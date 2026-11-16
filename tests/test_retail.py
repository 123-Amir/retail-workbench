import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.errors import PipelineError
from app.retail.data import RetailData
from app.retail.filters import apply_filters
from app.retail.inventory import inventory_metrics
from app.retail.joins import join_declared, safe_left_join
from app.retail.metrics import FILTER_FIELDS, build_enriched_orders, sales_metrics
from app.storage.runs import RunService
from app.storage.typed import typed_frame

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def T(rows, cols, schema):
    return typed_frame(pd.DataFrame(rows, columns=cols), schema)


ORDER_COLS = ["order_id", "order_date", "customer_id", "store_id", "sku", "quantity", "unit_price", "discount", "status"]
ORDER_SCHEMA = {"order_id": "string", "order_date": "date", "customer_id": "string", "store_id": "string", "sku": "string",
                "quantity": "integer", "unit_price": "integer", "discount": "float", "status": "string"}


def mini(extra_orders=(), products=None):
    orders = T([
        ["O1", "2025-01-10", "C1", "S1", "A", 2, 100, 0, "completed"],
        ["O2", "2025-01-20", "C2", "S2", "B", 1, 200, 0.5, "completed"],
        ["O3", "2025-02-05", "C1", "S1", "A", 1, 100, 0, "returned"],
        ["O4", "2025-02-06", "C3", "S1", "A", 5, 100, 0, "cancelled"],
        ["O5", "2025-02-07", "C3", "S1", "A", -1, 100, 0, "completed"],      # invalid quantity
        ["O6", "2025-02-08", "C3", "S1", "A", 1, 100, 0, "pending"],         # unknown status
        ["O7", "2025-02-09", "C3", "S1", "A", 1, 50, None, "completed"],     # missing discount
        ["O8", "2025-03-01", "C9", "S9", "C", 1, 10, 0, "completed"],        # orphan store + orphan sku
        *extra_orders], ORDER_COLS, ORDER_SCHEMA)
    prods = products if products is not None else T([["A", "Prod A", "T-Shirt", "Nike", 60], ["B", "Prod B", "Jeans", "Levis", 150]],
                                                    ["sku", "product_name", "category", "brand", "cost"],
                                                    {"sku": "string", "product_name": "string", "category": "string", "brand": "string", "cost": "integer"})
    stores = T([["S1", "Mumbai Central", "Mumbai", "West"], ["S2", "Delhi Central", "Delhi", "East"]],
               ["store_id", "store_name", "city", "region"], {c: "string" for c in ["store_id", "store_name", "city", "region"]})
    customers = T([["C1", "Pune"], ["C2", "Delhi"], ["C3", "Pune"]], ["customer_id", "city"], {"customer_id": "string", "city": "string"})
    return RetailData({"orders": orders, "products": prods, "stores": stores, "customers": customers},
                      {"orders": "orders@v1", "products": "products@v1", "stores": "stores@v1", "customers": "customers@v1"})


def grp(res, **key):
    return next(g for g in res["groups"] if all(g["group"][k] == v for k, v in key.items()))["metrics"]


# =============================== safe joins ===============================
def test_join_reports_orphans_and_never_changes_row_count():
    d = mini()
    r = join_declared(d.frames, "orders", "products")
    assert len(r.df) == len(d.frames["orders"]) and r.df["sku"].tolist() == d.frames["orders"]["sku"].tolist()   # same rows, same order
    rep = r.report
    assert rep["fan_out_free"] and rep["rows_after"] == rep["left_rows"] == 8
    assert (rep["matched_rows"], rep["unmatched_rows"]) == (7, 1) and rep["unmatched_key_examples"][0]["key"] == ["C"]
    assert rep["match_rate"] == 0.875 and rep["right_keys_never_referenced"] == 0
    assert r.df.loc[r.df["sku"] == "C", "products__category"].isna().all()


def test_fanout_is_refused_with_evidence():
    dup = T([["A", "x", "T", "N", 1], ["A", "y", "T", "N", 2]], ["sku", "product_name", "category", "brand", "cost"],
            {"sku": "string", "product_name": "string", "category": "string", "brand": "string", "cost": "integer"})
    with pytest.raises(PipelineError) as e:
        join_declared(mini(products=dup).frames, "orders", "products")
    assert e.value.code == "JOIN_FANOUT" and e.value.details["examples"] == [{"sku": "A"}]
    with pytest.raises(PipelineError) as e:                     # analytics surface it too, they never silently skip it
        sales_metrics(mini(products=dup), group_by=["category"])
    assert e.value.code == "JOIN_FANOUT"


def test_null_keys_never_match_each_other():
    left = T([["1", "a"], [None, "b"]], ["k", "v"], {"k": "string", "v": "string"})
    right = T([[None, "NULLROW"], ["1", "ONE"]], ["k", "name"], {"k": "string", "name": "string"})
    r = safe_left_join(left, right, ["k"], ["k"], "dim", ["name"], "fact")
    assert r.df["dim__name"].tolist()[0] == "ONE" and pd.isna(r.df["dim__name"].tolist()[1])
    assert r.report["null_key_rows"] == 1 and r.report["right_rows_with_null_key_ignored"] == 1


def test_join_input_errors_are_structured():
    a = T([["1"]], ["k"], {"k": "string"})
    n = T([[1]], ["k"], {"k": "integer"})
    for call, code in [(lambda: safe_left_join(a, a, ["zz"], ["k"], "r"), "JOIN_KEY_MISSING"),
                       (lambda: safe_left_join(a, n, ["k"], ["k"], "r"), "JOIN_KEY_TYPE_MISMATCH"),
                       (lambda: safe_left_join(a, a, ["k"], ["k"], "r", ["nope"]), "JOIN_COLUMN_MISSING"),
                       (lambda: safe_left_join(a.assign(r__v="x"), a.assign(v="y"), ["k"], ["k"], "r", ["v"]), "JOIN_COLUMN_COLLISION")]:
        with pytest.raises(PipelineError) as e:
            call()
        assert e.value.code == code


def test_only_declared_relationships_are_allowed():
    d = mini()
    for l, r in [("products", "orders"), ("orders", "inventory"), ("stores", "products")]:
        with pytest.raises(PipelineError) as e:
            join_declared(d.frames, l, r)
        assert e.value.code == "JOIN_NOT_ALLOWED" and "orders->products" in e.value.details["allowed"]


def test_multi_key_join_and_missing_dimension_datasets_are_reported():
    left = T([["S1", "A"], ["S1", "B"], ["S2", "A"]], ["store_id", "sku"], {"store_id": "string", "sku": "string"})
    right = T([["S1", "A", 5], ["S2", "A", 7]], ["store_id", "sku", "n"], {"store_id": "string", "sku": "string", "n": "integer"})
    r = safe_left_join(left, right, ["store_id", "sku"], ["store_id", "sku"], "x", ["n"], "l")
    assert r.df["x__n"].tolist()[0] == 5 and pd.isna(r.df["x__n"].tolist()[1]) and r.df["x__n"].tolist()[2] == 7
    d = mini()
    d.frames.pop("products")
    d.frames.pop("customers")
    _, notes = build_enriched_orders(d)
    assert {s["join"] for s in notes["joins_skipped"]} == {"orders->products", "orders->customers"}


# =============================== sales metrics: hand-checked numbers ===============================
def test_totals_match_hand_calculation():
    t = sales_metrics(mini())["totals"]
    assert t["gross_sales"] == 410.0 and t["returned_sales"] == 100.0 and t["net_revenue"] == 310.0
    assert t["orders"] == 3 and t["aov"] == 103.33
    assert t["units_shipped"] == 5 and t["units_returned"] == 1 and t["return_rate_units"] == 0.2
    assert t["return_rate_value"] == round(100 / 410, 4) and t["cancellation_rate"] == round(1 / 7, 4)
    assert t["gross_margin"] == 30.0 and t["margin_pct"] == 0.1 and t["cost_coverage"] == round(300 / 310, 4)


def test_exclusions_are_counted_and_explained_not_repaired():
    ev = sales_metrics(mini())["evidence"]
    assert ev["exclusions"] == {"discount_missing_or_out_of_range": 1, "quantity_missing_or_not_positive": 1, "status_unknown": 1}
    text = " ".join(ev["missing_evidence"])
    assert "no match in products" in text and "no match in stores" in text and "excluded from margin" in text
    assert ev["rows_total"] == ev["rows_considered"] == 8 and ev["datasets"]["orders"] == "orders@v1"


def test_group_by_region_with_unknown_bucket():
    r = sales_metrics(mini(), group_by=["region"])
    assert grp(r, region="West") == {**grp(r, region="West")} and grp(r, region="West")["net_revenue"] == 200.0
    assert grp(r, region="West")["gross_sales"] == 300.0 and grp(r, region="West")["returned_sales"] == 100.0
    assert grp(r, region="West")["return_rate_units"] == round(1 / 3, 4)
    assert grp(r, region="East")["net_revenue"] == 100.0 and grp(r, region="(unknown)")["net_revenue"] == 10.0
    assert [g["group"]["region"] for g in r["groups"]] == ["West", "East", "(unknown)"]           # sorted by net_revenue desc


@pytest.mark.parametrize("dims", [["region"], ["category"], ["month"], ["region", "category"], ["brand", "month"], ["sku"]])
def test_partition_invariant_groups_sum_to_total(dims):
    r = sales_metrics(mini(), group_by=dims, metrics=["net_revenue", "gross_sales", "orders", "units_shipped"])
    for m in ("net_revenue", "gross_sales", "orders", "units_shipped"):
        assert abs(sum(g["metrics"][m] for g in r["groups"]) - r["totals"][m]) < 1e-6


def test_month_and_category_groupings():
    m = sales_metrics(mini(), group_by=["month"])
    assert grp(m, month="2025-01")["net_revenue"] == 300.0 and grp(m, month="2025-02")["net_revenue"] == 0.0
    assert grp(m, month="2025-02")["return_rate_value"] == 1.0 and grp(m, month="2025-02")["aov"] is None
    c = sales_metrics(mini(), group_by=["category"])
    assert grp(c, category="T-Shirt")["net_revenue"] == 200.0 and grp(c, category="Jeans")["net_revenue"] == 100.0
    assert grp(c, category="(unknown)")["net_revenue"] == 10.0


def test_multi_turn_followup_keeps_state_through_filters():
    turn1 = [{"field": "region", "op": "eq", "value": "west"}]                    # "only stores in the West region"
    r1 = sales_metrics(mini(), filters=turn1)
    assert r1["totals"]["net_revenue"] == 200.0 and r1["evidence"]["rows_considered"] == 6      # all 6 lines of store S1
    r2 = sales_metrics(mini(), group_by=["category"], filters=turn1)              # "among those, which categories ..."
    assert [g["group"]["category"] for g in r2["groups"]] == ["T-Shirt"]
    assert grp(r2, category="T-Shirt")["return_rate_units"] == round(1 / 3, 4)
    assert r2["evidence"]["filters"] == [{"field": "region", "op": "eq", "value": "west", "rows_before": 8, "rows_after": 6}]


def test_null_semantics_in_filters_follow_sql():
    r = sales_metrics(mini(), filters=[{"field": "region", "op": "ne", "value": "west"}])
    assert r["totals"]["net_revenue"] == 100.0                                   # unknown-region row is NOT 'not west'
    r = sales_metrics(mini(), filters=[{"field": "region", "op": "is_null"}])
    assert r["totals"]["net_revenue"] == 10.0


def test_sort_limit_and_bounds():
    r = sales_metrics(mini(), group_by=["region"], limit=2)
    assert len(r["groups"]) == 2 and r["truncated"] and r["group_count"] == 3
    asc = sales_metrics(mini(), group_by=["region"], direction="asc")
    assert asc["groups"][0]["group"]["region"] == "(unknown)"
    for bad in (0, 201, 10**9):
        with pytest.raises(PipelineError) as e:
            sales_metrics(mini(), group_by=["region"], limit=bad)
        assert e.value.code == "LIMIT_TOO_LARGE"


@pytest.mark.parametrize("kw,code", [({"group_by": ["salary"]}, "INVALID_GROUP_BY"), ({"group_by": ["region"] * 2}, "INVALID_GROUP_BY"),
                                     ({"group_by": ["region", "category", "brand", "month"]}, "INVALID_GROUP_BY"),
                                     ({"metrics": ["revenue_per_ghost"]}, "INVALID_METRIC"),
                                     ({"metrics": ["aov"], "sort_by": "net_revenue"}, "INVALID_SORT"),
                                     ({"direction": "sideways", "group_by": ["region"]}, "INVALID_SORT"),
                                     ({"filters": [{"field": "salary", "op": "eq", "value": 1}]}, "INVALID_FILTER"),
                                     ({"filters": [{"field": "region", "op": "drop_table", "value": 1}]}, "INVALID_FILTER"),
                                     ({"filters": [{"field": "quantity", "op": "gt", "value": "abc"}]}, "INVALID_FILTER"),
                                     ({"filters": [{"field": "order_date", "op": "gt", "value": "03/04/2025"}]}, "INVALID_FILTER"),
                                     ({"filters": [{"field": "region", "op": "in", "value": "West"}]}, "INVALID_FILTER"),
                                     ({"filters": [{"field": "region", "op": "eq", "value": "x", "sql": "1=1"}]}, "INVALID_FILTER"),
                                     ({"filters": [{"field": "region", "op": "eq", "value": "x"}] * 11}, "INVALID_FILTER")])
def test_every_malformed_request_is_rejected_with_a_structured_error(kw, code):
    with pytest.raises(PipelineError) as e:
        sales_metrics(mini(), **kw)
    assert e.value.code == code


def test_unavailable_data_is_reported_not_invented():
    d = mini()
    d.frames.pop("stores")
    with pytest.raises(PipelineError) as e:
        sales_metrics(d, group_by=["region"])
    assert e.value.code == "DIMENSION_UNAVAILABLE"
    d.frames.pop("products")
    r = sales_metrics(d)                                                          # still answerable without cost
    assert r["totals"]["net_revenue"] == 310.0 and r["totals"]["gross_margin"] is None and r["totals"]["margin_pct"] is None
    assert any("cost unavailable" in m for m in r["evidence"]["missing_evidence"])
    with pytest.raises(PipelineError) as e:
        sales_metrics(RetailData({}, {}))
    assert e.value.code == "DATASET_NOT_LOADED"


def test_empty_scope_returns_nulls_not_errors():
    r = sales_metrics(mini(), group_by=["region"], filters=[{"field": "region", "op": "eq", "value": "nowhere"}])
    assert r["groups"] == [] and r["evidence"]["rows_considered"] == 0 and r["totals"]["net_revenue"] is None


def test_results_are_deterministic_and_json_safe():
    a, b = sales_metrics(mini(), group_by=["region", "category"]), sales_metrics(mini(), group_by=["region", "category"])
    assert a == b and json.dumps(a)


# =============================== inventory ===============================
def inv_data(dup=False, products=None):
    base = mini(extra_orders=[["O9", "2025-03-02", "C1", "S1", "A", 4, 100, 0, "completed"],
                              ["O10", "2025-03-03", "C1", "S3", "A", 3, 100, 0, "completed"]])
    # completed units S1/A = 2 (O1) + 4 (O9) = 6 ; S2/B = 1 ; S3/A = 3 (no inventory record)
    rows = [["S1", "A", 0, 10, "2025-06-01"], ["S1", "B", 5, 10, "2025-01-01"], ["S2", "A", 20, 10, "2025-08-01"], ["S2", "B", None, 10, "2025-07-01"]]
    if dup:
        rows.append(["S1", "A", 0, 10, "2025-06-01"])
    inv = T(rows, ["store_id", "sku", "stock_on_hand", "reorder_level", "last_restock_date"],
            {"store_id": "string", "sku": "string", "stock_on_hand": "integer", "reorder_level": "integer", "last_restock_date": "date"})
    base.frames["inventory"], base.versions["inventory"] = inv, "inventory@v1"
    if products is not None:
        base.frames["products"] = products
    return base


def test_inventory_totals_match_hand_calculation():
    t = inventory_metrics(inv_data(), as_of="2025-09-01")["totals"]
    assert t["positions"] == 4 and t["units_on_hand"] == 25 and t["unknown_stock_positions"] == 1
    assert t["stock_out_positions"] == 1 and t["stock_out_rate"] == round(1 / 3, 4) and t["low_stock_positions"] == 1
    assert t["units_sold_net"] == 6 and t["sell_through"] == round(6 / 31, 4)      # S1/A=6, S1/B=0, S2/A=0 ; S2/B has unknown stock
    assert t["avg_age_days"] == 107.0 and t["aged_90_plus_positions"] == 2 and t["unknown_age_positions"] == 0


def test_inventory_by_region_and_evidence():
    r = inventory_metrics(inv_data(), group_by=["region"], as_of="2025-09-01")
    west = next(g["metrics"] for g in r["groups"] if g["group"]["region"] == "West")
    assert west["positions"] == 2 and west["stock_out_rate"] == 0.5 and west["units_sold_net"] == 6
    ev = r["evidence"]
    assert ev["as_of"] == "2025-09-01" and ev["as_of_source"] == "provided"
    text = " ".join(ev["missing_evidence"])
    assert "snapshot" in text and "4 net units sold" in text                       # S3/A (3) + orphan store S9/C (1) have no inventory record
    assert {j["join"].split("(")[0] for j in ev["joins"]} == {"inventory", "inventory"} and all(j["fan_out_free"] for j in ev["joins"])


def test_inventory_duplicate_positions_do_not_double_count_sales():
    t = inventory_metrics(inv_data(dup=True), as_of="2025-09-01")
    assert t["totals"]["positions"] == 5 and t["totals"]["units_sold_net"] == 6
    assert any("duplicate store+sku" in m for m in t["evidence"]["missing_evidence"])


def test_inventory_default_as_of_comes_from_data_and_bad_as_of_is_rejected():
    r = inventory_metrics(inv_data())
    assert r["evidence"]["as_of"] == "2025-08-01"                                  # max(last restock 08-01, last order 03-03)
    assert "not today" in r["evidence"]["as_of_source"]
    with pytest.raises(PipelineError) as e:
        inventory_metrics(inv_data(), as_of="01/09/2025")
    assert e.value.code == "INVALID_AS_OF"


def test_inventory_refuses_fanout_and_missing_dataset():
    dup = T([["A", "x", "T", "N", 1], ["A", "y", "T", "N", 2]], ["sku", "product_name", "category", "brand", "cost"],
            {"sku": "string", "product_name": "string", "category": "string", "brand": "string", "cost": "integer"})
    with pytest.raises(PipelineError) as e:
        inventory_metrics(inv_data(products=dup))
    assert e.value.code == "JOIN_FANOUT"
    with pytest.raises(PipelineError) as e:
        inventory_metrics(mini())
    assert e.value.code == "DATASET_NOT_LOADED"


# =============================== end to end on the cleaned sample data ===============================
@pytest.fixture(scope="module")
def sample_data(tmp_path_factory):
    svc = RunService(root=tmp_path_factory.mktemp("store"))
    files = [(p.name, p.read_bytes()) for p in sorted(SAMPLES.glob("*.csv"))]
    r = svc.submit("u", files, wait=True)
    return RetailData.from_run(svc, "u", r["run_id"])


def test_sample_revenue_matches_an_independent_pandas_calculation(sample_data):
    o = sample_data.frames["orders"]
    ok = o["status"].str.casefold().eq("completed") & (o["quantity"] > 0) & o["unit_price"].notna() & o["discount"].notna()
    expected = float((o.loc[ok, "quantity"].astype(float) * o.loc[ok, "unit_price"].astype(float) * (1 - o.loc[ok, "discount"])).sum())
    r = sales_metrics(sample_data)
    assert abs(r["totals"]["net_revenue"] - round(expected, 2)) < 0.01
    assert r["totals"]["orders"] == int(o.loc[ok, "order_id"].nunique())


def test_sample_partitions_sum_to_total_and_orphan_sku_is_visible(sample_data):
    for dims in (["region"], ["category"], ["month"], ["region", "category"]):
        r = sales_metrics(sample_data, group_by=dims, metrics=["net_revenue", "orders"])
        assert abs(sum(g["metrics"]["net_revenue"] for g in r["groups"]) - r["totals"]["net_revenue"]) < 0.5
        assert sum(g["metrics"]["orders"] for g in r["groups"]) == r["totals"]["orders"]
    c = sales_metrics(sample_data, group_by=["category"])
    assert any(g["group"]["category"] == "(unknown)" for g in c["groups"])       # SKU-9999 lines are not dropped
    assert any("SKU-9999" in m for m in c["evidence"]["missing_evidence"])
    assert len(c["groups"]) == 6                                                  # 5 canonical categories + unknown
    assert json.dumps(c)


def test_sample_retail_flow_return_rate_investigation(sample_data):
    west = sales_metrics(sample_data, group_by=["category"], metrics=["return_rate_units", "units_shipped"],
                         filters=[{"field": "region", "op": "eq", "value": "West"}], sort_by="return_rate_units")
    assert west["evidence"]["filters"][0]["rows_after"] < west["evidence"]["rows_total"]
    assert west["groups"][0]["metrics"]["return_rate_units"] >= west["groups"][-1]["metrics"]["return_rate_units"]
    inv = inventory_metrics(sample_data, group_by=["category"], metrics=["stock_out_rate", "sell_through"])
    assert inv["evidence"]["rows_total"] == 320 and all(j["fan_out_free"] for j in inv["evidence"]["joins"])
