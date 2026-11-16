import json
import time
from pathlib import Path

import pandas as pd
import pytest

from app.chat import executor
from app.chat.catalog import build_catalog
from app.chat.executor import answer_plan
from app.chat.plan import parse_plan
from app.chat.validator import validate_plan
from app.errors import PipelineError
from app.retail.data import RetailData
from app.retail.metrics import sales_metrics
from app.storage.runs import RunService
from tests.test_retail import T, inv_data, mini

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def agg(metrics, **kw):
    return {"intent": "aggregate", "dataset": "orders", "metrics": metrics, **kw}


def ok(plan, data=None):
    out = answer_plan(plan, data or mini())
    assert out["status"] == "ok", out.get("error")
    return out


def err(plan, data=None):
    out = answer_plan(plan, data or mini())
    assert out["status"] == "error", out
    return out["error"]


def rows(out):
    return out["result"]["rows"]


# ================================ plan schema ================================
def test_valid_plan_parses_from_dict_and_json_string():
    p = {"intent": "aggregate", "dataset": "orders", "filters": [{"field": "region", "op": "eq", "value": "West"}],
         "group_by": ["category"], "metrics": [{"agg": "sum", "field": "quantity", "as": "units"}], "sort": [{"field": "units", "dir": "desc"}], "limit": 10}
    assert parse_plan(p).metrics[0].as_ == "units" and parse_plan(json.dumps(p)).limit == 10


@pytest.mark.parametrize("bad", [
    {"intent": "aggregate", "dataset": "orders", "metrics": [], "sql": "DROP TABLE x"},                    # unknown key
    {"intent": "execute", "dataset": "orders"},                                                              # unknown intent
    {"intent": "select", "dataset": "orders", "select": ["quantity; DROP TABLE orders"]},                    # injection in field name
    {"intent": "select", "dataset": "orders", "select": ["a"], "limit": "10"},                               # strict types: no coercion
    {"intent": "select", "dataset": "Orders; --", "select": ["a"]},
    {"intent": "select", "dataset": "orders", "select": ["a"] * 13},
    {"intent": "aggregate", "dataset": "orders", "metrics": [{"agg": "count", "as": "n"}], "group_by": ["a", "b", "c", "d"]},
    {"intent": "select", "dataset": "orders", "select": ["a"], "filters": [{"field": "a", "op": "eq", "value": 1}] * 11},
    {"intent": "select", "dataset": "orders", "select": ["a"], "filters": [{"field": "a", "op": "in", "value": list(range(101))}]},
    {"intent": "select", "dataset": "orders", "select": ["a"], "filters": [{"field": "a", "op": "eq", "value": {"$ne": 1}}]},
    {"intent": "clarify", "question": "x" * 301}, {"intent": "clarify", "question": "q", "options": ["a"] * 6},
    {"intent": "retail_metric", "domain": "sales", "extra": 1}, ["list"], "not json{", 42,
])
def test_malformed_or_hostile_plans_are_rejected_without_echoing_input(bad):
    with pytest.raises(PipelineError) as e:
        parse_plan(bad)
    assert e.value.code == "INVALID_PLAN" and "DROP TABLE" not in json.dumps(e.value.details)


def test_oversized_plan_string_is_rejected():
    with pytest.raises(PipelineError) as e:
        parse_plan("{" + " " * 25_000 + "}")
    assert e.value.code == "INVALID_PLAN"


# ================================ validator: allow-list ================================
@pytest.mark.parametrize("plan,code", [
    ({"intent": "select", "dataset": "ghost", "select": ["a"]}, "UNKNOWN_DATASET"),
    (agg([{"agg": "sum", "field": "revenue", "as": "r"}]), "UNKNOWN_FIELD"),                                # model invents a column
    (agg([{"agg": "sum", "field": "quantity", "as": "r"}], group_by=["products.category"]), "FIELD_NOT_JOINED"),
    (agg([{"agg": "count", "as": "n"}], joins=["inventory"]), "JOIN_NOT_ALLOWED"),
    ({"intent": "aggregate", "dataset": "products", "joins": ["orders"], "metrics": [{"agg": "count", "as": "n"}]}, "JOIN_NOT_ALLOWED"),  # reverse = fan-out
    (agg([{"agg": "median_absolute", "field": "quantity", "as": "r"}]), "INVALID_AGGREGATION"),
    (agg([{"agg": "sum", "field": "status", "as": "r"}]), "INVALID_AGGREGATION"),                          # sum of text
    (agg([{"agg": "max", "field": "status", "as": "r"}]), "INVALID_AGGREGATION"),
    (agg([{"agg": "sum", "as": "r"}]), "INVALID_AGGREGATION"),
    (agg([{"agg": "count", "as": "n"}, {"agg": "sum", "field": "quantity", "as": "n"}]), "INVALID_ALIAS"),
    (agg([{"agg": "count", "as": "status"}], group_by=["status"]), "INVALID_ALIAS"),
    (agg([{"agg": "count", "as": "N1"}]), "INVALID_PLAN"),
    (agg([{"agg": "count", "as": "n"}], filters=[{"field": "status", "op": "regex", "value": ".*"}]), "INVALID_OPERATOR"),
    (agg([{"agg": "count", "as": "n"}], filters=[{"field": "quantity", "op": "gt", "value": "abc"}]), "INVALID_FILTER"),
    (agg([{"agg": "count", "as": "n"}], filters=[{"field": "order_date", "op": "gt", "value": "03/04/2025"}]), "INVALID_FILTER"),
    (agg([{"agg": "count", "as": "n"}], limit=201), "LIMIT_TOO_LARGE"),
    (agg([{"agg": "count", "as": "n"}], limit=10**9), "LIMIT_TOO_LARGE"),
    (agg([{"agg": "count", "as": "n"}], limit=0), "INVALID_LIMIT"),
    (agg([{"agg": "count", "as": "n"}], group_by=["status"], sort=[{"field": "quantity", "dir": "asc"}]), "INVALID_SORT"),
    (agg([{"agg": "count", "as": "n"}], sort=[{"field": "n", "dir": "sideways"}]), "INVALID_SORT"),
    (agg([], group_by=["status"]), "INVALID_PLAN"),
    ({"intent": "select", "dataset": "orders", "select": ["status"], "metrics": [{"agg": "count", "as": "n"}]}, "INVALID_PLAN"),
    ({"intent": "distinct", "dataset": "orders", "group_by": ["status", "sku", "store_id"]}, "INVALID_PLAN"),
    ({"intent": "describe", "dataset": "orders", "select": ["status"], "group_by": ["sku"]}, "INVALID_PLAN"),
])
def test_validator_rejects_with_stable_codes(plan, code):
    assert err(plan)["code"] == code


def test_unknown_field_error_offers_suggestions_for_the_repair_loop():
    e = err(agg([{"agg": "sum", "field": "quantiy", "as": "r"}]))
    assert e["code"] == "UNKNOWN_FIELD" and "quantity" in e["details"]["suggestions"] and "quantity" in e["details"]["available"]


def test_field_from_unloaded_dataset_is_unknown_or_not_joined():
    d = mini()
    d.frames.pop("stores")
    assert err(agg([{"agg": "count", "as": "n"}], joins=["stores"]), d)["code"] == "DATASET_NOT_LOADED"
    assert err(agg([{"agg": "count", "as": "n"}], group_by=["stores.region"]), d)["code"] == "UNKNOWN_FIELD"


def test_normalisation_resolves_aliases_and_drops_unused_joins():
    v = validate_plan(parse_plan(agg([{"agg": "sum", "field": "orders.quantity", "as": "u"}], joins=["products", "stores"],
                                    group_by=["products.category"])), mini())
    assert v.plan.metrics[0].field == "quantity" and v.plan.joins == ["products"] and v.right_columns == {"products": ["category"]}


# ================================ privacy ================================
def customers_data():
    c = T([["C1", "Asha", "a@x.com", "Pune"], ["C2", "Ben", "b@x.com", "Pune"], ["C3", "Cy", None, "Delhi"]],
          ["customer_id", "name", "email", "city"], {k: "string" for k in ["customer_id", "name", "email", "city"]})
    d = mini()
    d.frames["customers"] = c
    return d


@pytest.mark.parametrize("plan", [
    {"intent": "select", "dataset": "customers", "select": ["email"]},
    {"intent": "select", "dataset": "customers", "select": ["customer_id", "name"]},
    {"intent": "distinct", "dataset": "customers", "group_by": ["email"]},
    {"intent": "aggregate", "dataset": "customers", "group_by": ["email"], "metrics": [{"agg": "count", "as": "n"}]},
    {"intent": "select", "dataset": "customers", "select": ["city"], "filters": [{"field": "email", "op": "eq", "value": "a@x.com"}]},
    {"intent": "describe", "dataset": "customers", "select": ["email"]},
    {"intent": "aggregate", "dataset": "customers", "metrics": [{"agg": "max", "field": "email", "as": "m"}]},
    {"intent": "aggregate", "dataset": "orders", "joins": ["customers"], "group_by": ["customers.name"], "metrics": [{"agg": "count", "as": "n"}]},
])
def test_personal_data_cannot_be_shown_grouped_filtered_or_described(plan):
    assert err(plan, customers_data())["code"] in ("FIELD_RESTRICTED", "INVALID_AGGREGATION")


def test_personal_data_can_still_be_counted():
    out = ok({"intent": "aggregate", "dataset": "customers", "metrics": [{"agg": "count_distinct", "field": "email", "as": "emails"}]}, customers_data())
    assert rows(out) == [{"emails": 2}]


def test_catalog_is_bounded_and_never_lists_personal_values():
    cat = build_catalog(customers_data())
    cols = {c["name"]: c for c in cat["datasets"]["customers"]["columns"]}
    assert cols["email"]["pii"] and "distinct_values" not in cols["email"] and "distinct_values" not in cols["name"]
    assert cols["city"]["distinct_values"] == ["Delhi", "Pune"]
    assert "orders -> products on sku (many_to_one)" in cat["joins"] and "products -> orders" not in " ".join(cat["joins"])
    assert "a@x.com" not in json.dumps(cat)


# ================================ execution: hand-checked answers ================================
def test_count_and_sum_by_status():
    out = ok(agg([{"agg": "count", "as": "lines"}, {"agg": "sum", "field": "quantity", "as": "units"}], group_by=["status"]))
    assert rows(out) == [{"status": "cancelled", "lines": 1, "units": 5}, {"status": "completed", "lines": 5, "units": 4},
                         {"status": "pending", "lines": 1, "units": 1}, {"status": "returned", "lines": 1, "units": 1}]


def test_scalar_aggregations():
    out = ok(agg([{"agg": "avg", "field": "unit_price", "as": "avg_price"}, {"agg": "min", "field": "order_date", "as": "first"},
                  {"agg": "max", "field": "order_date", "as": "last"}, {"agg": "median", "field": "quantity", "as": "med"},
                  {"agg": "count_distinct", "field": "customer_id", "as": "customers"}, {"agg": "count", "field": "discount", "as": "with_discount"}]))
    assert rows(out) == [{"avg_price": 95.0, "first": "2025-01-10", "last": "2025-03-01", "med": 1.0, "customers": 4, "with_discount": 7}]


def test_join_group_null_bucket_sort_and_stability():
    out = ok(agg([{"agg": "sum", "field": "quantity", "as": "units"}], joins=["products"], group_by=["products.category"], sort=[{"field": "units", "dir": "desc"}]))
    assert rows(out) == [{"products.category": "T-Shirt", "units": 9}, {"products.category": "Jeans", "units": 1}, {"products.category": None, "units": 1}]
    ev = out["evidence"]
    assert ev["joins"][0]["unmatched_rows"] == 1 and ev["joins"][0]["fan_out_free"] and ev["datasets"] == {"orders": "orders@v1", "products": "products@v1"}
    assert ev["columns_used"] == ["orders.quantity", "products.category"]


def test_filters_on_joined_fields_are_case_insensitive_and_counted():
    out = ok(agg([{"agg": "count", "as": "n"}], joins=["stores"], filters=[{"field": "stores.region", "op": "eq", "value": "west"}]))
    assert rows(out) == [{"n": 6}] and out["evidence"]["filters"][0]["rows_after"] == 6 and out["evidence"]["rows_total"] == 8


def test_select_sort_limit_and_truncation_flag():
    out = ok({"intent": "select", "dataset": "orders", "select": ["order_id", "quantity"], "sort": [{"field": "quantity", "dir": "desc"}], "limit": 3})
    assert [r["order_id"] for r in rows(out)] == ["O4", "O1", "O2"]
    assert out["result"]["total_rows"] == 8 and out["result"]["row_count"] == 3 and out["result"]["truncated"]


def test_distinct_counts_and_describe():
    out = ok({"intent": "distinct", "dataset": "orders", "group_by": ["status"]})
    assert rows(out) == [{"status": "completed", "count": 5}, {"status": "cancelled", "count": 1}, {"status": "pending", "count": 1}, {"status": "returned", "count": 1}]
    d = rows(ok({"intent": "describe", "dataset": "orders", "select": ["quantity", "status", "order_date"]}))
    q, s, t = d
    assert (q["count"], q["nulls"], q["unique"], q["min"], q["max"], q["mean"], q["median"]) == (8, 0, 4, -1.0, 5.0, 1.375, 1.0)
    assert s["top_values"][0] == {"value": "completed", "count": 5} and t["min"] == "2025-01-10" and t["max"] == "2025-03-01"


def test_empty_result_is_explicit():
    out = ok(agg([{"agg": "count", "as": "n"}], group_by=["status"], filters=[{"field": "status", "op": "eq", "value": "nope"}]))
    assert out["result"]["row_count"] == 0 and "No rows matched" in out["answer"]


def test_results_are_deterministic_and_json_safe():
    p = agg([{"agg": "sum", "field": "quantity", "as": "u"}], joins=["products", "stores"], group_by=["stores.region", "products.category"])
    a, b = ok(p), ok(p)
    assert a == b and json.dumps(a)


# ================================ retail intent through the same gate ================================
def test_retail_plan_matches_the_retail_engine_exactly():
    plan = {"intent": "retail_metric", "domain": "sales", "group_by": ["region"], "metrics": ["net_revenue", "aov", "return_rate_units"]}
    out = ok(plan)
    direct = sales_metrics(mini(), group_by=["region"], metrics=["net_revenue", "aov", "return_rate_units"])
    assert rows(out) == [{**g["group"], **g["metrics"]} for g in direct["groups"]] and out["result"]["totals"]["net_revenue"] == 310.0
    assert rows(out)[0] == {"region": "West", "net_revenue": 200.0, "aov": 200.0, "return_rate_units": round(1 / 3, 4)}
    assert out["evidence"]["exclusions"] and out["evidence"]["missing_evidence"] and "Caveat:" in out["answer"]


def test_retail_without_groups_returns_one_totals_row_and_inventory_works():
    out = ok({"intent": "retail_metric", "domain": "sales", "metrics": ["net_revenue", "orders"]})
    assert rows(out) == [{"net_revenue": 310.0, "orders": 3}]
    inv = ok({"intent": "retail_metric", "domain": "inventory", "group_by": ["region"], "metrics": ["stock_out_rate"], "as_of": "2025-09-01"}, inv_data())
    assert {r["region"]: r["stock_out_rate"] for r in rows(inv)} == {"West": 0.5, "East": 0.0}


@pytest.mark.parametrize("plan,code", [
    ({"intent": "retail_metric", "domain": "payroll"}, "INVALID_PLAN"),
    ({"intent": "retail_metric", "domain": "sales", "group_by": ["salary"]}, "INVALID_GROUP_BY"),
    ({"intent": "retail_metric", "domain": "sales", "metrics": ["profit_per_unicorn"]}, "INVALID_METRIC"),
    ({"intent": "retail_metric", "domain": "sales", "metrics": ["aov"], "sort_by": "net_revenue"}, "INVALID_SORT"),
    ({"intent": "retail_metric", "domain": "sales", "limit": 5000}, "LIMIT_TOO_LARGE"),
    ({"intent": "retail_metric", "domain": "sales", "filters": [{"field": "salary", "op": "eq", "value": 1}]}, "UNKNOWN_FIELD"),
    ({"intent": "retail_metric", "domain": "sales", "filters": [{"field": "region", "op": "like", "value": "W%"}]}, "INVALID_OPERATOR"),
    ({"intent": "retail_metric", "domain": "sales", "filters": [{"field": "quantity", "op": "gt", "value": "x"}]}, "INVALID_FILTER"),
    ({"intent": "retail_metric", "domain": "inventory"}, "DATASET_NOT_LOADED"),
])
def test_retail_plans_are_validated_before_running(plan, code):
    assert err(plan)["code"] == code


@pytest.mark.parametrize("plan,code", [
    ({"intent": "retail_metric", "domain": "inventory", "as_of": "09/01/2025"}, "INVALID_AS_OF"),
    ({"intent": "retail_metric", "domain": "inventory", "as_of": "2025-13-45"}, "INVALID_AS_OF"),
    ({"intent": "retail_metric", "domain": "sales", "as_of": "2025-09-01"}, "INVALID_PLAN"),
    ({"intent": "retail_metric", "domain": "inventory", "group_by": ["month"]}, "INVALID_GROUP_BY"),
])
def test_inventory_specific_validation_runs_on_data_that_has_inventory(plan, code):
    assert err(plan, inv_data())["code"] == code


def test_retail_dimension_needing_a_missing_dataset_is_refused_upfront():
    d = mini()
    d.frames.pop("stores")
    assert err({"intent": "retail_metric", "domain": "sales", "group_by": ["region"]}, d)["code"] == "DIMENSION_UNAVAILABLE"
    assert err({"intent": "retail_metric", "domain": "sales", "filters": [{"field": "region", "op": "eq", "value": "West"}]}, d)["code"] == "DIMENSION_UNAVAILABLE"


# ================================ clarify / refuse / failure paths ================================
def test_clarification_and_refusal_are_first_class_and_execute_nothing():
    c = answer_plan({"intent": "clarify", "question": "Do you mean units or revenue?", "options": ["units", "revenue"]}, mini())
    assert c["status"] == "clarification_needed" and c["clarification"]["options"] == ["units", "revenue"] and "result" not in c
    u = answer_plan({"intent": "unanswerable", "reason": "No supplier data is loaded.", "missing": ["suppliers"]}, mini())
    assert u["status"] == "unanswerable" and u["missing"] == ["suppliers"] and "Missing: suppliers" in u["answer"] and "result" not in u


def test_timeout_is_a_structured_error(monkeypatch):
    monkeypatch.setattr(executor, "_execute", lambda v, d: time.sleep(0.5))
    out = answer_plan(agg([{"agg": "count", "as": "n"}]), mini(), timeout=0.05)
    assert out["error"]["code"] == "QUERY_TIMEOUT" and out["trace"]["stages_completed"] == ["parse", "validate"]


def test_scan_cap_blocks_expensive_work(monkeypatch):
    monkeypatch.setattr(executor, "MAX_ROWS_SCANNED", 5)
    assert err(agg([{"agg": "count", "as": "n"}]))["code"] == "EXECUTION_TOO_EXPENSIVE"


def test_unexpected_exception_does_not_leak_internals(monkeypatch):
    def boom(v, d):
        raise ValueError("secret path /etc/shadow")
    monkeypatch.setattr(executor, "_execute", boom)
    out = answer_plan(agg([{"agg": "count", "as": "n"}]), mini())
    assert out["error"]["code"] == "INTERNAL_ERROR" and "secret" not in json.dumps(out)


def test_join_fanout_in_data_surfaces_as_a_structured_error():
    dup = T([["A", "x", "T", "N", 1], ["A", "y", "T", "N", 2]], ["sku", "product_name", "category", "brand", "cost"],
            {"sku": "string", "product_name": "string", "category": "string", "brand": "string", "cost": "integer"})
    out = answer_plan(agg([{"agg": "count", "as": "n"}], joins=["products"], group_by=["products.category"]), mini(products=dup))
    assert out["error"]["code"] == "JOIN_FANOUT"


# ================================ end to end on the real cleaned sample data ================================
@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    svc = RunService(root=tmp_path_factory.mktemp("chat"))
    files = [(p.name, p.read_bytes()) for p in sorted(SAMPLES.glob("*.csv"))]
    return RetailData.from_run(svc, "u", svc.submit("u", files, wait=True)["run_id"])


def test_answers_match_independent_pandas(sample):
    o = sample.frames["orders"]
    out = ok(agg([{"agg": "sum", "field": "quantity", "as": "units"}, {"agg": "count", "as": "lines"}], group_by=["status"]), sample)
    exp = o.groupby("status", dropna=False).agg(units=("quantity", "sum"), lines=("order_id", "size")).reset_index().sort_values("status")
    assert rows(out) == [{"status": r.status, "units": int(r.units), "lines": int(r.lines)} for r in exp.itertuples()]
    out = ok(agg([{"agg": "count", "as": "n"}], joins=["products"], group_by=["products.category"]), sample)
    merged = o.merge(sample.frames["products"][["sku", "category"]], on="sku", how="left")
    assert sum(r["n"] for r in rows(out)) == len(o) == len(merged) and out["evidence"]["datasets"] == {k: sample.versions[k] for k in ("orders", "products")}
    got = {r["products.category"]: r["n"] for r in rows(out)}
    assert got[None] == int(merged["category"].isna().sum())


def test_prompt_injection_in_a_cell_is_data_not_instructions(sample):
    out = ok({"intent": "select", "dataset": "products", "select": ["sku", "product_name"],
              "filters": [{"field": "product_name", "op": "contains", "value": "ignore all previous"}]}, sample)
    assert out["result"]["row_count"] >= 1 and out["flags"]["suspected_prompt_injection_in_results"]
    assert out["status"] == "ok" and set(out["plan"]) >= {"intent", "dataset"} and len(out["answer"].splitlines()) <= 12
    assert all(len(line) < 200 for line in out["answer"].splitlines())              # cell text is truncated in the answer
    clean = ok({"intent": "select", "dataset": "stores", "select": ["store_id"]}, sample)
    assert not clean["flags"]["suspected_prompt_injection_in_results"]


def test_dataset_versions_make_answers_reproducible(sample):
    out = ok({"intent": "retail_metric", "domain": "sales", "group_by": ["category"], "metrics": ["return_rate_units"],
              "filters": [{"field": "region", "op": "eq", "value": "West"}], "sort_by": "return_rate_units"}, sample)
    assert out["evidence"]["datasets"]["orders"] == sample.versions["orders"] and out["evidence"]["filters"][0]["rows_after"] == 380
    assert f"orders@{sample.versions['orders'].split('@')[1]}" in out["answer"]


# ================================ evidence contract: the assignment's required fields, for EVERY answer type ================================
REQUIRED_EVIDENCE = ["datasets", "columns_used", "filters", "joins", "operations", "rows_considered", "rows_total", "preview_limit"]


@pytest.mark.parametrize("plan", [
    {"intent": "retail_metric", "domain": "sales", "group_by": ["region"], "metrics": ["net_revenue"]},
    {"intent": "retail_metric", "domain": "inventory", "group_by": ["region"], "metrics": ["stock_out_rate"], "filters": [{"field": "category", "op": "eq", "value": "T-Shirt"}]},
    {"intent": "aggregate", "dataset": "orders", "joins": ["products"], "group_by": ["products.category"], "metrics": [{"agg": "count", "as": "n"}]},
    {"intent": "select", "dataset": "stores", "select": ["store_id"]},
    {"intent": "distinct", "dataset": "orders", "group_by": ["status"]},
    {"intent": "describe", "dataset": "orders", "select": ["quantity"]},
], ids=["sales", "inventory", "aggregate", "select", "distinct", "describe"])
def test_every_answer_type_carries_the_full_evidence_contract(sample, plan):
    out = ok(plan, sample)
    ev = out["evidence"]
    assert [k for k in REQUIRED_EVIDENCE if k not in ev] == []
    assert ev["columns_used"] and all("." in c for c in ev["columns_used"]) and all(isinstance(v, str) for v in ev["datasets"].values())
    assert ev["rows_considered"] <= ev["rows_total"] and ev["preview_limit"] <= 200 and out["result"]["row_count"] <= ev["preview_limit"]
    assert any(c.startswith("inventory.stock_on_hand") for c in ev["columns_used"]) or plan.get("domain") != "inventory"
