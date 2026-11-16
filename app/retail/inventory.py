"""Inventory analytics over a stock SNAPSHOT (positions = store x sku).

Definitions:
  stock_out        stock_on_hand == 0                 low_stock   0 < stock_on_hand <= reorder_level
  stock_out_rate   stock_out_positions / positions with known stock
  units_sold_net   units on valid COMPLETED order lines for the same store+sku (all dates in orders)
  sell_through     units_sold_net / (units_sold_net + units_on_hand), over positions with known stock
  age_days         as_of - last_restock_date;  aged_90_plus = age_days >= 90
Honest limits (always reported in missing_evidence): stock has no as-of timestamp, so stock-out is a
point-in-time share, not a frequency over time, and sell-through is approximate.
Sales are pre-aggregated to (store_id, sku) BEFORE joining, so the store x sku many-to-many cannot fan out."""
import numpy as np
import pandas as pd

from app.errors import PipelineError
from app.retail.filters import apply_filters
from app.retail.joins import join_declared, safe_left_join
from app.retail.metrics import UNKNOWN, _div, _label, _sort_limit, build_enriched_orders

INV_METRICS = {
    "positions": "Number of store x sku stock positions.",
    "units_on_hand": "Sum of stock_on_hand (known values).",
    "unknown_stock_positions": "Positions with missing stock_on_hand.",
    "stock_out_positions": "Positions with stock_on_hand == 0.",
    "stock_out_rate": "stock_out_positions / positions with known stock.",
    "low_stock_positions": "Positions with 0 < stock_on_hand <= reorder_level.",
    "units_sold_net": "Units on valid completed order lines for the same store+sku (positions with known stock).",
    "sell_through": "units_sold_net / (units_sold_net + units_on_hand).",
    "avg_age_days": "Mean days since last restock (known dates).",
    "aged_90_plus_positions": "Positions restocked 90+ days before as_of.",
    "unknown_age_positions": "Positions with no usable last_restock_date.",
}
INV_DIMENSIONS = {"region": "stores__region", "store": "stores__store_name", "store_id": "store_id", "city": "stores__city",
                  "category": "products__category", "brand": "products__brand", "product": "products__product_name", "sku": "sku"}
INV_FILTER_FIELDS = {**INV_DIMENSIONS, "stock_on_hand": "stock_on_hand", "reorder_level": "reorder_level", "last_restock_date": "last_restock_date"}


def applied_fields(filters):
    return [f for f in filters if isinstance(f, dict) and f.get("field") in INV_FILTER_FIELDS]


def inventory_metrics(data, group_by=None, metrics=None, filters=None, as_of=None, sort_by=None, direction="desc", limit=50):
    group_by = group_by or []
    if not isinstance(group_by, list) or len(group_by) > 3 or len(set(group_by)) != len(group_by) or any(g not in INV_DIMENSIONS for g in group_by):
        raise PipelineError("INVALID_GROUP_BY", "Unknown or too many dimensions.", {"allowed": sorted(INV_DIMENSIONS)})
    metrics = list(metrics or INV_METRICS)
    if bad := [m for m in metrics if m not in INV_METRICS]:
        raise PipelineError("INVALID_METRIC", "Unknown metric.", {"unknown": [str(b)[:30] for b in bad], "allowed": sorted(INV_METRICS)})
    sort_by = sort_by or metrics[0]
    if sort_by not in metrics:
        raise PipelineError("INVALID_SORT", "sort_by must be one of the requested metrics.")

    inv = data.get("inventory")
    if inv is None:
        raise PipelineError("DATASET_NOT_LOADED", "Inventory metrics need an 'inventory' dataset.", {"dataset": "inventory"})
    miss = [c for c in ("store_id", "sku", "stock_on_hand") if c not in inv.columns]
    if miss:
        raise PipelineError("COLUMNS_MISSING", "inventory is missing required columns.", {"missing": miss})
    df, joins, skipped, missing = inv.reset_index(drop=True).copy(), [], [], []
    for right in ("products", "stores"):
        if right not in data.frames:
            skipped.append({"join": f"inventory->{right}", "reason": f"dataset '{right}' is not loaded"})
            continue
        try:
            res = join_declared(data.frames, "inventory", right, left_df=df)
        except PipelineError as e:
            if e.code in ("JOIN_KEY_MISSING", "JOIN_COLUMN_MISSING"):
                skipped.append({"join": f"inventory->{right}", "reason": e.message})
                continue
            raise
        df = res.df
        joins.append(res.report)

    # --- sales, pre-aggregated to the inventory grain (no fan-out possible) ---
    sold_total = None
    if "orders" in data.frames:
        enr, _ = build_enriched_orders(data)
        ok = enr["_value_valid"] & (enr["_state"] == "completed") & enr["store_id"].notna() & enr["sku"].notna()
        agg = enr[ok].groupby(["store_id", "sku"], as_index=False)["_qty"].sum().rename(columns={"_qty": "units_sold_net"})
        sold_total = float(agg["units_sold_net"].sum())
        res = safe_left_join(df, agg, ["store_id", "sku"], ["store_id", "sku"], "sales", ["units_sold_net"], "inventory")
        df = res.df
        joins.append(res.report)
        inv_keys = set(zip(df["store_id"], df["sku"]))
        orphan_units = float(agg.loc[[k not in inv_keys for k in zip(agg["store_id"], agg["sku"])], "units_sold_net"].sum())
        if orphan_units:
            missing.append(f"{int(orphan_units)} net units sold in store+sku combinations with no inventory record are not in sell-through.")
        df["_sold"] = df["sales__units_sold_net"].astype("float64").fillna(0.0)
    else:
        df["_sold"] = 0.0
        missing.append("orders not loaded: units_sold_net and sell_through are not computed (0 / null).")

    dup = df.duplicated(subset=["store_id", "sku"], keep="first") & df["store_id"].notna() & df["sku"].notna()
    if dup.any():
        df.loc[dup, "_sold"] = 0.0
        missing.append(f"{int(dup.sum())} duplicate store+sku positions: sales attributed to the first occurrence only.")

    # --- age ---
    has_date = "last_restock_date" in df.columns
    if as_of is not None:
        try:
            as_of_ts = pd.to_datetime(str(as_of), format="%Y-%m-%d")
        except (ValueError, TypeError):
            raise PipelineError("INVALID_AS_OF", "as_of must be an ISO date YYYY-MM-DD.", {"value": str(as_of)[:30]})
        as_of_src = "provided"
    else:
        cands = [df["last_restock_date"].max()] if has_date else []
        if "orders" in data.frames and "order_date" in data.frames["orders"].columns:
            cands.append(data.frames["orders"]["order_date"].max())
        cands = [c for c in cands if pd.notna(c)]
        as_of_ts, as_of_src = (max(cands) if cands else None), "latest date found in the data (not today's date)"
    if has_date and as_of_ts is not None:
        age = (as_of_ts - df["last_restock_date"]).dt.days.astype("float64")
        future = int((age < 0).sum())
        age = age.where(age >= 0)
        if future:
            missing.append(f"{future} positions have last_restock_date after as_of; their age is unknown.")
    else:
        age = pd.Series(np.nan, index=df.index)
        missing.append("No last_restock_date: ageing metrics are unknown.")
    df["_age"] = age

    d, applied = apply_filters(df, filters or [], INV_FILTER_FIELDS)
    d = d.copy()
    cols = [INV_DIMENSIONS[g] for g in group_by]
    for g, c in zip(group_by, cols):
        if c not in d.columns:
            raise PipelineError("DIMENSION_UNAVAILABLE", f"Dimension '{g}' needs a dataset that is not loaded.", {"dimension": g})
        d[c] = _label(d[c])

    stock = d["stock_on_hand"].astype("float64")
    reorder = d["reorder_level"].astype("float64") if "reorder_level" in d.columns else pd.Series(np.nan, index=d.index)
    known = stock.notna()
    d["_pos"], d["_known"], d["_stock"] = 1.0, known.astype(float), stock.fillna(0.0)
    d["_out"], d["_low"] = (known & (stock == 0)).astype(float), (known & reorder.notna() & (stock > 0) & (stock <= reorder)).astype(float)
    d["_soldk"] = d["_sold"].where(known, 0.0)
    d["_agen"], d["_agesum"] = d["_age"].notna().astype(float), d["_age"].fillna(0.0)
    d["_aged"] = (d["_age"] >= 90).astype(float)
    d["_all"] = "all"
    sums = ["_pos", "_known", "_stock", "_out", "_low", "_soldk", "_agen", "_agesum", "_aged"]

    def table(keys):
        s = d.groupby(keys, dropna=False)[sums].sum()
        out = []
        for idx, r in s.iterrows():
            idx = idx if isinstance(idx, tuple) else (idx,)
            m = {"positions": int(r["_pos"]), "units_on_hand": int(r["_stock"]), "unknown_stock_positions": int(r["_pos"] - r["_known"]),
                 "stock_out_positions": int(r["_out"]), "stock_out_rate": _div(r["_out"], r["_known"]), "low_stock_positions": int(r["_low"]),
                 "units_sold_net": int(r["_soldk"]), "sell_through": _div(r["_soldk"], r["_soldk"] + r["_stock"]),
                 "avg_age_days": _div(r["_agesum"], r["_agen"], 1), "aged_90_plus_positions": int(r["_aged"]),
                 "unknown_age_positions": int(r["_pos"] - r["_agen"])}
            out.append({"group": dict(zip(keys, idx)), "metrics": {k: m[k] for k in metrics}, "positions": int(r["_pos"])})
        return out

    totals = table(["_all"])[0]["metrics"] if len(d) else {k: None for k in metrics}
    rows = [{"group": {g: r["group"][c] for g, c in zip(group_by, cols)}, "metrics": r["metrics"], "positions": r["positions"]}
            for r in (table(cols) if group_by and len(d) else [])]
    groups, n_groups, truncated = _sort_limit(rows, sort_by, direction, limit, group_by) if rows else ([], 0, False)

    missing.insert(0, "stock_on_hand is a snapshot with no as-of timestamp: stock_out_rate is a point-in-time share (not a frequency over time) and sell_through is approximate.")
    for j in joins:
        if j["unmatched_rows"] and j["join"].startswith("inventory(sku)"):
            missing.append(f"{j['unmatched_rows']} inventory positions have no match in products "
                           f"(e.g. {', '.join('/'.join(x['key']) for x in j['unmatched_key_examples'][:3])}); shown as '{UNKNOWN}' for product dimensions.")
    for sk in skipped:
        missing.append(f"{sk['join']} skipped: {sk['reason']}")

    used = {"inventory.store_id", "inventory.sku", "inventory.stock_on_hand"}
    used |= {f"inventory.{c}" for c in ("reorder_level", "last_restock_date") if c in inv.columns}
    if "orders" in data.frames:
        used |= {f"orders.{c}" for c in ("order_id", "store_id", "sku", "quantity", "unit_price", "discount", "status") if c in data.frames["orders"].columns}
    used |= {(c.replace("__", ".") if "__" in c else f"inventory.{c}") for c in cols}
    if filters:
        used |= {(INV_FILTER_FIELDS[f["field"]].replace("__", ".") if "__" in INV_FILTER_FIELDS[f["field"]] else f"inventory.{INV_FILTER_FIELDS[f['field']]}") for f in applied_fields(filters)}

    return {"kind": "inventory_metrics", "group_by": group_by, "metrics": metrics, "sort": {"by": sort_by, "dir": direction},
            "groups": groups, "group_count": n_groups, "truncated": truncated, "totals": totals,
            "evidence": {"datasets": {k: v for k, v in data.versions.items() if k in ("inventory", "orders", "products", "stores")},
                         "columns_used": sorted(used), "filters": applied, "joins": joins, "joins_skipped": skipped,
                         "as_of": None if as_of_ts is None else as_of_ts.date().isoformat(), "as_of_source": as_of_src,
                         "operations": (["filter"] if filters else []) + ["join_sales_at_store_sku_grain", "aggregate"],
                         "rows_total": len(df), "rows_considered": len(d), "formulas": {m: INV_METRICS[m] for m in metrics},
                         "missing_evidence": missing}}
