"""Auditable retail metrics. Pure code over typed, cleaned frames; the LLM never computes or invents a number.

Business definitions (explicit, so every number can be reproduced by hand):
  Each orders row is one order LINE.  line_value = quantity * unit_price * (1 - discount)
  state: completed | returned | cancelled | unknown (anything else / null status)
  A line is VALUE-VALID when quantity > 0, unit_price >= 0 and 0 <= discount <= 1 (all present).
  Invalid completed/returned lines are EXCLUDED and counted - never repaired or guessed.
  gross_sales  = sum(line_value) over valid completed + returned lines
  returned_sales = sum(line_value) over valid returned lines
  net_revenue  = gross_sales - returned_sales   (= valid completed lines)
  orders       = distinct order_id of valid completed lines;  aov = net_revenue / orders
  return_rate_units = units_returned / units_shipped (completed + returned);  return_rate_value likewise on value
  cancellation_rate = cancelled lines / lines with a known status
  gross_margin = sum(line_value - quantity*cost) over valid completed lines whose cost is known;  margin_pct = that / its revenue
"""
import numpy as np
import pandas as pd

from app.errors import PipelineError
from app.retail.filters import apply_filters
from app.retail.joins import join_declared
from app.retail.relationships import RELATIONSHIPS

UNKNOWN = "(unknown)"
MAX_GROUPS, MAX_GROUP_BY = 200, 3

METRICS = {
    "gross_sales": "Sum of line value over valid completed + returned lines.",
    "returned_sales": "Sum of line value over valid returned lines.",
    "net_revenue": "gross_sales - returned_sales (valid completed lines).",
    "orders": "Distinct order_id among valid completed lines.",
    "aov": "net_revenue / orders.",
    "units_shipped": "Units on valid completed + returned lines.",
    "units_returned": "Units on valid returned lines.",
    "return_rate_units": "units_returned / units_shipped.",
    "return_rate_value": "returned_sales / gross_sales.",
    "cancellation_rate": "Cancelled lines / lines with a known status.",
    "gross_margin": "Sum(line_value - quantity*cost) over valid completed lines with known cost.",
    "margin_pct": "gross_margin / revenue of those same lines.",
    "cost_coverage": "Share of net_revenue that has a known product cost.",
}
MONEY = {"gross_sales", "returned_sales", "net_revenue", "aov", "gross_margin"}

# public dimension name -> (column in enriched orders, description)
DIMENSIONS = {
    "region": ("stores__region", "Store region"), "store": ("stores__store_name", "Store name"),
    "store_id": ("store_id", "Store id"), "city": ("stores__city", "Store city"),
    "category": ("products__category", "Product category"), "brand": ("products__brand", "Product brand"),
    "product": ("products__product_name", "Product name"), "sku": ("sku", "SKU"),
    "month": ("order_month", "Order month (YYYY-MM)"), "status": ("status", "Line status"),
    "customer_city": ("customers__city", "Customer city"),
}
FILTER_FIELDS = {**{k: v[0] for k, v in DIMENSIONS.items()}, "order_date": "order_date", "quantity": "quantity",
                 "unit_price": "unit_price", "discount": "discount"}


# ----------------------------------------------------------------------------- enrichment
def build_enriched_orders(data):
    """orders + products/stores/customers (each join validated). Missing datasets are skipped and reported."""
    orders = data.get("orders")
    if orders is None:
        raise PipelineError("DATASET_NOT_LOADED", "Sales metrics need an 'orders' dataset.", {"dataset": "orders"})
    need = ["order_id", "quantity", "unit_price", "status"]
    miss = [c for c in need if c not in orders.columns]
    if miss:
        raise PipelineError("COLUMNS_MISSING", "orders is missing columns required for sales metrics.", {"missing": miss})
    df, joins, skipped = orders.reset_index(drop=True).copy(), [], []
    for rel in [r for r in RELATIONSHIPS if r.left == "orders"]:
        if rel.right not in data.frames:
            skipped.append({"join": f"orders->{rel.right}", "reason": f"dataset '{rel.right}' is not loaded"})
            continue
        try:
            res = join_declared(data.frames, "orders", rel.right, left_df=df)
        except PipelineError as e:
            if e.code in ("JOIN_KEY_MISSING", "JOIN_COLUMN_MISSING"):
                skipped.append({"join": f"orders->{rel.right}", "reason": e.message, "details": e.details})
                continue
            raise                                  # JOIN_FANOUT etc. must surface, never be skipped silently
        df = res.df
        joins.append(res.report)

    qty = df["quantity"].astype("float64")
    price = df["unit_price"].astype("float64")
    disc = df["discount"].astype("float64") if "discount" in df.columns else pd.Series(0.0, index=df.index)
    status = df["status"].astype("string").str.strip().str.casefold()
    df["_state"] = status.where(status.isin(["completed", "returned", "cancelled"]), "unknown").fillna("unknown").astype(object)
    q_ok, p_ok, d_ok = (qty > 0).fillna(False), (price >= 0).fillna(False), ((disc >= 0) & (disc <= 1)).fillna(False)
    shipped = df["_state"].isin(["completed", "returned"])
    df["_value_valid"] = shipped & q_ok & p_ok & d_ok
    df["_excl"] = np.select([shipped & ~q_ok, shipped & ~p_ok, shipped & ~d_ok, df["_state"] == "unknown"],
                            ["quantity_missing_or_not_positive", "unit_price_missing_or_negative", "discount_missing_or_out_of_range", "status_unknown"], default="")
    df["_qty"] = qty
    df["line_value"] = (qty * price * (1 - disc)).where(df["_value_valid"])
    cost = df["products__cost"].astype("float64") if "products__cost" in df.columns else pd.Series(np.nan, index=df.index)
    df["_cost_known"] = df["_value_valid"] & cost.notna()
    df["line_cost"] = (qty * cost).where(df["_cost_known"])
    df["order_month"] = (df["order_date"].dt.strftime("%Y-%m") if "order_date" in df.columns else pd.Series(pd.NA, index=df.index))
    notes = {"joins": joins, "joins_skipped": skipped, "discount_column_present": "discount" in orders.columns,
             "order_date_present": "order_date" in orders.columns}
    return df, notes


# ----------------------------------------------------------------------------- helpers
def _dims(group_by):
    if not isinstance(group_by, list) or len(group_by) > MAX_GROUP_BY or len(set(group_by)) != len(group_by):
        raise PipelineError("INVALID_GROUP_BY", f"group_by must be a list of up to {MAX_GROUP_BY} distinct dimensions.")
    bad = [g for g in group_by if g not in DIMENSIONS]
    if bad:
        raise PipelineError("INVALID_GROUP_BY", "Unknown dimension.", {"unknown": [str(b)[:30] for b in bad], "allowed": sorted(DIMENSIONS)})
    return group_by


def _div(a, b, nd=4):
    return None if b in (0, None) or a is None or b != b else round(float(a) / float(b), nd)


def _r2(x):
    return round(float(x), 2)


def _label(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna(UNKNOWN).astype(object)


def _sort_limit(rows, sort_by, direction, limit, keys):
    if direction not in ("asc", "desc"):
        raise PipelineError("INVALID_SORT", "direction must be 'asc' or 'desc'.")
    limit = MAX_GROUPS if limit is None else int(limit)
    if not 1 <= limit <= MAX_GROUPS:
        raise PipelineError("LIMIT_TOO_LARGE", f"limit must be between 1 and {MAX_GROUPS}.", {"max": MAX_GROUPS})
    rows = sorted(rows, key=lambda r: tuple(str(r["group"][k]) for k in keys))
    present = [r for r in rows if r["metrics"][sort_by] is not None]
    absent = [r for r in rows if r["metrics"][sort_by] is None]
    present.sort(key=lambda r: r["metrics"][sort_by], reverse=(direction == "desc"))   # stable: ties keep key order
    ordered = present + absent
    return ordered[:limit], len(ordered), len(ordered) > limit


# ----------------------------------------------------------------------------- sales metrics
def sales_metrics(data, group_by=None, metrics=None, filters=None, sort_by=None, direction="desc", limit=50):
    group_by = _dims(group_by or [])
    metrics = list(metrics or METRICS)
    if bad := [m for m in metrics if m not in METRICS]:
        raise PipelineError("INVALID_METRIC", "Unknown metric.", {"unknown": [str(b)[:30] for b in bad], "allowed": sorted(METRICS)})
    sort_by = sort_by or ("net_revenue" if "net_revenue" in metrics else metrics[0])
    if sort_by not in metrics:
        raise PipelineError("INVALID_SORT", "sort_by must be one of the requested metrics.", {"sort_by": str(sort_by)[:30]})

    enriched, notes = build_enriched_orders(data)
    cols = [DIMENSIONS[g][0] for g in group_by]
    for g, c in zip(group_by, cols):
        if c not in enriched.columns:
            raise PipelineError("DIMENSION_UNAVAILABLE", f"Dimension '{g}' needs a dataset/column that is not loaded.", {"dimension": g})
    d, applied = apply_filters(enriched, filters or [], FILTER_FIELDS)
    d = d.copy()
    for c in cols:
        d[c] = _label(d[c])
    has_cost = "products__cost" in enriched.columns

    v, st = d["line_value"].fillna(0.0), d["_state"]
    comp, ret = d["_value_valid"] & (st == "completed"), d["_value_valid"] & (st == "returned")
    cm = comp & d["_cost_known"]
    d["_gross"], d["_ret"], d["_comp"] = v.where(comp | ret, 0.0), v.where(ret, 0.0), v.where(comp, 0.0)
    d["_u_ship"], d["_u_ret"] = d["_qty"].where(comp | ret, 0.0), d["_qty"].where(ret, 0.0)
    d["_cancel"], d["_known"] = (st == "cancelled").astype(float), (st != "unknown").astype(float)
    d["_m_rev"], d["_m_cost"] = v.where(cm, 0.0), d["line_cost"].where(cm, 0.0).fillna(0.0)
    d["_is_comp"], d["_n"] = comp, 1.0
    d["_all"] = "all"
    sums = ["_gross", "_ret", "_comp", "_u_ship", "_u_ret", "_cancel", "_known", "_m_rev", "_m_cost", "_n"]

    def table(keys):
        g = d.groupby(keys, dropna=False)
        s = g[sums].sum()
        s["_orders"] = d[d["_is_comp"]].groupby(keys, dropna=False)["order_id"].nunique().reindex(s.index).fillna(0)
        out = []
        for idx, r in s.iterrows():
            idx = idx if isinstance(idx, tuple) else (idx,)
            m = {"gross_sales": _r2(r["_gross"]), "returned_sales": _r2(r["_ret"]), "net_revenue": _r2(r["_comp"]),
                 "orders": int(r["_orders"]), "aov": _div(r["_comp"], r["_orders"], 2), "units_shipped": int(r["_u_ship"]),
                 "units_returned": int(r["_u_ret"]), "return_rate_units": _div(r["_u_ret"], r["_u_ship"]),
                 "return_rate_value": _div(r["_ret"], r["_gross"]), "cancellation_rate": _div(r["_cancel"], r["_known"]),
                 "gross_margin": _r2(r["_m_rev"] - r["_m_cost"]) if has_cost and r["_m_rev"] else None,
                 "margin_pct": _div(r["_m_rev"] - r["_m_cost"], r["_m_rev"]) if has_cost else None,
                 "cost_coverage": _div(r["_m_rev"], r["_comp"]) if has_cost else None}
            out.append({"group": dict(zip(keys, idx)), "metrics": {k: m[k] for k in metrics}, "lines": int(r["_n"])})
        return out

    totals = table(["_all"])[0]["metrics"] if len(d) else {k: None for k in metrics}
    rows = [{"group": {g: r["group"][c] for g, c in zip(group_by, cols)}, "metrics": r["metrics"], "lines": r["lines"]}
            for r in (table(cols) if group_by and len(d) else [])]
    groups, n_groups, truncated = _sort_limit(rows, sort_by, direction, limit, group_by) if rows else ([], 0, False)

    excl = d.loc[d["_excl"] != "", "_excl"].value_counts().to_dict()
    missing = []
    for j in notes["joins_skipped"]:
        missing.append(f"{j['join']} skipped: {j['reason']}")
    for j in notes["joins"]:
        if j["unmatched_rows"]:
            missing.append(f"{j['unmatched_rows']} order lines have no match in {j['join'].split('-> ')[1].split('(')[0]} "
                           f"(e.g. {', '.join('/'.join(x['key']) for x in j['unmatched_key_examples'][:3])}); they appear under '{UNKNOWN}' for dimensions from that dataset.")
        if j["null_key_rows"]:
            missing.append(f"{j['null_key_rows']} order lines have a null join key for {j['join'].split(' ->')[0]}.")
    if not has_cost:
        missing.append("Product cost unavailable: margin metrics are null.")
    elif int(d.loc[comp, "_cost_known"].eq(False).sum()):
        missing.append(f"{int(d.loc[comp, '_cost_known'].eq(False).sum())} completed lines have no product cost and are excluded from margin.")
    if not notes["discount_column_present"]:
        missing.append("No discount column: revenue is computed at unit_price.")
    if "month" in group_by and not notes["order_date_present"]:
        missing.append("No order_date column: month grouping is all unknown.")
    if excl:
        missing.append("Lines excluded from value metrics: " + ", ".join(f"{k}={v}" for k, v in sorted(excl.items())) + ".")

    return {"kind": "sales_metrics", "group_by": group_by, "metrics": metrics, "sort": {"by": sort_by, "dir": direction},
            "groups": groups, "group_count": n_groups, "truncated": truncated, "totals": totals,
            "evidence": {"datasets": {k: v for k, v in data.versions.items() if k in ("orders", "products", "stores", "customers")},
                         "columns_used": sorted({"orders.quantity", "orders.unit_price", "orders.discount", "orders.status", "orders.order_id"}
                                                | {f"orders.{c}" for c in ("order_date",) if "month" in group_by}
                                                | {c.replace("__", ".") for c in cols}),
                         "filters": applied, "joins": notes["joins"], "joins_skipped": notes["joins_skipped"],
                         "operations": ["filter", "validate_lines", "aggregate"] if filters else ["validate_lines", "aggregate"],
                         "rows_total": len(enriched), "rows_considered": len(d),
                         "exclusions": {k: int(v) for k, v in sorted(excl.items())},
                         "formulas": {m: METRICS[m] for m in metrics}, "missing_evidence": missing}}
