"""Strict structured filters. Unknown fields/operators/values are rejected, never guessed.
Strings compare case-insensitively (cleaned data has one canonical spelling per value). NULL never matches
eq/ne/in/not_in/comparisons (SQL semantics); use is_null / not_null to ask about nulls."""
import pandas as pd

from app.errors import PipelineError

OPS = {"eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "between", "contains", "is_null", "not_null"}
MAX_FILTERS, MAX_IN = 10, 100


def _bad(msg, **d):
    return PipelineError("INVALID_FILTER", msg, d)


def _num(v):
    if isinstance(v, bool) or v is None:
        raise _bad("Numeric filter value expected.", value=str(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise _bad("Numeric filter value expected.", value=str(v)[:30])
    if f != f:
        raise _bad("NaN is not a valid filter value.")
    return f


def _date(v):
    try:
        return pd.to_datetime(str(v), format="%Y-%m-%d")
    except (ValueError, TypeError):
        raise _bad("Dates must be ISO YYYY-MM-DD.", value=str(v)[:30])


def apply_filters(df: pd.DataFrame, filters: list, field_map: dict):
    """field_map: public field name -> column in df. Returns (filtered_df, applied_filters_with_counts)."""
    if not isinstance(filters, list) or len(filters) > MAX_FILTERS:
        raise _bad(f"At most {MAX_FILTERS} filters are allowed.")
    applied, cur = [], df
    for f in filters:
        if not isinstance(f, dict) or set(f) - {"field", "op", "value"} or "field" not in f or "op" not in f:
            raise _bad("Each filter must be {field, op, value}.", filter=str(f)[:80])
        field, op, val = f["field"], f["op"], f.get("value")
        if field not in field_map:
            raise _bad("Unknown filter field.", field=str(field)[:40], allowed=sorted(field_map))
        if op not in OPS:
            raise _bad("Unknown filter operator.", op=str(op)[:20], allowed=sorted(OPS))
        col = field_map[field]
        if col not in cur.columns:
            raise PipelineError("FIELD_UNAVAILABLE", f"Field '{field}' needs a dataset that is not loaded.", {"field": field})
        s = cur[col]
        if op in ("is_null", "not_null"):
            mask = s.isna() if op == "is_null" else s.notna()
        else:
            if pd.api.types.is_datetime64_any_dtype(s):
                conv, ser = _date, s
            elif pd.api.types.is_numeric_dtype(s):
                conv, ser = _num, s.astype("float64")
            else:
                conv, ser = (lambda x: str(x).strip().casefold()), s.astype("string").str.strip().str.casefold()
            if op in ("in", "not_in"):
                if not isinstance(val, list) or not val or len(val) > MAX_IN:
                    raise _bad(f"'{op}' needs a list of 1-{MAX_IN} values.")
                vals = [conv(x) for x in val]
                mask = ser.isin(vals) if op == "in" else (~ser.isin(vals) & ser.notna())
            elif op == "between":
                if not isinstance(val, list) or len(val) != 2:
                    raise _bad("'between' needs [low, high].")
                lo, hi = conv(val[0]), conv(val[1])
                mask = (ser >= lo) & (ser <= hi)
            elif op == "contains":
                if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
                    raise _bad("'contains' only works on text fields.", field=field)
                mask = ser.str.contains(str(val).strip().casefold(), regex=False)
            else:
                v = conv(val)
                mask = {"eq": ser == v, "ne": (ser != v) & ser.notna(), "gt": ser > v, "gte": ser >= v, "lt": ser < v, "lte": ser <= v}[op]
        before = len(cur)
        cur = cur[mask.fillna(False).to_numpy(dtype=bool)]
        applied.append({"field": field, "op": op, "value": val, "rows_before": before, "rows_after": len(cur)})
    return cur, applied
