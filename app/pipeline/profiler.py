"""Dataset profiler. Pure function of a DataFrame: no mutation, no I/O, deterministic output.

Null handling (documented decision):
  null_count              -> truly empty cells (NaN)
  placeholder_null_count  -> blank/whitespace cells and tokens like 'N/A', 'null', '-'
  Both are excluded from type inference and from parse-failure counts, but reported separately
  so the cleaner can decide what to do. Nothing is changed here.

Privacy: columns that look like PII get their sample values redacted in the profile.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

from app.pipeline.parsers import NULL_TOKENS, parse_date, parse_number

PROFILE_VERSION = "1.0"
TYPE_THRESHOLD = 0.8      # >=80% of present values parse -> column has that type
MIXED_THRESHOLD = 0.2     # 20-80% parse -> 'mixed'
MAX_EXAMPLES = 5

_PII_NAME = re.compile(r"email|phone|mobile|address|dob|birth|ssn|aadhaar|^(customer_|first_|last_|full_)?name$", re.I)
_ID_NAME = re.compile(r"(^|_)(id|sku|code|key)$", re.I)


def _mask(x) -> np.ndarray:
    return x.fillna(False).to_numpy(dtype=bool)


def _r(x):
    return None if x is None else round(float(x), 4)


def profile_column(name: str, series: pd.Series) -> dict:
    s = series.astype("string")
    n = len(s)
    is_null = _mask(s.isna())
    stripped = s.str.strip()
    blank = _mask(stripped == "")
    token = _mask(stripped.str.lower().isin(NULL_TOKENS))
    placeholder = (blank | token) & ~is_null
    present = s[~is_null & ~placeholder]
    n_present = len(present)

    pii = bool(_PII_NAME.search(name))
    items = sorted(((str(k), int(c)) for k, c in present.value_counts().items()), key=lambda kc: (-kc[1], kc[0]))
    unique_count = len(items)

    # ---- per-unique-value parsing (cheap even for 100k rows) ----
    num_vals, date_info = {}, {}
    for v, _ in items:
        pn = parse_number(v)
        if pn is not None:
            num_vals[v] = pn
        pd_ = parse_date(v)
        if pd_ is not None:
            date_info[v] = pd_
    cnt = dict(items)
    num_rows = sum(cnt[v] for v in num_vals)
    date_rows = sum(cnt[v] for v in date_info)

    if n_present == 0:
        ctype = "empty"
    else:
        nf, df_ = num_rows / n_present, date_rows / n_present
        if nf >= TYPE_THRESHOLD:
            all_int = all(float(x[0]).is_integer() for x in num_vals.values())
            ctype = "integer" if all_int else "float"
        elif df_ >= TYPE_THRESHOLD:
            ctype = "date"
        elif max(nf, df_) >= MIXED_THRESHOLD:
            ctype = "mixed"
        else:
            ctype = "string"

    # ---- parse failures relative to the chosen type ----
    failures = {"count": 0, "examples": []}
    if ctype in ("integer", "float"):
        bad = [(v, c) for v, c in items if v not in num_vals]
    elif ctype == "date":
        bad = [(v, c) for v, c in items if v not in date_info]
    else:
        bad = []
    failures["count"] = sum(c for _, c in bad)
    failures["examples"] = ["[redacted]"] * min(len(bad), MAX_EXAMPLES) if pii else [v for v, _ in bad[:MAX_EXAMPLES]]

    prof = {
        "name": name,
        "inferred_type": ctype,
        "non_null_count": n_present,
        "null_count": int(is_null.sum()),
        "placeholder_null_count": int(placeholder.sum()),
        "null_pct": _r(100.0 * (n - n_present) / n) if n else 0.0,
        "is_all_null": bool(n > 0 and n_present == 0),
        "unique_count": unique_count,
        "unique_ratio": _r(unique_count / n_present) if n_present else 0.0,
        "duplicate_value_rows": n_present - unique_count,
        "looks_like_identifier": bool(_ID_NAME.search(name) and n_present and unique_count / n_present >= 0.9),
        "pii_suspected": pii,
        "parse_failures": failures,
        "whitespace_issue_count": int((present != present.str.strip()).sum()) if n_present else 0,
        "top_values": [{"value": "[redacted]" if pii else v, "count": c} for v, c in items[:5]],
    }

    # ---- spelling/case variants (strings only) ----
    if ctype in ("string", "mixed") and unique_count <= 5000:
        groups = {}
        for v, c in items:
            groups.setdefault(" ".join(v.lower().split()), []).append((v, c))
        variant_groups = [
            {"key": k, "total": sum(c for _, c in vs),
             "variants": [{"value": v, "count": c} for v, c in vs[:5]]}
            for k, vs in groups.items() if len(vs) > 1
        ]
        variant_groups.sort(key=lambda g: (-g["total"], g["key"]))
        prof["case_whitespace_variant_groups"] = [] if pii else variant_groups[:5]
        prof["case_whitespace_variant_group_count"] = len(variant_groups)

    # ---- numeric detail ----
    if ctype in ("integer", "float", "mixed") and num_vals:
        vals = np.repeat([num_vals[v][0] for v in num_vals], [cnt[v] for v in num_vals])
        prof["numeric_stats"] = {
            "count": int(len(vals)), "min": _r(vals.min()), "p25": _r(np.percentile(vals, 25)),
            "median": _r(np.median(vals)), "p75": _r(np.percentile(vals, 75)),
            "max": _r(vals.max()), "mean": _r(vals.mean()), "negative_count": int((vals < 0).sum()),
        }
        prof["symbol_formatted_count"] = sum(cnt[v] for v, (_, sym) in num_vals.items() if sym)

    # ---- date detail ----
    if ctype in ("date", "mixed") and date_info:
        fmts = Counter()
        for v, (_, fmt, _) in date_info.items():
            fmts[fmt] += cnt[v]
        sure = [d for v, (d, _, amb) in date_info.items() if not amb]
        prof["date_stats"] = {
            "formats": dict(sorted(fmts.items())),
            "ambiguous_count": sum(cnt[v] for v, (_, _, amb) in date_info.items() if amb),
            "min_unambiguous": min(sure).isoformat() if sure else None,
            "max_unambiguous": max(sure).isoformat() if sure else None,
        }
    return prof


def profile_dataframe(df: pd.DataFrame, name: str = "dataset") -> dict:
    """Profile without mutating `df`."""
    n_rows = len(df)
    sdf = df.astype("string")
    norm = sdf.apply(lambda c: c.str.strip().str.lower())
    cols = [profile_column(str(c), df[c]) for c in df.columns]
    return {
        "profile_version": PROFILE_VERSION,
        "name": name,
        "row_count": n_rows,
        "column_count": int(df.shape[1]),
        "duplicate_rows": int(sdf.duplicated().sum()) if n_rows else 0,
        "duplicate_rows_after_normalization": int(norm.duplicated().sum()) if n_rows else 0,
        "all_null_columns": [c["name"] for c in cols if c["is_all_null"]],
        "columns": cols,
    }
PII_NAME = _PII_NAME  # public alias used by the cleaner
ID_NAME = _ID_NAME  # public alias used by the chat catalog
