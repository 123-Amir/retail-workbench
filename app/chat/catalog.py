"""What the chat layer is allowed to talk about, derived from the loaded data (never hand-written).
Used by the validator (allow-list) and, in Step 8, to build a bounded, PII-free prompt for the LLM."""
import pandas as pd

from app.config import settings
from app.pipeline.profiler import ID_NAME, PII_NAME
from app.retail.inventory import INV_DIMENSIONS, INV_METRICS
from app.retail.metrics import DIMENSIONS, METRICS
from app.retail.relationships import RELATIONSHIPS

MAX_DISTINCT_VALUES, MAX_VALUE_CHARS = 30, 40


def kind_of(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "text"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    return "text"


def is_pii(column: str) -> bool:
    return bool(PII_NAME.search(column))


def build_catalog(data) -> dict:
    datasets = {}
    for name, df in sorted(data.frames.items()):
        cols = []
        for c in df.columns:
            k, pii = kind_of(df[c]), is_pii(c)
            entry = {"name": c, "type": k, "pii": pii}
            if k == "text" and not pii and not ID_NAME.search(c) and df[c].nunique() <= MAX_DISTINCT_VALUES:   # identifiers are row-level, never labels
                entry["distinct_values"] = [str(v)[:MAX_VALUE_CHARS] for v in sorted(df[c].dropna().unique(), key=str)]
            cols.append(entry)
        datasets[name] = {"rows": len(df), "version": data.versions.get(name), "columns": cols}
    return {"datasets": datasets,
            "joins": [f"{r.left} -> {r.right} on {','.join(r.left_keys)} ({r.cardinality})" for r in RELATIONSHIPS
                      if r.left in data.frames and r.right in data.frames],
            "retail": {"sales": {"needs": ["orders"], "dimensions": {k: v[1] for k, v in DIMENSIONS.items()}, "metrics": METRICS},
                       "inventory": {"needs": ["inventory"], "dimensions": sorted(INV_DIMENSIONS), "metrics": INV_METRICS}},
            "aggregations": ["count", "count_distinct", "sum", "avg", "min", "max", "median"],
            "limits": {"max_rows": settings.max_result_rows, "max_filters": 10, "max_group_by": 3, "max_joins": 3}}
