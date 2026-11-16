"""Safe joins. Guarantees, enforced on every call:
 1. explicit keys that exist on both sides, with compatible types
 2. right-side keys unique (else JOIN_FANOUT, never silently multiplied rows)
 3. NULL keys never match each other (pandas would match NaN with NaN)
 4. the output has exactly as many rows, in the same order, as the left input (checked, not assumed)
 5. a report: match rate, orphan keys, unreferenced dimension keys, columns added
The join is positional (index lookup), so row multiplication is impossible by construction."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.errors import PipelineError
from app.retail.relationships import DEFAULT_RIGHT_COLUMNS, get_relationship


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    return "text"


def _index(df: pd.DataFrame, keys: list):
    return pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])


@dataclass
class JoinResult:
    df: pd.DataFrame
    report: dict


def safe_left_join(left, right, left_keys, right_keys, right_name, right_columns=None, left_name="left") -> JoinResult:
    left_keys, right_keys = list(left_keys), list(right_keys)
    if len(left_keys) != len(right_keys) or not left_keys:
        raise PipelineError("JOIN_KEY_MISSING", "Join needs the same, non-zero number of keys on both sides.")
    for df, keys, side in ((left, left_keys, left_name), (right, right_keys, right_name)):
        miss = [k for k in keys if k not in df.columns]
        if miss:
            raise PipelineError("JOIN_KEY_MISSING", f"Join key column(s) missing in {side}.", {"dataset": side, "missing": miss})
    for lk, rk in zip(left_keys, right_keys):
        if _kind(left[lk]) != _kind(right[rk]):
            raise PipelineError("JOIN_KEY_TYPE_MISMATCH", "Join key types are incompatible.",
                                {"left": f"{left_name}.{lk}:{_kind(left[lk])}", "right": f"{right_name}.{rk}:{_kind(right[rk])}"})
    rcols = right_columns if right_columns is not None else [c for c in right.columns if c not in right_keys]
    missing_cols = [c for c in rcols if c not in right.columns]
    if missing_cols:
        raise PipelineError("JOIN_COLUMN_MISSING", "Requested columns are not in the right dataset.", {"missing": missing_cols})
    out_names = {c: f"{right_name}__{c}" for c in rcols}
    clash = [n for n in out_names.values() if n in left.columns]
    if clash:
        raise PipelineError("JOIN_COLUMN_COLLISION", "Joined column names already exist.", {"columns": clash})

    r = right.reset_index(drop=True)
    r_null = r[right_keys].isna().any(axis=1)
    r_valid = r[~r_null].reset_index(drop=True)
    dup = r_valid.duplicated(subset=right_keys, keep=False)
    if dup.any():
        ex = r_valid.loc[dup, right_keys].drop_duplicates().head(5)
        raise PipelineError("JOIN_FANOUT", "Right-side join keys are not unique; joining would multiply rows.",
                            {"right": right_name, "keys": right_keys, "duplicate_key_count": int(r_valid.loc[dup, right_keys].drop_duplicates().shape[0]),
                             "examples": [dict(zip(right_keys, map(str, row))) for row in ex.itertuples(index=False)]})

    n = len(left)
    l_ok = left[left_keys].notna().all(axis=1).to_numpy()
    pos = np.full(n, -1, dtype=np.int64)
    if n and len(r_valid):
        pos = _index(r_valid, right_keys).get_indexer(_index(left, left_keys))
        pos = np.where(l_ok, pos, -1)
    matched = pos >= 0
    taken = r_valid[rcols].iloc[np.where(matched, pos, 0)].reset_index(drop=True) if len(r_valid) else r_valid[rcols].reindex(range(n))
    taken = taken.where(pd.Series(matched, index=taken.index))
    taken.columns = [out_names[c] for c in rcols]
    taken.index = left.index
    out = pd.concat([left, taken], axis=1)
    if len(out) != n or not out.index.equals(left.index):          # invariant, checked not assumed
        raise PipelineError("JOIN_INVARIANT_VIOLATED", "Join changed the row count or order.")

    unmatched = l_ok & ~matched
    orphan = left.loc[unmatched, left_keys]
    top = orphan.value_counts().head(5)
    ref = set(_index(left[l_ok], left_keys)) if l_ok.any() else set()
    report = {"join": f"{left_name}({','.join(left_keys)}) -> {right_name}({','.join(right_keys)})", "how": "left",
              "cardinality": "many_to_one", "left_rows": n, "right_rows": len(right), "rows_after": len(out), "fan_out_free": True,
              "null_key_rows": int((~l_ok).sum()), "matched_rows": int(matched.sum()), "unmatched_rows": int(unmatched.sum()),
              "match_rate": round(float(matched.sum()) / n, 4) if n else None,
              "unmatched_distinct_keys": int(orphan.drop_duplicates().shape[0]),
              "unmatched_key_examples": [{"key": [str(x) for x in (idx if isinstance(idx, tuple) else (idx,))], "rows": int(c)} for idx, c in top.items()],
              "right_rows_with_null_key_ignored": int(r_null.sum()),
              "right_keys_never_referenced": int(sum(1 for k in _index(r_valid, right_keys) if k not in ref)) if len(r_valid) else 0,
              "columns_added": list(out_names.values())}
    return JoinResult(out, report)


def join_declared(frames: dict, left_name: str, right_name: str, left_df=None, right_columns=None) -> JoinResult:
    """Join through the relationship allow-list (what the chat layer must use)."""
    rel = get_relationship(left_name, right_name)
    for nm in (left_name, right_name):
        if nm not in frames:
            raise PipelineError("DATASET_NOT_LOADED", f"Dataset '{nm}' is not loaded.", {"dataset": nm})
    left = frames[left_name] if left_df is None else left_df
    cols = right_columns or [c for c in DEFAULT_RIGHT_COLUMNS.get(right_name, []) if c in frames[right_name].columns]
    return safe_left_join(left, frames[right_name], rel.left_keys, rel.right_keys, right_name, cols, left_name)
