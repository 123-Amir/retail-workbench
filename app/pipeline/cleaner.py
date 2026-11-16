"""Deterministic cleaner. Executes the cleaning plan step by step.

Guarantees:
- The input DataFrame is never mutated.
- Every executor is a pure function (df, step, fields) -> (new_df, log). If it raises, the step is
  marked 'failed', the data is exactly as before that step, and the run continues.
- Every executor logs cells_changed / cells_nulled / rows_removed. validate.py cross-checks these
  logs against the real before/after frames, so silent loss is detected, not trusted.
- Only the plan decides what happens. Steps flagged 'flag_only' never touch data.
"""
import copy
from collections import Counter
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from app.errors import PipelineError
from app.pipeline.analyze import analyze_dataframe
from app.pipeline.issues import SKU_RE, detect_issues, hid
from app.pipeline.parsers import NULL_TOKENS, NUMDATE, parse_date, parse_number
from app.pipeline.profiler import PII_NAME, profile_dataframe
from app.pipeline.validate import validate_cleaning

MAX_EXAMPLES = 5


def fmt_number(v: float) -> str:
    """Shortest stable decimal text: 1299.0 -> '1299', 0.1 -> '0.1'. Re-parsing gives the same text."""
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.10f}".rstrip("0").rstrip(".")


def _remap(series: pd.Series, fn):
    """Apply fn(value)->(out_or_None, tag_or_None) once per unique value (fast for 100k+ rows)."""
    uniq = series.dropna().unique()
    res = {u: fn(u) for u in uniq}
    new = series.map({u: r[0] for u, r in res.items()}).astype(object)
    new = new.where(new.notna(), np.nan)
    vc = series.value_counts().to_dict()
    tags = {}
    tag_of = {u: r[1] for u, r in res.items() if r[1]}
    for u, (_, tag) in sorted(res.items(), key=lambda kv: (-vc[kv[0]], str(kv[0]))):
        if tag:
            t = tags.setdefault(tag, {"cells": 0, "examples": []})
            t["cells"] += int(vc[u])
            if len(t["examples"]) < MAX_EXAMPLES:
                t["examples"].append(u)
    return new, tags, tag_of


def _log(old: pd.Series, new: pd.Series, tags=None, redact=False) -> dict:
    both_na = old.isna() & new.isna()                       # no sentinel strings: NUL bytes break pandas string hashing
    changed = ~both_na & ~old.eq(new).fillna(False).astype(bool)
    log = {"cells_changed": int(changed.sum()), "cells_nulled": int((new.isna() & old.notna()).sum()), "tags": tags or {}}
    ex = []
    if not redact:
        for i in changed[changed].index[:3]:
            ex.append({"row": int(i) + 2, "before": old[i], "after": None if pd.isna(new[i]) else new[i]})
    log["examples"] = ex
    if redact:
        log["tags"] = {t: {"cells": v["cells"], "examples": ["[redacted]"] * len(v["examples"])} for t, v in log["tags"].items()}
    return log


LOSSY_TAGS = {"parse_failure", "unresolved_ambiguous"}   # values destroyed by policy (placeholders are not lossy)


def _column_step(df, fields, fn):
    col = fields[0]
    old = df[col]
    new, tags, tag_of = _remap(old, fn)
    out = df.copy()
    out[col] = new
    log = _log(old, new, tags, redact=bool(PII_NAME.search(col)))
    lossy_vals = {u for u, tg in tag_of.items() if tg in LOSSY_TAGS}
    if lossy_vals:
        log["_lossy"] = old.where(old.isin(lossy_vals))      # raw text of destroyed cells (internal, popped by _execute)
    return out, log


# ---------------- executors ----------------
def ex_rename(df, step, fields):
    mapping = step["params"]["mapping"]
    return df.rename(columns=mapping), {"columns_renamed": len(mapping), "cells_changed": 0, "cells_nulled": 0, "tags": {}}


def ex_trim(df, step, fields):
    return _column_step(df, fields, lambda v: (v.strip(), None))


def ex_null_placeholders(df, step, fields):
    tokens = set(step["params"]["tokens"])
    def fn(v):
        t = v.strip().lower()
        return (None, "placeholder") if t == "" or t in tokens else (v, None)
    return _column_step(df, fields, fn)


def ex_sku(df, step, fields):
    def fn(v):
        m = SKU_RE.match(v.strip())
        return (f"{m[1].upper()}-{m[2]}", None) if m else (v, "unmatched_kept")
    return _column_step(df, fields, fn)


def ex_categories(df, step, fields):
    mapping = step["params"]["mapping"]
    return _column_step(df, fields, lambda v: (mapping.get(v.strip(), v), None))


def ex_numbers(df, step, fields):
    def fn(v):
        t = v.strip()
        if t == "" or t.lower() in NULL_TOKENS:
            return None, "placeholder"
        r = parse_number(v)
        return (None, "parse_failure") if r is None else (fmt_number(r[0]), None)
    return _column_step(df, fields, fn)


def ex_dates(df, step, fields):
    order = step["params"]["separator_order"]
    def fn(v):
        t = v.strip()
        if t == "" or t.lower() in NULL_TOKENS:
            return None, "placeholder"
        p = parse_date(v)
        if p is None:
            return None, "parse_failure"
        d, _, amb = p
        if amb:
            m = NUMDATE.match(t)
            o = order.get("slash" if m[2] == "/" else "dash")
            if o is None:
                return None, "unresolved_ambiguous"
            a, b, y = int(m[1]), int(m[3]), int(m[4])
            d = date(y, b, a) if o == "dmy" else date(y, a, b)
        return d.isoformat(), None
    return _column_step(df, fields, fn)


def ex_drop_duplicates(df, step, fields):
    mask = df.duplicated(keep="first")
    removed = df.index[mask]
    return df[~mask].copy(), {"rows_removed": int(mask.sum()), "cells_changed": 0, "cells_nulled": 0, "tags": {},
                              "removed_row_numbers_sample": [int(i) + 2 for i in removed[:20]]}


EXEC = {"rename_columns": ex_rename, "trim_whitespace": ex_trim, "null_placeholders": ex_null_placeholders,
        "normalize_sku": ex_sku, "standardize_categories": ex_categories, "parse_numbers": ex_numbers,
        "parse_dates": ex_dates, "drop_exact_duplicates": ex_drop_duplicates}


def _check_decisions(decisions):
    decisions = decisions or {}
    for k, v in decisions.items():
        if v not in ("apply", "skip"):
            raise PipelineError("INVALID_DECISION", "Decision must be 'apply' or 'skip'.", {"step_id": k, "value": str(v)[:20]})
    return decisions


def _lossy_merged(df, lossy):
    """Rows removed as duplicates ONLY because a value was nulled by policy (they differed in the raw text)."""
    if not lossy:
        return 0, []
    shadow = df.copy()
    for col, raw in lossy.items():
        sentinel = raw.map(lambda x: f"\u27e6LOSSY\u27e7:{x}", na_action="ignore")
        shadow[col] = shadow[col].where(raw.isna(), sentinel)
    extra = df.duplicated(keep="first") & ~shadow.duplicated(keep="first")
    return int(extra.sum()), [int(i) + 2 for i in df.index[extra][:20]]


def _execute(df, name, decisions):
    analysis = analyze_dataframe(df, name)
    steps = copy.deepcopy(analysis["cleaning_plan"])
    st = {"work": df.copy(deep=True), "null_pre": None, "rename": {}, "lossy": {}}

    def run(s):
        if s["status"] == "flag_only":
            s["final_status"], s["execution"] = "flagged", {"effective": False}
            return
        decision = decisions.get(s["id"], s["default_action"])
        s["decision"], s["decision_source"] = decision, ("override" if s["id"] in decisions else "default")
        if decision != "apply":
            s["final_status"], s["execution"] = "skipped", {"effective": False}
            return
        if s["action"] == "drop_exact_duplicates" and st["null_pre"] is None:
            st["null_pre"] = st["work"].isna().sum().tolist()
        fields = [st["rename"].get(f, f) for f in s["fields"]]
        try:
            work2, log = EXEC[s["action"]](st["work"], s, fields)
            if "_lossy" in log:
                st["lossy"][fields[0]] = log.pop("_lossy")
            if s["action"] == "drop_exact_duplicates":
                n, rows = _lossy_merged(st["work"], st["lossy"])
                log["rows_removed_due_to_lossy_nulling"], log["lossy_merged_row_numbers_sample"] = n, rows
            log["effective"] = any(log.get(k, 0) for k in ("cells_changed", "rows_removed", "columns_renamed"))
            st["work"], s["final_status"], s["execution"] = work2, "applied", log
            if s["action"] == "rename_columns":
                st["rename"] = dict(s["params"]["mapping"])
        except Exception as e:  # step is isolated: data stays as it was before this step
            s["final_status"] = "failed"
            s["execution"] = {"effective": False, "error": {"type": type(e).__name__, "message": str(e)[:200]}}

    for s in steps:
        run(s)

    # Cleaning itself can create duplicates (e.g. two rows whose bad dates were nulled). Handle them
    # visibly so the output is a fixed point (idempotent) - unless the reviewer skipped dedupe.
    extra = []
    if not any(s["action"] == "drop_exact_duplicates" for s in steps):
        k = int(st["work"].duplicated().sum())
        if k:
            iss = {"id": hid("ISS", name, "DUPLICATES_CREATED_BY_CLEANING", "*"), "dataset": name,
                   "type": "DUPLICATES_CREATED_BY_CLEANING", "severity": "medium", "columns": [], "affected_rows": k,
                   "summary": "Rows became identical during cleaning (values were normalised or nulled).",
                   "evidence": {"rows": k}, "detected_by": "code", "detected_at": "post_cleaning"}
            extra.append(iss)
            s = {"id": hid("STP", name, "drop_exact_duplicates", "*"), "dataset": name, "action": "drop_exact_duplicates",
                 "fields": [], "issue_ids": [iss["id"]], "risk": "medium", "source": "code", "status": "planned",
                 "default_action": "apply", "params": {"keep": "first", "subset": "all_columns"},
                 "reason": "Remove rows that became identical during cleaning, so re-running the cleaner is stable. Merged rows are reported.",
                 "expected_effect": {"rows_removed_min": k}, "final_status": "pending", "dynamic": True,
                 "order": len(steps) + 1}
            steps.append(s)
            run(s)
    if st["null_pre"] is None:
        st["null_pre"] = st["work"].isna().sum().tolist()
    return st["work"], steps, st["null_pre"], analysis, extra


@dataclass
class CleanResult:
    df: pd.DataFrame
    report: dict


def clean_dataframe(df: pd.DataFrame, name: str, decisions: dict | None = None, check_idempotent: bool = True) -> CleanResult:
    decisions = _check_decisions(decisions)
    cleaned, steps, null_pre, analysis, extra_issues = _execute(df, name, decisions)
    cleaned = cleaned.reset_index(drop=True)
    profile_after = profile_dataframe(cleaned, name)
    issues_after = detect_issues(cleaned, profile_after)

    idem = {"checked": False}
    if check_idempotent:
        c2, steps2, *_ = _execute(cleaned, name, decisions)
        c2 = c2.reset_index(drop=True)
        effective = [s["id"] for s in steps2 if s.get("execution", {}).get("effective")]
        idem = {"checked": True, "stable": bool(c2.equals(cleaned) and list(c2.columns) == list(cleaned.columns) and not effective),
                "second_run_effective_steps": effective}

    validation = validate_cleaning(df, cleaned, steps, null_pre, issues_after, idem)
    report = {
        "dataset": {"name": name, "rows_in": len(df), "rows_out": len(cleaned), "columns": list(cleaned.columns)},
        "profile_before": analysis["profile_before"], "issues": analysis["issues"] + extra_issues, "cleaning_plan": steps,
        "profile_after": profile_after, "issues_after": issues_after, "validation": validation,
        "schema_after": {c["name"]: c["inferred_type"] for c in profile_after["columns"]},
    }
    return CleanResult(df=cleaned, report=report)
check_decisions = _check_decisions  # public alias used by the run service
