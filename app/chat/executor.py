"""Executes ONLY validated plans and returns {answer, result, evidence}. Every number in the result comes from
pandas/retail code run here; the answer text is a deterministic template over that result (no model involved).

Bounds: row-scan cap, limit cap (validator), join cap, per-query wall-clock timeout, no dynamic code, no SQL.
Note: the timeout is a soft deadline (a Python thread cannot be killed); the row cap bounds the worst case."""
import math
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime

import numpy as np
import pandas as pd

from app.chat.catalog import kind_of
from app.chat.plan import ClarifyPlan, QueryPlan, RefusePlan, RetailPlan, parse_plan
from app.chat.validator import Validated, validate_plan
from app.config import settings
from app.errors import PipelineError
from app.pipeline.issues import INJECTION_RE
from app.retail.filters import apply_filters
from app.retail.inventory import inventory_metrics
from app.retail.joins import join_declared
from app.retail.metrics import sales_metrics

MAX_ROWS_SCANNED = 2_000_000
PREVIEW_ROWS_IN_ANSWER = 5
_POOL = ThreadPoolExecutor(max_workers=4)
_FUNCS = {"sum": "sum", "avg": "mean", "min": "min", "max": "max", "median": "median", "count_distinct": "nunique", "count": "count"}


def _py(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 4)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.date().isoformat() if (v.hour, v.minute, v.second) == (0, 0, 0) else v.isoformat()
    return str(v)


def _table(df: pd.DataFrame, total: int, limit_hit: bool, types: dict | None = None) -> dict:
    cols = [{"name": c, "type": (types or {}).get(c) or kind_of(df[c])} for c in df.columns]
    rows = [{c: _py(v) for c, v in zip(df.columns, rec)} for rec in df.itertuples(index=False, name=None)]
    return {"columns": cols, "rows": rows, "row_count": len(rows), "total_rows": int(total), "truncated": bool(limit_hit)}


def _sorted(df, sort, ascending_default=True):
    if not sort:
        return df
    return df.sort_values(by=[s[0] for s in sort], ascending=[s[1] == "asc" for s in sort], kind="mergesort", na_position="last")


# ------------------------------------------------------------------------------ generic queries
def _aggregate(df, p, fmap):
    keys = [fmap[g] for g in p.group_by]
    if keys:
        g = df.groupby(keys, dropna=False, sort=True)
        out = pd.concat({m.as_: (g.size() if m.field is None else g[fmap[m.field]].agg(_FUNCS[m.agg])) for m in p.metrics}, axis=1)
        out = out.reset_index().rename(columns={fmap[n]: n for n in p.group_by})
        out = out.sort_values(by=list(p.group_by), kind="mergesort", na_position="last")
    else:
        row = {m.as_: (len(df) if m.field is None else df[fmap[m.field]].agg(_FUNCS[m.agg])) for m in p.metrics}
        out = pd.DataFrame([row])
    return _sorted(out, [(s.field, s.dir) for s in p.sort])


def _describe(df, p, fmap):
    rows = []
    for name in p.select:
        s = df[fmap[name]]
        k = kind_of(s)
        r = {"field": name, "type": k, "count": int(s.notna().sum()), "nulls": int(s.isna().sum()), "unique": int(s.nunique())}
        if k == "number":
            x = s.astype("float64")
            r.update(min=x.min(), p25=x.quantile(0.25), median=x.median(), p75=x.quantile(0.75), max=x.max(), mean=x.mean(), std=x.std())
        elif k == "date":
            r.update(min=s.min(), max=s.max())
        else:
            r["top_values"] = [{"value": str(v)[:60], "count": int(c)} for v, c in s.value_counts().head(5).items()]
        rows.append(r)
    return rows


def _run_query(v: Validated, data):
    p = v.plan
    df = data.frames[p.dataset]
    if len(df) > MAX_ROWS_SCANNED:
        raise PipelineError("EXECUTION_TOO_EXPENSIVE", "Dataset is larger than the allowed scan size.", {"rows": len(df), "max": MAX_ROWS_SCANNED})
    reports, ops = [], []
    for j in p.joins:
        res = join_declared(data.frames, p.dataset, j, left_df=df, right_columns=v.right_columns[j])
        df = res.df
        reports.append(res.report)
        ops.append(f"join:{j}")
    fmap = {r.public: r.column for r in v.fields.values()}
    rows_total = len(data.frames[p.dataset])
    df, applied = apply_filters(df, [f.model_dump() for f in p.filters], fmap)
    if applied:
        ops.append("filter")

    if p.intent == "aggregate":
        out = _aggregate(df, p, fmap)
        ops += (["group_by"] if p.group_by else []) + [f"aggregate:{m.agg}({m.field or '*'})" for m in p.metrics] + (["sort"] if p.sort else [])
        total = len(out)
        res_tbl = _table(out.head(v.limit), total, total > v.limit)
    elif p.intent == "select":
        d = _sorted(df.assign(), [(fmap[s.field], s.dir) for s in p.sort])
        out = d[[fmap[s] for s in p.select]].rename(columns={fmap[s]: s for s in p.select})
        ops += (["sort"] if p.sort else []) + ["select"]
        res_tbl = _table(out.head(v.limit), len(out), len(out) > v.limit)
    elif p.intent == "distinct":
        keys = [fmap[g] for g in p.group_by]
        out = df.groupby(keys, dropna=False).size().reset_index(name="count").rename(columns={fmap[n]: n for n in p.group_by})
        out = out.sort_values(by=["count"] + list(p.group_by), ascending=[False] + [True] * len(keys), kind="mergesort", na_position="last")
        out = _sorted(out, [(s.field, s.dir) for s in p.sort]) if p.sort else out
        ops += ["distinct", "count"]
        res_tbl = _table(out.head(v.limit), len(out), len(out) > v.limit)
    else:
        rows = _describe(df, p, fmap)
        ops.append("describe")
        res_tbl = {"columns": [{"name": c, "type": "text"} for c in ("field", "type", "count", "nulls", "unique", "min", "p25", "median", "p75", "max", "mean", "std", "top_values")],
                   "rows": [{k: (x if k == "top_values" else _py(x)) for k, x in r.items()} for r in rows],
                   "row_count": len(rows), "total_rows": len(rows), "truncated": False}
    used = sorted({f"{r.dataset}.{r.column.split('__', 1)[-1]}" for r in v.fields.values()})
    evidence = {"datasets": {n: data.versions.get(n) for n in [p.dataset] + list(p.joins)}, "columns_used": used, "filters": applied,
                "joins": reports, "operations": ops, "rows_total": rows_total, "rows_considered": len(df), "preview_limit": v.limit}
    return res_tbl, evidence


# ------------------------------------------------------------------------------ retail metrics
def _run_retail(v: Validated, data):
    p = v.plan
    kw = dict(group_by=p.group_by, metrics=p.metrics or None, filters=[f.model_dump() for f in p.filters],
              sort_by=p.sort_by, direction=p.direction, limit=v.limit)
    res = sales_metrics(data, **kw) if p.domain == "sales" else inventory_metrics(data, as_of=p.as_of, **kw)
    names = res["metrics"]
    rows = [{**g["group"], **g["metrics"]} for g in res["groups"]] if p.group_by else [dict(res["totals"])]
    df = pd.DataFrame(rows, columns=list(p.group_by) + names)
    tbl = _table(df, res["group_count"] if p.group_by else 1, res["truncated"], {n: "number" for n in names})
    tbl["totals"] = {k: _py(x) for k, x in res["totals"].items()}
    ev = dict(res["evidence"])
    ev.update(operations=[f"retail:{p.domain}"] + ev["operations"], preview_limit=v.limit)
    return tbl, ev


# ------------------------------------------------------------------------------ answer text
def _fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:,.4f}".rstrip("0").rstrip(".")
    if isinstance(x, list):
        return "[" + ", ".join(_fmt(i) for i in x[:5]) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}={_fmt(i)}" for k, i in x.items()) + "}"
    return str(x)[:60]


def describe_plan(p) -> str:
    flt = "".join(f" where {f.field} {f.op} {f.value}" if f.value is not None else f" where {f.field} {f.op}" for f in p.filters)
    if isinstance(p, RetailPlan):
        return f"{p.domain} metrics ({', '.join(p.metrics) or 'all'})" + (f" by {', '.join(p.group_by)}" if p.group_by else "") + flt
    if p.intent == "aggregate":
        m = ", ".join(f"{x.agg}({x.field or '*'}) as {x.as_}" for x in p.metrics)
        return f"{m} on {p.dataset}" + (f" by {', '.join(p.group_by)}" if p.group_by else "") + flt
    if p.intent == "select":
        return f"{', '.join(p.select)} from {p.dataset}" + flt
    if p.intent == "distinct":
        return f"distinct {', '.join(p.group_by)} in {p.dataset}" + flt
    return f"summary of {', '.join(p.select)} in {p.dataset}" + flt


def render_answer(p, result, evidence) -> str:
    lines = [f"{describe_plan(p)}."]
    if result["row_count"] == 0:
        lines.append("No rows matched, so there is nothing to report.")
    else:
        shown = result["rows"][:PREVIEW_ROWS_IN_ANSWER]
        lines.append(f"{result['total_rows']} result row(s)" + (f", first {len(shown)} shown" if result["total_rows"] > len(shown) else "") + ":")
        lines += ["- " + "; ".join(f"{k}={_fmt(x)}" for k, x in r.items()) for r in shown]
    vers = ", ".join(f"{k}@{str(x).split('@')[-1]}" if x else k for k, x in evidence["datasets"].items())
    lines.append(f"Data: {vers}; {evidence['rows_considered']} of {evidence['rows_total']} rows considered.")
    for m in evidence.get("missing_evidence", [])[:3]:
        lines.append(f"Caveat: {m}")
    return "\n".join(lines)


def _flags(result) -> dict:
    def strings(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for i in x.values():
                yield from strings(i)
        elif isinstance(x, list):
            for i in x:
                yield from strings(i)
    return {"suspected_prompt_injection_in_results": any(INJECTION_RE.search(s) for s in strings(result["rows"])),
            "note": "Cell text is untrusted data and is never treated as an instruction."}


def _execute(v: Validated, data):
    return _run_retail(v, data) if isinstance(v.plan, RetailPlan) else _run_query(v, data)


# ------------------------------------------------------------------------------ public entry point
def answer_plan(raw, data, timeout: float | None = None) -> dict:
    """parse -> validate -> execute -> evidence. Never raises for bad plans: returns status 'error' with a structured error."""
    out = {"status": "error", "trace": {"stages_completed": []}}
    try:
        plan = parse_plan(raw)
        out["trace"]["stages_completed"].append("parse")
        v = validate_plan(plan, data)
        p = v.plan
        out["plan"] = p.model_dump(by_alias=True)
        out["trace"]["stages_completed"].append("validate")
        if isinstance(p, ClarifyPlan):
            out.update(status="clarification_needed", clarification={"question": p.question, "options": p.options}, answer=p.question)
            return out
        if isinstance(p, RefusePlan):
            out.update(status="unanswerable", reason=p.reason, missing=p.missing,
                       answer=p.reason + (" Missing: " + ", ".join(p.missing) + "." if p.missing else ""))
            return out
        fut = _POOL.submit(_execute, v, data)
        try:
            result, evidence = fut.result(timeout=timeout if timeout is not None else settings.query_timeout)
        except FutureTimeout:
            fut.cancel()
            raise PipelineError("QUERY_TIMEOUT", "Query exceeded its time budget.", {"seconds": timeout if timeout is not None else settings.query_timeout})
        out["trace"]["stages_completed"].append("execute")
        out.update(status="ok", result=result, evidence=evidence, flags=_flags(result), answer=render_answer(p, result, evidence))
    except PipelineError as e:
        out["error"] = e.to_dict()
    except Exception as e:   # never leak internals to the caller
        out["error"] = {"code": "INTERNAL_ERROR", "message": "Unexpected error while answering.", "details": {"type": type(e).__name__}}
    return out
