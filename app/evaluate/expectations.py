"""Expectation engine: turns 'expected properties / exact answers' into pass/fail checks with expected vs actual.
Every check is {name, passed, expected, actual}; actual/expected are truncated so results stay small."""
import json
import re

MISSING = object()
_TOKEN = re.compile(r'\["([^"]+)"\]|([^.\[\]]+)')


def get_path(obj, path: str):
    """'result.rows.0.net_revenue' or 'result.rows.0["products.category"]' (bracket form for keys containing dots)."""
    cur = obj
    for quoted, plain in _TOKEN.findall(path):
        key = quoted or plain
        if isinstance(cur, dict):
            if key not in cur:
                return MISSING
            cur = cur[key]
        elif isinstance(cur, list):
            try:
                cur = cur[int(key)]
            except (ValueError, IndexError):
                return MISSING
        else:
            return MISSING
    return cur


def _short(v, n=300):
    if v is MISSING:
        return "<missing>"
    s = json.dumps(v, default=str, sort_keys=True)
    return v if len(s) <= n else s[:n] + "..."


def _chk(name, passed, expected, actual):
    return {"name": name, "passed": bool(passed), "expected": _short(expected), "actual": _short(actual)}


def is_subset(exp, act) -> bool:
    if isinstance(exp, dict):
        return isinstance(act, dict) and all(k in act and is_subset(v, act[k]) for k, v in exp.items())
    if isinstance(exp, list):
        return isinstance(act, list) and all(any(is_subset(e, a) for a in act) for e in exp)
    return exp == act


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _has(val, exp) -> bool:
    """str: case-insensitive substring. list: equal element, or (for text) a string element containing the text."""
    if isinstance(val, str):
        return isinstance(exp, str) and exp.casefold() in val.casefold()
    if isinstance(val, list):
        return exp in val or (isinstance(exp, str) and any(isinstance(x, str) and exp.casefold() in x.casefold() for x in val))
    return False


def run_assertion(a: dict, doc: dict) -> dict:
    path = a.get("path", "")
    val = get_path(doc, path)
    name = a.get("name") or f"assert:{path}"
    ops = [k for k in a if k not in ("path", "name", "tol", "dir")]
    if len(ops) != 1:
        return _chk(name, False, "exactly one operator", ops)
    op, exp = ops[0], a[ops[0]]
    present = val is not MISSING
    if op == "equals":
        ok = present and val == exp
    elif op == "approx":
        ok = present and _num(val) and _num(exp) and abs(val - exp) <= a.get("tol", 1e-6)
    elif op == "contains":
        ok = present and _has(val, exp)
    elif op == "not_contains":
        ok = present and isinstance(val, (str, list)) and not _has(val, exp)
    elif op in ("gte", "lte", "gt", "lt"):
        ok = present and _num(val) and _num(exp) and {"gte": val >= exp, "lte": val <= exp, "gt": val > exp, "lt": val < exp}[op]
    elif op == "in":
        ok = present and isinstance(exp, list) and val in exp
    elif op == "is_null":
        ok = present and (val is None) == bool(exp)
    elif op == "exists":
        ok = present == bool(exp)
    elif op == "length":
        ok = present and isinstance(val, (list, str, dict)) and len(val) == exp
    elif op == "regex":
        ok = present and isinstance(val, str) and re.search(exp, val) is not None
    elif op == "sorted_by":
        d = a.get("dir", "desc")
        try:
            vals = [r[exp] for r in val if r.get(exp) is not None]
            ok = vals == sorted(vals, reverse=(d == "desc"))
        except (TypeError, KeyError, AttributeError):
            ok = False
    else:
        return _chk(name, False, "known operator", op)
    return _chk(name, ok, {op: exp}, val)


# ------------------------------------------------------------------ chat question expectations
def check_question(item: dict, expect: dict) -> list:
    c = []
    if "status" in expect:
        c.append(_chk("status", item["status"] == expect["status"], expect["status"], item["status"]))
    if "error_code" in expect:
        c.append(_chk("error_code", (item.get("error") or {}).get("code") == expect["error_code"], expect["error_code"], (item.get("error") or {}).get("code")))
    if "plan_intent" in expect:
        c.append(_chk("plan_intent", (item.get("plan") or {}).get("intent") == expect["plan_intent"], expect["plan_intent"], (item.get("plan") or {}).get("intent")))
    if "plan_subset" in expect:
        c.append(_chk("plan_subset", is_subset(expect["plan_subset"], item.get("plan")), expect["plan_subset"], item.get("plan")))
    if "planner_source" in expect:
        got = (item.get("trace") or {}).get("planner", {}).get("source")
        c.append(_chk("planner_source", got == expect["planner_source"], expect["planner_source"], got))
    rows = ((item.get("result") or {}).get("rows"))
    if "row_count" in expect:
        c.append(_chk("row_count", rows is not None and len(rows) == expect["row_count"], expect["row_count"], None if rows is None else len(rows)))
    if "rows" in expect:
        c.append(_chk("rows(exact, ordered)", rows == expect["rows"], expect["rows"], rows))
    if "rows_unordered" in expect:
        key = lambda r: json.dumps(r, sort_keys=True, default=str)  # noqa: E731
        c.append(_chk("rows(exact, any order)", rows is not None and sorted(rows, key=key) == sorted(expect["rows_unordered"], key=key), expect["rows_unordered"], rows))
    text = (item.get("answer") or {}).get("text") or ""
    for s in expect.get("answer_contains", []):
        c.append(_chk(f"answer_contains:{s}", s.casefold() in text.casefold(), s, text[:200]))
    for s in expect.get("answer_not_contains", []):
        c.append(_chk(f"answer_not_contains:{s}", s.casefold() not in text.casefold(), s, text[:200]))
    if expect.get("numbers_grounded"):
        c.append(_chk("numbers_grounded", item.get("grounded") is True, True, item.get("grounded")))
    for k in expect.get("evidence_has", []):
        c.append(_chk(f"evidence_has:{k}", k in (item.get("evidence") or {}), k, sorted((item.get("evidence") or {}))))
    for a in expect.get("assertions", []):
        c.append(run_assertion(a, item))
    return c


# ------------------------------------------------------------------ dataset expectations
def check_dataset(name: str, entry: dict, issues: list, plan: list, cleaned, exp: dict) -> list:
    ds, c = entry["dataset"], []
    pre = f"{name}:"
    if "status" in exp:
        c.append(_chk(pre + "status", ds.get("status") == exp["status"], exp["status"], ds.get("status")))
    if "error_code" in exp:
        c.append(_chk(pre + "error_code", (ds.get("error") or {}).get("code") == exp["error_code"], exp["error_code"], (ds.get("error") or {}).get("code")))
    for k in ("rows_in", "rows_out"):
        if k in exp:
            c.append(_chk(pre + k, ds.get(k) == exp[k], exp[k], ds.get(k)))
    if "rows_removed" in exp:
        got = None if ds.get("rows_in") is None or ds.get("rows_out") is None else ds["rows_in"] - ds["rows_out"]
        c.append(_chk(pre + "rows_removed", got == exp["rows_removed"], exp["rows_removed"], got))
    if "validation_status" in exp:
        c.append(_chk(pre + "validation_status", (entry.get("validation") or {}).get("status") == exp["validation_status"], exp["validation_status"], (entry.get("validation") or {}).get("status")))
    if "idempotent" in exp:
        got = ((entry.get("validation") or {}).get("idempotency") or {}).get("stable")
        c.append(_chk(pre + "idempotent", got == exp["idempotent"], exp["idempotent"], got))
    if "issue_count" in exp:
        c.append(_chk(pre + "issue_count", len(issues) == exp["issue_count"], exp["issue_count"], len(issues)))
    if "plan_steps" in exp:
        c.append(_chk(pre + "plan_steps", len(plan) == exp["plan_steps"], exp["plan_steps"], len(plan)))
    got_types = {i["type"] for i in issues}
    for t in exp.get("issue_types_include", []):
        c.append(_chk(pre + f"issue:{t}", t in got_types, t, sorted(got_types)))
    for t in exp.get("issue_types_exclude", []):
        c.append(_chk(pre + f"no_issue:{t}", t not in got_types, f"not {t}", sorted(got_types)))
    got_actions = {s["action"] for s in plan}
    for a in exp.get("plan_actions_include", []):
        c.append(_chk(pre + f"plan_action:{a}", a in got_actions, a, sorted(got_actions)))
    if "columns" in exp:
        c.append(_chk(pre + "columns", ds.get("columns") == exp["columns"], exp["columns"], ds.get("columns")))
    for col, rule in (exp.get("cleaned_columns") or {}).items():
        if cleaned is None or col not in cleaned.columns:
            c.append(_chk(pre + f"col:{col}", False, rule, "<no cleaned artifact or column>"))
            continue
        s = cleaned[col]
        nn = s.dropna().astype(str)
        if "all_match" in rule:
            c.append(_chk(pre + f"col:{col}:all_match", bool(nn.map(lambda v: re.fullmatch(rule["all_match"], v) is not None).all()), rule["all_match"], int((~nn.map(lambda v: re.fullmatch(rule["all_match"], v) is not None)).sum())))
        if "unique_count" in rule:
            c.append(_chk(pre + f"col:{col}:unique_count", int(nn.nunique()) == rule["unique_count"], rule["unique_count"], int(nn.nunique())))
        if "null_count" in rule:
            c.append(_chk(pre + f"col:{col}:null_count", int(s.isna().sum()) == rule["null_count"], rule["null_count"], int(s.isna().sum())))
        if "values_in" in rule:
            bad = sorted(set(nn) - set(rule["values_in"]))
            c.append(_chk(pre + f"col:{col}:values_in", not bad, rule["values_in"], bad[:5]))
    return c
