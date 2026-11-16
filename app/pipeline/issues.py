"""Issue detection. Read-only: inspects a raw (string) DataFrame + its profile and returns issues.
Issue ids are stable: sha1(dataset | type | columns), independent of row order and of counts."""
import hashlib
import re
from collections import Counter

import pandas as pd

from app.pipeline.parsers import NULL_TOKENS, NUMDATE, parse_date, parse_number
from app.pipeline.variants import cluster_variants
from app.retail.aliases import alias_table_for

MIN_DATE_EVIDENCE = 5
DATE_CONFIDENCE = 0.95
_CAT_NAME = re.compile(r"category|brand|status|region|city|type|state|store_name|location", re.I)
_SKU_NAME = re.compile(r"(^|_)sku$", re.I)
_QTY_PRICE = re.compile(r"quantity|qty|price|cost|stock|amount|total", re.I)
_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
_SKU = re.compile(r"^([A-Za-z]+)[-_ ]?(\d+)$")
_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above)|disregard .{0,25}instruction|system prompt|"
    r"reveal .{0,30}(prompt|key|secret|password)|you are now|<\s*/?\s*system\s*>", re.I)
_SEV = {"high": 0, "medium": 1, "low": 2}


def hid(prefix: str, *parts: str) -> str:
    return f"{prefix}-{hashlib.sha1('|'.join(parts).encode()).hexdigest()[:8]}"


def _issue(ds, itype, sev, cols, rows, summary, evidence):
    return {"id": hid("ISS", ds, itype, ",".join(cols) or "*"), "dataset": ds, "type": itype,
            "severity": sev, "columns": cols, "affected_rows": int(rows), "summary": summary,
            "evidence": evidence, "detected_by": "code"}


def _present(series):
    s = series.astype("string").dropna()
    st = s.str.strip()
    keep = ~(st.eq("") | st.str.lower().isin(NULL_TOKENS))
    vc = s[keep.to_numpy(dtype=bool)].value_counts()
    return sorted(((str(k), int(c)) for k, c in vc.items()), key=lambda kc: (-kc[1], kc[0]))


def _snake(name, i):
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or f"col_{i}"


def date_order_evidence(items):
    ev = {k: {"dmy": 0, "mdy": 0, "ambiguous": 0} for k in ("slash", "dash")}
    for v, c in items:
        m = NUMDATE.match(v.strip())
        if not m or parse_date(v) is None:
            continue
        a, b, kind = int(m[1]), int(m[3]), ("slash" if m[2] == "/" else "dash")
        if a > 12:
            ev[kind]["dmy"] += c
        elif b > 12:
            ev[kind]["mdy"] += c
        elif a != b:
            ev[kind]["ambiguous"] += c
    out = {}
    for kind, e in ev.items():
        if not (e["dmy"] or e["mdy"] or e["ambiguous"]):
            continue
        tot = e["dmy"] + e["mdy"]
        top = max(e["dmy"], e["mdy"])
        ok = tot >= MIN_DATE_EVIDENCE and top / tot >= DATE_CONFIDENCE
        e["inferred_order"] = ("dmy" if e["dmy"] > e["mdy"] else "mdy") if ok else None
        e["confidence"] = round(top / tot, 3) if tot else None
        out[kind] = e
    return out


def detect_issues(df: pd.DataFrame, profile: dict) -> list:
    ds = profile["name"]
    if profile["row_count"] == 0:
        return [_issue(ds, "EMPTY_DATASET", "high", [], 0, "Dataset has a header but no data rows.", {})]

    issues = []
    names = [c["name"] for c in profile["columns"]]

    bad_names = {n: _snake(n, i) for i, n in enumerate(names) if not _SNAKE.match(n)}
    if bad_names:
        collision = len(set(bad_names.values()) | (set(names) - set(bad_names))) < len(names)
        issues.append(_issue(ds, "COLUMN_NAME_FORMAT", "low", sorted(bad_names), 0,
                             "Column names are not snake_case.", {"mapping": bad_names, "collision": collision}))

    if profile["duplicate_rows"]:
        issues.append(_issue(ds, "DUPLICATE_ROWS", "medium", [], profile["duplicate_rows"],
                             "Exact duplicate rows found.",
                             {"exact": profile["duplicate_rows"],
                              "after_normalization": profile["duplicate_rows_after_normalization"]}))

    for c in profile["columns"]:
        n, ctype, pii = c["name"], c["inferred_type"], c["pii_suspected"]
        series = df[n]
        items = _present(series)
        red = lambda xs: ["[redacted]"] * len(xs) if pii else xs  # noqa: E731

        if c["is_all_null"]:
            issues.append(_issue(ds, "ALL_NULL_COLUMN", "low", [n], len(df), "Column has no values.", {}))
            continue

        if c["whitespace_issue_count"]:
            ex = [v for v, _ in items if v != v.strip()][:3]
            issues.append(_issue(ds, "WHITESPACE", "low", [n], c["whitespace_issue_count"],
                                 "Leading/trailing whitespace.", {"examples": red(ex)}))

        if c["placeholder_null_count"]:
            s = series.astype("string").dropna()
            st, low = s.str.strip(), s.str.strip().str.lower()
            toks = {str(t): int(k) for t, k in low[low.isin(NULL_TOKENS)].value_counts().items()}
            issues.append(_issue(ds, "PLACEHOLDER_NULLS", "low", [n], c["placeholder_null_count"],
                                 "Placeholder text used instead of null.",
                                 {"tokens": dict(sorted(toks.items())), "blank_cells": int(st.eq("").sum())}))

        if _SKU_NAME.search(n):
            fixes, unmatched = [], 0
            rows = 0
            for v, k in items:
                st = v.strip()
                m = _SKU.match(st)
                if not m:
                    unmatched += k
                    continue
                canon = f"{m[1].upper()}-{m[2]}"
                if canon != st:
                    rows += k
                    fixes.append({"raw": v, "canonical": canon})
            if rows or unmatched:
                issues.append(_issue(ds, "SKU_FORMAT", "medium", [n], rows,
                                     "SKU formatting varies (case, separator).",
                                     {"non_canonical_rows": rows, "unmatched_rows": unmatched,
                                      "examples": fixes[:5], "pattern": "PREFIX-digits"}))

        if ctype == "string" and not pii and not c["looks_like_identifier"] and not _SKU_NAME.search(n) \
                and c["unique_count"] <= 200 and (_CAT_NAME.search(n) or c["unique_ratio"] <= 0.2):
            agg = Counter()
            for v, k in items:
                agg[v.strip()] += k
            clusters = cluster_variants(sorted(agg.items(), key=lambda kc: (-kc[1], kc[0])), alias_table_for(n))
            if clusters:
                issues.append(_issue(ds, "CATEGORY_VARIANTS", "medium", [n], sum(cl["total"] for cl in clusters),
                                     "Spelling/case variants of the same category value.",
                                     {"cluster_count": len(clusters), "clusters": clusters}))

        if ctype in ("string", "mixed") and not pii:
            hits = [(v, k) for v, k in items if _INJECTION.search(v)]
            if hits:
                issues.append(_issue(ds, "PROMPT_INJECTION_SUSPECTED", "medium", [n], sum(k for _, k in hits),
                                     "Cell text looks like an instruction to an AI. Treated strictly as data.",
                                     {"examples": [v[:80] for v, _ in hits[:3]]}))

        if ctype in ("integer", "float"):
            cur = pct = com = dec = 0
            for v, k in items:
                t = v.strip().lower()
                if parse_number(v) is None:
                    continue
                if t.startswith(("rs", "inr", "usd", "$", "₹")):
                    cur += k
                if t.endswith("%"):
                    pct += k
                if "," in t:
                    com += k
                if ctype == "integer" and "." in t:
                    dec += k
            if cur or pct or com or dec:
                issues.append(_issue(ds, "MIXED_NUMERIC_FORMAT", "medium", [n], cur + pct + com + dec,
                                     "Numbers stored with currency symbols, %, commas or decimal noise.",
                                     {"currency_prefixed": cur, "percent": pct, "thousand_commas": com,
                                      "integer_with_decimal_point": dec}))
            if c.get("numeric_stats", {}).get("negative_count") and _QTY_PRICE.search(n):
                issues.append(_issue(ds, "NEGATIVE_VALUES", "medium", [n], c["numeric_stats"]["negative_count"],
                                     "Negative values in a quantity/price-like column (could be returns).",
                                     {"min": c["numeric_stats"]["min"]}))

        if ctype in ("integer", "float", "date") and c["parse_failures"]["count"]:
            issues.append(_issue(ds, "PARSE_FAILURES", "medium", [n], c["parse_failures"]["count"],
                                 f"Values cannot be parsed as {ctype}.",
                                 {"expected_type": ctype, "examples": c["parse_failures"]["examples"]}))

        if ctype == "date":
            ds_ = c["date_stats"]
            fams = sorted({f.split("_")[0] for f in ds_["formats"]})
            if len(fams) > 1:
                issues.append(_issue(ds, "MIXED_DATE_FORMATS", "medium", [n], sum(ds_["formats"].values()),
                                     "Several date formats in one column.", {"formats": ds_["formats"]}))
            if ds_["ambiguous_count"]:
                issues.append(_issue(ds, "AMBIGUOUS_DATES", "high", [n], ds_["ambiguous_count"],
                                     "Dates like 03/04/2025 could be day-first or month-first.",
                                     {"by_separator": date_order_evidence(items)}))

    # identifier collisions that survive exact-duplicate removal
    id_cols = [c["name"] for c in profile["columns"] if c["looks_like_identifier"]]
    if id_cols:
        norm = df.astype("string").apply(lambda col: col.str.strip().str.lower()).drop_duplicates()
        for n in id_cols:
            k = int(norm[n].dropna().duplicated().sum())
            if k:
                issues.append(_issue(ds, "KEY_CONFLICT", "high", [n], k,
                                     "Same identifier appears in rows that differ. Needs a human decision.", {}))

    return sorted(issues, key=lambda i: (_SEV[i["severity"]], i["type"], i["columns"]))
SKU_RE = _SKU  # public alias used by the cleaner
INJECTION_RE = _INJECTION  # public alias used by the chat layer
