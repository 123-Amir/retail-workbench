"""Cleaning plan: turns issues into ordered, auditable steps. NO data is mutated here.

Step fields: id (stable), order, action, fields, issue_ids, reason, risk (low/medium/high),
source ('code' | 'ai'), status ('planned' | 'needs_review' | 'flag_only'),
default_action ('apply' | 'skip'), params, expected_effect, final_status ('pending' until Step 4 runs it).
Allowed actions never include dropping columns: schema changes are never silent.
"""
from app.pipeline.issues import hid

ACTIONS = ["rename_columns", "trim_whitespace", "null_placeholders", "normalize_sku",
           "standardize_categories", "parse_numbers", "parse_dates", "drop_exact_duplicates", "flag_for_review"]
_ORDER = {a: i for i, a in enumerate(ACTIONS)}
STATUSES = ("planned", "needs_review", "flag_only")


def build_cleaning_plan(dataset: str, issues: list, profile: dict) -> list:
    idx = {(i["type"], i["columns"][0] if len(i["columns"]) == 1 else None): i for i in issues}
    by_type = {i["type"]: i for i in issues}
    cols = {c["name"]: c for c in profile["columns"]}
    steps = []

    def add(action, fields, issue_ids, reason, risk, status, default, params=None, effect=None):
        steps.append({"id": hid("STP", dataset, action, ",".join(fields) or "*"), "dataset": dataset,
                      "action": action, "fields": fields, "issue_ids": issue_ids, "reason": reason,
                      "risk": risk, "source": "code", "status": status, "default_action": default,
                      "params": params or {}, "expected_effect": effect or {}, "final_status": "pending"})

    if (i := by_type.get("COLUMN_NAME_FORMAT")):
        add("rename_columns", i["columns"], [i["id"]], "Standardise column names to snake_case. Renames change the schema, so apply only after review.",
            "medium", "needs_review", "skip", {"mapping": i["evidence"]["mapping"]})

    for n, c in cols.items():
        ids = lambda *types: [idx[(t, n)]["id"] for t in types if (t, n) in idx]  # noqa: E731
        if (t := idx.get(("WHITESPACE", n))):
            add("trim_whitespace", [n], [t["id"]], "Strip leading/trailing spaces so equal values compare equal.",
                "low", "planned", "apply", {}, {"cells_changed": t["affected_rows"]})
        if (t := idx.get(("PLACEHOLDER_NULLS", n))):
            add("null_placeholders", [n], [t["id"]], "Convert placeholder text (N/A, -, blank) to real nulls.",
                "low", "planned", "apply", {"tokens": sorted(t["evidence"]["tokens"]), "blank_cells": t["evidence"]["blank_cells"]},
                {"cells_nulled": t["affected_rows"]})
        if (t := idx.get(("SKU_FORMAT", n))):
            add("normalize_sku", [n], [t["id"]], "Canonical SKU form PREFIX-digits so joins across datasets match.",
                "low", "planned", "apply", {"pattern": "PREFIX-digits", "unmatched_rows": t["evidence"]["unmatched_rows"]},
                {"cells_changed": t["affected_rows"]})
        if (t := idx.get(("CATEGORY_VARIANTS", n))):
            mapping, review = {}, False
            for cl in t["evidence"]["clusters"]:
                mapping.update(cl["mapping"])
                review = review or cl["needs_review"]
            add("standardize_categories", [n], [t["id"]],
                "Merge case/spelling variants; alias-table matches (confidence 0.8) need review." if review
                else "Merge case/spelling variants of the same value.",
                "medium" if review else "low", "needs_review" if review else "planned", "apply",
                {"mapping": dict(sorted(mapping.items())),
                 "clusters": [{"canonical": cl["canonical"], "variants": cl["variants"]} for cl in t["evidence"]["clusters"]]},
                {"cells_changed_max": t["affected_rows"]})

        if c["inferred_type"] in ("integer", "float"):
            ii = ids("MIXED_NUMERIC_FORMAT", "PARSE_FAILURES")
            if ii:
                f = c["parse_failures"]["count"]
                add("parse_numbers", [n], ii, "Parse to numbers ($, Rs, commas removed; 10% becomes 0.1). Unparseable values become null and are recorded.",
                    "medium" if f else "low", "needs_review" if f else "planned", "apply",
                    {"percent_as_fraction": True, "on_failure": "set_null_and_record", "integer": c["inferred_type"] == "integer"},
                    {"values_nulled": f})
        if c["inferred_type"] == "date":
            ii = ids("MIXED_DATE_FORMATS", "AMBIGUOUS_DATES", "PARSE_FAILURES")
            if ii:
                f, amb = c["parse_failures"]["count"], c["date_stats"]["ambiguous_count"]
                order = {}
                unresolved = 0
                if (a := idx.get(("AMBIGUOUS_DATES", n))):
                    for sep, e in a["evidence"]["by_separator"].items():
                        if e["ambiguous"]:
                            order[sep] = e["inferred_order"]
                            if e["inferred_order"] is None:
                                unresolved += e["ambiguous"]
                risk = "high" if amb else ("medium" if f else "low")
                add("parse_dates", [n], ii,
                    "Convert to ISO dates. Ambiguous day/month values use the order proven by unambiguous dates with the same separator; otherwise they are nulled, never guessed.",
                    risk, "needs_review" if risk != "low" else "planned", "apply",
                    {"output_format": "iso", "separator_order": order, "unresolved_ambiguous": "set_null_and_flag",
                     "on_failure": "set_null_and_record"},
                    {"values_nulled_min": f + unresolved, "ambiguous_values": amb})

    if (i := by_type.get("DUPLICATE_ROWS")):
        add("drop_exact_duplicates", [], [i["id"]],
            "Remove rows identical in every column (checked after normalisation, keeping the first). Count is verified in validation.",
            "medium", "planned", "apply", {"keep": "first", "subset": "all_columns"},
            {"rows_removed_min": i["evidence"]["exact"]})

    for t, why in [("ALL_NULL_COLUMN", "Column is entirely empty. Kept (no silent schema change) and reported."),
                   ("NEGATIVE_VALUES", "Negative values might be legitimate returns. Not changed; needs a business decision."),
                   ("PROMPT_INJECTION_SUSPECTED", "Text resembles an AI instruction. Kept as data; the chat layer must never obey it."),
                   ("KEY_CONFLICT", "Same identifier with different row content. Cannot be auto-resolved.")]:
        for i in (x for x in issues if x["type"] == t):
            add("flag_for_review", i["columns"], [i["id"]], why, "low" if t == "ALL_NULL_COLUMN" else "medium",
                "flag_only", "skip", {"issue_type": t})

    steps.sort(key=lambda s: (_ORDER[s["action"]], s["fields"]))
    for k, s in enumerate(steps, 1):
        s["order"] = k
    return steps


def summarize_plan(steps: list) -> dict:
    cnt = lambda key: {k: sum(1 for s in steps if s[key] == k) for k in sorted({s[key] for s in steps})}  # noqa: E731
    return {"total_steps": len(steps), "by_action": cnt("action"), "by_risk": cnt("risk"), "by_status": cnt("status"),
            "will_apply_by_default": sum(1 for s in steps if s["default_action"] == "apply")}
