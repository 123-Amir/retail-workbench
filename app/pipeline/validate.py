"""Independent validation of a cleaning run. It never trusts executor logs blindly:
it recomputes row counts, schema and null counts from the real before/after frames and compares."""
import pandas as pd

INFORMATIONAL = {"ALL_NULL_COLUMN", "NEGATIVE_VALUES", "PROMPT_INJECTION_SUSPECTED", "KEY_CONFLICT", "EMPTY_DATASET"}


def validate_cleaning(before: pd.DataFrame, after: pd.DataFrame, steps: list, null_pre: list, issues_after: list, idem: dict) -> dict:
    checks = []

    def chk(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})

    applied = [s for s in steps if s["final_status"] == "applied"]
    by_action = {s["action"]: s for s in applied}

    # 1. schema
    renamed = by_action["rename_columns"]["params"]["mapping"] if "rename_columns" in by_action else {}
    expected = [renamed.get(c, c) for c in before.columns]
    got = list(after.columns)
    added, removed = [c for c in got if c not in expected], [c for c in expected if c not in got]
    reordered = not added and not removed and got != expected
    schema_delta = {"columns_before": list(before.columns), "columns_after": got, "added": added,
                    "removed": removed, "renamed": renamed, "reordered": reordered}
    chk("schema_preserved", "pass" if not (added or removed or reordered) else "fail",
        "Columns match the plan." if not (added or removed or reordered) else schema_delta)

    # 2. row loss must be fully explained by the duplicate step
    lost = len(before) - len(after)
    explained = by_action["drop_exact_duplicates"]["execution"]["rows_removed"] if "drop_exact_duplicates" in by_action else 0
    row_delta = {"before": len(before), "after": len(after), "removed": lost, "explained_by_duplicate_step": explained,
                 "unexplained": lost - explained}
    ok = lost == explained and not (len(after) == 0 and len(before) > 0)
    chk("row_loss_explained", "pass" if ok else "fail", row_delta)

    # 3. duplicate step actually did its job, and did at least what the plan promised
    if "drop_exact_duplicates" in by_action:
        left = int(after.duplicated().sum())
        chk("no_duplicates_remaining", "pass" if left == 0 else "fail", {"duplicates_left": left})
        promised = by_action["drop_exact_duplicates"]["expected_effect"].get("rows_removed_min", 0)
        chk("expected_effect_met", "pass" if explained >= promised else "warn", {"promised_min": promised, "actual": explained})

        lossy_n = by_action["drop_exact_duplicates"]["execution"].get("rows_removed_due_to_lossy_nulling", 0)
        chk("no_rows_merged_by_data_loss", "warn" if lossy_n else "pass",
            {"rows_merged": lossy_n, "row_numbers_sample": by_action["drop_exact_duplicates"]["execution"].get("lossy_merged_row_numbers_sample", []),
             "note": "These rows differed in the raw file and became identical only because a value was nulled by policy. Review them."}
            if lossy_n else "No rows were merged because of nulled values.")

    # 4. every new null must be explained by a logged step (recomputed from real frames)
    nb = before.isna().sum().tolist()
    null_delta, bad = {}, False
    for i, col in enumerate(before.columns):
        inc = null_pre[i] - nb[i]
        logged = sum(s["execution"].get("cells_nulled", 0) for s in applied if s["fields"] == [col])
        if inc or logged:
            null_delta[col] = {"nulls_before": nb[i], "nulls_after_cleaning": null_pre[i], "increase": inc,
                               "explained_by_logs": logged, "unexplained": inc - logged}
            bad = bad or inc != logged
    chk("null_accounting", "fail" if bad else "pass",
        "Every new null is explained by a logged step." if not bad else null_delta)

    # 5. failed steps
    failed = [s["id"] for s in steps if s["final_status"] == "failed"]
    chk("no_failed_steps", "warn" if failed else "pass", {"failed_step_ids": failed} if failed else "All executed steps succeeded.")

    # 6. what was lost on purpose (policy), by reason
    loss = {}
    for s in applied:
        for tag, v in s["execution"].get("tags", {}).items():
            if tag in ("placeholder", "parse_failure", "unresolved_ambiguous"):
                loss[tag] = loss.get(tag, 0) + v["cells"]

    # 7. unresolved issues after cleaning
    actionable = [i for i in issues_after if i["type"] not in INFORMATIONAL]
    info = [i for i in issues_after if i["type"] in INFORMATIONAL]
    brief = lambda L: [{"id": i["id"], "type": i["type"], "columns": i["columns"], "rows": i["affected_rows"]} for i in L]  # noqa: E731
    chk("no_actionable_issues_left", "warn" if actionable else "pass", brief(actionable) if actionable else "None.")

    # 8. idempotency
    if idem.get("checked"):
        chk("idempotent", "pass" if idem["stable"] else "fail",
            "Re-running the cleaner on its output changes nothing." if idem["stable"] else idem)
    else:
        chk("idempotent", "skipped", "Not checked.")

    statuses = {c["status"] for c in checks}
    return {"status": "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass"),
            "checks": checks, "row_delta": row_delta, "schema_delta": schema_delta, "null_delta": null_delta,
            "values_nulled_by_reason": dict(sorted(loss.items())),
            "applied_without_human_review": [s["id"] for s in applied if s["status"] == "needs_review"],
            "unresolved_issues": {"actionable": brief(actionable), "informational": brief(info)},
            "idempotency": idem}
