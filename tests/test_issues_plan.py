import json
from pathlib import Path

import pandas as pd
import pytest

from app.pipeline.analyze import analyze_dataframe
from app.pipeline.loader import load_csv
from app.pipeline.planner import ACTIONS, STATUSES
from app.pipeline.variants import cluster_variants, norm_key

ROOT = Path(__file__).resolve().parent.parent
SAMPLES, EDGE = ROOT / "data" / "samples", ROOT / "data" / "edge_cases"
NAMES = ["products", "stores", "customers", "orders", "inventory"]


def analyze(name, folder=SAMPLES):
    r = load_csv(folder / f"{name}.csv")
    return analyze_dataframe(r.df, name), r.df


def issue(a, typ, col=None):
    return next((i for i in a["issues"] if i["type"] == typ and (col is None or col in i["columns"])), None)


def step(a, action, col=None):
    return next((s for s in a["cleaning_plan"] if s["action"] == action and (col is None or col in s["fields"])), None)


# ---------- structural invariants over every dataset ----------
@pytest.mark.parametrize("name", NAMES)
def test_plan_is_well_formed(name):
    a, _ = analyze(name)
    issue_ids = {i["id"] for i in a["issues"]}
    step_ids = [s["id"] for s in a["cleaning_plan"]]
    assert len(issue_ids) == len(a["issues"]) and len(set(step_ids)) == len(step_ids)
    for s in a["cleaning_plan"]:
        assert s["action"] in ACTIONS and s["status"] in STATUSES
        assert s["risk"] in ("low", "medium", "high") and s["source"] in ("code", "ai")
        assert s["default_action"] in ("apply", "skip") and s["reason"] and s["final_status"] == "pending"
        assert s["issue_ids"] and set(s["issue_ids"]) <= issue_ids
        if s["status"] == "flag_only":
            assert s["default_action"] == "skip"          # flags never mutate data
    assert json.dumps(a)


@pytest.mark.parametrize("name", NAMES)
def test_ids_are_stable_across_reruns_and_row_order(name):
    a, df = analyze(name)
    shuffled = analyze_dataframe(df.sample(frac=1, random_state=7).reset_index(drop=True), name)
    assert [i["id"] for i in a["issues"]] == [i["id"] for i in shuffled["issues"]]
    assert [s["id"] for s in a["cleaning_plan"]] == [s["id"] for s in shuffled["cleaning_plan"]]
    assert a == analyze(name)[0]                            # fully deterministic


def test_analysis_never_mutates_input():
    r = load_csv(SAMPLES / "orders.csv")
    before = r.df.copy(deep=True)
    analyze_dataframe(r.df, "orders")
    pd.testing.assert_frame_equal(before, r.df)


def test_no_step_can_drop_a_column_or_row_silently():
    for n in NAMES:
        for s in analyze(n)[0]["cleaning_plan"]:
            assert "drop_column" not in s["action"]
    # the only row-removing action is exact-duplicate removal and it declares its expected effect
    a, _ = analyze("orders")
    d = step(a, "drop_exact_duplicates")
    assert d["expected_effect"]["rows_removed_min"] == 15


def test_steps_run_in_a_safe_order():
    a, _ = analyze("orders")
    pos = {s["action"]: s["order"] for s in reversed(a["cleaning_plan"])}
    assert pos["trim_whitespace"] < pos["normalize_sku"] < pos["parse_dates"] < pos["drop_exact_duplicates"]
    mutating = [s for s in a["cleaning_plan"] if s["status"] != "flag_only"]
    assert mutating[-1]["action"] == "drop_exact_duplicates"


# ---------- no phantom work, edge cases ----------
def test_already_clean_file_gets_empty_plan():
    a, _ = analyze("already_clean", EDGE)
    assert a["issues"] == [] and a["cleaning_plan"] == []


def test_header_only_reports_empty_dataset_and_no_steps():
    a, _ = analyze("header_only", EDGE)
    assert [i["type"] for i in a["issues"]] == ["EMPTY_DATASET"] and a["cleaning_plan"] == []


# ---------- retail-specific detection ----------
def test_ambiguous_dates_use_evidence_not_guessing():
    a, _ = analyze("orders")
    by_sep = issue(a, "AMBIGUOUS_DATES", "order_date")["evidence"]["by_separator"]
    assert by_sep["slash"]["inferred_order"] == "dmy" and by_sep["dash"]["inferred_order"] == "mdy"
    s = step(a, "parse_dates", "order_date")
    assert s["risk"] == "high" and s["status"] == "needs_review"
    assert s["params"]["unresolved_ambiguous"] == "set_null_and_flag"


def test_ambiguity_without_evidence_is_left_unresolved():
    df = pd.DataFrame({"d": ["01/02/2025", "03/04/2025", "05/06/2025", "2025-01-01", "2025-01-02"]})
    a = analyze_dataframe(df, "t")
    e = issue(a, "AMBIGUOUS_DATES")["evidence"]["by_separator"]["slash"]
    assert e["inferred_order"] is None
    assert step(a, "parse_dates")["expected_effect"]["values_nulled_min"] == 3


def test_category_variants_merge_to_one_canonical_with_review_for_alias():
    a, df = analyze("products")
    s = step(a, "standardize_categories", "category")
    present = {v.strip() for v in df["category"].dropna()}
    tshirt = {"T-Shirt", "Tee", "tshirt", "T Shirt", "t-shirts"} & present
    assert len({s["params"]["mapping"].get(v, v) for v in tshirt}) == 1
    assert s["status"] == "needs_review"                    # 'Tee' came from the alias table (conf 0.8)
    methods = {v["method"] for c in s["params"]["clusters"] for v in c["variants"]}
    assert "alias_table" in methods


def test_region_abbreviation_resolves_but_is_reviewed():
    df = pd.DataFrame({"region": ["West", "west ", "WEST", "W", "North", "North", "East", "East"]})
    s = step(analyze_dataframe(df, "stores"), "standardize_categories", "region")
    assert s["status"] == "needs_review"                   # 'W' -> 'West' comes from the alias table
    assert {s["params"]["mapping"].get(v, v) for v in ["West", "west", "WEST", "W"]} == {"West"}


def test_stores_sample_has_case_only_region_variants_so_no_review_needed():
    s = step(analyze("stores")[0], "standardize_categories", "region")
    assert s["status"] == "planned" and s["risk"] == "low"


def test_case_only_variants_are_auto_planned_without_review():
    a, _ = analyze("orders")
    s = step(a, "standardize_categories", "status")
    assert s["status"] == "planned" and s["risk"] == "low"
    assert set(s["params"]["mapping"].values()) <= {"Completed", "completed", "Returned", "returned", "Cancelled", "cancelled"}


def test_sku_normalisation_planned_for_every_sku_column():
    for n in ("products", "orders", "inventory"):
        a, _ = analyze(n)
        assert step(a, "normalize_sku", "sku") is not None
        assert issue(a, "SKU_FORMAT")["evidence"]["unmatched_rows"] >= 0
    ex = issue(analyze("orders")[0], "SKU_FORMAT")["evidence"]["examples"][0]
    assert ex["canonical"].startswith("SKU-")


def test_prompt_injection_cell_is_flagged_never_acted_on():
    a, _ = analyze("products")
    i = issue(a, "PROMPT_INJECTION_SUSPECTED", "product_name")
    assert i and all(len(x) <= 80 for x in i["evidence"]["examples"])
    s = step(a, "flag_for_review", "product_name")
    assert s["status"] == "flag_only" and s["default_action"] == "skip"


def test_all_null_column_kept_and_flagged():
    a, _ = analyze("inventory")
    s = step(a, "flag_for_review", "notes")
    assert s and s["params"]["issue_type"] == "ALL_NULL_COLUMN"


def test_numeric_formats_and_failures_linked_to_one_parse_step():
    a, _ = analyze("orders")
    s = step(a, "parse_numbers", "quantity")
    types = {i["type"] for i in a["issues"] if i["id"] in s["issue_ids"]}
    assert "PARSE_FAILURES" in types and s["status"] == "needs_review"
    assert s["params"]["on_failure"] == "set_null_and_record"
    assert issue(a, "NEGATIVE_VALUES", "quantity")        # quantity=-1 flagged, not auto-changed
    p = step(a, "parse_numbers", "unit_price")
    assert p["status"] == "planned"                       # formats only: lossless


def test_pii_never_leaks_into_issue_evidence():
    a, _ = analyze("customers")
    assert "@" not in json.dumps([i for i in a["issues"] if "email" in i["columns"]])


def test_key_conflict_detected_when_ids_collide_with_different_content():
    ids = [f"A{i}" for i in range(19)] + ["A0"]       # 95% unique -> identifier-like, one collision
    df = pd.DataFrame({"order_id": ids, "qty": [str(i) for i in range(20)]})
    a = analyze_dataframe(df, "t")
    assert issue(a, "KEY_CONFLICT", "order_id")["severity"] == "high"
    assert issue(a, "DUPLICATE_ROWS") is None


# ---------- variant clustering unit tests ----------
def test_variant_clusters_explain_every_mapping():
    cl = cluster_variants([("Jeans", 10), ("jeans", 3), ("Denim", 2), ("Watch", 5)], {"denim": "jean"})
    assert len(cl) == 1 and cl[0]["canonical"] == "Jeans"
    assert {v["value"]: v["method"] for v in cl[0]["variants"]} == {
        "Jeans": "case_whitespace", "jeans": "case_whitespace", "Denim": "alias_table"}
    assert norm_key("T-Shirts") == norm_key("tshirt") == norm_key("T Shirt")


def test_shouty_variant_is_not_chosen_as_canonical():
    assert cluster_variants([("WEST", 50), ("West", 10)])[0]["canonical"] == "West"


def test_canonical_prefers_title_case_over_lowercase_even_when_less_frequent():
    assert cluster_variants([("jeans", 10), ("Jeans", 3)])[0]["canonical"] == "Jeans"
    assert cluster_variants([("ST01", 5), ("st01", 9)])[0]["canonical"] == "st01"      # IDs: no style preference, frequency decides
