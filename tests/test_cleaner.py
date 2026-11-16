import re
from pathlib import Path

import pandas as pd
import pytest

from app.errors import PipelineError
from app.pipeline import cleaner
from app.pipeline.cleaner import clean_dataframe, fmt_number
from app.pipeline.loader import load_csv

ROOT = Path(__file__).resolve().parent.parent
SAMPLES, EDGE = ROOT / "data" / "samples", ROOT / "data" / "edge_cases"
NAMES = ["products", "stores", "customers", "orders", "inventory"]


def clean(name, folder=SAMPLES, **kw):
    df = load_csv(folder / f"{name}.csv").df
    return df, clean_dataframe(df, name, **kw)


def check(res, name):
    return next(c for c in res.report["validation"]["checks"] if c["name"] == name)


def step(res, action, col=None):
    return next(s for s in res.report["cleaning_plan"] if s["action"] == action and (col is None or col in s["fields"]))


# ---------- the engineering gate: validated, stable, no silent loss ----------
@pytest.mark.parametrize("name", NAMES)
def test_every_sample_validates_and_is_idempotent(name):
    df, res = clean(name)
    v = res.report["validation"]
    assert v["status"] == "pass", [c for c in v["checks"] if c["status"] != "pass"]
    again = clean_dataframe(res.df, name)
    assert again.df.equals(res.df)                       # cleaner(cleaner(x)) == cleaner(x)
    assert not [s for s in again.report["cleaning_plan"] if s["execution"].get("effective")]


@pytest.mark.parametrize("name", NAMES)
def test_input_is_never_mutated_and_schema_is_unchanged(name):
    df, _ = clean(name)
    pristine = load_csv(SAMPLES / f"{name}.csv").df
    pd.testing.assert_frame_equal(df, pristine)
    assert list(clean(name)[1].df.columns) == list(df.columns)


def test_row_loss_is_exactly_the_known_duplicates():
    for name, dups in [("orders", 15), ("products", 3), ("customers", 4), ("inventory", 6), ("stores", 0)]:
        df, res = clean(name)
        assert len(df) - len(res.df) == dups
        assert res.report["validation"]["row_delta"]["unexplained"] == 0
    assert clean("orders")[1].df["order_id"].is_unique


# ---------- transformation correctness ----------
def test_orders_values_are_canonical_after_cleaning():
    _, res = clean("orders")
    d = res.df
    assert d["sku"].dropna().map(lambda x: bool(re.fullmatch(r"SKU-\d{4}", x))).all()
    assert d["store_id"].nunique() == 8 and d["status"].nunique() == 3
    assert d["order_date"].dropna().map(lambda x: bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", x))).all()
    assert d["unit_price"].dropna().map(lambda x: re.fullmatch(r"\d+", x) is not None).all()
    assert set(d["discount"].dropna()) <= {"0", "0.1", "0.2"}
    assert "two" not in set(d["quantity"].dropna()) and "-1" in set(d["quantity"].dropna())   # -1 kept (flag only)


def test_products_categories_collapse_to_five_and_injection_text_is_preserved_verbatim():
    df, res = clean("products")
    assert res.df["category"].nunique() == 5
    assert res.df["brand"].nunique() == 6
    original = [v for v in df["product_name"] if "IGNORE ALL PREVIOUS" in v]
    assert original and all(v in set(res.df["product_name"]) for v in original)   # data is never sanitised away


def test_all_null_column_is_kept():
    _, res = clean("inventory")
    assert "notes" in res.df.columns and res.df["notes"].isna().all()


@pytest.mark.parametrize("v,out", [(1299.0, "1299"), (0.1, "0.1"), (12.5, "12.5"), (-3.0, "-3"), (0.125, "0.125")])
def test_fmt_number_is_stable(v, out):
    assert fmt_number(v) == out and fmt_number(float(fmt_number(v))) == out


def test_ambiguous_dates_follow_evidence_and_unproven_ones_become_null():
    ev = ["13/04/2025"] * 5 + ["14/04/2025"] * 2 + ["04-13-2025"] * 5            # slash=dmy, dash=mdy proven
    df = pd.DataFrame({"d": ev + ["03/04/2025", "03-04-2025", "2025-01-02"]})
    res = clean_dataframe(df, "t")
    assert res.df["d"].tolist()[-3:] == ["2025-04-03", "2025-03-04", "2025-01-02"]

    df2 = pd.DataFrame({"d": ["01/02/2025", "03/04/2025", "2025-01-01", "2025-01-02", "2025-01-03"]})
    res2 = clean_dataframe(df2, "t")
    assert res2.df["d"].isna().sum() == 1 and "2025-04-03" not in set(res2.df["d"].dropna())   # never guessed
    assert res2.report["validation"]["values_nulled_by_reason"]["unresolved_ambiguous"] == 2
    assert len(res2.df) == 4                                                       # the two nulled rows became identical -> merged...
    assert check(res2, "no_rows_merged_by_data_loss")["status"] == "warn"           # ...but visibly, never silently
    assert check(res2, "idempotent")["status"] == "pass"
    assert res2.report["validation"]["status"] == "warn"
    assert any(i["type"] == "DUPLICATES_CREATED_BY_CLEANING" for i in res2.report["issues"])


def test_row_numbers_of_removed_duplicates_are_traceable():
    _, res = clean("orders")
    sample = step(res, "drop_exact_duplicates")["execution"]["removed_row_numbers_sample"]
    assert 0 < len(sample) <= 20 and all(n >= 2 for n in sample)                     # CSV line numbers (header = line 1)


# ---------- decisions / review ----------
def test_skip_decision_leaves_data_alone_and_reports_unresolved_work():
    df = load_csv(SAMPLES / "orders.csv").df
    from app.pipeline.analyze import analyze_dataframe
    dedupe_id = next(s["id"] for s in analyze_dataframe(df, "orders")["cleaning_plan"] if s["action"] == "drop_exact_duplicates")
    res = clean_dataframe(df, "orders", decisions={dedupe_id: "skip"})
    assert len(res.df) == len(df) and step(res, "drop_exact_duplicates")["final_status"] == "skipped"
    v = res.report["validation"]
    assert v["row_delta"]["unexplained"] == 0
    assert check(res, "no_actionable_issues_left")["status"] == "warn"               # DUPLICATE_ROWS still open
    assert any(i["type"] == "DUPLICATE_ROWS" for i in v["unresolved_issues"]["actionable"])


def test_rename_is_skipped_by_default_but_works_when_approved():
    df = pd.DataFrame({"Order ID": ["a", "b"], "Qty": ["1", "2"]})
    res = clean_dataframe(df, "t")
    assert list(res.df.columns) == ["Order ID", "Qty"]                               # no silent schema change
    sid = step(res, "rename_columns")["id"]
    res2 = clean_dataframe(df, "t", decisions={sid: "apply"})
    assert list(res2.df.columns) == ["order_id", "qty"] and res2.report["validation"]["status"] == "pass"


def test_invalid_decision_is_a_structured_error():
    with pytest.raises(PipelineError) as e:
        clean_dataframe(pd.DataFrame({"a": ["1"]}), "t", decisions={"STP-x": "maybe"})
    assert e.value.code == "INVALID_DECISION"


def test_needs_review_steps_that_ran_are_reported():
    _, res = clean("products")
    assert step(res, "standardize_categories", "category")["id"] in res.report["validation"]["applied_without_human_review"]


# ---------- failure & validator-honesty tests ----------
def test_failed_step_is_isolated_and_does_not_corrupt_data(monkeypatch):
    def boom(df, step_, fields):
        raise RuntimeError("simulated crash")
    monkeypatch.setitem(cleaner.EXEC, "parse_numbers", boom)
    df, res = clean("orders")
    failed = [s for s in res.report["cleaning_plan"] if s["final_status"] == "failed"]
    assert {s["action"] for s in failed} == {"parse_numbers"} and failed[0]["execution"]["error"]["type"] == "RuntimeError"
    assert res.df["unit_price"].astype(str).str.contains(r"[$,]|Rs", regex=True).any()   # untouched, not half-converted
    assert res.df["sku"].dropna().str.match(r"^SKU-\d{4}$").all()                        # later steps still ran
    assert len(res.df) == 1500 and res.report["validation"]["status"] == "warn"
    assert check(res, "no_failed_steps")["status"] == "warn"


def test_validator_catches_a_silent_null_that_the_executor_did_not_log(monkeypatch):
    real = cleaner.EXEC["trim_whitespace"]

    def sneaky(df, step_, fields):
        out, log = real(df, step_, fields)
        i = out[fields[0]].first_valid_index()
        out.loc[i, fields[0]] = None                                                    # silent data loss
        return out, log

    monkeypatch.setitem(cleaner.EXEC, "trim_whitespace", sneaky)
    _, res = clean("orders", check_idempotent=False)
    assert check(res, "null_accounting")["status"] == "fail" and res.report["validation"]["status"] == "fail"


def test_validator_catches_unexplained_row_loss(monkeypatch):
    real = cleaner.EXEC["trim_whitespace"]
    monkeypatch.setitem(cleaner.EXEC, "trim_whitespace", lambda df, s, f: (real(df, s, f)[0].iloc[:-3], real(df, s, f)[1]))
    _, res = clean("stores", check_idempotent=False)
    assert check(res, "row_loss_explained")["status"] == "fail"


def test_validator_catches_a_schema_mutation(monkeypatch):
    real = cleaner.EXEC["trim_whitespace"]

    def adds_col(df, s, f):
        out, log = real(df, s, f)
        out["extra"] = "x"
        return out, log

    monkeypatch.setitem(cleaner.EXEC, "trim_whitespace", adds_col)
    _, res = clean("stores", check_idempotent=False)
    assert check(res, "schema_preserved")["status"] == "fail"


# ---------- edge cases ----------
def test_already_clean_and_header_only_pass_untouched():
    df, res = clean("already_clean", EDGE)
    assert res.df.equals(df) and res.report["validation"]["status"] == "pass" and res.report["cleaning_plan"] == []
    df, res = clean("header_only", EDGE)
    assert len(res.df) == 0 and list(res.df.columns) == list(df.columns) and res.report["validation"]["status"] == "pass"


def test_large_input_runs_and_stays_stable():
    n = 120_000
    df = pd.DataFrame({"sku": [f"sku{i % 500:04d}" for i in range(n)], "qty": [str(i % 7) for i in range(n)],
                       "price": ["$1,299.00" if i % 3 else "499" for i in range(n)]})
    res = clean_dataframe(df, "big")
    assert len(res.df) == len(df.drop_duplicates()) and res.report["validation"]["status"] == "pass"
    assert res.df["price"].isin(["1299", "499"]).all()


def test_rows_merged_only_because_of_nulling_are_counted_separately_from_true_duplicates():
    df = pd.DataFrame({"k": ["a", "a", "b", "b"], "d": ["2025-01-01", "2025-01-01", "01/02/2025", "03/04/2025"]})
    # extra rows give the date column enough evidence to be typed as date
    df = pd.concat([df, pd.DataFrame({"k": list("defgh"), "d": ["2025-02-01"] * 5})], ignore_index=True)
    res = clean_dataframe(df, "t")
    ex = step(res, "drop_exact_duplicates")["execution"]
    assert ex["rows_removed"] == 2 and ex["rows_removed_due_to_lossy_nulling"] == 1   # 1 true duplicate + 1 merged by nulling
    assert len(res.df) == len(df) - 2 and check(res, "idempotent")["status"] == "pass"


def test_skipping_dedupe_is_respected_even_for_cleaning_created_duplicates():
    df = pd.DataFrame({"d": ["01/02/2025", "03/04/2025", "2025-01-01", "2025-01-02", "2025-01-03"]})
    sid = step(clean_dataframe(df, "t"), "drop_exact_duplicates")["id"]
    res = clean_dataframe(df, "t", decisions={sid: "skip"}, check_idempotent=False)
    assert len(res.df) == 5 and step(res, "drop_exact_duplicates")["final_status"] == "skipped"
