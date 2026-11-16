import json
from pathlib import Path

import pandas as pd
import pytest

from app.errors import PipelineError
from app.pipeline.loader import load_csv, sanitize_filename
from app.pipeline.parsers import parse_date, parse_number
from app.pipeline.profiler import profile_dataframe

ROOT = Path(__file__).resolve().parent.parent
SAMPLES, EDGE = ROOT / "data" / "samples", ROOT / "data" / "edge_cases"


def prof(name):
    r = load_csv(SAMPLES / f"{name}.csv")
    return profile_dataframe(r.df, name), r.df


def col(p, name):
    return next(c for c in p["columns"] if c["name"] == name)


# ---------- parsers ----------
def test_parse_number_formats():
    assert parse_number("$1,299.00") == (1299.0, True)
    assert parse_number("Rs 499") == (499.0, True)
    assert parse_number("10%") == (0.1, True)
    assert parse_number("42") == (42.0, False)
    assert parse_number("two") is None and parse_number("SKU-0001") is None


def test_parse_date_flags_ambiguity_instead_of_guessing():
    assert parse_date("2025-03-14")[2] is False
    assert parse_date("13/04/2025")[1] == "slash_dmy"
    assert parse_date("04/13/2025")[1] == "slash_mdy"
    assert parse_date("03/04/2025")[2] is True        # ambiguous: flagged
    assert parse_date("05/05/2025")[2] is False       # same either way
    assert parse_date("31/02/2025") is None           # impossible date
    assert parse_date("not a date") is None


# ---------- loader ----------
def test_loader_rejects_bad_inputs(tmp_path):
    (tmp_path / "x.txt").write_text("a,b\n1,2\n")
    (tmp_path / "e.csv").write_bytes(b"")
    (tmp_path / "big.csv").write_text("a\n" + "1\n" * 2000)
    for path, kw, code in [(tmp_path / "x.txt", {}, "UNSUPPORTED_FILE_TYPE"),
                           (tmp_path / "nope.csv", {}, "FILE_NOT_FOUND"),
                           (tmp_path / "e.csv", {}, "EMPTY_FILE"),
                           (tmp_path / "big.csv", {"max_mb": 0.001}, "FILE_TOO_LARGE")]:
        with pytest.raises(PipelineError) as ei:
            load_csv(path, **kw)
        assert ei.value.code == code


def test_loader_header_only_and_encoding(tmp_path):
    r = load_csv(EDGE / "header_only.csv")
    assert len(r.df) == 0 and r.df.shape[1] == 6
    assert any(w["code"] == "HEADER_ONLY" for w in r.warnings)
    (tmp_path / "l1.csv").write_bytes("name,city\nJosé,Zürich\n".encode("cp1252"))
    r = load_csv(tmp_path / "l1.csv")
    assert r.encoding == "cp1252" and r.df.iloc[0]["name"] == "José"


def test_loader_keeps_leading_zeros_and_sanitizes_name(tmp_path):
    (tmp_path / "z.csv").write_text("code\n007\n")
    assert load_csv(tmp_path / "z.csv").df.iloc[0, 0] == "007"
    assert sanitize_filename("../../etc/passwd.csv") == "passwd.csv"


# ---------- profiler invariants ----------
@pytest.mark.parametrize("name", ["products", "stores", "customers", "orders", "inventory"])
def test_null_accounting_adds_up_for_every_column(name):
    p, df = prof(name)
    assert p["row_count"] == len(df)
    for c in p["columns"]:
        assert c["non_null_count"] + c["null_count"] + c["placeholder_null_count"] == p["row_count"], c["name"]


def test_known_duplicate_counts_from_generator():
    assert prof("orders")[0]["duplicate_rows"] == 15
    assert prof("products")[0]["duplicate_rows"] == 3
    assert prof("customers")[0]["duplicate_rows"] == 4
    assert prof("inventory")[0]["duplicate_rows"] == 6


def test_normalisation_reveals_more_duplicates_than_exact_match():
    p, _ = prof("orders")
    assert p["duplicate_rows_after_normalization"] >= p["duplicate_rows"]


def test_all_null_column_detected():
    p, _ = prof("inventory")
    assert "notes" in p["all_null_columns"] and col(p, "notes")["inferred_type"] == "empty"


def test_mixed_quantity_reports_parse_failures():
    c = col(prof("orders")[0], "quantity")
    assert c["inferred_type"] == "integer"
    assert c["parse_failures"]["count"] > 0 and c["placeholder_null_count"] > 0


def test_messy_dates_and_prices_detected():
    p, _ = prof("orders")
    d = col(p, "order_date")
    assert d["inferred_type"] == "date" and len(d["date_stats"]["formats"]) > 1
    assert d["date_stats"]["ambiguous_count"] > 0 and d["parse_failures"]["count"] > 0
    assert col(p, "unit_price")["symbol_formatted_count"] > 0


def test_category_and_region_variants_found():
    pp, _ = prof("products")
    assert col(pp, "category")["whitespace_issue_count"] > 0
    sp, _ = prof("stores")
    assert col(sp, "region")["case_whitespace_variant_group_count"] >= 1


def test_pii_columns_are_redacted():
    c = col(prof("customers")[0], "email")
    assert c["pii_suspected"] and all(t["value"] == "[redacted]" for t in c["top_values"])


def test_identifier_with_duplicates_is_visible():
    c = col(prof("orders")[0], "order_id")
    assert c["looks_like_identifier"] and c["duplicate_value_rows"] == 15


def test_profile_is_deterministic_json_and_non_mutating():
    r = load_csv(SAMPLES / "orders.csv")
    before = r.df.copy(deep=True)
    a, b = profile_dataframe(r.df), profile_dataframe(r.df)
    assert a == b and json.dumps(a)
    pd.testing.assert_frame_equal(before, r.df)


def test_header_only_and_clean_files_do_not_crash():
    r = load_csv(EDGE / "header_only.csv")
    p = profile_dataframe(r.df)
    assert p["row_count"] == 0 and all(c["inferred_type"] == "empty" for c in p["columns"])
    p = profile_dataframe(load_csv(EDGE / "already_clean.csv").df)
    assert p["duplicate_rows"] == 0 and col(p, "price")["inferred_type"] == "integer"
