import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from app.errors import PipelineError
from app.pipeline import cleaner
from app.pipeline.cleaner import clean_dataframe
from app.storage import runs as runs_mod
from app.storage.paths import dataset_name_from_filename, safe_join, validate_owner
from app.storage.runs import RunService

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"


def sample(*names):
    return [(f"{n}.csv", (SAMPLES / f"{n}.csv").read_bytes()) for n in names]


@pytest.fixture
def svc(tmp_path):
    return RunService(root=tmp_path / "store", max_upload_mb=5)


def ds(run, name):
    return next(d for d in run["datasets"] if d["name"] == name)


# ---------- happy path & artifacts ----------
def test_run_persists_original_and_cleaned_artifacts_with_hashes(svc):
    files = sample("orders", "products")
    r = svc.submit("alice", files, wait=True)
    run = svc.get_run("alice", r["run_id"])
    assert run["status"] == "succeeded" and [e["event"] for e in run["events"]][:2] == ["submitted", "started"]
    o = ds(run, "orders")
    assert (o["rows_in"], o["rows_out"], o["validation_status"]) == (1515, 1500, "pass")
    orig = safe_join(svc.root, "alice", o["original_path"]).read_bytes()
    assert orig == files[0][1] or orig == files[1][1]                          # original stored byte-for-byte
    assert hashlib.sha256(orig).hexdigest() == o["sha256"]
    cleaned = safe_join(svc.root, "alice", o["cleaned_path"]).read_bytes()
    assert hashlib.sha256(cleaned).hexdigest() == o["cleaned_sha256"] and o["data_version"] == f"orders@{o['cleaned_sha256'][:12]}"
    rep = svc.get_report("alice", r["run_id"], "orders")
    for k in ("dataset", "profile_before", "issues", "cleaning_plan", "profile_after", "validation", "artifacts"):
        assert k in rep
    assert not list(svc.root.rglob("*.tmp-*"))                                  # atomic writes leave no temp files


def test_cleaned_artifact_is_a_fixed_point_and_typed_read_restores_dtypes(svc):
    r = svc.submit("alice", sample("orders"), wait=True)
    df = svc.read_cleaned("alice", r["run_id"], "orders")
    again = clean_dataframe(df, "orders")                                       # re-cleaning the *stored file* changes nothing
    assert again.df.equals(df) and not [s for s in again.report["cleaning_plan"] if s["execution"].get("effective")]
    t = svc.read_cleaned("alice", r["run_id"], "orders", typed=True)
    assert str(t["quantity"].dtype) == "Int64" and str(t["discount"].dtype) == "float64"
    assert pd.api.types.is_datetime64_any_dtype(t["order_date"]) and len(t) == 1500


# ---------- idempotency / concurrency ----------
def test_resubmitting_same_content_reuses_the_run_without_recomputing(svc):
    a = svc.submit("alice", sample("stores", "products"), wait=True)
    path = safe_join(svc.root, "alice", ds(svc.get_run("alice", a["run_id"]), "stores")["cleaned_path"])
    mtime = path.stat().st_mtime_ns
    b = svc.submit("alice", list(reversed(sample("stores", "products"))), wait=True)   # file order must not matter
    assert b["reused"] and b["run_id"] == a["run_id"] and path.stat().st_mtime_ns == mtime
    assert len(svc.list_runs("alice")) == 1


def test_different_decisions_or_content_create_a_new_run(svc):
    base = svc.submit("alice", sample("orders"), wait=True)
    step_id = next(s["id"] for s in svc.get_report("alice", base["run_id"], "orders")["cleaning_plan"] if s["action"] == "drop_exact_duplicates")
    other = svc.submit("alice", sample("orders"), decisions={step_id: "skip"}, wait=True)
    assert not other["reused"] and other["run_id"] != base["run_id"]
    assert ds(svc.get_run("alice", other["run_id"]), "orders")["rows_out"] == 1515
    changed = svc.submit("alice", [("orders.csv", sample("orders")[0][1] + b"ORD-99999,2025-01-01,C0001,ST01,SKU-0001,1,499,0,completed\n")], wait=True)
    assert not changed["reused"]


def test_concurrent_identical_submits_produce_exactly_one_run(svc):
    files = sample("stores", "products")
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda _: svc.submit("alice", files, wait=True), range(8)))
    assert len({r["run_id"] for r in results}) == 1 and sum(1 for r in results if not r["reused"]) == 1
    assert len(svc.list_runs("alice")) == 1
    assert svc.get_run("alice", results[0]["run_id"])["status"] == "succeeded"


def test_background_processing_and_double_process_guard(svc):
    r = svc.submit("alice", sample("stores"), wait=False)
    svc.wait(r["run_id"], timeout=30)
    run = svc.get_run("alice", r["run_id"])
    assert run["status"] == "succeeded"
    n_events = len(run["events"])
    svc.process("alice", r["run_id"])                                          # already finished: must be a no-op
    assert len(svc.get_run("alice", r["run_id"])["events"]) == n_events


# ---------- isolation & input safety ----------
def test_owners_are_fully_isolated(svc):
    a = svc.submit("alice", sample("stores"), wait=True)
    for call in (lambda: svc.get_run("bob", a["run_id"]), lambda: svc.get_report("bob", a["run_id"], "stores"),
                 lambda: svc.read_cleaned("bob", a["run_id"], "stores")):
        with pytest.raises(PipelineError) as e:
            call()
        assert e.value.code == "RUN_NOT_FOUND"                                  # same answer as a non-existent run: no existence leak
    assert svc.list_runs("bob") == []
    b = svc.submit("bob", sample("stores"), wait=True)                          # same content, different owner => separate run
    assert b["run_id"] != a["run_id"] and not b["reused"]
    assert (svc.root / "alice" / a["run_id"]).is_dir() and (svc.root / "bob" / b["run_id"]).is_dir()


def test_owner_and_path_inputs_are_validated():
    for bad in ("../x", "a/b", "", "x" * 65, "a b", None):
        with pytest.raises(PipelineError) as e:
            validate_owner(bad)
        assert e.value.code == "INVALID_OWNER"
    with pytest.raises(PipelineError) as e:
        safe_join(Path("/tmp/store"), "alice", "../../etc/passwd")
    assert e.value.code == "PATH_ESCAPE"
    assert dataset_name_from_filename("../../Evil Name!.csv") == "evil_name"


def test_safe_join_accepts_a_child_path_under_the_storage_root(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    child = safe_join(root, "alice", "run_123")
    assert child == root / "alice" / "run_123"


def test_malicious_filename_cannot_escape_the_run_directory(svc):
    r = svc.submit("alice", [("../../../evil.csv", b"a,b\n1,2\n")], wait=True)
    run = svc.get_run("alice", r["run_id"])
    assert ds(run, "evil")["status"] == "succeeded"
    assert not (svc.root.parent / "evil.csv").exists() and (svc.root / "alice" / r["run_id"] / "original" / "evil.csv").exists()


# ---------- failure handling ----------
def test_bad_files_fail_individually_and_the_run_is_partial(svc):
    files = sample("stores") + [("notes.txt", b"hello"), ("empty.csv", b""), ("huge.csv", b"a\n" + b"1\n" * 3_000_000)]
    run = svc.get_run("alice", svc.submit("alice", files, wait=True)["run_id"])
    assert run["status"] == "partial" and ds(run, "stores")["status"] == "succeeded"
    assert {n: ds(run, n)["error"]["code"] for n in ("notes", "empty", "huge")} == {
        "notes": "UNSUPPORTED_FILE_TYPE", "empty": "EMPTY_FILE", "huge": "FILE_TOO_LARGE"}


def test_all_files_rejected_gives_failed_run_and_can_be_retried(svc):
    r = svc.submit("alice", [("a.txt", b"x")], wait=True)
    assert r["status"] == "failed"
    r2 = svc.submit("alice", [("a.txt", b"x")], wait=True)
    assert not r2["reused"] and r2["run_id"] != r["run_id"]                      # failed runs never block a retry


def test_submit_level_errors_are_structured(svc):
    for args, code in [(([],), "NO_FILES"), (([("a.csv", b"x"), ("A.csv", b"y")],), "DUPLICATE_DATASET_NAME"),
                       (([(f"f{i}.csv", b"a\n1\n") for i in range(11)],), "TOO_MANY_FILES")]:
        with pytest.raises(PipelineError) as e:
            svc.submit("alice", *args)
        assert e.value.code == code
    with pytest.raises(PipelineError) as e:
        svc.submit("alice", sample("stores"), decisions={"STP-1": "maybe"})
    assert e.value.code == "INVALID_DECISION"


def test_unexpected_crash_in_one_dataset_does_not_abort_the_run(svc, monkeypatch):
    real = runs_mod.clean_dataframe

    def flaky(df, name, decisions=None, **kw):
        if name == "products":
            raise RuntimeError("boom")
        return real(df, name, decisions, **kw)

    monkeypatch.setattr(runs_mod, "clean_dataframe", flaky)
    run = svc.get_run("alice", svc.submit("alice", sample("products", "stores"), wait=True)["run_id"])
    assert run["status"] == "partial" and ds(run, "stores")["status"] == "succeeded"
    p = ds(run, "products")
    assert p["status"] == "failed" and p["error"]["code"] == "INTERNAL_ERROR" and p["cleaned_path"] is None
    assert not (svc.root / "alice" / run["run_id"] / "cleaned" / "products.csv").exists()
    with pytest.raises(PipelineError) as e:
        svc.read_cleaned("alice", run["run_id"], "products")
    assert e.value.code == "DATASET_NOT_READY"


def test_failed_validation_publishes_no_cleaned_artifact_but_keeps_evidence(svc, monkeypatch):
    real = cleaner.EXEC["trim_whitespace"]

    def sneaky(df, step, fields):
        out, log = real(df, step, fields)
        out.loc[out[fields[0]].first_valid_index(), fields[0]] = None            # silent data loss
        return out, log

    monkeypatch.setitem(cleaner.EXEC, "trim_whitespace", sneaky)
    run = svc.get_run("alice", svc.submit("alice", sample("orders"), wait=True)["run_id"])
    o = ds(run, "orders")
    assert run["status"] == "failed" and o["error"]["code"] == "VALIDATION_FAILED" and "null_accounting" in o["error"]["details"]["failed_checks"]
    assert o["cleaned_path"] is None and o["original_path"]                      # original kept, nothing corrupt is usable
    assert svc.get_report("alice", run["run_id"], "orders")["validation"]["status"] == "fail"


def test_tampered_artifact_is_detected_on_read(svc):
    r = svc.submit("alice", sample("stores"), wait=True)
    d = ds(svc.get_run("alice", r["run_id"]), "stores")
    p = safe_join(svc.root, "alice", d["cleaned_path"])
    p.write_text(p.read_text().replace("ST01", "ST99"))
    with pytest.raises(PipelineError) as e:
        svc.read_cleaned("alice", r["run_id"], "stores")
    assert e.value.code == "ARTIFACT_CORRUPT"
    assert len(svc.read_cleaned("alice", r["run_id"], "stores", verify=False)) == 8


def test_roundtrip_mismatch_blocks_publication(svc, monkeypatch):
    monkeypatch.setattr(runs_mod, "_equivalent", lambda a, b: False)
    run = svc.get_run("alice", svc.submit("alice", sample("stores"), wait=True)["run_id"])
    assert ds(run, "stores")["error"]["code"] == "ARTIFACT_ROUNDTRIP_MISMATCH" and ds(run, "stores")["cleaned_path"] is None


def test_crash_recovery_marks_interrupted_runs_failed(svc):
    r = svc.submit("alice", sample("stores"), wait=False)
    svc.wait(r["run_id"], 30)
    with svc.db.connect() as con:                                                # simulate a process that died mid-run
        con.execute("UPDATE runs SET status='running' WHERE run_id=?", (r["run_id"],))
        con.execute("UPDATE datasets SET status='running' WHERE run_id=?", (r["run_id"],))
    assert svc.recover_stale_runs() == 1
    run = svc.get_run("alice", r["run_id"])
    assert run["status"] == "failed" and run["error"]["code"] == "INTERRUPTED" and ds(run, "stores")["error"]["code"] == "INTERRUPTED"
    assert not svc.submit("alice", sample("stores"), wait=True)["reused"]       # retry is allowed


# ---------- edge cases ----------
def test_header_only_dataset_succeeds_with_zero_rows(svc):
    run = svc.get_run("alice", svc.submit("alice", [("h.csv", b"a,b,c\n")], wait=True)["run_id"])
    d = ds(run, "h")
    assert d["status"] == "succeeded" and (d["rows_in"], d["rows_out"]) == (0, 0)
    assert list(svc.read_cleaned("alice", run["run_id"], "h").columns) == ["a", "b", "c"]


def test_database_is_durable_across_service_restarts(tmp_path):
    root = tmp_path / "store"
    r = RunService(root=root).submit("alice", sample("stores"), wait=True)
    s2 = RunService(root=root)                                                  # fresh process, same storage
    assert s2.get_run("alice", r["run_id"])["status"] == "succeeded" and len(s2.read_cleaned("alice", r["run_id"], "stores")) == 8
    assert s2.submit("alice", sample("stores"), wait=True)["reused"]
