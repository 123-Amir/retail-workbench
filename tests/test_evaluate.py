import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.evaluate import runner
from app.evaluate.cli import main
from app.evaluate.expectations import MISSING, get_path, is_subset, run_assertion
from app.evaluate.runner import evaluate_file
from app.llm import base as llm_base

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CASES = ROOT / "cases" / "sample_cases.json"
STORES, ORDERS = str(ROOT / "data/samples/stores.csv"), str(ROOT / "data/samples/orders.csv")
MINIMUM_KEYS = {"dataset", "profile_before", "issues", "cleaning_plan", "profile_after", "validation", "chat_evaluation", "artifacts"}


def run(tmp_path, cases, name="r", **kw):
    inp = tmp_path / f"{name}_cases.json"
    inp.write_text(json.dumps({"cases": cases} if isinstance(cases, list) else cases))
    out, art = tmp_path / f"{name}_results.json", tmp_path / f"{name}_art"
    doc, code = evaluate_file(str(inp), str(out), str(art), **kw)
    return doc, code, out, art


def by_id(doc):
    return {r["case_id"]: r for r in doc["results"]}


def tree_hashes(d: Path):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def sample_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sample")
    doc, code, out, art = run(tmp, json.loads(SAMPLE_CASES.read_text()))
    return doc, code, out, art, tmp


# ================================ the shipped sample cases are a living regression suite ================================
def test_all_sample_cases_pass_with_every_check(sample_run):
    doc, code, out, art, _ = sample_run
    s = doc["summary"]
    assert code == 0 and s["cases_total"] == 15 and s["cases_passed"] == 15 and s["checks_failed"] == 0 and s["checks_total"] > 150
    assert s["datasets_failed"] == 3 and s["questions_total"] == 22 and s["provider"] == "mock"      # the 3 failures are the intentional bad inputs


def test_results_json_is_the_returned_document(sample_run):
    doc, _, out, _, _ = sample_run
    assert json.loads(out.read_text()) == json.loads(json.dumps(doc, sort_keys=True))


def test_every_result_has_the_minimum_machine_readable_shape(sample_run):
    for r in sample_run[0]["results"]:
        assert MINIMUM_KEYS <= set(r), r["case_id"]
        assert isinstance(r["issues"], list) and isinstance(r["cleaning_plan"], list) and isinstance(r["chat_evaluation"], list)
        for q in r["chat_evaluation"]:
            assert {"question", "plan", "answer", "trace", "status"} <= set(q)


def test_single_dataset_cases_are_flat_and_multi_dataset_cases_are_keyed(sample_run):
    r = by_id(sample_run[0])
    assert r["clean-orders"]["dataset"]["name"] == "orders" and r["clean-orders"]["validation"]["status"] == "pass"
    multi = r["retail-multi-dataset-conversation"]
    assert set(multi["dataset"]) == {"customers", "inventory", "orders", "products", "stores"} and multi["validation"]["orders"]["status"] == "pass"
    assert set(multi["datasets"]) == set(multi["dataset"])                                          # extension: always keyed
    assert {i["dataset"] for i in multi["issues"]} >= {"orders", "products"} and {s["dataset"] for s in multi["cleaning_plan"]} >= {"orders"}


def test_stable_ids_and_data_versions_give_full_traceability(sample_run):
    r = by_id(sample_run[0])["clean-orders"]
    assert all(i["id"].startswith("ISS-") for i in r["issues"]) and all(s["id"].startswith("STP-") and s["final_status"] in ("applied", "skipped", "flagged", "failed") for s in r["cleaning_plan"])
    ver = r["artifacts"]["datasets"]["orders"]["data_version"]
    chat = by_id(sample_run[0])["retail-multi-dataset-conversation"]["chat_evaluation"]
    assert ver.startswith("orders@") and chat[0]["evidence"]["datasets"]["orders"].startswith("orders@")
    assert all(s["execution"] is not None for s in r["cleaning_plan"])


def test_retail_chat_answers_carry_evidence_and_state(sample_run):
    q = by_id(sample_run[0])["retail-multi-dataset-conversation"]["chat_evaluation"]
    assert [x["status"] for x in q] == ["ok", "ok", "ok", "ok", "ok", "clarification_needed", "unanswerable", "unanswerable"]
    assert q[2]["plan"]["filters"][0]["value"] == "West" and q[2]["evidence"]["filters"][0]["rows_after"] == 380 and q[2]["grounded"] is True


# ================================ determinism ================================
def test_two_runs_are_byte_identical_including_artifacts(sample_run, tmp_path):
    doc, _, out, art, _ = sample_run
    doc2, _, out2, art2 = run(tmp_path, json.loads(SAMPLE_CASES.read_text()), name="again")
    assert out.read_bytes() == out2.read_bytes() and tree_hashes(art) == tree_hashes(art2)


def test_results_contain_no_volatile_or_host_specific_values(sample_run):
    import re
    text = sample_run[2].read_text()
    assert not re.search(r"\brun_[0-9a-f]{12}\b|\bchat_[0-9a-f]{12}\b|eval_\d+", text)                  # run/session/owner ids
    for needle in ("eval-store", "/tmp/", str(ROOT), "created_at", "asked_at", "finished_at", "@example.com"):
        assert needle not in text, needle


# ================================ artifacts ================================
def test_artifacts_exist_match_their_recorded_hashes_and_originals_are_untouched(sample_run):
    doc, _, _, art, _ = sample_run
    a = by_id(doc)["clean-orders"]["artifacts"]["datasets"]["orders"]
    for kind in ("original", "cleaned", "report"):
        assert (art / a[kind]).is_file()
    assert hashlib.sha256((art / a["original"]).read_bytes()).hexdigest() == a["original_sha256"] == hashlib.sha256(Path(ORDERS).read_bytes()).hexdigest()
    assert hashlib.sha256((art / a["cleaned"]).read_bytes()).hexdigest() == a["cleaned_sha256"] and a["data_version"] == f"orders@{a['cleaned_sha256'][:12]}"
    rep = json.loads((art / a["report"]).read_text())
    assert rep["validation"]["status"] == "pass" and "run" not in rep and rep["artifacts"]["cleaned_sha256"] == a["cleaned_sha256"]
    assert len((art / a["cleaned"]).read_text().splitlines()) == 1501                                # header + 1500 rows


def test_failed_datasets_publish_no_cleaned_artifact(sample_run):
    doc, _, _, art, _ = sample_run
    a = by_id(doc)["edge-batch-bad-files-do-not-abort"]["artifacts"]["datasets"]
    assert "cleaned" not in a["malformed"] and "cleaned" not in a["empty"] and "cleaned" in a["stores"]
    assert not (art / "edge-batch-bad-files-do-not-abort" / "malformed.cleaned.csv").exists()


def test_chat_trace_is_replayable_and_free_of_ids_and_prompts(sample_run):
    doc, _, _, art, _ = sample_run
    tr = json.loads((art / by_id(doc)["retail-multi-dataset-conversation"]["artifacts"]["chat_trace"]).read_text())
    assert [t["turn"] for t in tr["turns"]][:3] == [1, 2, 3] and {"question", "raw_plan", "plan", "execution", "answer"} <= set(tr["turns"][0])
    assert "session_id" not in tr and "You translate ONE" not in json.dumps(tr)


def test_rerun_clears_stale_files_in_a_case_directory(tmp_path):
    cases = [{"id": "c1", "files": [STORES]}]
    _, _, _, art = run(tmp_path, cases)
    stale = art / "c1" / "stale.txt"
    stale.write_text("old")
    run(tmp_path, cases)
    assert not stale.exists() and (art / "c1" / "stores.cleaned.csv").exists()


# ================================ failure isolation ================================
def test_one_bad_case_never_aborts_the_batch(tmp_path, monkeypatch):
    real = runner.Evaluator._execute

    def boom(self, case, idx, cdir, out):
        if case["id"] == "boom":
            raise ValueError("secret internal detail /etc/passwd")
        return real(self, case, idx, cdir, out)
    monkeypatch.setattr(runner.Evaluator, "_execute", boom)
    cases = [{"id": "g1", "files": [STORES]}, {"id": "bad-files", "files": "x"}, "not an object", {"id": "no-files"}, {"id": "boom", "files": [STORES]},
             {"id": "q-both", "files": [STORES], "questions": [{"question": "a", "plan": {"intent": "clarify", "question": "q"}}]},
             {"id": "g2", "files": [STORES], "questions": ["how many rows in stores"]}]
    doc, code, _, _ = run(tmp_path, cases)
    r = by_id(doc)
    assert code == 0 and [x["status"] for x in doc["results"]] == ["passed", "error", "error", "error", "error", "error", "passed"]
    assert r["bad-files"]["error"]["code"] == "INVALID_CASE" and r["no-files"]["error"]["code"] == "INVALID_CASE" and r["q-both"]["error"]["code"] == "INVALID_CASE"
    assert r["boom"]["error"] == {"code": "CASE_FAILED", "message": "Unexpected error while running the case.", "details": {"type": "ValueError"}}
    assert "secret" not in json.dumps(doc) and r["g2"]["chat_evaluation"][0]["result"]["rows"] == [{"row_count": 8}]
    assert all(MINIMUM_KEYS <= set(x) for x in doc["results"])                                       # error cases keep the output shape
    assert doc["summary"]["cases_error"] == 5


def test_duplicate_case_ids_are_reported_not_overwritten(tmp_path):
    doc, _, _, art = run(tmp_path, [{"id": "same", "files": [STORES]}, {"id": "same", "files": [ORDERS]}])
    assert [r["status"] for r in doc["results"]] == ["passed", "error"] and doc["results"][1]["error"]["code"] == "DUPLICATE_CASE_ID"
    assert (art / "same" / "stores.cleaned.csv").exists() and not (art / "same" / "orders.cleaned.csv").exists()   # first case's artifacts survive


def test_bad_inputs_inside_a_case_become_per_dataset_errors(tmp_path):
    (tmp_path / "dir.csv").mkdir()
    (tmp_path / "x.txt").write_text("hi")
    doc, _, _, _ = run(tmp_path, [{"id": "c", "files": [STORES, str(tmp_path / "dir.csv"), str(tmp_path / "nope.csv"), str(tmp_path / "x.txt")],
                                   "questions": ["how many rows in stores"]}])
    r = doc["results"][0]
    assert {n: (e["dataset"]["error"] or {}).get("code") for n, e in r["datasets"].items()} == {
        "stores": None, "dir": "FILE_NOT_FOUND", "nope": "FILE_NOT_FOUND", "x": "UNSUPPORTED_FILE_TYPE"}
    assert r["chat_evaluation"][0]["status"] == "ok" and r["status"] == "passed"


def test_oversized_inputs_fail_individually_and_never_get_read_into_memory(tmp_path):
    doc, _, _, art = run(tmp_path, [{"id": "big", "files": [ORDERS, STORES]}], max_upload_mb=0.01)       # 10 KB limit: orders.csv is ~90 KB
    d = doc["results"][0]["datasets"]
    assert d["orders"]["dataset"]["error"]["code"] == "FILE_TOO_LARGE" and d["stores"]["dataset"]["status"] == "succeeded"
    assert not (art / "big" / "orders.original.csv").exists()


def test_when_nothing_could_be_cleaned_every_question_gets_a_structured_error(tmp_path):
    doc, _, _, _ = run(tmp_path, [{"id": "c", "files": [str(tmp_path / "nope.csv")], "questions": ["net revenue by region", {"plan": {"intent": "clarify", "question": "q"}}]}])
    qs = doc["results"][0]["chat_evaluation"]
    assert [q["status"] for q in qs] == ["error", "error"] and {q["error"]["code"] for q in qs} == {"RUN_NOT_READY"} and doc["results"][0]["status"] == "passed"


def test_model_script_exhaustion_is_just_an_outage(tmp_path):
    doc, _, _, _ = run(tmp_path, [{"id": "c", "files": [ORDERS, str(ROOT / "data/samples/products.csv"), STORES], "model_script": [],
                                   "questions": ["net revenue by region"]}])
    q = doc["results"][0]["chat_evaluation"][0]
    assert q["status"] == "ok" and q["trace"]["planner"]["source"] == "rules"


# ================================ failing expectations are reported honestly ================================
def test_unmet_expectations_fail_the_case_with_expected_and_actual(tmp_path):
    doc, _, _, _ = run(tmp_path, [{"id": "wrong", "files": [STORES], "expect": {"datasets": {"stores": {"rows_out": 999, "status": "failed"}}},
                                   "questions": [{"question": "how many rows in stores", "expect": {"rows": [{"row_count": 7}], "status": "ok"}}]}])
    r = doc["results"][0]
    failed = {c["name"]: c for c in r["checks"] if not c["passed"]}
    assert r["status"] == "failed" and set(failed) == {"stores:rows_out", "stores:status", "q1:rows(exact, ordered)"}
    assert failed["stores:rows_out"]["expected"] == 999 and failed["stores:rows_out"]["actual"] == 8
    assert doc["summary"]["cases_failed"] == 1 and doc["summary"]["checks_failed"] == 3


# ================================ hidden datasets: arbitrary file names ================================
def test_files_can_be_mapped_to_the_dataset_names_retail_features_expect(tmp_path):
    for src, dst in [("orders", "sales_data"), ("products", "catalogue"), ("stores", "locations")]:
        (tmp_path / f"{dst}.csv").write_bytes((ROOT / f"data/samples/{src}.csv").read_bytes())
    files = [str(tmp_path / f"{n}.csv") for n in ("sales_data", "catalogue", "locations")]
    unmapped, _, _, _ = run(tmp_path, [{"id": "u", "files": files, "questions": ["net revenue by region", {"plan": {"intent": "aggregate", "dataset": "sales_data", "metrics": [{"agg": "count", "as": "n"}]}}]}], name="u")
    q = unmapped["results"][0]["chat_evaluation"]
    assert set(unmapped["results"][0]["datasets"]) == {"sales_data", "catalogue", "locations"}
    assert q[0]["status"] == "error" and q[0]["error"]["code"] == "DATASET_NOT_LOADED"               # retail features need canonical names...
    assert q[1]["status"] == "ok" and q[1]["result"]["rows"] == [{"n": 1500}]                         # ...generic chat works on any name
    mapped_files = [{"path": files[0], "name": "orders"}, {"path": files[1], "name": "products"}, {"path": files[2], "name": "stores"}]
    mapped, _, _, _ = run(tmp_path, [{"id": "m", "files": mapped_files, "questions": ["net revenue by region"]}], name="m")
    assert mapped["results"][0]["chat_evaluation"][0]["status"] == "ok" and set(mapped["results"][0]["datasets"]) == {"orders", "products", "stores"}


# ================================ scale ================================
def test_100k_plus_rows_run_through_the_same_pipeline(tmp_path):
    n = 120_000
    lines = ["order_id,order_date,store_id,sku,quantity,unit_price,status"]
    for i in range(n):
        lines.append(f"ORD-{i % 119_000:06d},2025-0{1 + i % 9}-1{i % 9},ST0{1 + i % 8}, sku{i % 500:04d} ,{1 + i % 5},${(i % 7 + 1) * 100}.00,{['completed', 'returned', 'Completed'][i % 3]}")
    big = tmp_path / "big_orders.csv"
    big.write_text("\n".join(lines) + "\n")
    doc, _, _, _ = run(tmp_path, [{"id": "big", "files": [{"path": str(big), "name": "orders"}],
                                   "expect": {"datasets": {"orders": {"status": "succeeded", "rows_in": n, "validation_status": "pass", "idempotent": True,
                                                                      "cleaned_columns": {"sku": {"all_match": r"SKU-\d{4}"}, "status": {"unique_count": 2}}}}},
                                   "questions": [{"plan": {"intent": "aggregate", "dataset": "orders", "group_by": ["status"], "metrics": [{"agg": "sum", "field": "quantity", "as": "u"}]},
                                                  "expect": {"status": "ok", "row_count": 2}}]}])
    r = doc["results"][0]
    assert r["status"] == "passed", [c for c in r["checks"] if not c["passed"]]
    assert r["dataset"]["rows_out"] == n                                                              # reused ids have different content: nothing may be dropped
    assert any(i["type"] == "KEY_CONFLICT" and i["columns"] == ["order_id"] for i in r["issues"])    # ...but the conflict is reported for a human


# ================================ CLI & fatal errors ================================
def test_cli_runs_as_a_module_exactly_like_the_assignment_command(tmp_path):
    out, art = tmp_path / "results.json", tmp_path / "evaluation_output"
    p = subprocess.run([sys.executable, "-m", "app.evaluate", "--input", str(SAMPLE_CASES), "--output", str(out), "--artifacts-dir", str(art)],
                       cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stderr
    assert "15 passed" in p.stderr and json.loads(out.read_text())["summary"]["cases_total"] == 15 and (art / "clean-orders" / "orders.cleaned.csv").exists()


def test_strict_mode_changes_only_the_exit_code(tmp_path):
    inp = tmp_path / "c.json"
    inp.write_text(json.dumps([{"id": "x", "files": [STORES], "expect": {"datasets": {"stores": {"rows_out": 1}}}}]))
    base = ["--input", str(inp), "--output", str(tmp_path / "o.json"), "--artifacts-dir", str(tmp_path / "a")]
    assert main(base) == 0 and main(base + ["--strict"]) == 1
    assert json.loads((tmp_path / "o.json").read_text())["summary"]["cases_failed"] == 1


@pytest.mark.parametrize("content,code", [("{not json", "INPUT_UNREADABLE"), ('{"cases": "nope"}', "INPUT_INVALID"), ('"just a string"', "INPUT_INVALID")])
def test_unusable_input_is_a_fatal_but_structured_result(tmp_path, content, code):
    inp = tmp_path / "bad.json"
    inp.write_text(content)
    out = tmp_path / "o.json"
    assert main(["--input", str(inp), "--output", str(out)]) == 2
    doc = json.loads(out.read_text())
    assert doc["fatal_error"]["code"] == code and doc["results"] == [] and doc["summary"] is None
    assert main(["--input", str(tmp_path / "missing.json"), "--output", str(out)]) == 2


def test_unconfigured_real_provider_is_a_fatal_config_error_not_a_mid_run_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_base, "settings", dataclasses.replace(llm_base.settings, anthropic_api_key=""))
    inp = tmp_path / "c.json"
    inp.write_text(json.dumps([{"id": "x", "files": [STORES]}]))
    assert main(["--input", str(inp), "--output", str(tmp_path / "o.json"), "--provider", "anthropic"]) == 2
    assert json.loads((tmp_path / "o.json").read_text())["fatal_error"]["code"] == "LLM_NOT_CONFIGURED"


def test_the_evaluator_drives_the_production_services(tmp_path, monkeypatch):
    calls = {"submit": 0, "ask": 0}
    from app.chat.service import ChatService
    from app.storage.runs import RunService
    rs, ca = RunService.submit, ChatService.ask
    monkeypatch.setattr(RunService, "submit", lambda self, *a, **k: (calls.__setitem__("submit", calls["submit"] + 1), rs(self, *a, **k))[1])
    monkeypatch.setattr(ChatService, "ask", lambda self, *a, **k: (calls.__setitem__("ask", calls["ask"] + 1), ca(self, *a, **k))[1])
    run(tmp_path, [{"id": "c", "files": [STORES], "questions": ["how many rows in stores", "how many rows in stores"]}])
    assert calls == {"submit": 1, "ask": 2}


# ================================ expectation engine ================================
DOC = {"result": {"rows": [{"products.category": "Tee", "n": 3}, {"products.category": "Jeans", "n": 1}], "totals": {"x": 1.5}},
       "answer": {"text": "Net revenue is 5,083,381"}, "evidence": {"missing_evidence": ["31 lines have no match in products", "snapshot caveat"]}}


def test_get_path_supports_indexes_and_dotted_keys():
    assert get_path(DOC, "result.rows.0.n") == 3 and get_path(DOC, 'result.rows.1["products.category"]') == "Jeans"
    assert get_path(DOC, "result.rows.9.n") is MISSING and get_path(DOC, "result.nope.x") is MISSING


@pytest.mark.parametrize("a,ok", [
    ({"path": "result.rows.0.n", "equals": 3}, True), ({"path": "result.rows.0.n", "equals": 4}, False),
    ({"path": "result.totals.x", "approx": 1.50004, "tol": 0.001}, True), ({"path": "result.totals.x", "approx": 2, "tol": 0.001}, False),
    ({"path": "answer.text", "contains": "net REVENUE"}, True), ({"path": "answer.text", "not_contains": "profit"}, True),
    ({"path": "evidence.missing_evidence", "contains": "no match in products"}, True), ({"path": "evidence.missing_evidence", "contains": "nonsense"}, False),
    ({"path": "evidence.missing_evidence", "not_contains": "nonsense"}, True),
    ({"path": "result.rows.0.n", "gte": 3}, True), ({"path": "result.rows.0.n", "gt": 3}, False), ({"path": "result.rows.0.n", "lt": 4}, True),
    ({"path": "result.rows", "length": 2}, True), ({"path": "answer.text", "regex": r"\d,\d{3},\d{3}"}, True),
    ({"path": "result.rows", "sorted_by": "n", "dir": "desc"}, True), ({"path": "result.rows", "sorted_by": "n", "dir": "asc"}, False),
    ({"path": "result.rows.0.n", "in": [1, 3]}, True), ({"path": "result.nope", "exists": False}, True), ({"path": "result.nope", "equals": 1}, False),
    ({"path": "result.rows.0.n", "is_null": True}, False), ({"path": "result.rows.0.n", "equals": 3, "gt": 1}, False), ({"path": "result.rows.0.n", "wat": 1}, False),
])
def test_assertion_operators(a, ok):
    assert run_assertion(a, DOC)["passed"] is ok


def test_plan_subset_matching_is_recursive_and_list_membership_based():
    plan = {"intent": "retail_metric", "group_by": ["category"], "filters": [{"field": "region", "op": "eq", "value": "West"}, {"field": "brand", "op": "eq", "value": "Nike"}]}
    assert is_subset({"filters": [{"field": "region", "value": "West"}]}, plan) and is_subset({"group_by": ["category"]}, plan)
    assert not is_subset({"filters": [{"field": "region", "value": "East"}]}, plan) and not is_subset({"intent": "aggregate"}, plan)
