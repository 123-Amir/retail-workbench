import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.llm.mock import MockProvider

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
H = {"X-Owner-Id": "alice"}


def files(*names):
    return [("files", (f"{n}.csv", (SAMPLES / f"{n}.csv").read_bytes(), "text/csv")) for n in names]


@pytest.fixture
def client(tmp_path):
    app = create_app(root=tmp_path / "store", provider=MockProvider(), max_upload_mb=5)
    with TestClient(app) as c:
        c.app_ = app
        yield c


def finished_run(client, *names, owner=H, **kw):
    r = client.post("/runs", files=files(*names), headers=owner, **kw)
    assert r.status_code == 202, r.text
    rid = r.json()["run_id"]
    client.app_.state.runs.wait(rid, 60)
    return rid


# ================================ upload -> clean -> inspect ================================
def test_health_needs_no_identity(client):
    assert client.get("/health").json() == {"status": "ok", "pipeline_version": "1.0", "llm_provider": "mock"}


def test_upload_is_accepted_asynchronously_and_reaches_a_finished_run(client):
    r = client.post("/runs", files=files("orders", "products", "stores"), headers=H)
    assert r.status_code == 202 and r.headers["location"] == f"/runs/{r.json()['run_id']}" and r.json()["reused"] is False
    rid = r.json()["run_id"]
    client.app_.state.runs.wait(rid, 60)
    run = client.get(f"/runs/{rid}", headers=H).json()
    assert run["status"] == "succeeded" and {d["name"]: d["rows_out"] for d in run["datasets"]} == {"orders": 1500, "products": 60, "stores": 8}
    assert all("_path" not in k for d in run["datasets"] for k in d)                      # internal storage paths are not exposed
    assert client.post("/runs", files=files("orders", "products", "stores"), headers=H).json() == {"run_id": rid, "status": "succeeded", "reused": True}


def test_report_and_cleaned_download_are_hash_verified(client):
    rid = finished_run(client, "orders")
    rep = client.get(f"/runs/{rid}/datasets/orders/report", headers=H).json()
    assert rep["validation"]["status"] == "pass" and "run" not in rep and set(rep["artifacts"]) == {"original_sha256", "cleaned_sha256"}
    dl = client.get(f"/runs/{rid}/datasets/orders/cleaned", headers=H)
    assert dl.headers["content-type"].startswith("text/csv") and hashlib.sha256(dl.content).hexdigest() == rep["artifacts"]["cleaned_sha256"]
    assert len(dl.text.splitlines()) == 1501


def test_decisions_form_field_changes_the_run(client):
    base = finished_run(client, "orders")
    step = next(s["id"] for s in client.get(f"/runs/{base}/datasets/orders/report", headers=H).json()["cleaning_plan"] if s["action"] == "drop_exact_duplicates")
    rid = finished_run(client, "orders", data={"decisions": json.dumps({step: "skip"})})
    assert rid != base and client.get(f"/runs/{rid}", headers=H).json()["datasets"][0]["rows_out"] == 1515


# ================================ chat ================================
def test_multi_turn_chat_over_http_keeps_state_and_evidence(client):
    rid = finished_run(client, "orders", "products", "stores", "customers", "inventory")
    sid = client.post(f"/runs/{rid}/sessions", headers=H).json()["session_id"]
    ask = lambda q: client.post(f"/sessions/{sid}/ask", json={"question": q}, headers=H)   # noqa: E731
    t1, t2, t3 = ask("net revenue by region").json(), ask("only stores in the West region").json(), ask("among those, which categories had the highest return rate?").json()
    assert [t["turn"] for t in (t1, t2, t3)] == [1, 2, 3] and t3["plan"]["filters"][0]["value"] == "West" and t3["evidence"]["filters"][0]["rows_after"] == 380
    assert t3["evidence"]["datasets"]["orders"].startswith("orders@") and t3["answer"]["text"]
    sess = client.get(f"/sessions/{sid}", headers=H).json()
    assert len(sess["turns"]) == 3 and sess["turns"][2]["execution"]["evidence"] and [s["session_id"] for s in client.get("/sessions", headers=H).json()["sessions"]] == [sid]


def test_structured_plan_and_analytic_errors_are_200_with_status_not_http_errors(client):
    rid = finished_run(client, "orders", "products")
    sid = client.post(f"/runs/{rid}/sessions", headers=H).json()["session_id"]
    bad = client.post(f"/sessions/{sid}/ask", json={"plan": {"intent": "aggregate", "dataset": "orders", "metrics": [{"agg": "sum", "field": "revenue", "as": "r"}]}}, headers=H)
    assert bad.status_code == 200 and bad.json()["status"] == "error" and bad.json()["error"]["code"] == "UNKNOWN_FIELD"
    good = client.post(f"/sessions/{sid}/ask", json={"plan": {"intent": "describe", "dataset": "orders", "select": ["quantity"]}}, headers=H).json()
    assert good["status"] == "ok" and good["trace"]["planner"]["source"] == "direct"


def test_no_prompts_or_internal_paths_in_any_response(client):
    rid = finished_run(client, "orders", "products", "stores")
    sid = client.post(f"/runs/{rid}/sessions", headers=H).json()["session_id"]
    bodies = [client.post(f"/sessions/{sid}/ask", json={"question": "net revenue by region"}, headers=H).text, client.get(f"/sessions/{sid}", headers=H).text,
              client.get(f"/runs/{rid}", headers=H).text, client.get(f"/runs/{rid}/datasets/orders/report", headers=H).text]
    for b in bodies:
        assert "You translate ONE" not in b and "/store/" not in b and "workbench.sqlite3" not in b


# ================================ isolation, auth, validation ================================
def test_other_owners_get_404_for_everything_of_mine(client):
    rid = finished_run(client, "stores")
    sid = client.post(f"/runs/{rid}/sessions", headers=H).json()["session_id"]
    eve = {"X-Owner-Id": "eve"}
    for method, url, kw in [("get", f"/runs/{rid}", {}), ("get", f"/runs/{rid}/datasets/stores/report", {}), ("get", f"/runs/{rid}/datasets/stores/cleaned", {}),
                            ("post", f"/runs/{rid}/sessions", {}), ("get", f"/sessions/{sid}", {}), ("post", f"/sessions/{sid}/ask", {"json": {"question": "x"}})]:
        r = getattr(client, method)(url, headers=eve, **kw)
        assert r.status_code == 404 and r.json()["error"]["code"] in ("RUN_NOT_FOUND", "SESSION_NOT_FOUND"), url
    assert client.get("/runs", headers=eve).json() == {"runs": []} and client.get("/sessions", headers=eve).json() == {"sessions": []}


@pytest.mark.parametrize("headers", [{}, {"X-Owner-Id": "../etc"}, {"X-Owner-Id": "a b"}, {"X-Owner-Id": "x" * 65}])
def test_identity_header_is_required_and_validated(client, headers):
    r = client.get("/runs", headers=headers)
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_OWNER"


def test_optional_bearer_token_is_enforced_when_configured(tmp_path):
    with TestClient(create_app(root=tmp_path / "s", provider=MockProvider(), api_token="s3cret")) as c:
        assert c.get("/runs", headers=H).status_code == 401
        assert c.get("/runs", headers={**H, "Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/runs", headers={**H, "Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200


def test_request_validation_errors_are_structured(client):
    cases = [(client.post("/runs", files=files("stores"), data={"decisions": "{not json"}, headers=H), 400, "INVALID_DECISION"),
             (client.post("/runs", files=files("stores"), data={"decisions": "[1]"}, headers=H), 400, "INVALID_DECISION"),
             (client.post("/runs", files=[("files", (f"f{i}.csv", b"a\n1\n", "text/csv")) for i in range(11)], headers=H), 400, "TOO_MANY_FILES")]
    rid = finished_run(client, "stores")
    sid = client.post(f"/runs/{rid}/sessions", headers=H).json()["session_id"]
    cases += [(client.post(f"/sessions/{sid}/ask", json={}, headers=H), 400, "INVALID_REQUEST"),
              (client.post(f"/sessions/{sid}/ask", json={"question": "a", "plan": {}}, headers=H), 400, "INVALID_REQUEST"),
              (client.post(f"/sessions/{sid}/ask", json={"question": "x" * 1001}, headers=H), 400, "INVALID_QUESTION"),
              (client.get(f"/runs/{rid}/datasets/ghost/report", headers=H), 404, "DATASET_NOT_FOUND")]
    for r, status, code in cases:
        assert r.status_code == status and r.json()["error"]["code"] == code, (r.text, code)
    assert client.post(f"/sessions/{sid}/ask", json={"question": "a", "surprise": 1}, headers=H).status_code == 422


def test_oversized_upload_is_rejected_per_file_without_sinking_the_run(client):
    big = ("files", ("huge.csv", b"a\n" + b"1\n" * 3_000_000, "text/csv"))          # 6 MB > 5 MB limit
    r = client.post("/runs", files=[big] + files("stores"), headers=H)
    rid = r.json()["run_id"]
    client.app_.state.runs.wait(rid, 60)
    run = client.get(f"/runs/{rid}", headers=H).json()
    assert run["status"] == "partial" and {d["name"]: (d["error"] or {}).get("code") for d in run["datasets"]} == {"huge": "FILE_TOO_LARGE", "stores": None}


def test_cleaned_download_before_ready_or_for_failed_dataset_is_a_409(client):
    r = client.post("/runs", files=[("files", ("bad.csv", b"a,b\n1,2\n3,4,5,6\n", "text/csv")), *files("stores")], headers=H)
    rid = r.json()["run_id"]
    client.app_.state.runs.wait(rid, 60)
    d = client.get(f"/runs/{rid}/datasets/bad/cleaned", headers=H)
    assert d.status_code == 409 and d.json()["error"]["code"] == "DATASET_NOT_READY"
    assert client.post(f"/runs/{rid}/sessions", headers=H).status_code == 201           # the good dataset is still chat-able


def test_unexpected_server_errors_do_not_leak_internals(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret /var/lib/db password=hunter2")
    monkeypatch.setattr(client.app_.state.runs, "list_runs", boom)
    r = TestClient(client.app_, raise_server_exceptions=False).get("/runs", headers=H)
    assert r.status_code == 500 and r.json()["error"]["code"] == "INTERNAL_ERROR" and "hunter2" not in r.text
