"""Run service: submit -> (background) process -> inspect. Owns state, artifacts and idempotency.

State model:  run: queued -> running -> succeeded | partial | failed
              dataset: pending -> running -> succeeded | failed
Guarantees:
- Idempotent: same owner + same file contents + same decisions + same pipeline version => same run (reused).
- A failed dataset never aborts the others; a dataset whose validation FAILS publishes no cleaned artifact.
- Artifacts are written atomically and hashed; reads verify the hash (tamper/corruption detection).
- Every query is scoped by owner; another owner's run looks exactly like a missing run.
- Memory is bounded by max_upload_mb per file (files are read whole; documented limit).
"""
import hashlib
import json
import shutil
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import settings
from app.errors import PipelineError
from app.pipeline.cleaner import check_decisions, clean_dataframe
from app.pipeline.loader import load_csv, sanitize_filename
from app.storage.db import Database
from app.storage.files import atomic_write_bytes
from app.storage.paths import dataset_name_from_filename, safe_join, validate_owner
from app.storage.typed import typed_frame

PIPELINE_VERSION = "1.0"
MAX_FILES = 10


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dumps(obj) -> str:
    return json.dumps(obj, default=str, allow_nan=False)


def _equivalent(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Same shape, columns, null positions and values (ignores string-vs-object dtype)."""
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    na = a.isna()
    if not na.equals(b.isna()):
        return False
    return bool((a.where(~na, "").astype(str).to_numpy() == b.where(~na, "").astype(str).to_numpy()).all())


class RunService:
    def __init__(self, root=None, max_upload_mb: float | None = None, max_workers: int = 2):
        self.root = Path(root or settings.data_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.root / "workbench.sqlite3")
        self.max_upload_mb = settings.max_upload_mb if max_upload_mb is None else max_upload_mb
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = {}

    # ------------------------------------------------------------------ submit
    def _precheck(self, files):
        entries, seen = [], set()
        for filename, data in files:
            name = dataset_name_from_filename(filename)
            if name in seen:
                raise PipelineError("DUPLICATE_DATASET_NAME", "Two files map to the same dataset name.", {"name": name})
            seen.add(name)
            e = {"name": name, "filename": sanitize_filename(filename), "data": None, "sha256": None, "size": len(data), "error": None}
            if not e["filename"].lower().endswith(".csv"):
                e["error"] = PipelineError("UNSUPPORTED_FILE_TYPE", "Only .csv files are accepted.").to_dict()
            elif len(data) == 0:
                e["error"] = PipelineError("EMPTY_FILE", "File has 0 bytes.").to_dict()
            elif len(data) > self.max_upload_mb * 1024 * 1024:
                e["error"] = PipelineError("FILE_TOO_LARGE", f"File exceeds {self.max_upload_mb} MB limit.", {"size_bytes": len(data)}).to_dict()
            else:
                e["data"], e["sha256"] = data, hashlib.sha256(data).hexdigest()
            entries.append(e)
        return sorted(entries, key=lambda e: e["name"])

    @staticmethod
    def _idem_key(owner, entries, decisions) -> str:
        payload = {"owner": owner, "pipeline": PIPELINE_VERSION, "decisions": dict(sorted(decisions.items())),
                   "files": [[e["name"], e["sha256"] or e["error"]["code"]] for e in entries]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _find_live(self, owner, key):
        with self.db.connect() as con:
            r = con.execute("SELECT run_id,status FROM runs WHERE owner=? AND idempotency_key=? AND status!='failed'", (owner, key)).fetchone()
        return (r["run_id"], r["status"]) if r else None

    def submit(self, owner: str, files: list, decisions: dict | None = None, wait: bool = False) -> dict:
        """files: [(filename, bytes)]. wait=True processes inline (used by the evaluator); otherwise in the background."""
        validate_owner(owner)
        decisions = check_decisions(decisions)
        if not files:
            raise PipelineError("NO_FILES", "At least one file is required.")
        if len(files) > MAX_FILES:
            raise PipelineError("TOO_MANY_FILES", f"At most {MAX_FILES} files per run.")
        entries = self._precheck(files)
        key = self._idem_key(owner, entries, decisions)
        live = self._find_live(owner, key)
        if live:
            return {"run_id": live[0], "status": live[1], "reused": True}

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run_dir = safe_join(self.root, owner, run_id)
        all_bad = all(e["error"] for e in entries)
        try:
            for e in entries:
                if e["data"] is not None:
                    e["rel"] = f"{run_id}/original/{e['name']}.csv"
                    atomic_write_bytes(safe_join(self.root, owner, e["rel"]), e["data"])
            with self.db.connect() as con:
                con.execute("INSERT INTO runs(run_id,owner,status,idempotency_key,options_json,pipeline_version,created_at,finished_at,error_json)"
                            " VALUES(?,?,?,?,?,?,?,?,?)",
                            (run_id, owner, "failed" if all_bad else "queued", key, _dumps({"decisions": decisions}),
                             PIPELINE_VERSION, _now(), _now() if all_bad else None,
                             _dumps({"code": "ALL_FILES_REJECTED"}) if all_bad else None))
                for e in entries:
                    con.execute("INSERT INTO datasets(run_id,name,status,original_filename,sha256,size_bytes,original_path,error_json)"
                                " VALUES(?,?,?,?,?,?,?,?)",
                                (run_id, e["name"], "failed" if e["error"] else "pending", e["filename"], e["sha256"], e["size"],
                                 e.get("rel"), _dumps(e["error"]) if e["error"] else None))
                self._event(con, run_id, "submitted", {"files": len(entries), "rejected": sum(1 for e in entries if e["error"])})
        except sqlite3.IntegrityError:   # a concurrent identical submit won the race
            shutil.rmtree(run_dir, ignore_errors=True)
            live = self._find_live(owner, key)
            if live:
                return {"run_id": live[0], "status": live[1], "reused": True}
            raise
        if all_bad:
            return {"run_id": run_id, "status": "failed", "reused": False}
        if wait:
            self.process(owner, run_id)
        else:
            self._futures[run_id] = self._pool.submit(self.process, owner, run_id)
        return {"run_id": run_id, "status": self.get_run(owner, run_id)["status"], "reused": False}

    def wait(self, run_id: str, timeout: float | None = None):
        f = self._futures.get(run_id)
        if f:
            f.result(timeout=timeout)

    # ----------------------------------------------------------------- process
    @staticmethod
    def _event(con, run_id, event, detail=None):
        con.execute("INSERT INTO events(run_id,ts,event,detail_json) VALUES(?,?,?,?)", (run_id, _now(), event, _dumps(detail or {})))

    def process(self, owner: str, run_id: str) -> dict:
        validate_owner(owner)
        with self.db.connect() as con:
            claimed = con.execute("UPDATE runs SET status='running', started_at=? WHERE run_id=? AND owner=? AND status='queued'",
                                  (_now(), run_id, owner)).rowcount == 1
            if claimed:
                self._event(con, run_id, "started")
        if not claimed:                      # already running/finished elsewhere: never process twice
            return self.get_run(owner, run_id)
        with self.db.connect() as con:
            decisions = json.loads(con.execute("SELECT options_json FROM runs WHERE run_id=?", (run_id,)).fetchone()[0])["decisions"]
            todo = [r["name"] for r in con.execute("SELECT name FROM datasets WHERE run_id=? AND status='pending' ORDER BY name", (run_id,))]
        for name in todo:
            self._process_dataset(owner, run_id, name, decisions)
        with self.db.connect() as con:
            st = [r[0] for r in con.execute("SELECT status FROM datasets WHERE run_id=?", (run_id,))]
            final = "succeeded" if all(s == "succeeded" for s in st) else ("failed" if not any(s == "succeeded" for s in st) else "partial")
            con.execute("UPDATE runs SET status=?, finished_at=? WHERE run_id=?", (final, _now(), run_id))
            self._event(con, run_id, "finished", {"status": final})
        return self.get_run(owner, run_id)

    def _fail_dataset(self, run_id, name, error: dict, **cols):
        with self.db.connect() as con:
            sets = ", ".join(f"{k}=?" for k in cols)
            con.execute(f"UPDATE datasets SET status='failed', error_json=?{', ' + sets if sets else ''} WHERE run_id=? AND name=?",
                        (_dumps(error), *cols.values(), run_id, name))
            self._event(con, run_id, "dataset_failed", {"name": name, "code": error.get("code")})

    def _process_dataset(self, owner, run_id, name, decisions):
        with self.db.connect() as con:
            con.execute("UPDATE datasets SET status='running' WHERE run_id=? AND name=?", (run_id, name))
            row = con.execute("SELECT original_path FROM datasets WHERE run_id=? AND name=?", (run_id, name)).fetchone()
        try:
            lr = load_csv(safe_join(self.root, owner, row["original_path"]), max_mb=self.max_upload_mb)
            res = clean_dataframe(lr.df, name, decisions)
            report, vstatus = res.report, res.report["validation"]["status"]
            artifacts = {"original": row["original_path"], "original_sha256": lr.sha256, "cleaned": None, "cleaned_sha256": None}
            cleaned_rel = cleaned_sha = data_version = None
            error = None

            if vstatus == "fail":
                bad = [c["name"] for c in report["validation"]["checks"] if c["status"] == "fail"]
                error = {"code": "VALIDATION_FAILED", "message": "Cleaning failed validation; no cleaned artifact was published.", "details": {"failed_checks": bad}}
            else:
                data = res.df.to_csv(index=False, lineterminator="\n").encode("utf-8")
                cleaned_rel = f"{run_id}/cleaned/{name}.csv"
                path = safe_join(self.root, owner, cleaned_rel)
                atomic_write_bytes(path, data)
                if not _equivalent(load_csv(path, max_mb=self.max_upload_mb + 1).df, res.df):   # read-back proof
                    path.unlink()
                    cleaned_rel = None
                    error = {"code": "ARTIFACT_ROUNDTRIP_MISMATCH", "message": "Written cleaned file does not match the in-memory result."}
                else:
                    cleaned_sha = hashlib.sha256(data).hexdigest()
                    data_version = f"{name}@{cleaned_sha[:12]}"
                    artifacts.update(cleaned=cleaned_rel, cleaned_sha256=cleaned_sha)

            report["dataset"].update({"filename": lr.filename, "encoding": lr.encoding, "sha256": lr.sha256, "size_bytes": lr.size_bytes,
                                      "loader_warnings": lr.warnings, "data_version": data_version})
            report["artifacts"] = artifacts
            report["run"] = {"run_id": run_id, "pipeline_version": PIPELINE_VERSION, "status": "failed" if error else "succeeded", "error": error}
            report_rel = f"{run_id}/reports/{name}.json"
            atomic_write_bytes(safe_join(self.root, owner, report_rel), _dumps(report).encode("utf-8"))

            common = dict(rows_in=report["dataset"]["rows_in"], rows_out=report["dataset"]["rows_out"], validation_status=vstatus,
                          report_path=report_rel, sha256=lr.sha256, size_bytes=lr.size_bytes, schema_json=_dumps(report["schema_after"]))
            if error:
                self._fail_dataset(run_id, name, error, **common)
                return
            with self.db.connect() as con:
                cols = dict(common, status="succeeded", data_version=data_version, cleaned_path=cleaned_rel, cleaned_sha256=cleaned_sha)
                con.execute(f"UPDATE datasets SET {', '.join(f'{k}=?' for k in cols)} WHERE run_id=? AND name=?", (*cols.values(), run_id, name))
                self._event(con, run_id, "dataset_succeeded", {"name": name, "validation": vstatus, "data_version": data_version})
        except PipelineError as e:
            self._fail_dataset(run_id, name, e.to_dict())
        except Exception as e:   # never leak internals or abort the run
            self._fail_dataset(run_id, name, {"code": "INTERNAL_ERROR", "message": "Unexpected error while processing dataset.",
                                              "details": {"type": type(e).__name__, "detail": str(e)[:150]}})

    # ------------------------------------------------------------------- reads
    def _run_row(self, con, owner, run_id):
        r = con.execute("SELECT * FROM runs WHERE run_id=? AND owner=?", (run_id, owner)).fetchone()
        if not r:   # same error whether the run is missing or belongs to someone else
            raise PipelineError("RUN_NOT_FOUND", "Run not found.", {"run_id": run_id})
        return r

    def get_run(self, owner: str, run_id: str) -> dict:
        validate_owner(owner)
        with self.db.connect() as con:
            r = self._run_row(con, owner, run_id)
            ds = con.execute("SELECT * FROM datasets WHERE run_id=? ORDER BY name", (run_id,)).fetchall()
            ev = con.execute("SELECT ts,event,detail_json FROM events WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
        return {"run_id": run_id, "status": r["status"], "created_at": r["created_at"], "started_at": r["started_at"],
                "finished_at": r["finished_at"], "pipeline_version": r["pipeline_version"], "options": json.loads(r["options_json"]),
                "error": json.loads(r["error_json"]) if r["error_json"] else None,
                "datasets": [{"name": d["name"], "status": d["status"], "filename": d["original_filename"], "rows_in": d["rows_in"],
                              "rows_out": d["rows_out"], "validation_status": d["validation_status"], "data_version": d["data_version"],
                              "sha256": d["sha256"], "cleaned_sha256": d["cleaned_sha256"], "original_path": d["original_path"],
                              "cleaned_path": d["cleaned_path"], "report_path": d["report_path"],
                              "schema": json.loads(d["schema_json"]) if d["schema_json"] else None,
                              "error": json.loads(d["error_json"]) if d["error_json"] else None} for d in ds],
                "events": [{"ts": e["ts"], "event": e["event"], "detail": json.loads(e["detail_json"])} for e in ev]}

    def list_runs(self, owner: str, limit: int = 50) -> list:
        validate_owner(owner)
        with self.db.connect() as con:
            rows = con.execute("SELECT run_id,status,created_at,finished_at FROM runs WHERE owner=? ORDER BY created_at DESC, run_id LIMIT ?",
                               (owner, min(max(int(limit), 1), 200))).fetchall()
        return [dict(r) for r in rows]

    def _dataset(self, owner, run_id, name):
        with self.db.connect() as con:
            self._run_row(con, owner, run_id)
            d = con.execute("SELECT * FROM datasets WHERE run_id=? AND name=?", (run_id, name)).fetchone()
        if not d:
            raise PipelineError("DATASET_NOT_FOUND", "Dataset not found in this run.", {"name": name})
        return d

    def get_report(self, owner: str, run_id: str, name: str) -> dict:
        validate_owner(owner)
        d = self._dataset(owner, run_id, name)
        if not d["report_path"]:
            raise PipelineError("REPORT_NOT_AVAILABLE", "No report for this dataset.", {"status": d["status"]})
        return json.loads(safe_join(self.root, owner, d["report_path"]).read_text("utf-8"))

    def read_cleaned(self, owner: str, run_id: str, name: str, typed: bool = False, verify: bool = True) -> pd.DataFrame:
        validate_owner(owner)
        d = self._dataset(owner, run_id, name)
        if d["status"] != "succeeded" or not d["cleaned_path"]:
            raise PipelineError("DATASET_NOT_READY", "No cleaned artifact is available for this dataset.", {"status": d["status"]})
        lr = load_csv(safe_join(self.root, owner, d["cleaned_path"]), max_mb=self.max_upload_mb + 1)
        if verify and lr.sha256 != d["cleaned_sha256"]:
            raise PipelineError("ARTIFACT_CORRUPT", "Cleaned artifact hash does not match the recorded hash.", {"name": name})
        return typed_frame(lr.df, json.loads(d["schema_json"])) if typed else lr.df

    def cleaned_bytes(self, owner: str, run_id: str, name: str) -> bytes:
        """Exact bytes of the cleaned artifact, hash-verified (what a download endpoint serves)."""
        validate_owner(owner)
        d = self._dataset(owner, run_id, name)
        if d["status"] != "succeeded" or not d["cleaned_path"]:
            raise PipelineError("DATASET_NOT_READY", "No cleaned artifact is available for this dataset.", {"status": d["status"]})
        data = safe_join(self.root, owner, d["cleaned_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != d["cleaned_sha256"]:
            raise PipelineError("ARTIFACT_CORRUPT", "Cleaned artifact hash does not match the recorded hash.", {"name": name})
        return data

    # ---------------------------------------------------------------- recovery
    def recover_stale_runs(self) -> int:
        """Call at startup: runs left 'queued'/'running' by a crashed process are marked failed (retry creates a new run)."""
        with self.db.connect() as con:
            ids = [r[0] for r in con.execute("SELECT run_id FROM runs WHERE status IN ('queued','running')")]
            err = _dumps({"code": "INTERRUPTED", "message": "Process stopped before the run finished."})
            for rid in ids:
                con.execute("UPDATE runs SET status='failed', finished_at=?, error_json=? WHERE run_id=?", (_now(), err, rid))
                con.execute("UPDATE datasets SET status='failed', error_json=? WHERE run_id=? AND status IN ('pending','running')", (err, rid))
                self._event(con, rid, "recovered_as_failed")
        return len(ids)
