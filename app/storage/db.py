"""SQLite persistence. One short-lived connection per operation (thread-safe), WAL mode, foreign keys on."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY,
  owner TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','partial','failed')),
  idempotency_key TEXT NOT NULL,
  options_json TEXT NOT NULL,
  pipeline_version TEXT NOT NULL,
  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
  error_json TEXT
);
-- only one live (non-failed) run per (owner, content+options): failed runs may be retried
CREATE UNIQUE INDEX IF NOT EXISTS ux_runs_idem ON runs(owner, idempotency_key) WHERE status != 'failed';
CREATE INDEX IF NOT EXISTS ix_runs_owner ON runs(owner, created_at);
CREATE TABLE IF NOT EXISTS datasets(
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed')),
  original_filename TEXT NOT NULL,
  sha256 TEXT, size_bytes INTEGER,
  rows_in INTEGER, rows_out INTEGER,
  validation_status TEXT, data_version TEXT,
  original_path TEXT, cleaned_path TEXT, report_path TEXT, cleaned_sha256 TEXT,
  schema_json TEXT, error_json TEXT,
  PRIMARY KEY(run_id, name)
);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  ts TEXT NOT NULL, event TEXT NOT NULL, detail_json TEXT
);
CREATE TABLE IF NOT EXISTS chat_sessions(
  session_id TEXT PRIMARY KEY,
  owner TEXT NOT NULL,
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  state_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_chat_owner ON chat_sessions(owner, created_at);
CREATE TABLE IF NOT EXISTS chat_turns(
  session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  question TEXT NOT NULL,
  status TEXT NOT NULL,
  source TEXT,
  plan_json TEXT, raw_plan_json TEXT, planner_json TEXT, execution_json TEXT, answer_json TEXT,
  PRIMARY KEY(session_id, idx)
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)
            con.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)", (SCHEMA_VERSION,))

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()
