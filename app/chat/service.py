"""Chat sessions: question -> plan -> execution -> answer, persisted so a reviewer can replay every turn.

Per turn we store: the question, the model's raw plan (only if it parsed as JSON), the final validated plan, planner
attempts/notes, execution trace + evidence + a bounded result preview, and the answer. We never store system prompts,
catalogs sent to the model, API keys, or unparseable model output (which could echo a hidden prompt).
Sessions are pinned to ONE run, so every answer cites the same data versions (raw/cleaned ambiguity cannot arise)."""
import hashlib
import json
import re
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone

from app.chat.catalog import build_catalog
from app.chat.executor import answer_plan
from app.chat.grounding import check_grounding
from app.chat.planner import Explainer, Planner
from app.errors import PipelineError
from app.retail.data import RetailData
from app.storage.paths import validate_owner
from app.storage.runs import RunService

MAX_QUESTION_CHARS, PERSISTED_PREVIEW_ROWS = 1000, 50
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dumps(o):
    return json.dumps(o, default=str, allow_nan=False)


class _LRU(OrderedDict):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def put(self, k, v):
        self[k] = v
        self.move_to_end(k)
        while len(self) > self.size:
            self.popitem(last=False)


class ChatService:
    def __init__(self, runs: RunService, provider=None, *, explain: bool = True, max_llm_calls: int = 3, sleep=time.sleep, breaker=None):
        self.runs, self.db = runs, runs.db
        self.planner = Planner(provider, max_llm_calls=max_llm_calls, sleep=sleep, breaker=breaker)
        self.explainer = Explainer(provider, self.planner.breaker, sleep) if (provider is not None and explain) else None
        self._data, self._answers, self._locks = _LRU(4), _LRU(128), defaultdict(threading.Lock)

    # ------------------------------------------------------------------ sessions
    def create_session(self, owner: str, run_id: str) -> dict:
        validate_owner(owner)
        run = self.runs.get_run(owner, run_id)                       # owner-scoped: someone else's run == missing
        ready = {d["name"]: d["data_version"] for d in run["datasets"] if d["status"] == "succeeded"}
        if not ready:
            raise PipelineError("RUN_NOT_READY", "This run has no successfully cleaned dataset to chat with.", {"status": run["status"]})
        sid = f"chat_{uuid.uuid4().hex[:12]}"
        with self.db.connect() as con:
            con.execute("INSERT INTO chat_sessions(session_id,owner,run_id,created_at,state_json) VALUES(?,?,?,?,?)",
                        (sid, owner, run_id, _now(), _dumps({"datasets": ready})))
        return {"session_id": sid, "run_id": run_id, "datasets": ready}

    def _session(self, con, owner, sid):
        r = con.execute("SELECT * FROM chat_sessions WHERE session_id=? AND owner=?", (sid, owner)).fetchone()
        if not r:
            raise PipelineError("SESSION_NOT_FOUND", "Chat session not found.", {"session_id": sid})
        return r

    def _load(self, owner, run_id):
        key = (owner, run_id)
        if key not in self._data:
            data = RetailData.from_run(self.runs, owner, run_id)       # verifies artifact hashes on read
            self._data.put(key, (data, build_catalog(data)))
        return self._data[key]

    # ------------------------------------------------------------------ one turn
    def ask(self, owner: str, session_id: str, question: str | None = None, plan: dict | None = None) -> dict:
        validate_owner(owner)
        if (question is None) == (plan is None):
            raise PipelineError("INVALID_REQUEST", "Provide exactly one of 'question' or 'plan'.")
        if question is not None:
            if not isinstance(question, str):
                raise PipelineError("INVALID_QUESTION", "Question must be text.")
            question = " ".join(_CTRL.sub(" ", question).split())
            if not question or len(question) > MAX_QUESTION_CHARS:
                raise PipelineError("INVALID_QUESTION", f"Question must be 1-{MAX_QUESTION_CHARS} characters.")
        with self._locks[session_id]:
            with self.db.connect() as con:
                s = self._session(con, owner, session_id)
                idx = con.execute("SELECT COALESCE(MAX(idx),0)+1 FROM chat_turns WHERE session_id=?", (session_id,)).fetchone()[0]
            state = json.loads(s["state_json"])
            data, catalog = self._load(owner, s["run_id"])
            shown_q = question if question is not None else "(structured plan)"

            if plan is not None:                                        # deterministic path: no model involved
                from app.chat.planner import apply_inheritance
                if not isinstance(plan, dict):
                    raise PipelineError("INVALID_PLAN", "Structured plan must be a JSON object.")
                try:
                    final, notes = apply_inheritance(plan, state.get("last_plan"))
                except (AttributeError, KeyError, TypeError, ValueError) as e:
                    raise PipelineError("INVALID_PLAN", "Structured plan inheritance is invalid.",
                                         {"reason": type(e).__name__}) from e
                po = {"plan": final, "raw_plan": plan if isinstance(plan, dict) else None, "source": "direct", "attempts": [], "notes": notes, "error": None}
            else:
                o = self.planner.plan(question, data, catalog, state)
                po = {"plan": o.plan, "raw_plan": o.raw_plan, "source": o.source, "attempts": o.attempts, "notes": o.notes, "error": o.error}

            cached = False
            if po["plan"] is None:
                out = {"status": "error", "trace": {"stages_completed": []}, "error": po["error"] or {"code": "PLANNING_FAILED", "message": "No valid plan."}}
            else:
                key = hashlib.sha256(_dumps({"o": owner, "r": s["run_id"], "p": po["plan"]}).encode()).hexdigest()
                if key in self._answers:
                    out, cached = json.loads(_dumps(self._answers[key])), True
                else:
                    out = answer_plan(po["plan"], data)
                    if out["status"] == "ok":
                        self._answers.put(key, out)

            llm_down = any(a["outcome"] in ("llm_error", "circuit_open") for a in po["attempts"])    # don't hammer a model that just failed
            answer = self._compose_answer(shown_q if question else "", out, allow_explain=not llm_down)
            turn = {"session_id": session_id, "turn": idx, "question": shown_q, "status": out["status"], "answer": answer,
                    "plan": out.get("plan"), "raw_plan": po["raw_plan"], "result": out.get("result"), "evidence": out.get("evidence"),
                    "flags": out.get("flags"), "error": out.get("error"),
                    "trace": {"planner": {"source": po["source"], "attempts": po["attempts"], "notes": po["notes"]},
                              "stages_completed": out["trace"]["stages_completed"], "cached": cached}}
            self._persist(session_id, idx, state, turn, out)
        return turn

    def _compose_answer(self, question: str, out: dict, allow_explain: bool = True) -> dict:
        if out["status"] != "ok":
            return {"text": out.get("answer") or (out.get("error") or {}).get("message", ""), "source": "system", "details": None, "grounding": None}
        template = out["answer"]
        ground = check_grounding(template, out["result"], out["evidence"], question)
        if self.explainer is not None and allow_explain:
            text, g = self.explainer.explain(question, out)
            if text:
                return {"text": text, "source": "llm_explanation", "details": template, "grounding": g}
            if g is not None:
                return {"text": template, "source": "template", "details": None,
                        "grounding": {**g, "action": "model explanation discarded: it contained numbers not present in the executed result"}}
        return {"text": template, "source": "template", "details": None, "grounding": ground}

    def _persist(self, sid, idx, state, turn, out):
        if out["status"] == "ok" and out["plan"]["intent"] in ("retail_metric", "aggregate", "select", "distinct", "describe"):
            state = {**state, "last_plan": out["plan"], "last_question": turn["question"]}
        preview = None
        if out.get("result"):
            preview = {**out["result"], "rows": out["result"]["rows"][:PERSISTED_PREVIEW_ROWS]}
        with self.db.connect() as con:
            con.execute("INSERT INTO chat_turns(session_id,idx,created_at,question,status,source,plan_json,raw_plan_json,planner_json,execution_json,answer_json)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (sid, idx, _now(), turn["question"], turn["status"], turn["trace"]["planner"]["source"], _dumps(turn["plan"]),
                         _dumps(turn["raw_plan"]), _dumps(turn["trace"]["planner"]),
                         _dumps({"stages_completed": turn["trace"]["stages_completed"], "cached": turn["trace"]["cached"], "evidence": turn["evidence"],
                                 "result_preview": preview, "flags": turn["flags"], "error": turn["error"]}),
                         _dumps(turn["answer"])))
            con.execute("UPDATE chat_sessions SET state_json=? WHERE session_id=?", (_dumps(state), sid))

    # ------------------------------------------------------------------ inspection
    def get_session(self, owner: str, session_id: str) -> dict:
        validate_owner(owner)
        with self.db.connect() as con:
            s = self._session(con, owner, session_id)
            rows = con.execute("SELECT * FROM chat_turns WHERE session_id=? ORDER BY idx", (session_id,)).fetchall()
        j = lambda x: json.loads(x) if x else None   # noqa: E731
        return {"session_id": session_id, "run_id": s["run_id"], "created_at": s["created_at"], "datasets": json.loads(s["state_json"]).get("datasets"),
                "turns": [{"turn": r["idx"], "asked_at": r["created_at"], "question": r["question"], "status": r["status"], "planner": j(r["planner_json"]),
                           "raw_plan": j(r["raw_plan_json"]), "plan": j(r["plan_json"]), "execution": j(r["execution_json"]), "answer": j(r["answer_json"])}
                          for r in rows]}

    def list_sessions(self, owner: str) -> list:
        validate_owner(owner)
        with self.db.connect() as con:
            rs = con.execute("SELECT s.session_id, s.run_id, s.created_at, (SELECT COUNT(*) FROM chat_turns t WHERE t.session_id=s.session_id) AS turns "
                             "FROM chat_sessions s WHERE s.owner=? ORDER BY s.created_at DESC, s.session_id", (owner,)).fetchall()
        return [dict(r) for r in rs]
