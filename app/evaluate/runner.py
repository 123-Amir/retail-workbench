"""Batch evaluator. Uses the SAME production code as the app (RunService for ingest/clean/persist, ChatService for chat).
Guarantees: one failing case/question/file never aborts the batch; output is deterministic for fixed data and model
responses (no timestamps, run ids, session ids, temp paths or durations are written to results/artifacts)."""
import hashlib
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from app.chat.grounding import check_grounding
from app.chat.service import ChatService
from app.config import settings
from app.errors import PipelineError
from app.evaluate.cases import normalize_case
from app.evaluate.expectations import check_dataset, check_question
from app.llm.base import LLMAuthError, LLMRateLimit, LLMTimeout, LLMUnavailable, get_provider
from app.llm.mock import ScriptedProvider
from app.storage.files import atomic_write_bytes
from app.storage.paths import dataset_name_from_filename, safe_join
from app.storage.runs import PIPELINE_VERSION, RunService

SCHEMA_VERSION = 1
MAX_ROWS_IN_OUTPUT = 100
_ERRORS = {"unavailable": LLMUnavailable, "timeout": LLMTimeout, "auth": LLMAuthError}


def _dumps(o, indent=None) -> str:
    return json.dumps(o, default=str, sort_keys=True, indent=indent, allow_nan=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_dir(case_id: str, used: set) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]", "_", case_id).strip("._") or "case"
    name, n = base[:60], 2
    while name in used:
        name, n = f"{base[:55]}_{n}", n + 1
    used.add(name)
    return name


def _script(items):
    out = []
    for it in items:
        if isinstance(it, dict) and "error" in it:
            kind = it["error"]
            out.append(LLMRateLimit("rate", retry_after=it.get("retry_after")) if kind == "rate_limit" else _ERRORS.get(kind, RuntimeError)(str(kind)))
        else:
            out.append(it)
    return ScriptedProvider(out)


def _read_inputs(case, base_dir: Path, max_mb: float):
    """-> (uploads [(filename, bytes)], pre_failed {dataset_name: error_dict}). Unreadable inputs become per-dataset errors."""
    uploads, pre = [], {}
    for f in case["files"]:
        p = Path(f["path"])
        if not p.is_absolute():
            p = (base_dir / p) if (base_dir / p).exists() else (Path.cwd() / p)
        filename = (f["name"] if f["name"].lower().endswith(".csv") else f["name"] + ".csv") if f["name"] else p.name
        name = dataset_name_from_filename(filename)
        try:
            if not p.is_file():
                raise PipelineError("FILE_NOT_FOUND", "Input file does not exist.", {"file": p.name})
            if p.stat().st_size > max_mb * 1024 * 1024:
                raise PipelineError("FILE_TOO_LARGE", f"File exceeds {max_mb} MB limit.", {"size_bytes": p.stat().st_size})
            uploads.append((filename, p.read_bytes()))
        except PipelineError as e:
            pre[name] = e.to_dict()
        except OSError:
            pre[name] = PipelineError("FILE_UNREADABLE", "Input file could not be read.", {"file": p.name}).to_dict()
    return uploads, pre


def _bounded(result):
    if not result:
        return result
    r = dict(result)
    if len(r["rows"]) > MAX_ROWS_IN_OUTPUT:
        r["rows_omitted_from_output"] = len(r["rows"]) - MAX_ROWS_IN_OUTPUT
        r["rows"] = r["rows"][:MAX_ROWS_IN_OUTPUT]
    return r


class Evaluator:
    def __init__(self, base_dir: Path, artifacts_dir: Path, provider_name: str | None, max_upload_mb: float, store_dir: str | None):
        self.base_dir, self.art = base_dir, artifacts_dir
        self.provider_name = provider_name or settings.llm_provider
        self.max_mb = max_upload_mb
        self.provider_probe = get_provider(self.provider_name)          # fail fast on bad configuration
        self._tmp = None if store_dir else tempfile.TemporaryDirectory(prefix="eval-store-")
        self.runs = RunService(root=store_dir or self._tmp.name, max_upload_mb=max_upload_mb)
        self.used_dirs: set = set()

    def close(self):
        if self._tmp:
            self._tmp.cleanup()

    # ------------------------------------------------------------------ one case
    def run_case(self, raw, idx: int) -> dict:
        cid = str(raw.get("id") or raw.get("case_id") or raw.get("name") or f"case_{idx}")[:80] if isinstance(raw, dict) else f"case_{idx}"
        out = {"case_id": cid, "status": "error", "dataset": {}, "profile_before": {}, "issues": [], "cleaning_plan": [], "profile_after": {},
               "validation": {}, "chat_evaluation": [], "artifacts": {}, "datasets": {}, "checks": [], "error": None}
        try:
            case = normalize_case(raw, idx)
            cdir = _safe_dir(case["id"], self.used_dirs)
            self._execute(case, idx, cdir, out)
        except PipelineError as e:
            out["error"], out["status"] = e.to_dict(), "error"
            exp = (raw.get("expect") or raw.get("expected") or {}) if isinstance(raw, dict) else {}
            if isinstance(exp, dict) and exp.get("case_error_code"):
                ok = e.code == exp["case_error_code"]
                out["checks"].append({"name": "case_error_code", "passed": ok, "expected": exp["case_error_code"], "actual": e.code})
                out["status"] = "passed" if ok else "error"
        except Exception as e:   # noqa: BLE001 - a bug in one case must not take down the batch
            out["error"], out["status"] = {"code": "CASE_FAILED", "message": "Unexpected error while running the case.", "details": {"type": type(e).__name__}}, "error"
        return out

    def _execute(self, case, idx, cdir, out):
        owner = f"eval_{idx}"
        case_dir = self.art / cdir
        shutil.rmtree(case_dir, ignore_errors=True)
        case_dir.mkdir(parents=True, exist_ok=True)
        uploads, pre_failed = _read_inputs(case, self.base_dir, self.max_mb)
        if not uploads and not pre_failed:
            raise PipelineError("INVALID_CASE", "A case needs at least one file.")

        per_ds, run_id, run = {}, None, None
        if uploads:
            run_id = self.runs.submit(owner, uploads, case["decisions"], wait=True)["run_id"]
            run = self.runs.get_run(owner, run_id)
        for name, err in pre_failed.items():
            per_ds[name] = {"dataset": {"name": name, "status": "failed", "error": err}, "profile_before": {}, "issues": [], "cleaning_plan": [],
                            "profile_after": {}, "validation": {}, "schema_after": {}, "artifacts": {}}
        cleaned_frames = {}
        for d in (run["datasets"] if run else []):
            per_ds[d["name"]] = self._dataset_entry(owner, run_id, d, case_dir, cdir, cleaned_frames)
        per_ds = dict(sorted(per_ds.items()))

        out["datasets"] = per_ds
        single = len(per_ds) == 1
        for key in ("dataset", "profile_before", "profile_after", "validation"):
            out[key] = next(iter(per_ds.values()))[key] if single else {n: e[key] for n, e in per_ds.items()}
        out["issues"] = [i for e in per_ds.values() for i in e["issues"]]
        out["cleaning_plan"] = [s for e in per_ds.values() for s in e["cleaning_plan"]]
        out["artifacts"] = {"dir": cdir, "datasets": {n: e["artifacts"] for n, e in per_ds.items()}}

        # dataset expectations
        exp = case["expect"]
        for n, rule in (exp.get("datasets") or {}).items():
            if n not in per_ds:
                out["checks"].append({"name": f"{n}:present", "passed": False, "expected": "dataset in case", "actual": sorted(per_ds)})
                continue
            e = per_ds[n]
            out["checks"] += check_dataset(n, e, e["issues"], e["cleaning_plan"], cleaned_frames.get(n), rule)
        if exp.get("all_datasets_succeed"):
            bad = [n for n, e in per_ds.items() if e["dataset"].get("status") != "succeeded"]
            out["checks"].append({"name": "all_datasets_succeed", "passed": not bad, "expected": True, "actual": bad or True})

        # chat
        if case["questions"]:
            self._chat(case, owner, run_id, run, out, case_dir, cdir)
        out["status"] = "failed" if any(not c["passed"] for c in out["checks"]) else "passed"

    def _dataset_entry(self, owner, run_id, d, case_dir: Path, cdir: str, cleaned_frames: dict) -> dict:
        name = d["name"]
        entry = {"dataset": {"name": name, "status": d["status"], "error": d["error"]}, "profile_before": {}, "issues": [], "cleaning_plan": [],
                 "profile_after": {}, "validation": {}, "schema_after": {}, "artifacts": {}}
        files = {}
        if d["original_path"]:
            src = safe_join(self.runs.root, owner, d["original_path"])
            dst = case_dir / f"{name}.original.csv"
            shutil.copyfile(src, dst)
            files["original"], files["original_sha256"] = f"{cdir}/{dst.name}", _sha(dst)
        if d["report_path"]:
            rep = self.runs.get_report(owner, run_id, name)
            ds = dict(rep["dataset"])
            ds.update(status=d["status"], error=d["error"], validation_status=rep["validation"]["status"])
            entry.update(dataset=ds, profile_before=rep["profile_before"], issues=rep["issues"], cleaning_plan=rep["cleaning_plan"],
                         profile_after=rep["profile_after"], validation=rep["validation"], schema_after=rep["schema_after"])
        if d["cleaned_path"]:
            src = safe_join(self.runs.root, owner, d["cleaned_path"])
            dst = case_dir / f"{name}.cleaned.csv"
            shutil.copyfile(src, dst)
            files["cleaned"], files["cleaned_sha256"] = f"{cdir}/{dst.name}", _sha(dst)
            files["data_version"] = d["data_version"]
            cleaned_frames[name] = self.runs.read_cleaned(owner, run_id, name)
        if d["report_path"]:
            rep_out = {k: v for k, v in rep.items() if k != "run"}                # drop run id / paths: not portable
            rep_out["artifacts"] = files
            dst = case_dir / f"{name}.report.json"
            atomic_write_bytes(dst, _dumps(rep_out, indent=2).encode())
            files["report"] = f"{cdir}/{dst.name}"
        entry["artifacts"] = files
        return entry

    # ------------------------------------------------------------------ chat
    def _chat(self, case, owner, run_id, run, out, case_dir: Path, cdir: str):
        ready = bool(run and any(d["status"] == "succeeded" for d in run["datasets"]))
        provider = _script(case["model_script"]) if case["model_script"] is not None else get_provider(self.provider_name)
        real = provider.name not in ("mock", "scripted")
        chat = ChatService(self.runs, provider, sleep=time.sleep if real else (lambda s: None)) if ready else None
        sid = chat.create_session(owner, run_id)["session_id"] if ready else None
        for i, q in enumerate(case["questions"], 1):
            text = q["question"] if q["question"] is not None else "(structured plan)"
            item = {"index": i, "question": text, "plan": None, "raw_plan": None, "answer": {}, "result": None, "evidence": None, "flags": None,
                    "trace": {}, "status": "error", "error": None, "grounded": None, "checks": []}
            try:
                if not ready:
                    raise PipelineError("RUN_NOT_READY", "No dataset was cleaned successfully, so there is nothing to chat with.")
                t = chat.ask(owner, sid, question=q["question"], plan=q["plan"])
                item.update(plan=t["plan"], raw_plan=t["raw_plan"], answer=t["answer"], result=_bounded(t["result"]), evidence=t["evidence"],
                            flags=t["flags"], trace=t["trace"], status=t["status"], error=t["error"])
                if t["status"] == "ok":
                    item["grounded"] = check_grounding(t["answer"]["text"], t["result"], t["evidence"], text)["ok"]
            except PipelineError as e:
                item.update(error=e.to_dict(), answer={"text": e.message, "source": "system"})
            except Exception as e:   # noqa: BLE001
                item.update(error={"code": "QUESTION_FAILED", "message": "Unexpected error.", "details": {"type": type(e).__name__}}, answer={"text": "Unexpected error.", "source": "system"})
            item["checks"] = check_question(item, q["expect"])
            out["checks"] += [{**c, "name": f"q{i}:{c['name']}"} for c in item["checks"]]
            out["chat_evaluation"].append(item)
        if ready:
            s = chat.get_session(owner, sid)
            for k in ("session_id", "run_id", "created_at"):
                s.pop(k, None)
            for t in s["turns"]:
                t.pop("asked_at", None)
            atomic_write_bytes(case_dir / "chat_trace.json", _dumps(s, indent=2).encode())
            out["artifacts"]["chat_trace"] = f"{cdir}/chat_trace.json"


# ------------------------------------------------------------------ batch
def summarize(results: list, provider: str) -> dict:
    ds = [e["dataset"] for r in results for e in r["datasets"].values()]
    qs = [q for r in results for q in r["chat_evaluation"]]
    checks = [c for r in results for c in r["checks"]]
    count = lambda xs, k, v: sum(1 for x in xs if x.get(k) == v)  # noqa: E731
    return {"cases_total": len(results), "cases_passed": count(results, "status", "passed"), "cases_failed": count(results, "status", "failed"),
            "cases_error": count(results, "status", "error"), "datasets_total": len(ds), "datasets_succeeded": count(ds, "status", "succeeded"),
            "datasets_failed": count(ds, "status", "failed"), "questions_total": len(qs),
            "questions_by_status": {s: count(qs, "status", s) for s in sorted({q["status"] for q in qs})},
            "checks_total": len(checks), "checks_passed": sum(1 for c in checks if c["passed"]), "checks_failed": sum(1 for c in checks if not c["passed"]),
            "provider": provider, "pipeline_version": PIPELINE_VERSION}


def evaluate_file(input_path: str, output_path: str, artifacts_dir: str, provider: str | None = None, max_upload_mb: float | None = None,
                  store_dir: str | None = None) -> tuple:
    """Returns (results_document, exit_code). exit_code 2 = fatal input/config problem (output still written)."""
    out_path, art = Path(output_path), Path(artifacts_dir)

    def finish(doc, code):
        atomic_write_bytes(out_path, _dumps(doc, indent=2).encode())
        return doc, code

    def fatal(err: PipelineError):
        return finish({"schema_version": SCHEMA_VERSION, "fatal_error": err.to_dict(), "summary": None, "results": []}, 2)

    try:
        src = Path(input_path)
        doc = json.loads(src.read_text("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        return fatal(PipelineError("INPUT_UNREADABLE", "Cases file could not be read as JSON.", {"detail": type(e).__name__}))
    cases = doc.get("cases") if isinstance(doc, dict) else doc
    if not isinstance(cases, list):
        return fatal(PipelineError("INPUT_INVALID", "Cases file must be a list of cases or an object with a 'cases' list."))
    art.mkdir(parents=True, exist_ok=True)
    try:
        ev = Evaluator(src.resolve().parent, art, provider, max_upload_mb if max_upload_mb is not None else settings.max_upload_mb, store_dir)
    except PipelineError as e:
        return fatal(e)
    try:
        results, seen = [], set()
        for i, raw in enumerate(cases, 1):
            r = ev.run_case(raw, i)
            if r["case_id"] in seen:   # duplicate ids would collide in artifacts: report, do not overwrite
                r.update(status="error", error={"code": "DUPLICATE_CASE_ID", "message": "Case ids must be unique.", "details": {"id": r["case_id"]}})
            seen.add(r["case_id"])
            results.append(r)
    finally:
        ev.close()
    return finish({"schema_version": SCHEMA_VERSION, "summary": summarize(results, ev.provider_name), "results": results}, 0)
