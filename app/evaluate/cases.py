"""Case-file parsing. Lenient about key spellings (reviewers may write their own cases), strict about types.
A malformed case becomes a structured error for THAT case only; it never aborts the batch."""
from app.errors import PipelineError

_FILES = ("files", "datasets", "inputs", "paths")
_QUESTIONS = ("questions", "chat", "chat_questions")


def _bad(msg, **d):
    return PipelineError("INVALID_CASE", msg, d)


def _first(raw, keys):
    for k in keys:
        if k in raw:
            return raw[k]
    return None


def normalize_case(raw, idx: int) -> dict:
    if not isinstance(raw, dict):
        raise _bad("A case must be a JSON object.")
    cid = str(raw.get("id") or raw.get("case_id") or raw.get("name") or f"case_{idx}")[:80]
    files_raw = _first(raw, _FILES)
    if files_raw is None:
        single = raw.get("file") or raw.get("path")
        files_raw = [single] if single else []
    if not isinstance(files_raw, list):
        raise _bad("'files' must be a list.")
    files = []
    for f in files_raw:
        if isinstance(f, str):
            files.append({"path": f, "name": None})
        elif isinstance(f, dict) and isinstance(f.get("path") or f.get("file") or f.get("csv"), str):
            name = f.get("name") or f.get("dataset")
            if name is not None and not isinstance(name, str):
                raise _bad("File 'name' must be text.")
            files.append({"path": f.get("path") or f.get("file") or f.get("csv"), "name": name})
        else:
            raise _bad("Each file must be a path string or {path, name}.")
    qs_raw = _first(raw, _QUESTIONS) or []
    if not isinstance(qs_raw, list):
        raise _bad("'questions' must be a list.")
    questions = []
    for q in qs_raw:
        if isinstance(q, str):
            questions.append({"question": q, "plan": None, "expect": {}})
            continue
        if not isinstance(q, dict):
            raise _bad("Each question must be a string or an object.")
        text = q.get("question") or q.get("q") or q.get("text")
        plan = q.get("plan")
        if (text is None) == (plan is None) or (text is not None and not isinstance(text, str)) or (plan is not None and not isinstance(plan, dict)):
            raise _bad("A question needs exactly one of 'question' (text) or 'plan' (object).")
        exp = q.get("expect") or q.get("expected") or {}
        if not isinstance(exp, dict):
            raise _bad("'expect' must be an object.")
        questions.append({"question": text, "plan": plan, "expect": exp})
    decisions = raw.get("decisions") or {}
    if not isinstance(decisions, dict):
        raise _bad("'decisions' must be an object {step_id: 'apply'|'skip'}.")
    script = raw.get("model_script")
    if script is not None and not isinstance(script, list):
        raise _bad("'model_script' must be a list.")
    exp = raw.get("expect") or raw.get("expected") or {}
    if not isinstance(exp, dict):
        raise _bad("'expect' must be an object.")
    return {"id": cid, "files": files, "questions": questions, "decisions": decisions, "model_script": script, "expect": exp}
