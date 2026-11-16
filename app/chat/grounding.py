"""Numeric grounding: every number in a model-written explanation must already exist in the executed
result/evidence (or in the user's question). If not, the explanation is discarded and the deterministic
template is used instead. A persuasive sentence without computational evidence is a wrong answer."""
import re

_VERSION = re.compile(r"\b[\w.]+@[0-9a-f]{6,}\b")
_NUM = re.compile(r"(?<![\w.@])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")   # a comma belongs to the number only as a thousands separator


def extract_numbers(text: str) -> list:
    return _NUM.findall(_VERSION.sub(" ", text or ""))


def _walk(obj, out: set):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for t in extract_numbers(obj):
            try:
                out.add(float(t.rstrip("%").replace(",", "")))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, out)


def check_grounding(text: str, result: dict, evidence: dict, question: str = "") -> dict:
    allowed: set = set()
    for src in (result, evidence, question):
        _walk(src, allowed)
    bad, tokens = [], extract_numbers(text)
    for tok in tokens:
        pct = tok.endswith("%")
        s = tok.rstrip("%").replace(",", "")
        v = float(s)
        d = len(s.split(".")[1]) if "." in s else 0
        if not any(abs(round(a, d) - round(v, d)) < 1e-9 or (pct and abs(round(a * 100, d) - round(v, d)) < 1e-9) for a in allowed):
            bad.append(tok)
    return {"ok": not bad, "ungrounded_numbers": bad[:10], "numbers_checked": len(tokens)}
