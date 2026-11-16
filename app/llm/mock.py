"""Providers for keyless operation and tests.
MockProvider    : deterministic. 'plan' purpose reads the same JSON prompt the real model gets and answers with rule_plan.
                  'explain' purpose writes a one-sentence summary built only from the result it was given.
ScriptedProvider: plays back canned responses/exceptions in order (failure-path tests)."""
import json

from app.chat.rules import rule_plan


class MockProvider:
    name, model = "mock", "mock-rules-1"

    def complete(self, *, purpose: str, system: str, user: str, max_tokens: int = 1024, timeout: float = 20.0) -> str:
        payload = json.loads(user)
        if purpose == "plan":
            return json.dumps(rule_plan(payload["question"], payload["catalog"], payload.get("previous_plan")))
        rows, cols = payload.get("rows", []), payload.get("columns", [])
        if not rows:
            return "No rows matched, so there is nothing to summarise."
        r = rows[0]
        keys = [k for k in r if isinstance(r[k], (int, float)) and not isinstance(r[k], bool)]
        label = ", ".join(str(r[k]) for k in r if k not in keys and r[k] is not None) or "Overall"
        facts = ", ".join(f"{k} = {r[k]}" for k in keys[:3])
        more = f" ({payload['total_rows']} result rows in total.)" if payload.get("total_rows", 0) > 1 else ""
        return f"{label}: {facts}.{more}" if facts else f"{label}.{more}"


class ScriptedProvider:
    name, model = "scripted", "scripted-1"

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def complete(self, *, purpose: str, system: str, user: str, max_tokens: int = 1024, timeout: float = 20.0) -> str:
        self.calls.append({"purpose": purpose, "system": system, "user": user})
        item = self.script.pop(0) if self.script else RuntimeError("script exhausted")
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, str) else json.dumps(item)
