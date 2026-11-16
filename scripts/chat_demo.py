"""python scripts/chat_demo.py - multi-turn chat with evidence, then an LLM outage and an invented-number explanation.
Uses the keyless mock provider (LLM_PROVIDER=mock). Set LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY to use a real model."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.chat.service import ChatService
from app.llm.base import LLMUnavailable, get_provider
from app.llm.mock import ScriptedProvider
from app.storage.runs import RunService

files = [(p.name, p.read_bytes()) for p in sorted(Path("data/samples").glob("*.csv"))]


def show(t):
    pl = t["trace"]["planner"]
    print(f"\nQ{t['turn']}: {t['question']}\n   status={t['status']}  planner={pl['source']}  answer_source={t['answer']['source']}")
    for line in t["answer"]["text"].splitlines()[:6]:
        print("   " + line)
    g = t["answer"]["grounding"]
    if g and g.get("action"):
        print("   grounding:", g["action"], "->", g["ungrounded_numbers"])
    if t["evidence"]:
        print("   filters:", [(f['field'], f['value'], f['rows_after']) for f in t["evidence"]["filters"]], "| ops:", t["evidence"]["operations"])
    for n in pl["notes"]:
        print("   note:", n)


with tempfile.TemporaryDirectory() as tmp:
    runs = RunService(root=tmp)
    run = runs.submit("demo", files, wait=True)["run_id"]

    print("=== Multi-turn (provider: %s) ===" % get_provider().name)
    chat = ChatService(runs, get_provider())
    sid = chat.create_session("demo", run)["session_id"]
    for q in ["What is the net revenue by region?", "only stores in the West region", "among those, which categories had the highest return rate?",
              "which store is best?", "forecast revenue for next month"]:
        show(chat.ask("demo", sid, q))

    print("\n=== LLM outage: backoff, then deterministic fallback ===")
    sleeps = []
    down = ChatService(runs, ScriptedProvider([LLMUnavailable("down")] * 9), sleep=sleeps.append)
    show(down.ask("demo", down.create_session("demo", run)["session_id"], "stock-out rate by region"))
    print("   backoff delays slept:", sleeps)

    print("\n=== Model writes an invented number: explanation is discarded ===")
    liar = ChatService(runs, ScriptedProvider(["Orders were 1,234,567 in total, up 99% year on year."]))
    show(liar.ask("demo", liar.create_session("demo", run)["session_id"], plan={"intent": "retail_metric", "domain": "sales", "metrics": ["orders"]}))
