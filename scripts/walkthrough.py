"""Guided demo for the 3-5 minute walkthrough.  python scripts/walkthrough.py [1-5]   (no argument = all five beats)
Beats: 1 cleaning run | 2 ambiguous decision | 3 retail-domain feature | 4 multi-turn chat with evidence | 5 failure + recovery.
Uses the keyless mock provider and a temporary store, so it is safe to run anywhere."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from app.chat.executor import answer_plan
from app.chat.service import ChatService
from app.errors import PipelineError
from app.llm.base import LLMUnavailable
from app.llm.mock import MockProvider, ScriptedProvider
from app.pipeline import cleaner
from app.pipeline.cleaner import clean_dataframe
from app.retail.data import RetailData
from app.retail.joins import join_declared
from app.retail.inventory import inventory_metrics
from app.retail.metrics import sales_metrics
from app.storage.runs import RunService

SAMPLES = Path("data/samples")
FILES = [(p.name, p.read_bytes()) for p in sorted(SAMPLES.glob("*.csv"))]
OWNER = "demo"


def head(n, title):
    print(f"\n{'=' * 78}\nBEAT {n}: {title}\n{'=' * 78}")


def world(tmp):
    runs = RunService(root=tmp)
    run_id = runs.submit(OWNER, FILES, wait=True)["run_id"]
    return runs, run_id, RetailData.from_run(runs, OWNER, run_id)


def beat1(runs, run_id, data):
    head(1, "One cleaning run: profile -> issues -> plan -> deterministic clean -> validate")
    rep = runs.get_report(OWNER, run_id, "orders")
    d = rep["dataset"]
    print(f"orders.csv  {d['rows_in']} rows -> {d['rows_out']} rows   data_version={d['data_version']}")
    kinds = {}
    for i in rep["issues"]:
        kinds.setdefault(i["type"], []).append(i["columns"][0] if i["columns"] else "*")
    print("issues found:", "; ".join(f"{k}({','.join(v)})" for k, v in sorted(kinds.items())))
    print("plan (id, action, field, risk, status -> outcome):")
    for s in rep["cleaning_plan"]:
        e = s["execution"]
        eff = f"changed={e.get('cells_changed', 0)} nulled={e.get('cells_nulled', 0)} removed={e.get('rows_removed', 0)}" if s["final_status"] == "applied" else ""
        print(f"  {s['id']}  {s['action']:<24}{','.join(s['fields']) or '-':<12}{s['risk']:<7}{s['status']:<13}-> {s['final_status']:<8}{eff}")
    v = rep["validation"]
    print(f"validation: {v['status']}  | " + ", ".join(f"{c['name']}={c['status']}" for c in v["checks"]))
    print(f"row loss explained by the duplicate step: {v['row_delta']}")
    print(f"idempotent (re-clean of own output changes nothing): {v['idempotency']['stable']}")


def beat2(runs, run_id, data):
    head(2, "An ambiguous decision: 03/04/2025 - is it 3 April or 4 March?")
    rep = runs.get_report(OWNER, run_id, "orders")
    issue = next(i for i in rep["issues"] if i["type"] == "AMBIGUOUS_DATES")
    print(f"{issue['affected_rows']} order dates are ambiguous. Evidence from the UNambiguous dates, per separator:")
    for sep, e in issue["evidence"]["by_separator"].items():
        print(f"  '{sep}': day-first={e['dmy']}  month-first={e['mdy']}  ambiguous={e['ambiguous']}  -> inferred={e['inferred_order']} (confidence {e['confidence']})")
    step = next(s for s in rep["cleaning_plan"] if s["action"] == "parse_dates")
    print(f"plan step {step['id']}: risk={step['risk']} status={step['status']} (a human may override via decisions); params={step['params']['separator_order']}")
    for ex in step["execution"]["examples"][:3]:
        print(f"  row {ex['row']}: {ex['before']!r} -> {ex['after']!r}")
    print("\nNo evidence? Then we refuse to guess (dates below have no day>12 anywhere):")
    df = pd.DataFrame({"id": list("abcde"), "d": ["01/02/2025", "03/04/2025", "2025-01-01", "2025-01-02", "2025-01-03"]})
    res = clean_dataframe(df, "demo")
    print(res.df.to_string(index=False))
    print("values nulled by reason:", res.report["validation"]["values_nulled_by_reason"], "| validation:", res.report["validation"]["status"])


def beat3(runs, run_id, data):
    head(3, "Retail domain: safe joins + auditable analytics (nothing is invented)")
    j = join_declared(data.frames, "orders", "products").report
    print(f"join {j['join']}: rows {j['left_rows']}->{j['rows_after']} (fan-out free={j['fan_out_free']}), match rate {j['match_rate']}, orphans {j['unmatched_rows']} e.g. {j['unmatched_key_examples']}")
    dup = data.frames["products"].copy()
    dup = pd.concat([dup, dup.head(1)], ignore_index=True)
    try:
        join_declared({**data.frames, "products": dup}, "orders", "products")
    except PipelineError as e:
        print(f"duplicate SKU in products -> join REFUSED: {e.code} {e.details['examples']}")
    r = sales_metrics(data, group_by=["category"], metrics=["net_revenue", "orders", "aov", "return_rate_units"], limit=6)
    print("\nnet revenue by category (orphan SKU lines are kept visible as '(unknown)', not dropped):")
    for g in r["groups"]:
        print(f"  {g['group']['category']:<10} {g['metrics']}")
    print("formula:", r["evidence"]["formulas"]["net_revenue"], "| exclusions:", r["evidence"]["exclusions"])
    for m in r["evidence"]["missing_evidence"][:3]:
        print("  missing evidence:", m)
    inv = inventory_metrics(data, group_by=["region"], metrics=["stock_out_rate", "sell_through"])
    print("\ninventory (as_of", inv["evidence"]["as_of"] + ", from the data, not today's date):", [(g["group"]["region"], g["metrics"]["stock_out_rate"]) for g in inv["groups"]])
    print("  caveat:", inv["evidence"]["missing_evidence"][0][:110] + "...")


def beat4(runs, run_id, data):
    head(4, "Multi-turn chat: plan -> validate -> execute -> evidence (state carried by code)")
    chat = ChatService(runs, MockProvider())
    sid = chat.create_session(OWNER, run_id)["session_id"]
    for q in ["What is the net revenue by region?", "only stores in the West region", "among those, which categories had the highest return rate?"]:
        t = chat.ask(OWNER, sid, q)
        ev = t["evidence"]
        print(f"\nQ{t['turn']}: {q}\n  plan filters={t['plan']['filters']} group_by={t['plan']['group_by']} metrics={t['plan']['metrics']}")
        print("  answer:", t["answer"]["text"].splitlines()[0][:100])
        print(f"  evidence: datasets={ev['datasets']}  filters(rows_after)={[f['rows_after'] for f in ev['filters']]}  joins={[j['join'] for j in ev['joins']]}  rows_considered={ev['rows_considered']}/{ev['rows_total']}")
    print("\nreplayable later:", [(x["turn"], x["status"], x["planner"]["source"]) for x in chat.get_session(OWNER, sid)["turns"]])
    for q in ["which store is best?", "forecast revenue for next month"]:
        t = chat.ask(OWNER, sid, q)
        print(f"\n'{q}' -> {t['status']}: {t['answer']['text'][:100]}")


def beat5(runs, run_id, data):
    head(5, "Failure and recovery: nothing fails silently, nothing corrupts data")
    print("a) LLM outage -> exponential backoff, then deterministic fallback:")
    slept = []
    chat = ChatService(runs, ScriptedProvider([LLMUnavailable("down")] * 6), sleep=slept.append)
    t = chat.ask(OWNER, chat.create_session(OWNER, run_id)["session_id"], "net revenue by region")
    print(f"   status={t['status']} planner={t['trace']['planner']['source']} backoff={slept} note={t['trace']['planner']['notes'][0]!r}")
    print("b) model invents a number -> explanation discarded, computed answer shown:")
    chat = ChatService(runs, ScriptedProvider(["Orders were 1,234,567, up 99% year on year."]))
    t = chat.ask(OWNER, chat.create_session(OWNER, run_id)["session_id"], plan={"intent": "retail_metric", "domain": "sales", "metrics": ["orders"]})
    print(f"   answer.source={t['answer']['source']} ungrounded={t['answer']['grounding']['ungrounded_numbers']} -> shown: {t['answer']['text'].splitlines()[2]}")
    print("c) hostile / hallucinated plans never reach data:")
    for name, p in {"invented column": {"intent": "aggregate", "dataset": "orders", "metrics": [{"agg": "sum", "field": "revenue", "as": "r"}]},
                    "PII": {"intent": "select", "dataset": "customers", "select": ["email"]},
                    "limit bomb": {"intent": "select", "dataset": "orders", "select": ["sku"], "limit": 10**9},
                    "fan-out join": {"intent": "aggregate", "dataset": "products", "joins": ["orders"], "metrics": [{"agg": "count", "as": "n"}]}}.items():
        print(f"   {name:<16} -> {answer_plan(p, data)['error']['code']}")
    print("d) a cleaning step that silently loses data is caught by the independent validator and publishes NOTHING:")
    real = cleaner.EXEC["trim_whitespace"]

    def sneaky(df, step, fields):
        out, log = real(df, step, fields)
        out.loc[out[fields[0]].first_valid_index(), fields[0]] = None
        return out, log
    cleaner.EXEC["trim_whitespace"] = sneaky
    try:
        with tempfile.TemporaryDirectory() as t2:
            r2 = RunService(root=t2)
            rid = r2.submit("x", [FILES[3]], wait=True)["run_id"]            # orders.csv
            run = r2.get_run("x", rid)
            d = run["datasets"][0]
            print(f"   run={run['status']}  dataset error={d['error']['code']} failed_checks={d['error']['details']['failed_checks']}  cleaned artifact published={d['cleaned_path'] is not None}  original kept={d['original_path'] is not None}")
    finally:
        cleaner.EXEC["trim_whitespace"] = real


BEATS = {1: beat1, 2: beat2, 3: beat3, 4: beat4, 5: beat5}

if __name__ == "__main__":
    pick = [int(a) for a in sys.argv[1:]] or sorted(BEATS)
    with tempfile.TemporaryDirectory() as tmp:
        w = world(tmp)
        for n in pick:
            BEATS[n](*w)
