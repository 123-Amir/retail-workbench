"""python scripts/chat_core_demo.py - plans go through parse -> validate -> execute. No LLM involved yet."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.chat.executor import answer_plan
from app.retail.data import RetailData
from app.storage.runs import RunService

files = [(p.name, p.read_bytes()) for p in sorted(Path("data/samples").glob("*.csv"))]
CASES = {
    "Good plan: units returned by category": {"intent": "aggregate", "dataset": "orders", "joins": ["products"],
        "filters": [{"field": "status", "op": "eq", "value": "returned"}], "group_by": ["products.category"],
        "metrics": [{"agg": "sum", "field": "quantity", "as": "units"}], "sort": [{"field": "units", "dir": "desc"}], "limit": 3},
    "Model invents a column": {"intent": "aggregate", "dataset": "orders", "metrics": [{"agg": "sum", "field": "revenue", "as": "r"}]},
    "Excessive limit": {"intent": "select", "dataset": "orders", "select": ["order_id"], "limit": 1000000},
    "Undeclared join (fan-out risk)": {"intent": "aggregate", "dataset": "products", "joins": ["orders"], "metrics": [{"agg": "count", "as": "n"}]},
    "Personal data": {"intent": "select", "dataset": "customers", "select": ["email"]},
    "Injection-shaped field name": {"intent": "select", "dataset": "orders", "select": ["sku; DROP TABLE orders"]},
    "Clarify": {"intent": "clarify", "question": "Revenue before or after returns?", "options": ["gross", "net"]},
    "Unanswerable": {"intent": "unanswerable", "reason": "Supplier lead times are not in the data.", "missing": ["suppliers"]},
    "Cell with injected text is only data": {"intent": "select", "dataset": "products", "select": ["product_name"],
        "filters": [{"field": "product_name", "op": "contains", "value": "ignore all previous"}]},
}
with tempfile.TemporaryDirectory() as tmp:
    svc = RunService(root=tmp)
    data = RetailData.from_run(svc, "demo", svc.submit("demo", files, wait=True)["run_id"])
    for title, plan in CASES.items():
        out = answer_plan(plan, data)
        print(f"\n## {title}\n   status: {out['status']}")
        if out["status"] == "error":
            print("   error:", out["error"]["code"], "-", out["error"]["message"])
        else:
            print("   " + out["answer"].replace("\n", "\n   "))
            if out.get("flags", {}).get("suspected_prompt_injection_in_results"):
                print("   flag: suspected prompt injection in result cells (treated as data)")
