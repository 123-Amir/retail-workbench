"""python scripts/retail_demo.py - sales + inventory analytics on the sample data, with evidence."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.retail.data import RetailData
from app.retail.inventory import inventory_metrics
from app.retail.metrics import sales_metrics
from app.storage.runs import RunService

files = [(p.name, p.read_bytes()) for p in sorted(Path("data/samples").glob("*.csv"))]
with tempfile.TemporaryDirectory() as tmp:
    svc = RunService(root=tmp)
    data = RetailData.from_run(svc, "demo", svc.submit("demo", files, wait=True)["run_id"])
    r = sales_metrics(data, group_by=["category"], metrics=["net_revenue", "return_rate_units"],
                      filters=[{"field": "region", "op": "eq", "value": "West"}], sort_by="return_rate_units")
    print("West stores - return rate by category")
    for g in r["groups"]:
        print(f"  {g['group']['category']:<10} {g['metrics']}")
    print("  filters:", r["evidence"]["filters"])
    print("  versions:", r["evidence"]["datasets"])
    for m in r["evidence"]["missing_evidence"]:
        print("  missing evidence:", m)
    i = inventory_metrics(data, group_by=["region"], metrics=["stock_out_rate", "sell_through"])
    print("\nInventory by region (as_of", i["evidence"]["as_of"] + ")")
    for g in i["groups"]:
        print(f"  {g['group']['region']:<6} {g['metrics']}")
