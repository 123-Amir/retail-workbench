"""python scripts/run_demo.py - submit all sample CSVs as a run, then resubmit to show idempotent reuse."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.storage.runs import RunService

files = [(p.name, p.read_bytes()) for p in sorted(Path("data/samples").glob("*.csv"))]
with tempfile.TemporaryDirectory() as tmp:
    svc = RunService(root=tmp)
    first = svc.submit("demo", files, wait=True)
    run = svc.get_run("demo", first["run_id"])
    print("run:", run["run_id"], run["status"], "reused:", first["reused"])
    for d in run["datasets"]:
        print(f"  {d['name']:<10} {d['status']:<10} {d['rows_in']:>5} -> {d['rows_out']:<5} validation={d['validation_status']} version={d['data_version']}")
    again = svc.submit("demo", files, wait=True)
    print("resubmit same files -> run:", again["run_id"], "reused:", again["reused"])
    print("events:", [e["event"] for e in run["events"]])
    print("typed orders dtypes:", dict(svc.read_cleaned("demo", first["run_id"], "orders", typed=True).dtypes.astype(str)))
