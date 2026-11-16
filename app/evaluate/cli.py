"""python -m app.evaluate --input cases.json --output results.json --artifacts-dir ./evaluation_output"""
import argparse
import sys
import time

from app.evaluate.runner import evaluate_file


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.evaluate", description="Run cleaning + chat cases through the production pipeline.")
    ap.add_argument("--input", required=True, help="cases JSON file")
    ap.add_argument("--output", required=True, help="results JSON file to write")
    ap.add_argument("--artifacts-dir", default="./evaluation_output", help="where original/cleaned CSVs, reports and chat traces are written")
    ap.add_argument("--provider", default=None, help="mock (default, keyless) or anthropic; overrides LLM_PROVIDER")
    ap.add_argument("--max-upload-mb", type=float, default=None)
    ap.add_argument("--store-dir", default=None, help="persist the run store here (default: temporary, discarded)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any case failed or errored (default: exit 0 once the batch completed)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    doc, code = evaluate_file(a.input, a.output, a.artifacts_dir, a.provider, a.max_upload_mb, a.store_dir)
    if code == 2:
        print(f"FATAL {doc['fatal_error']['code']}: {doc['fatal_error']['message']}", file=sys.stderr)
        return 2
    s = doc["summary"]
    print(f"cases: {s['cases_passed']} passed, {s['cases_failed']} failed, {s['cases_error']} error / {s['cases_total']} | "
          f"datasets: {s['datasets_succeeded']}/{s['datasets_total']} cleaned | questions: {s['questions_total']} | checks: {s['checks_passed']}/{s['checks_total']} "
          f"| {time.perf_counter() - t0:.1f}s -> {a.output}", file=sys.stderr)
    return 1 if a.strict and (s["cases_failed"] or s["cases_error"]) else 0
