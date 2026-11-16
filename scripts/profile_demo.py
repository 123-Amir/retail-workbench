"""Quick look: python scripts/profile_demo.py [path/to/file.csv ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.errors import PipelineError
from app.pipeline.loader import load_csv
from app.pipeline.profiler import profile_dataframe

paths = sys.argv[1:] or sorted(str(p) for p in Path("data/samples").glob("*.csv"))
for p in paths:
    try:
        r = load_csv(p)
    except PipelineError as e:
        print(f"\n{p}: ERROR {e.code}"); continue
    pr = profile_dataframe(r.df, r.filename)
    print(f"\n== {r.filename}: {pr['row_count']} rows x {pr['column_count']} cols | exact dups={pr['duplicate_rows']} | dups after normalisation={pr['duplicate_rows_after_normalization']}")
    for c in pr["columns"]:
        extra = []
        if c["parse_failures"]["count"]: extra.append(f"parse_fail={c['parse_failures']['count']}")
        if c["placeholder_null_count"]: extra.append(f"placeholders={c['placeholder_null_count']}")
        if c["whitespace_issue_count"]: extra.append(f"whitespace={c['whitespace_issue_count']}")
        if c.get("case_whitespace_variant_group_count"): extra.append(f"variant_groups={c['case_whitespace_variant_group_count']}")
        if c.get("date_stats", {}).get("ambiguous_count"): extra.append(f"ambiguous_dates={c['date_stats']['ambiguous_count']}")
        print(f"  {c['name']:<18}{c['inferred_type']:<9} nulls={c['null_count']:<4} unique={c['unique_count']:<5} {' '.join(extra)}")
