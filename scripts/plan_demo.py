"""python scripts/plan_demo.py [files...]  - shows issues and the cleaning plan (no data is changed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline.analyze import analyze_dataframe
from app.pipeline.loader import load_csv

for p in sys.argv[1:] or sorted(str(x) for x in Path("data/samples").glob("*.csv")):
    r = load_csv(p)
    a = analyze_dataframe(r.df, r.filename.replace(".csv", ""))
    print(f"\n=== {r.filename}: {len(a['issues'])} issues, {a['plan_summary']['total_steps']} steps")
    for i in a["issues"]:
        print(f"  [{i['severity']:<6}] {i['id']} {i['type']:<26} {','.join(i['columns']) or '-':<14} rows={i['affected_rows']}")
    for s in a["cleaning_plan"]:
        print(f"  {s['order']:>2}. {s['id']} {s['action']:<24} {','.join(s['fields']) or '-':<14} risk={s['risk']:<6} {s['status']:<12} default={s['default_action']}")
