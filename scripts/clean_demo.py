"""python scripts/clean_demo.py [files...]  - clean, validate, write cleaned CSVs to evaluation_output/demo/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline.cleaner import clean_dataframe
from app.pipeline.loader import load_csv

out = Path("evaluation_output/demo"); out.mkdir(parents=True, exist_ok=True)
for p in sys.argv[1:] or sorted(str(x) for x in Path("data/samples").glob("*.csv")):
    r = load_csv(p)
    res = clean_dataframe(r.df, Path(p).stem)
    rep, v = res.report, res.report["validation"]
    res.df.to_csv(out / f"{Path(p).stem}_clean.csv", index=False)
    print(f"\n=== {r.filename}: {rep['dataset']['rows_in']} -> {rep['dataset']['rows_out']} rows | validation: {v['status'].upper()}")
    for s in rep["cleaning_plan"]:
        e = s["execution"]
        print(f"  {s['final_status']:<8} {s['action']:<24} {','.join(s['fields']) or '-':<18} changed={e.get('cells_changed', '-')} nulled={e.get('cells_nulled', '-')} removed={e.get('rows_removed', '-')}")
    for c in v["checks"]:
        print(f"  check {c['status']:<7} {c['name']}")
    if v["values_nulled_by_reason"]:
        print("  nulled by reason:", v["values_nulled_by_reason"])
