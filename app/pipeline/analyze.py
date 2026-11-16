"""Pre-mutation analysis: profile -> issues -> plan. Nothing here changes the data."""
import pandas as pd

from app.pipeline.issues import detect_issues
from app.pipeline.planner import build_cleaning_plan, summarize_plan
from app.pipeline.profiler import profile_dataframe


def analyze_dataframe(df: pd.DataFrame, name: str) -> dict:
    profile = profile_dataframe(df, name)
    issues = detect_issues(df, profile)
    plan = build_cleaning_plan(name, issues, profile)
    return {"profile_before": profile, "issues": issues, "cleaning_plan": plan, "plan_summary": summarize_plan(plan)}
