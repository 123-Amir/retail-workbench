"""Cleaned artifacts are CSV text. This restores real dtypes from the schema recorded at cleaning time,
so analytics (chat, retail metrics) never has to guess types again."""
import pandas as pd

from app.errors import PipelineError


def typed_frame(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    out = df.copy()
    for col, ctype in schema.items():
        if col not in out.columns:
            continue
        try:
            if ctype == "integer":
                out[col] = pd.to_numeric(out[col], errors="raise").astype("Int64")
            elif ctype == "float":
                out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
            elif ctype == "date":
                out[col] = pd.to_datetime(out[col], format="%Y-%m-%d", errors="raise")
            else:
                out[col] = out[col].astype("string")
        except (ValueError, TypeError) as e:
            raise PipelineError("TYPE_CAST_FAILED", f"Column '{col}' could not be cast to {ctype}.", {"detail": str(e)[:150]})
    return out
