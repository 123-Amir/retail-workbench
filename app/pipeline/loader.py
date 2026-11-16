"""Safe CSV loading. Filenames and contents are untrusted.

Design choices:
- Everything is read as *string* (dtype=str). We never let pandas guess types, because
  guessing silently changes data (e.g. 'SKU-0001' stays fine, but '007' would become 7).
- Empty cells become NaN; any other text (including 'N/A') is preserved for the profiler to judge.
- Encoding fallback: utf-8-sig -> cp1252 -> latin-1. The encoding used is reported.
"""
import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.config import settings
from app.errors import PipelineError

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def sanitize_filename(name: str) -> str:
    base = Path(name).name  # drops any directory part, blocks '../' tricks
    cleaned = _SAFE_CHARS.sub("_", base)[:100].lstrip(".")
    return cleaned or "upload.csv"


@dataclass
class LoadResult:
    df: pd.DataFrame
    filename: str
    encoding: str
    sha256: str
    size_bytes: int
    warnings: list = field(default_factory=list)


def load_csv(path, max_mb: float | None = None) -> LoadResult:
    p = Path(path)
    if p.suffix.lower() != ".csv":
        raise PipelineError("UNSUPPORTED_FILE_TYPE", "Only .csv files are accepted.", {"suffix": p.suffix})
    if not p.is_file():
        raise PipelineError("FILE_NOT_FOUND", "File does not exist.", {"filename": sanitize_filename(p.name)})

    size = p.stat().st_size
    limit_mb = settings.max_upload_mb if max_mb is None else max_mb
    if size > limit_mb * 1024 * 1024:
        raise PipelineError("FILE_TOO_LARGE", f"File exceeds {limit_mb} MB limit.", {"size_bytes": size})
    if size == 0:
        raise PipelineError("EMPTY_FILE", "File has 0 bytes.")

    raw = p.read_bytes()  # bounded by the size limit above
    text, used = None, None
    for enc in ENCODINGS:
        try:
            text, used = raw.decode(enc), enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:  # latin-1 cannot fail, so this is only a safety net
        raise PipelineError("UNDECODABLE_FILE", "Could not decode file.")

    warnings = []
    if used != "utf-8-sig":
        warnings.append({"code": "NON_UTF8_ENCODING", "message": f"Decoded using {used}."})

    header = next(csv.reader(io.StringIO(text)), [])
    if not any(h.strip() for h in header):
        raise PipelineError("NO_HEADER", "File has no usable header row.")
    dupes = sorted({h for h in header if header.count(h) > 1})
    if dupes:
        warnings.append({"code": "DUPLICATE_COLUMN_NAMES", "message": "Duplicate header names.", "columns": dupes})

    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, na_values=[""])
    except pd.errors.EmptyDataError:
        raise PipelineError("EMPTY_FILE", "File has no data.")
    except pd.errors.ParserError as e:
        raise PipelineError("MALFORMED_CSV", "CSV could not be parsed.", {"detail": str(e)[:200]})

    if len(df) == 0:
        warnings.append({"code": "HEADER_ONLY", "message": "File has a header but zero data rows."})

    return LoadResult(df=df, filename=sanitize_filename(p.name), encoding=used,
                      sha256=hashlib.sha256(raw).hexdigest(), size_bytes=size, warnings=warnings)
