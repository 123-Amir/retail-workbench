"""Path and identifier safety. Owners, filenames and dataset names are untrusted input."""
import os
import re
from pathlib import Path

from app.errors import PipelineError
from app.pipeline.loader import sanitize_filename

_OWNER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _without_extended_prefix(path: str) -> str:
    if path.startswith("\\\\?\\UNC\\"):
        return "\\\\" + path[8:]
    if path.startswith("\\\\?\\"):
        return path[4:]
    return path


def validate_owner(owner: str) -> str:
    if not isinstance(owner, str) or not _OWNER.match(owner):
        raise PipelineError("INVALID_OWNER", "Owner id must be 1-64 chars of letters, digits, '_' or '-'.")
    return owner


def dataset_name_from_filename(filename: str) -> str:
    stem = Path(sanitize_filename(filename)).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:50] or "dataset"


def safe_join(root: Path, *parts: str) -> Path:
    """Join and verify the result is still inside root (blocks '..' and absolute-path tricks)."""
    root_path = Path(_without_extended_prefix(os.path.realpath(root)))
    candidate = Path(_without_extended_prefix(os.path.realpath(root_path.joinpath(*parts))))
    try:
        common = os.path.commonpath((os.path.normcase(str(root_path)), os.path.normcase(str(candidate))))
    except ValueError:
        common = None
    if common != os.path.normcase(str(root_path)):
        raise PipelineError("PATH_ESCAPE", "Resolved path is outside the storage root.")
    return candidate
