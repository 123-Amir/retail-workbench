"""Keeps the documentation honest: claims that can be checked mechanically are checked."""
import json
import re
from pathlib import Path

from app.pipeline.planner import ACTIONS
from app.retail.inventory import INV_METRICS
from app.retail.metrics import METRICS

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_counts_quoted_in_the_readme_match_the_code():
    assert len(METRICS) == 13 and "13 sales metrics" in README
    assert len(INV_METRICS) == 11 and "11 inventory metrics" in README
    assert len(ACTIONS) == 9
    issues_src = (ROOT / "app/pipeline/issues.py").read_text(encoding="utf-8") + (ROOT / "app/pipeline/cleaner.py").read_text(encoding="utf-8")
    types = set(re.findall(r'_issue\(ds, "([A-Z_]+)"', issues_src)) | set(re.findall(r'"type": "([A-Z_]+)"', (ROOT / "app/pipeline/cleaner.py").read_text(encoding="utf-8")))
    assert len(types) == 16 and "16 issue types" in README, sorted(types)


def test_every_script_and_doc_mentioned_in_the_readme_exists():
    for path in set(re.findall(r"`((?:scripts|docs|cases)/[\w./-]+)`", README)) | set(re.findall(r"\b(docs/[A-Z_]+\.md)", README)):
        assert (ROOT / path).exists(), path


def test_env_example_documents_every_variable_the_code_reads():
    used = set(re.findall(r'os\.getenv\("([A-Z_]+)"', (ROOT / "app/config.py").read_text(encoding="utf-8") + (ROOT / "app/api/main.py").read_text(encoding="utf-8")))
    documented = set(re.findall(r"^([A-Z_]+)=", (ROOT / ".env.example").read_text(encoding="utf-8"), re.M))
    assert used and used <= documented, used - documented


def test_the_default_provider_needs_no_key_and_the_root_cases_file_is_the_sample():
    assert "LLM_PROVIDER=mock" in (ROOT / ".env.example").read_text(encoding="utf-8")
    assert (ROOT / "cases.json").read_text(encoding="utf-8") == (ROOT / "cases/sample_cases.json").read_text(encoding="utf-8")


def test_every_sample_case_file_exists_and_ids_are_unique():
    cases = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))["cases"]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        if isinstance(c.get("files"), list):
            for f in c["files"]:
                assert (ROOT / f).exists() or "does_not_exist" in f, f
