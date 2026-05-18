from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "messy_raw_corpus"
QA_PATH = REPO_ROOT / "tests" / "fixtures" / "messy_raw_corpus_qa.json"

