from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "messy_raw_corpus"
QA_PATH = REPO_ROOT / "tests" / "fixtures" / "messy_raw_corpus_qa.json"
BROAD_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "broad_raw_world"
BROAD_QA_PATH = REPO_ROOT / "tests" / "fixtures" / "broad_raw_world_qa.json"
NOISE_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "hardcore_noise"
NOISE_QA_PATH = REPO_ROOT / "tests" / "fixtures" / "hardcore_noise_qa.json"
HARD_REASONING_ROOT = REPO_ROOT / "tests" / "fixtures" / "hard_raw_reasoning"
HARD_REASONING_QA_PATH = REPO_ROOT / "tests" / "fixtures" / "hard_raw_reasoning_qa.json"


def pytest_configure() -> None:
    os.environ.setdefault("KMD_TEST_ALLOW_NO_MODEL", "1")
    os.environ.setdefault("KMD_TEST_ALLOW_MODEL_EVIDENCE_TOOLS", "1")
