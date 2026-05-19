from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "messy_raw_corpus"
QA_PATH = REPO_ROOT / "tests" / "fixtures" / "messy_raw_corpus_qa.json"
BROAD_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "broad_raw_world"
BROAD_QA_PATH = REPO_ROOT / "tests" / "fixtures" / "broad_raw_world_qa.json"
NOISE_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "hardcore_noise"
NOISE_QA_PATH = REPO_ROOT / "tests" / "fixtures" / "hardcore_noise_qa.json"
