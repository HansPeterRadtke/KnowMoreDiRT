from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "broad_raw_world_qa.json": 65,
    "hardcore_noise_qa.json": 8,
    "hard_raw_reasoning_qa.json": 134,
    "messy_raw_corpus_qa.json": 60,
    "structured_record_json_qa.json": 6,
}


def test_internal_benchmark_is_broad_and_intact():
    categories = set()
    for name, count in EXPECTED.items():
        payload = json.loads((ROOT / "tests" / "fixtures" / name).read_text())
        assert len(payload["questions"]) == count
        assert all({"id", "category", "question", "answer"}.issubset(item) for item in payload["questions"])
        categories.update(item["category"] for item in payload["questions"])
    assert len(categories) >= 80
