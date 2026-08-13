from __future__ import annotations

import json
from pathlib import Path


def test_recording_requirements_fixture_covers_recorded_contract() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures"
    payload = json.loads((root / "recording_requirements_qa.json").read_text(encoding="utf-8"))
    categories = {item["category"] for item in payload["questions"]}
    assert {
        "dream_real_world_scope",
        "dream_requested_scope",
        "reported_real_world_scope",
        "reported_requested_scope",
        "asserted_official_scope",
        "semantic_vector_retrieval",
        "source_only_no_world_knowledge",
    }.issubset(categories)
    assert (root / "recording_requirements" / "timmy_dream.txt").is_file()
    assert (root / "recording_requirements" / "vector_lexicon.txt").is_file()
