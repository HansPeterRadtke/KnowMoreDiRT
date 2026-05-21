from __future__ import annotations

import json

from knowmoredirt.evaluation import evaluate_fixture

from conftest import HARD_REASONING_QA_PATH, HARD_REASONING_ROOT


REQUIRED_CATEGORIES = {
    "wrong_answer_type",
    "identifier_family",
    "url_confusion",
    "organization_person_confusion",
    "content_phrase",
    "unanswerable_false_positive",
    "nested_json",
    "multi_hop",
    "temporal_state",
    "context_discourse",
    "noise_pollution",
    "canonical_output",
    "counts_aggregation",
    "tables_logs",
    "mixed_formats",
}


def test_hard_raw_reasoning_fixture_is_broad_and_failure_driven() -> None:
    payload = json.loads(HARD_REASONING_QA_PATH.read_text(encoding="utf-8"))
    categories = {entry["category"] for entry in payload["questions"]}

    assert HARD_REASONING_ROOT.exists()
    assert len(payload["questions"]) >= 80
    assert REQUIRED_CATEGORIES.issubset(categories)


def test_hard_raw_reasoning_evaluation_reaches_full_correctness() -> None:
    result = evaluate_fixture(HARD_REASONING_ROOT, HARD_REASONING_QA_PATH)

    assert result.total == 134
    assert result.correct == 134
    assert all(values["correct"] == values["total"] for values in result.by_category.values())
