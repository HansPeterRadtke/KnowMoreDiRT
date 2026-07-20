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


def test_hard_raw_reasoning_evaluation_reports_no_model_regression_floor() -> None:
    result = evaluate_fixture(HARD_REASONING_ROOT, HARD_REASONING_QA_PATH)

    assert result.total == 134
    # The pytest-only no-model path intentionally exercises deterministic infrastructure only.
    # Semantic role disambiguation belongs to the required model path, so do not hard-code the
    # remaining actor-role and epistemic decisions into Python merely to make this fixture perfect.
    assert result.correct == 130
    assert result.by_category["actor_identifier_role_chain"]["correct"] == 2
    assert result.by_category["actor_identifier_role_chain"]["total"] == 5
    model_dependent_categories = {"actor_identifier_role_chain", "context_discourse"}
    assert len(
        [
            values
            for name, values in result.by_category.items()
            if name not in model_dependent_categories and values["correct"] == values["total"]
        ]
    ) == len(result.by_category) - len(model_dependent_categories)
    assert result.by_category["context_discourse"]["correct"] == result.by_category["context_discourse"]["total"] - 1
