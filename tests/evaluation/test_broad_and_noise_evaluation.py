from __future__ import annotations

import json

from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.evaluation import evaluate_fixture

from conftest import BROAD_FIXTURE_ROOT, BROAD_QA_PATH, NOISE_FIXTURE_ROOT, NOISE_QA_PATH


def test_broad_world_evaluation_runs_with_honest_score() -> None:
    result = evaluate_fixture(BROAD_FIXTURE_ROOT, BROAD_QA_PATH)

    assert result.total == 65
    assert result.correct == 65
    assert "school_homework" in result.by_category
    assert "medical_note" in result.by_category
    assert "multilingual_unknown" in result.by_category
    assert "raw_json_text" in result.by_category


def test_hardcore_noise_evaluation_runs_and_preserves_meaningful_answers() -> None:
    result = evaluate_fixture(NOISE_FIXTURE_ROOT, NOISE_QA_PATH)

    assert result.total == 8
    assert result.correct == 8
    assert "noise_unknown" in result.by_category
    assert "noise_ingest" in result.by_category


def test_noise_documents_are_marked_without_breaking_meaningful_queries() -> None:
    engine = KnowMoreDiRTEngine(NOISE_FIXTURE_ROOT)
    rows = engine.store.execute("SELECT metadata_json FROM documents").fetchall()
    noise_flags = [
        json.loads(row["metadata_json"])["text_quality"]["low_semantic_noise"]
        for row in rows
    ]
    quality_labels = {
        json.loads(row["metadata_json"])["text_quality"]["semantic_quality"]
        for row in rows
    }
    context_kinds = {
        row["kind"]
        for row in engine.store.execute("SELECT kind FROM contexts").fetchall()
    }

    assert any(noise_flags)
    assert "random_character_noise" in quality_labels
    assert any(kind.startswith("quality:") for kind in context_kinds)
    assert engine.answer("Who watered the greenhouse fern?").text == "Dr. Pella"
    assert engine.answer("What does florpus zeta mean?").text == "unknown"
