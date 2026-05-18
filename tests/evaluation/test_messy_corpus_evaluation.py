from __future__ import annotations

from knowmoredirt.evaluation import evaluate_fixture

from conftest import FIXTURE_ROOT, QA_PATH


def test_messy_corpus_evaluation_runs_and_reports_categories() -> None:
    result = evaluate_fixture(FIXTURE_ROOT, QA_PATH)

    assert result.total == 60
    assert 0.0 <= result.score <= 1.0
    assert result.correct >= 20
    assert "unanswerable" in result.by_category
    assert "dream_vs_fact" in result.by_category
    assert "temporal" in result.by_category
    assert len(result.results) == 60

