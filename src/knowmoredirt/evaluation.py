"""Internal evaluation helpers for fixture QA reports."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .engine import KnowMoreDiRTEngine
from .text import normalize


@dataclass(frozen=True)
class QuestionResult:
    id: str
    category: str
    question: str
    expected: str
    predicted: str
    correct: bool


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    correct: int
    score: float
    by_category: dict[str, dict[str, float | int]]
    results: list[QuestionResult]


def answer_matches(predicted: str, expected: str) -> bool:
    if normalize(expected) == "unknown":
        return normalize(predicted) == "unknown"
    return normalize(predicted) == normalize(expected)


def evaluate_fixture(corpus_root: str | Path, qa_path: str | Path) -> EvaluationResult:
    engine = KnowMoreDiRTEngine(corpus_root)
    payload = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    results: list[QuestionResult] = []
    category_counts: dict[str, list[bool]] = defaultdict(list)
    for entry in payload["questions"]:
        answer = engine.answer(entry["question"]).text
        correct = answer_matches(answer, entry["answer"])
        results.append(
            QuestionResult(
                id=entry["id"],
                category=entry["category"],
                question=entry["question"],
                expected=entry["answer"],
                predicted=answer,
                correct=correct,
            )
        )
        category_counts[entry["category"]].append(correct)
    correct_count = sum(1 for item in results if item.correct)
    by_category = {
        category: {
            "total": len(values),
            "correct": sum(1 for value in values if value),
            "score": (sum(1 for value in values if value) / len(values)) if values else 0.0,
        }
        for category, values in sorted(category_counts.items())
    }
    return EvaluationResult(
        total=len(results),
        correct=correct_count,
        score=(correct_count / len(results)) if results else 0.0,
        by_category=by_category,
        results=results,
    )


def evaluation_to_dict(result: EvaluationResult) -> dict:
    data = asdict(result)
    data["results"] = [asdict(item) for item in result.results]
    return data

