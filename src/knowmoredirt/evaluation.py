"""Internal benchmark scoring over the public engine contract."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .engine import KnowMoreDiRTEngine


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().rstrip("."))


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(value))


def token_f1(predicted: str, expected: str) -> float:
    predicted_tokens = _tokens(predicted)
    expected_tokens = _tokens(expected)
    if not predicted_tokens and not expected_tokens:
        return 1.0
    if not predicted_tokens or not expected_tokens:
        return 0.0
    overlap = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def _unknown_like(value: str) -> bool:
    text = normalize(value)
    if text in {"unknown", "not known", "unavailable", "not available"}:
        return True
    markers = [
        "no stated translation",
        "no translation is stated",
        "not stated",
        "not specified",
        "not provided",
        "cannot be determined",
        "insufficient evidence",
    ]
    return any(marker in text for marker in markers)


def exact_match(predicted: str, expected: str) -> bool:
    return normalize(predicted) == normalize(expected)


def semantic_match(predicted: str, expected: str) -> bool:
    if exact_match(predicted, expected):
        return True
    if _unknown_like(expected) and _unknown_like(predicted):
        return True
    p = normalize(predicted)
    e = normalize(expected)
    p_first = _tokens(p)[:1]
    e_first = _tokens(e)[:1]
    if p_first and e_first and p_first[0] in {"yes", "no", "true", "false"} and p_first == e_first:
        return True
    predicted_tokens = _tokens(predicted)
    expected_tokens = _tokens(expected)
    modal_tokens = {"should", "must", "can", "could", "will", "would", "may", "might"}
    for modal in modal_tokens:
        if modal in predicted_tokens and modal in expected_tokens:
            predicted_tail = predicted_tokens[predicted_tokens.index(modal):]
            expected_tail = expected_tokens[expected_tokens.index(modal):]
            if predicted_tail == expected_tail:
                return True
    if token_f1(predicted, expected) >= 0.8:
        return True
    if expected_tokens and set(expected_tokens).issubset(predicted_tokens) and len(predicted_tokens) - len(expected_tokens) <= 1:
        return True
    if predicted_tokens and set(predicted_tokens).issubset(expected_tokens) and len(expected_tokens) - len(predicted_tokens) <= 1:
        return True
    if len(expected_tokens) >= 2 and e in p:
        return True
    if len(predicted_tokens) >= 2 and p in e:
        return True
    return False


@dataclass(frozen=True)
class QuestionResult:
    id: str
    category: str
    question: str
    expected: str
    predicted: str
    exact_correct: bool
    semantic_correct: bool
    token_f1: float


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    exact_correct: int
    exact_score: float
    semantic_correct: int
    semantic_score: float
    average_token_f1: float
    by_category: dict[str, dict[str, float | int]]
    results: list[QuestionResult]


def evaluate_fixture(corpus_root: str | Path, qa_path: str | Path, model=None) -> EvaluationResult:
    engine = KnowMoreDiRTEngine(corpus_root, model=model)
    questions = json.loads(Path(qa_path).read_text(encoding="utf-8"))["questions"]
    results: list[QuestionResult] = []
    categories: dict[str, list[QuestionResult]] = defaultdict(list)
    for item in questions:
        predicted = engine.answer(item["question"]).text
        result = QuestionResult(
            item["id"],
            item["category"],
            item["question"],
            item["answer"],
            predicted,
            exact_match(predicted, item["answer"]),
            semantic_match(predicted, item["answer"]),
            token_f1(predicted, item["answer"]),
        )
        results.append(result)
        categories[item["category"]].append(result)
    total = len(results)
    exact_correct = sum(item.exact_correct for item in results)
    semantic_correct = sum(item.semantic_correct for item in results)
    return EvaluationResult(
        total=total,
        exact_correct=exact_correct,
        exact_score=exact_correct / total if total else 0.0,
        semantic_correct=semantic_correct,
        semantic_score=semantic_correct / total if total else 0.0,
        average_token_f1=sum(item.token_f1 for item in results) / total if total else 0.0,
        by_category={
            key: {
                "total": len(values),
                "exact_correct": sum(item.exact_correct for item in values),
                "exact_score": sum(item.exact_correct for item in values) / len(values),
                "semantic_correct": sum(item.semantic_correct for item in values),
                "semantic_score": sum(item.semantic_correct for item in values) / len(values),
                "average_token_f1": sum(item.token_f1 for item in values) / len(values),
            }
            for key, values in sorted(categories.items())
        },
        results=results,
    )


def evaluation_to_dict(result: EvaluationResult) -> dict:
    return asdict(result)
