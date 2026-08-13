"""Internal evaluation helpers for fixture QA reports."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kmd_runtime_config import boolean as _config_boolean, text as _config_text, model_cache_dir as _model_cache_dir

from .answer_types import is_unknown_text
from .engine import KnowMoreDiRTEngine
from .model import LocalModelClient, complete_json_with_transport_retry
from .runtime_logging import get_logger
from .text import normalize


LOGGER = get_logger("evaluation")
JUDGE_SCHEMA_VERSION = "kmd-answer-equivalence-v1"
JUDGE_PROMPT_VERSION = "answer-equivalence-source-independent-v1"
JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["equivalent", "reason"],
    "properties": {
        "equivalent": {"type": "boolean"},
        "reason": {"type": "string"},
    },
}


@dataclass(frozen=True)
class QuestionResult:
    id: str
    category: str
    question: str
    expected: str
    predicted: str
    correct: bool
    evaluation_reason: str = ""


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    correct: int
    score: float
    by_category: dict[str, dict[str, float | int]]
    results: list[QuestionResult]


def answer_matches(predicted: str, expected: str) -> bool:
    predicted_norm = normalize(predicted).rstrip(".?!")
    expected_norm = normalize(expected).rstrip(".?!")
    if expected_norm == "unknown":
        return is_unknown_text(predicted)
    if predicted_norm == expected_norm:
        return True
    if predicted_norm in {"yes", "no"} and expected_norm.startswith(predicted_norm + ";"):
        return True
    return False


def _judge_enabled() -> bool:
    return _config_boolean("KMD_EVALUATION_USE_LOCAL_JUDGE")


def _judge_cache_root() -> Path:
    return _model_cache_dir("KMD_EVALUATION_JUDGE_CACHE_DIR")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)


def semantic_answer_judgment(
    question: str,
    predicted: str,
    expected: str,
    *,
    client: LocalModelClient | None = None,
) -> dict[str, Any]:
    """Judge semantic equivalence without supplying or inferring source facts.

    The judge is evaluation-only.  It receives the evaluation question and two
    already-produced answer strings; it never participates in KMD retrieval or
    reasoning and therefore cannot inject world knowledge into the KMD answer.
    """

    if answer_matches(predicted, expected):
        return {
            "equivalent": True,
            "reason": "deterministic_match",
            "judge_used": False,
            "cache_hit": False,
        }
    if client is None and not _judge_enabled():
        return {
            "equivalent": False,
            "reason": "deterministic_mismatch_local_judge_disabled",
            "judge_used": False,
            "cache_hit": False,
        }
    model = client or LocalModelClient(
        endpoint=_config_text("KMD_LOCAL_MODEL_ENDPOINT")
    )
    fingerprint = model.cache_fingerprint()
    cache_context = {
        "schema": JUDGE_SCHEMA_VERSION,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "question": str(question),
        "expected": str(expected),
        "predicted": str(predicted),
        "model_fingerprint": fingerprint,
    }
    digest = hashlib.sha256(_canonical_json(cache_context).encode("utf-8")).hexdigest()
    path = _judge_cache_root() / f"{digest}.json"
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("cache_context") == cache_context:
            result = payload.get("result")
            if isinstance(result, dict) and isinstance(result.get("equivalent"), bool):
                return {**result, "judge_used": True, "cache_hit": True}
    prompt = (
        "Decide whether the EXPECTED ANSWER and PREDICTED ANSWER are semantically equivalent answers "
        "to the QUESTION. Judge meaning, not wording. Do not add facts, repair either answer, or use world "
        "knowledge to make a false answer true. Treat paraphrases, reordered conjunctions, equivalent date/number "
        "formats, and concise versus expanded wording as equivalent only when they preserve all material meaning. "
        "Treat an explicit unknown/unsupported answer as equivalent to another explicit unknown/unsupported answer, "
        "including one that additionally mentions subordinate evidence while still clearly saying the real answer is unknown.\n\n"
        f"QUESTION:\n{question}\n\nEXPECTED ANSWER:\n{expected}\n\nPREDICTED ANSWER:\n{predicted}"
    )
    raw = complete_json_with_transport_retry(
        model,
        prompt,
        n_predict=256,
        json_schema=JUDGE_JSON_SCHEMA,
    )
    equivalent = bool(raw.get("equivalent"))
    reason = str(raw.get("reason") or "").strip() or "judge_returned_no_reason"
    result = {"equivalent": equivalent, "reason": reason}
    _atomic_write_json(path, {"cache_context": cache_context, "result": result})
    return {**result, "judge_used": True, "cache_hit": False}


def evaluate_fixture(corpus_root: str | Path, qa_path: str | Path) -> EvaluationResult:
    engine = KnowMoreDiRTEngine(corpus_root)
    payload = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    results: list[QuestionResult] = []
    category_counts: dict[str, list[bool]] = defaultdict(list)
    progress = _config_boolean("KMD_EVAL_PROGRESS")
    questions = payload["questions"]
    for index, entry in enumerate(questions, start=1):
        progress_message = f"kmd-eval {Path(qa_path).name} {index}/{len(questions)} {entry['id']}"
        LOGGER.info(progress_message)
        if progress:
            print(progress_message, flush=True)
        answer = engine.answer(entry["question"]).text
        judgment = semantic_answer_judgment(entry["question"], answer, entry["answer"])
        correct = bool(judgment["equivalent"])
        results.append(
            QuestionResult(
                id=entry["id"],
                category=entry["category"],
                question=entry["question"],
                expected=entry["answer"],
                predicted=answer,
                correct=correct,
                evaluation_reason=str(judgment.get("reason") or ""),
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
