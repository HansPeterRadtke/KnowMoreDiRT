"""Optional local-model helpers for generic query frames.

Model use is isolated and local-only.  The planner asks for a generic
relation/query frame, never an external label or hardcoded semantic intent.
Evidence answering is constrained to bounded raw-text snippets and is validated
against source grounding before it can leave the engine.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from .model import LocalModelClient
from .query import QueryFrame, frame_from_mapping, plan_question


QUERY_FRAME_GRAMMAR = r'''
root ::= "{" ws "\"query_frame\"" ws ":" ws "{" ws "\"target_anchors\"" ws ":" ws string_array ws "," ws "\"requested_relation\"" ws ":" ws string ws "," ws "\"relation_terms\"" ws ":" ws string_array ws "," ws "\"constraints\"" ws ":" ws string_array ws "," ws "\"answer_type\"" ws ":" ws answer_type ws "," ws "\"temporal_scope\"" ws ":" ws string ws "," ws "\"negated\"" ws ":" ws bool ws "," ws "\"aggregation\"" ws ":" ws string ws "," ws "\"requires_evidence\"" ws ":" ws bool ws "}" ws "}"
answer_type ::= "\"person\"" | "\"actor\"" | "\"organization\"" | "\"identifier\"" | "\"url\"" | "\"file_path\"" | "\"count\"" | "\"state\"" | "\"date_time\"" | "\"boolean\"" | "\"content_phrase\"" | "\"metadata_value\"" | "\"unknown\""
string_array ::= "[" ws (string (ws "," ws string)*)? ws "]"
bool ::= "true" | "false"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" ["\\/bfnrt])*
ws ::= [ \t\n\r]*
'''

EVIDENCE_EXTRACTION_GRAMMAR = r'''
root ::= "{" ws "\"answer\"" ws ":" ws "{" ws "\"sufficient_evidence\"" ws ":" ws bool ws "," ws "\"answer_type\"" ws ":" ws answer_type ws "," ws "\"answer\"" ws ":" ws string ws "," ws "\"evidence_span\"" ws ":" ws string ws "}" ws "}"
answer_type ::= "\"person\"" | "\"actor\"" | "\"organization\"" | "\"identifier\"" | "\"url\"" | "\"file_path\"" | "\"count\"" | "\"state\"" | "\"date_time\"" | "\"boolean\"" | "\"content_phrase\"" | "\"metadata_value\"" | "\"unknown\""
bool ::= "true" | "false"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" ["\\/bfnrt])*
ws ::= [ \t\n\r]*
'''


@dataclass
class ModelQueryTrace:
    enabled: bool = False
    call_count: int = 0
    parsed_count: int = 0
    accepted_count: int = 0
    model_answer_count: int = 0
    evidence_call_count: int = 0
    evidence_parsed_count: int = 0
    evidence_accepted_count: int = 0
    evidence_rejected_count: int = 0
    prompt_hashes: list[str] | None = None
    response_hashes: list[str] | None = None
    last_plan: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "call_count": self.call_count,
            "parsed_count": self.parsed_count,
            "accepted_count": self.accepted_count,
            "model_answer_count": self.model_answer_count,
            "evidence_call_count": self.evidence_call_count,
            "evidence_parsed_count": self.evidence_parsed_count,
            "evidence_accepted_count": self.evidence_accepted_count,
            "evidence_rejected_count": self.evidence_rejected_count,
            "prompt_hashes": self.prompt_hashes or [],
            "response_hashes": self.response_hashes or [],
            "last_plan": self.last_plan,
        }


def deterministic_plan(question: str) -> dict[str, Any]:
    """Compatibility wrapper returning the deterministic query frame as dict."""

    return plan_question(question).as_dict()


def normalize_model_plan(question: str, model: dict[str, Any] | None, det: dict[str, Any]) -> dict[str, Any] | None:
    if model and model.get("accepted"):
        frame = frame_from_mapping(question, model.get("query_frame") if isinstance(model.get("query_frame"), dict) else model, source="model")
        return {**frame.as_dict(), "accepted": True}
    if det:
        return det
    return plan_question(question).as_dict()


def build_query_plan_prompt(question: str) -> str:
    frame = plan_question(question).as_dict()
    return (
        "JSON only. Convert the question into a generic DRT/DSPG query frame; do not answer it. "
        "Use only relation text visible in the question. Do not choose dataset labels, hidden categories, "
        "or source-specific shortcuts. The frame must contain target anchors, requested relation text, "
        "relation terms, constraints, broad answer type, temporal scope, negation, aggregation, and "
        "whether source evidence is required."
        + json.dumps({"question": question, "deterministic_frame": frame}, ensure_ascii=False)
    )


def call_model_query_plan(question: str, client: LocalModelClient, *, n_predict: int = 160) -> dict[str, Any]:
    prompt = build_query_plan_prompt(question)
    start = time.time()
    try:
        parsed = client.complete_json(prompt, n_predict=n_predict, grammar=QUERY_FRAME_GRAMMAR)
    except Exception as exc:
        return {
            **plan_question(question).as_dict(),
            "source": "model",
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "grammar_hash": hashlib.sha256(QUERY_FRAME_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    frame_payload = parsed.get("query_frame") if isinstance(parsed, dict) else None
    if not isinstance(frame_payload, dict):
        return {
            **plan_question(question).as_dict(),
            "source": "model",
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "grammar_hash": hashlib.sha256(QUERY_FRAME_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    frame = frame_from_mapping(question, frame_payload, source="model").as_dict()
    return {
        **frame,
        "accepted": True,
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "stop_reason": "parsed_json",
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "grammar_hash": hashlib.sha256(QUERY_FRAME_GRAMMAR.encode()).hexdigest(),
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }


def build_evidence_extraction_prompt(question: str, expected_answer_type: str, evidence_items: list[dict[str, str]]) -> str:
    return (
        "JSON only. Answer the question only from the provided raw-text evidence. "
        "Return sufficient_evidence=false and answer='unknown' when the evidence does not state a complete answer. "
        "The answer must be the shortest exact grounded value compatible with the expected answer type. "
        "Do not use outside knowledge."
        + json.dumps(
            {
                "question": question,
                "expected_answer_type": expected_answer_type,
                "evidence": evidence_items,
            },
            ensure_ascii=False,
        )
    )


def call_model_evidence_answer(
    question: str,
    expected_answer_type: str,
    evidence_items: list[dict[str, str]],
    client: LocalModelClient,
    *,
    n_predict: int = 160,
) -> dict[str, Any]:
    prompt = build_evidence_extraction_prompt(question, expected_answer_type, evidence_items)
    start = time.time()
    try:
        parsed = client.complete_json(prompt, n_predict=n_predict, grammar=EVIDENCE_EXTRACTION_GRAMMAR)
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "grammar_hash": hashlib.sha256(EVIDENCE_EXTRACTION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    answer = parsed.get("answer") if isinstance(parsed, dict) else None
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    if not isinstance(answer, dict):
        return {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "grammar_hash": hashlib.sha256(EVIDENCE_EXTRACTION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    return {
        "accepted": True,
        "sufficient_evidence": bool(answer.get("sufficient_evidence")),
        "answer_type": str(answer.get("answer_type") or "unknown"),
        "answer": str(answer.get("answer") or ""),
        "evidence_span": str(answer.get("evidence_span") or ""),
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "grammar_hash": hashlib.sha256(EVIDENCE_EXTRACTION_GRAMMAR.encode()).hexdigest(),
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
    }
