"""Optional local-model helpers for generic query frames.

Model use is isolated and local-only.  The planner asks for a generic
relation/query frame, never an external label or hardcoded semantic intent.
Evidence answering is constrained to bounded raw-text snippets and is validated
against source grounding before it can leave the engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import LocalModelClient
from .extractors import identifiers, urls
from .query import QueryFrame, frame_from_mapping, visible_anchors
from .text import content_tokens


ANSWER_TYPES = {
    "person",
    "actor",
    "organization",
    "identifier",
    "url",
    "file_path",
    "count",
    "state",
    "date_time",
    "boolean",
    "content_phrase",
    "metadata_value",
    "unknown",
}

QUERY_FRAME_GRAMMAR = r'''
root ::= "{" ws "\"query_frame\"" ws ":" ws "{" ws "\"target_anchors\"" ws ":" ws string_array ws "," ws "\"requested_relation\"" ws ":" ws string ws "," ws "\"relation_terms\"" ws ":" ws string_array ws "," ws "\"constraints\"" ws ":" ws string_array ws "," ws "\"answer_type\"" ws ":" ws answer_type ws "," ws "\"temporal_scope\"" ws ":" ws string ws "," ws "\"negated\"" ws ":" ws bool ws "," ws "\"aggregation\"" ws ":" ws string ws "," ws "\"requires_evidence\"" ws ":" ws bool ws "}" ws "}"
answer_type ::= "\"person\"" | "\"actor\"" | "\"organization\"" | "\"identifier\"" | "\"url\"" | "\"file_path\"" | "\"count\"" | "\"state\"" | "\"date_time\"" | "\"boolean\"" | "\"content_phrase\"" | "\"metadata_value\"" | "\"unknown\""
string_array ::= "[" ws (string (ws "," ws string)*)? ws "]"
bool ::= "true" | "false"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" ["\\/bfnrt])*
ws ::= [ \t\n\r]*
'''


def _optional_grammar(grammar: str) -> str | None:
    return grammar if os.environ.get("KMD_LOCAL_MODEL_GRAMMAR", "").strip().lower() in {"1", "true", "yes", "on"} else None


def _coerce_confidence(value: Any, default: float = 0.65) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    text = str(value or "").strip().lower()
    if not text:
        return default
    qualitative = {
        "very high": 0.95,
        "high": 0.85,
        "medium": 0.65,
        "moderate": 0.65,
        "low": 0.35,
        "very low": 0.15,
    }
    if text in qualitative:
        return qualitative[text]
    try:
        parsed = float(text)
    except ValueError:
        return default
    return max(0.0, min(1.0, parsed))


def _cache_path(env_var: str, prompt_hash: str) -> Path | None:
    cache_dir = os.environ.get(env_var, "").strip()
    return Path(cache_dir) / f"{prompt_hash}.json" if cache_dir else None


def _read_cache(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(cached, dict):
        cached["fresh_or_cached"] = "cache"
        return cached
    return None


def _write_cache(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        pass


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _valid_query_frame_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = {
        "target_anchors",
        "requested_relation",
        "relation_terms",
        "constraints",
        "answer_type",
        "temporal_scope",
        "negated",
        "aggregation",
        "requires_evidence",
    }
    if not required.issubset(value):
        return False
    return (
        _is_string_list(value.get("target_anchors"))
        and isinstance(value.get("requested_relation"), str)
        and _is_string_list(value.get("relation_terms"))
        and _is_string_list(value.get("constraints"))
        and str(value.get("answer_type")) in ANSWER_TYPES
        and isinstance(value.get("temporal_scope"), str)
        and isinstance(value.get("negated"), bool)
        and isinstance(value.get("aggregation"), str)
        and isinstance(value.get("requires_evidence"), bool)
    )


def _repair_query_frame_payload(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    repaired = dict(value)
    if "requested_relation" not in repaired and "requested_relations" in repaired:
        raw = repaired.get("requested_relations")
        if isinstance(raw, list):
            repaired["requested_relation"] = " ".join(str(item) for item in raw if str(item).strip())
        else:
            repaired["requested_relation"] = str(raw or "")
    if "answer_type" not in repaired and "broad_answer_type" in repaired:
        repaired["answer_type"] = str(repaired.get("broad_answer_type") or "")
    if "negated" not in repaired and "negation" in repaired:
        repaired["negated"] = bool(repaired.get("negation"))
    if "requires_evidence" not in repaired and "source_evidence_required" in repaired:
        repaired["requires_evidence"] = bool(repaired.get("source_evidence_required"))
    if "aggregation" in repaired and isinstance(repaired.get("aggregation"), bool):
        repaired["aggregation"] = "count" if repaired.get("aggregation") else ""
    for key in ["target_anchors", "relation_terms", "constraints"]:
        if key not in repaired:
            repaired[key] = []
        elif not isinstance(repaired.get(key), list):
            repaired[key] = []
    for key in ["requested_relation", "answer_type", "temporal_scope", "aggregation"]:
        if key not in repaired:
            repaired[key] = ""
    if "negated" not in repaired:
        repaired["negated"] = False
    if "requires_evidence" not in repaired:
        repaired["requires_evidence"] = True
    return repaired


def _valid_answer_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = {"sufficient_evidence", "answer_type", "answer", "evidence_span"}
    if not required.issubset(value):
        return False
    return (
        isinstance(value.get("sufficient_evidence"), bool)
        and str(value.get("answer_type")) in ANSWER_TYPES
        and isinstance(value.get("answer"), str)
        and isinstance(value.get("evidence_span"), str)
    )


def _repair_evidence_span(answer: dict[str, Any], evidence_items: list[dict[str, str]]) -> dict[str, Any]:
    repaired = dict(answer)
    span = str(repaired.get("evidence_span") or "")
    if span and any(span in str(item.get("text") or "") for item in evidence_items):
        return repaired
    proposed = str(repaired.get("answer") or "")
    if proposed and proposed.lower() != "unknown":
        for item in evidence_items:
            text = str(item.get("text") or "")
            if proposed in text:
                repaired["evidence_span"] = proposed
                return repaired
    return repaired

EVIDENCE_EXTRACTION_GRAMMAR = r'''
root ::= "{" ws "\"answer\"" ws ":" ws "{" ws "\"sufficient_evidence\"" ws ":" ws bool ws "," ws "\"answer_type\"" ws ":" ws answer_type ws "," ws "\"answer\"" ws ":" ws string ws "," ws "\"evidence_span\"" ws ":" ws string ws "}" ws "}"
answer_type ::= "\"person\"" | "\"actor\"" | "\"organization\"" | "\"identifier\"" | "\"url\"" | "\"file_path\"" | "\"count\"" | "\"state\"" | "\"date_time\"" | "\"boolean\"" | "\"content_phrase\"" | "\"metadata_value\"" | "\"unknown\""
bool ::= "true" | "false"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" ["\\/bfnrt])*
ws ::= [ \t\n\r]*
'''

FRAME_EXTRACTION_GRAMMAR = r'''
root ::= "{" ws "\"frames\"" ws ":" ws frame_array ws "}"
frame_array ::= "[" ws (frame (ws "," ws frame)*)? ws "]"
frame ::= "{" ws "\"frame_type\"" ws ":" ws string ws "," ws "\"predicate\"" ws ":" ws string ws "," ws "\"arguments\"" ws ":" ws arg_array ws "," ws "\"polarity\"" ws ":" ws string ws "," ws "\"modality\"" ws ":" ws string ws "," ws "\"temporal_text\"" ws ":" ws string ws "," ws "\"evidence_text\"" ws ":" ws string ws "," ws "\"confidence\"" ws ":" ws number ws "}"
arg_array ::= "[" ws (argument (ws "," ws argument)*)? ws "]"
argument ::= "{" ws "\"role\"" ws ":" ws string ws "," ws "\"text\"" ws ":" ws string ws "," ws "\"value_type\"" ws ":" ws string ws "}"
number ::= "-"? [0-9]+ ("." [0-9]+)?
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" ["\\/bfnrt])*
ws ::= [ \t\n\r]*
'''

ANSWER_VERIFICATION_GRAMMAR = r'''
root ::= "{" ws "\"verification\"" ws ":" ws "{" ws "\"entailed\"" ws ":" ws bool ws "," ws "\"answer_type\"" ws ":" ws answer_type ws "," ws "\"answer\"" ws ":" ws string ws "," ws "\"evidence_span\"" ws ":" ws string ws "," ws "\"reason\"" ws ":" ws string ws "}" ws "}"
answer_type ::= "\"person\"" | "\"actor\"" | "\"organization\"" | "\"identifier\"" | "\"url\"" | "\"file_path\"" | "\"count\"" | "\"state\"" | "\"date_time\"" | "\"boolean\"" | "\"content_phrase\"" | "\"metadata_value\"" | "\"unknown\""
bool ::= "true" | "false"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" ["\\/bfnrt])*
ws ::= [ \t\n\r]*
'''

IDENTITY_CANONICALIZATION_GRAMMAR = r'''
root ::= "{" ws "\"canonicalization\"" ws ":" ws "{" ws "\"same_referent\"" ws ":" ws bool ws "," ws "\"answer\"" ws ":" ws string ws "," ws "\"evidence_span\"" ws ":" ws string ws "," ws "\"reason\"" ws ":" ws string ws "}" ws "}"
bool ::= "true" | "false"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" ["\\/bfnrt])*
ws ::= [ \t\n\r]*
'''

QUERY_EVIDENCE_ANSWER_GRAMMAR = r'''
root ::= "{" ws "\"result\"" ws ":" ws "{" ws "\"query_frame\"" ws ":" ws "{" ws "\"target_anchors\"" ws ":" ws string_array ws "," ws "\"requested_relation\"" ws ":" ws string ws "," ws "\"relation_terms\"" ws ":" ws string_array ws "," ws "\"constraints\"" ws ":" ws string_array ws "," ws "\"answer_type\"" ws ":" ws answer_type ws "," ws "\"temporal_scope\"" ws ":" ws string ws "," ws "\"negated\"" ws ":" ws bool ws "," ws "\"aggregation\"" ws ":" ws string ws "," ws "\"requires_evidence\"" ws ":" ws bool ws "}" ws "," ws "\"sufficient_evidence\"" ws ":" ws bool ws "," ws "\"answer_type\"" ws ":" ws answer_type ws "," ws "\"answer\"" ws ":" ws string ws "," ws "\"evidence_span\"" ws ":" ws string ws "," ws "\"reason\"" ws ":" ws string ws "}" ws "}"
answer_type ::= "\"person\"" | "\"actor\"" | "\"organization\"" | "\"identifier\"" | "\"url\"" | "\"file_path\"" | "\"count\"" | "\"state\"" | "\"date_time\"" | "\"boolean\"" | "\"content_phrase\"" | "\"metadata_value\"" | "\"unknown\""
string_array ::= "[" ws (string (ws "," ws string)*)? ws "]"
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
    chunk_frame_call_count: int = 0
    chunk_frame_parsed_count: int = 0
    chunk_frame_accepted_count: int = 0
    verifier_call_count: int = 0
    verifier_parsed_count: int = 0
    verifier_accepted_count: int = 0
    verifier_rejected_count: int = 0
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
            "chunk_frame_call_count": self.chunk_frame_call_count,
            "chunk_frame_parsed_count": self.chunk_frame_parsed_count,
            "chunk_frame_accepted_count": self.chunk_frame_accepted_count,
            "verifier_call_count": self.verifier_call_count,
            "verifier_parsed_count": self.verifier_parsed_count,
            "verifier_accepted_count": self.verifier_accepted_count,
            "verifier_rejected_count": self.verifier_rejected_count,
            "prompt_hashes": self.prompt_hashes or [],
            "response_hashes": self.response_hashes or [],
            "last_plan": self.last_plan,
        }


def deterministic_plan(question: str) -> dict[str, Any]:
    """Compatibility wrapper returning the deterministic query frame as dict."""

    from .query import plan_question

    return plan_question(question).as_dict()


def normalize_model_plan(question: str, model: dict[str, Any] | None, det: dict[str, Any]) -> dict[str, Any] | None:
    if model and model.get("accepted"):
        frame = frame_from_mapping(question, model.get("query_frame") if isinstance(model.get("query_frame"), dict) else model, source="model")
        return {**frame.as_dict(), "accepted": True}
    if det:
        return det
    from .query import plan_question

    return plan_question(question).as_dict()


def build_query_plan_prompt(question: str) -> str:
    surface = {
        "visible_anchors": visible_anchors(question),
        "urls": urls(question),
        "identifiers": identifiers(question),
        "content_tokens": content_tokens(question)[:32],
    }
    return (
        "JSON only. Convert the question into a generic DRT/DSPG query frame; do not answer it. "
        "Use only relation text visible in the question. Do not choose dataset labels, hidden categories, "
        "or source-specific shortcuts. The frame must contain target anchors, requested relation text, "
        "relation terms, constraints, broad answer type, temporal scope, negation, aggregation, and "
        "whether source evidence is required. Surface observations are syntactic hints only; they are "
        "not a semantic answer plan."
        + json.dumps({"question": question, "surface_observations": surface}, ensure_ascii=False)
    )


def call_model_query_plan(question: str, client: LocalModelClient, *, n_predict: int | None = None) -> dict[str, Any]:
    if n_predict is None:
        n_predict = int(os.environ.get("KMD_QUERY_PLAN_N_PREDICT", "128"))
    prompt = build_query_plan_prompt(question)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    cache_path = _cache_path("KMD_QUERY_PLAN_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    start = time.time()
    try:
        parsed = client.complete_json(prompt, n_predict=n_predict, grammar=_optional_grammar(QUERY_FRAME_GRAMMAR))
    except Exception as exc:
        from .query import plan_question

        payload = {
            **plan_question(question).as_dict(),
            "source": "model",
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(QUERY_FRAME_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
        _write_cache(cache_path, payload)
        return payload
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    frame_payload = parsed.get("query_frame") if isinstance(parsed, dict) else None
    if frame_payload is None and isinstance(parsed, dict) and any(key in parsed for key in ["target_anchors", "requested_relation", "answer_type"]):
        frame_payload = parsed
    if not isinstance(frame_payload, dict):
        from .query import plan_question

        payload = {
            **plan_question(question).as_dict(),
            "source": "model",
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(QUERY_FRAME_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
        _write_cache(cache_path, payload)
        return payload
    frame_payload = _repair_query_frame_payload(frame_payload)
    if not _valid_query_frame_payload(frame_payload):
        from .query import plan_question

        payload = {
            **plan_question(question).as_dict(),
            "source": "model",
            "accepted": False,
            "reason": "schema_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(QUERY_FRAME_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
        _write_cache(cache_path, payload)
        return payload
    frame = frame_from_mapping(question, frame_payload, source="model").as_dict()
    payload = {
        **frame,
        "accepted": True,
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "stop_reason": "parsed_json",
        "prompt_hash": prompt_hash,
        "grammar_hash": hashlib.sha256(QUERY_FRAME_GRAMMAR.encode()).hexdigest(),
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    _write_cache(cache_path, payload)
    return payload


def build_evidence_extraction_prompt(question: str, expected_answer_type: str, evidence_items: list[dict[str, str]]) -> str:
    return (
        "JSON only. Answer the question only from the provided raw-text evidence. "
        "Return sufficient_evidence=false and answer='unknown' when the evidence does not state a complete answer. "
        "The answer must be the shortest exact grounded value compatible with the expected answer type, not a "
        "full clause containing the question relation words. "
        "For boolean questions, answer with 'Yes; ...' or 'No; ...' plus a concise grounded reason when the "
        "evidence entails support or denial; a statement that "
        "holds only inside a dream, fiction, belief, quote, allegation, or hypothetical context is not asserted "
        "as a real-world fact, and asserted contradiction can support a no answer. Do not return unknown merely "
        "because the evidence is negative; evidence saying no proof, no support, only a different state/value, "
        "or an explicit contradiction can settle a no answer. Return unknown only when the evidence does not "
        "settle yes or no. "
        "If multiple evidence items support the same discourse referent, use the most complete exact surface form "
        "available in the evidence. Interpret identity, coreference, roles, polarity, modality, and answer selection "
        "yourself from the evidence; do not use outside knowledge."
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
    n_predict: int | None = None,
) -> dict[str, Any]:
    if n_predict is None:
        n_predict = int(os.environ.get("KMD_EVIDENCE_ANSWER_N_PREDICT", "128"))
    prompt = build_evidence_extraction_prompt(question, expected_answer_type, evidence_items)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    cache_path = _cache_path("KMD_EVIDENCE_ANSWER_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    start = time.time()
    try:
        parsed = client.complete_json(prompt, n_predict=n_predict, grammar=_optional_grammar(EVIDENCE_EXTRACTION_GRAMMAR))
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(EVIDENCE_EXTRACTION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    answer = parsed.get("answer") if isinstance(parsed, dict) else None
    if isinstance(answer, str) and isinstance(parsed, dict):
        answer = {
            "sufficient_evidence": parsed.get("sufficient_evidence", True),
            "answer_type": parsed.get("answer_type", expected_answer_type),
            "answer": answer,
            "evidence_span": parsed.get("evidence_span", ""),
        }
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    if not isinstance(answer, dict):
        return {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(EVIDENCE_EXTRACTION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    answer = _repair_evidence_span(answer, evidence_items)
    if not _valid_answer_payload(answer):
        return {
            "accepted": False,
            "reason": "schema_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(EVIDENCE_EXTRACTION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    payload = {
        "accepted": True,
        "sufficient_evidence": bool(answer.get("sufficient_evidence")),
        "answer_type": str(answer.get("answer_type") or "unknown"),
        "answer": str(answer.get("answer") or ""),
        "evidence_span": str(answer.get("evidence_span") or ""),
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": prompt_hash,
        "grammar_hash": hashlib.sha256(EVIDENCE_EXTRACTION_GRAMMAR.encode()).hexdigest(),
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    _write_cache(cache_path, payload)
    return payload


def build_query_evidence_answer_prompt(question: str, evidence_items: list[dict[str, str]]) -> str:
    surface = {
        "visible_anchors": visible_anchors(question),
        "urls": urls(question),
        "identifiers": identifiers(question),
        "content_tokens": content_tokens(question)[:32],
    }
    return (
        "JSON only. Perform a bounded DRT/DSPG question analysis and grounded entailment check. "
        "First convert the question into a generic query_frame. Then answer only if the provided raw-text "
        "evidence entails a complete answer to that frame. Relation words are data from the question and "
        "evidence, not handler names. Reject nearby wrong-type values, sibling record values, URL/identifier "
        "confusion, unsupported claims, inaccessible dream/fiction/belief contexts, and temporal mismatches. "
        "For boolean questions, answer with 'Yes; ...' or 'No; ...' plus a concise grounded reason. Answer no "
        "when the evidence entails denial, contradiction, or only inaccessible scope for the queried proposition; "
        "evidence saying no proof, no support, only a different state/value, or an explicit contradiction can "
        "settle a no answer. Answer unknown only when support and denial are both unsupported. "
        "If evidence links a short mention to a fuller mention of the same discourse referent, answer with the "
        "fuller exact evidence surface. "
        "If evidence is insufficient, return sufficient_evidence=false and answer='unknown'. Copy evidence_span "
        "exactly from one provided evidence item."
        + json.dumps(
            {
                "question": question,
                "surface_observations": surface,
                "evidence": evidence_items,
            },
            ensure_ascii=False,
        )
    )


def call_model_query_evidence_answer(
    question: str,
    evidence_items: list[dict[str, str]],
    client: LocalModelClient,
    *,
    n_predict: int | None = None,
) -> dict[str, Any]:
    if n_predict is None:
        n_predict = int(os.environ.get("KMD_QUERY_EVIDENCE_N_PREDICT", "128"))
    prompt = build_query_evidence_answer_prompt(question, evidence_items)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    cache_path = _cache_path("KMD_QUERY_EVIDENCE_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    start = time.time()
    try:
        parsed = client.complete_json(prompt, n_predict=n_predict, grammar=_optional_grammar(QUERY_EVIDENCE_ANSWER_GRAMMAR))
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(QUERY_EVIDENCE_ANSWER_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    result = parsed.get("result") if isinstance(parsed, dict) else None
    if result is None and isinstance(parsed, dict) and "answer" in parsed:
        result = parsed
    if not isinstance(result, dict):
        return {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(QUERY_EVIDENCE_ANSWER_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    frame_payload = result.get("query_frame") if isinstance(result.get("query_frame"), dict) else {}
    frame_payload = _repair_query_frame_payload(frame_payload)
    result = _repair_evidence_span(result, evidence_items)
    if not _valid_query_frame_payload(frame_payload) or not _valid_answer_payload(result):
        return {
            "accepted": False,
            "reason": "schema_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(QUERY_EVIDENCE_ANSWER_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    frame = frame_from_mapping(question, frame_payload, source="model").as_dict()
    payload = {
        "accepted": True,
        "query_frame": frame,
        "sufficient_evidence": bool(result.get("sufficient_evidence")),
        "answer_type": str(result.get("answer_type") or frame.get("answer_type") or "unknown"),
        "answer": str(result.get("answer") or ""),
        "evidence_span": str(result.get("evidence_span") or ""),
        "reason": str(result.get("reason") or ""),
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": prompt_hash,
        "grammar_hash": hashlib.sha256(QUERY_EVIDENCE_ANSWER_GRAMMAR.encode()).hexdigest(),
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    _write_cache(cache_path, payload)
    return payload


def build_chunk_frame_prompt(chunk_text: str, *, rel_path: str = "") -> str:
    return (
        "JSON only. Extract generic DRT/DSPG discourse frames from this raw text chunk. "
        "Do not answer questions. Do not use dataset labels or handler names. "
        "Represent only grounded source statements as frames. Use arbitrary relation/predicate words as data, "
        "not categories. For each frame include frame_type, predicate, arguments with generic roles, polarity, "
        "modality, temporal_text, exact evidence_text copied from the chunk, and confidence. "
        "Use modality values such as asserted, reported, belief, quote, dream, fiction, hypothetical, uncertain "
        "when the text itself marks those scopes."
        + json.dumps({"source": rel_path, "chunk": chunk_text[:2400]}, ensure_ascii=False)
    )


def call_model_chunk_frames(
    chunk_text: str,
    client: LocalModelClient,
    *,
    rel_path: str = "",
    n_predict: int | None = None,
) -> dict[str, Any]:
    if n_predict is None:
        n_predict = int(os.environ.get("KMD_CHUNK_FRAME_N_PREDICT", "192"))
    prompt = build_chunk_frame_prompt(chunk_text, rel_path=rel_path)
    start = time.time()
    try:
        parsed = client.complete_json(prompt, n_predict=n_predict, grammar=_optional_grammar(FRAME_EXTRACTION_GRAMMAR))
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "grammar_hash": hashlib.sha256(FRAME_EXTRACTION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    frames = parsed.get("frames") if isinstance(parsed, dict) else None
    if frames is None and isinstance(parsed, dict):
        frames = parsed.get("items")
    if frames is None and isinstance(parsed, dict) and any(key in parsed for key in ["frame_type", "predicate", "evidence_text"]):
        frames = [parsed]
    if not isinstance(frames, list):
        return {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "grammar_hash": hashlib.sha256(FRAME_EXTRACTION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    grounded: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        evidence_text = str(frame.get("evidence_text") or "").strip()
        predicate = str(frame.get("predicate") or "").strip()
        if not evidence_text or evidence_text not in chunk_text or not predicate:
            continue
        arguments = frame.get("arguments")
        if isinstance(arguments, dict):
            arguments = [
                {"role": str(role), "text": str(text), "value_type": "unknown"}
                for role, text in arguments.items()
            ]
        grounded.append(
            {
                "frame_type": str(frame.get("frame_type") or "relation"),
                "predicate": predicate,
                "arguments": arguments if isinstance(arguments, list) else [],
                "polarity": str(frame.get("polarity") or "positive"),
                "modality": str(frame.get("modality") or "asserted"),
                "temporal_text": str(frame.get("temporal_text") or ""),
                "evidence_text": evidence_text,
                "confidence": _coerce_confidence(frame.get("confidence")),
            }
        )
    return {
        "accepted": True,
        "frames": grounded,
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "grammar_hash": hashlib.sha256(FRAME_EXTRACTION_GRAMMAR.encode()).hexdigest(),
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
    }


def build_answer_verification_prompt(
    question: str,
    query_frame: dict[str, Any],
    candidate_answer: str,
    evidence_items: list[dict[str, str]],
    discourse_frames: list[dict[str, Any]],
) -> str:
    return (
        "JSON only. Verify whether the candidate answer is entailed by the bounded raw-text evidence and "
        "generic discourse frames. Reject wrong type, wrong scope, wrong temporal state, wrong relation role, "
        "nearby identifier or URL confusion, unsupported inference, and false positives. "
        "Return the shortest exact answer value entailed by the evidence, not a full clause containing the "
        "question relation words. "
        "If the candidate is a short mention and the evidence entails a fuller exact surface for the same "
        "discourse referent, return the fuller surface as answer. "
        "Do not use outside knowledge. If evidence is insufficient, return entailed=false and answer='unknown'."
        + json.dumps(
            {
                "question": question,
                "query_frame": query_frame,
                "candidate_answer": candidate_answer,
                "evidence": evidence_items,
                "discourse_frames": discourse_frames,
            },
            ensure_ascii=False,
        )
    )


def call_model_answer_verification(
    question: str,
    query_frame: dict[str, Any],
    candidate_answer: str,
    evidence_items: list[dict[str, str]],
    discourse_frames: list[dict[str, Any]],
    client: LocalModelClient,
    *,
    n_predict: int | None = None,
) -> dict[str, Any]:
    if n_predict is None:
        n_predict = int(os.environ.get("KMD_VERIFIER_N_PREDICT", "128"))
    prompt = build_answer_verification_prompt(question, query_frame, candidate_answer, evidence_items, discourse_frames)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    cache_path = _cache_path("KMD_VERIFIER_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    start = time.time()
    try:
        parsed = client.complete_json(prompt, n_predict=n_predict, grammar=_optional_grammar(ANSWER_VERIFICATION_GRAMMAR))
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(ANSWER_VERIFICATION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    verification = parsed.get("verification") if isinstance(parsed, dict) else None
    if verification is None and isinstance(parsed, dict) and any(key in parsed for key in ["entailed", "answer"]):
        verification = parsed
    if not isinstance(verification, dict):
        return {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(ANSWER_VERIFICATION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    payload = {
        "accepted": True,
        "entailed": bool(verification.get("entailed")),
        "answer_type": str(verification.get("answer_type") or "unknown"),
        "answer": str(verification.get("answer") or ""),
        "evidence_span": str(verification.get("evidence_span") or ""),
        "reason": str(verification.get("reason") or ""),
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": prompt_hash,
        "grammar_hash": hashlib.sha256(ANSWER_VERIFICATION_GRAMMAR.encode()).hexdigest(),
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    _write_cache(cache_path, payload)
    return payload


def build_identity_canonicalization_prompt(
    question: str,
    candidate_answer: str,
    fuller_candidates: list[str],
    evidence_items: list[dict[str, str]],
) -> str:
    return (
        "JSON only. Decide whether a short candidate answer and one fuller candidate surface refer to the same "
        "discourse referent in the provided evidence for this question. The fuller answer is allowed only when "
        "the evidence entails the identity/coreference; otherwise keep the original candidate answer. Treat a "
        "first-name-only answer plus a single non-conflicting fuller name sharing that first name as an identity "
        "hypothesis to accept when the surrounding evidence describes the same role, event, state, or record. Do "
        "not use outside knowledge."
        + json.dumps(
            {
                "question": question,
                "candidate_answer": candidate_answer,
                "fuller_candidates": fuller_candidates,
                "evidence": evidence_items,
            },
            ensure_ascii=False,
        )
    )


def call_model_identity_canonicalization(
    question: str,
    candidate_answer: str,
    fuller_candidates: list[str],
    evidence_items: list[dict[str, str]],
    client: LocalModelClient,
    *,
    n_predict: int | None = None,
) -> dict[str, Any]:
    if n_predict is None:
        n_predict = int(os.environ.get("KMD_IDENTITY_N_PREDICT", "96"))
    prompt = build_identity_canonicalization_prompt(question, candidate_answer, fuller_candidates, evidence_items)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    cache_path = _cache_path("KMD_IDENTITY_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    start = time.time()
    try:
        parsed = client.complete_json(prompt, n_predict=n_predict, grammar=_optional_grammar(IDENTITY_CANONICALIZATION_GRAMMAR))
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(IDENTITY_CANONICALIZATION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    result = parsed.get("canonicalization") if isinstance(parsed, dict) else None
    if result is None and isinstance(parsed, dict) and any(key in parsed for key in ["same_referent", "answer"]):
        result = parsed
    if result is None and isinstance(parsed, dict) and any(key in parsed for key in ["decision", "justified_answer", "canonical_answer"]):
        decision = str(parsed.get("decision") or "").strip().lower()
        answer = str(parsed.get("justified_answer") or parsed.get("canonical_answer") or parsed.get("answer") or candidate_answer)
        result = {
            "same_referent": decision in {"accept", "accepted", "true", "yes", "same", "same_referent"},
            "answer": answer,
            "evidence_span": str(parsed.get("evidence_span") or ""),
            "reason": str(parsed.get("rationale") or parsed.get("reason") or ""),
        }
    if not isinstance(result, dict):
        return {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(IDENTITY_CANONICALIZATION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    answer = str(result.get("answer") or "")
    span = str(result.get("evidence_span") or "")
    if not answer or (answer != candidate_answer and answer not in fuller_candidates):
        return {
            "accepted": False,
            "reason": "schema_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(IDENTITY_CANONICALIZATION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    if not span and answer != candidate_answer:
        for item in evidence_items:
            text = str(item.get("text") or "")
            if answer in text:
                span = answer
                break
    if answer != candidate_answer and not any((span or answer) in str(item.get("text") or "") for item in evidence_items):
        return {
            "accepted": False,
            "reason": "ungrounded_answer",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": hashlib.sha256(IDENTITY_CANONICALIZATION_GRAMMAR.encode()).hexdigest(),
            "elapsed": round(time.time() - start, 3),
        }
    payload = {
        "accepted": True,
        "same_referent": bool(result.get("same_referent")),
        "answer": answer,
        "evidence_span": span,
        "reason": str(result.get("reason") or ""),
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": prompt_hash,
        "grammar_hash": hashlib.sha256(IDENTITY_CANONICALIZATION_GRAMMAR.encode()).hexdigest(),
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    _write_cache(cache_path, payload)
    return payload
