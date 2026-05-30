"""Optional local-model helpers for generic query frames.

Model use is isolated and local-only.  The planner asks for a generic
relation/query frame, never an external label or hardcoded semantic intent.
Evidence answering is constrained to bounded raw-text snippets and is validated
against source grounding before it can leave the engine.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import LocalModelClient, LocalModelJSONError
from .extractors import identifiers, urls
from .query import QueryFrame, frame_from_mapping, visible_anchors
from .text import content_tokens, normalize


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
DRS_CONTEXT_KINDS = {
    "asserted",
    "negated",
    "conditional_antecedent",
    "conditional_consequent",
    "reported",
    "quoted",
    "believed",
    "possible",
    "uncertain",
    "hypothetical",
    "fictional",
    "dreamed",
}
DRS_POLARITIES = {"positive", "negative", "unknown"}
DRS_IDENTITY_STATUSES = {"accepted", "candidate", "rejected", "ambiguous"}

PROMPT_VERSION = "kmd-drt-2026-05-28-v35"
CHUNK_FRAME_SCHEMA_VERSION = "chunk-frames-v5"
CHUNK_DRS_SCHEMA_VERSION = "chunk-drs-v2"
CHUNK_DRS_STAGED_FALLBACK_POLICY = "retry-invalid-json-schema-grounding-staged-temporal-scope-v3"
CHUNK_DRS_GROUNDING_REPAIR_POLICY = "model-label-value-escaped-evidence-span-v3"
CHUNK_DRS_IDENTITY_PROVENANCE_POLICY = "identity-evidence-bilateral-surface-v1"
CHUNK_DRS_TEMPORAL_PROVENANCE_POLICY = "condition-stage-declared-temporal-records-v2"
CHUNK_DRS_SPARSE_RETRY_POLICY = "retry-validated-sparse-drs-staged-v1"
CHUNK_DRS_STRUCTURE_VALIDATION_POLICY = "acyclic-box-condition-arguments-v1"
CHUNK_DRS_BOX_COMPLETION_POLICY = "model-complete-missing-box-declarations-v1"
CHUNK_DRS_SOURCE_SPAN_POLICY = "chunk-drs-delimiter-source-span-enum-v2"
CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY = "stage1-source-span-evidence-enum-v1"
CHUNK_DRS_SKELETON_ID_POLICY = "stage1-stable-id-enums-v1"
CHUNK_DRS_MONOLITHIC_ID_POLICY = "monolithic-stable-id-enums-v1"
CHUNK_DRS_COMPACT_UNDERCOVERAGE_POLICY = "retry-delimiter-rich-low-condition-density-v1"
CHUNK_DRS_STAGED_RETRY_DIAGNOSTICS_POLICY = "record-non-improving-staged-retry-v1"
CHUNK_DRS_STAGE_FAILURE_CACHE_POLICY = "cache-invalid-json-stage-failures-v1"
QUERY_DRS_SCHEMA_VERSION = "query-drs-v3"
QUERY_DRS_VALIDATION_POLICY = "strict-query-drs-version-question-evidence-repair-v8"
QUERY_DRS_ARRAY_CAP_POLICY = "reserved_output_tokens_div_96_4_8-v1"
QUERY_FRAME_SCHEMA_VERSION = "query-frame-v4"
ANSWER_SCHEMA_VERSION = "answer-v4"

QUERY_FRAME_GRAMMAR = r'''
root ::= "{" ws "\"query_frame\"" ws ":" ws "{" ws "\"target_anchors\"" ws ":" ws string_array ws "," ws "\"answer_variables\"" ws ":" ws string_array ws "," ws "\"requested_relation\"" ws ":" ws string ws "," ws "\"relation_terms\"" ws ":" ws string_array ws "," ws "\"constraints\"" ws ":" ws string_array ws "," ws "\"scope_requirements\"" ws ":" ws string_array ws "," ws "\"modality_requirements\"" ws ":" ws string_array ws "," ws "\"answer_type\"" ws ":" ws answer_type ws "," ws "\"temporal_scope\"" ws ":" ws string ws "," ws "\"negated\"" ws ":" ws bool ws "," ws "\"aggregation\"" ws ":" ws string ws "," ws "\"requires_evidence\"" ws ":" ws bool ws "}" ws "}"
answer_type ::= "\"person\"" | "\"actor\"" | "\"organization\"" | "\"identifier\"" | "\"url\"" | "\"file_path\"" | "\"count\"" | "\"state\"" | "\"date_time\"" | "\"boolean\"" | "\"content_phrase\"" | "\"metadata_value\"" | "\"unknown\""
string_array ::= "[" ws (string (ws "," ws string)*)? ws "]"
bool ::= "true" | "false"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" ["\\/bfnrt])*
ws ::= [ \t\n\r]*
'''


def _optional_grammar(grammar: str) -> str | None:
    return None if os.environ.get("KMD_LOCAL_MODEL_GRAMMAR", "1").strip().lower() in {"0", "false", "no", "off"} else grammar


def _json_schema_enabled() -> bool:
    return os.environ.get("KMD_LOCAL_MODEL_JSON_SCHEMA", "1").strip().lower() not in {"0", "false", "no", "off"}


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _client_fingerprint(client: LocalModelClient | None) -> dict[str, Any]:
    if client is None:
        return {}
    method = getattr(client, "cache_fingerprint", None)
    if callable(method):
        try:
            payload = method()
        except Exception as exc:
            return {"endpoint": getattr(client, "endpoint", ""), "metadata_error": f"{type(exc).__name__}: {exc}"}
        return payload if isinstance(payload, dict) else {}
    return {
        "endpoint": getattr(client, "endpoint", ""),
        "timeout_seconds": getattr(client, "timeout_seconds", ""),
        "seed": os.environ.get("KMD_LOCAL_MODEL_SEED", "1778779265"),
    }


def _client_context_size(client: LocalModelClient | None) -> int:
    if client is None:
        return 0
    method = getattr(client, "context_size", None)
    if callable(method):
        try:
            return max(0, int(method()))
        except Exception:
            return 0
    return 0


def default_chunk_frame_n_predict(client: LocalModelClient | None = None) -> int:
    configured = os.environ.get("KMD_CHUNK_FRAME_N_PREDICT")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    context_size = _client_context_size(client)
    if context_size > 0:
        return max(192, min(1024, context_size // 32))
    return 192


def default_chunk_drs_n_predict(client: LocalModelClient | None = None) -> int:
    configured = os.environ.get("KMD_CHUNK_DRS_N_PREDICT")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    context_size = _client_context_size(client)
    if context_size > 0:
        return max(384, min(1536, context_size // 24))
    return 384


def default_query_drs_n_predict(client: LocalModelClient | None = None) -> int:
    configured = os.environ.get("KMD_QUERY_DRS_N_PREDICT")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    context_size = _client_context_size(client)
    if context_size > 0:
        return max(256, min(768, context_size // 48))
    return 256


ANSWER_TYPE_ALIASES = {
    "amount": "count",
    "contact": "content_phrase",
    "contact_info": "content_phrase",
    "date": "date_time",
    "datetime": "date_time",
    "definition": "content_phrase",
    "email": "content_phrase",
    "entity": "content_phrase",
    "integer": "count",
    "link": "url",
    "location": "content_phrase",
    "name": "content_phrase",
    "number": "count",
    "object": "content_phrase",
    "phone": "content_phrase",
    "phone_number": "content_phrase",
    "phrase": "content_phrase",
    "place": "content_phrase",
    "quantity": "count",
    "string": "content_phrase",
    "text": "content_phrase",
    "uri": "url",
    "word": "content_phrase",
    "yes_no": "boolean",
}


def _normalize_answer_type(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    normalized = ANSWER_TYPE_ALIASES.get(text, text)
    if normalized in ANSWER_TYPES:
        return normalized
    return default if default in ANSWER_TYPES else "unknown"


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


def _cache_material(stage: str, prompt: str, client: LocalModelClient | None, settings: dict[str, Any] | None = None) -> str:
    payload = {
        "stage": stage,
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt,
        "model_endpoint": getattr(client, "endpoint", os.environ.get("KMD_LOCAL_MODEL_ENDPOINT", "")),
        "model_timeout": getattr(client, "timeout_seconds", os.environ.get("KMD_LOCAL_MODEL_TIMEOUT", "")),
        "model_identity": os.environ.get("KMD_LOCAL_MODEL_ID", ""),
        "seed": os.environ.get("KMD_LOCAL_MODEL_SEED", "1778779265"),
        "model_fingerprint": _client_fingerprint(client),
        "settings": settings or {},
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _cache_hash(stage: str, prompt: str, client: LocalModelClient | None, settings: dict[str, Any] | None = None) -> str:
    return hashlib.sha256(_cache_material(stage, prompt, client, settings).encode("utf-8")).hexdigest()


def _grammar_hash(grammar: str, schema_version: str) -> str:
    return hashlib.sha256((grammar + schema_version).encode()).hexdigest()


def _json_schema_hash(schema: dict[str, Any] | None, schema_version: str) -> str:
    return hashlib.sha256(json.dumps({"schema": schema or {}, "version": schema_version}, sort_keys=True).encode()).hexdigest()


def _constraint_settings(grammar: str, json_schema: dict[str, Any] | None, schema_version: str) -> dict[str, Any]:
    use_json_schema = bool(json_schema) and _json_schema_enabled()
    return {
        "constraint_mode": "json_schema" if use_json_schema else ("gbnf" if _optional_grammar(grammar) else "none"),
        "grammar_hash": _grammar_hash(grammar, schema_version),
        "json_schema_hash": _json_schema_hash(json_schema, schema_version) if json_schema else "",
    }


def _complete_structured(
    client: LocalModelClient,
    prompt: str,
    *,
    n_predict: int,
    grammar: str,
    json_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    if json_schema and _json_schema_enabled():
        try:
            return client.complete_json(prompt, n_predict=n_predict, json_schema=json_schema)
        except TypeError:
            pass
    return client.complete_json(prompt, n_predict=n_predict, grammar=_optional_grammar(grammar))


def _cache_path(env_var: str, prompt_hash: str) -> Path | None:
    cache_dir = os.environ.get(env_var, "").strip()
    if not cache_dir:
        cache_name = env_var.lower()
        if cache_name.startswith("kmd_"):
            cache_name = cache_name[4:]
        cache_dir = str(Path.home() / ".cache" / "knowmoredirt" / cache_name)
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
        "answer_variables",
        "requested_relation",
        "relation_terms",
        "constraints",
        "scope_requirements",
        "modality_requirements",
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
        and _is_string_list(value.get("answer_variables"))
        and isinstance(value.get("requested_relation"), str)
        and _is_string_list(value.get("relation_terms"))
        and _is_string_list(value.get("constraints"))
        and _is_string_list(value.get("scope_requirements"))
        and _is_string_list(value.get("modality_requirements"))
        and str(value.get("answer_type")) in ANSWER_TYPES
        and isinstance(value.get("temporal_scope"), str)
        and isinstance(value.get("negated"), bool)
        and isinstance(value.get("aggregation"), str)
        and isinstance(value.get("requires_evidence"), bool)
    )


def _query_grounded_terms(items: list[str], question: str) -> list[str]:
    if not question:
        return items
    question_norm = normalize(question)
    question_tokens = set(content_tokens(question))
    grounded: list[str] = []
    for item in items:
        item_text = str(item or "").strip()
        item_norm = normalize(item_text).replace("_", " ")
        item_tokens = [token for token in content_tokens(item_norm) if token not in {"of", "for", "to", "in", "on"}]
        if not item_tokens:
            continue
        if item_norm in question_norm or all(token in question_tokens for token in item_tokens):
            grounded.append(item_text)
    return list(dict.fromkeys(grounded))


def _repair_query_frame_payload(value: Any, question: str = "") -> Any:
    if not isinstance(value, dict):
        return value
    repaired = dict(value)
    if "target_anchors" not in repaired and "target_anchor" in repaired:
        anchor = repaired.get("target_anchor")
        repaired["target_anchors"] = [str(anchor)] if str(anchor or "").strip() else []
    if "requested_relation" not in repaired and "requested_relations" in repaired:
        raw = repaired.get("requested_relations")
        if isinstance(raw, list):
            repaired["requested_relation"] = " ".join(str(item) for item in raw if str(item).strip())
        else:
            repaired["requested_relation"] = str(raw or "")
    if "requested_relation" not in repaired:
        for key in ["relation", "predicate"]:
            if str(repaired.get(key) or "").strip():
                repaired["requested_relation"] = str(repaired.get(key) or "")
                break
    if "target_anchors" not in repaired:
        anchors = []
        for key in ["subject", "target", "entity", "topic", "arg1", "object", "arg2"]:
            raw_anchor = str(repaired.get(key) or "").strip()
            if not raw_anchor or raw_anchor.lower() in {"who", "what", "where", "when", "which", "answer", "value", "location", "person", "unknown"}:
                continue
            anchors.append(raw_anchor)
        if anchors:
            repaired["target_anchors"] = list(dict.fromkeys(anchors))
    if "answer_type" not in repaired and "broad_answer_type" in repaired:
        repaired["answer_type"] = str(repaired.get("broad_answer_type") or "")
    if "answer_variables" not in repaired:
        raw_variable = repaired.get("answer_variable") or repaired.get("variable") or repaired.get("slot")
        repaired["answer_variables"] = [str(raw_variable)] if str(raw_variable or "").strip() else []
    if "answer_type" in repaired:
        repaired["answer_type"] = _normalize_answer_type(repaired.get("answer_type"), "unknown")
    if "negated" not in repaired and "negation" in repaired:
        repaired["negated"] = bool(repaired.get("negation"))
    if "requires_evidence" not in repaired and "source_evidence_required" in repaired:
        repaired["requires_evidence"] = bool(repaired.get("source_evidence_required"))
    if "aggregation" in repaired and isinstance(repaired.get("aggregation"), bool):
        repaired["aggregation"] = "count" if repaired.get("aggregation") else ""
    for key in ["target_anchors", "answer_variables", "relation_terms", "constraints", "scope_requirements", "modality_requirements"]:
        if key not in repaired:
            repaired[key] = []
        elif isinstance(repaired.get(key), dict):
            repaired[key] = [
                str(item)
                for pair in repaired[key].items()
                for item in pair
                if str(item).strip()
            ]
        elif not isinstance(repaired.get(key), list):
            repaired[key] = []
    for key in ["requested_relation", "answer_type", "temporal_scope", "aggregation"]:
        if key not in repaired:
            repaired[key] = ""
    if not repaired.get("answer_type"):
        repaired["answer_type"] = "unknown"
    for key in ["relation_terms", "constraints"]:
        repaired[key] = _query_grounded_terms([str(item) for item in repaired.get(key, [])], question)
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


def _repair_answer_payload(value: Any, default_answer_type: str = "unknown") -> Any:
    if not isinstance(value, dict):
        return value
    repaired = dict(value)
    nested_answer = repaired.get("answer")
    if isinstance(nested_answer, dict):
        if "sufficient_evidence" not in repaired and isinstance(nested_answer.get("sufficient_evidence"), bool):
            repaired["sufficient_evidence"] = nested_answer.get("sufficient_evidence")
        if "evidence_span" not in repaired and isinstance(nested_answer.get("evidence_span"), str):
            repaired["evidence_span"] = nested_answer.get("evidence_span")
        if "answer_type" not in repaired and isinstance(nested_answer.get("answer_type"), str):
            repaired["answer_type"] = nested_answer.get("answer_type")
        scalar_answer = ""
        for key, item in nested_answer.items():
            if key in {"sufficient_evidence", "evidence_span", "answer_type", "reason", "rationale"}:
                continue
            if isinstance(item, (str, int, float, bool)) and str(item).strip():
                scalar_answer = str(item)
                break
        repaired["answer"] = scalar_answer
    if "sufficient_evidence" not in repaired:
        answer = str(repaired.get("answer") or "").strip()
        repaired["sufficient_evidence"] = bool(answer and answer.lower() != "unknown")
    repaired["answer_type"] = _normalize_answer_type(repaired.get("answer_type"), default_answer_type)
    if "answer" not in repaired:
        repaired["answer"] = ""
    if "evidence_span" not in repaired:
        repaired["evidence_span"] = ""
    return repaired


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
frame ::= "{" ws "\"frame_type\"" ws ":" ws string ws "," ws "\"predicate\"" ws ":" ws string ws "," ws "\"arguments\"" ws ":" ws arg_array ws "," ws "\"identity_hypotheses\"" ws ":" ws identity_array ws "," ws "\"polarity\"" ws ":" ws string ws "," ws "\"modality\"" ws ":" ws string ws "," ws "\"context_holder\"" ws ":" ws string ws "," ws "\"temporal_text\"" ws ":" ws string ws "," ws "\"evidence_text\"" ws ":" ws string ws "," ws "\"confidence\"" ws ":" ws number ws "}"
arg_array ::= "[" ws (argument (ws "," ws argument)*)? ws "]"
argument ::= "{" ws "\"role\"" ws ":" ws string ws "," ws "\"text\"" ws ":" ws string ws "," ws "\"value_type\"" ws ":" ws string ws "}"
identity_array ::= "[" ws (identity (ws "," ws identity)*)? ws "]"
identity ::= "{" ws "\"left_text\"" ws ":" ws string ws "," ws "\"right_text\"" ws ":" ws string ws "," ws "\"relation\"" ws ":" ws string ws "," ws "\"evidence_text\"" ws ":" ws string ws "," ws "\"confidence\"" ws ":" ws number ws "}"
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

ANSWER_CANONICALIZATION_GRAMMAR = r'''
root ::= "{" ws "\"canonical_answer\"" ws ":" ws "{" ws "\"answer\"" ws ":" ws string ws "," ws "\"evidence_span\"" ws ":" ws string ws "," ws "\"reason\"" ws ":" ws string ws "}" ws "}"
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
root ::= "{" ws "\"result\"" ws ":" ws "{" ws "\"query_frame\"" ws ":" ws "{" ws "\"target_anchors\"" ws ":" ws string_array ws "," ws "\"answer_variables\"" ws ":" ws string_array ws "," ws "\"requested_relation\"" ws ":" ws string ws "," ws "\"relation_terms\"" ws ":" ws string_array ws "," ws "\"constraints\"" ws ":" ws string_array ws "," ws "\"scope_requirements\"" ws ":" ws string_array ws "," ws "\"modality_requirements\"" ws ":" ws string_array ws "," ws "\"answer_type\"" ws ":" ws answer_type ws "," ws "\"temporal_scope\"" ws ":" ws string ws "," ws "\"negated\"" ws ":" ws bool ws "," ws "\"aggregation\"" ws ":" ws string ws "," ws "\"requires_evidence\"" ws ":" ws bool ws "}" ws "," ws "\"sufficient_evidence\"" ws ":" ws bool ws "," ws "\"answer_type\"" ws ":" ws answer_type ws "," ws "\"answer\"" ws ":" ws string ws "," ws "\"evidence_span\"" ws ":" ws string ws "," ws "\"reason\"" ws ":" ws string ws "}" ws "}"
answer_type ::= "\"person\"" | "\"actor\"" | "\"organization\"" | "\"identifier\"" | "\"url\"" | "\"file_path\"" | "\"count\"" | "\"state\"" | "\"date_time\"" | "\"boolean\"" | "\"content_phrase\"" | "\"metadata_value\"" | "\"unknown\""
string_array ::= "[" ws (string (ws "," ws string)*)? ws "]"
bool ::= "true" | "false"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" ["\\/bfnrt])*
ws ::= [ \t\n\r]*
'''


def _schema_obj(required: list[str], props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": required, "properties": props}


def _schema_array(item: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": item}


def _schema_enum(values: set[str]) -> dict[str, Any]:
    return {"type": "string", "enum": sorted(values)}


STRING_SCHEMA = {"type": "string"}
BOOL_SCHEMA = {"type": "boolean"}
NUMBER_SCHEMA = {"type": "number"}
ANSWER_TYPE_SCHEMA = _schema_enum(ANSWER_TYPES)
STRING_ARRAY_SCHEMA = _schema_array(STRING_SCHEMA)

QUERY_FRAME_JSON_SCHEMA = _schema_obj(
    ["query_frame"],
    {
        "query_frame": _schema_obj(
            [
                "target_anchors",
                "answer_variables",
                "requested_relation",
                "relation_terms",
                "constraints",
                "scope_requirements",
                "modality_requirements",
                "answer_type",
                "temporal_scope",
                "negated",
                "aggregation",
                "requires_evidence",
            ],
            {
                "target_anchors": STRING_ARRAY_SCHEMA,
                "answer_variables": STRING_ARRAY_SCHEMA,
                "requested_relation": STRING_SCHEMA,
                "relation_terms": STRING_ARRAY_SCHEMA,
                "constraints": STRING_ARRAY_SCHEMA,
                "scope_requirements": STRING_ARRAY_SCHEMA,
                "modality_requirements": STRING_ARRAY_SCHEMA,
                "answer_type": ANSWER_TYPE_SCHEMA,
                "temporal_scope": STRING_SCHEMA,
                "negated": BOOL_SCHEMA,
                "aggregation": STRING_SCHEMA,
                "requires_evidence": BOOL_SCHEMA,
            },
        )
    },
)

ANSWER_JSON_SCHEMA = _schema_obj(
    ["answer"],
    {
        "answer": _schema_obj(
            ["sufficient_evidence", "answer_type", "answer", "evidence_span"],
            {
                "sufficient_evidence": BOOL_SCHEMA,
                "answer_type": ANSWER_TYPE_SCHEMA,
                "answer": STRING_SCHEMA,
                "evidence_span": STRING_SCHEMA,
            },
        )
    },
)

QUERY_EVIDENCE_ANSWER_JSON_SCHEMA = _schema_obj(
    ["result"],
    {
        "result": _schema_obj(
            ["query_frame", "sufficient_evidence", "answer_type", "answer", "evidence_span", "reason"],
            {
                "query_frame": QUERY_FRAME_JSON_SCHEMA["properties"]["query_frame"],
                "sufficient_evidence": BOOL_SCHEMA,
                "answer_type": ANSWER_TYPE_SCHEMA,
                "answer": STRING_SCHEMA,
                "evidence_span": STRING_SCHEMA,
                "reason": STRING_SCHEMA,
            },
        )
    },
)

FRAME_JSON_SCHEMA = _schema_obj(
    ["frames"],
    {
        "frames": _schema_array(
            _schema_obj(
                [
                    "frame_type",
                    "predicate",
                    "arguments",
                    "identity_hypotheses",
                    "polarity",
                    "modality",
                    "context_holder",
                    "temporal_text",
                    "evidence_text",
                    "confidence",
                ],
                {
                    "frame_type": STRING_SCHEMA,
                    "predicate": STRING_SCHEMA,
                    "arguments": _schema_array(
                        _schema_obj(
                            ["role", "text", "value_type"],
                            {"role": STRING_SCHEMA, "text": STRING_SCHEMA, "value_type": STRING_SCHEMA},
                        )
                    ),
                    "identity_hypotheses": _schema_array(
                        _schema_obj(
                            ["left_text", "right_text", "relation", "evidence_text", "confidence"],
                            {
                                "left_text": STRING_SCHEMA,
                                "right_text": STRING_SCHEMA,
                                "relation": STRING_SCHEMA,
                                "evidence_text": STRING_SCHEMA,
                                "confidence": NUMBER_SCHEMA,
                            },
                        )
                    ),
                    "polarity": STRING_SCHEMA,
                    "modality": STRING_SCHEMA,
                    "context_holder": STRING_SCHEMA,
                    "temporal_text": STRING_SCHEMA,
                    "evidence_text": STRING_SCHEMA,
                    "confidence": NUMBER_SCHEMA,
                },
            )
        )
    },
)

DRS_ARGUMENT_JSON_SCHEMA = _schema_obj(
    ["role", "target_kind", "target_id", "value", "value_type", "evidence_text"],
    {
        "role": STRING_SCHEMA,
        "target_kind": _schema_enum({"referent", "box", "condition", "literal", "unknown"}),
        "target_id": STRING_SCHEMA,
        "value": STRING_SCHEMA,
        "value_type": STRING_SCHEMA,
        "evidence_text": STRING_SCHEMA,
    },
)

QUERY_VARIABLE_JSON_SCHEMA = _schema_obj(
    ["id", "label", "answer_type", "evidence_text"],
    {
        "id": STRING_SCHEMA,
        "label": STRING_SCHEMA,
        "answer_type": ANSWER_TYPE_SCHEMA,
        "evidence_text": STRING_SCHEMA,
    },
)

QUERY_DRS_ARGUMENT_JSON_SCHEMA = _schema_obj(
    ["role", "target_kind", "target_id", "value", "value_type", "evidence_text"],
    {
        "role": STRING_SCHEMA,
        "target_kind": _schema_enum({"answer_variable", "referent", "box", "condition", "temporal", "literal", "unknown"}),
        "target_id": STRING_SCHEMA,
        "value": STRING_SCHEMA,
        "value_type": STRING_SCHEMA,
        "evidence_text": STRING_SCHEMA,
    },
)

QUERY_DRS_CONDITION_JSON_SCHEMA = _schema_obj(
    ["id", "predicate", "box_id", "polarity", "modality", "temporal_id", "arguments", "evidence_text"],
    {
        "id": STRING_SCHEMA,
        "predicate": STRING_SCHEMA,
        "box_id": STRING_SCHEMA,
        "polarity": _schema_enum(DRS_POLARITIES),
        "modality": _schema_enum(DRS_CONTEXT_KINDS),
        "temporal_id": STRING_SCHEMA,
        "arguments": _schema_array(QUERY_DRS_ARGUMENT_JSON_SCHEMA),
        "evidence_text": STRING_SCHEMA,
    },
)

DRS_REFERENT_JSON_SCHEMA = _schema_obj(
    ["id", "label", "kind", "evidence_text"],
    {
        "id": STRING_SCHEMA,
        "label": STRING_SCHEMA,
        "kind": STRING_SCHEMA,
        "evidence_text": STRING_SCHEMA,
    },
)

DRS_BOX_JSON_SCHEMA = _schema_obj(
    ["id", "kind", "parent_id", "holder_referent_id", "evidence_text"],
    {
        "id": STRING_SCHEMA,
        "kind": _schema_enum(DRS_CONTEXT_KINDS),
        "parent_id": STRING_SCHEMA,
        "holder_referent_id": STRING_SCHEMA,
        "evidence_text": STRING_SCHEMA,
    },
)

DRS_TEMPORAL_JSON_SCHEMA = _schema_obj(
    ["id", "value", "value_type", "evidence_text"],
    {
        "id": STRING_SCHEMA,
        "value": STRING_SCHEMA,
        "value_type": STRING_SCHEMA,
        "evidence_text": STRING_SCHEMA,
    },
)

DRS_CONDITION_JSON_SCHEMA = _schema_obj(
    ["id", "predicate", "box_id", "polarity", "modality", "temporal_id", "arguments", "evidence_text"],
    {
        "id": STRING_SCHEMA,
        "predicate": STRING_SCHEMA,
        "box_id": STRING_SCHEMA,
        "polarity": _schema_enum(DRS_POLARITIES),
        "modality": _schema_enum(DRS_CONTEXT_KINDS),
        "temporal_id": STRING_SCHEMA,
        "arguments": _schema_array(DRS_ARGUMENT_JSON_SCHEMA),
        "evidence_text": STRING_SCHEMA,
    },
)

DRS_IDENTITY_JSON_SCHEMA = _schema_obj(
    ["left_referent_id", "right_referent_id", "status", "evidence_text", "confidence"],
    {
        "left_referent_id": STRING_SCHEMA,
        "right_referent_id": STRING_SCHEMA,
        "status": _schema_enum(DRS_IDENTITY_STATUSES),
        "evidence_text": STRING_SCHEMA,
        "confidence": NUMBER_SCHEMA,
    },
)

DRS_JSON_SCHEMA = _schema_obj(
    ["drs"],
    {
        "drs": _schema_obj(
            [
                "schema_version",
                "source_id",
                "referents",
                "boxes",
                "conditions",
                "identity_hypotheses",
                "temporal_records",
                "evidence_spans",
                "semantic_notes",
            ],
            {
                "schema_version": STRING_SCHEMA,
                "source_id": STRING_SCHEMA,
                "referents": _schema_array(DRS_REFERENT_JSON_SCHEMA),
                "boxes": _schema_array(DRS_BOX_JSON_SCHEMA),
                "conditions": _schema_array(DRS_CONDITION_JSON_SCHEMA),
                "identity_hypotheses": _schema_array(DRS_IDENTITY_JSON_SCHEMA),
                "temporal_records": _schema_array(DRS_TEMPORAL_JSON_SCHEMA),
                "evidence_spans": STRING_ARRAY_SCHEMA,
                "semantic_notes": STRING_ARRAY_SCHEMA,
            },
        )
    },
)


def chunk_drs_json_schema(
    max_evidence_chars: int | None = None,
    max_array_items: int | None = None,
    *,
    include_auxiliary_fields: bool = True,
    source_id: str | None = None,
    evidence_text_values: list[str] | None = None,
    constrain_stable_ids: bool = False,
) -> dict[str, Any]:
    schema = json.loads(json.dumps(DRS_JSON_SCHEMA))
    drs_schema = schema["properties"]["drs"]
    if not include_auxiliary_fields:
        required = drs_schema.get("required")
        if isinstance(required, list):
            drs_schema["required"] = [key for key in required if key not in {"evidence_spans", "semantic_notes"}]
        properties = drs_schema.get("properties")
        if isinstance(properties, dict):
            properties.pop("evidence_spans", None)
            properties.pop("semantic_notes", None)
    if (
        not max_evidence_chars
        and not max_array_items
        and source_id is None
        and not evidence_text_values
        and not constrain_stable_ids
    ):
        return schema
    max_length = max(1, int(max_evidence_chars)) if max_evidence_chars else None
    max_items = max(1, int(max_array_items)) if max_array_items else None

    def visit(node: Any, parent_key: str = "") -> None:
        if isinstance(node, dict):
            if max_length is not None and parent_key == "evidence_text" and node.get("type") == "string":
                node["maxLength"] = max_length
            if max_length is not None and parent_key == "evidence_spans" and isinstance(node.get("items"), dict):
                node["items"]["maxLength"] = max_length
            if (
                max_items is not None
                and node.get("type") == "array"
                and parent_key
                in {
                    "referents",
                    "boxes",
                    "conditions",
                    "arguments",
                    "identity_hypotheses",
                    "temporal_records",
                    "evidence_spans",
                    "semantic_notes",
                }
            ):
                node["maxItems"] = max_items
            for key, value in node.items():
                visit(value, key)
        elif isinstance(node, list):
            for item in node:
                visit(item, parent_key)

    visit(schema)
    drs_properties = drs_schema.get("properties")
    if not isinstance(drs_properties, dict):
        return schema
    if source_id is not None:
        drs_properties["source_id"] = _schema_enum({source_id})
    if constrain_stable_ids:
        stable_item_count = max_items or 8
        referent_ids = [f"r{index}" for index in range(stable_item_count)]
        box_ids = [f"b{index}" for index in range(stable_item_count)]
        condition_ids = [f"c{index}" for index in range(stable_item_count)]
        temporal_ids = [f"t{index}" for index in range(stable_item_count)]
        referent_schema = drs_properties["referents"]["items"]
        box_schema = drs_properties["boxes"]["items"]
        condition_schema = drs_properties["conditions"]["items"]
        argument_schema = condition_schema["properties"]["arguments"]["items"]
        identity_schema = drs_properties["identity_hypotheses"]["items"]
        temporal_schema = drs_properties["temporal_records"]["items"]
        referent_schema["properties"]["id"] = {"type": "string", "enum": referent_ids}
        box_schema["properties"]["id"] = {"type": "string", "enum": box_ids}
        box_schema["properties"]["parent_id"] = {"type": "string", "enum": ["", *box_ids]}
        box_schema["properties"]["holder_referent_id"] = {"type": "string", "enum": ["", *referent_ids]}
        condition_schema["properties"]["id"] = {"type": "string", "enum": condition_ids}
        condition_schema["properties"]["box_id"] = {"type": "string", "enum": box_ids}
        condition_schema["properties"]["temporal_id"] = {"type": "string", "enum": ["", *temporal_ids]}
        argument_schema["properties"]["target_id"] = {
            "type": "string",
            "enum": sorted(set(["", *box_ids, *condition_ids, *referent_ids])),
        }
        identity_schema["properties"]["left_referent_id"] = {"type": "string", "enum": referent_ids}
        identity_schema["properties"]["right_referent_id"] = {"type": "string", "enum": referent_ids}
        temporal_schema["properties"]["id"] = {"type": "string", "enum": temporal_ids}
    if evidence_text_values:
        evidence_values = list(dict.fromkeys(str(value) for value in evidence_text_values))
        evidence_schema: dict[str, Any] = {"type": "string", "enum": evidence_values}
        if max_length is not None:
            evidence_schema["maxLength"] = max_length
        condition_schema = drs_properties["conditions"]["items"]
        argument_schema = condition_schema["properties"]["arguments"]["items"]
        condition_schema["properties"]["evidence_text"] = copy.deepcopy(evidence_schema)
        argument_schema["properties"]["evidence_text"] = copy.deepcopy(evidence_schema)
    return schema


def chunk_drs_evidence_max_chars(chunk_text: str, n_predict: int | None = None) -> int | None:
    if not chunk_text:
        return None
    configured = os.environ.get("KMD_CHUNK_DRS_MAX_EVIDENCE_CHARS")
    if configured:
        try:
            return max(1, min(len(chunk_text), int(configured)))
        except ValueError:
            pass
    if not n_predict:
        return len(chunk_text)
    budgeted = max(96, min(256, int(n_predict) // 4))
    return max(1, min(len(chunk_text), budgeted))


def chunk_drs_array_max_items(n_predict: int | None = None) -> int | None:
    configured = os.environ.get("KMD_CHUNK_DRS_MAX_ARRAY_ITEMS")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    if not n_predict:
        return None
    return max(4, min(10, int(n_predict) // 96))


def _staged_chunk_drs_enabled() -> bool:
    return os.environ.get("KMD_CHUNK_DRS_STAGED_FALLBACK", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _validation_count(validation: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(validation.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def _staged_fallback_failure_summary(fallback: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "accepted": False,
        "reason": fallback.get("reason"),
        "stage": fallback.get("stage"),
    }
    for key in ("error", "raw_snippet", "grounding_failures", "validation", "elapsed"):
        value = fallback.get(key)
        if value:
            summary[key] = value
    raw_text = str(fallback.get("raw_text") or "")
    if raw_text:
        summary["raw_snippet"] = summary.get("raw_snippet") or raw_text[:4000]
    return summary


def _chunk_drs_structurally_sparse(validation: dict[str, Any]) -> bool:
    """Return true for model-produced DRS shells that need a second extraction pass."""

    condition_count = _validation_count(validation, "condition_count")
    referent_count = _validation_count(validation, "referent_count")
    box_count = _validation_count(validation, "box_count")
    return condition_count == 0 and box_count > 0 and referent_count > 0


def _chunk_drs_structural_condition_floor(source_text: str, max_evidence_chars: int | None = None) -> int:
    field_like_spans = []
    source_surface = source_text.strip()
    for span in chunk_drs_source_span_candidates(source_text, max_evidence_chars):
        if not span or span == source_surface:
            continue
        if (":" in span or "=" in span) and not span.endswith(":"):
            field_like_spans.append(span)
    return len(field_like_spans)


def _chunk_drs_staged_retry_reason(
    validation: dict[str, Any],
    source_text: str = "",
    context_budget: dict[str, Any] | None = None,
) -> str:
    if _chunk_drs_structurally_sparse(validation):
        return "structural_sparsity"
    condition_count = _validation_count(validation, "condition_count")
    field_like_span_count = _chunk_drs_structural_condition_floor(
        source_text,
        (context_budget or {}).get("max_evidence_chars"),
    )
    if field_like_span_count >= 3 and condition_count < 2:
        return "structural_undercoverage"
    return ""


def chunk_drs_source_span_candidates(
    chunk_text: str,
    max_evidence_chars: int | None = None,
    *,
    max_candidates: int = 24,
) -> list[str]:
    candidates = [""]
    if not chunk_text:
        return candidates
    max_len = max(1, int(max_evidence_chars)) if max_evidence_chars else 0

    def add(candidate: str) -> None:
        span = candidate.strip()
        if not span or span in candidates or span not in chunk_text:
            return
        if span.endswith(":"):
            return
        if max_len and len(span) > max_len:
            return
        candidates.append(span)

    def add_value_spans(segment: str) -> None:
        text = segment.strip()
        if not text:
            return
        for separator in (":", "="):
            if separator not in text:
                continue
            _head, tail = text.split(separator, 1)
            value = tail.strip()
            if not value:
                continue
            add(value)
            unquoted = value.strip("\"'")
            if unquoted != value:
                add(unquoted)
            break

    add(chunk_text)
    normalized_separators = chunk_text
    for separator in ("\n", "\t", "|", ";", ",", "{", "}", "[", "]"):
        normalized_separators = normalized_separators.replace(separator, "|")
    for segment in normalized_separators.split("|"):
        add(segment)
        add_value_spans(segment)
        if len(candidates) >= max_candidates:
            break
    return candidates[:max_candidates]


def default_staged_chunk_drs_skeleton_n_predict(n_predict: int) -> int:
    configured = os.environ.get("KMD_CHUNK_DRS_STAGED_SKELETON_N_PREDICT")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    return max(192, min(int(n_predict), 384))


def default_staged_chunk_drs_condition_n_predict(n_predict: int) -> int:
    configured = os.environ.get("KMD_CHUNK_DRS_STAGED_CONDITION_N_PREDICT")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    return max(int(n_predict), 768)


def default_chunk_drs_box_completion_n_predict(n_predict: int) -> int:
    configured = os.environ.get("KMD_CHUNK_DRS_BOX_COMPLETION_N_PREDICT")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    return max(128, min(int(n_predict), 384))


def _schema_array_limited(item: dict[str, Any], max_items: int | None = None) -> dict[str, Any]:
    schema = _schema_array(item)
    if max_items:
        schema["maxItems"] = max(1, int(max_items))
    return schema


def chunk_drs_skeleton_json_schema(
    source_id: str,
    max_array_items: int | None = None,
    evidence_text_values: list[str] | None = None,
) -> dict[str, Any]:
    max_items = max(1, int(max_array_items)) if max_array_items else 8
    referent_ids = [f"r{index}" for index in range(max_items)]
    box_ids = [f"b{index}" for index in range(max_items)]
    temporal_ids = [f"t{index}" for index in range(max_items)]
    referent_schema = copy.deepcopy(DRS_REFERENT_JSON_SCHEMA)
    box_schema = copy.deepcopy(DRS_BOX_JSON_SCHEMA)
    temporal_schema = copy.deepcopy(DRS_TEMPORAL_JSON_SCHEMA)
    referent_schema["properties"]["id"] = {"type": "string", "enum": referent_ids}
    box_schema["properties"]["id"] = {"type": "string", "enum": box_ids}
    box_schema["properties"]["parent_id"] = {"type": "string", "enum": ["", *box_ids]}
    box_schema["properties"]["holder_referent_id"] = {"type": "string", "enum": ["", *referent_ids]}
    temporal_schema["properties"]["id"] = {"type": "string", "enum": temporal_ids}
    if evidence_text_values:
        evidence_values = list(dict.fromkeys(str(value) for value in evidence_text_values))
        evidence_schema = {"type": "string", "enum": evidence_values}
        referent_schema["properties"]["evidence_text"] = copy.deepcopy(evidence_schema)
        box_schema["properties"]["evidence_text"] = copy.deepcopy(evidence_schema)
        temporal_schema["properties"]["evidence_text"] = copy.deepcopy(evidence_schema)
    return _schema_obj(
        ["drs_skeleton"],
        {
            "drs_skeleton": _schema_obj(
                ["schema_version", "source_id", "referents", "boxes", "temporal_records"],
                {
                    "schema_version": _schema_enum({CHUNK_DRS_SCHEMA_VERSION}),
                    "source_id": _schema_enum({source_id}),
                    "referents": _schema_array_limited(referent_schema, max_array_items),
                    "boxes": _schema_array_limited(box_schema, max_array_items),
                    "temporal_records": _schema_array_limited(temporal_schema, max_array_items),
                },
            )
        },
    )


def chunk_drs_condition_json_schema(
    *,
    source_id: str,
    box_ids: list[str],
    referent_ids: list[str],
    temporal_ids: list[str] | None = None,
    max_conditions: int | None = None,
    max_arguments: int | None = None,
    evidence_text_values: list[str] | None = None,
) -> dict[str, Any]:
    condition_schema = copy.deepcopy(DRS_CONDITION_JSON_SCHEMA)
    argument_schema = copy.deepcopy(DRS_ARGUMENT_JSON_SCHEMA)
    allowed_targets = sorted(set(["", *box_ids, *referent_ids]))
    allowed_temporals = sorted(set(["", *(temporal_ids or [])]))
    argument_schema["properties"]["target_id"] = {"type": "string", "enum": allowed_targets}
    condition_schema["properties"]["box_id"] = {"type": "string", "enum": box_ids or [""]}
    condition_schema["properties"]["temporal_id"] = {"type": "string", "enum": allowed_temporals}
    if max_conditions:
        condition_schema["properties"]["id"] = {
            "type": "string",
            "enum": [f"c{index}" for index in range(max(1, int(max_conditions)))],
        }
    if evidence_text_values:
        evidence_values = list(dict.fromkeys(str(value) for value in evidence_text_values))
        condition_schema["properties"]["evidence_text"] = {"type": "string", "enum": evidence_values}
        argument_schema["properties"]["evidence_text"] = {"type": "string", "enum": evidence_values}
    condition_schema["properties"]["arguments"] = _schema_array_limited(argument_schema, max_arguments)
    return _schema_obj(
        ["condition_stage"],
        {
            "condition_stage": _schema_obj(
                ["schema_version", "source_id", "conditions"],
                {
                    "schema_version": _schema_enum({CHUNK_DRS_SCHEMA_VERSION}),
                    "source_id": _schema_enum({source_id}),
                    "conditions": _schema_array_limited(condition_schema, max_conditions),
                },
            )
        },
    )


def chunk_drs_box_completion_json_schema(
    *,
    source_id: str,
    missing_box_ids: list[str],
    existing_box_ids: list[str],
    referent_ids: list[str],
    max_boxes: int | None = None,
) -> dict[str, Any]:
    box_schema = copy.deepcopy(DRS_BOX_JSON_SCHEMA)
    box_schema["properties"]["id"] = {"type": "string", "enum": sorted(set(missing_box_ids))}
    box_schema["properties"]["parent_id"] = {"type": "string", "enum": sorted(set(["", *existing_box_ids]))}
    box_schema["properties"]["holder_referent_id"] = {"type": "string", "enum": sorted(set(["", *referent_ids]))}
    return _schema_obj(
        ["box_completion"],
        {
            "box_completion": _schema_obj(
                ["schema_version", "source_id", "boxes"],
                {
                    "schema_version": _schema_enum({CHUNK_DRS_SCHEMA_VERSION}),
                    "source_id": _schema_enum({source_id}),
                    "boxes": _schema_array_limited(box_schema, max_boxes),
                },
            )
        },
    )


QUERY_DRS_JSON_SCHEMA = _schema_obj(
    ["query_drs"],
    {
        "query_drs": _schema_obj(
            [
                "schema_version",
                "question",
                "answer_variables",
                "target_referents",
                "temporal_records",
                "requested_conditions",
                "constraints",
                "box_requirements",
                "temporal_scope",
                "aggregation",
                "answer_type",
                "requires_evidence",
            ],
            {
                "schema_version": STRING_SCHEMA,
                "question": STRING_SCHEMA,
                "answer_variables": _schema_array(QUERY_VARIABLE_JSON_SCHEMA),
                "target_referents": _schema_array(DRS_REFERENT_JSON_SCHEMA),
                "temporal_records": _schema_array(DRS_TEMPORAL_JSON_SCHEMA),
                "requested_conditions": _schema_array(QUERY_DRS_CONDITION_JSON_SCHEMA),
                "constraints": STRING_ARRAY_SCHEMA,
                "box_requirements": _schema_array(DRS_BOX_JSON_SCHEMA),
                "temporal_scope": STRING_SCHEMA,
                "aggregation": STRING_SCHEMA,
                "answer_type": ANSWER_TYPE_SCHEMA,
                "requires_evidence": BOOL_SCHEMA,
            },
        )
    },
)


def query_drs_array_max_items(n_predict: int | None = None) -> int | None:
    configured = os.environ.get("KMD_QUERY_DRS_MAX_ARRAY_ITEMS")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    if not n_predict:
        return None
    return max(4, min(8, int(n_predict) // 96))


def query_drs_json_schema(question: str | None = None, max_array_items: int | None = None) -> dict[str, Any]:
    schema = copy.deepcopy(QUERY_DRS_JSON_SCHEMA)
    query_schema = schema["properties"]["query_drs"]
    query_schema["properties"]["schema_version"] = _schema_enum({QUERY_DRS_SCHEMA_VERSION})
    if question is not None:
        query_schema["properties"]["question"] = _schema_enum({question})
    if max_array_items:
        capped = max(1, int(max_array_items))

        def visit(node: Any, parent_key: str = "") -> None:
            if isinstance(node, dict):
                if node.get("type") == "array" and parent_key in {
                    "answer_variables",
                    "target_referents",
                    "temporal_records",
                    "requested_conditions",
                    "constraints",
                    "box_requirements",
                    "arguments",
                }:
                    node["maxItems"] = capped
                for key, value in node.items():
                    visit(value, key)
            elif isinstance(node, list):
                for item in node:
                    visit(item, parent_key)

        visit(schema)
    return schema

VERIFICATION_JSON_SCHEMA = _schema_obj(
    ["verification"],
    {
        "verification": _schema_obj(
            ["entailed", "answer_type", "answer", "evidence_span", "reason"],
            {
                "entailed": BOOL_SCHEMA,
                "answer_type": ANSWER_TYPE_SCHEMA,
                "answer": STRING_SCHEMA,
                "evidence_span": STRING_SCHEMA,
                "reason": STRING_SCHEMA,
            },
        )
    },
)

CANONICAL_ANSWER_JSON_SCHEMA = _schema_obj(
    ["canonical_answer"],
    {
        "canonical_answer": _schema_obj(
            ["answer", "evidence_span", "reason"],
            {"answer": STRING_SCHEMA, "evidence_span": STRING_SCHEMA, "reason": STRING_SCHEMA},
        )
    },
)

IDENTITY_CANONICALIZATION_JSON_SCHEMA = _schema_obj(
    ["canonicalization"],
    {
        "canonicalization": _schema_obj(
            ["same_referent", "answer", "evidence_span", "reason"],
            {
                "same_referent": BOOL_SCHEMA,
                "answer": STRING_SCHEMA,
                "evidence_span": STRING_SCHEMA,
                "reason": STRING_SCHEMA,
            },
        )
    },
)


def _evidence_contains_span(span: str, evidence_items: list[dict[str, str]]) -> bool:
    return bool(span) and any(span in str(item.get("text") or "") for item in evidence_items)


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
    canonicalization_call_count: int = 0
    canonicalization_accepted_count: int = 0
    canonicalization_rejected_count: int = 0
    cache_hit_count: int = 0
    rejected_output_count: int = 0
    invalid_json_count: int = 0
    schema_rejection_count: int = 0
    grounding_rejection_count: int = 0
    time_spent_seconds: float = 0.0
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
            "canonicalization_call_count": self.canonicalization_call_count,
            "canonicalization_accepted_count": self.canonicalization_accepted_count,
            "canonicalization_rejected_count": self.canonicalization_rejected_count,
            "cache_hit_count": self.cache_hit_count,
            "rejected_output_count": self.rejected_output_count,
            "invalid_json_count": self.invalid_json_count,
            "schema_rejection_count": self.schema_rejection_count,
            "grounding_rejection_count": self.grounding_rejection_count,
            "time_spent_seconds": round(self.time_spent_seconds, 3),
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
        "Use this exact shape: {\"query_frame\":{\"target_anchors\":[],\"answer_variables\":[],"
        "\"requested_relation\":\"\",\"relation_terms\":[],\"constraints\":[],\"scope_requirements\":[],"
        "\"modality_requirements\":[],\"answer_type\":\"unknown\",\"temporal_scope\":\"\","
        "\"negated\":false,\"aggregation\":\"\",\"requires_evidence\":true}}. "
        "All semantic decisions about requested relation, answer variables, answer type, scope, polarity, "
        "temporal constraints, modality, and aggregation belong in this JSON. The broad answer_type must be one "
        "of the schema values: person, actor, organization, identifier, url, file_path, count, state, date_time, "
        "boolean, content_phrase, metadata_value, or unknown. Use unknown only when the query DRS leaves the "
        "answer variable type underspecified. Put any quantity, list, temporal, modal, polarity, or qualifier "
        "requirements into aggregation, temporal_scope, modality_requirements, scope_requirements, negated, "
        "constraints, and answer_variables as DRS data rather than as prose. If the answer is requested inside a "
        "subordinate or non-asserted DRS, represent that accessibility requirement in modality_requirements or "
        "scope_requirements as well as any predicate text; do not leave the scope marker only in requested_relation. Relation terms should describe the "
        "predicate or answer slot requested by the question, not hidden labels. Use only text visible in the "
        "question and no outside knowledge. Surface observations are syntactic hints only."
        + json.dumps({"question": question, "surface_observations": surface}, ensure_ascii=False)
    )


def call_model_query_plan(question: str, client: LocalModelClient, *, n_predict: int | None = None) -> dict[str, Any]:
    if n_predict is None:
        n_predict = int(os.environ.get("KMD_QUERY_PLAN_N_PREDICT", "128"))
    prompt = build_query_plan_prompt(question)
    constraint = _constraint_settings(QUERY_FRAME_GRAMMAR, QUERY_FRAME_JSON_SCHEMA, QUERY_FRAME_SCHEMA_VERSION)
    grammar_hash = str(constraint["grammar_hash"])
    prompt_hash = _cache_hash(
        "query_frame",
        prompt,
        client,
        {"n_predict": n_predict, "schema": QUERY_FRAME_SCHEMA_VERSION, **constraint},
    )
    cache_path = _cache_path("KMD_QUERY_PLAN_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and not (
        cached.get("accepted") is False
        and cached.get("reason") in {"invalid_json", "schema_validation_failed", "request_failed"}
    ):
        return cached
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=QUERY_FRAME_GRAMMAR,
            json_schema=QUERY_FRAME_JSON_SCHEMA,
        )
    except Exception as exc:
        from .query import plan_question

        payload = {
            **plan_question(question).as_dict(),
            "source": "model",
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            **constraint,
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
            **constraint,
            "elapsed": round(time.time() - start, 3),
        }
        _write_cache(cache_path, payload)
        return payload
    frame_payload = _repair_query_frame_payload(frame_payload, question)
    if not _valid_query_frame_payload(frame_payload):
        from .query import plan_question

        payload = {
            **plan_question(question).as_dict(),
            "source": "model",
            "accepted": False,
            "reason": "schema_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            **constraint,
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
        **constraint,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    _write_cache(cache_path, payload)
    return payload


QUERY_DRS_GRAMMAR = ""


def build_query_drs_prompt(question: str) -> str:
    surface = {
        "visible_anchors": visible_anchors(question),
        "urls": urls(question),
        "identifiers": identifiers(question),
        "content_tokens": content_tokens(question)[:32],
    }
    return (
        "JSON only. Convert the question into a generic DRT query DRS; do not answer it. "
        "Every semantic decision about answer variables, target referents, requested conditions, constraints, "
        "scope, modality, temporal scope, polarity, and aggregation must be represented in the query_drs JSON. "
        "Use only text visible in the question and no outside knowledge. Use subordinate box_requirements for "
        "questions about reported, believed, negated, conditional, uncertain, hypothetical, fictional, or quoted "
        "content. If a requested condition is in the main asserted query scope and no explicit box_requirement is "
        "needed, set its box_id to the empty string; do not invent a box id without declaring that box. "
        "Declare answer variables as objects with stable local ids such as qv0, a short label for the requested "
        "answer variable, a broad answer_type, and evidence_text copied exactly from the question. Put visible "
        "non-answer discourse anchors that the requested condition is about into target_referents, including named "
        "and common-noun anchors, and put visible temporal phrases into "
        "temporal_records with ids such as qt0. Make condition arguments point to those ids when they are the same "
        "discourse referent or temporal value, and use temporal_id for the condition's temporal record when applicable. "
        "Requested condition arguments must use target_kind='answer_variable' and target_id equal to the declared qv "
        "id for the answer slot. Choose the "
        "top-level answer_type from the schema values based on the answer variable requested by the question; use "
        "unknown only when the query DRS leaves the answer variable type underspecified. "
        "Arguments use target_kind and target_id exactly as declared in the query DRS namespace. "
        "Return this shape with schema_version query-drs-v3: {\"query_drs\":{\"schema_version\":\"query-drs-v3\","
        "\"question\":\"\",\"answer_variables\":[{\"id\":\"qv0\",\"label\":\"\",\"answer_type\":\"unknown\","
        "\"evidence_text\":\"\"}],\"target_referents\":[],\"temporal_records\":[],\"requested_conditions\":[],"
        "\"constraints\":[],\"box_requirements\":[],\"temporal_scope\":\"\",\"aggregation\":\"\","
        "\"answer_type\":\"unknown\",\"requires_evidence\":true}}."
        + json.dumps({"question": question, "surface_observations": surface}, ensure_ascii=False)
    )


def _repair_query_drs_payload(payload: Any, question: str) -> Any:
    if not isinstance(payload, dict) or not isinstance(payload.get("query_drs"), dict):
        return payload
    query_drs = {**payload["query_drs"]}

    def grounded_question_surface(candidate: str) -> str:
        value = candidate.strip()
        if not value:
            return ""
        if value in question:
            return value
        index = question.lower().find(value.lower())
        if index >= 0:
            return question[index : index + len(value)]
        return ""

    def repair_item(item: dict[str, Any], fields: tuple[str, ...], *, use_full_question: bool = False) -> bool:
        evidence_text = str(item.get("evidence_text") or "").strip()
        if not evidence_text:
            return False
        grounded_evidence = grounded_question_surface(evidence_text)
        if grounded_evidence:
            if grounded_evidence != evidence_text:
                item["evidence_text"] = grounded_evidence
                return True
            return False
        for field in fields:
            candidate = str(item.get(field) or "").strip()
            for variant in (candidate, candidate.replace("_", " "), candidate.replace("-", " ")):
                grounded_variant = grounded_question_surface(variant)
                if grounded_variant:
                    item["evidence_text"] = grounded_variant
                    return True
        if use_full_question and question:
            item["evidence_text"] = question
            return True
        return False

    repaired = False
    for key, fields, use_full_question in [
        ("answer_variables", ("label",), False),
        ("target_referents", ("label",), False),
        ("temporal_records", ("value",), False),
        ("box_requirements", (), True),
        ("requested_conditions", (), True),
    ]:
        items = query_drs.get(key)
        if isinstance(items, list):
            repaired_items = [item for item in items if isinstance(item, dict)]
            for item in repaired_items:
                repaired |= repair_item(item, fields, use_full_question=use_full_question)
            if len(repaired_items) != len(items):
                query_drs[key] = repaired_items
                repaired = True
    answer_variable_ids = {
        str(item.get("id") or "").strip()
        for item in query_drs.get("answer_variables", [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    answer_variable_surfaces_by_id = {
        str(item.get("id") or "").strip(): {
            normalize(str(value or ""))
            for value in [item.get("label"), item.get("evidence_text")]
            if str(value or "").strip()
        }
        for item in query_drs.get("answer_variables", [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    target_ids = {
        str(item.get("id") or "").strip()
        for item in query_drs.get("target_referents", [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    target_id_by_surface: dict[str, str] = {}
    for item in query_drs.get("target_referents", []):
        if not isinstance(item, dict):
            continue
        target_id = str(item.get("id") or "").strip()
        if not target_id:
            continue
        for value in [item.get("label"), item.get("evidence_text")]:
            surface = normalize(str(value or ""))
            if surface:
                target_id_by_surface[surface] = target_id
    temporal_ids = {
        str(item.get("id") or "").strip()
        for item in query_drs.get("temporal_records", [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    box_ids = {
        str(item.get("id") or "").strip()
        for item in query_drs.get("box_requirements", [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    condition_ids = {
        str(item.get("id") or "").strip()
        for item in query_drs.get("requested_conditions", [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    conditions = query_drs.get("requested_conditions")
    if isinstance(conditions, list):
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            arguments = condition.get("arguments")
            if not isinstance(arguments, list):
                continue
            repaired_arguments = [item for item in arguments if isinstance(item, dict)]
            for argument in repaired_arguments:
                repaired |= repair_item(argument, ("value", "role"), use_full_question=False)
                target_kind = str(argument.get("target_kind") or "").strip()
                target_id = str(argument.get("target_id") or "").strip()
                declared_kind = ""
                if target_id in answer_variable_ids:
                    declared_kind = "answer_variable"
                elif target_id in target_ids:
                    declared_kind = "referent"
                elif target_id in temporal_ids:
                    declared_kind = "temporal"
                elif target_id in box_ids:
                    declared_kind = "box"
                elif target_id in condition_ids:
                    declared_kind = "condition"
                if declared_kind and target_kind != declared_kind:
                    argument["target_kind"] = declared_kind
                    target_kind = declared_kind
                    repaired = True
                if target_kind == "answer_variable":
                    argument_surfaces = {
                        normalize(str(value or ""))
                        for value in [argument.get("evidence_text"), argument.get("value")]
                        if str(value or "").strip()
                    }
                    target_surface_ids = {
                        target_id_by_surface[surface]
                        for surface in argument_surfaces
                        if surface in target_id_by_surface
                    }
                    if len(target_surface_ids) == 1:
                        argument["target_kind"] = "referent"
                        argument["target_id"] = next(iter(target_surface_ids))
                        argument["value"] = ""
                        target_kind = "referent"
                        repaired = True
                value = str(argument.get("value") or "").strip()
                if target_kind not in {"literal", "unknown"} and value and value not in question:
                    argument["value"] = ""
                    repaired = True
            deduped_arguments: list[dict[str, Any]] = []
            seen_answer_argument_refs: set[tuple[str, str, str, str]] = set()
            grounded_answer_ref_seen: set[str] = set()
            for argument in repaired_arguments:
                target_kind = str(argument.get("target_kind") or "").strip()
                target_id = str(argument.get("target_id") or "").strip()
                if target_kind == "answer_variable" and target_id:
                    argument_surfaces = {
                        normalize(str(value or ""))
                        for value in [argument.get("evidence_text"), argument.get("value"), argument.get("role")]
                        if str(value or "").strip()
                    }
                    answer_surfaces = answer_variable_surfaces_by_id.get(target_id, set())
                    is_grounded_answer_ref = bool(answer_surfaces.intersection(argument_surfaces))
                    if target_id in grounded_answer_ref_seen and not is_grounded_answer_ref:
                        repaired = True
                        continue
                    signature = (
                        target_id,
                        str(argument.get("value") or "").strip(),
                        str(argument.get("value_type") or "").strip(),
                        str(argument.get("evidence_text") or "").strip(),
                    )
                    if signature in seen_answer_argument_refs:
                        repaired = True
                        continue
                    seen_answer_argument_refs.add(signature)
                    if is_grounded_answer_ref:
                        grounded_answer_ref_seen.add(target_id)
                deduped_arguments.append(argument)
            if len(deduped_arguments) != len(arguments):
                condition["arguments"] = deduped_arguments
                repaired = True
    if not repaired:
        return payload
    return {**payload, "query_drs": query_drs}


def _validate_query_drs_payload(payload: Any, question: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("query_drs"), dict):
        return {"schema_valid": False, "errors": ["missing_query_drs_object"]}
    query_drs = payload["query_drs"]
    errors: list[str] = []
    grounding_failures: list[str] = []

    def collection(name: str) -> list[dict[str, Any]]:
        value = query_drs.get(name)
        if not isinstance(value, list):
            errors.append(f"not_list:{name}")
            return []
        return [item for item in value if isinstance(item, dict)]

    def optional_collection(name: str) -> list[dict[str, Any]]:
        value = query_drs.get(name)
        if value is None:
            return []
        if not isinstance(value, list):
            errors.append(f"not_list:{name}")
            return []
        return [item for item in value if isinstance(item, dict)]

    def check_grounding(value: Any, label: str) -> None:
        span = str(value or "").strip()
        if span and span not in question:
            grounding_failures.append(f"{label}:{span[:100]}")

    if query_drs.get("question") != question:
        errors.append("question_mismatch")
    if str(query_drs.get("schema_version") or "") != QUERY_DRS_SCHEMA_VERSION:
        errors.append(f"schema_version_mismatch:{query_drs.get('schema_version')}")
    if str(query_drs.get("answer_type") or "") not in ANSWER_TYPES:
        errors.append(f"bad_answer_type:{query_drs.get('answer_type')}")
    raw_answer_variables = query_drs.get("answer_variables")
    if not isinstance(raw_answer_variables, list):
        errors.append("not_list:answer_variables")
    if not isinstance(query_drs.get("constraints"), list):
        errors.append("not_list:constraints")
    answer_variable_ids: set[str] = set()
    answer_variable_labels: set[str] = set()
    if isinstance(raw_answer_variables, list):
        for index, variable in enumerate(raw_answer_variables):
            if isinstance(variable, dict):
                variable_id = str(variable.get("id") or "").strip()
                label = str(variable.get("label") or "").strip()
                if not variable_id:
                    errors.append(f"answer_variable_missing_id:{index}")
                if not label:
                    errors.append(f"answer_variable_missing_label:{variable_id or index}")
                if str(variable.get("answer_type") or "") not in ANSWER_TYPES:
                    errors.append(f"bad_answer_variable_type:{variable_id}:{variable.get('answer_type')}")
                check_grounding(variable.get("evidence_text"), f"answer_variable:{variable_id or index}")
                if variable_id:
                    answer_variable_ids.add(variable_id)
                if label:
                    answer_variable_labels.add(label)
            elif isinstance(variable, str):
                label = variable.strip()
                if label:
                    answer_variable_labels.add(label)
            else:
                errors.append(f"bad_answer_variable:{index}")
    targets = collection("target_referents")
    temporals = optional_collection("temporal_records")
    boxes = collection("box_requirements")
    conditions = collection("requested_conditions")
    target_ids = {str(item.get("id") or "") for item in targets if str(item.get("id") or "")}
    temporal_ids = {str(item.get("id") or "") for item in temporals if str(item.get("id") or "")}
    box_ids = {str(item.get("id") or "") for item in boxes if str(item.get("id") or "")}
    condition_ids = {str(item.get("id") or "") for item in conditions if str(item.get("id") or "")}
    for box in boxes:
        box_id = str(box.get("id") or "")
        parent_id = str(box.get("parent_id") or "")
        holder_id = str(box.get("holder_referent_id") or "")
        if str(box.get("kind") or "") not in DRS_CONTEXT_KINDS:
            errors.append(f"bad_box_kind:{box_id}:{box.get('kind')}")
        if parent_id and parent_id not in box_ids:
            errors.append(f"missing_parent_box:{box_id}->{parent_id}")
        if holder_id and holder_id not in target_ids:
            errors.append(f"missing_holder_referent:{box_id}->{holder_id}")
        check_grounding(box.get("evidence_text"), f"box:{box_id}")
    for target in targets:
        target_id = str(target.get("id") or "")
        if not target_id or not str(target.get("label") or "").strip():
            errors.append(f"bad_target_referent:{target_id}")
        check_grounding(target.get("evidence_text"), f"target:{target_id}")
    for temporal in temporals:
        temporal_id = str(temporal.get("id") or "")
        if not temporal_id or not str(temporal.get("value") or "").strip():
            errors.append(f"bad_temporal:{temporal_id}")
        check_grounding(temporal.get("evidence_text"), f"temporal:{temporal_id}")
    for condition in conditions:
        condition_id = str(condition.get("id") or "")
        box_id = str(condition.get("box_id") or "")
        temporal_id = str(condition.get("temporal_id") or "")
        if not condition_id or not str(condition.get("predicate") or "").strip():
            errors.append(f"bad_condition:{condition_id}")
        if box_id and box_id not in box_ids:
            errors.append(f"missing_condition_box:{condition_id}->{box_id}")
        if temporal_id and temporal_id not in temporal_ids:
            errors.append(f"missing_condition_temporal:{condition_id}->{temporal_id}")
        if str(condition.get("polarity") or "") not in DRS_POLARITIES:
            errors.append(f"bad_polarity:{condition_id}:{condition.get('polarity')}")
        if str(condition.get("modality") or "") not in DRS_CONTEXT_KINDS:
            errors.append(f"bad_modality:{condition_id}:{condition.get('modality')}")
        check_grounding(condition.get("evidence_text"), f"condition:{condition_id}")
        arguments = condition.get("arguments")
        if not isinstance(arguments, list):
            errors.append(f"bad_arguments:{condition_id}")
            continue
        for arg in arguments:
            if not isinstance(arg, dict):
                continue
            target_kind = str(arg.get("target_kind") or "")
            target_id = str(arg.get("target_id") or "")
            if target_kind == "answer_variable":
                if answer_variable_ids and target_id not in answer_variable_ids:
                    errors.append(f"missing_answer_variable:{condition_id}->{target_id}")
                elif not answer_variable_ids and target_id and target_id not in answer_variable_labels:
                    errors.append(f"missing_answer_variable:{condition_id}->{target_id}")
            elif target_kind == "referent" and target_id and target_id not in target_ids:
                errors.append(f"missing_argument_referent:{condition_id}->{target_id}")
            elif target_kind == "box" and target_id and target_id not in box_ids:
                errors.append(f"missing_argument_box:{condition_id}->{target_id}")
            elif target_kind == "condition" and target_id and target_id not in condition_ids:
                errors.append(f"missing_argument_condition:{condition_id}->{target_id}")
            elif target_kind == "temporal" and target_id and target_id not in temporal_ids:
                errors.append(f"missing_argument_temporal:{condition_id}->{target_id}")
            elif target_kind not in {"answer_variable", "referent", "box", "condition", "temporal", "literal", "unknown"}:
                errors.append(f"bad_argument_target_kind:{condition_id}:{target_kind}")
            check_grounding(arg.get("evidence_text"), f"argument:{condition_id}:{arg.get('role')}")
    return {
        "schema_valid": not errors and not grounding_failures,
        "errors": errors[:50],
        "grounding_failures": grounding_failures[:50],
        "grounding_failure_count": len(grounding_failures),
        "answer_variable_count": len(answer_variable_ids) or len(answer_variable_labels),
        "target_count": len(targets),
        "temporal_record_count": len(temporals),
        "condition_count": len(conditions),
        "box_requirement_count": len(boxes),
    }


def call_model_query_drs(question: str, client: LocalModelClient, *, n_predict: int | None = None) -> dict[str, Any]:
    if n_predict is None:
        n_predict = default_query_drs_n_predict(client)
    prompt = build_query_drs_prompt(question)
    max_array_items = query_drs_array_max_items(n_predict)
    json_schema = query_drs_json_schema(question, max_array_items=max_array_items)
    constraint = _constraint_settings(QUERY_DRS_GRAMMAR, json_schema, QUERY_DRS_SCHEMA_VERSION)
    prompt_hash = _cache_hash(
        "query_drs",
        prompt,
        client,
        {
            "n_predict": n_predict,
            "schema": QUERY_DRS_SCHEMA_VERSION,
            "validation_policy": QUERY_DRS_VALIDATION_POLICY,
            "array_cap_policy": QUERY_DRS_ARRAY_CAP_POLICY,
            "max_array_items": max_array_items,
            **constraint,
        },
    )
    cache_path = _cache_path("KMD_QUERY_DRS_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and cached.get("reason") != "request_failed":
        return cached
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=QUERY_DRS_GRAMMAR,
            json_schema=json_schema,
        )
    except LocalModelJSONError as exc:
        payload = {
            "accepted": False,
            "reason": "invalid_json",
            "error": str(exc),
            "raw_text": exc.raw_text,
            "raw_snippet": exc.snippet,
            "prompt_hash": prompt_hash,
            **constraint,
            "validation_policy": QUERY_DRS_VALIDATION_POLICY,
            "array_cap_policy": QUERY_DRS_ARRAY_CAP_POLICY,
            "max_array_items": max_array_items,
            "elapsed": round(time.time() - start, 3),
        }
        _write_cache(cache_path, payload)
        return payload
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            **constraint,
            "validation_policy": QUERY_DRS_VALIDATION_POLICY,
            "array_cap_policy": QUERY_DRS_ARRAY_CAP_POLICY,
            "max_array_items": max_array_items,
            "elapsed": round(time.time() - start, 3),
        }
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    parsed = _repair_query_drs_payload(parsed, question)
    validation = _validate_query_drs_payload(parsed, question)
    if not validation.get("schema_valid"):
        payload = {
            "accepted": False,
            "reason": "schema_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            **constraint,
            "validation_policy": QUERY_DRS_VALIDATION_POLICY,
            "array_cap_policy": QUERY_DRS_ARRAY_CAP_POLICY,
            "max_array_items": max_array_items,
            "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
            "validation": validation,
        }
        _write_cache(cache_path, payload)
        return payload
    payload = {
        "accepted": True,
        "query_drs": parsed["query_drs"],
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": prompt_hash,
        **constraint,
        "validation_policy": QUERY_DRS_VALIDATION_POLICY,
        "array_cap_policy": QUERY_DRS_ARRAY_CAP_POLICY,
        "max_array_items": max_array_items,
        "validation": validation,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    _write_cache(cache_path, payload)
    return payload


def query_frame_from_query_drs(question: str, query_drs: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(query_drs, dict):
        return None
    target_referents = query_drs.get("target_referents")
    requested_conditions = query_drs.get("requested_conditions")
    box_requirements = query_drs.get("box_requirements")
    temporal_records = query_drs.get("temporal_records")
    if not isinstance(target_referents, list) or not isinstance(requested_conditions, list):
        return None
    answer_variables_raw = query_drs.get("answer_variables")
    answer_variables: list[str] = []
    answer_variable_labels_by_id: dict[str, str] = {}
    if isinstance(answer_variables_raw, list):
        for item in answer_variables_raw:
            if isinstance(item, dict):
                variable_id = str(item.get("id") or "").strip()
                label = str(item.get("label") or "").strip()
                if label:
                    answer_variables.append(label)
                if variable_id and label:
                    answer_variable_labels_by_id[variable_id] = label
            elif str(item or "").strip():
                answer_variables.append(str(item).strip())
    target_anchors = [
        str(item.get("label") or "").strip()
        for item in target_referents
        if isinstance(item, dict) and str(item.get("label") or "").strip()
    ]
    temporal_terms: list[str] = []
    temporal_values_by_id: dict[str, str] = {}
    if isinstance(temporal_records, list):
        for item in temporal_records:
            if not isinstance(item, dict):
                continue
            temporal_id = str(item.get("id") or "").strip()
            value = str(item.get("value") or "").strip()
            evidence = str(item.get("evidence_text") or "").strip()
            temporal_text = value or evidence
            if temporal_text:
                temporal_terms.append(temporal_text)
            if temporal_id and temporal_text:
                temporal_values_by_id[temporal_id] = temporal_text
    predicates = [
        str(item.get("predicate") or "").strip()
        for item in requested_conditions
        if isinstance(item, dict) and str(item.get("predicate") or "").strip()
    ]
    argument_terms: list[str] = []
    modality_terms: list[str] = []
    for condition in requested_conditions:
        if not isinstance(condition, dict):
            continue
        modality = str(condition.get("modality") or "").strip()
        if modality and modality != "asserted":
            modality_terms.append(modality)
        for argument in condition.get("arguments") or []:
            if not isinstance(argument, dict):
                continue
            target_kind = str(argument.get("target_kind") or "").strip()
            target_id = str(argument.get("target_id") or "").strip()
            value = str(argument.get("value") or "").strip()
            role = str(argument.get("role") or "").strip()
            if target_kind == "answer_variable" and target_id in answer_variable_labels_by_id:
                argument_terms.append(answer_variable_labels_by_id[target_id])
            if target_kind == "temporal" and target_id in temporal_values_by_id:
                argument_terms.append(temporal_values_by_id[target_id])
            if value:
                argument_terms.append(value)
            if role:
                argument_terms.append(role)
    scope_terms = [
        str(item.get("kind") or "").strip()
        for item in box_requirements or []
        if isinstance(item, dict) and str(item.get("kind") or "").strip() and str(item.get("kind") or "") != "asserted"
    ]
    temporal_scope = query_drs.get("temporal_scope") if isinstance(query_drs.get("temporal_scope"), str) else ""
    if not temporal_scope and temporal_terms:
        temporal_scope = " ".join(dict.fromkeys(temporal_terms))
    frame = frame_from_mapping(
        question,
        {
            "target_anchors": list(dict.fromkeys(target_anchors)),
            "answer_variables": list(dict.fromkeys(answer_variables)),
            "requested_relation": " ".join(dict.fromkeys(predicates)),
            "relation_terms": list(dict.fromkeys([*predicates, *argument_terms, *temporal_terms])),
            "constraints": query_drs.get("constraints") if isinstance(query_drs.get("constraints"), list) else [],
            "scope_requirements": list(dict.fromkeys(scope_terms)),
            "modality_requirements": list(dict.fromkeys(modality_terms)),
            "answer_type": query_drs.get("answer_type") if isinstance(query_drs.get("answer_type"), str) else "unknown",
            "temporal_scope": temporal_scope,
            "aggregation": query_drs.get("aggregation") if isinstance(query_drs.get("aggregation"), str) else "",
            "requires_evidence": bool(query_drs.get("requires_evidence", True)),
        },
        source="model_query_drs",
    )
    return frame.as_dict()


def build_evidence_extraction_prompt(question: str, expected_answer_type: str, evidence_items: list[dict[str, str]]) -> str:
    return (
        "JSON only. Answer the question only from the provided raw-text evidence. "
        "Return sufficient_evidence=false and answer='unknown' when the evidence does not state a complete answer. "
        "The answer must be the grounded value bound to the question's answer variable and compatible with the "
        "expected answer type. Preserve the source wording needed for that binding, and do not include unrelated "
        "predicate or context text. If the question requires multiple bindings or an aggregate, encode the scalar "
        "public answer requested by that query and separate multiple grounded bindings with '; '. Interpret the "
        "DRT conditions in the evidence, including referents, roles, identity, polarity, modality, temporal scope, "
        "and accessibility. Do not use outside knowledge or hidden labels. The evidence_span must be copied "
        "exactly from one provided evidence item when sufficient_evidence is true."
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
    constraint = _constraint_settings(EVIDENCE_EXTRACTION_GRAMMAR, ANSWER_JSON_SCHEMA, ANSWER_SCHEMA_VERSION)
    grammar_hash = str(constraint["grammar_hash"])
    prompt_hash = _cache_hash(
        "evidence_answer",
        prompt,
        client,
        {
            "n_predict": n_predict,
            "schema": ANSWER_SCHEMA_VERSION,
            "expected_answer_type": expected_answer_type,
            **constraint,
        },
    )
    cache_path = _cache_path("KMD_EVIDENCE_ANSWER_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=EVIDENCE_EXTRACTION_GRAMMAR,
            json_schema=ANSWER_JSON_SCHEMA,
        )
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            **constraint,
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
        payload = {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            **constraint,
            "elapsed": round(time.time() - start, 3),
        }
        _write_cache(cache_path, payload)
        return payload
    answer = _repair_answer_payload(answer, expected_answer_type)
    answer = _repair_evidence_span(answer, evidence_items)
    if not _valid_answer_payload(answer):
        payload = {
            "accepted": False,
            "reason": "schema_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            **constraint,
            "elapsed": round(time.time() - start, 3),
        }
        _write_cache(cache_path, payload)
        return payload
    payload = {
        "accepted": True,
        "sufficient_evidence": bool(answer.get("sufficient_evidence")),
        "answer_type": str(answer.get("answer_type") or "unknown"),
        "answer": str(answer.get("answer") or ""),
        "evidence_span": str(answer.get("evidence_span") or ""),
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": prompt_hash,
        **constraint,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    _write_cache(cache_path, payload)
    return payload


def build_query_evidence_answer_prompt(
    question: str,
    evidence_items: list[dict[str, str]],
    discourse_records: list[dict[str, Any]] | None = None,
) -> str:
    surface = {
        "visible_anchors": visible_anchors(question),
        "urls": urls(question),
        "identifiers": identifiers(question),
        "content_tokens": content_tokens(question)[:32],
    }
    return (
        "JSON only. Perform a bounded DRT/DSPG question analysis and grounded entailment check. "
        "First convert the question into the required generic query_frame object. Then answer only if the provided "
        "DRS/DSPG discourse records and raw-text evidence entail a complete answer to that frame. Relation words "
        "are data from the question, discourse records, and evidence, not handler names. Reject values that do not "
        "satisfy the query frame's type, role, scope, polarity, modality, temporal, identity, and provenance "
        "requirements. "
        "The output must include all required fields in the exact result schema. Do not omit answer_type, "
        "sufficient_evidence, reason, or evidence_span. If sufficient_evidence is true, evidence_span must be "
        "one exact supporting sentence or line copied from the provided evidence. In query_frame, set answer_type "
        "to the broad requested variable type whenever derivable from the question, and use unknown only when it "
        "is genuinely not derivable. "
        "Return exactly one result object with a query_frame object and a scalar answer string; do not use "
        "generic_query_frame and do not nest the answer inside another object. "
        "Copy evidence_span as one exact supporting sentence or line, not a multi-line evidence window. "
        "Return the grounded binding or aggregate requested by the query frame. For multiple bindings, return all "
        "and only the grounded values that satisfy the same frame, separated with '; '. "
        "If evidence is insufficient, return sufficient_evidence=false and answer='unknown'. Copy evidence_span "
        "exactly from one provided evidence item."
        + json.dumps(
            {
                "question": question,
                "surface_observations": surface,
                "evidence": evidence_items,
                "discourse_records": discourse_records or [],
            },
            ensure_ascii=False,
        )
    )


def build_query_evidence_answer_repair_prompt(
    question: str,
    evidence_items: list[dict[str, str]],
    raw_response: str,
    discourse_records: list[dict[str, Any]] | None = None,
) -> str:
    return (
        "JSON only. Repair the previous local-model output into the exact bounded DRT/DSPG answer schema. "
        "Use only the question, bounded evidence, and previous output shown here. This is an LLM semantic repair: "
        "do not fill missing truth conditions by formatting guesses. If the evidence does not entail the previous "
        "answer, or no exact supporting sentence/line can be copied as evidence_span, return sufficient_evidence=false "
        "and answer='unknown'. The repaired answer must preserve the query frame's referents, roles, type, scope, "
        "polarity, modality, temporal constraints, identity constraints, and provenance. Return exactly "
        "{\"result\":{\"query_frame\":{\"target_anchors\":[],\"answer_variables\":[],"
        "\"requested_relation\":\"\",\"relation_terms\":[],\"constraints\":[],\"scope_requirements\":[],"
        "\"modality_requirements\":[],\"answer_type\":\"unknown\",\"temporal_scope\":\"\","
        "\"negated\":false,\"aggregation\":\"\",\"requires_evidence\":true},\"sufficient_evidence\":false,"
        "\"answer_type\":\"unknown\",\"answer\":\"unknown\",\"evidence_span\":\"\",\"reason\":\"\"}}."
        + json.dumps(
            {
                "question": question,
                "evidence": evidence_items,
                "discourse_records": discourse_records or [],
                "previous_model_output": raw_response,
            },
            ensure_ascii=False,
        )
    )


def _query_evidence_payload_from_result(
    question: str,
    result: dict[str, Any],
    evidence_items: list[dict[str, str]],
    raw: str,
    elapsed: float,
    prompt_hash: str,
    grammar_hash: str,
    *,
    fresh_or_cached: str,
    repair_prompt_hash: str = "",
) -> dict[str, Any]:
    frame_payload = result.get("query_frame") if isinstance(result.get("query_frame"), dict) else {}
    if not frame_payload and isinstance(result.get("generic_query_frame"), dict):
        frame_payload = result.get("generic_query_frame")
    frame_payload = _repair_query_frame_payload(frame_payload, question)
    answer_payload = _repair_answer_payload(result, "unknown")
    answer_payload = _repair_evidence_span(answer_payload, evidence_items)
    if not _valid_query_frame_payload(frame_payload):
        frame_payload = _repair_query_frame_payload({}, question)
    if not _valid_query_frame_payload(frame_payload) or not _valid_answer_payload(answer_payload):
        return {
            "accepted": False,
            "reason": "schema_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "elapsed": elapsed,
        }
    sufficient = bool(answer_payload.get("sufficient_evidence"))
    evidence_span = str(answer_payload.get("evidence_span") or "")
    if sufficient and not _evidence_contains_span(evidence_span, evidence_items):
        return {
            "accepted": False,
            "reason": "grounding_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "elapsed": elapsed,
        }
    frame = frame_from_mapping(question, frame_payload, source="model").as_dict()
    payload = {
        "accepted": True,
        "query_frame": frame,
        "sufficient_evidence": sufficient,
        "answer_type": str(answer_payload.get("answer_type") or frame.get("answer_type") or "unknown"),
        "answer": str(answer_payload.get("answer") or ""),
        "evidence_span": evidence_span,
        "reason": str(answer_payload.get("reason") or ""),
        "raw_text": raw,
        "elapsed": elapsed,
        "prompt_hash": prompt_hash,
        "grammar_hash": grammar_hash,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": fresh_or_cached,
    }
    if repair_prompt_hash:
        payload["repair_prompt_hash"] = repair_prompt_hash
    return payload


def _call_model_query_evidence_answer_repair(
    question: str,
    evidence_items: list[dict[str, str]],
    raw_response: str,
    client: LocalModelClient,
    *,
    n_predict: int,
    discourse_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prompt = build_query_evidence_answer_repair_prompt(question, evidence_items, raw_response, discourse_records)
    constraint = _constraint_settings(QUERY_EVIDENCE_ANSWER_GRAMMAR, QUERY_EVIDENCE_ANSWER_JSON_SCHEMA, ANSWER_SCHEMA_VERSION)
    grammar_hash = str(constraint["grammar_hash"])
    prompt_hash = _cache_hash(
        "query_evidence_answer_repair",
        prompt,
        client,
        {"n_predict": n_predict, "schema": ANSWER_SCHEMA_VERSION, **constraint},
    )
    cache_path = _cache_path("KMD_QUERY_EVIDENCE_REPAIR_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and cached.get("reason") not in {"invalid_json", "schema_validation_failed", "grounding_validation_failed"}:
        return cached
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=QUERY_EVIDENCE_ANSWER_GRAMMAR,
            json_schema=QUERY_EVIDENCE_ANSWER_JSON_SCHEMA,
        )
    except Exception as exc:
        payload = {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "elapsed": round(time.time() - start, 3),
        }
        _write_cache(cache_path, payload)
        return payload
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    result = parsed.get("result") if isinstance(parsed, dict) else None
    if result is None and isinstance(parsed, dict) and "answer" in parsed:
        result = parsed
    if not isinstance(result, dict):
        payload = {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "elapsed": round(time.time() - start, 3),
        }
        _write_cache(cache_path, payload)
        return payload
    payload = _query_evidence_payload_from_result(
        question,
        result,
        evidence_items,
        raw,
        parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        prompt_hash,
        grammar_hash,
        fresh_or_cached="fresh_repair",
    )
    _write_cache(cache_path, payload)
    return payload


def call_model_query_evidence_answer(
    question: str,
    evidence_items: list[dict[str, str]],
    client: LocalModelClient,
    *,
    n_predict: int | None = None,
    discourse_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if n_predict is None:
        n_predict = int(os.environ.get("KMD_QUERY_EVIDENCE_N_PREDICT", "128"))
    prompt = build_query_evidence_answer_prompt(question, evidence_items, discourse_records)
    constraint = _constraint_settings(QUERY_EVIDENCE_ANSWER_GRAMMAR, QUERY_EVIDENCE_ANSWER_JSON_SCHEMA, ANSWER_SCHEMA_VERSION)
    grammar_hash = str(constraint["grammar_hash"])
    prompt_hash = _cache_hash(
        "query_evidence_answer",
        prompt,
        client,
        {"n_predict": n_predict, "schema": ANSWER_SCHEMA_VERSION, **constraint},
    )
    cache_path = _cache_path("KMD_QUERY_EVIDENCE_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=QUERY_EVIDENCE_ANSWER_GRAMMAR,
            json_schema=QUERY_EVIDENCE_ANSWER_JSON_SCHEMA,
        )
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "elapsed": round(time.time() - start, 3),
        }
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    result = parsed.get("result") if isinstance(parsed, dict) else None
    if result is None and isinstance(parsed, dict) and "answer" in parsed:
        result = parsed
    if not isinstance(result, dict):
        repaired = _call_model_query_evidence_answer_repair(
            question,
            evidence_items,
            raw,
            client,
            n_predict=n_predict,
            discourse_records=discourse_records,
        )
        if repaired.get("accepted"):
            payload = {**repaired, "repair_of_prompt_hash": prompt_hash}
            _write_cache(cache_path, payload)
            return payload
        payload = {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "elapsed": round(time.time() - start, 3),
        }
        _write_cache(cache_path, payload)
        return payload
    missing_required = not {"query_frame", "sufficient_evidence", "answer_type", "answer", "evidence_span", "reason"}.issubset(result)
    payload = _query_evidence_payload_from_result(
        question,
        result,
        evidence_items,
        raw,
        parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        prompt_hash,
        grammar_hash,
        fresh_or_cached="fresh",
    )
    needs_repair = missing_required or payload.get("reason") in {"schema_validation_failed", "grounding_validation_failed"}
    if needs_repair:
        repaired = _call_model_query_evidence_answer_repair(
            question,
            evidence_items,
            raw,
            client,
            n_predict=n_predict,
            discourse_records=discourse_records,
        )
        if repaired.get("accepted"):
            payload = {**repaired, "repair_of_prompt_hash": prompt_hash}
    _write_cache(cache_path, payload)
    return payload


def build_chunk_frame_prompt(chunk_text: str, *, rel_path: str = "", context_budget: dict[str, Any] | None = None) -> str:
    return (
        "JSON only. Extract generic DRT/DSPG discourse frames and grounded DRT structures from this raw text chunk. "
        "Use this exact shape: {\"frames\":[{\"frame_type\":\"relation\",\"predicate\":\"\","
        "\"arguments\":[{\"role\":\"argument\",\"text\":\"\",\"value_type\":\"unknown\"}],"
        "\"identity_hypotheses\":[{\"left_text\":\"\",\"right_text\":\"\",\"relation\":\"same_referent\","
        "\"evidence_text\":\"\",\"confidence\":0.0}],"
        "\"polarity\":\"positive\",\"modality\":\"asserted\",\"context_holder\":\"\",\"temporal_text\":\"\","
        "\"evidence_text\":\"\",\"confidence\":0.0}]}. "
        "Do not answer questions. Do not use dataset labels, hidden categories, or handler names. "
        "Represent only source-grounded discourse conditions. Predicate and role words are data supplied by "
        "your semantic parse, not control-flow labels. evidence_text must be copied exactly from the chunk. "
        "Each non-empty argument text and identity_hypotheses evidence_text/left_text/right_text must also be "
        "copied exactly from the chunk. Arguments should include every grounded phrase needed to bind the "
        "condition's discourse referents, participants, complements, attributes, quantities, locations, times, "
        "and values when those phrases appear in the chunk. Do not bury a bound value only inside predicate text "
        "when the same value appears as an exact argument phrase in the chunk. Include identity_hypotheses only when the chunk itself supports alias, "
        "coreference, pronoun, speaker, or same-referent links between distinct mentions; do not include self-links. Include modality, polarity, context_holder, "
        "and temporal_text only when the chunk itself supports that DRT interpretation."
        + json.dumps({"source": rel_path, "context_budget": context_budget or {}, "chunk": chunk_text}, ensure_ascii=False)
    )


def _context_limited_chunk_frame_text(
    chunk_text: str,
    client: LocalModelClient,
    *,
    rel_path: str,
    n_predict: int,
) -> tuple[str, dict[str, Any]]:
    context_size = _client_context_size(client)
    budget: dict[str, Any] = {
        "runtime_context_size": context_size,
        "reserved_output_tokens": int(n_predict),
        "context_source": "client_metadata" if context_size > 0 else "unavailable",
    }
    if context_size <= 0:
        configured_chars = os.environ.get("KMD_CHUNK_FRAME_MAX_CHARS")
        if configured_chars:
            try:
                max_chars = max(1, int(configured_chars))
            except ValueError:
                max_chars = len(chunk_text)
            limited = chunk_text[:max_chars]
        else:
            limited = chunk_text
        budget.update(
            {
                "prompt_budget_tokens": 0,
                "prompt_overhead_tokens": 0,
                "chunk_budget_tokens": _estimate_tokens(limited),
                "input_chars": len(chunk_text),
                "prompt_chunk_chars": len(limited),
                "input_truncated": len(limited) < len(chunk_text),
            }
        )
        return limited, budget
    seed_budget = {**budget, "prompt_budget_tokens": max(0, context_size - int(n_predict)), "chunk_budget_tokens": 0}
    overhead_tokens = _estimate_tokens(build_chunk_frame_prompt("", rel_path=rel_path, context_budget=seed_budget))
    prompt_budget_tokens = max(0, context_size - int(n_predict) - overhead_tokens)
    max_chars = max(0, prompt_budget_tokens * 4)
    limited = chunk_text[:max_chars] if max_chars else ""
    budget.update(
        {
            "prompt_budget_tokens": prompt_budget_tokens,
            "prompt_overhead_tokens": overhead_tokens,
            "chunk_budget_tokens": _estimate_tokens(limited),
            "input_chars": len(chunk_text),
            "prompt_chunk_chars": len(limited),
            "input_truncated": len(limited) < len(chunk_text),
        }
    )
    return limited, budget


def chunk_frame_cache_context(client: LocalModelClient | None, *, n_predict: int | None = None) -> dict[str, Any]:
    constraint = _constraint_settings(FRAME_EXTRACTION_GRAMMAR, FRAME_JSON_SCHEMA, CHUNK_FRAME_SCHEMA_VERSION)
    if n_predict is None:
        n_predict = default_chunk_frame_n_predict(client)
    return {
        "prompt_version": PROMPT_VERSION,
        "schema_version": CHUNK_FRAME_SCHEMA_VERSION,
        **constraint,
        "n_predict": int(n_predict),
        "model_fingerprint": _client_fingerprint(client),
    }


def call_model_chunk_frames(
    chunk_text: str,
    client: LocalModelClient,
    *,
    rel_path: str = "",
    n_predict: int | None = None,
) -> dict[str, Any]:
    if n_predict is None:
        n_predict = default_chunk_frame_n_predict(client)
    prompt_chunk, context_budget = _context_limited_chunk_frame_text(
        chunk_text,
        client,
        rel_path=rel_path,
        n_predict=n_predict,
    )
    prompt = build_chunk_frame_prompt(prompt_chunk, rel_path=rel_path, context_budget=context_budget)
    constraint = _constraint_settings(FRAME_EXTRACTION_GRAMMAR, FRAME_JSON_SCHEMA, CHUNK_FRAME_SCHEMA_VERSION)
    grammar_hash = str(constraint["grammar_hash"])
    prompt_hash = _cache_hash(
        "chunk_frames",
        prompt,
        client,
        {
            "n_predict": n_predict,
            "schema": CHUNK_FRAME_SCHEMA_VERSION,
            **constraint,
            "context_budget": context_budget,
        },
    )
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=FRAME_EXTRACTION_GRAMMAR,
            json_schema=FRAME_JSON_SCHEMA,
        )
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
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
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "elapsed": round(time.time() - start, 3),
        }
    grounded: list[dict[str, Any]] = []
    rejected_for_grounding = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        evidence_text = str(frame.get("evidence_text") or "").strip()
        predicate = str(frame.get("predicate") or "").strip()
        if not evidence_text or evidence_text not in prompt_chunk or not predicate:
            rejected_for_grounding += 1
            continue
        arguments = frame.get("arguments")
        if isinstance(arguments, dict):
            arguments = [
                {"role": str(role), "text": str(text), "value_type": "unknown"}
                for role, text in arguments.items()
            ]
        grounded_arguments: list[dict[str, Any]] = []
        if isinstance(arguments, list):
            for argument in arguments:
                if not isinstance(argument, dict):
                    continue
                text = str(argument.get("text") or argument.get("value") or "").strip()
                if text and text not in prompt_chunk:
                    rejected_for_grounding += 1
                    continue
                grounded_arguments.append(
                    {
                        "role": str(argument.get("role") or "argument"),
                        "text": text,
                        "value_type": str(argument.get("value_type") or "unknown"),
                    }
                )
        identity_hypotheses: list[dict[str, Any]] = []
        raw_identity_hypotheses = frame.get("identity_hypotheses")
        if isinstance(raw_identity_hypotheses, list):
            for hypothesis in raw_identity_hypotheses:
                if not isinstance(hypothesis, dict):
                    continue
                left_text = str(hypothesis.get("left_text") or "").strip()
                right_text = str(hypothesis.get("right_text") or "").strip()
                identity_evidence = str(hypothesis.get("evidence_text") or evidence_text).strip()
                if not left_text or not right_text or not identity_evidence:
                    continue
                if left_text not in prompt_chunk or right_text not in prompt_chunk or identity_evidence not in prompt_chunk:
                    rejected_for_grounding += 1
                    continue
                identity_hypotheses.append(
                    {
                        "left_text": left_text,
                        "right_text": right_text,
                        "relation": str(hypothesis.get("relation") or "same_referent"),
                        "evidence_text": identity_evidence,
                        "confidence": _coerce_confidence(hypothesis.get("confidence")),
                    }
                )
        context_holder = str(frame.get("context_holder") or "").strip()
        if context_holder and context_holder not in prompt_chunk:
            rejected_for_grounding += 1
            continue
        temporal_text = str(frame.get("temporal_text") or "").strip()
        if temporal_text and temporal_text not in prompt_chunk:
            rejected_for_grounding += 1
            continue
        grounded.append(
            {
                "frame_type": str(frame.get("frame_type") or "relation"),
                "predicate": predicate,
                "arguments": grounded_arguments,
                "identity_hypotheses": identity_hypotheses,
                "polarity": str(frame.get("polarity") or "positive"),
                "modality": str(frame.get("modality") or "asserted"),
                "context_holder": context_holder,
                "temporal_text": temporal_text,
                "evidence_text": evidence_text,
                "confidence": _coerce_confidence(frame.get("confidence")),
            }
        )
    if frames and not grounded:
        return {
            "accepted": False,
            "reason": "grounding_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
            "rejected_for_grounding": rejected_for_grounding,
            "context_budget": context_budget,
        }
    return {
        "accepted": True,
        "frames": grounded,
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": prompt_hash,
        "grammar_hash": grammar_hash,
        "context_budget": context_budget,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
        "rejected_for_grounding": rejected_for_grounding,
    }


CHUNK_DRS_GRAMMAR = ""


def build_chunk_drs_prompt(chunk_text: str, *, rel_path: str = "", context_budget: dict[str, Any] | None = None) -> str:
    max_evidence_chars = int((context_budget or {}).get("max_evidence_chars") or 0)
    max_array_items = int((context_budget or {}).get("max_array_items") or 0)
    source_span_policy = str((context_budget or {}).get("source_span_policy") or "")
    evidence_budget_text = (
        f" Each evidence_text item must be at most {max_evidence_chars} characters."
        if max_evidence_chars > 0
        else ""
    )
    array_budget_text = f" Each JSON array must contain at most {max_array_items} items." if max_array_items > 0 else ""
    source_span_text = (
        " The JSON schema constrains condition and argument evidence_text to deterministic source-span options; "
        "choose one exact listed source span or ''. "
        if source_span_policy
        else ""
    )
    return (
        "JSON only. Convert the raw text chunk into one source-grounded DRS object. "
        "Every semantic decision must be represented as referents, boxes, conditions, temporal_records, "
        "and identity_hypotheses. Do not answer questions, use outside knowledge, infer hidden answers, "
        "or use handler names. The root asserted box should have id b0 and parent_id ''. Use subordinate boxes "
        "for negation, reports, quotes, beliefs, conditionals, uncertainty, dreams, fiction, and modality. "
        "A condition must not use target_kind=box with target_id equal to its own box_id; scoped complements "
        "belong in a distinct subordinate box whose parent_id is the containing box. "
        "Do not create boxes only to stand for ordinary events; boxes are for scoped DRS contexts. If an event "
        "complement is not itself a scoped DRS, represent it as a grounded literal argument or as a declared "
        "condition referenced with target_kind=condition. "
        "Arguments use target_kind and target_id; use target_kind=box when an argument is a subordinate DRS box, "
        "target_kind=condition when an argument is another condition, and target_kind=referent for discourse "
        "referents. Identity hypotheses must be model-provided DRT data, not same-name merging; do not include "
        "self identity hypotheses where left_referent_id equals right_referent_id. Every target_id using "
        "target_kind referent, box, or condition must match an id declared in the corresponding array. If a grounded "
        "participant has no declared id, declare it first or use target_kind literal or unknown; never emit undeclared "
        "ids. Identity hypotheses must reference declared distinct referents and should be [] unless the source "
        "explicitly supports an identity, alias, or coreference link. Use temporal_records only for explicit "
        "source-grounded temporal or ordering phrases; otherwise temporal_id must be ''. "
        "For compact records, key/value lists, JSON-like objects, TSV/CSV rows, and log entries, still emit "
        "grounded DRS conditions for visible source-supported field/value or row structure; do not leave "
        "conditions empty solely because the chunk is terse or delimiter-heavy. "
        + source_span_text
        + "Every evidence_text item must be one contiguous substring copied exactly from the chunk."
        + evidence_budget_text
        + array_budget_text
        + " "
        "Copy each evidence substring at most once; never concatenate or repeat the chunk inside a string."
        + json.dumps(
            {
                "source_id": rel_path,
                "schema_version": CHUNK_DRS_SCHEMA_VERSION,
                "context_budget": context_budget or {},
                "required_top_shape": {
                    "drs": {
                        "schema_version": CHUNK_DRS_SCHEMA_VERSION,
                        "source_id": rel_path,
                        "referents": [],
                        "boxes": [],
                        "conditions": [],
                        "identity_hypotheses": [],
                        "temporal_records": [],
                    }
                },
                "chunk": chunk_text,
            },
            ensure_ascii=False,
        )
    )


def build_chunk_drs_skeleton_prompt(chunk_text: str, *, rel_path: str = "", context_budget: dict[str, Any] | None = None) -> str:
    source_span_candidates = (context_budget or {}).get("source_span_candidates")
    span_candidate_text = (
        "When source_span_candidates are provided, each evidence_text must be exactly one listed source span or ''. "
        if isinstance(source_span_candidates, list) and source_span_candidates
        else ""
    )
    return (
        "JSON only. Stage 1 of source-grounded DRS extraction. Extract only declared discourse referents "
        "DRS boxes, and explicit temporal records from the chunk. Do not emit conditions, identity hypotheses, answers, "
        "outside knowledge, or handler names. Declare one root asserted box with id b0 and parent_id ''. Use "
        "stable referent ids r0, r1, ...; box ids b0, b1, ...; and temporal ids t0, t1, ... in order. Use "
        "subordinate boxes only for scoped DRS contexts such as reports, quotes, beliefs, negation, conditionals, "
        "uncertainty, dreams, fiction, and modality; subordinate boxes must be distinct from the containing box. "
        "When a scoped context contains embedded proposition content, declare a distinct subordinate box for that "
        "content so stage 2 can reference it; do not require a condition to point back to its own box. "
        + span_candidate_text
        + "Every evidence_text item must be one contiguous substring "
        "copied exactly from the chunk."
        + json.dumps(
            {
                "source_id": rel_path,
                "schema_version": CHUNK_DRS_SCHEMA_VERSION,
                "context_budget": context_budget or {},
                "required_top_shape": {
                    "drs_skeleton": {
                        "schema_version": CHUNK_DRS_SCHEMA_VERSION,
                        "source_id": rel_path,
                        "referents": [],
                        "boxes": [],
                        "temporal_records": [],
                    }
                },
                "chunk": chunk_text,
            },
            ensure_ascii=False,
        )
    )


def build_chunk_drs_condition_prompt(
    chunk_text: str,
    *,
    rel_path: str,
    referents: list[dict[str, Any]],
    boxes: list[dict[str, Any]],
    temporal_records: list[dict[str, Any]] | None = None,
    context_budget: dict[str, Any] | None = None,
) -> str:
    source_span_candidates = (context_budget or {}).get("source_span_candidates")
    span_candidate_text = (
        "When source_span_candidates are provided, each evidence_text must be exactly one listed source span or ''. "
        if isinstance(source_span_candidates, list) and source_span_candidates
        else ""
    )
    return (
        "JSON only. Stage 2 of source-grounded DRS extraction. Emit conditions using only the declared "
        "referent, box, and temporal ids. Do not invent ids; target_id is schema-constrained to declared ids or ''. "
        "Use stable condition ids c0, c1, c2, ... in order. "
        "If an argument is a literal phrase rather than a declared id, set target_id to '' and put the exact "
        "phrase in value and/or evidence_text. Do not emit identity hypotheses or temporal records in this stage. "
        "When a declared temporal record scopes a condition, set that condition's temporal_id to the declared "
        "temporal id; otherwise temporal_id must be ''. "
        "A condition must not point a target_kind=box argument at its own box_id; use a distinct declared "
        "subordinate box for scoped content, a declared condition, or a literal argument. "
        "For compact records, key/value lists, JSON-like objects, TSV/CSV rows, and log entries, emit grounded "
        "conditions for visible source-supported field/value or row structure when declared referents or literals "
        "can participate. "
        + span_candidate_text
        + "Every evidence_text item must be one contiguous substring copied exactly from the chunk."
        + json.dumps(
            {
                "source_id": rel_path,
                "schema_version": CHUNK_DRS_SCHEMA_VERSION,
                "context_budget": context_budget or {},
                "declared_referents": referents,
                "declared_boxes": boxes,
                "declared_temporal_records": temporal_records or [],
                "required_top_shape": {
                    "condition_stage": {
                        "schema_version": CHUNK_DRS_SCHEMA_VERSION,
                        "source_id": rel_path,
                        "conditions": [],
                    }
                },
                "chunk": chunk_text,
            },
            ensure_ascii=False,
        )
    )


def build_chunk_drs_box_completion_prompt(
    chunk_text: str,
    *,
    rel_path: str,
    candidate_drs: dict[str, Any],
    validation_errors: list[str],
    missing_box_ids: list[str],
    context_budget: dict[str, Any] | None = None,
) -> str:
    return (
        "JSON only. Complete missing source-grounded DRS box declarations for an otherwise model-produced DRS. "
        "This is a structural DRT repair call, not question answering. Do not add referents, conditions, "
        "identity hypotheses, hidden answers, outside knowledge, or handler names. Emit only boxes for ids listed "
        "in missing_box_ids when the source supports that referenced DRS box; otherwise emit an empty boxes array. "
        "Each box evidence_text must be one exact contiguous substring from the chunk. Parent ids and holder ids "
        "must use declared ids. For scoped complements such as beliefs, reports, quotes, negation, conditionals, "
        "uncertainty, dreams, fiction, or modality, a missing content box may be subordinate to the containing box. "
        + json.dumps(
            {
                "source_id": rel_path,
                "schema_version": CHUNK_DRS_SCHEMA_VERSION,
                "context_budget": context_budget or {},
                "missing_box_ids": missing_box_ids,
                "validation_errors": validation_errors[:50],
                "candidate_drs": candidate_drs,
                "required_top_shape": {
                    "box_completion": {
                        "schema_version": CHUNK_DRS_SCHEMA_VERSION,
                        "source_id": rel_path,
                        "boxes": [],
                    }
                },
                "chunk": chunk_text,
            },
            ensure_ascii=False,
        )
    )


def _context_limited_chunk_drs_text(
    chunk_text: str,
    client: LocalModelClient,
    *,
    rel_path: str,
    n_predict: int,
) -> tuple[str, dict[str, Any]]:
    context_size = _client_context_size(client)
    budget: dict[str, Any] = {
        "runtime_context_size": context_size,
        "reserved_output_tokens": int(n_predict),
        "context_source": "client_metadata" if context_size > 0 else "unavailable",
    }
    if context_size <= 0:
        configured_chars = os.environ.get("KMD_CHUNK_DRS_MAX_CHARS")
        if configured_chars:
            try:
                max_chars = max(1, int(configured_chars))
            except ValueError:
                max_chars = len(chunk_text)
            limited = chunk_text[:max_chars]
        else:
            limited = chunk_text
        budget.update(
            {
                "prompt_budget_tokens": 0,
                "prompt_overhead_tokens": 0,
                "chunk_budget_tokens": _estimate_tokens(limited),
                "input_chars": len(chunk_text),
                "prompt_chunk_chars": len(limited),
                "max_evidence_chars": chunk_drs_evidence_max_chars(limited, n_predict),
                "max_array_items": chunk_drs_array_max_items(n_predict),
                "input_truncated": len(limited) < len(chunk_text),
            }
        )
        return limited, budget
    seed_budget = {**budget, "prompt_budget_tokens": max(0, context_size - int(n_predict)), "chunk_budget_tokens": 0}
    overhead_tokens = _estimate_tokens(build_chunk_drs_prompt("", rel_path=rel_path, context_budget=seed_budget))
    prompt_budget_tokens = max(0, context_size - int(n_predict) - overhead_tokens)
    max_chars = max(0, prompt_budget_tokens * 4)
    limited = chunk_text[:max_chars] if max_chars else ""
    budget.update(
        {
            "prompt_budget_tokens": prompt_budget_tokens,
            "prompt_overhead_tokens": overhead_tokens,
            "chunk_budget_tokens": _estimate_tokens(limited),
            "input_chars": len(chunk_text),
            "prompt_chunk_chars": len(limited),
            "max_evidence_chars": chunk_drs_evidence_max_chars(limited, n_predict),
            "max_array_items": chunk_drs_array_max_items(n_predict),
            "input_truncated": len(limited) < len(chunk_text),
        }
    )
    return limited, budget


def _validate_chunk_drs_payload(payload: Any, source_text: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("drs"), dict):
        return {"schema_valid": False, "errors": ["missing_drs_object"], "grounding_failures": []}
    drs = payload["drs"]
    errors: list[str] = []
    grounding_failures: list[str] = []

    def collection(name: str) -> list[dict[str, Any]]:
        value = drs.get(name)
        if not isinstance(value, list):
            errors.append(f"not_list:{name}")
            return []
        return [item for item in value if isinstance(item, dict)]

    referents = collection("referents")
    boxes = collection("boxes")
    conditions = collection("conditions")
    identities = collection("identity_hypotheses")
    temporals = collection("temporal_records")
    evidence_spans = drs.get("evidence_spans", [])
    if evidence_spans is None:
        evidence_spans = []
    if not isinstance(evidence_spans, list):
        errors.append("not_list:evidence_spans")
        evidence_spans = []

    referent_ids = {str(item.get("id") or "") for item in referents if str(item.get("id") or "")}
    box_ids = {str(item.get("id") or "") for item in boxes if str(item.get("id") or "")}
    condition_ids = {str(item.get("id") or "") for item in conditions if str(item.get("id") or "")}
    temporal_ids = {str(item.get("id") or "") for item in temporals if str(item.get("id") or "")}

    def check_span(value: Any, label: str) -> None:
        span = str(value or "").strip()
        if span and span not in source_text:
            grounding_failures.append(f"{label}:{span[:100]}")

    if not box_ids:
        errors.append("missing_box")
    for span in evidence_spans:
        check_span(span, "evidence_spans")
    for item in referents:
        ref_id = str(item.get("id") or "")
        if not ref_id or not str(item.get("label") or "").strip():
            errors.append(f"bad_referent:{ref_id}")
        check_span(item.get("evidence_text"), f"referent:{ref_id}")
    for item in boxes:
        box_id = str(item.get("id") or "")
        parent_id = str(item.get("parent_id") or "")
        holder_id = str(item.get("holder_referent_id") or "")
        if str(item.get("kind") or "") not in DRS_CONTEXT_KINDS:
            errors.append(f"bad_box_kind:{box_id}:{item.get('kind')}")
        if parent_id and parent_id not in box_ids:
            errors.append(f"missing_parent_box:{box_id}->{parent_id}")
        if parent_id and parent_id == box_id:
            errors.append(f"self_parent_box:{box_id}")
        if holder_id and holder_id not in referent_ids:
            errors.append(f"missing_holder_referent:{box_id}->{holder_id}")
        check_span(item.get("evidence_text"), f"box:{box_id}")
    for item in temporals:
        temporal_id = str(item.get("id") or "")
        if not temporal_id or not str(item.get("value") or "").strip():
            errors.append(f"bad_temporal:{temporal_id}")
        check_span(item.get("evidence_text"), f"temporal:{temporal_id}")
    for item in conditions:
        condition_id = str(item.get("id") or "")
        box_id = str(item.get("box_id") or "")
        temporal_id = str(item.get("temporal_id") or "")
        if not condition_id or not str(item.get("predicate") or "").strip():
            errors.append(f"bad_condition:{condition_id}")
        if box_id not in box_ids:
            errors.append(f"missing_condition_box:{condition_id}->{box_id}")
        if str(item.get("polarity") or "") not in DRS_POLARITIES:
            errors.append(f"bad_polarity:{condition_id}:{item.get('polarity')}")
        if str(item.get("modality") or "") not in DRS_CONTEXT_KINDS:
            errors.append(f"bad_modality:{condition_id}:{item.get('modality')}")
        if temporal_id and temporal_id not in temporal_ids:
            errors.append(f"missing_temporal:{condition_id}->{temporal_id}")
        check_span(item.get("evidence_text"), f"condition:{condition_id}")
        arguments = item.get("arguments")
        if not isinstance(arguments, list):
            errors.append(f"bad_arguments:{condition_id}")
            continue
        for arg in arguments:
            if not isinstance(arg, dict):
                continue
            target_kind = str(arg.get("target_kind") or "")
            target_id = str(arg.get("target_id") or "")
            if target_kind == "referent" and target_id and target_id not in referent_ids:
                errors.append(f"missing_argument_referent:{condition_id}->{target_id}")
            elif target_kind == "box" and target_id and target_id not in box_ids:
                errors.append(f"missing_argument_box:{condition_id}->{target_id}")
            elif target_kind == "box" and target_id and target_id == box_id:
                errors.append(f"self_argument_box:{condition_id}->{target_id}")
            elif target_kind == "condition" and target_id and target_id not in condition_ids:
                errors.append(f"missing_argument_condition:{condition_id}->{target_id}")
            elif target_kind == "condition" and target_id and target_id == condition_id:
                errors.append(f"self_argument_condition:{condition_id}->{target_id}")
            elif target_kind in {"literal", "unknown"} and target_id:
                errors.append(f"literal_argument_has_target_id:{condition_id}->{target_id}")
            elif target_kind not in {"referent", "box", "condition", "literal", "unknown"}:
                errors.append(f"bad_argument_target_kind:{condition_id}:{target_kind}")
            check_span(arg.get("evidence_text"), f"argument:{condition_id}:{arg.get('role')}")
    for item in identities:
        left_id = str(item.get("left_referent_id") or "")
        right_id = str(item.get("right_referent_id") or "")
        if left_id not in referent_ids:
            errors.append(f"missing_identity_left:{left_id}")
        if right_id not in referent_ids:
            errors.append(f"missing_identity_right:{right_id}")
        if str(item.get("status") or "") not in DRS_IDENTITY_STATUSES:
            errors.append(f"bad_identity_status:{item.get('status')}")
        check_span(item.get("evidence_text"), f"identity:{left_id}:{right_id}")
    return {
        "schema_valid": not errors and not grounding_failures,
        "errors": errors[:50],
        "grounding_failures": grounding_failures[:50],
        "grounding_failure_count": len(grounding_failures),
        "referent_count": len(referents),
        "box_count": len(boxes),
        "condition_count": len(conditions),
        "identity_hypothesis_count": len(identities),
        "temporal_record_count": len(temporals),
    }


def _repair_evidence_text_from_declared_value(
    item: dict[str, Any],
    source_text: str,
    field_names: tuple[str, ...],
) -> bool:
    evidence_text = str(item.get("evidence_text") or "").strip()
    if not source_text or (evidence_text and evidence_text in source_text):
        return False
    if evidence_text:
        for candidate in (
            evidence_text.replace('\\"', '"'),
            evidence_text.replace("\\/", "/"),
            evidence_text.replace('\\"', '"').replace("\\/", "/"),
        ):
            if candidate and candidate in source_text:
                item["evidence_text"] = candidate
                return True
    for field_name in field_names:
        candidate = str(item.get(field_name) or "").strip()
        if candidate and candidate in source_text:
            item["evidence_text"] = candidate
            return True
    return False


def _repair_chunk_drs_payload(payload: Any, source_text: str = "", *, prune_unreferenced_temporals: bool = True) -> Any:
    if not isinstance(payload, dict) or not isinstance(payload.get("drs"), dict):
        return payload
    drs = {**payload["drs"]}
    referents = drs.get("referents")
    boxes = drs.get("boxes")
    conditions = drs.get("conditions")
    if not isinstance(referents, list) or not isinstance(boxes, list) or not isinstance(conditions, list):
        return payload
    repaired_referents = [item for item in referents if isinstance(item, dict)]
    repaired_boxes = [item for item in boxes if isinstance(item, dict)]
    repaired_conditions = [item for item in conditions if isinstance(item, dict)]
    referent_ids = {str(item.get("id") or "") for item in repaired_referents}
    referents_by_id = {str(item.get("id") or ""): item for item in repaired_referents if str(item.get("id") or "")}
    box_ids = {str(item.get("id") or "") for item in repaired_boxes if str(item.get("id") or "")}
    namespace_repaired = False
    grounding_repaired = False
    if source_text:
        for item in repaired_referents:
            grounding_repaired |= _repair_evidence_text_from_declared_value(item, source_text, ("label",))
        for item in repaired_boxes:
            grounding_repaired |= _repair_evidence_text_from_declared_value(item, source_text, ())
        for item in repaired_conditions:
            grounding_repaired |= _repair_evidence_text_from_declared_value(item, source_text, ())
        temporals = drs.get("temporal_records")
        if isinstance(temporals, list):
            for item in temporals:
                if isinstance(item, dict):
                    grounding_repaired |= _repair_evidence_text_from_declared_value(item, source_text, ("value",))
    temporal_records = drs.get("temporal_records")
    repaired_temporals = temporal_records
    temporal_repaired = False
    if prune_unreferenced_temporals and isinstance(temporal_records, list):
        referenced_temporal_ids = {
            str(condition.get("temporal_id") or "").strip()
            for condition in repaired_conditions
            if str(condition.get("temporal_id") or "").strip()
        }
        repaired_temporals = [
            item
            for item in temporal_records
            if isinstance(item, dict) and str(item.get("id") or "").strip() in referenced_temporal_ids
        ]
        if len(repaired_temporals) != len(temporal_records):
            drs["temporal_records"] = repaired_temporals
            temporal_repaired = True
    for condition in repaired_conditions:
        if not isinstance(condition.get("arguments"), list):
            continue
        for argument in condition["arguments"]:
            if not isinstance(argument, dict):
                continue
            if source_text:
                grounding_repaired |= _repair_evidence_text_from_declared_value(argument, source_text, ("value",))
            target_id = str(argument.get("target_id") or "").strip()
            target_kind = str(argument.get("target_kind") or "").strip()
            if target_id in box_ids and target_kind != "box":
                argument["target_kind"] = "box"
                namespace_repaired = True
                continue
            if target_id in referent_ids and target_kind != "referent":
                argument["target_kind"] = "referent"
                namespace_repaired = True
                target_kind = "referent"
            if target_kind in {"literal", "unknown"} and target_id:
                argument["target_id"] = ""
                namespace_repaired = True
                target_id = ""
            if str(argument.get("target_kind") or "") != "referent":
                continue
            value = str(argument.get("value") or "").strip()
            evidence_text = str(argument.get("evidence_text") or "").strip()
            if not target_id or target_id in referent_ids or not value:
                continue
            repaired_referents.append(
                {
                    "id": target_id,
                    "label": value,
                    "kind": str(argument.get("value_type") or "unknown") or "unknown",
                    "evidence_text": evidence_text or value,
                }
            )
            referent_ids.add(target_id)
    identities = drs.get("identity_hypotheses")
    repaired_identities = identities
    if isinstance(identities, list):
        repaired_identities = []
        for item in identities:
            if not isinstance(item, dict):
                continue
            left_id = str(item.get("left_referent_id") or "").strip()
            right_id = str(item.get("right_referent_id") or "").strip()
            if left_id and left_id == right_id:
                continue
            if source_text:
                evidence_text = str(item.get("evidence_text") or "").strip()

                def supported_by_evidence(ref_id: str) -> bool:
                    referent = referents_by_id.get(ref_id)
                    if not referent:
                        return False
                    surfaces = [str(referent.get("label") or "").strip(), str(referent.get("evidence_text") or "").strip()]
                    return any(surface and surface in evidence_text for surface in surfaces)

                if not evidence_text or not supported_by_evidence(left_id) or not supported_by_evidence(right_id):
                    continue
            repaired_identities.append(item)
        if len(repaired_identities) != len(identities):
            drs["identity_hypotheses"] = repaired_identities
    if (
        len(repaired_referents) == len(referents)
        and len(repaired_boxes) == len(boxes)
        and len(repaired_conditions) == len(conditions)
        and not temporal_repaired
        and repaired_identities is identities
        and not namespace_repaired
        and not grounding_repaired
    ):
        return payload
    drs["referents"] = repaired_referents
    drs["boxes"] = repaired_boxes
    drs["conditions"] = repaired_conditions
    if temporal_repaired:
        drs["temporal_records"] = repaired_temporals
    return {**payload, "drs": drs}


def _drs_exact_span_failures(items: list[dict[str, Any]], source_text: str, prefix: str) -> list[str]:
    failures: list[str] = []
    for item in items:
        span = str(item.get("evidence_text") or "").strip()
        item_id = str(item.get("id") or item.get("role") or "")
        if span and span not in source_text:
            failures.append(f"{prefix}:{item_id}:{span[:100]}")
    return failures


def _complete_chunk_drs_stage(
    client: LocalModelClient,
    cache_path: Path | None,
    prompt: str,
    schema: dict[str, Any],
    *,
    stage: str,
    n_predict: int,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    constraint = _constraint_settings(CHUNK_DRS_GRAMMAR, schema, CHUNK_DRS_SCHEMA_VERSION)
    prompt_hash = _cache_hash(
        stage,
        prompt,
        client,
        {
            "n_predict": n_predict,
            "schema": CHUNK_DRS_SCHEMA_VERSION,
            "stage_failure_cache_policy": CHUNK_DRS_STAGE_FAILURE_CACHE_POLICY,
            **constraint,
        },
    )
    path = cache_path.parent / f"{prompt_hash}.json" if cache_path is not None else None
    cached = _read_cache(path)
    if cached is not None and cached.get("reason") != "request_failed":
        return cached, 0.0, {"prompt_hash": prompt_hash, **constraint}
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=CHUNK_DRS_GRAMMAR,
            json_schema=schema,
        )
    except LocalModelJSONError as exc:
        elapsed = round(time.time() - start, 3)
        payload = {
            "accepted": False,
            "reason": "invalid_json",
            "error": str(exc),
            "raw_text": exc.raw_text,
            "raw_snippet": exc.snippet[:4000],
            "elapsed": elapsed,
            "prompt_hash": prompt_hash,
            **constraint,
        }
        _write_cache(path, payload)
        return (
            payload,
            elapsed,
            {"prompt_hash": prompt_hash, **constraint},
        )
    except Exception as exc:
        return (
            {"accepted": False, "reason": "request_failed", "error": str(exc), "raw_text": ""},
            round(time.time() - start, 3),
            {"prompt_hash": prompt_hash, **constraint},
        )
    elapsed = parsed.get("_model_elapsed_seconds", round(time.time() - start, 3))
    _write_cache(path, parsed)
    return parsed, float(elapsed), {"prompt_hash": prompt_hash, **constraint}


def _missing_argument_box_ids(validation: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for error in validation.get("errors") or []:
        text = str(error or "")
        if not text.startswith("missing_argument_box:") or "->" not in text:
            continue
        box_id = text.rsplit("->", 1)[-1].strip()
        if box_id and box_id not in ids:
            ids.append(box_id)
    return ids


def _call_model_chunk_drs_box_completion(
    prompt_chunk: str,
    client: LocalModelClient,
    *,
    rel_path: str,
    n_predict: int,
    context_budget: dict[str, Any],
    cache_path: Path | None,
    payload: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    drs = payload.get("drs") if isinstance(payload, dict) else None
    if not isinstance(drs, dict):
        return {"accepted": False, "reason": "missing_drs_object", "stage": "box_completion"}
    missing_box_ids = _missing_argument_box_ids(validation)
    if not missing_box_ids:
        return {"accepted": False, "reason": "no_missing_argument_box", "stage": "box_completion"}
    boxes = [item for item in drs.get("boxes", []) if isinstance(item, dict)] if isinstance(drs.get("boxes"), list) else []
    referents = (
        [item for item in drs.get("referents", []) if isinstance(item, dict)]
        if isinstance(drs.get("referents"), list)
        else []
    )
    existing_box_ids = [str(item.get("id") or "") for item in boxes if str(item.get("id") or "")]
    referent_ids = [str(item.get("id") or "") for item in referents if str(item.get("id") or "")]
    missing_box_ids = [box_id for box_id in missing_box_ids if box_id not in existing_box_ids]
    if not missing_box_ids or not existing_box_ids:
        return {"accepted": False, "reason": "no_completable_missing_box", "stage": "box_completion"}
    box_n_predict = default_chunk_drs_box_completion_n_predict(n_predict)
    prompt = build_chunk_drs_box_completion_prompt(
        prompt_chunk,
        rel_path=rel_path,
        candidate_drs=drs,
        validation_errors=[str(error) for error in validation.get("errors") or []],
        missing_box_ids=missing_box_ids,
        context_budget=context_budget,
    )
    schema = chunk_drs_box_completion_json_schema(
        source_id=rel_path,
        missing_box_ids=missing_box_ids,
        existing_box_ids=existing_box_ids,
        referent_ids=referent_ids,
        max_boxes=len(missing_box_ids),
    )
    completion, elapsed, constraint = _complete_chunk_drs_stage(
        client,
        cache_path,
        prompt,
        schema,
        stage="chunk_drs_box_completion",
        n_predict=box_n_predict,
    )
    completion_payload = completion.get("box_completion") if isinstance(completion, dict) else None
    if not isinstance(completion_payload, dict):
        return {
            "accepted": False,
            "reason": str(completion.get("reason") or "schema_validation_failed")
            if isinstance(completion, dict)
            else "schema_validation_failed",
            "stage": "box_completion",
            "raw_text": str(completion.get("raw_text") or completion.get("_model_raw") or "")
            if isinstance(completion, dict)
            else "",
            "elapsed": elapsed,
            "box_completion_n_predict": box_n_predict,
            **constraint,
        }
    new_boxes = completion_payload.get("boxes")
    new_boxes = [item for item in new_boxes if isinstance(item, dict)] if isinstance(new_boxes, list) else []
    allowed_missing = set(missing_box_ids)
    new_boxes = [item for item in new_boxes if str(item.get("id") or "") in allowed_missing]
    if not new_boxes:
        return {
            "accepted": False,
            "reason": "empty_box_completion",
            "stage": "box_completion",
            "raw_text": str(completion.get("_model_raw") or "") if isinstance(completion, dict) else "",
            "elapsed": elapsed,
            "box_completion_n_predict": box_n_predict,
            **constraint,
        }
    merged = {
        **payload,
        "drs": {
            **drs,
            "boxes": [*boxes, *new_boxes],
        },
    }
    merged = _repair_chunk_drs_payload(merged, prompt_chunk)
    merged_validation = _validate_chunk_drs_payload(merged, prompt_chunk)
    if not merged_validation.get("schema_valid"):
        reason = "grounding_validation_failed" if merged_validation.get("grounding_failure_count") else "schema_validation_failed"
        return {
            "accepted": False,
            "reason": reason,
            "stage": "box_completion",
            "raw_text": str(completion.get("_model_raw") or "") if isinstance(completion, dict) else "",
            "elapsed": elapsed,
            "validation": merged_validation,
            "box_completion_n_predict": box_n_predict,
            **constraint,
        }
    raw = json.dumps(
        {
            "candidate": json.dumps(drs, sort_keys=True),
            "box_completion": completion.get("_model_raw") if isinstance(completion, dict) else "",
        },
        sort_keys=True,
    )
    return {
        "accepted": True,
        "reason": "box_completion_repair",
        "drs": merged["drs"],
        "raw_text": raw,
        "elapsed": elapsed,
        "prompt_hash": constraint.get("prompt_hash"),
        "constraint_mode": constraint.get("constraint_mode"),
        "validation": merged_validation,
        "context_budget": {
            **context_budget,
            "box_completion_policy": CHUNK_DRS_BOX_COMPLETION_POLICY,
            "box_completion_n_predict": box_n_predict,
        },
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
        "box_completion": {
            "accepted": True,
            "missing_box_ids": missing_box_ids,
            "added_box_count": len(new_boxes),
            "prompt_hash": constraint.get("prompt_hash"),
        },
    }


def _call_model_chunk_drs_staged(
    prompt_chunk: str,
    client: LocalModelClient,
    *,
    rel_path: str,
    n_predict: int,
    context_budget: dict[str, Any],
    cache_path: Path | None,
) -> dict[str, Any]:
    skeleton_n_predict = default_staged_chunk_drs_skeleton_n_predict(n_predict)
    condition_n_predict = default_staged_chunk_drs_condition_n_predict(n_predict)
    max_items = context_budget.get("max_array_items") or chunk_drs_array_max_items(n_predict)
    source_span_candidates = chunk_drs_source_span_candidates(
        prompt_chunk,
        context_budget.get("max_evidence_chars"),
    )
    skeleton_context_budget = {
        **context_budget,
        "source_span_policy": CHUNK_DRS_SOURCE_SPAN_POLICY,
        "skeleton_source_span_policy": CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY,
        "source_span_candidates": source_span_candidates,
    }
    skeleton_prompt = build_chunk_drs_skeleton_prompt(
        prompt_chunk,
        rel_path=rel_path,
        context_budget=skeleton_context_budget,
    )
    skeleton_schema = chunk_drs_skeleton_json_schema(rel_path, max_items, source_span_candidates)
    skeleton, skeleton_elapsed, skeleton_constraint = _complete_chunk_drs_stage(
        client,
        cache_path,
        skeleton_prompt,
        skeleton_schema,
        stage="chunk_drs_skeleton",
        n_predict=skeleton_n_predict,
    )
    skeleton_payload = skeleton.get("drs_skeleton") if isinstance(skeleton, dict) else None
    if not isinstance(skeleton_payload, dict):
        return {
            "accepted": False,
            "reason": str(skeleton.get("reason") or "schema_validation_failed") if isinstance(skeleton, dict) else "schema_validation_failed",
            "stage": "skeleton",
            "error": str(skeleton.get("error") or "") if isinstance(skeleton, dict) else "",
            "raw_snippet": str(skeleton.get("raw_snippet") or "") if isinstance(skeleton, dict) else "",
            "raw_text": str(skeleton.get("raw_text") or skeleton.get("_model_raw") or "") if isinstance(skeleton, dict) else "",
            "elapsed": skeleton_elapsed,
            "fresh_or_cached": str(skeleton.get("fresh_or_cached") or "fresh") if isinstance(skeleton, dict) else "fresh",
            **skeleton_constraint,
        }
    referents = skeleton_payload.get("referents")
    boxes = skeleton_payload.get("boxes")
    temporals = skeleton_payload.get("temporal_records")
    referents = [item for item in referents if isinstance(item, dict)] if isinstance(referents, list) else []
    boxes = [item for item in boxes if isinstance(item, dict)] if isinstance(boxes, list) else []
    temporals = [item for item in temporals if isinstance(item, dict)] if isinstance(temporals, list) else []
    skeleton_payload = _repair_chunk_drs_payload(
        {
            "drs": {
                "schema_version": CHUNK_DRS_SCHEMA_VERSION,
                "source_id": rel_path,
                "referents": referents,
                "boxes": boxes,
                "conditions": [],
                "identity_hypotheses": [],
                "temporal_records": temporals,
            }
        },
        prompt_chunk,
        prune_unreferenced_temporals=False,
    )["drs"]
    referents = skeleton_payload["referents"]
    boxes = skeleton_payload["boxes"]
    temporals = skeleton_payload["temporal_records"]
    skeleton_span_failures = (
        _drs_exact_span_failures(referents, prompt_chunk, "referent")
        + _drs_exact_span_failures(boxes, prompt_chunk, "box")
        + _drs_exact_span_failures(temporals, prompt_chunk, "temporal")
    )
    if skeleton_span_failures:
        return {
            "accepted": False,
            "reason": "grounding_validation_failed",
            "stage": "skeleton",
            "grounding_failures": skeleton_span_failures[:50],
            "elapsed": skeleton_elapsed,
            **skeleton_constraint,
        }
    box_ids = [str(item.get("id") or "") for item in boxes if str(item.get("id") or "")]
    referent_ids = [str(item.get("id") or "") for item in referents if str(item.get("id") or "")]
    temporal_ids = [str(item.get("id") or "") for item in temporals if str(item.get("id") or "")]
    condition_context_budget = {
        **context_budget,
        "source_span_policy": CHUNK_DRS_SOURCE_SPAN_POLICY,
        "skeleton_source_span_policy": CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY,
        "source_span_candidates": source_span_candidates,
    }
    condition_prompt = build_chunk_drs_condition_prompt(
        prompt_chunk,
        rel_path=rel_path,
        referents=referents,
        boxes=boxes,
        temporal_records=temporals,
        context_budget=condition_context_budget,
    )
    condition_schema = chunk_drs_condition_json_schema(
        source_id=rel_path,
        box_ids=box_ids,
        referent_ids=referent_ids,
        temporal_ids=temporal_ids,
        max_conditions=max_items,
        max_arguments=max_items,
        evidence_text_values=source_span_candidates,
    )
    condition_stage, condition_elapsed, condition_constraint = _complete_chunk_drs_stage(
        client,
        cache_path,
        condition_prompt,
        condition_schema,
        stage="chunk_drs_conditions",
        n_predict=condition_n_predict,
    )
    condition_payload = condition_stage.get("condition_stage") if isinstance(condition_stage, dict) else None
    if not isinstance(condition_payload, dict):
        return {
            "accepted": False,
            "reason": str(condition_stage.get("reason") or "schema_validation_failed") if isinstance(condition_stage, dict) else "schema_validation_failed",
            "stage": "conditions",
            "error": str(condition_stage.get("error") or "") if isinstance(condition_stage, dict) else "",
            "raw_snippet": str(condition_stage.get("raw_snippet") or "") if isinstance(condition_stage, dict) else "",
            "raw_text": str(condition_stage.get("raw_text") or condition_stage.get("_model_raw") or "") if isinstance(condition_stage, dict) else "",
            "elapsed": skeleton_elapsed + condition_elapsed,
            "fresh_or_cached": str(condition_stage.get("fresh_or_cached") or "fresh")
            if isinstance(condition_stage, dict)
            else "fresh",
            **condition_constraint,
        }
    conditions = condition_payload.get("conditions")
    conditions = [item for item in conditions if isinstance(item, dict)] if isinstance(conditions, list) else []
    merged = {
        "drs": {
            "schema_version": CHUNK_DRS_SCHEMA_VERSION,
            "source_id": rel_path,
            "referents": referents,
            "boxes": boxes,
            "conditions": conditions,
            "identity_hypotheses": [],
            "temporal_records": temporals,
        }
    }
    merged = _repair_chunk_drs_payload(merged, prompt_chunk)
    validation = _validate_chunk_drs_payload(merged, prompt_chunk)
    elapsed = skeleton_elapsed + condition_elapsed
    if not validation.get("schema_valid"):
        box_completion = _call_model_chunk_drs_box_completion(
            prompt_chunk,
            client,
            rel_path=rel_path,
            n_predict=n_predict,
            context_budget=context_budget,
            cache_path=cache_path,
            payload=merged,
            validation=validation,
        )
        if box_completion.get("accepted"):
            raw = json.dumps(
                {
                    "skeleton": skeleton.get("_model_raw") if isinstance(skeleton, dict) else "",
                    "conditions": condition_stage.get("_model_raw") if isinstance(condition_stage, dict) else "",
                    "box_completion": box_completion.get("raw_text") or "",
                },
                sort_keys=True,
            )
            staged_prompt_hash = hashlib.sha256(
                json.dumps(
                    {
                        "skeleton_prompt_hash": skeleton_constraint.get("prompt_hash"),
                        "condition_prompt_hash": condition_constraint.get("prompt_hash"),
                        "box_completion_prompt_hash": box_completion.get("prompt_hash"),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            return {
                "accepted": True,
                "reason": "staged_fallback",
                "drs": box_completion["drs"],
                "raw_text": raw,
                "elapsed": elapsed + float(box_completion.get("elapsed") or 0.0),
                "prompt_hash": staged_prompt_hash,
                "constraint_mode": condition_constraint.get("constraint_mode"),
                "validation": box_completion.get("validation"),
                "context_budget": {
                    **context_budget,
                    "staged_fallback_policy": CHUNK_DRS_STAGED_FALLBACK_POLICY,
                    "grounding_repair_policy": CHUNK_DRS_GROUNDING_REPAIR_POLICY,
                    "identity_provenance_policy": CHUNK_DRS_IDENTITY_PROVENANCE_POLICY,
                    "temporal_provenance_policy": CHUNK_DRS_TEMPORAL_PROVENANCE_POLICY,
                    "sparse_retry_policy": CHUNK_DRS_SPARSE_RETRY_POLICY,
                    "structure_validation_policy": CHUNK_DRS_STRUCTURE_VALIDATION_POLICY,
                    "box_completion_policy": CHUNK_DRS_BOX_COMPLETION_POLICY,
                    "source_span_policy": CHUNK_DRS_SOURCE_SPAN_POLICY,
                    "skeleton_source_span_policy": CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY,
                    "skeleton_id_policy": CHUNK_DRS_SKELETON_ID_POLICY,
                    "staged_skeleton_n_predict": skeleton_n_predict,
                    "staged_condition_n_predict": condition_n_predict,
                    "box_completion_n_predict": box_completion["context_budget"]["box_completion_n_predict"],
                },
                "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
                "fresh_or_cached": "fresh",
                "staged": True,
                "box_completion": box_completion.get("box_completion"),
            }
        reason = "grounding_validation_failed" if validation.get("grounding_failure_count") else "schema_validation_failed"
        return {
            "accepted": False,
            "reason": reason,
            "stage": "merged",
            "validation": validation,
            "box_completion": {
                "accepted": False,
                "reason": box_completion.get("reason"),
                "stage": box_completion.get("stage"),
            },
            "elapsed": elapsed,
            **condition_constraint,
        }
    raw = json.dumps(
        {
            "skeleton": skeleton.get("_model_raw") if isinstance(skeleton, dict) else "",
            "conditions": condition_stage.get("_model_raw") if isinstance(condition_stage, dict) else "",
        },
        sort_keys=True,
    )
    staged_prompt_hash = hashlib.sha256(
        json.dumps(
            {
                "skeleton_prompt_hash": skeleton_constraint.get("prompt_hash"),
                "condition_prompt_hash": condition_constraint.get("prompt_hash"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "accepted": True,
        "reason": "staged_fallback",
        "drs": merged["drs"],
        "raw_text": raw,
        "elapsed": elapsed,
        "prompt_hash": staged_prompt_hash,
        "constraint_mode": condition_constraint.get("constraint_mode"),
        "validation": validation,
        "context_budget": {
            **context_budget,
            "staged_fallback_policy": CHUNK_DRS_STAGED_FALLBACK_POLICY,
            "grounding_repair_policy": CHUNK_DRS_GROUNDING_REPAIR_POLICY,
            "identity_provenance_policy": CHUNK_DRS_IDENTITY_PROVENANCE_POLICY,
            "temporal_provenance_policy": CHUNK_DRS_TEMPORAL_PROVENANCE_POLICY,
            "sparse_retry_policy": CHUNK_DRS_SPARSE_RETRY_POLICY,
            "structure_validation_policy": CHUNK_DRS_STRUCTURE_VALIDATION_POLICY,
            "box_completion_policy": CHUNK_DRS_BOX_COMPLETION_POLICY,
            "source_span_policy": CHUNK_DRS_SOURCE_SPAN_POLICY,
            "skeleton_source_span_policy": CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY,
            "skeleton_id_policy": CHUNK_DRS_SKELETON_ID_POLICY,
            "staged_skeleton_n_predict": skeleton_n_predict,
            "staged_condition_n_predict": condition_n_predict,
            "box_completion_n_predict": default_chunk_drs_box_completion_n_predict(n_predict),
        },
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
        "staged": True,
    }


def chunk_drs_cache_context(client: LocalModelClient | None, *, n_predict: int | None = None) -> dict[str, Any]:
    if n_predict is None:
        n_predict = default_chunk_drs_n_predict(client)
    production_schema = chunk_drs_json_schema(include_auxiliary_fields=False)
    constraint = _constraint_settings(CHUNK_DRS_GRAMMAR, production_schema, CHUNK_DRS_SCHEMA_VERSION)
    return {
        "prompt_version": PROMPT_VERSION,
        "schema_version": CHUNK_DRS_SCHEMA_VERSION,
        "evidence_cap_policy": "min_chunk_or_reserved_output_quarter_96_256",
        "array_cap_policy": "reserved_output_tokens_div_96_4_10",
        "staged_fallback": _staged_chunk_drs_enabled(),
        "staged_fallback_policy": CHUNK_DRS_STAGED_FALLBACK_POLICY,
        "grounding_repair_policy": CHUNK_DRS_GROUNDING_REPAIR_POLICY,
        "identity_provenance_policy": CHUNK_DRS_IDENTITY_PROVENANCE_POLICY,
        "temporal_provenance_policy": CHUNK_DRS_TEMPORAL_PROVENANCE_POLICY,
        "sparse_retry_policy": CHUNK_DRS_SPARSE_RETRY_POLICY,
        "structure_validation_policy": CHUNK_DRS_STRUCTURE_VALIDATION_POLICY,
        "box_completion_policy": CHUNK_DRS_BOX_COMPLETION_POLICY,
        "source_span_policy": CHUNK_DRS_SOURCE_SPAN_POLICY,
        "skeleton_source_span_policy": CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY,
        "skeleton_id_policy": CHUNK_DRS_SKELETON_ID_POLICY,
        "monolithic_id_policy": CHUNK_DRS_MONOLITHIC_ID_POLICY,
        "compact_undercoverage_policy": CHUNK_DRS_COMPACT_UNDERCOVERAGE_POLICY,
        "staged_retry_diagnostics_policy": CHUNK_DRS_STAGED_RETRY_DIAGNOSTICS_POLICY,
        "stage_failure_cache_policy": CHUNK_DRS_STAGE_FAILURE_CACHE_POLICY,
        "staged_skeleton_n_predict": default_staged_chunk_drs_skeleton_n_predict(int(n_predict)),
        "staged_condition_n_predict": default_staged_chunk_drs_condition_n_predict(int(n_predict)),
        "box_completion_n_predict": default_chunk_drs_box_completion_n_predict(int(n_predict)),
        **constraint,
        "n_predict": int(n_predict),
        "model_fingerprint": _client_fingerprint(client),
    }


def call_model_chunk_drs(
    chunk_text: str,
    client: LocalModelClient,
    *,
    rel_path: str = "",
    n_predict: int | None = None,
) -> dict[str, Any]:
    if n_predict is None:
        n_predict = default_chunk_drs_n_predict(client)
    prompt_chunk, context_budget = _context_limited_chunk_drs_text(
        chunk_text,
        client,
        rel_path=rel_path,
        n_predict=n_predict,
    )
    source_span_candidates = chunk_drs_source_span_candidates(
        prompt_chunk,
        context_budget.get("max_evidence_chars"),
    )
    context_budget = {
        **context_budget,
        "source_span_policy": CHUNK_DRS_SOURCE_SPAN_POLICY,
        "source_span_candidate_count": len(source_span_candidates),
        "skeleton_source_span_policy": CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY,
        "monolithic_id_policy": CHUNK_DRS_MONOLITHIC_ID_POLICY,
        "compact_undercoverage_policy": CHUNK_DRS_COMPACT_UNDERCOVERAGE_POLICY,
        "staged_retry_diagnostics_policy": CHUNK_DRS_STAGED_RETRY_DIAGNOSTICS_POLICY,
        "stage_failure_cache_policy": CHUNK_DRS_STAGE_FAILURE_CACHE_POLICY,
    }
    prompt = build_chunk_drs_prompt(prompt_chunk, rel_path=rel_path, context_budget=context_budget)
    drs_json_schema = chunk_drs_json_schema(
        context_budget.get("max_evidence_chars"),
        context_budget.get("max_array_items"),
        include_auxiliary_fields=False,
        source_id=rel_path,
        evidence_text_values=source_span_candidates,
        constrain_stable_ids=True,
    )
    constraint = _constraint_settings(CHUNK_DRS_GRAMMAR, drs_json_schema, CHUNK_DRS_SCHEMA_VERSION)
    prompt_hash = _cache_hash(
        "chunk_drs",
        prompt,
        client,
        {
            "n_predict": n_predict,
            "schema": CHUNK_DRS_SCHEMA_VERSION,
            **constraint,
            "context_budget": context_budget,
            "staged_fallback": _staged_chunk_drs_enabled(),
            "staged_fallback_policy": CHUNK_DRS_STAGED_FALLBACK_POLICY,
            "grounding_repair_policy": CHUNK_DRS_GROUNDING_REPAIR_POLICY,
            "identity_provenance_policy": CHUNK_DRS_IDENTITY_PROVENANCE_POLICY,
            "temporal_provenance_policy": CHUNK_DRS_TEMPORAL_PROVENANCE_POLICY,
            "sparse_retry_policy": CHUNK_DRS_SPARSE_RETRY_POLICY,
            "structure_validation_policy": CHUNK_DRS_STRUCTURE_VALIDATION_POLICY,
            "box_completion_policy": CHUNK_DRS_BOX_COMPLETION_POLICY,
            "source_span_policy": CHUNK_DRS_SOURCE_SPAN_POLICY,
            "skeleton_source_span_policy": CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY,
            "skeleton_id_policy": CHUNK_DRS_SKELETON_ID_POLICY,
            "monolithic_id_policy": CHUNK_DRS_MONOLITHIC_ID_POLICY,
            "compact_undercoverage_policy": CHUNK_DRS_COMPACT_UNDERCOVERAGE_POLICY,
            "staged_retry_diagnostics_policy": CHUNK_DRS_STAGED_RETRY_DIAGNOSTICS_POLICY,
            "stage_failure_cache_policy": CHUNK_DRS_STAGE_FAILURE_CACHE_POLICY,
            "source_span_candidate_count": len(source_span_candidates),
            "staged_skeleton_n_predict": default_staged_chunk_drs_skeleton_n_predict(int(n_predict)),
            "staged_condition_n_predict": default_staged_chunk_drs_condition_n_predict(int(n_predict)),
            "box_completion_n_predict": default_chunk_drs_box_completion_n_predict(int(n_predict)),
        },
    )
    cache_path = _cache_path("KMD_CHUNK_DRS_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and cached.get("reason") != "request_failed":
        return cached
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=CHUNK_DRS_GRAMMAR,
            json_schema=drs_json_schema,
        )
    except LocalModelJSONError as exc:
        payload = {
            "accepted": False,
            "reason": "invalid_json",
            "error": str(exc),
            "raw_text": exc.raw_text,
            "raw_snippet": exc.snippet[:4000],
            "prompt_hash": prompt_hash,
            **constraint,
            "elapsed": round(time.time() - start, 3),
            "context_budget": context_budget,
        }
        if _staged_chunk_drs_enabled():
            fallback = _call_model_chunk_drs_staged(
                prompt_chunk,
                client,
                rel_path=rel_path,
                n_predict=n_predict,
                context_budget=context_budget,
                cache_path=cache_path,
            )
            if fallback.get("accepted"):
                payload = {**fallback, "fallback_from_reason": "invalid_json", "monolithic_prompt_hash": prompt_hash}
                _write_cache(cache_path, payload)
                return payload
            payload["staged_fallback"] = _staged_fallback_failure_summary(fallback)
        _write_cache(cache_path, payload)
        return payload
    except Exception as exc:
        raw_text = str(getattr(exc, "raw_text", "") or "")
        payload = {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "raw_text": raw_text,
            "raw_snippet": str(getattr(exc, "snippet", "") or raw_text)[:4000],
            "prompt_hash": prompt_hash,
            **constraint,
            "elapsed": round(time.time() - start, 3),
        }
        return payload
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    parsed = _repair_chunk_drs_payload(parsed, prompt_chunk)
    validation = _validate_chunk_drs_payload(parsed, prompt_chunk)
    if not validation.get("schema_valid"):
        reason = "grounding_validation_failed" if validation.get("grounding_failure_count") else "schema_validation_failed"
        monolithic_elapsed = float(parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)))
        staged_elapsed = 0.0
        payload = {
            "accepted": False,
            "reason": reason,
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            **constraint,
            "elapsed": monolithic_elapsed,
            "validation": validation,
            "context_budget": context_budget,
        }
        if _staged_chunk_drs_enabled():
            fallback = _call_model_chunk_drs_staged(
                prompt_chunk,
                client,
                rel_path=rel_path,
                n_predict=n_predict,
                context_budget=context_budget,
                cache_path=cache_path,
            )
            staged_elapsed = float(fallback.get("elapsed") or 0.0)
            if fallback.get("accepted"):
                payload = {**fallback, "fallback_from_reason": reason, "monolithic_prompt_hash": prompt_hash}
                _write_cache(cache_path, payload)
                return payload
            payload["staged_fallback"] = _staged_fallback_failure_summary(fallback)
        box_completion = _call_model_chunk_drs_box_completion(
            prompt_chunk,
            client,
            rel_path=rel_path,
            n_predict=n_predict,
            context_budget=context_budget,
            cache_path=cache_path,
            payload=parsed,
            validation=validation,
        )
        if box_completion.get("accepted"):
            payload = {
                **box_completion,
                "elapsed": monolithic_elapsed + staged_elapsed + float(box_completion.get("elapsed") or 0.0),
                "fallback_from_reason": reason,
                "monolithic_prompt_hash": prompt_hash,
            }
            _write_cache(cache_path, payload)
            return payload
        payload["box_completion"] = {
            "accepted": False,
            "reason": box_completion.get("reason"),
            "stage": box_completion.get("stage"),
        }
        _write_cache(cache_path, payload)
        return payload
    if validation.get("grounding_failure_count"):
        payload = {
            "accepted": False,
            "reason": "grounding_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            **constraint,
            "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
            "validation": validation,
            "context_budget": context_budget,
        }
        _write_cache(cache_path, payload)
        return payload
    staged_retry_reason = _chunk_drs_staged_retry_reason(validation, prompt_chunk, context_budget)
    staged_retry_summary: dict[str, Any] | None = None
    if _staged_chunk_drs_enabled() and staged_retry_reason:
        fallback = _call_model_chunk_drs_staged(
            prompt_chunk,
            client,
            rel_path=rel_path,
            n_predict=n_predict,
            context_budget=context_budget,
            cache_path=cache_path,
        )
        fallback_validation = fallback.get("validation") if isinstance(fallback.get("validation"), dict) else {}
        if fallback.get("accepted") and _validation_count(fallback_validation, "condition_count") > _validation_count(
            validation, "condition_count"
        ):
            payload = {**fallback, "fallback_from_reason": staged_retry_reason, "monolithic_prompt_hash": prompt_hash}
            _write_cache(cache_path, payload)
            return payload
        staged_retry_summary = _staged_fallback_failure_summary(fallback)
        staged_retry_summary.update(
            {
                "accepted": bool(fallback.get("accepted")),
                "fallback_from_reason": staged_retry_reason,
                "monolithic_condition_count": _validation_count(validation, "condition_count"),
                "fallback_condition_count": _validation_count(fallback_validation, "condition_count"),
            }
        )
    payload = {
        "accepted": True,
        "drs": parsed["drs"],
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": prompt_hash,
        **constraint,
        "context_budget": context_budget,
        "validation": validation,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    if staged_retry_summary:
        payload["staged_retry"] = staged_retry_summary
    _write_cache(cache_path, payload)
    return payload


def build_answer_verification_prompt(
    question: str,
    query_frame: dict[str, Any],
    candidate_answer: str,
    evidence_items: list[dict[str, str]],
    discourse_frames: list[dict[str, Any]],
) -> str:
    return (
        "JSON only. Verify whether the candidate answer is entailed by the bounded raw-text evidence and "
        "generic discourse frames. Reject candidates that do not satisfy the query frame's answer type, predicate, "
        "argument roles, referents, identity links, context accessibility, polarity, modality, temporal constraints, "
        "and provenance. Treat answer_type as a broad schema compatibility label for the bound variable, not as a "
        "word that must appear in the answer surface. "
        "For scoped DRS queries, the candidate may be the embedded proposition or scoped value rather than the "
        "scope holder/source; verify that binding against the evidence and discourse frames instead of requiring "
        "the candidate text itself to repeat the target anchor. "
        "Return exactly {\"verification\":{\"entailed\":false,\"answer_type\":\"unknown\",\"answer\":\"unknown\","
        "\"evidence_span\":\"\",\"reason\":\"\"}} with the appropriate values. "
        "Return the grounded answer binding or aggregate entailed by the evidence, using an exact evidence_span "
        "copied from the provided evidence. If the candidate contains multiple values, verify every value against "
        "the same query frame and omit any unentailed value. If evidence is insufficient, return entailed=false "
        "and answer='unknown'. "
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
    constraint = _constraint_settings(ANSWER_VERIFICATION_GRAMMAR, VERIFICATION_JSON_SCHEMA, ANSWER_SCHEMA_VERSION)
    grammar_hash = str(constraint["grammar_hash"])
    prompt_hash = _cache_hash(
        "answer_verification",
        prompt,
        client,
        {"n_predict": n_predict, "schema": ANSWER_SCHEMA_VERSION, **constraint},
    )
    cache_path = _cache_path("KMD_VERIFIER_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=ANSWER_VERIFICATION_GRAMMAR,
            json_schema=VERIFICATION_JSON_SCHEMA,
        )
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
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
            "grammar_hash": grammar_hash,
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
        "grammar_hash": grammar_hash,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    _write_cache(cache_path, payload)
    return payload


def build_answer_canonicalization_prompt(
    question: str,
    candidate_answer: str,
    answer_type: str,
    evidence_items: list[dict[str, str]],
) -> str:
    return (
        "JSON only. Canonicalize a model-selected final answer without changing its truth conditions or referent. "
        "Return the shortest grounded public answer that preserves the same DRS binding, answer type, polarity, "
        "modality, temporal scope, and provenance. The canonical answer may remove only redundant wording that is "
        "not part of the bound value or required aggregate. It must not introduce new referents, choose a sibling "
        "condition, change a scoped proposition into an asserted one, or use outside knowledge. evidence_span must "
        "be copied exactly from one provided evidence item whenever the answer is changed. Return exactly "
        "{\"canonical_answer\":{\"answer\":\"\",\"evidence_span\":\"\",\"reason\":\"\"}}."
        + json.dumps(
            {
                "question": question,
                "candidate_answer": candidate_answer,
                "answer_type": answer_type,
                "evidence": evidence_items,
            },
            ensure_ascii=False,
        )
    )


def call_model_answer_canonicalization(
    question: str,
    candidate_answer: str,
    answer_type: str,
    evidence_items: list[dict[str, str]],
    client: LocalModelClient,
    *,
    n_predict: int | None = None,
) -> dict[str, Any]:
    if n_predict is None:
        n_predict = int(os.environ.get("KMD_ANSWER_CANONICALIZATION_N_PREDICT", "96"))
    prompt = build_answer_canonicalization_prompt(question, candidate_answer, answer_type, evidence_items)
    constraint = _constraint_settings(ANSWER_CANONICALIZATION_GRAMMAR, CANONICAL_ANSWER_JSON_SCHEMA, ANSWER_SCHEMA_VERSION)
    grammar_hash = str(constraint["grammar_hash"])
    prompt_hash = _cache_hash(
        "answer_canonicalization",
        prompt,
        client,
        {"n_predict": n_predict, "schema": ANSWER_SCHEMA_VERSION, **constraint},
    )
    cache_path = _cache_path("KMD_ANSWER_CANONICALIZATION_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and cached.get("reason") not in {"ungrounded_answer", "schema_validation_failed", "invalid_json"}:
        return cached
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=ANSWER_CANONICALIZATION_GRAMMAR,
            json_schema=CANONICAL_ANSWER_JSON_SCHEMA,
        )
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "elapsed": round(time.time() - start, 3),
        }
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    result = parsed.get("canonical_answer") if isinstance(parsed, dict) else None
    if result is None and isinstance(parsed, dict) and "answer" in parsed:
        result = parsed
    if not isinstance(result, dict):
        return {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "elapsed": round(time.time() - start, 3),
        }
    answer = str(result.get("answer") or "").strip()
    reason_text = str(result.get("reason") or "").strip()
    span = str(result.get("evidence_span") or "").strip()
    if not answer:
        return {
            "accepted": False,
            "reason": "schema_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "elapsed": round(time.time() - start, 3),
        }
    span_grounded = False
    if span:
        span_grounded = any(span in str(item.get("text") or "") for item in evidence_items)
    if not span:
        for item in evidence_items:
            text = str(item.get("text") or "")
            if answer in text:
                span = answer
                span_grounded = True
                break
    answer_grounded = any(answer in str(item.get("text") or "") for item in evidence_items)
    if not span_grounded and not answer_grounded:
        return {
            "accepted": False,
            "reason": "ungrounded_answer",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "elapsed": round(time.time() - start, 3),
        }
    payload = {
        "accepted": True,
        "answer": answer,
        "evidence_span": span,
        "reason": reason_text,
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": prompt_hash,
        "grammar_hash": grammar_hash,
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
        "the evidence entails the identity/coreference in the same relevant DRS context; otherwise keep the "
        "original candidate answer. Do not use outside knowledge or name-shape heuristics."
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
    constraint = _constraint_settings(IDENTITY_CANONICALIZATION_GRAMMAR, IDENTITY_CANONICALIZATION_JSON_SCHEMA, ANSWER_SCHEMA_VERSION)
    grammar_hash = str(constraint["grammar_hash"])
    prompt_hash = _cache_hash(
        "identity_canonicalization",
        prompt,
        client,
        {"n_predict": n_predict, "schema": ANSWER_SCHEMA_VERSION, **constraint},
    )
    cache_path = _cache_path("KMD_IDENTITY_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None:
        if not (cached.get("accepted") is False and cached.get("reason") in {"invalid_json", "schema_validation_failed"}):
            return cached
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=IDENTITY_CANONICALIZATION_GRAMMAR,
            json_schema=IDENTITY_CANONICALIZATION_JSON_SCHEMA,
        )
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
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
    if result is None and isinstance(parsed, dict) and any(key in parsed for key in ["identity_hypothesis_accepted", "fuller_candidate", "fuller_answer"]):
        answer = str(parsed.get("answer") or parsed.get("fuller_candidate") or parsed.get("fuller_answer") or candidate_answer)
        result = {
            "same_referent": bool(parsed.get("identity_hypothesis_accepted")),
            "answer": answer,
            "evidence_span": str(parsed.get("evidence_span") or ""),
            "reason": str(parsed.get("reason") or parsed.get("rationale") or ""),
        }
    if not isinstance(result, dict):
        return {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
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
            "grammar_hash": grammar_hash,
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
            "grammar_hash": grammar_hash,
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
        "grammar_hash": grammar_hash,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    _write_cache(cache_path, payload)
    return payload
