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

PROMPT_VERSION = "kmd-drt-2026-05-27-v28"
CHUNK_FRAME_SCHEMA_VERSION = "chunk-frames-v5"
CHUNK_DRS_SCHEMA_VERSION = "chunk-drs-v1"
QUERY_DRS_SCHEMA_VERSION = "query-drs-v1"
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


def chunk_drs_json_schema(max_evidence_chars: int | None = None) -> dict[str, Any]:
    schema = json.loads(json.dumps(DRS_JSON_SCHEMA))
    if not max_evidence_chars:
        return schema
    max_length = max(1, int(max_evidence_chars))

    def visit(node: Any, parent_key: str = "") -> None:
        if isinstance(node, dict):
            if parent_key == "evidence_text" and node.get("type") == "string":
                node["maxLength"] = max_length
            if parent_key == "evidence_spans" and isinstance(node.get("items"), dict):
                node["items"]["maxLength"] = max_length
            for key, value in node.items():
                visit(value, key)
        elif isinstance(node, list):
            for item in node:
                visit(item, parent_key)

    visit(schema)
    return schema


QUERY_DRS_JSON_SCHEMA = _schema_obj(
    ["query_drs"],
    {
        "query_drs": _schema_obj(
            [
                "schema_version",
                "question",
                "answer_variables",
                "target_referents",
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
                "answer_variables": STRING_ARRAY_SCHEMA,
                "target_referents": _schema_array(DRS_REFERENT_JSON_SCHEMA),
                "requested_conditions": _schema_array(DRS_CONDITION_JSON_SCHEMA),
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
        "Put visible named anchors that the requested condition is about into target_referents, and make condition "
        "arguments point to those referent ids when they are the same discourse referent. Choose the broad "
        "answer_type from the schema values based on the answer variable requested by the question; use unknown "
        "only when the query DRS leaves the answer variable type underspecified. "
        "Arguments use target_kind and target_id exactly as in chunk DRS extraction. "
        "Return this shape with schema_version query-drs-v1: {\"query_drs\":{\"schema_version\":\"query-drs-v1\","
        "\"question\":\"\",\"answer_variables\":[],\"target_referents\":[],\"requested_conditions\":[],"
        "\"constraints\":[],\"box_requirements\":[],\"temporal_scope\":\"\",\"aggregation\":\"\","
        "\"answer_type\":\"unknown\",\"requires_evidence\":true}}."
        + json.dumps({"question": question, "surface_observations": surface}, ensure_ascii=False)
    )


def _validate_query_drs_payload(payload: Any, question: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("query_drs"), dict):
        return {"schema_valid": False, "errors": ["missing_query_drs_object"]}
    query_drs = payload["query_drs"]
    errors: list[str] = []

    def collection(name: str) -> list[dict[str, Any]]:
        value = query_drs.get(name)
        if not isinstance(value, list):
            errors.append(f"not_list:{name}")
            return []
        return [item for item in value if isinstance(item, dict)]

    if query_drs.get("question") != question:
        errors.append("question_mismatch")
    if str(query_drs.get("answer_type") or "") not in ANSWER_TYPES:
        errors.append(f"bad_answer_type:{query_drs.get('answer_type')}")
    if not isinstance(query_drs.get("answer_variables"), list):
        errors.append("not_list:answer_variables")
    if not isinstance(query_drs.get("constraints"), list):
        errors.append("not_list:constraints")
    targets = collection("target_referents")
    boxes = collection("box_requirements")
    conditions = collection("requested_conditions")
    target_ids = {str(item.get("id") or "") for item in targets if str(item.get("id") or "")}
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
    for condition in conditions:
        condition_id = str(condition.get("id") or "")
        box_id = str(condition.get("box_id") or "")
        if not condition_id or not str(condition.get("predicate") or "").strip():
            errors.append(f"bad_condition:{condition_id}")
        if box_id and box_id not in box_ids:
            errors.append(f"missing_condition_box:{condition_id}->{box_id}")
        if str(condition.get("polarity") or "") not in DRS_POLARITIES:
            errors.append(f"bad_polarity:{condition_id}:{condition.get('polarity')}")
        if str(condition.get("modality") or "") not in DRS_CONTEXT_KINDS:
            errors.append(f"bad_modality:{condition_id}:{condition.get('modality')}")
        arguments = condition.get("arguments")
        if not isinstance(arguments, list):
            errors.append(f"bad_arguments:{condition_id}")
            continue
        for arg in arguments:
            if not isinstance(arg, dict):
                continue
            target_kind = str(arg.get("target_kind") or "")
            target_id = str(arg.get("target_id") or "")
            if target_kind == "referent" and target_id and target_id not in target_ids:
                errors.append(f"missing_argument_referent:{condition_id}->{target_id}")
            elif target_kind == "box" and target_id and target_id not in box_ids:
                errors.append(f"missing_argument_box:{condition_id}->{target_id}")
            elif target_kind == "condition" and target_id and target_id not in condition_ids:
                errors.append(f"missing_argument_condition:{condition_id}->{target_id}")
            elif target_kind not in {"referent", "box", "condition", "literal", "unknown"}:
                errors.append(f"bad_argument_target_kind:{condition_id}:{target_kind}")
    return {
        "schema_valid": not errors,
        "errors": errors[:50],
        "target_count": len(targets),
        "condition_count": len(conditions),
        "box_requirement_count": len(boxes),
    }


def call_model_query_drs(question: str, client: LocalModelClient, *, n_predict: int | None = None) -> dict[str, Any]:
    if n_predict is None:
        n_predict = default_query_drs_n_predict(client)
    prompt = build_query_drs_prompt(question)
    constraint = _constraint_settings(QUERY_DRS_GRAMMAR, QUERY_DRS_JSON_SCHEMA, QUERY_DRS_SCHEMA_VERSION)
    prompt_hash = _cache_hash(
        "query_drs",
        prompt,
        client,
        {"n_predict": n_predict, "schema": QUERY_DRS_SCHEMA_VERSION, **constraint},
    )
    cache_path = _cache_path("KMD_QUERY_DRS_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=QUERY_DRS_GRAMMAR,
            json_schema=QUERY_DRS_JSON_SCHEMA,
        )
    except Exception as exc:
        payload = {
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
    validation = _validate_query_drs_payload(parsed, question)
    if not validation.get("schema_valid"):
        payload = {
            "accepted": False,
            "reason": "schema_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            **constraint,
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
    if not isinstance(target_referents, list) or not isinstance(requested_conditions, list):
        return None
    target_anchors = [
        str(item.get("label") or "").strip()
        for item in target_referents
        if isinstance(item, dict) and str(item.get("label") or "").strip()
    ]
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
            value = str(argument.get("value") or "").strip()
            role = str(argument.get("role") or "").strip()
            if value:
                argument_terms.append(value)
            if role:
                argument_terms.append(role)
    scope_terms = [
        str(item.get("kind") or "").strip()
        for item in box_requirements or []
        if isinstance(item, dict) and str(item.get("kind") or "").strip() and str(item.get("kind") or "") != "asserted"
    ]
    frame = frame_from_mapping(
        question,
        {
            "target_anchors": list(dict.fromkeys(target_anchors)),
            "answer_variables": query_drs.get("answer_variables") if isinstance(query_drs.get("answer_variables"), list) else [],
            "requested_relation": " ".join(dict.fromkeys(predicates)),
            "relation_terms": list(dict.fromkeys([*predicates, *argument_terms])),
            "constraints": query_drs.get("constraints") if isinstance(query_drs.get("constraints"), list) else [],
            "scope_requirements": list(dict.fromkeys(scope_terms)),
            "modality_requirements": list(dict.fromkeys(modality_terms)),
            "answer_type": query_drs.get("answer_type") if isinstance(query_drs.get("answer_type"), str) else "unknown",
            "temporal_scope": query_drs.get("temporal_scope") if isinstance(query_drs.get("temporal_scope"), str) else "",
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
        "coreference, pronoun, speaker, or same-referent links. Include modality, polarity, context_holder, "
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
    return (
        "JSON only. Convert the raw text chunk into one source-grounded DRS object. "
        "Every semantic decision must be represented as referents, boxes, conditions, temporal_records, "
        "and identity_hypotheses. Do not answer questions, use outside knowledge, infer hidden answers, "
        "or use handler names. The root asserted box should have id b0 and parent_id ''. Use subordinate boxes "
        "for negation, reports, quotes, beliefs, conditionals, uncertainty, dreams, fiction, and modality. "
        "Arguments use target_kind and target_id; use target_kind=box when an argument is a subordinate DRS box, "
        "target_kind=condition when an argument is another condition, and target_kind=referent for discourse "
        "referents. Identity hypotheses must be model-provided DRT data, not same-name merging. "
        "Every evidence_text and evidence_spans item must be one contiguous substring copied exactly from the chunk. "
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
                        "evidence_spans": [],
                        "semantic_notes": [],
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
    evidence_spans = drs.get("evidence_spans")
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
            elif target_kind == "condition" and target_id and target_id not in condition_ids:
                errors.append(f"missing_argument_condition:{condition_id}->{target_id}")
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
        "schema_valid": not errors,
        "errors": errors[:50],
        "grounding_failures": grounding_failures[:50],
        "grounding_failure_count": len(grounding_failures),
        "referent_count": len(referents),
        "box_count": len(boxes),
        "condition_count": len(conditions),
        "identity_hypothesis_count": len(identities),
    }


def _repair_chunk_drs_payload(payload: Any) -> Any:
    if not isinstance(payload, dict) or not isinstance(payload.get("drs"), dict):
        return payload
    drs = {**payload["drs"]}
    referents = drs.get("referents")
    conditions = drs.get("conditions")
    if not isinstance(referents, list) or not isinstance(conditions, list):
        return payload
    repaired_referents = [item for item in referents if isinstance(item, dict)]
    referent_ids = {str(item.get("id") or "") for item in repaired_referents}
    for condition in conditions:
        if not isinstance(condition, dict) or not isinstance(condition.get("arguments"), list):
            continue
        for argument in condition["arguments"]:
            if not isinstance(argument, dict) or str(argument.get("target_kind") or "") != "referent":
                continue
            target_id = str(argument.get("target_id") or "").strip()
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
    if len(repaired_referents) == len(referents):
        return payload
    drs["referents"] = repaired_referents
    return {**payload, "drs": drs}


def chunk_drs_cache_context(client: LocalModelClient | None, *, n_predict: int | None = None) -> dict[str, Any]:
    constraint = _constraint_settings(CHUNK_DRS_GRAMMAR, DRS_JSON_SCHEMA, CHUNK_DRS_SCHEMA_VERSION)
    if n_predict is None:
        n_predict = default_chunk_drs_n_predict(client)
    return {
        "prompt_version": PROMPT_VERSION,
        "schema_version": CHUNK_DRS_SCHEMA_VERSION,
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
    prompt = build_chunk_drs_prompt(prompt_chunk, rel_path=rel_path, context_budget=context_budget)
    drs_json_schema = chunk_drs_json_schema(len(prompt_chunk) if prompt_chunk else None)
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
        },
    )
    cache_path = _cache_path("KMD_CHUNK_DRS_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None:
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
        _write_cache(cache_path, payload)
        return payload
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    parsed = _repair_chunk_drs_payload(parsed)
    validation = _validate_chunk_drs_payload(parsed, prompt_chunk)
    if not validation.get("schema_valid"):
        payload = {
            "accepted": False,
            "reason": "schema_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            **constraint,
            "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
            "validation": validation,
            "context_budget": context_budget,
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
