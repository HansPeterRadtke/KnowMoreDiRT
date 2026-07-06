"""Local-model helpers for DRS/query planning.

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
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context_budget import context_ratio, context_relative_budget
from .drs_validation import box_parent_cycle_errors, box_root_errors, condition_argument_cycle_errors
from .model import LocalModelClient, LocalModelJSONError
from .extractors import identifiers, urls
from .query import QueryFrame, frame_from_mapping, term_variants, visible_anchors
from .relations import extract_relations
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
QUERY_SLOT_GENERIC_TERMS = {
    "actor",
    "answer",
    "code",
    "content",
    "count",
    "date",
    "entity",
    "file",
    "identifier",
    "id",
    "item",
    "link",
    "metadata",
    "number",
    "organization",
    "path",
    "person",
    "state",
    "status",
    "time",
    "unknown",
    "url",
    "value",
}
QUERY_QUESTION_COVERAGE_SKIP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

PROMPT_VERSION = "kmd-drt-2026-05-28-v35"
CHUNK_FRAME_SCHEMA_VERSION = "chunk-frames-v5"
CHUNK_FRAME_CONTEXT_BUDGET_POLICY = "runtime-context-minus-output-prompt-overhead-v1"
CHUNK_DRS_SCHEMA_VERSION = "chunk-drs-v2"
CHUNK_DRS_STAGED_FALLBACK_POLICY = "retry-invalid-json-schema-grounding-staged-temporal-scope-v3"
CHUNK_DRS_GROUNDING_REPAIR_POLICY = "model-label-value-escaped-evidence-span-v3"
CHUNK_DRS_IDENTITY_PROVENANCE_POLICY = "identity-evidence-bilateral-surface-box-scope-v2"
CHUNK_DRS_TEMPORAL_PROVENANCE_POLICY = "condition-stage-declared-temporal-records-v2"
CHUNK_DRS_SPARSE_RETRY_POLICY = "retry-validated-sparse-drs-staged-v2"
CHUNK_DRS_STRUCTURE_VALIDATION_POLICY = "single-root-acyclic-box-parent-condition-arguments-v4"
CHUNK_DRS_BOX_COMPLETION_POLICY = "model-complete-missing-box-declarations-v1"
CHUNK_DRS_SOURCE_SPAN_POLICY = "chunk-drs-delimiter-source-span-enum-v2"
CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY = "stage1-source-span-evidence-enum-v1"
CHUNK_DRS_SKELETON_ID_POLICY = "stage1-stable-id-enums-v1"
CHUNK_DRS_MONOLITHIC_ID_POLICY = "monolithic-stable-id-enums-v1"
CHUNK_DRS_COMPACT_UNDERCOVERAGE_POLICY = "retry-delimiter-rich-low-condition-density-v1"
CHUNK_DRS_STAGED_RETRY_DIAGNOSTICS_POLICY = "record-non-improving-staged-retry-v1"
CHUNK_DRS_STAGE_FAILURE_CACHE_POLICY = "cache-invalid-json-stage-failures-v2"
CHUNK_DRS_DYNAMIC_SKELETON_BUDGET_POLICY = "nested-field-like-source-spans-allow-768-v2"
CHUNK_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY = "source-aware-tiny-prose-544-short-768-1024-v2"
CHUNK_DRS_DYNAMIC_CONDITION_BUDGET_POLICY = "compact-nontemporal-condition-stage-floor-528-v2"
CHUNK_DRS_STAGED_FIRST_POLICY = "field-like-source-spans-before-monolithic-v1"
CHUNK_DRS_COMPACT_FACT_POLICY_LEGACY = "compact-model-facts-to-root-drs-v1"
CHUNK_DRS_COMPACT_FACT_POLICY_PREVIOUS = "compact-model-facts-to-root-drs-v2"
CHUNK_DRS_COMPACT_FACT_POLICY = "compact-model-facts-with-embedded-scope-predicate-v5"
CHUNK_DRS_COMPACT_TEMPORAL_SOURCE_POLICY = "compact-source-span-explicit-timestamp-v1"
CHUNK_DRS_COMPACT_RETRY_POLICY = "retry-compact-invalid-json-larger-budget-v2"
QUERY_DRS_SCHEMA_VERSION = "query-drs-v3"
QUERY_DRS_VALIDATION_POLICY = "strict-query-drs-version-question-evidence-box-dag-repair-operators-v10"
QUERY_DRS_ARRAY_CAP_POLICY = "reserved_output_tokens_div_96_4_8-v1"
QUERY_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY = "surface-token-budget-short384-mid512-long-context-v1"
QUERY_DRS_COMPACT_PLAN_POLICY = "compact-model-plan-to-query-drs-v2"
CONSTRAINT_TRANSPORT_POLICY = "bounded-json-schema-min4096-structured-record-route-v1"
QUERY_DRS_COMPACT_UNDERCOVERAGE_POLICY = "broad-slot-uncovered-token-full-fallback-v1"
QUERY_DRS_REQUEST_FAILURE_RETRY_POLICY = "smaller-full-query-drs-output-budget-v1"
CHUNK_DRS_STRUCTURED_RECORD_ROUTE_POLICY = "structured-records-use-deterministic-extraction-no-drs-skip-v1"
QUERY_OPERATOR_SCHEMA_POLICY = "query-temporal-aggregation-operator-enums-v1"
QUERY_FRAME_SCHEMA_VERSION = "query-frame-v6"
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


def _local_model_transport_fingerprint() -> dict[str, Any]:
    return {
        "api": os.environ.get("KMD_LOCAL_MODEL_API", "chat").strip().lower() or "chat",
    }


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
        "per_token_timeout_seconds": getattr(client, "timeout_seconds", ""),
        "seed": os.environ.get("KMD_LOCAL_MODEL_SEED", "1778779265"),
        "transport_settings": _local_model_transport_fingerprint(),
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


def default_chunk_drs_n_predict(client: LocalModelClient | None = None, chunk_text: str = "") -> int:
    configured = os.environ.get("KMD_CHUNK_DRS_N_PREDICT")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    context_size = _client_context_size(client)
    source_text = str(chunk_text or "")
    if context_size > 0:
        context_budget = context_relative_budget(
            context_size,
            output_ratio_names=("KMD_CHUNK_DRS_OUTPUT_RATIO",),
        )
        if not source_text:
            return context_budget.output_tokens
        source_tokens = max(1, _estimate_tokens(source_text))
        floor_ratio = context_ratio(("KMD_CHUNK_DRS_OUTPUT_FLOOR_RATIO",), 1.0 / 512.0)
        source_ratio = float(os.environ.get("KMD_CHUNK_DRS_OUTPUT_SOURCE_TOKEN_RATIO", "64"))
        if source_ratio <= 0.0:
            source_ratio = 64.0
        source_scaled = int(round(source_tokens * source_ratio))
        floor_tokens = int(round(context_size * floor_ratio))
        return max(1, min(context_budget.output_tokens, max(floor_tokens, source_scaled)))
    fallback = os.environ.get("KMD_CHUNK_DRS_FALLBACK_N_PREDICT", "").strip()
    if fallback:
        try:
            return max(1, int(fallback))
        except ValueError:
            pass
    return 384


def default_query_drs_n_predict(client: LocalModelClient | None = None, question: str = "") -> int:
    configured = os.environ.get("KMD_QUERY_DRS_N_PREDICT")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    context_size = _client_context_size(client)
    if context_size <= 0:
        return 256
    context_budget = max(256, min(768, context_size // 48))
    token_count = len(content_tokens(question))
    anchor_count = len(visible_anchors(question))
    if question and token_count <= 14 and anchor_count <= 3:
        return min(context_budget, 384)
    if question and token_count <= 32 and anchor_count <= 6:
        return min(context_budget, 512)
    return context_budget


def default_compact_query_drs_n_predict(question: str = "") -> int:
    configured = os.environ.get("KMD_QUERY_DRS_COMPACT_N_PREDICT")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    token_count = len(content_tokens(question))
    return 64 if token_count <= 20 else 96


def default_compact_chunk_drs_n_predict(chunk_text: str = "") -> int:
    configured = os.environ.get("KMD_CHUNK_DRS_COMPACT_N_PREDICT")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    token_count = len(content_tokens(chunk_text))
    if token_count <= 20:
        return 384
    if token_count <= 60:
        return 768
    return 1024


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
        "model_per_token_timeout_seconds": getattr(client, "timeout_seconds", os.environ.get("KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS", "")),
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
        "constraint_transport_policy": CONSTRAINT_TRANSPORT_POLICY,
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


def _cached_structured_failure_retryable(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    retryable_reasons = {
        "request_failed",
        "invalid_json",
        "schema_validation_failed",
        "grounding_validation_failed",
    }
    return str(payload.get("reason") or "") in retryable_reasons or str(
        payload.get("repair_failure_reason") or ""
    ) in retryable_reasons


def _cached_request_failed(payload: dict[str, Any] | None) -> bool:
    return _cached_structured_failure_retryable(payload)


def _cached_evidence_answer_retryable(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    return str(payload.get("reason") or "") == "request_failed" or str(
        payload.get("repair_failure_reason") or ""
    ) == "request_failed"


def _query_drs_cached_retryable_failure(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    return payload.get("accepted") is False and str(payload.get("reason") or "") in {
        "request_failed",
        "invalid_json",
        "schema_validation_failed",
        "grounding_validation_failed",
    }


def _write_cache(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _with_model_input_audits(payload, payload)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        pass


def _model_input_audit_from(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        audit = value.get("_model_input_audit") or value.get("model_input_audit")
        if isinstance(audit, dict):
            return copy.deepcopy(audit)
    audit = getattr(value, "model_input_audit", None)
    if isinstance(audit, dict):
        return copy.deepcopy(audit)
    return None


def _model_input_audits_from(*values: Any) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            nested = value.get("model_input_audits")
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, dict):
                        marker = str(item.get("request_body_sha256") or json.dumps(item, sort_keys=True, default=str))
                        if marker not in seen:
                            audits.append(copy.deepcopy(item))
                            seen.add(marker)
            audit = _model_input_audit_from(value)
        else:
            audit = _model_input_audit_from(value)
        if isinstance(audit, dict):
            marker = str(audit.get("request_body_sha256") or json.dumps(audit, sort_keys=True, default=str))
            if marker not in seen:
                audits.append(audit)
                seen.add(marker)
    return audits


def _with_model_input_audits(payload: dict[str, Any], *sources: Any) -> dict[str, Any]:
    audits = _model_input_audits_from(*sources)
    if not audits:
        return payload
    enriched = {**payload}
    enriched["model_input_audit_count"] = len(audits)
    enriched["model_input_audits"] = audits
    if len(audits) == 1:
        enriched["model_input_audit"] = audits[0]
    return enriched


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
        and str(value.get("temporal_scope") or "") in {"", "earliest", "latest"}
        and isinstance(value.get("negated"), bool)
        and str(value.get("aggregation") or "") in {"", "count", "list", "set"}
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

SOURCE_RESOLVED_ANSWER_GRAMMAR = r'''
root ::= "{" ws "\"source_resolved_answer\"" ws ":" ws "{" ws "\"answer\"" ws ":" ws string ws "," ws "\"evidence_span\"" ws ":" ws string ws "," ws "\"reason\"" ws ":" ws string ws "}" ws "}"
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


def _schema_string_limited(max_length: int) -> dict[str, Any]:
    return {"type": "string", "maxLength": max(1, int(max_length))}


def _schema_array_bounded(item: dict[str, Any], max_items: int) -> dict[str, Any]:
    return {"type": "array", "items": item, "maxItems": max(0, int(max_items))}


def _schema_enum(values: set[str]) -> dict[str, Any]:
    return {"type": "string", "enum": sorted(values)}


STRING_SCHEMA = {"type": "string"}
BOOL_SCHEMA = {"type": "boolean"}
NUMBER_SCHEMA = {"type": "number"}
ANSWER_TYPE_SCHEMA = _schema_enum(ANSWER_TYPES)
TEMPORAL_SCOPE_SCHEMA = _schema_enum({"", "earliest", "latest"})
AGGREGATION_SCHEMA = _schema_enum({"", "count", "list", "set"})
STRING_ARRAY_SCHEMA = _schema_array(STRING_SCHEMA)

COMPACT_QUERY_DRS_JSON_SCHEMA = _schema_obj(
    ["a", "answer", "targets", "predicates", "constraints", "temporal_scope", "aggregation"],
    {
        "a": ANSWER_TYPE_SCHEMA,
        "answer": _schema_string_limited(96),
        "targets": _schema_array_bounded(_schema_string_limited(128), 8),
        "predicates": _schema_array_bounded(_schema_string_limited(64), 8),
        "constraints": _schema_array_bounded(_schema_string_limited(160), 8),
        "temporal_scope": TEMPORAL_SCOPE_SCHEMA,
        "aggregation": AGGREGATION_SCHEMA,
    },
)

COMPACT_CHUNK_FACT_ARGUMENT_JSON_SCHEMA = _schema_obj(
    ["role", "value"],
    {"role": _schema_string_limited(48), "value": _schema_string_limited(160)},
)

COMPACT_CHUNK_FACT_JSON_SCHEMA = _schema_obj(
    ["p", "e", "arguments", "temporal_text", "scope"],
    {
        "p": _schema_string_limited(64),
        "e": _schema_string_limited(260),
        "arguments": _schema_array_bounded(COMPACT_CHUNK_FACT_ARGUMENT_JSON_SCHEMA, 6),
        "temporal_text": _schema_string_limited(96),
        "scope": _schema_enum(DRS_CONTEXT_KINDS),
    },
)

COMPACT_CHUNK_DRS_JSON_SCHEMA = _schema_obj(
    ["facts"],
    {"facts": _schema_array_bounded(COMPACT_CHUNK_FACT_JSON_SCHEMA, 8)},
)

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
                "temporal_scope": TEMPORAL_SCOPE_SCHEMA,
                "negated": BOOL_SCHEMA,
                "aggregation": AGGREGATION_SCHEMA,
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
        "box_id": STRING_SCHEMA,
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

    string_limits = {
        "id": 24,
        "schema_version": 24,
        "source_id": 256,
        "label": 160,
        "kind": 48,
        "predicate": 64,
        "box_id": 24,
        "parent_id": 24,
        "holder_referent_id": 24,
        "temporal_id": 24,
        "role": 48,
        "target_kind": 24,
        "target_id": 24,
        "value": 160,
        "value_type": 48,
        "left_referent_id": 24,
        "right_referent_id": 24,
        "status": 24,
    }

    def visit(node: Any, parent_key: str = "") -> None:
        if isinstance(node, dict):
            if node.get("type") == "string" and parent_key in string_limits:
                node["maxLength"] = string_limits[parent_key]
            if max_length is not None and parent_key == "evidence_text" and node.get("type") == "string":
                node["maxLength"] = min(max_length, 260)
            if max_length is not None and parent_key == "evidence_spans" and isinstance(node.get("items"), dict):
                node["items"]["maxLength"] = min(max_length, 260)
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
        identity_schema["properties"]["box_id"] = {"type": "string", "enum": ["", *box_ids]}
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
    evidence_ratio = context_ratio(("KMD_CHUNK_DRS_EVIDENCE_OUTPUT_RATIO",), 0.25)
    budgeted = max(1, int(round(int(n_predict) * evidence_ratio)))
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
    item_ratio = context_ratio(("KMD_CHUNK_DRS_ARRAY_ITEM_OUTPUT_RATIO",), 1.0 / 96.0)
    return max(1, int(round(int(n_predict) * item_ratio)))


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
    for key in (
        "error",
        "raw_snippet",
        "grounding_failures",
        "validation",
        "elapsed",
        "prompt_hash",
        "grammar_hash",
        "json_schema_hash",
        "constraint_mode",
    ):
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


def _chunk_drs_has_temporal_surface(source_text: str) -> bool:
    try:
        return any(relation.relation_type == "temporal" for relation in extract_relations(source_text))
    except Exception:
        return False


def _chunk_drs_staged_retry_reason(
    validation: dict[str, Any],
    source_text: str = "",
    context_budget: dict[str, Any] | None = None,
) -> str:
    if _chunk_drs_structurally_sparse(validation):
        return "structural_sparsity"
    condition_count = _validation_count(validation, "condition_count")
    box_count = _validation_count(validation, "box_count")
    if box_count >= 3 and condition_count < box_count - 1:
        return "scoped_box_undercoverage"
    field_like_span_count = _chunk_drs_structural_condition_floor(
        source_text,
        (context_budget or {}).get("max_evidence_chars"),
    )
    if field_like_span_count >= 3 and condition_count < 2:
        return "structural_undercoverage"
    return ""


def _chunk_drs_staged_first_reason(
    source_text: str,
    context_budget: dict[str, Any] | None = None,
) -> str:
    field_like_span_count = _chunk_drs_structural_condition_floor(
        source_text,
        (context_budget or {}).get("max_evidence_chars"),
    )
    if field_like_span_count >= 4:
        return "field_like_source_spans"
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


def default_staged_chunk_drs_skeleton_n_predict(
    n_predict: int,
    source_text: str = "",
    max_evidence_chars: int | None = None,
) -> int:
    configured = os.environ.get("KMD_CHUNK_DRS_STAGED_SKELETON_N_PREDICT")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    base = max(192, min(int(n_predict), 384))
    if (
        source_text
        and _chunk_drs_structural_condition_floor(source_text, max_evidence_chars) >= 4
        and any(delimiter in source_text for delimiter in "{}[]")
    ):
        return max(base, 768)
    return base


def default_staged_chunk_drs_condition_n_predict(
    n_predict: int,
    source_text: str = "",
    max_evidence_chars: int | None = None,
) -> int:
    configured = os.environ.get("KMD_CHUNK_DRS_STAGED_CONDITION_N_PREDICT")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    if (
        source_text
        and _estimate_tokens(source_text) <= 64
        and _chunk_drs_structural_condition_floor(source_text, max_evidence_chars) <= 8
        and not _chunk_drs_has_temporal_surface(source_text)
    ):
        return 528
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
    referent_schema["properties"]["label"] = _schema_string_limited(160)
    referent_schema["properties"]["kind"] = _schema_string_limited(48)
    box_schema["properties"]["kind"] = _schema_enum(DRS_CONTEXT_KINDS)
    temporal_schema["properties"]["value"] = _schema_string_limited(160)
    temporal_schema["properties"]["value_type"] = _schema_string_limited(48)
    if evidence_text_values:
        evidence_values = list(dict.fromkeys(str(value) for value in evidence_text_values))
        evidence_schema = {"type": "string", "enum": evidence_values}
        referent_schema["properties"]["evidence_text"] = copy.deepcopy(evidence_schema)
        box_schema["properties"]["evidence_text"] = copy.deepcopy(evidence_schema)
        temporal_schema["properties"]["evidence_text"] = copy.deepcopy(evidence_schema)
    else:
        referent_schema["properties"]["evidence_text"] = _schema_string_limited(260)
        box_schema["properties"]["evidence_text"] = _schema_string_limited(260)
        temporal_schema["properties"]["evidence_text"] = _schema_string_limited(260)
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
    condition_ids = [f"c{index}" for index in range(max(1, int(max_conditions)))] if max_conditions else []
    allowed_targets = sorted(set(["", *box_ids, *condition_ids, *referent_ids]))
    allowed_temporals = sorted(set(["", *(temporal_ids or [])]))
    condition_schema["properties"]["predicate"] = _schema_string_limited(64)
    argument_schema["properties"]["role"] = _schema_string_limited(48)
    argument_schema["properties"]["value"] = _schema_string_limited(160)
    argument_schema["properties"]["value_type"] = _schema_string_limited(48)
    argument_schema["properties"]["target_id"] = {"type": "string", "enum": allowed_targets}
    condition_schema["properties"]["box_id"] = {"type": "string", "enum": box_ids or [""]}
    condition_schema["properties"]["temporal_id"] = {"type": "string", "enum": allowed_temporals}
    if condition_ids:
        condition_schema["properties"]["id"] = {"type": "string", "enum": condition_ids}
    if evidence_text_values:
        evidence_values = list(dict.fromkeys(str(value) for value in evidence_text_values))
        condition_schema["properties"]["evidence_text"] = {"type": "string", "enum": evidence_values}
        argument_schema["properties"]["evidence_text"] = {"type": "string", "enum": evidence_values}
    else:
        condition_schema["properties"]["evidence_text"] = _schema_string_limited(260)
        argument_schema["properties"]["evidence_text"] = _schema_string_limited(260)
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
    box_schema["properties"]["kind"] = _schema_enum(DRS_CONTEXT_KINDS)
    box_schema["properties"]["parent_id"] = {"type": "string", "enum": sorted(set(["", *existing_box_ids]))}
    box_schema["properties"]["holder_referent_id"] = {"type": "string", "enum": sorted(set(["", *referent_ids]))}
    box_schema["properties"]["evidence_text"] = _schema_string_limited(260)
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
                "temporal_scope": TEMPORAL_SCOPE_SCHEMA,
                "aggregation": AGGREGATION_SCHEMA,
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

SOURCE_RESOLVED_ANSWER_JSON_SCHEMA = _schema_obj(
    ["source_resolved_answer"],
    {
        "source_resolved_answer": _schema_obj(
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
        "answer variable type underspecified. temporal_scope must be '', 'latest', or 'earliest'; put current, "
        "latest, final, first, earliest, or ordering requirements there as a normalized DRS operator rather "
        "than leaving them only in requested_relation. aggregation must be '', 'count', 'list', or 'set'. "
        "Put any quantity, list, temporal, modal, polarity, or qualifier "
        "requirements into aggregation, temporal_scope, modality_requirements, scope_requirements, negated, "
        "constraints, and answer_variables as DRS data rather than as prose. If the answer is requested inside a "
        "subordinate or non-asserted DRS, represent that accessibility requirement in modality_requirements or "
        "scope_requirements as well as any predicate text; do not leave the scope marker only in requested_relation. Relation terms should describe the "
        "predicate or answer slot requested by the question, not hidden labels. Use only text visible in the "
        "question and no outside knowledge. Surface observations are syntactic hints only."
        + json.dumps({"question": question, "surface_observations": surface}, ensure_ascii=False)
    )


def call_model_query_plan_test_only(question: str, client: LocalModelClient, *, n_predict: int | None = None) -> dict[str, Any]:
    if n_predict is None:
        n_predict = int(os.environ.get("KMD_QUERY_PLAN_N_PREDICT", "128"))
    prompt = build_query_plan_prompt(question)
    constraint = _constraint_settings(QUERY_FRAME_GRAMMAR, QUERY_FRAME_JSON_SCHEMA, QUERY_FRAME_SCHEMA_VERSION)
    grammar_hash = str(constraint["grammar_hash"])
    cache_settings = {
        "n_predict": n_predict,
        "schema": QUERY_FRAME_SCHEMA_VERSION,
        "operator_schema_policy": QUERY_OPERATOR_SCHEMA_POLICY,
        **constraint,
    }
    cache_context = {**cache_settings, "model_fingerprint": _client_fingerprint(client)}
    prompt_hash = _cache_hash(
        "query_frame",
        prompt,
        client,
        cache_settings,
    )
    cache_path = _cache_path("KMD_QUERY_PLAN_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and not _cached_structured_failure_retryable(cached):
        cached.setdefault("cache_context", cache_context)
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
            "operator_schema_policy": QUERY_OPERATOR_SCHEMA_POLICY,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
        return _with_model_input_audits(payload, exc)
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
            "operator_schema_policy": QUERY_OPERATOR_SCHEMA_POLICY,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
        payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
            "operator_schema_policy": QUERY_OPERATOR_SCHEMA_POLICY,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
        payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
        "operator_schema_policy": QUERY_OPERATOR_SCHEMA_POLICY,
        "cache_context": cache_context,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    payload = _with_model_input_audits(payload, parsed)
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
        "answer variable, a broad answer_type, and evidence_text copied exactly from the question. The label must "
        "preserve visible modifiers that distinguish the requested slot from neighboring slots; do not reduce it "
        "to only a broad type word when the question includes a narrower phrase. Put visible "
        "non-answer discourse anchors that the requested condition is about into target_referents, including named "
        "and common-noun anchors, and put visible temporal phrases into "
        "temporal_records with ids such as qt0. Make condition arguments point to those ids when they are the same "
        "discourse referent or temporal value, and use temporal_id for the condition's temporal record when applicable. "
        "Requested condition arguments must use target_kind='answer_variable' and target_id equal to the declared qv "
        "id for the answer slot. Choose the "
        "top-level answer_type from the schema values based on the answer variable requested by the question; use "
        "unknown only when the query DRS leaves the answer variable type underspecified. "
        "temporal_scope must be '', 'latest', or 'earliest'. aggregation must be '', 'count', 'list', or 'set'. "
        "Arguments use target_kind and target_id exactly as declared in the query DRS namespace. "
        "Return this shape with schema_version query-drs-v3: {\"query_drs\":{\"schema_version\":\"query-drs-v3\","
        "\"question\":\"\",\"answer_variables\":[{\"id\":\"qv0\",\"label\":\"\",\"answer_type\":\"unknown\","
        "\"evidence_text\":\"\"}],\"target_referents\":[],\"temporal_records\":[],\"requested_conditions\":[],"
        "\"constraints\":[],\"box_requirements\":[],\"temporal_scope\":\"\",\"aggregation\":\"\","
        "\"answer_type\":\"unknown\",\"requires_evidence\":true}}."
        + json.dumps({"question": question, "surface_observations": surface}, ensure_ascii=False)
    )


def build_compact_query_drs_prompt(question: str) -> str:
    return (
        "JSON only. Plan this question as compact DRS query data; do not answer it. "
        "Output exactly one object with mandatory keys a, answer, targets, predicates, constraints, temporal_scope, aggregation. "
        "a is one broad answer type: person, actor, organization, identifier, url, file_path, count, state, "
        "date_time, boolean, content_phrase, metadata_value, or unknown. answer is the visible question word or "
        "answer slot phrase; preserve visible modifiers that distinguish the slot, instead of only the broad type "
        "word, when a narrower phrase appears. targets are non-answer noun phrases the answer is about, excluding verbs. predicates "
        "are verbs or relation words requested by the question. constraints are other visible qualifiers. "
        "temporal_scope is '', 'latest', or 'earliest'. aggregation is '', 'count', 'list', or 'set'. "
        "Use only words visible in the question and no outside knowledge. "
        + json.dumps({"question": question}, ensure_ascii=False)
    )


def _grounded_question_text(question: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text in question:
        return text
    index = question.lower().find(text.lower())
    if index >= 0:
        return question[index : index + len(text)]
    return ""


def _compact_query_drs_to_payload(question: str, compact: dict[str, Any]) -> dict[str, Any]:
    answer_type = _normalize_answer_type(compact.get("a") or compact.get("answer_type"))
    answer_label = str(compact.get("answer") or compact.get("answer_label") or answer_type or "answer").strip()
    answer_evidence = _grounded_question_text(question, answer_label) or question
    raw_targets = compact.get("targets")
    raw_predicates = compact.get("predicates") or compact.get("p")
    raw_constraints = compact.get("constraints")
    targets = [
        grounded
        for value in (raw_targets if isinstance(raw_targets, list) else [])
        if (grounded := _grounded_question_text(question, value))
    ]
    predicates = [
        grounded
        for value in (raw_predicates if isinstance(raw_predicates, list) else [])
        if (grounded := _grounded_question_text(question, value))
    ]
    constraints = [
        grounded
        for value in (raw_constraints if isinstance(raw_constraints, list) else [])
        if (grounded := _grounded_question_text(question, value))
    ]
    temporal_scope = str(compact.get("temporal_scope") or compact.get("time") or "").strip()
    if temporal_scope not in {"", "latest", "earliest"}:
        temporal_scope = ""
    aggregation = str(compact.get("aggregation") or "").strip()
    if aggregation not in {"", "count", "list", "set"}:
        aggregation = ""
    answer_variables = [
        {
            "id": "qv0",
            "label": answer_label or "answer",
            "answer_type": answer_type,
            "evidence_text": answer_evidence,
        }
    ]
    target_referents = [
        {"id": f"qr{index}", "label": target, "kind": "unknown", "evidence_text": target}
        for index, target in enumerate(dict.fromkeys(targets))
    ]
    requested_conditions: list[dict[str, Any]] = []
    condition_predicates = list(dict.fromkeys(predicates))
    if condition_predicates:
        arguments = [
            {
                "role": "answer",
                "target_kind": "answer_variable",
                "target_id": "qv0",
                "value": "",
                "value_type": answer_type,
                "evidence_text": answer_evidence,
            }
        ]
        for referent in target_referents:
            arguments.append(
                {
                    "role": "argument",
                    "target_kind": "referent",
                    "target_id": referent["id"],
                    "value": "",
                    "value_type": "unknown",
                    "evidence_text": str(referent["evidence_text"]),
                }
            )
        requested_conditions.append(
            {
                "id": "qc0",
                "box_id": "",
                "predicate": " ".join(condition_predicates),
                "polarity": "positive",
                "modality": "asserted",
                "temporal_id": "",
                "evidence_text": question,
                "arguments": arguments,
            }
        )
    return {
        "query_drs": {
            "schema_version": QUERY_DRS_SCHEMA_VERSION,
            "question": question,
            "answer_variables": answer_variables,
            "target_referents": target_referents,
            "temporal_records": [],
            "requested_conditions": requested_conditions,
            "constraints": constraints,
            "box_requirements": [],
            "temporal_scope": temporal_scope,
            "aggregation": aggregation,
            "answer_type": answer_type,
            "requires_evidence": True,
        }
    }


def _compact_query_drs_answer_slot_undercovered(question: str, payload: dict[str, Any]) -> bool:
    query_drs = payload.get("query_drs") if isinstance(payload.get("query_drs"), dict) else {}
    answer_variables = query_drs.get("answer_variables") if isinstance(query_drs, dict) else []
    if not isinstance(answer_variables, list):
        return False
    answer_tokens: set[str] = set()
    generic_answer = False
    for variable in answer_variables:
        if not isinstance(variable, dict):
            continue
        label_tokens = set(content_tokens(str(variable.get("label") or "")))
        evidence_tokens = set(content_tokens(str(variable.get("evidence_text") or "")))
        tokens = label_tokens or evidence_tokens
        answer_tokens.update(tokens)
        answer_type = normalize(str(variable.get("answer_type") or ""))
        if len(tokens) <= 1 and ((tokens and next(iter(tokens)) in QUERY_SLOT_GENERIC_TERMS) or answer_type in QUERY_SLOT_GENERIC_TERMS):
            generic_answer = True
    covered = set(answer_tokens) | QUERY_QUESTION_COVERAGE_SKIP_TERMS
    for item in query_drs.get("target_referents") or []:
        if isinstance(item, dict):
            covered.update(content_tokens(str(item.get("label") or "")))
            covered.update(content_tokens(str(item.get("evidence_text") or "")))
    requested_conditions = [
        condition for condition in query_drs.get("requested_conditions") or []
        if isinstance(condition, dict)
    ]
    for condition in query_drs.get("requested_conditions") or []:
        if not isinstance(condition, dict):
            continue
        covered.update(content_tokens(str(condition.get("predicate") or "")))
        for argument in condition.get("arguments") or []:
            if not isinstance(argument, dict):
                continue
            covered.update(content_tokens(str(argument.get("role") or "")))
            if str(argument.get("target_kind") or "") != "answer_variable":
                covered.update(content_tokens(str(argument.get("value") or "")))
                covered.update(content_tokens(str(argument.get("evidence_text") or "")))
    constraints = [value for value in query_drs.get("constraints") or [] if str(value or "").strip()]
    for value in constraints:
        covered.update(content_tokens(str(value or "")))
    uncovered = [token for token in content_tokens(question) if token not in covered]
    if generic_answer:
        return bool(uncovered)
    if uncovered and not requested_conditions and not constraints:
        return True
    return False


def call_model_query_drs_compact(question: str, client: LocalModelClient, *, n_predict: int | None = None) -> dict[str, Any]:
    if n_predict is None:
        n_predict = default_compact_query_drs_n_predict(question)
    prompt = build_compact_query_drs_prompt(question)
    constraint = _constraint_settings(QUERY_DRS_GRAMMAR, COMPACT_QUERY_DRS_JSON_SCHEMA, QUERY_DRS_SCHEMA_VERSION)
    cache_settings = {
        "n_predict": n_predict,
        "schema": QUERY_DRS_SCHEMA_VERSION,
        "compact_plan_policy": QUERY_DRS_COMPACT_PLAN_POLICY,
        **constraint,
    }
    cache_context = {**cache_settings, "model_fingerprint": _client_fingerprint(client)}
    prompt_hash = _cache_hash("query_drs_compact", prompt, client, cache_settings)
    cache_path = _cache_path("KMD_QUERY_DRS_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and not _query_drs_cached_retryable_failure(cached):
        finalized = {**cached}
        finalized.setdefault("cache_context", cache_context)
        if finalized.get("accepted") and isinstance(finalized.get("query_drs"), dict):
            repaired = _repair_query_drs_payload({"query_drs": finalized["query_drs"]}, question)
            validation = _validate_query_drs_payload(repaired, question)
            if validation.get("schema_valid"):
                finalized["query_drs"] = repaired["query_drs"]
                finalized["validation"] = validation
                if cache_path is not None and finalized != cached:
                    finalized = _with_model_input_audits(finalized, locals().get("parsed"), locals().get("cached"), locals().get("source_payload"), locals().get("payload"))
                    _write_cache(cache_path, finalized)
        return finalized
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=QUERY_DRS_GRAMMAR,
            json_schema=COMPACT_QUERY_DRS_JSON_SCHEMA,
        )
    except LocalModelJSONError as exc:
        payload = {
            "accepted": False,
            "reason": "invalid_json",
            "error": str(exc),
            "raw_text": exc.raw_text,
            "raw_snippet": exc.snippet,
            "prompt_hash": prompt_hash,
            "compact_plan_policy": QUERY_DRS_COMPACT_PLAN_POLICY,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
        payload = _with_model_input_audits(payload, exc)
        _write_cache(cache_path, payload)
        return payload
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "compact_plan_policy": QUERY_DRS_COMPACT_PLAN_POLICY,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    payload_drs = _compact_query_drs_to_payload(question, parsed if isinstance(parsed, dict) else {})
    payload_drs = _repair_query_drs_payload(payload_drs, question)
    validation = _validate_query_drs_payload(payload_drs, question)
    if not validation.get("schema_valid"):
        payload = {
            "accepted": False,
            "reason": "schema_validation_failed",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "compact_plan_policy": QUERY_DRS_COMPACT_PLAN_POLICY,
            "cache_context": cache_context,
            "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
            "validation": validation,
        }
        payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
        _write_cache(cache_path, payload)
        return payload
    payload = {
        "accepted": True,
        "query_drs": payload_drs["query_drs"],
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": prompt_hash,
        "compact_plan_policy": QUERY_DRS_COMPACT_PLAN_POLICY,
        "cache_context": cache_context,
        "validation": validation,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
        "compact": True,
    }
    payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
    _write_cache(cache_path, payload)
    return payload


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
    if not query_drs.get("requested_conditions"):
        covered = set(QUERY_QUESTION_COVERAGE_SKIP_TERMS)
        answer_items = [item for item in query_drs.get("answer_variables", []) if isinstance(item, dict)]
        target_items = [item for item in query_drs.get("target_referents", []) if isinstance(item, dict)]
        for item in [*answer_items, *target_items]:
            covered.update(content_tokens(str(item.get("label") or "")))
            covered.update(content_tokens(str(item.get("evidence_text") or "")))
        for value in query_drs.get("constraints") or []:
            covered.update(content_tokens(str(value or "")))
        uncovered_predicate_tokens = [
            token for token in content_tokens(question)
            if token not in covered and token not in QUERY_SLOT_GENERIC_TERMS
        ]
        if uncovered_predicate_tokens:
            answer_id = str(answer_items[0].get("id") or "qv0") if answer_items else "qv0"
            answer_type = str(query_drs.get("answer_type") or "unknown")
            arguments = [
                {
                    "role": "answer",
                    "target_kind": "answer_variable",
                    "target_id": answer_id,
                    "value": "",
                    "value_type": answer_type,
                    "evidence_text": str(answer_items[0].get("evidence_text") or answer_items[0].get("label") or "")
                    if answer_items
                    else "",
                }
            ]
            for referent in target_items:
                arguments.append(
                    {
                        "role": "argument",
                        "target_kind": "referent",
                        "target_id": str(referent.get("id") or ""),
                        "value": "",
                        "value_type": str(referent.get("kind") or "unknown"),
                        "evidence_text": str(referent.get("evidence_text") or referent.get("label") or ""),
                    }
                )
            query_drs["requested_conditions"] = [
                {
                    "id": "qc0",
                    "box_id": "",
                    "predicate": " ".join(dict.fromkeys(uncovered_predicate_tokens)),
                    "polarity": "positive",
                    "modality": "asserted",
                    "temporal_id": "",
                    "evidence_text": question,
                    "arguments": arguments,
                }
            ]
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
    answer_variable_ids_by_surface: dict[str, set[str]] = {}
    for variable_id, surfaces in answer_variable_surfaces_by_id.items():
        for surface in surfaces:
            if surface:
                answer_variable_ids_by_surface.setdefault(surface, set()).add(variable_id)
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
                role = normalize(str(argument.get("role") or ""))
                argument_surfaces = {
                    normalize(str(value or ""))
                    for value in [argument.get("evidence_text"), argument.get("value")]
                    if str(value or "").strip()
                }
                if role == "answer" and target_kind != "answer_variable":
                    answer_surface_ids: set[str] = set()
                    for surface in argument_surfaces:
                        answer_surface_ids.update(answer_variable_ids_by_surface.get(surface, set()))
                    if len(answer_surface_ids) == 1:
                        argument["target_kind"] = "answer_variable"
                        argument["target_id"] = next(iter(answer_surface_ids))
                        argument["value"] = ""
                        target_kind = "answer_variable"
                        target_id = str(argument.get("target_id") or "").strip()
                        repaired = True
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
                if target_kind == "answer_variable" and role != "answer":
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
    if str(query_drs.get("temporal_scope") or "") not in {"", "earliest", "latest"}:
        errors.append(f"bad_temporal_scope:{query_drs.get('temporal_scope')}")
    if str(query_drs.get("aggregation") or "") not in {"", "count", "list", "set"}:
        errors.append(f"bad_aggregation:{query_drs.get('aggregation')}")
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
        if parent_id and parent_id == box_id:
            errors.append(f"self_parent_box:{box_id}")
        if holder_id and holder_id not in target_ids:
            errors.append(f"missing_holder_referent:{box_id}->{holder_id}")
        check_grounding(box.get("evidence_text"), f"box:{box_id}")
    errors.extend(box_parent_cycle_errors(boxes))
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


def _query_drs_retry_budgets(n_predict: int) -> list[int]:
    configured = os.environ.get("KMD_QUERY_DRS_RETRY_N_PREDICTS", "").strip()
    if configured:
        budgets: list[int] = []
        for item in configured.split(","):
            try:
                value = int(item.strip())
            except ValueError:
                continue
            if value > 0 and value != n_predict and value not in budgets:
                budgets.append(value)
        return budgets
    if n_predict > 256:
        return [256]
    return []


def _call_model_query_drs_full_once(
    question: str,
    client: LocalModelClient,
    *,
    n_predict: int,
    retry_index: int = 0,
    retry_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = build_query_drs_prompt(question)
    max_array_items = query_drs_array_max_items(n_predict)
    json_schema = query_drs_json_schema(question, max_array_items=max_array_items)
    constraint = _constraint_settings(QUERY_DRS_GRAMMAR, json_schema, QUERY_DRS_SCHEMA_VERSION)
    cache_settings = {
        "n_predict": n_predict,
        "schema": QUERY_DRS_SCHEMA_VERSION,
        "validation_policy": QUERY_DRS_VALIDATION_POLICY,
        "array_cap_policy": QUERY_DRS_ARRAY_CAP_POLICY,
        "output_budget_policy": QUERY_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY,
        "operator_schema_policy": QUERY_OPERATOR_SCHEMA_POLICY,
        "max_array_items": max_array_items,
        **constraint,
    }
    if retry_index:
        cache_settings = {
            **cache_settings,
            "request_failure_retry_policy": QUERY_DRS_REQUEST_FAILURE_RETRY_POLICY,
            "request_failure_retry_index": retry_index,
            "request_failure_retry_after": retry_after or {},
        }
    cache_context = {**cache_settings, "model_fingerprint": _client_fingerprint(client)}
    prompt_hash = _cache_hash(
        "query_drs",
        prompt,
        client,
        cache_settings,
    )
    cache_path = _cache_path("KMD_QUERY_DRS_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and not _query_drs_cached_retryable_failure(cached):
        cached.setdefault("cache_context", cache_context)
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
            "output_budget_policy": QUERY_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY,
            "operator_schema_policy": QUERY_OPERATOR_SCHEMA_POLICY,
            "max_array_items": max_array_items,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
        if retry_index:
            payload["request_failure_retry_policy"] = QUERY_DRS_REQUEST_FAILURE_RETRY_POLICY
            payload["request_failure_retry_index"] = retry_index
            payload["request_failure_retry_after"] = retry_after or {}
        payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
        _write_cache(cache_path, payload)
        return payload
    except Exception as exc:
        payload = {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            **constraint,
            "validation_policy": QUERY_DRS_VALIDATION_POLICY,
            "array_cap_policy": QUERY_DRS_ARRAY_CAP_POLICY,
            "output_budget_policy": QUERY_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY,
            "operator_schema_policy": QUERY_OPERATOR_SCHEMA_POLICY,
            "max_array_items": max_array_items,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
        if retry_index:
            payload["request_failure_retry_policy"] = QUERY_DRS_REQUEST_FAILURE_RETRY_POLICY
            payload["request_failure_retry_index"] = retry_index
            payload["request_failure_retry_after"] = retry_after or {}
        return payload
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
            "output_budget_policy": QUERY_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY,
            "operator_schema_policy": QUERY_OPERATOR_SCHEMA_POLICY,
            "max_array_items": max_array_items,
            "cache_context": cache_context,
            "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
            "validation": validation,
        }
        if retry_index:
            payload["request_failure_retry_policy"] = QUERY_DRS_REQUEST_FAILURE_RETRY_POLICY
            payload["request_failure_retry_index"] = retry_index
            payload["request_failure_retry_after"] = retry_after or {}
        payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
        "output_budget_policy": QUERY_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY,
        "operator_schema_policy": QUERY_OPERATOR_SCHEMA_POLICY,
        "max_array_items": max_array_items,
        "cache_context": cache_context,
        "validation": validation,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    if retry_index:
        payload["request_failure_retry_policy"] = QUERY_DRS_REQUEST_FAILURE_RETRY_POLICY
        payload["request_failure_retry_index"] = retry_index
        payload["request_failure_retry_after"] = retry_after or {}
    payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
    _write_cache(cache_path, payload)
    return payload


def call_model_query_drs(question: str, client: LocalModelClient, *, n_predict: int | None = None) -> dict[str, Any]:
    if (
        _compact_live_model_path_allowed(client)
        and os.environ.get("KMD_QUERY_DRS_COMPACT_FIRST", "1").strip().lower() not in {"0", "false", "no", "off"}
    ):
        compact = call_model_query_drs_compact(question, client)
        if compact.get("accepted"):
            if _compact_query_drs_answer_slot_undercovered(question, compact):
                fallback_n_predict = default_query_drs_n_predict(client, question) if n_predict is None else n_predict
                full = _call_model_query_drs_full_once(question, client, n_predict=fallback_n_predict)
                fallback_attempt = {
                    "policy": QUERY_DRS_COMPACT_UNDERCOVERAGE_POLICY,
                    "compact_prompt_hash": compact.get("prompt_hash"),
                    "full_prompt_hash": full.get("prompt_hash"),
                    "full_reason": full.get("reason"),
                    "full_accepted": bool(full.get("accepted")),
                }
                if full.get("accepted"):
                    full["compact_fallback_attempt"] = fallback_attempt
                    return full
                compact["compact_fallback_attempt"] = fallback_attempt
            return compact
    if n_predict is None:
        n_predict = default_query_drs_n_predict(client, question)
    result = _call_model_query_drs_full_once(question, client, n_predict=n_predict)
    if result.get("reason") != "request_failed":
        return result
    retry_attempts: list[dict[str, Any]] = [
        {
            "n_predict": n_predict,
            "reason": result.get("reason"),
            "error": result.get("error"),
            "elapsed": result.get("elapsed"),
            "prompt_hash": result.get("prompt_hash"),
        }
    ]
    last = result
    for retry_index, retry_budget in enumerate(_query_drs_retry_budgets(n_predict), start=1):
        retry_after = {
            "n_predict": n_predict if retry_index == 1 else retry_attempts[-1].get("n_predict"),
            "reason": last.get("reason"),
            "error": last.get("error"),
            "prompt_hash": last.get("prompt_hash"),
        }
        last = _call_model_query_drs_full_once(
            question,
            client,
            n_predict=retry_budget,
            retry_index=retry_index,
            retry_after=retry_after,
        )
        retry_attempts.append(
            {
                "n_predict": retry_budget,
                "reason": last.get("reason"),
                "error": last.get("error"),
                "elapsed": last.get("elapsed"),
                "prompt_hash": last.get("prompt_hash"),
            }
        )
        if last.get("reason") != "request_failed":
            last["query_drs_retry_attempts"] = retry_attempts
            return last
    last["query_drs_retry_attempts"] = retry_attempts
    return last


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
    cache_settings = {
        "n_predict": n_predict,
        "schema": ANSWER_SCHEMA_VERSION,
        "expected_answer_type": expected_answer_type,
        **constraint,
    }
    cache_context = {
        **cache_settings,
        "model_fingerprint": _client_fingerprint(client),
        "evidence_count": len(evidence_items),
    }
    prompt_hash = _cache_hash(
        "evidence_answer",
        prompt,
        client,
        cache_settings,
    )
    cache_path = _cache_path("KMD_EVIDENCE_ANSWER_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and not _cached_evidence_answer_retryable(cached):
        cached.setdefault("cache_context", cache_context)
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
            "cache_context": cache_context,
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
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
        payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
        payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
        "cache_context": cache_context,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
    cache_context: dict[str, Any] | None = None,
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
            "cache_context": cache_context or {},
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
            "cache_context": cache_context or {},
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
        "cache_context": cache_context or {},
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
    cache_settings = {"n_predict": n_predict, "schema": ANSWER_SCHEMA_VERSION, **constraint}
    cache_context = {
        **cache_settings,
        "model_fingerprint": _client_fingerprint(client),
        "evidence_count": len(evidence_items),
        "discourse_record_count": len(discourse_records or []),
        "repair": True,
    }
    prompt_hash = _cache_hash(
        "query_evidence_answer_repair",
        prompt,
        client,
        cache_settings,
    )
    cache_path = _cache_path("KMD_QUERY_EVIDENCE_REPAIR_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and cached.get("reason") not in {
        "invalid_json",
        "schema_validation_failed",
        "grounding_validation_failed",
        "request_failed",
    }:
        cached.setdefault("cache_context", cache_context)
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
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
        return _with_model_input_audits(payload, exc)
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
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
        payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
        cache_context=cache_context,
    )
    payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
    cache_settings = {"n_predict": n_predict, "schema": ANSWER_SCHEMA_VERSION, **constraint}
    cache_context = {
        **cache_settings,
        "model_fingerprint": _client_fingerprint(client),
        "evidence_count": len(evidence_items),
        "discourse_record_count": len(discourse_records or []),
        "repair": False,
    }
    prompt_hash = _cache_hash(
        "query_evidence_answer",
        prompt,
        client,
        cache_settings,
    )
    cache_path = _cache_path("KMD_QUERY_EVIDENCE_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and not _cached_request_failed(cached):
        cached.setdefault("cache_context", cache_context)
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
            "cache_context": cache_context,
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
            payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
            _write_cache(cache_path, payload)
            return payload
        payload = {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
            "repair_failure_reason": repaired.get("reason"),
            "repair_prompt_hash": repaired.get("prompt_hash"),
            "repair_cache_context": repaired.get("cache_context"),
        }
        if repaired.get("reason") != "request_failed":
            payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
        cache_context=cache_context,
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
    payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
    client: LocalModelClient | None,
    *,
    rel_path: str,
    n_predict: int,
) -> tuple[str, dict[str, Any]]:
    context_size = _client_context_size(client)
    budget: dict[str, Any] = {
        "runtime_context_size": context_size,
        "reserved_output_tokens": int(n_predict),
        "context_source": "client_metadata" if context_size > 0 else "unavailable",
        "context_budget_policy": CHUNK_FRAME_CONTEXT_BUDGET_POLICY,
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


def chunk_frame_cache_context(
    client: LocalModelClient | None,
    *,
    n_predict: int | None = None,
    rel_path: str = "",
    chunk_text: str = "",
) -> dict[str, Any]:
    constraint = _constraint_settings(FRAME_EXTRACTION_GRAMMAR, FRAME_JSON_SCHEMA, CHUNK_FRAME_SCHEMA_VERSION)
    if n_predict is None:
        n_predict = default_chunk_frame_n_predict(client)
    context = {
        "prompt_version": PROMPT_VERSION,
        "schema_version": CHUNK_FRAME_SCHEMA_VERSION,
        "context_budget_policy": CHUNK_FRAME_CONTEXT_BUDGET_POLICY,
        **constraint,
        "n_predict": int(n_predict),
        "model_fingerprint": _client_fingerprint(client),
    }
    if rel_path:
        context["source_rel_path"] = rel_path
    if chunk_text:
        _prompt_chunk, context_budget = _context_limited_chunk_frame_text(
            str(chunk_text),
            client,
            rel_path=rel_path,
            n_predict=int(n_predict),
        )
        context["context_budget"] = context_budget
    return context


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
    cache_settings = {
        "n_predict": n_predict,
        "schema": CHUNK_FRAME_SCHEMA_VERSION,
        **constraint,
        "context_budget": context_budget,
    }
    cache_context = {**cache_settings, "model_fingerprint": _client_fingerprint(client)}
    prompt_hash = _cache_hash(
        "chunk_frames",
        prompt,
        client,
        cache_settings,
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
            "context_budget": context_budget,
            "cache_context": cache_context,
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
            "context_budget": context_budget,
            "cache_context": cache_context,
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
            "cache_context": cache_context,
        }
    return {
        "accepted": True,
        "frames": grounded,
        "raw_text": raw,
        "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
        "prompt_hash": prompt_hash,
        "grammar_hash": grammar_hash,
        "context_budget": context_budget,
        "cache_context": cache_context,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
        "rejected_for_grounding": rejected_for_grounding,
    }


CHUNK_DRS_GRAMMAR = ""
COMPACT_SOURCE_TEMPORAL_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2})?|\d{1,2}:\d{2})\b")


def _build_compact_chunk_drs_prompt_v1(chunk_text: str, *, rel_path: str = "") -> str:
    return (
        "JSON only. Extract compact source-grounded DRS facts from this raw text chunk. "
        "Output exactly {\"facts\":[{\"p\":\"\",\"agent\":\"\",\"patient\":\"\",\"value\":\"\",\"e\":\"\"}]}. "
        "p is the model-chosen predicate or relation word. agent, patient, and value are exact source strings when "
        "the source supports those roles; leave a field empty when unsupported. e is one exact contiguous source "
        "span containing the non-empty role values. Use only source-grounded asserted, reported, negated, or scoped "
        "conditions from the chunk. Return {\"facts\":[]} when the chunk asserts no useful source-grounded DRS "
        "condition. Do not answer questions and do not use outside knowledge. "
        + json.dumps({"source_id": rel_path, "chunk": chunk_text}, ensure_ascii=False)
    )


def _build_compact_chunk_drs_prompt_v2(chunk_text: str, *, rel_path: str = "") -> str:
    return (
        "JSON only. Extract compact source-grounded DRS facts from this raw text chunk. "
        "Output exactly {\"facts\":[{\"p\":\"\",\"agent\":\"\",\"patient\":\"\",\"value\":\"\",\"time\":\"\",\"e\":\"\"}]}. "
        "p is the model-chosen predicate or relation word. agent, patient, and value are exact source strings when "
        "the source supports those roles; time is an exact source string only when an explicit timestamp, date, or "
        "ordering phrase scopes the fact; leave a field empty when unsupported. e is one exact contiguous source "
        "span containing the non-empty role values and time when present. Use only source-grounded asserted, reported, negated, or scoped "
        "conditions from the chunk. Return {\"facts\":[]} when the chunk asserts no useful source-grounded DRS "
        "condition. Do not answer questions and do not use outside knowledge. "
        + json.dumps({"source_id": rel_path, "chunk": chunk_text}, ensure_ascii=False)
    )


def build_compact_chunk_drs_prompt(chunk_text: str, *, rel_path: str = "") -> str:
    return (
        "JSON only. Extract compact source-grounded DRS facts from this raw text chunk. "
        "Output exactly one object with mandatory key facts. Each fact must have mandatory keys p, e, arguments, "
        "temporal_text, scope. facts and arguments must always be JSON arrays, even when empty. "
        "Use this shape: {\"facts\":[{\"p\":predicate,\"e\":evidence_span,\"arguments\":[{\"role\":role,\"value\":value}],"
        "\"temporal_text\":optional_time,\"scope\":asserted_or_negated_or_reported_or_possible_or_hypothetical}]}. "
        "Each e and every argument value must be exact substrings of the chunk. Use short predicate labels from the chunk. Extract all useful "
        "conditions from the chunk. Include source-stated definitions, meanings, names, aliases, and terminology as ordinary "
        "relations when the chunk asserts them. Return {\"facts\":[]} when the chunk asserts no useful source-grounded DRS "
        "condition. Do not answer questions and do not use outside knowledge. "
        + json.dumps({"source_id": rel_path, "chunk": chunk_text}, ensure_ascii=False)
    )


def _compact_fact_items(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    facts = parsed.get("facts")
    if not isinstance(facts, list):
        return []
    return [item for item in facts if isinstance(item, dict)]


def _compact_cached_payload_has_conditions(cached: dict[str, Any]) -> bool:
    drs = cached.get("drs") if isinstance(cached, dict) else None
    if not isinstance(drs, dict):
        return False
    conditions = drs.get("conditions")
    return isinstance(conditions, list) and any(isinstance(item, dict) for item in conditions)


def _source_segment_for_values(source_text: str, values: list[str], fallback: str = "") -> str:
    fallback = str(fallback or "").strip()
    if fallback and fallback in source_text:
        return fallback
    required = [value for value in values if value and value in source_text]
    if not required:
        return ""
    candidates = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", source_text) if part.strip()]
    candidates.append(source_text.strip())
    for candidate in candidates:
        if all(value in candidate for value in required) and candidate in source_text:
            return candidate
    return ""


COMPACT_TEMPORAL_FIELDS = {"time", "timestamp", "temporal", "temporal_text"}
COMPACT_LITERAL_ARGUMENT_ROLES = {"state", "value"}
COMPACT_SCOPE_FIELDS = {"scope", "context", "modality", "box_kind"}
COMPACT_SCOPE_ALIASES = {
    "assert": "asserted",
    "assertion": "asserted",
    "belief": "believed",
    "believe": "believed",
    "conditional antecedent": "conditional_antecedent",
    "conditional consequent": "conditional_consequent",
    "dream": "dreamed",
    "fiction": "fictional",
    "hypothesis": "hypothetical",
    "negation": "negated",
    "possibility": "possible",
    "quotation": "quoted",
    "quote": "quoted",
    "report": "reported",
    "uncertainty": "uncertain",
}


def _compact_fact_temporal_text(fact: dict[str, Any], source_text: str) -> str:
    for key in COMPACT_TEMPORAL_FIELDS:
        text = str(fact.get(key) or "").strip()
        if text and text in source_text:
            return text
    role_values = fact.get("roles")
    if isinstance(role_values, dict):
        for key in COMPACT_TEMPORAL_FIELDS:
            text = str(role_values.get(key) or "").strip()
            if text and text in source_text:
                return text
    raw_arguments = fact.get("arguments")
    if isinstance(raw_arguments, list):
        for item in raw_arguments:
            if not isinstance(item, dict):
                continue
            for key in COMPACT_TEMPORAL_FIELDS:
                text = str(item.get(key) or "").strip()
                if text and text in source_text:
                    return text
    return ""


def _compact_fact_scope(fact: dict[str, Any]) -> str:
    for key in COMPACT_SCOPE_FIELDS:
        raw = normalize(str(fact.get(key) or ""))
        if not raw:
            continue
        value = COMPACT_SCOPE_ALIASES.get(raw, raw)
        if value in DRS_CONTEXT_KINDS:
            return value
    role_values = fact.get("roles")
    if isinstance(role_values, dict):
        for key in COMPACT_SCOPE_FIELDS:
            raw = normalize(str(role_values.get(key) or ""))
            if not raw:
                continue
            value = COMPACT_SCOPE_ALIASES.get(raw, raw)
            if value in DRS_CONTEXT_KINDS:
                return value
    return "asserted"


def _source_temporal_text_for_evidence(source_text: str, evidence: str) -> str:
    evidence = str(evidence or "").strip()
    if not evidence or evidence not in source_text:
        return ""
    start = source_text.find(evidence)
    search_text = source_text[max(0, start - 80) : start + min(len(evidence), 80)]
    matches = [match.group(0) for match in COMPACT_SOURCE_TEMPORAL_RE.finditer(search_text)]
    return matches[-1] if matches else ""


def _compact_fact_arguments(fact: dict[str, Any], source_text: str) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    role_values = fact.get("roles")
    if isinstance(role_values, dict):
        for role, value in role_values.items():
            role_key = str(role or "").strip()
            if role_key in COMPACT_TEMPORAL_FIELDS or role_key in COMPACT_SCOPE_FIELDS:
                continue
            text = str(value or "").strip()
            if text and text in source_text:
                values.append((role_key or "argument", text))
    raw_arguments = fact.get("arguments")
    if isinstance(raw_arguments, list):
        for item in raw_arguments:
            if isinstance(item, str):
                text = item.strip()
                if text and text in source_text:
                    values.append(("argument", text))
            elif isinstance(item, dict):
                explicit_role = str(item.get("role") or "").strip()
                explicit_value = str(item.get("value") or "").strip()
                if explicit_value and explicit_value in source_text:
                    values.append((explicit_role or "argument", explicit_value))
                    continue
                for role, value in item.items():
                    role_key = str(role or "").strip()
                    if role_key in {"role", "value"} or role_key in COMPACT_TEMPORAL_FIELDS or role_key in COMPACT_SCOPE_FIELDS:
                        continue
                    text = str(value or "").strip()
                    if text and text in source_text:
                        values.append((role_key or "argument", text))
    for role in ["agent", "patient", "theme", "holder", "topic", "value", "state", "identifier", "location"]:
        text = str(fact.get(role) or "").strip()
        if text and text in source_text:
            values.append((role, text))
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for role, value in values:
        key = (role, value)
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _compact_literal_argument_value(argument: dict[str, Any], referents_by_id: dict[str, dict[str, Any]]) -> str:
    value = str(argument.get("value") or "").strip()
    if value:
        return value
    referent = referents_by_id.get(str(argument.get("target_id") or ""))
    if not referent:
        return ""
    return str(referent.get("label") or referent.get("evidence_text") or "").strip()


def _attach_compact_source_temporals(payload: dict[str, Any], source_text: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("drs"), dict):
        return payload
    drs = {**payload["drs"]}
    conditions = [dict(item) for item in drs.get("conditions", []) if isinstance(item, dict)]
    referents = [dict(item) for item in drs.get("referents", []) if isinstance(item, dict)]
    temporals = [dict(item) for item in drs.get("temporal_records", []) if isinstance(item, dict)]
    referents_by_id = {str(item.get("id") or ""): item for item in referents}
    temporal_ids_by_value = {str(item.get("value") or ""): str(item.get("id") or "") for item in temporals}
    changed = False
    for condition in conditions:
        temporal_id = str(condition.get("temporal_id") or "").strip()
        if not temporal_id:
            temporal_text = _source_temporal_text_for_evidence(source_text, str(condition.get("evidence_text") or ""))
            if temporal_text:
                temporal_id = temporal_ids_by_value.get(temporal_text, "")
                if not temporal_id:
                    temporal_id = f"t{len(temporals)}"
                    temporal_ids_by_value[temporal_text] = temporal_id
                    temporals.append(
                        {
                            "id": temporal_id,
                            "value": temporal_text,
                            "value_type": "timestamp",
                            "evidence_text": temporal_text,
                        }
                    )
                condition["temporal_id"] = temporal_id
                changed = True
        if temporal_id and isinstance(condition.get("arguments"), list):
            repaired_args = []
            for argument in condition["arguments"]:
                if not isinstance(argument, dict):
                    continue
                argument = {**argument}
                role_norm = normalize(str(argument.get("role") or ""))
                if role_norm in COMPACT_LITERAL_ARGUMENT_ROLES and str(argument.get("target_kind") or "") == "referent":
                    value = _compact_literal_argument_value(argument, referents_by_id)
                    if value:
                        argument["target_kind"] = "literal"
                        argument["target_id"] = ""
                        argument["value"] = value
                        argument["value_type"] = role_norm
                        argument["evidence_text"] = str(argument.get("evidence_text") or value)
                        changed = True
                repaired_args.append(argument)
            condition["arguments"] = repaired_args
    if not changed:
        return payload
    drs["conditions"] = conditions
    drs["referents"] = referents
    drs["temporal_records"] = temporals
    return {**payload, "drs": drs, "compact_temporal_source_policy": CHUNK_DRS_COMPACT_TEMPORAL_SOURCE_POLICY}


def _compact_chunk_drs_to_payload(parsed: dict[str, Any], source_text: str, *, rel_path: str = "") -> dict[str, Any]:
    referents: list[dict[str, Any]] = []
    referent_ids_by_value: dict[str, str] = {}
    conditions: list[dict[str, Any]] = []
    temporal_records: list[dict[str, Any]] = []
    temporal_ids_by_value: dict[str, str] = {}
    evidence_spans: list[str] = []
    boxes: list[dict[str, Any]] = [
        {
            "id": "b0",
            "kind": "asserted",
            "parent_id": "",
            "holder_referent_id": "",
            "evidence_text": "",
            "confidence": 0.72,
        }
    ]
    box_ids_by_scope_evidence: dict[tuple[str, str], str] = {}

    def referent_id_for(value: str) -> str:
        key = normalize(value)
        existing = referent_ids_by_value.get(key)
        if existing:
            return existing
        referent_id = f"r{len(referents)}"
        referent_ids_by_value[key] = referent_id
        referents.append(
            {
                "id": referent_id,
                "label": value,
                "kind": "unknown",
                "evidence_text": value,
            }
        )
        return referent_id

    for fact in _compact_fact_items(parsed):
        predicate = str(fact.get("p") or fact.get("predicate") or "").strip()
        arguments = _compact_fact_arguments(fact, source_text)
        temporal_text = _compact_fact_temporal_text(fact, source_text)
        evidence = _source_segment_for_values(
            source_text,
            [*[value for _role, value in arguments], temporal_text],
            str(fact.get("e") or fact.get("evidence_text") or ""),
        )
        if not predicate or not evidence:
            continue
        scope = _compact_fact_scope(fact)
        box_id = "b0"
        if scope != "asserted":
            box_key = (scope, evidence)
            box_id = box_ids_by_scope_evidence.get(box_key, "")
            if not box_id:
                box_id = f"b{len(boxes)}"
                box_ids_by_scope_evidence[box_key] = box_id
                boxes.append(
                    {
                        "id": box_id,
                        "kind": scope,
                        "parent_id": "b0",
                        "holder_referent_id": "",
                        "evidence_text": evidence,
                        "confidence": 0.72,
                    }
                )
        temporal_id = ""
        if temporal_text:
            temporal_id = temporal_ids_by_value.get(temporal_text, "")
            if not temporal_id:
                temporal_id = f"t{len(temporal_records)}"
                temporal_ids_by_value[temporal_text] = temporal_id
                temporal_records.append(
                    {
                        "id": temporal_id,
                        "value": temporal_text,
                        "value_type": "timestamp",
                        "evidence_text": temporal_text,
                    }
                )
        condition_arguments = []
        for role, value in arguments:
            if normalize(role) in COMPACT_LITERAL_ARGUMENT_ROLES:
                condition_arguments.append(
                    {
                        "role": role,
                        "target_kind": "literal",
                        "target_id": "",
                        "value": value,
                        "value_type": normalize(role) or "unknown",
                        "evidence_text": value,
                    }
                )
            else:
                condition_arguments.append(
                    {
                        "role": role,
                        "target_kind": "referent",
                        "target_id": referent_id_for(value),
                        "value": "",
                        "value_type": "unknown",
                        "evidence_text": value,
                    }
                )
        condition_id = f"c{len(conditions)}"
        conditions.append(
            {
                "id": condition_id,
                "box_id": box_id,
                "predicate": predicate,
                "polarity": "positive",
                "modality": "asserted",
                "temporal_id": temporal_id,
                "evidence_text": evidence,
                "arguments": condition_arguments,
            }
        )
        if evidence not in evidence_spans:
            evidence_spans.append(evidence)
    return _attach_compact_source_temporals({
        "drs": {
            "schema_version": CHUNK_DRS_SCHEMA_VERSION,
            "source_id": rel_path,
            "referents": referents,
            "boxes": boxes,
            "conditions": conditions,
            "identity_hypotheses": [],
            "temporal_records": temporal_records,
            "evidence_spans": evidence_spans,
        }
    }, source_text)


def _compact_chunk_drs_enabled() -> bool:
    return os.environ.get("KMD_CHUNK_DRS_COMPACT_FIRST", "1").strip().lower() not in {"0", "false", "no", "off"}


def _compact_live_model_path_allowed(client: LocalModelClient) -> bool:
    return isinstance(client, LocalModelClient) or os.environ.get(
        "KMD_FORCE_COMPACT_MODEL_PATH",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _compact_chunk_drs_eligible(chunk_text: str) -> bool:
    configured = os.environ.get("KMD_CHUNK_DRS_COMPACT_MAX_CHARS", "")
    try:
        max_chars = int(configured) if configured else 1200
    except ValueError:
        max_chars = 1200
    return len(str(chunk_text or "")) <= max_chars


def _compact_chunk_drs_retry_budgets(n_predict: int) -> list[int]:
    configured = os.environ.get("KMD_CHUNK_DRS_COMPACT_RETRY_N_PREDICTS", "").strip()
    if configured:
        budgets: list[int] = []
        for item in configured.split(","):
            try:
                value = int(item.strip())
            except ValueError:
                continue
            if value > 0 and value != n_predict and value not in budgets:
                budgets.append(value)
        return budgets
    budgets = [max(768, n_predict * 2), max(1536, n_predict * 4)]
    return [value for value in dict.fromkeys(budgets) if value > n_predict]


def _finalize_compact_cached_payload(
    payload: dict[str, Any],
    chunk_text: str,
    cache_context: dict[str, Any],
    *,
    cache_path: Path | None = None,
    migrated_from_prompt_hash: str = "",
) -> dict[str, Any]:
    payload = {**payload}
    payload.setdefault("cache_context", cache_context)
    if not payload.get("accepted") or not isinstance(payload.get("drs"), dict):
        return payload
    upgraded = _attach_compact_source_temporals(payload, chunk_text)
    repaired = _repair_chunk_drs_payload({"drs": upgraded["drs"]}, chunk_text)
    validation = _validate_chunk_drs_payload(repaired, chunk_text)
    if not validation.get("schema_valid"):
        return payload
    finalized = {
        **upgraded,
        "drs": repaired["drs"],
        "validation": validation,
        "cache_context": cache_context,
        "compact_fact_policy": CHUNK_DRS_COMPACT_FACT_POLICY,
        "fresh_or_cached": "cache",
    }
    if migrated_from_prompt_hash:
        finalized["compact_legacy_cache_migration"] = {
            "from_policy": CHUNK_DRS_COMPACT_FACT_POLICY_LEGACY,
            "from_prompt_hash": migrated_from_prompt_hash,
        }
    if cache_path is not None and finalized != payload:
        finalized = _with_model_input_audits(finalized, locals().get("parsed"), locals().get("cached"), locals().get("source_payload"), locals().get("payload"))
        _write_cache(cache_path, finalized)
    return finalized


def call_model_chunk_drs_compact(
    chunk_text: str,
    client: LocalModelClient,
    *,
    rel_path: str = "",
    n_predict: int | None = None,
    refresh_empty_legacy: bool = False,
) -> dict[str, Any]:
    if n_predict is None:
        n_predict = default_compact_chunk_drs_n_predict(chunk_text)
    prompt = build_compact_chunk_drs_prompt(chunk_text, rel_path=rel_path)
    source_text_hash = hashlib.sha256(str(chunk_text or "").encode("utf-8", errors="replace")).hexdigest()
    budgets = [n_predict, *_compact_chunk_drs_retry_budgets(n_predict)]
    failures: list[dict[str, Any]] = []
    last_payload: dict[str, Any] | None = None

    def condition_retry_cache_available(next_index: int, retry_after: dict[str, Any]) -> bool:
        if next_index >= len(budgets):
            return False
        retry_budget = budgets[next_index]
        compact_constraint = _constraint_settings(CHUNK_DRS_GRAMMAR, COMPACT_CHUNK_DRS_JSON_SCHEMA, CHUNK_DRS_SCHEMA_VERSION)
        retry_settings = {
            "n_predict": retry_budget,
            "schema": CHUNK_DRS_SCHEMA_VERSION,
            "compact_fact_policy": CHUNK_DRS_COMPACT_FACT_POLICY,
            **compact_constraint,
            "source_text_hash": source_text_hash,
            "compact_retry_policy": CHUNK_DRS_COMPACT_RETRY_POLICY,
            "compact_retry_index": next_index,
            "compact_retry_after": retry_after,
        }
        retry_prompt_hash = _cache_hash("chunk_drs_compact", prompt, client, retry_settings)
        retry_cache_path = _cache_path("KMD_CHUNK_DRS_CACHE_DIR", retry_prompt_hash)
        retry_cached = _read_cache(retry_cache_path)
        return bool(
            retry_cached is not None
            and not _query_drs_cached_retryable_failure(retry_cached)
            and _compact_cached_payload_has_conditions(retry_cached)
        )

    def condition_source_cache(cache_path: Path | None, cache_context: dict[str, Any]) -> dict[str, Any] | None:
        if cache_path is None:
            return None
        cache_dir = cache_path.parent
        if not cache_dir.exists():
            return None
        candidates: list[tuple[int, float, Path, dict[str, Any]]] = []
        for candidate_path in cache_dir.glob("*.json"):
            if candidate_path == cache_path:
                continue
            candidate = _read_cache(candidate_path)
            if (
                not isinstance(candidate, dict)
                or _query_drs_cached_retryable_failure(candidate)
                or not _compact_cached_payload_has_conditions(candidate)
            ):
                continue
            if candidate.get("accepted") is not True:
                continue
            candidate_context = candidate.get("cache_context") if isinstance(candidate.get("cache_context"), dict) else {}
            if str(candidate_context.get("constraint_transport_policy") or "") != str(cache_context.get("constraint_transport_policy") or ""):
                continue
            if str(candidate_context.get("constraint_mode") or "") != str(cache_context.get("constraint_mode") or ""):
                continue
            if str(candidate_context.get("json_schema_hash") or "") != str(cache_context.get("json_schema_hash") or ""):
                continue
            if str(candidate_context.get("source_text_hash") or "") != source_text_hash:
                continue
            candidate_rel_path = str(candidate_context.get("source_rel_path") or "")
            if rel_path and candidate_rel_path and candidate_rel_path != rel_path:
                continue
            candidate_policy = str(candidate.get("compact_fact_policy") or candidate_context.get("compact_fact_policy") or "")
            if candidate_policy != CHUNK_DRS_COMPACT_FACT_POLICY:
                continue
            candidate_schema = str(candidate_context.get("schema") or candidate.get("schema_version") or "")
            if candidate_schema and candidate_schema != CHUNK_DRS_SCHEMA_VERSION:
                continue
            conditions = candidate.get("drs", {}).get("conditions") if isinstance(candidate.get("drs"), dict) else []
            condition_count = len([item for item in conditions if isinstance(item, dict)]) if isinstance(conditions, list) else 0
            try:
                mtime = candidate_path.stat().st_mtime
            except OSError:
                mtime = 0.0
            candidates.append((condition_count, mtime, candidate_path, candidate))
        if not candidates:
            return None
        _, _, source_path, source_payload = max(candidates, key=lambda item: (item[0], item[1], item[2].name))
        finalized = _finalize_compact_cached_payload(
            source_payload,
            chunk_text,
            cache_context,
            cache_path=cache_path,
        )
        finalized["compact_source_cache_reuse"] = {
            "from_prompt_hash": str(source_payload.get("prompt_hash") or source_path.stem),
            "source_text_hash": source_text_hash,
        }
        finalized = _with_model_input_audits(finalized, locals().get("parsed"), locals().get("cached"), locals().get("source_payload"), locals().get("payload"))
        _write_cache(cache_path, finalized)
        return finalized

    compact_constraint = _constraint_settings(CHUNK_DRS_GRAMMAR, COMPACT_CHUNK_DRS_JSON_SCHEMA, CHUNK_DRS_SCHEMA_VERSION)
    for retry_index, budget in enumerate(budgets):
        cache_settings = {
            "n_predict": budget,
            "schema": CHUNK_DRS_SCHEMA_VERSION,
            "compact_fact_policy": CHUNK_DRS_COMPACT_FACT_POLICY,
            **compact_constraint,
            "source_text_hash": source_text_hash,
        }
        if retry_index:
            cache_settings = {
                **cache_settings,
                "compact_retry_policy": CHUNK_DRS_COMPACT_RETRY_POLICY,
                "compact_retry_index": retry_index,
                "compact_retry_after": failures[-1] if failures else {},
            }
        cache_context = {
            **cache_settings,
            "model_fingerprint": _client_fingerprint(client),
            "source_rel_path": rel_path,
        }
        prompt_hash = _cache_hash("chunk_drs_compact", prompt, client, cache_settings)
        cache_path = _cache_path("KMD_CHUNK_DRS_CACHE_DIR", prompt_hash)
        cached = _read_cache(cache_path)
        cached_empty_undercoverage = False
        if cached is not None and not _query_drs_cached_retryable_failure(cached):
            finalized = _finalize_compact_cached_payload(cached, chunk_text, cache_context, cache_path=cache_path)
            if refresh_empty_legacy and not _compact_cached_payload_has_conditions(finalized):
                failures.append(
                    {
                        "n_predict": budget,
                        "reason": "empty_compact_drs_cache",
                        "elapsed": finalized.get("elapsed"),
                        "prompt_hash": prompt_hash,
                    }
                )
                last_payload = finalized
                cached_empty_undercoverage = True
            elif not _compact_cached_payload_has_conditions(finalized):
                failure_marker = {
                    "n_predict": budget,
                    "reason": "empty_compact_drs_cache",
                    "elapsed": finalized.get("elapsed"),
                    "prompt_hash": prompt_hash,
                }
                if condition_retry_cache_available(retry_index + 1, failure_marker):
                    failures.append(failure_marker)
                    last_payload = finalized
                    cached_empty_undercoverage = True
                else:
                    source_cached = condition_source_cache(cache_path, cache_context)
                    if source_cached is not None:
                        return source_cached
                    return finalized
            else:
                return finalized
        if not retry_index:
            legacy_variants = [
                (_build_compact_chunk_drs_prompt_v2(chunk_text, rel_path=rel_path), CHUNK_DRS_COMPACT_FACT_POLICY_PREVIOUS),
                (_build_compact_chunk_drs_prompt_v1(chunk_text, rel_path=rel_path), CHUNK_DRS_COMPACT_FACT_POLICY_LEGACY),
            ]
            for legacy_prompt, legacy_policy in legacy_variants:
                legacy_settings = {
                    **cache_settings,
                    "compact_fact_policy": legacy_policy,
                }
                legacy_prompt_hash = _cache_hash("chunk_drs_compact", legacy_prompt, client, legacy_settings)
                legacy_cache_path = _cache_path("KMD_CHUNK_DRS_CACHE_DIR", legacy_prompt_hash)
                legacy_cached = _read_cache(legacy_cache_path)
                if (
                    legacy_cached is not None
                    and not _query_drs_cached_retryable_failure(legacy_cached)
                    and (not refresh_empty_legacy or _compact_cached_payload_has_conditions(legacy_cached))
                ):
                    finalized_legacy = _finalize_compact_cached_payload(
                        legacy_cached,
                        chunk_text,
                        cache_context,
                        cache_path=cache_path,
                        migrated_from_prompt_hash=legacy_prompt_hash,
                    )
                    if not _compact_cached_payload_has_conditions(finalized_legacy):
                        source_cached = condition_source_cache(cache_path, cache_context)
                        if source_cached is not None:
                            return source_cached
                    return finalized_legacy
        source_cached = condition_source_cache(cache_path, cache_context)
        if source_cached is not None:
            return source_cached
        if cached_empty_undercoverage:
            continue
        start = time.time()
        try:
            parsed = _complete_structured(
                client,
                prompt,
                n_predict=budget,
                grammar=CHUNK_DRS_GRAMMAR,
                json_schema=COMPACT_CHUNK_DRS_JSON_SCHEMA,
            )
        except LocalModelJSONError as exc:
            payload = {
                "accepted": False,
                "reason": "invalid_json",
                "error": str(exc),
                "raw_text": exc.raw_text,
                "raw_snippet": exc.snippet,
                "prompt_hash": prompt_hash,
                "compact_fact_policy": CHUNK_DRS_COMPACT_FACT_POLICY,
                "cache_context": cache_context,
                "elapsed": round(time.time() - start, 3),
            }
            if retry_index:
                payload["compact_retry_policy"] = CHUNK_DRS_COMPACT_RETRY_POLICY
                payload["compact_retry_index"] = retry_index
            payload = _with_model_input_audits(payload, exc)
            _write_cache(cache_path, payload)
            failures.append(
                {
                    "n_predict": budget,
                    "reason": payload.get("reason"),
                    "error": payload.get("error"),
                    "elapsed": payload.get("elapsed"),
                    "prompt_hash": prompt_hash,
                }
            )
            last_payload = payload
            continue
        except Exception as exc:
            payload = {
                "accepted": False,
                "reason": "request_failed",
                "error": str(exc),
                "prompt_hash": prompt_hash,
                "compact_fact_policy": CHUNK_DRS_COMPACT_FACT_POLICY,
                "cache_context": cache_context,
                "elapsed": round(time.time() - start, 3),
            }
            if retry_index:
                payload["compact_retry_policy"] = CHUNK_DRS_COMPACT_RETRY_POLICY
                payload["compact_retry_index"] = retry_index
            payload = _with_model_input_audits(payload, exc)
            failures.append(
                {
                    "n_predict": budget,
                    "reason": payload.get("reason"),
                    "error": payload.get("error"),
                    "elapsed": payload.get("elapsed"),
                    "prompt_hash": prompt_hash,
                }
            )
            last_payload = payload
            continue
        raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
        drs_payload = _compact_chunk_drs_to_payload(parsed if isinstance(parsed, dict) else {}, chunk_text, rel_path=rel_path)
        drs_payload = _repair_chunk_drs_payload(drs_payload, chunk_text)
        validation = _validate_chunk_drs_payload(drs_payload, chunk_text)
        if not validation.get("schema_valid"):
            payload = {
                "accepted": False,
                "reason": "schema_validation_failed",
                "raw_text": raw,
                "prompt_hash": prompt_hash,
                "compact_fact_policy": CHUNK_DRS_COMPACT_FACT_POLICY,
                "cache_context": cache_context,
                "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
                "validation": validation,
            }
            if retry_index:
                payload["compact_retry_policy"] = CHUNK_DRS_COMPACT_RETRY_POLICY
                payload["compact_retry_index"] = retry_index
            payload = _with_model_input_audits(payload, parsed)
            _write_cache(cache_path, payload)
            failures.append(
                {
                    "n_predict": budget,
                    "reason": payload.get("reason"),
                    "elapsed": payload.get("elapsed"),
                    "prompt_hash": prompt_hash,
                    "validation": validation,
                }
            )
            last_payload = payload
            continue
        payload = {
            "accepted": True,
            "reason": "compact_drs",
            "drs": drs_payload["drs"],
            "raw_text": raw,
            "elapsed": parsed.get("_model_elapsed_seconds", round(time.time() - start, 3)),
            "prompt_hash": prompt_hash,
            "compact_fact_policy": CHUNK_DRS_COMPACT_FACT_POLICY,
            "cache_context": cache_context,
            "validation": validation,
            "context_budget": {
                "compact": True,
                "input_chars": len(chunk_text),
                "reserved_output_tokens": budget,
                "source_text_hash": source_text_hash,
            },
            "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
            "fresh_or_cached": "fresh",
            "compact": True,
        }
        if failures:
            payload["compact_retry_attempts"] = failures
        if retry_index:
            payload["compact_retry_policy"] = CHUNK_DRS_COMPACT_RETRY_POLICY
            payload["compact_retry_index"] = retry_index
        payload = _with_model_input_audits(payload, parsed)
        _write_cache(cache_path, payload)
        if refresh_empty_legacy and not _compact_cached_payload_has_conditions(payload):
            failures.append(
                {
                    "n_predict": budget,
                    "reason": "empty_compact_drs",
                    "elapsed": payload.get("elapsed"),
                    "prompt_hash": prompt_hash,
                    "validation": validation,
                }
            )
            last_payload = payload
            continue
        return payload
    if last_payload is not None:
        if failures:
            last_payload["compact_retry_attempts"] = failures
        return last_payload
    return {
        "accepted": False,
        "reason": "schema_validation_failed",
        "compact_fact_policy": CHUNK_DRS_COMPACT_FACT_POLICY,
        "cache_context": {"source_rel_path": rel_path, "source_text_hash": source_text_hash},
    }


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
        "or use handler names. Emit exactly one root asserted box with id b0 and parent_id ''; that root is only "
        "the containing discourse, not permission to flatten embedded scoped propositions into asserted fact. "
        "Use subordinate boxes for negation, reports, quotes, beliefs, conditionals, uncertainty, dreams, fiction, "
        "and modality. Conditions for embedded propositions inside those contexts must use the subordinate box_id, "
        "not b0, unless the chunk separately asserts that proposition as fact. "
        "Box parent links must be acyclic. "
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
        "explicitly supports an identity, alias, or coreference link. When an identity belongs inside a subordinate "
        "DRS context, put that context's declared box id in the hypothesis box_id; use box_id '' only for the root "
        "asserted DRS. Use temporal_records only for explicit "
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


def _chunk_drs_validation_feedback_text(context_budget: dict[str, Any] | None) -> str:
    feedback = (context_budget or {}).get("validation_feedback")
    if not isinstance(feedback, dict) or not feedback:
        return ""
    return (
        "Validation retry: the previous model-produced DRS failed KMD post-validation. "
        "Produce a fresh corrected model-owned DRS structure, not a patch, and preserve only source-grounded "
        "semantics that the chunk supports. Do not repair by renaming ids in a way that changes referents or "
        "box scope. Satisfy every listed validator error using proper DRT structure. Diagnostics: "
        + json.dumps(feedback, ensure_ascii=False)
        + " "
    )


def _chunk_drs_validation_feedback_payload(validation: dict[str, Any], *, stage: str) -> dict[str, Any]:
    errors = [str(error) for error in validation.get("errors") or []][:50]
    grounding_failures = [str(error) for error in validation.get("grounding_failures") or []][:50]
    corrections: list[str] = []
    for error in errors:
        if error == "duplicate_or_missing_referent_id":
            corrections.append("Every discourse referent must have one unique id r0, r1, ...; do not reuse a referent id for two discourse referents, and if a referent cannot be uniquely declared, omit it rather than duplicating an id.")
        elif error == "duplicate_or_missing_box_id":
            corrections.append("Every DRS box must have one unique id b0, b1, ...; do not reuse a box id for two scoped contexts.")
        elif error == "duplicate_or_missing_condition_id":
            corrections.append("Every condition must have one unique id c0, c1, ...; do not reuse a condition id.")
        elif error == "missing_box" or error == "missing_root_box":
            corrections.append("Declare exactly one root asserted box b0 with parent_id ''.")
        elif error.startswith("self_argument_box:"):
            corrections.append("A condition must not use target_kind=box with target_id equal to its own box_id; use a distinct subordinate content box, a declared condition target, or a literal argument.")
        elif error.startswith("missing_argument_box:"):
            corrections.append("If an argument targets a DRS box, that box id must be declared in the boxes array with grounded evidence and proper parent scope.")
        elif error.startswith("missing_argument_referent:"):
            corrections.append("If an argument targets a discourse referent, that referent id must be declared in the referents array with grounded evidence.")
        elif error.startswith("literal_argument_has_target_id:"):
            corrections.append("Literal and unknown arguments must use target_id ''.")
    return {
        "stage": stage,
        "errors": errors,
        "grounding_failures": grounding_failures,
        "required_corrections": list(dict.fromkeys(corrections))[:20],
        "validation_retry_policy": "model-corrects-drs-topology-no-deterministic-id-rewrite-v1",
    }


def build_chunk_drs_skeleton_prompt(chunk_text: str, *, rel_path: str = "", context_budget: dict[str, Any] | None = None) -> str:
    source_span_candidates = (context_budget or {}).get("source_span_candidates")
    span_candidate_text = (
        "When source_span_candidates are provided, each evidence_text must be exactly one listed source span or ''. "
        if isinstance(source_span_candidates, list) and source_span_candidates
        else ""
    )
    validation_feedback_text = _chunk_drs_validation_feedback_text(context_budget)
    return (
        validation_feedback_text
        + "JSON only. Stage 1 of source-grounded DRS extraction. Extract only declared discourse referents "
        "DRS boxes, and explicit temporal records from the chunk. Do not emit conditions, identity hypotheses, answers, "
        "outside knowledge, or handler names. Declare exactly one root asserted box with id b0 and parent_id ''; "
        "that root is only the containing discourse, not permission to flatten embedded scoped propositions into "
        "asserted fact. Use "
        "stable referent ids r0, r1, ...; box ids b0, b1, ...; and temporal ids t0, t1, ... in order. "
        "IDs are single-use: never emit two referents with the same id, two boxes with the same id, or two temporal records with the same id. If uncertain, emit fewer declared items rather than repeating an id. Use "
        "subordinate boxes only for scoped DRS contexts such as reports, quotes, beliefs, negation, conditionals, "
        "uncertainty, dreams, fiction, and modality; subordinate boxes must be distinct from the containing box "
        "and parent links must be acyclic. Conditions for embedded propositions inside those contexts must use "
        "the subordinate box_id, not b0, unless the chunk separately asserts that proposition as fact. "
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
    validation_feedback_text = _chunk_drs_validation_feedback_text(context_budget)
    return (
        validation_feedback_text
        + "JSON only. Stage 2 of source-grounded DRS extraction. Emit conditions using only the declared "
        "referent, box, and temporal ids. Do not invent ids; target_id is schema-constrained to declared ids or ''. "
        "Use stable condition ids c0, c1, c2, ... in order. IDs are single-use: never emit two conditions with the same id, and use only declared referent, box, and temporal ids from the stage input. "
        "If an argument is a literal phrase rather than a declared id, set target_id to '' and put the exact "
        "phrase in value and/or evidence_text. Do not emit identity hypotheses or temporal records in this stage. "
        "When a declared temporal record scopes a condition, set that condition's temporal_id to the declared "
        "temporal id; otherwise temporal_id must be ''. "
        "If a condition is an embedded proposition inside a declared non-asserted box, use that subordinate "
        "box_id; do not flatten reported, quoted, believed, negated, hypothetical, fictional, dreamed, uncertain, "
        "conditional, or modal content into b0 unless the chunk separately asserts it as fact. "
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

    referent_id_values = [str(item.get("id") or "") for item in referents]
    box_id_values = [str(item.get("id") or "") for item in boxes]
    condition_id_values = [str(item.get("id") or "") for item in conditions]
    temporal_id_values = [str(item.get("id") or "") for item in temporals]
    referent_ids = {value for value in referent_id_values if value}
    box_ids = {value for value in box_id_values if value}
    condition_ids = {value for value in condition_id_values if value}
    temporal_ids = {value for value in temporal_id_values if value}
    if len(referent_ids) != len([value for value in referent_id_values if value]):
        errors.append("duplicate_or_missing_referent_id")
    if len(box_ids) != len([value for value in box_id_values if value]):
        errors.append("duplicate_or_missing_box_id")
    if len(condition_ids) != len([value for value in condition_id_values if value]):
        errors.append("duplicate_or_missing_condition_id")
    if len(temporal_ids) != len([value for value in temporal_id_values if value]):
        errors.append("duplicate_or_missing_temporal_id")

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
    errors.extend(box_root_errors(boxes))
    errors.extend(box_parent_cycle_errors(boxes))
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
    errors.extend(condition_argument_cycle_errors(conditions))
    for item in identities:
        left_id = str(item.get("left_referent_id") or "")
        right_id = str(item.get("right_referent_id") or "")
        if left_id not in referent_ids:
            errors.append(f"missing_identity_left:{left_id}")
        if right_id not in referent_ids:
            errors.append(f"missing_identity_right:{right_id}")
        box_id = str(item.get("box_id") or "")
        if box_id and box_id not in box_ids:
            errors.append(f"missing_identity_box:{left_id}:{right_id}->{box_id}")
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


def _normalize_chunk_drs_shape(payload: Any) -> Any:
    if not isinstance(payload, dict) or not isinstance(payload.get("drs"), dict):
        return payload
    drs = {**payload["drs"]}
    changed = False
    # These list-valued fields are structural JSON shape, not semantic decisions.
    # Missing auxiliary lists mean empty collections; non-list core collections remain invalid.
    for field_name in ("identity_hypotheses", "temporal_records", "evidence_spans", "semantic_notes"):
        value = drs.get(field_name)
        if value is None:
            drs[field_name] = []
            changed = True
        elif isinstance(value, list):
            if field_name in {"identity_hypotheses", "temporal_records"}:
                repaired = [item for item in value if isinstance(item, dict)]
            else:
                repaired = [str(item) for item in value if str(item or "").strip()]
            if repaired != value:
                drs[field_name] = repaired
                changed = True
    return {**payload, "drs": drs} if changed else payload


def _repair_chunk_drs_payload(payload: Any, source_text: str = "", *, prune_unreferenced_temporals: bool = True) -> Any:
    payload = _normalize_chunk_drs_shape(payload)
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
            box_evidence = str(item.get("evidence_text") or "").strip()
            if box_evidence and box_evidence not in source_text:
                item["evidence_text"] = ""
                grounding_repaired = True
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
    if cached is not None and not _cached_structured_failure_retryable(cached):
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
        payload = _with_model_input_audits(payload, exc)
        _write_cache(path, payload)
        return (
            payload,
            elapsed,
            {"prompt_hash": prompt_hash, **constraint},
        )
    except Exception as exc:
        payload = _with_model_input_audits(
            {"accepted": False, "reason": "request_failed", "error": str(exc), "raw_text": ""},
            exc,
        )
        return (
            payload,
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
        return _with_model_input_audits(
            {
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
            },
            completion,
        )
    new_boxes = completion_payload.get("boxes")
    new_boxes = [item for item in new_boxes if isinstance(item, dict)] if isinstance(new_boxes, list) else []
    allowed_missing = set(missing_box_ids)
    new_boxes = [item for item in new_boxes if str(item.get("id") or "") in allowed_missing]
    if not new_boxes:
        return _with_model_input_audits(
            {
                "accepted": False,
                "reason": "empty_box_completion",
                "stage": "box_completion",
                "raw_text": str(completion.get("_model_raw") or "") if isinstance(completion, dict) else "",
                "elapsed": elapsed,
                "box_completion_n_predict": box_n_predict,
                **constraint,
            },
            completion,
        )
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
        return _with_model_input_audits(
            {
                "accepted": False,
                "reason": reason,
                "stage": "box_completion",
                "raw_text": str(completion.get("_model_raw") or "") if isinstance(completion, dict) else "",
                "elapsed": elapsed,
                "validation": merged_validation,
                "box_completion_n_predict": box_n_predict,
                **constraint,
            },
            completion,
        )
    raw = json.dumps(
        {
            "candidate": json.dumps(drs, sort_keys=True),
            "box_completion": completion.get("_model_raw") if isinstance(completion, dict) else "",
        },
        sort_keys=True,
    )
    return _with_model_input_audits(
        {
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
        },
        payload,
        completion,
    )


def _call_model_chunk_drs_staged(
    prompt_chunk: str,
    client: LocalModelClient,
    *,
    rel_path: str,
    n_predict: int,
    context_budget: dict[str, Any],
    cache_path: Path | None,
    validation_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if validation_feedback is not None:
        context_budget = {
            **context_budget,
            "validation_feedback": validation_feedback,
            "validation_retry_policy": "model-corrects-drs-topology-no-deterministic-id-rewrite-v1",
        }
    condition_n_predict = default_staged_chunk_drs_condition_n_predict(
        n_predict,
        prompt_chunk,
        context_budget.get("max_evidence_chars"),
    )
    max_items = context_budget.get("max_array_items") or chunk_drs_array_max_items(n_predict)
    source_span_candidates = chunk_drs_source_span_candidates(
        prompt_chunk,
        context_budget.get("max_evidence_chars"),
    )
    skeleton_n_predict = default_staged_chunk_drs_skeleton_n_predict(
        n_predict,
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
        return _with_model_input_audits(
            {
                "accepted": False,
                "reason": str(skeleton.get("reason") or "schema_validation_failed")
                if isinstance(skeleton, dict)
                else "schema_validation_failed",
                "stage": "skeleton",
                "error": str(skeleton.get("error") or "") if isinstance(skeleton, dict) else "",
                "raw_snippet": str(skeleton.get("raw_snippet") or "") if isinstance(skeleton, dict) else "",
                "raw_text": str(skeleton.get("raw_text") or skeleton.get("_model_raw") or "")
                if isinstance(skeleton, dict)
                else "",
                "elapsed": skeleton_elapsed,
                "fresh_or_cached": str(skeleton.get("fresh_or_cached") or "fresh")
                if isinstance(skeleton, dict)
                else "fresh",
                **skeleton_constraint,
            },
            skeleton,
        )
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
        return _with_model_input_audits(
            {
                "accepted": False,
                "reason": "grounding_validation_failed",
                "stage": "skeleton",
                "grounding_failures": skeleton_span_failures[:50],
                "elapsed": skeleton_elapsed,
                **skeleton_constraint,
            },
            skeleton,
        )
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
        return _with_model_input_audits(
            {
                "accepted": False,
                "reason": str(condition_stage.get("reason") or "schema_validation_failed")
                if isinstance(condition_stage, dict)
                else "schema_validation_failed",
                "stage": "conditions",
                "error": str(condition_stage.get("error") or "") if isinstance(condition_stage, dict) else "",
                "raw_snippet": str(condition_stage.get("raw_snippet") or "") if isinstance(condition_stage, dict) else "",
                "raw_text": str(condition_stage.get("raw_text") or condition_stage.get("_model_raw") or "")
                if isinstance(condition_stage, dict)
                else "",
                "elapsed": skeleton_elapsed + condition_elapsed,
                "fresh_or_cached": str(condition_stage.get("fresh_or_cached") or "fresh")
                if isinstance(condition_stage, dict)
                else "fresh",
                **condition_constraint,
            },
            skeleton,
            condition_stage,
        )
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
            return _with_model_input_audits(
                {
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
                        "dynamic_skeleton_budget_policy": CHUNK_DRS_DYNAMIC_SKELETON_BUDGET_POLICY,
                        "dynamic_condition_budget_policy": CHUNK_DRS_DYNAMIC_CONDITION_BUDGET_POLICY,
                        "staged_skeleton_n_predict": skeleton_n_predict,
                        "staged_condition_n_predict": condition_n_predict,
                        "box_completion_n_predict": box_completion["context_budget"]["box_completion_n_predict"],
                    },
                    "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
                    "fresh_or_cached": "fresh",
                    "staged": True,
                    "box_completion": box_completion.get("box_completion"),
                },
                skeleton,
                condition_stage,
                box_completion,
            )
        reason = "grounding_validation_failed" if validation.get("grounding_failure_count") else "schema_validation_failed"
        validation_retry_summary: dict[str, Any] | None = None
        if validation_feedback is None and reason == "schema_validation_failed":
            retry_feedback = _chunk_drs_validation_feedback_payload(validation, stage="staged_merge")
            retry = _call_model_chunk_drs_staged(
                prompt_chunk,
                client,
                rel_path=rel_path,
                n_predict=n_predict,
                context_budget=context_budget,
                cache_path=cache_path,
                validation_feedback=retry_feedback,
            )
            validation_retry_summary = _staged_fallback_failure_summary(retry)
            validation_retry_summary["feedback"] = retry_feedback
            if retry.get("accepted"):
                raw = json.dumps(
                    {
                        "initial_skeleton": skeleton.get("_model_raw") if isinstance(skeleton, dict) else "",
                        "initial_conditions": condition_stage.get("_model_raw") if isinstance(condition_stage, dict) else "",
                        "validation_retry": retry.get("raw_text") or "",
                    },
                    sort_keys=True,
                )
                return _with_model_input_audits(
                    {
                        **retry,
                        "raw_text": raw,
                        "elapsed": elapsed + float(retry.get("elapsed") or 0.0),
                        "fallback_from_reason": reason,
                        "validation_retry": {
                            "accepted": True,
                            "feedback": retry_feedback,
                            "prompt_hash": retry.get("prompt_hash"),
                        },
                        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
                    },
                    skeleton,
                    condition_stage,
                    retry,
                )
        payload = {
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
        if validation_retry_summary is not None:
            payload["validation_retry"] = validation_retry_summary
        return _with_model_input_audits(payload, skeleton, condition_stage, box_completion)
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
    return _with_model_input_audits(
        {
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
                "dynamic_skeleton_budget_policy": CHUNK_DRS_DYNAMIC_SKELETON_BUDGET_POLICY,
                "dynamic_condition_budget_policy": CHUNK_DRS_DYNAMIC_CONDITION_BUDGET_POLICY,
                "staged_skeleton_n_predict": skeleton_n_predict,
                "staged_condition_n_predict": condition_n_predict,
                "box_completion_n_predict": default_chunk_drs_box_completion_n_predict(n_predict),
            },
            "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
            "fresh_or_cached": "fresh",
            "staged": True,
        },
        skeleton,
        condition_stage,
    )


def chunk_drs_cache_context(
    client: LocalModelClient | None,
    *,
    n_predict: int | None = None,
    rel_path: str = "",
    chunk_text: str = "",
) -> dict[str, Any]:
    if n_predict is None:
        n_predict = default_chunk_drs_n_predict(client, chunk_text)
    production_schema = chunk_drs_json_schema(include_auxiliary_fields=False)
    constraint = _constraint_settings(CHUNK_DRS_GRAMMAR, production_schema, CHUNK_DRS_SCHEMA_VERSION)
    context = {
        "prompt_version": PROMPT_VERSION,
        "schema_version": CHUNK_DRS_SCHEMA_VERSION,
        "evidence_cap_policy": "min_chunk_or_reserved_output_ratio",
        "array_cap_policy": "reserved_output_tokens_ratio",
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
        "structured_record_route_policy": CHUNK_DRS_STRUCTURED_RECORD_ROUTE_POLICY,
        "staged_retry_diagnostics_policy": CHUNK_DRS_STAGED_RETRY_DIAGNOSTICS_POLICY,
        "stage_failure_cache_policy": CHUNK_DRS_STAGE_FAILURE_CACHE_POLICY,
        "dynamic_skeleton_budget_policy": CHUNK_DRS_DYNAMIC_SKELETON_BUDGET_POLICY,
        "dynamic_condition_budget_policy": CHUNK_DRS_DYNAMIC_CONDITION_BUDGET_POLICY,
        "dynamic_output_budget_policy": CHUNK_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY,
        "staged_first_policy": CHUNK_DRS_STAGED_FIRST_POLICY,
        "compact_first": _compact_chunk_drs_enabled() and _compact_chunk_drs_eligible(chunk_text),
        "compact_fact_policy": CHUNK_DRS_COMPACT_FACT_POLICY,
        "compact_n_predict": default_compact_chunk_drs_n_predict(chunk_text),
        "staged_skeleton_n_predict": default_staged_chunk_drs_skeleton_n_predict(int(n_predict)),
        "staged_condition_n_predict": default_staged_chunk_drs_condition_n_predict(int(n_predict)),
        "box_completion_n_predict": default_chunk_drs_box_completion_n_predict(int(n_predict)),
        **constraint,
        "n_predict": int(n_predict),
        "model_fingerprint": _client_fingerprint(client),
    }
    if rel_path:
        context["source_rel_path"] = rel_path
    if chunk_text:
        context["source_text_hash"] = hashlib.sha256(str(chunk_text or "").encode("utf-8", errors="replace")).hexdigest()
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
        context["context_budget"] = {
            **context_budget,
            "source_span_policy": CHUNK_DRS_SOURCE_SPAN_POLICY,
            "source_span_candidate_count": len(source_span_candidates),
            "skeleton_source_span_policy": CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY,
        }
    return context




def call_model_chunk_drs(
    chunk_text: str,
    client: LocalModelClient,
    *,
    rel_path: str = "",
    n_predict: int | None = None,
    refresh_empty_compact_legacy: bool = False,
) -> dict[str, Any]:
    if _compact_live_model_path_allowed(client) and _compact_chunk_drs_enabled() and _compact_chunk_drs_eligible(chunk_text):
        compact = call_model_chunk_drs_compact(
            chunk_text,
            client,
            rel_path=rel_path,
            refresh_empty_legacy=refresh_empty_compact_legacy,
        )
        if compact.get("accepted") or compact.get("reason") == "request_failed":
            return compact
    if n_predict is None:
        n_predict = default_chunk_drs_n_predict(client, chunk_text)
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
        "structured_record_route_policy": CHUNK_DRS_STRUCTURED_RECORD_ROUTE_POLICY,
        "staged_retry_diagnostics_policy": CHUNK_DRS_STAGED_RETRY_DIAGNOSTICS_POLICY,
        "stage_failure_cache_policy": CHUNK_DRS_STAGE_FAILURE_CACHE_POLICY,
        "dynamic_skeleton_budget_policy": CHUNK_DRS_DYNAMIC_SKELETON_BUDGET_POLICY,
        "dynamic_condition_budget_policy": CHUNK_DRS_DYNAMIC_CONDITION_BUDGET_POLICY,
        "dynamic_output_budget_policy": CHUNK_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY,
        "staged_first_policy": CHUNK_DRS_STAGED_FIRST_POLICY,
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
    cache_settings = {
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
        "structured_record_route_policy": CHUNK_DRS_STRUCTURED_RECORD_ROUTE_POLICY,
        "staged_retry_diagnostics_policy": CHUNK_DRS_STAGED_RETRY_DIAGNOSTICS_POLICY,
        "stage_failure_cache_policy": CHUNK_DRS_STAGE_FAILURE_CACHE_POLICY,
        "dynamic_skeleton_budget_policy": CHUNK_DRS_DYNAMIC_SKELETON_BUDGET_POLICY,
        "dynamic_condition_budget_policy": CHUNK_DRS_DYNAMIC_CONDITION_BUDGET_POLICY,
        "dynamic_output_budget_policy": CHUNK_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY,
        "staged_first_policy": CHUNK_DRS_STAGED_FIRST_POLICY,
        "source_span_candidate_count": len(source_span_candidates),
        "staged_skeleton_n_predict": default_staged_chunk_drs_skeleton_n_predict(
            int(n_predict),
            prompt_chunk,
            context_budget.get("max_evidence_chars"),
        ),
        "staged_condition_n_predict": default_staged_chunk_drs_condition_n_predict(
            int(n_predict),
            prompt_chunk,
            context_budget.get("max_evidence_chars"),
        ),
        "box_completion_n_predict": default_chunk_drs_box_completion_n_predict(int(n_predict)),
    }
    cache_context = {
        **cache_settings,
        "prompt_version": PROMPT_VERSION,
        "model_fingerprint": _client_fingerprint(client),
    }
    if rel_path:
        cache_context["source_rel_path"] = rel_path
    prompt_hash = _cache_hash(
        "chunk_drs",
        prompt,
        client,
        cache_settings,
    )
    cache_path = _cache_path("KMD_CHUNK_DRS_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and not _cached_structured_failure_retryable(cached):
        cached.setdefault("cache_context", cache_context)
        return cached
    staged_first_reason = _chunk_drs_staged_first_reason(prompt_chunk, context_budget)
    staged_first_summary: dict[str, Any] | None = None
    if _staged_chunk_drs_enabled() and staged_first_reason:
        fallback = _call_model_chunk_drs_staged(
            prompt_chunk,
            client,
            rel_path=rel_path,
            n_predict=n_predict,
            context_budget=context_budget,
            cache_path=cache_path,
        )
        if fallback.get("accepted"):
            payload = {
                **fallback,
                "fallback_from_reason": staged_first_reason,
                "monolithic_prompt_hash": prompt_hash,
                "staged_first": True,
            }
            payload.setdefault("cache_context", cache_context)
            payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
            _write_cache(cache_path, payload)
            return payload
        staged_first_summary = _staged_fallback_failure_summary(fallback)
        staged_first_summary.update({"fallback_from_reason": staged_first_reason, "staged_first": True})
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
            "cache_context": cache_context,
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
                payload.setdefault("cache_context", cache_context)
                payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
                _write_cache(cache_path, payload)
                return payload
            payload["staged_fallback"] = _staged_fallback_failure_summary(fallback)
        if staged_first_summary:
            payload["staged_first"] = staged_first_summary
        payload = _with_model_input_audits(payload, exc)
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
            "context_budget": context_budget,
            "cache_context": cache_context,
        }
        return _with_model_input_audits(payload, exc)
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
            "cache_context": cache_context,
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
                payload.setdefault("cache_context", cache_context)
                payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
            payload.setdefault("cache_context", cache_context)
            payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
            _write_cache(cache_path, payload)
            return payload
        payload["box_completion"] = {
            "accepted": False,
            "reason": box_completion.get("reason"),
            "stage": box_completion.get("stage"),
        }
        if staged_first_summary:
            payload["staged_first"] = staged_first_summary
        payload = _with_model_input_audits(payload, parsed, box_completion)
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
            "cache_context": cache_context,
        }
        payload = _with_model_input_audits(payload, parsed)
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
            payload.setdefault("cache_context", cache_context)
            payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
        "cache_context": cache_context,
        "validation": validation,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    if staged_retry_summary:
        payload["staged_retry"] = staged_retry_summary
    if staged_first_summary:
        payload["staged_first"] = staged_first_summary
    payload = _with_model_input_audits(payload, parsed)
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
    cache_settings = {"n_predict": n_predict, "schema": ANSWER_SCHEMA_VERSION, **constraint}
    cache_context = {
        **cache_settings,
        "model_fingerprint": _client_fingerprint(client),
        "expected_answer_type": str(query_frame.get("answer_type") or "unknown"),
        "evidence_count": len(evidence_items),
        "discourse_frame_count": len(discourse_frames),
    }
    prompt_hash = _cache_hash(
        "answer_verification",
        prompt,
        client,
        cache_settings,
    )
    cache_path = _cache_path("KMD_VERIFIER_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and not _cached_request_failed(cached):
        cached.setdefault("cache_context", cache_context)
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
    except LocalModelJSONError as exc:
        return {
            "accepted": False,
            "reason": "invalid_json",
            "error": str(exc),
            "raw_text": exc.raw_text,
            "snippet": exc.snippet,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "cache_context": cache_context,
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
            "cache_context": cache_context,
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
        "cache_context": cache_context,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
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
        "condition, change a scoped proposition into an asserted one, or use outside knowledge. When evidence "
        "contains explicit speaker/source identity for quoted, reported, or message content, the canonical answer "
        "should normalize first-person or second-person deictic wording into a public source-resolved answer when "
        "the question asks what the source says/reports/believes rather than asking for exact wording. Use the "
        "referent surface requested by the question when that surface is grounded, and keep the same scoped "
        "proposition. If the candidate is "
        "not a complete answer binding but instead states that no binding is available, return answer='unknown' and "
        "copy the evidence_span that supports that absence. evidence_span must "
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
    cache_settings = {"n_predict": n_predict, "schema": ANSWER_SCHEMA_VERSION, **constraint}
    cache_context = {
        **cache_settings,
        "model_fingerprint": _client_fingerprint(client),
        "answer_type": answer_type,
        "evidence_count": len(evidence_items),
    }
    prompt_hash = _cache_hash(
        "answer_canonicalization",
        prompt,
        client,
        cache_settings,
    )
    cache_path = _cache_path("KMD_ANSWER_CANONICALIZATION_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and cached.get("reason") not in {
        "request_failed",
        "ungrounded_answer",
        "schema_validation_failed",
        "invalid_json",
    }:
        cached.setdefault("cache_context", cache_context)
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
    except LocalModelJSONError as exc:
        return {
            "accepted": False,
            "reason": "invalid_json",
            "error": str(exc),
            "raw_text": exc.raw_text,
            "snippet": exc.snippet,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "cache_context": cache_context,
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
            "cache_context": cache_context,
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
            "cache_context": cache_context,
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
    if normalize(answer) == "unknown":
        answer_grounded = bool(span_grounded)
    else:
        answer_grounded = any(answer in str(item.get("text") or "") for item in evidence_items)
        if (
            not answer_grounded
            and span_grounded
            and answer_type in {"content_phrase", "metadata_value", "state"}
            and _answer_terms_grounded_by_evidence(answer, evidence_items)
        ):
            answer_grounded = True
    if not span_grounded and not answer_grounded:
        return {
            "accepted": False,
            "reason": "ungrounded_answer",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "cache_context": cache_context,
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
        "cache_context": cache_context,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
    _write_cache(cache_path, payload)
    return payload


def build_source_resolved_answer_prompt(
    question: str,
    candidate_answer: str,
    answer_type: str,
    evidence_items: list[dict[str, str]],
) -> str:
    return (
        "JSON only. Convert a quoted/reported/message-content candidate into a public reported answer for the "
        "question. This task is not quote extraction. If evidence gives a source identity, answer as an outside "
        "narrator reporting what that source said/reported/believed; do not keep the quote's grammatical "
        "perspective. For questions phrased with past reporting auxiliaries such as did/was/were, the public "
        "reported answer must use past-tense reported wording for the main finite predicate when fluent, even "
        "when the quoted content itself uses present-tense wording. Preserve future and relative time words "
        "separately; tense alignment must not delete or alter those time words. Use a shorter "
        "source surface from the question when it is grounded inside the evidence identity. Preserve proposition, "
        "polarity, modality, object names, file paths, URLs, IDs, and negation. If exact words or a quote are "
        "requested, keep exact wording. If no grounded public reported answer is entailed, return the candidate "
        "unchanged. evidence_span must be copied exactly from one evidence item. Return exactly "
        "{\"source_resolved_answer\":{\"answer\":\"\",\"evidence_span\":\"\",\"reason\":\"\"}}."
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


def _source_resolution_cache_path(prompt_hash: str) -> Path | None:
    cache_dir = os.environ.get("KMD_SOURCE_RESOLUTION_CACHE_DIR", "").strip()
    if not cache_dir:
        cache_dir = os.environ.get("KMD_ANSWER_CANONICALIZATION_CACHE_DIR", "").strip()
    if cache_dir:
        return Path(cache_dir) / f"{prompt_hash}.json"
    return _cache_path("KMD_SOURCE_RESOLUTION_CACHE_DIR", prompt_hash)


def call_model_source_resolved_answer(
    question: str,
    candidate_answer: str,
    answer_type: str,
    evidence_items: list[dict[str, str]],
    client: LocalModelClient,
    *,
    n_predict: int | None = None,
) -> dict[str, Any]:
    if n_predict is None:
        n_predict = int(os.environ.get("KMD_SOURCE_RESOLUTION_N_PREDICT", "160"))
    prompt = build_source_resolved_answer_prompt(question, candidate_answer, answer_type, evidence_items)
    constraint = _constraint_settings(SOURCE_RESOLVED_ANSWER_GRAMMAR, None, ANSWER_SCHEMA_VERSION)
    grammar_hash = str(constraint["grammar_hash"])
    cache_settings = {"n_predict": n_predict, "schema": ANSWER_SCHEMA_VERSION, **constraint}
    cache_context = {
        **cache_settings,
        "model_fingerprint": _client_fingerprint(client),
        "answer_type": answer_type,
        "evidence_count": len(evidence_items),
    }
    prompt_hash = _cache_hash(
        "source_resolved_answer",
        prompt,
        client,
        cache_settings,
    )
    cache_path = _source_resolution_cache_path(prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None and cached.get("reason") not in {
        "request_failed",
        "ungrounded_answer",
        "schema_validation_failed",
        "invalid_json",
    }:
        cached.setdefault("cache_context", cache_context)
        return cached
    start = time.time()
    try:
        parsed = _complete_structured(
            client,
            prompt,
            n_predict=n_predict,
            grammar=SOURCE_RESOLVED_ANSWER_GRAMMAR,
            json_schema=None,
        )
    except LocalModelJSONError as exc:
        return {
            "accepted": False,
            "reason": "invalid_json",
            "error": str(exc),
            "raw_text": exc.raw_text,
            "snippet": exc.snippet,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    result = parsed.get("source_resolved_answer") if isinstance(parsed, dict) else None
    if result is None and isinstance(parsed, dict) and "answer" in parsed:
        result = parsed
    if not isinstance(result, dict):
        return {
            "accepted": False,
            "reason": "invalid_json",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "cache_context": cache_context,
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
            "cache_context": cache_context,
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
    if normalize(answer) == "unknown":
        answer_grounded = bool(span_grounded)
    else:
        answer_grounded = any(answer in str(item.get("text") or "") for item in evidence_items)
        if (
            not answer_grounded
            and span_grounded
            and answer_type in {"content_phrase", "metadata_value", "state"}
            and _answer_terms_grounded_by_evidence(answer, evidence_items)
        ):
            answer_grounded = True
    if not span_grounded and not answer_grounded:
        return {
            "accepted": False,
            "reason": "ungrounded_answer",
            "raw_text": raw,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "cache_context": cache_context,
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
        "cache_context": cache_context,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
    _write_cache(cache_path, payload)
    return payload


def _answer_terms_grounded_by_evidence(answer: str, evidence_items: list[dict[str, str]]) -> bool:
    answer_terms = [term for term in content_tokens(answer) if term]
    if not answer_terms:
        return False
    evidence_terms: set[str] = set()
    for item in evidence_items:
        for term in content_tokens(str(item.get("text") or "")):
            evidence_terms.update(term_variants(term))
    if not evidence_terms:
        return False
    for term in answer_terms:
        if not (term_variants(term) & evidence_terms):
            return False
    return True


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
    cache_settings = {"n_predict": n_predict, "schema": ANSWER_SCHEMA_VERSION, **constraint}
    cache_context = {
        **cache_settings,
        "model_fingerprint": _client_fingerprint(client),
        "fuller_candidate_count": len(fuller_candidates),
        "evidence_count": len(evidence_items),
    }
    prompt_hash = _cache_hash(
        "identity_canonicalization",
        prompt,
        client,
        cache_settings,
    )
    cache_path = _cache_path("KMD_IDENTITY_CACHE_DIR", prompt_hash)
    cached = _read_cache(cache_path)
    if cached is not None:
        if not (
            cached.get("accepted") is False
            and cached.get("reason") in {"request_failed", "invalid_json", "schema_validation_failed"}
        ):
            cached.setdefault("cache_context", cache_context)
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
    except LocalModelJSONError as exc:
        return {
            "accepted": False,
            "reason": "invalid_json",
            "error": str(exc),
            "raw_text": exc.raw_text,
            "snippet": exc.snippet,
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "request_failed",
            "error": str(exc),
            "prompt_hash": prompt_hash,
            "grammar_hash": grammar_hash,
            "cache_context": cache_context,
            "elapsed": round(time.time() - start, 3),
        }
    raw = str(parsed.get("_model_raw") or "") if isinstance(parsed, dict) else ""
    result = parsed.get("canonicalization") if isinstance(parsed, dict) else None
    if result is None and isinstance(parsed, dict) and any(key in parsed for key in ["same_referent", "answer"]):
        result = parsed
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
            "cache_context": cache_context,
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
            "cache_context": cache_context,
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
            "cache_context": cache_context,
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
        "cache_context": cache_context,
        "output_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "fresh_or_cached": "fresh",
    }
    payload = _with_model_input_audits(payload, locals().get("parsed"), locals().get("exc"), locals().get("repaired"), locals().get("fallback"), locals().get("box_completion"))
    _write_cache(cache_path, payload)
    return payload
