"""Portable closed JSON Schemas for every model-owned semantic stage."""
from __future__ import annotations

from typing import Any


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _arr(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def _bounded_arr(items: dict[str, Any], maximum: int) -> dict[str, Any]:
    return {"type": "array", "items": items, "maxItems": maximum}


STRING = {"type": "string"}
BOOL = {"type": "boolean"}
INTEGER = {"type": "integer"}
NUMBER = {"type": "number"}
ANSWER_SHAPE = {
    "type": "string",
    "enum": ["text", "list", "number", "boolean", "url", "identifier", "date_time"],
}


def dataset_profile_schema(fingerprint: str) -> dict[str, Any]:
    collection = _obj(
        {
            "collection_path": STRING,
            "purpose": STRING,
            "record_granularity": {
                "type": "string",
                "enum": ["document", "record", "event", "message", "row", "line", "unknown"],
            },
            "identity_fields": _arr(STRING),
            "temporal_fields": _arr(STRING),
            "text_fields": _arr(STRING),
            "extraction_notes": STRING,
        }
    )
    collections = _arr(collection)
    collections["maxItems"] = 8
    return _obj(
        {
            "dataset_profile": _obj(
                {
                    "fingerprint": {"type": "string", "enum": [fingerprint]},
                    "summary": STRING,
                    "collections": collections,
                    "general_notes": STRING,
                }
            )
        }
    )


def semantic_contract_schema(question: str, contract_id: str) -> dict[str, Any]:
    return _obj(
        {
            "semantic_contract": _obj(
                {
                    "contract_id": {"type": "string", "enum": [contract_id]},
                    "question": {"type": "string", "enum": [question]},
                    "intent_summary": STRING,
                    "answer_shape": ANSWER_SHAPE,
                    "answer_slot": STRING,
                    "semantic_kind": {
                        "type": "string",
                        "enum": [
                            "entity_attribute",
                            "event_fact",
                            "source_classification",
                            "reported_content",
                            "definition",
                            "calculation",
                            "unknown",
                        ],
                    },
                    "world_scope": {
                        "type": "string",
                        "enum": [
                            "asserted_world",
                            "reported_content",
                            "nonactual_internal",
                            "nonactual_external_effect",
                            "source_metadata",
                            "unknown",
                        ],
                    },
                    "source_scope": {
                        "type": "string",
                        "enum": ["any", "non_cache", "cache_only", "semantic_only", "unknown"],
                    },
                    "authority_mode": {
                        "type": "string",
                        "enum": ["any", "explicit_official", "unknown"],
                    },
                    "target_phrases": _bounded_arr(STRING, 8),
                    "scope_phrases": _bounded_arr(STRING, 8),
                    "relation_phrases": _bounded_arr(STRING, 8),
                    "constraint_phrases": _bounded_arr(STRING, 8),
                    "polarity": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral", "unknown"],
                    },
                    "temporal_mode": {
                        "type": "string",
                        "enum": [
                            "none",
                            "current",
                            "latest",
                            "earliest",
                            "final",
                            "before",
                            "after",
                            "at_time",
                        ],
                    },
                    "epistemic_mode": {
                        "type": "string",
                        "enum": [
                            "asserted",
                            "reported",
                            "alleged",
                            "fictional",
                            "dream",
                            "hypothetical",
                            "quoted",
                            "unknown",
                        ],
                    },
                    "reporting_tense": {
                        "type": "string",
                        "enum": ["none", "present", "past", "unknown"],
                    },
                    "requires_explicit_evidence": BOOL,
                    "compound_request": BOOL,
                }
            )
        }
    )


def query_program_schema(contract_id: str) -> dict[str, Any]:
    filter_item = _obj(
        {
            "field_path": STRING,
            "operator": {
                "type": "string",
                "enum": [
                    "equals", "not_equals", "contains", "contains_all", "contains_any",
                    "exists", "in", "less_than", "less_or_equal", "greater_than", "greater_or_equal",
                ],
            },
            "value": STRING,
            "values": _bounded_arr(STRING, 8),
        }
    )
    argument = _obj(
        {
            "name": {
                "type": "string",
                "enum": [
                    "mode", "left_field", "right_field", "sort_field", "direction",
                    "aggregate", "operation", "extractor", "label", "start_phrase",
                    "end_phrase", "pattern", "value_group", "time_group", "occurrence",
                    "value_kind", "strip_chars", "distinct",
                ],
            },
            "value": STRING,
            "values": _bounded_arr(STRING, 8),
            "numbers": _bounded_arr(NUMBER, 8),
        }
    )
    step = _obj(
        {
            "tool": {
                "type": "string",
                "enum": [
                    "sample_records", "search_records", "expand_source_context", "filter_records",
                    "project_values", "extract_values", "model_extract", "join_records", "union_values",
                    "intersect_values", "sort_records", "aggregate_values", "calculate",
                ],
            },
            "inputs": _bounded_arr(INTEGER, 5),
            "collection": STRING,
            "terms": _bounded_arr(STRING, 8),
            "fields": _bounded_arr(STRING, 8),
            "filters": _bounded_arr(filter_item, 6),
            "arguments": _bounded_arr(argument, 8),
            "limit": INTEGER,
        }
    )
    return _obj(
        {
            "query_program": _obj(
                {
                    "contract_id": {"type": "string", "enum": [contract_id]},
                    "steps": _bounded_arr(step, 8),
                }
            )
        }
    )


def tool_extraction_schema(contract_id: str) -> dict[str, Any]:
    return _obj(
        {
            "tool_extraction": _obj(
                {
                    "contract_id": {"type": "string", "enum": [contract_id]},
                    "status": {"type": "string", "enum": ["extracted", "unknown"]},
                    "values": _bounded_arr(STRING, 20),
                    "answer_shape": ANSWER_SHAPE,
                    "evidence_record_ids": _bounded_arr(STRING, 20),
                    "evidence_relation": {
                        "type": "string",
                        "enum": [
                            "direct_support",
                            "direct_contradiction",
                            "structured_field",
                            "derived",
                            "state_only",
                            "absence",
                            "nonactual_content",
                            "unknown",
                        ],
                    },
                    "reason": STRING,
                }
            )
        }
    )


def numeric_value_repair_schema(contract_id: str) -> dict[str, Any]:
    return _obj(
        {
            "numeric_value_repair": _obj(
                {
                    "contract_id": {"type": "string", "enum": [contract_id]},
                    "status": {"type": "string", "enum": ["extracted", "unknown"]},
                    "value": NUMBER,
                    "evidence_record_ids": _bounded_arr(STRING, 20),
                    "evidence_relation": {
                        "type": "string",
                        "enum": [
                            "direct_support",
                            "structured_field",
                            "derived",
                            "absence",
                            "unknown",
                        ],
                    },
                    "reason": STRING,
                }
            )
        }
    )


def event_fact_verdict_schema(contract_id: str) -> dict[str, Any]:
    return _obj(
        {
            "event_fact_verdict": _obj(
                {
                    "contract_id": {"type": "string", "enum": [contract_id]},
                    "verdict": {
                        "type": "string",
                        "enum": ["supports", "contradicts", "insufficient"],
                    },
                    "scope_binding": {
                        "type": "string",
                        "enum": ["direct", "title_to_body", "document_scope", "none"],
                    },
                    "evidence_basis": {
                        "type": "string",
                        "enum": [
                            "explicit_support",
                            "explicit_denial",
                            "authoritative_not_proven",
                            "impossibility",
                            "state_only",
                            "absence_only",
                            "nonactual_only",
                            "mixed_or_other",
                        ],
                    },
                    "evidence_record_ids": _bounded_arr(STRING, 20),
                    "authority_label": STRING,
                    "decisive_predicate": STRING,
                    "correction_clause": STRING,
                    "reason": STRING,
                }
            )
        }
    )


def grounded_answer_schema(contract_id: str) -> dict[str, Any]:
    return _obj(
        {
            "grounded_answer": _obj(
                {
                    "contract_id": {"type": "string", "enum": [contract_id]},
                    "status": {"type": "string", "enum": ["answered", "unknown"]},
                    "answer": STRING,
                    "answer_shape": ANSWER_SHAPE,
                    "evidence_record_ids": _arr(STRING),
                    "derivation": {
                        "type": "string",
                        "enum": ["direct", "extraction", "arithmetic", "comparison", "summary", "unknown"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": STRING,
                }
            )
        }
    )



def evidence_review_schema(contract_id: str) -> dict[str, Any]:
    search = _obj(
        {
            "collection": STRING,
            "terms": _bounded_arr(STRING, 8),
            "mode": {"type": "string", "enum": ["all", "any", "phrase"]},
            "fields": _bounded_arr(STRING, 8),
            "limit": INTEGER,
        }
    )
    return _obj(
        {
            "evidence_review": _obj(
                {
                    "contract_id": {"type": "string", "enum": [contract_id]},
                    "status": {"type": "string", "enum": ["answered", "search", "unknown"]},
                    "answer": STRING,
                    "answer_items": _bounded_arr(STRING, 60),
                    "answer_shape": ANSWER_SHAPE,
                    "evidence_record_ids": _bounded_arr(STRING, 60),
                    "searches": _bounded_arr(search, 4),
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": STRING,
                }
            )
        }
    )

def assert_portable_closed_schema(schema: dict[str, Any]) -> None:
    def visit(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if "const" in node:
                raise ValueError(f"bare const is not portable: {path}")
            if node.get("type") == "object":
                properties = node.get("properties")
                if not isinstance(properties, dict):
                    raise ValueError(f"object has no properties: {path}")
                if node.get("additionalProperties") is not False:
                    raise ValueError(f"object is open: {path}")
                if set(node.get("required", [])) != set(properties):
                    raise ValueError(f"not every property is required: {path}")
            for key, value in node.items():
                visit(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{path}[{index}]")

    visit(schema, "$")
