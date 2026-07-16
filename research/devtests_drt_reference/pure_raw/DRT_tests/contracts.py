#!/usr/bin/env python3
from __future__ import annotations

DSPG_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "chunks",
        "mentions",
        "referents",
        "contexts",
        "propositions",
        "relations",
        "identity_hypotheses",
    ],
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "source_name", "text", "order", "token_estimate", "attributes"],
                "properties": {
                    "id": {"type": "string"},
                    "source_name": {"type": "string"},
                    "text": {"type": "string"},
                    "order": {"type": "integer"},
                    "token_estimate": {"type": "integer"},
                    "attributes": {"type": "object"},
                },
            },
        },
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "text", "mention_type", "chunk_id", "sentence_index", "order", "attributes"],
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "mention_type": {"type": "string"},
                    "chunk_id": {"type": "string"},
                    "sentence_index": {"type": "integer"},
                    "order": {"type": "integer"},
                    "attributes": {"type": "object"},
                },
            },
        },
        "referents": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "type", "label", "aliases", "attributes"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "label": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "attributes": {"type": "object"},
                },
            },
        },
        "contexts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "kind", "parent_id", "holder", "attributes"],
                "properties": {
                    "id": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "asserted",
                            "negated",
                            "conditional_antecedent",
                            "conditional_consequent",
                            "believed",
                            "reported",
                            "quoted",
                            "dreamed",
                            "possible",
                            "uncertain",
                            "hypothetical",
                        ],
                    },
                    "parent_id": {"type": ["string", "null"]},
                    "holder": {"type": ["string", "null"]},
                    "attributes": {"type": "object"},
                },
            },
        },
        "propositions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "predicate", "surface", "context_id", "confidence", "attributes"],
                "properties": {
                    "id": {"type": "string"},
                    "predicate": {"type": "string"},
                    "surface": {"type": "string"},
                    "context_id": {"type": "string"},
                    "confidence": {"type": "number"},
                    "attributes": {"type": "object"},
                },
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source", "target", "type", "confidence"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"},
                    "confidence": {"type": "number"},
                    "context_id": {"type": ["string", "null"]},
                    "attributes": {"type": "object"},
                },
            },
        },
        "identity_hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "mention_id", "referent_id", "confidence", "status", "reason", "rank"],
                "properties": {
                    "id": {"type": "string"},
                    "mention_id": {"type": "string"},
                    "referent_id": {"type": "string"},
                    "confidence": {"type": "number"},
                    "status": {"type": "string", "enum": ["accepted", "candidate", "rejected", "ambiguous"]},
                    "reason": {"type": "string"},
                    "rank": {"type": "integer"},
                    "attributes": {"type": "object"},
                },
            },
        },
    },
}


DSPG_TOPLEVEL_JSON_GRAMMAR = r'''
root ::= "{" ws "\"chunks\"" ws ":" ws array ws "," ws "\"mentions\"" ws ":" ws array ws "," ws "\"referents\"" ws ":" ws array ws "," ws "\"contexts\"" ws ":" ws array ws "," ws "\"propositions\"" ws ":" ws array ws "," ws "\"relations\"" ws ":" ws array ws "," ws "\"identity_hypotheses\"" ws ":" ws array ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
object ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
pair ::= string ws ":" ws value
value ::= object | array | string | number | boolean | "null"
boolean ::= "true" | "false"
string ::= "\"" chars "\""
chars ::= [a-zA-Z0-9_ .,:;!?/#@+=<>()-]*
number ::= "-"? int frac?
int ::= "0" | [1-9] [0-9]*
frac ::= "." [0-9]+
ws ::= [ \t\n\r]*
'''
