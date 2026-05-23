"""Generic query-frame construction for DSPG retrieval.

The core does not route questions through content-specific intent names.  It
turns a question into a relation-agnostic frame: visible anchors, requested
relation text, constraints, answer type, temporal/aggregation flags, and
evidence requirements.  The terms remain data; they do not select bespoke code
paths.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .answer_types import infer_expected_answer
from .extractors import capitalized_phrases, identifiers, urls
from .text import content_tokens, normalize, tokenize


QUESTION_WORDS = {
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "did",
    "does",
    "do",
    "is",
    "are",
    "was",
    "were",
    "can",
    "could",
    "should",
    "has",
    "have",
    "find",
    "provide",
    "show",
    "return",
    "give",
    "the",
    "a",
    "an",
    "for",
    "of",
    "to",
    "in",
    "on",
    "at",
    "about",
    "according",
}

ANCHOR_SKIP = {
    "Who",
    "What",
    "Which",
    "Where",
    "When",
    "How",
    "Can",
    "Could",
    "Did",
    "Does",
    "Do",
    "Is",
    "Are",
    "Was",
    "Were",
    "Find",
    "Return",
    "Show",
    "Give",
    "ID",
    "IDs",
    "URL",
    "URLs",
    "JSON",
}

GENERIC_NOUNS = {
    "answer",
    "content",
    "document",
    "entity",
    "fact",
    "field",
    "folder",
    "item",
    "name",
    "note",
    "object",
    "record",
    "source",
    "text",
    "thing",
    "value",
}


@dataclass(frozen=True)
class QueryFrame:
    """A relation-agnostic internal representation of a question."""

    question_text: str
    answer_type: str
    target_anchors: tuple[str, ...]
    requested_relation: str
    relation_terms: tuple[str, ...]
    constraints: tuple[str, ...]
    temporal_scope: str = ""
    negated: bool = False
    aggregation: str = ""
    requires_evidence: bool = True
    source: str = "deterministic"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def term_variants(term: str) -> set[str]:
    """Return small morphology-only variants without semantic labels."""

    token = normalize(term)
    if not token:
        return set()
    variants = {token}
    for suffix in ("ing", "ied", "ed", "ers", "er", "ors", "or", "ies", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            stem = token[: -len(suffix)]
            if suffix == "ies":
                stem = f"{stem}y"
            variants.add(stem)
    if len(token) >= 3:
        variants.add(f"{token}er")
        variants.add(f"{token}or")
    return {value for value in variants if len(value) > 1}


def expand_terms(terms: list[str] | tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for term in terms:
        for variant in term_variants(term):
            if variant not in values:
                values.append(variant)
    return values


def visible_anchors(text: str) -> list[str]:
    values: list[str] = []
    values.extend(urls(text))
    values.extend(identifiers(text))
    for phrase in capitalized_phrases(text):
        first = phrase.split()[0]
        if first not in ANCHOR_SKIP and phrase not in ANCHOR_SKIP and phrase not in values:
            values.append(phrase)
    return list(dict.fromkeys(value for value in values if value))


def _question_relation_terms(question: str) -> list[str]:
    qnorm = normalize(question)
    anchors = [normalize(anchor) for anchor in visible_anchors(question)]
    tokens = [
        token
        for token in tokenize(qnorm)
        if token not in QUESTION_WORDS
        and token not in GENERIC_NOUNS
        and len(token) > 1
        and not any(token in anchor for anchor in anchors)
    ]
    terms: list[str] = []
    for token in tokens:
        for variant in term_variants(token):
            if variant not in terms:
                terms.append(variant)
    return terms


def _requested_relation(question: str, relation_terms: list[str]) -> str:
    if not relation_terms:
        return ""
    tokens = tokenize(question)
    selected = [token for token in tokens if normalize(token) in set(relation_terms)]
    return " ".join(selected[:8]) or " ".join(relation_terms[:8])


def plan_question(question: str) -> QueryFrame:
    qnorm = normalize(question)
    anchors = tuple(visible_anchors(question))
    relation_terms = _question_relation_terms(question)
    expected = infer_expected_answer(question)
    temporal_scope = ""
    if any(term in qnorm for term in ("current", "latest", "final")):
        temporal_scope = "latest"
    elif any(term in qnorm for term in ("earliest", "first", "initial")):
        temporal_scope = "earliest"
    aggregation = "count" if expected.answer_type == "count" else ""
    constraints = tuple(
        term
        for term in relation_terms
        if term not in {normalize(anchor) for anchor in anchors}
    )
    return QueryFrame(
        question_text=question,
        answer_type=expected.answer_type,
        target_anchors=anchors,
        requested_relation=_requested_relation(question, relation_terms),
        relation_terms=tuple(relation_terms),
        constraints=constraints,
        temporal_scope=temporal_scope,
        negated=bool(re.search(r"\b(?:not|no|never|without|denied|unsupported)\b", qnorm)),
        aggregation=aggregation,
        requires_evidence=True,
    )


def frame_from_mapping(question: str, mapping: dict[str, Any] | None, *, source: str = "model") -> QueryFrame:
    """Normalize a model/dict frame into the internal dataclass."""

    base = plan_question(question)
    if not mapping:
        return base
    raw = mapping.get("query_frame") if "query_frame" in mapping and isinstance(mapping.get("query_frame"), dict) else mapping
    if not isinstance(raw, dict):
        return base
    anchors = raw.get("target_anchors")
    if isinstance(anchors, str):
        anchor_tuple = tuple(value.strip() for value in anchors.split(";") if value.strip())
    elif isinstance(anchors, list):
        anchor_tuple = tuple(str(value).strip() for value in anchors if str(value).strip())
    else:
        anchor_tuple = base.target_anchors
    relation_terms_raw = raw.get("relation_terms")
    if isinstance(relation_terms_raw, list):
        relation_terms = tuple(expand_terms([str(value) for value in relation_terms_raw if str(value).strip()]))
    else:
        relation_terms = base.relation_terms
    constraints_raw = raw.get("constraints")
    if isinstance(constraints_raw, list):
        constraints = tuple(expand_terms([str(value) for value in constraints_raw if str(value).strip()]))
    else:
        constraints = base.constraints
    answer_type = str(raw.get("answer_type") or base.answer_type)
    if answer_type not in {
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
    }:
        answer_type = base.answer_type
    return QueryFrame(
        question_text=question,
        answer_type=answer_type,
        target_anchors=anchor_tuple or base.target_anchors,
        requested_relation=str(raw.get("requested_relation") or base.requested_relation),
        relation_terms=relation_terms or base.relation_terms,
        constraints=constraints or base.constraints,
        temporal_scope=str(raw.get("temporal_scope") or base.temporal_scope),
        negated=bool(raw.get("negated", base.negated)),
        aggregation=str(raw.get("aggregation") or base.aggregation),
        requires_evidence=bool(raw.get("requires_evidence", True)),
        source=source,
    )
