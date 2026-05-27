"""Generic query-frame containers for DSPG retrieval.

Natural-language question semantics are model-owned.  The deterministic helper
in this module builds only a lexical skeleton used for bounded retrieval when a
model query DRS is missing or being repaired: exact URLs, identifiers,
capitalized surface anchors, and content tokens.  It does not decide answer
type, negation, temporal scope, aggregation, or the requested semantic
relation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

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
    "many",
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
    "records",
    "row",
    "rows",
    "entry",
    "entries",
    "source",
    "text",
    "thing",
    "value",
    "count",
    "number",
}


@dataclass(frozen=True)
class QueryFrame:
    """A relation-agnostic internal representation of a question."""

    question_text: str
    answer_type: str
    answer_variables: tuple[str, ...]
    target_anchors: tuple[str, ...]
    requested_relation: str
    relation_terms: tuple[str, ...]
    constraints: tuple[str, ...]
    scope_requirements: tuple[str, ...] = ()
    modality_requirements: tuple[str, ...] = ()
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
    if not re.fullmatch(r"[a-z]+", token):
        return {token}
    variants = {token}
    for suffix in ("ing", "ied", "ed", "ers", "er", "ors", "or", "ies", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            stem = token[: -len(suffix)]
            if suffix == "ies":
                stem = f"{stem}y"
            variants.add(stem)
            if suffix == "s":
                variants.add(f"{stem}er")
                variants.add(f"{stem}or")
    if len(token) >= 3:
        variants.add(f"{token}er")
        variants.add(f"{token}or")
    return {value for value in variants if len(value) > 1}


def normalize_temporal_scope(value: str) -> str:
    """Normalize model-produced temporal operators into executor enums."""

    scope = normalize(value)
    aliases = {
        "current": "latest",
        "currently": "latest",
        "latest": "latest",
        "most_recent": "latest",
        "most recent": "latest",
        "recent": "latest",
        "final": "latest",
        "last": "latest",
        "earliest": "earliest",
        "oldest": "earliest",
        "first": "earliest",
        "initial": "earliest",
    }
    return aliases.get(scope, scope)


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
        if (
            first not in ANCHOR_SKIP
            and phrase not in ANCHOR_SKIP
            and not (phrase.isupper() and len(phrase) <= 5)
            and phrase not in values
        ):
            values.append(phrase)
    return list(dict.fromkeys(value for value in values if value))


def _question_relation_terms(question: str) -> list[str]:
    qnorm = re.sub(r"\baccording to\b.+", " ", normalize(question))
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
    anchors = tuple(visible_anchors(question))
    relation_terms = tuple(_question_relation_terms(question))
    constraints = tuple(
        term
        for term in relation_terms
        if term not in {normalize(anchor) for anchor in anchors}
    )
    return QueryFrame(
        question_text=question,
        answer_type="unknown",
        answer_variables=(),
        target_anchors=anchors,
        requested_relation="",
        relation_terms=relation_terms,
        constraints=constraints,
        scope_requirements=(),
        modality_requirements=(),
        temporal_scope="",
        negated=False,
        aggregation="",
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
    elif isinstance(anchors, (list, tuple)):
        anchor_tuple = tuple(str(value).strip() for value in anchors if str(value).strip())
    else:
        anchor_tuple = base.target_anchors
    relation_terms_raw = raw.get("relation_terms")
    relation_terms_supplied = isinstance(relation_terms_raw, (list, tuple))
    if isinstance(relation_terms_raw, (list, tuple)):
        relation_values = [str(value).strip() for value in relation_terms_raw if str(value).strip()]
        relation_terms = tuple(relation_values if source == "model" else expand_terms(relation_values))
    else:
        relation_terms = base.relation_terms
    constraints_raw = raw.get("constraints")
    constraints_supplied = isinstance(constraints_raw, (list, tuple))
    if isinstance(constraints_raw, (list, tuple)):
        constraint_values = [str(value).strip() for value in constraints_raw if str(value).strip()]
        constraints = tuple(constraint_values if source == "model" else expand_terms(constraint_values))
    else:
        constraints = base.constraints
    answer_variables_raw = raw.get("answer_variables")
    if isinstance(answer_variables_raw, str):
        answer_variables = tuple(value.strip() for value in answer_variables_raw.split(";") if value.strip())
    elif isinstance(answer_variables_raw, (list, tuple)):
        answer_variables = tuple(str(value).strip() for value in answer_variables_raw if str(value).strip())
    else:
        answer_variables = base.answer_variables
    scope_requirements_raw = raw.get("scope_requirements")
    if isinstance(scope_requirements_raw, (list, tuple)):
        scope_requirements = tuple(str(value).strip() for value in scope_requirements_raw if str(value).strip())
    else:
        scope_requirements = base.scope_requirements
    modality_requirements_raw = raw.get("modality_requirements")
    if isinstance(modality_requirements_raw, (list, tuple)):
        modality_requirements = tuple(str(value).strip() for value in modality_requirements_raw if str(value).strip())
    else:
        modality_requirements = base.modality_requirements
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
    combined_anchors = tuple(dict.fromkeys([*anchor_tuple, *base.target_anchors])) if anchor_tuple else base.target_anchors
    return QueryFrame(
        question_text=question,
        answer_type=answer_type,
        answer_variables=answer_variables,
        target_anchors=combined_anchors,
        requested_relation=str(raw.get("requested_relation") or base.requested_relation),
        relation_terms=relation_terms if relation_terms_supplied else base.relation_terms,
        constraints=constraints if constraints_supplied else base.constraints,
        scope_requirements=scope_requirements,
        modality_requirements=modality_requirements,
        temporal_scope=normalize_temporal_scope(str(raw.get("temporal_scope") or base.temporal_scope)),
        negated=bool(raw.get("negated", base.negated)),
        aggregation=str(raw.get("aggregation") or base.aggregation),
        requires_evidence=bool(raw.get("requires_evidence", True)),
        source=source,
    )
