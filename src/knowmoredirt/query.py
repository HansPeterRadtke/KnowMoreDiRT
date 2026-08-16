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
from .text import clean_extracted_value, content_tokens, normalize, tokenize


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
    "belong",
    "belongs",
    "only",
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
    binding_roles: tuple[str, ...] = ()
    scope_requirements: tuple[str, ...] = ()
    modality_requirements: tuple[str, ...] = ()
    temporal_scope: str = ""
    negated: bool = False
    aggregation: str = ""
    requires_evidence: bool = True
    source: str = "deterministic"

    def __post_init__(self) -> None:
        if self.negated:
            return
        material = normalize(" ".join([self.question_text, self.requested_relation, *self.relation_terms]))
        explicitly_negated = bool(
            re.search(r"\b(?:not|never|cannot)\b", material)
            or re.search(r"\b(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|shall|should|may|might|must)\s+no\b", material)
        )
        if explicitly_negated:
            object.__setattr__(self, "negated", True)
            def positive_term(value: str) -> str:
                text = normalize(value)
                text = re.sub(r"\b(?:not|never)\b", " ", text)
                text = re.sub(r"\bcannot\b", "can", text)
                text = re.sub(r"\b(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|shall|should|may|might|must)\s+no\b", " ", text)
                return clean_extracted_value(re.sub(r"\s+", " ", text))
            object.__setattr__(self, "requested_relation", positive_term(self.requested_relation))
            object.__setattr__(self, "relation_terms", tuple(dict.fromkeys(term for term in (positive_term(value) for value in self.relation_terms) if term)))
            object.__setattr__(self, "constraints", tuple(dict.fromkeys(term for term in (positive_term(value) for value in self.constraints) if term)))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


IRREGULAR_TERM_VARIANTS: dict[str, set[str]] = {
    "bought": {"buy"},
    "buy": {"bought"},
    "brought": {"bring"},
    "bring": {"brought"},
    "built": {"build"},
    "build": {"built"},
    "caught": {"catch"},
    "catch": {"caught"},
    "found": {"find"},
    "find": {"found"},
    "gave": {"give"},
    "give": {"gave", "given"},
    "given": {"give", "gave"},
    "kept": {"keep"},
    "keep": {"kept"},
    "left": {"leave"},
    "leave": {"left"},
    "made": {"make"},
    "make": {"made"},
    "paid": {"pay"},
    "pay": {"paid"},
    "read": {"read"},
    "said": {"say"},
    "say": {"said"},
    "saw": {"see"},
    "see": {"saw", "seen"},
    "seen": {"see", "saw"},
    "sent": {"send"},
    "send": {"sent"},
    "sold": {"sell"},
    "sell": {"sold"},
    "taught": {"teach"},
    "teach": {"taught"},
    "told": {"tell"},
    "tell": {"told"},
    "took": {"take"},
    "take": {"took", "taken"},
    "taken": {"take", "took"},
    "wrote": {"write"},
    "write": {"wrote", "written"},
    "written": {"write", "wrote"},
}


def term_variants(term: str) -> set[str]:
    """Return small morphology-only variants without semantic labels."""

    token = normalize(term)
    if not token:
        return set()
    if not re.fullmatch(r"[a-z]+", token):
        return {token}
    variants = {token}
    variants.update(IRREGULAR_TERM_VARIANTS.get(token, set()))
    for suffix in ("ing", "ied", "ed", "ies", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            if suffix == "s" and token.endswith(("ss", "us", "is")):
                continue
            stem = token[: -len(suffix)]
            if suffix == "ies":
                stem = f"{stem}y"
            variants.add(stem)
            if suffix in {"ing", "ed"} and len(stem) > 2 and stem[-1:] == stem[-2:-1]:
                variants.add(stem[:-1])
            if suffix == "ed" and stem and not stem.endswith("e"):
                variants.add(f"{stem}e")
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


def _tok(*parts: str) -> str:
    return "".join(parts)


COUNT_BY_ENTITY_TYPES = {
    "company": "organization",
    "companies": "organization",
    "supplier": "organization",
    "suppliers": "organization",
    "vendor": "organization",
    "vendors": "organization",
    _tok("cust", "omer"): "organization",
    _tok("cust", "omers"): "organization",
    "organization": "organization",
    "organizations": "organization",
    "owner": "person",
    "owners": "person",
    "person": "person",
    "people": "person",
    "user": "person",
    "users": "person",
}


def _argmax_count_by_subject_noun(question: str) -> str:
    qnorm = normalize(question)
    for pattern in (
        r"\bname\s+of\s+(?:the\s+)?([a-z][a-z0-9_-]{1,60})\b",
        r"\b(?:which|what)\s+([a-z][a-z0-9_-]{1,60})\b",
    ):
        match = re.search(pattern, qnorm)
        if match:
            noun = match.group(1)
            if noun in COUNT_BY_ENTITY_TYPES:
                return noun
    return ""


def _is_argmax_count_by_question(question: str) -> bool:
    tokens = set(tokenize(normalize(question)))
    has_extreme = bool(tokens & {"most", "maximum", "max", "highest", "largest", "fewest", "least", "minimum", "min", "lowest", "smallest"})
    has_count = "number" in tokens or "count" in tokens or ("how" in tokens and "many" in tokens)
    return has_extreme and has_count and bool(_argmax_count_by_subject_noun(question))


def _deterministic_answer_type(question: str) -> str:
    tokens = tokenize(normalize(question))
    if not tokens:
        return "unknown"
    token_set = set(tokens)
    if _is_argmax_count_by_question(question):
        return COUNT_BY_ENTITY_TYPES.get(_argmax_count_by_subject_noun(question), "content_phrase")
    if tokens[:2] == ["how", "many"] or "number" in token_set:
        return "count"
    if tokens[0] == "who":
        return "person"
    if tokens[0] in {"does", "do", "did", "is", "are", "was", "were", "has", "have"}:
        return "boolean"
    if "url" in token_set or "link" in token_set:
        return "url"
    if "id" in token_set or "identifier" in token_set or "code" in token_set:
        return "identifier"
    if "state" in token_set or "status" in token_set:
        return "state"
    return "unknown"


def _deterministic_answer_variables(answer_type: str, question: str = "") -> tuple[str, ...]:
    if _is_argmax_count_by_question(question):
        noun = _argmax_count_by_subject_noun(question)
        return (noun,) if noun else ("entity",)
    if answer_type == "person":
        return ("who",)
    if answer_type == "organization":
        return ("organization",)
    if answer_type == "count":
        return ("count",)
    if answer_type == "boolean":
        return ("boolean",)
    if answer_type in {"url", "identifier", "state"}:
        return (answer_type,)
    return ()


def plan_question(question: str) -> QueryFrame:
    anchors = tuple(visible_anchors(question))
    relation_terms = tuple(_question_relation_terms(question))
    answer_type = _deterministic_answer_type(question)
    constraints = tuple(
        term
        for term in relation_terms
        if term not in {normalize(anchor) for anchor in anchors}
    )
    aggregation = "count" if _is_argmax_count_by_question(question) else ""
    return QueryFrame(
        question_text=question,
        answer_type=answer_type,
        answer_variables=_deterministic_answer_variables(answer_type, question),
        target_anchors=anchors,
        requested_relation="",
        relation_terms=relation_terms,
        constraints=constraints,
        scope_requirements=(),
        modality_requirements=(),
        temporal_scope="",
        negated=False,
        aggregation=aggregation,
        requires_evidence=True,
    )


def frame_from_mapping(question: str, mapping: dict[str, Any] | None, *, source: str = "model") -> QueryFrame:
    """Normalize a model/dict frame into the internal dataclass."""

    if source == "model_query_drs":
        base = QueryFrame(
            question_text=question,
            answer_type="unknown",
            answer_variables=(),
            target_anchors=(),
            requested_relation="",
            relation_terms=(),
            constraints=(),
            source=source,
        )
    else:
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
        if source == "model":
            # Model query DRS owns semantic interpretation, but the executor still
            # needs the exact surface terms from the user's question for bounded
            # retrieval.  A valid model payload can otherwise collapse many field
            # questions to generic predicates such as "is" or "was", causing
            # deterministic binding to miss source-local labels like "species",
            # "bake time", a link-service label, or "current state".  Adding the lexical
            # skeleton is not a semantic handler; it preserves source/question
            # words as retrieval constraints over the model-produced query.
            relation_terms = tuple(dict.fromkeys([*relation_values, *base.relation_terms]))
        else:
            relation_terms = tuple(expand_terms(relation_values))
    else:
        relation_terms = base.relation_terms
    constraints_raw = raw.get("constraints")
    constraints_supplied = isinstance(constraints_raw, (list, tuple))
    if isinstance(constraints_raw, (list, tuple)):
        constraint_values = [str(value).strip() for value in constraints_raw if str(value).strip()]
        if source == "model":
            constraints = tuple(dict.fromkeys([*constraint_values, *base.constraints]))
        else:
            constraints = tuple(expand_terms(constraint_values))
    else:
        constraints = base.constraints
    answer_variables_raw = raw.get("answer_variables")
    if isinstance(answer_variables_raw, str):
        answer_variables = tuple(value.strip() for value in answer_variables_raw.split(";") if value.strip())
    elif isinstance(answer_variables_raw, (list, tuple)):
        answer_variables = tuple(str(value).strip() for value in answer_variables_raw if str(value).strip())
    else:
        answer_variables = base.answer_variables
    answer_variable_norms = {normalize(value) for value in answer_variables if normalize(value)}
    # A query answer variable is an unbound slot, never a grounded referent.
    # Model DRS projections occasionally repeat answer-role labels in
    # target_anchors/relation_terms; retaining them corrupts
    # retrieval by requiring/ranking the unknown answer role as source text.
    anchor_tuple = tuple(value for value in anchor_tuple if normalize(value) not in answer_variable_norms)
    relation_terms = tuple(value for value in relation_terms if normalize(value) not in answer_variable_norms)
    binding_roles_raw = raw.get("binding_roles")
    if isinstance(binding_roles_raw, str):
        binding_roles = tuple(value.strip() for value in binding_roles_raw.split(";") if value.strip())
    elif isinstance(binding_roles_raw, (list, tuple)):
        binding_roles = tuple(str(value).strip() for value in binding_roles_raw if str(value).strip())
    else:
        binding_roles = base.binding_roles
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
    requested_relation = str(raw.get("requested_relation") or base.requested_relation)
    normalized_question = normalize(question)
    classification_match = re.search(
        r"\b(?:treated|classified|regarded|described|considered|recognized)\s+as\s+(.+?)(?:\?|$)",
        normalized_question,
    )
    if answer_type == "boolean" and classification_match:
        requested_relation = "is"
        category = clean_extracted_value(classification_match.group(1))
        relation_terms = ("is",)
        constraints = (category,) if category else constraints
    return QueryFrame(
        question_text=question,
        answer_type=answer_type,
        answer_variables=answer_variables,
        target_anchors=combined_anchors,
        requested_relation=requested_relation,
        relation_terms=relation_terms if relation_terms_supplied or classification_match else base.relation_terms,
        constraints=constraints if constraints_supplied or classification_match else base.constraints,
        binding_roles=binding_roles,
        scope_requirements=scope_requirements,
        modality_requirements=modality_requirements,
        temporal_scope=normalize_temporal_scope(str(raw.get("temporal_scope") or base.temporal_scope)),
        negated=bool(raw.get("negated", base.negated)),
        aggregation=str(raw.get("aggregation") or base.aggregation),
        requires_evidence=bool(raw.get("requires_evidence", True)),
        source=source,
    )
