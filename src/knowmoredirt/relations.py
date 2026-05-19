"""Generic source-grounded relation extraction for the DSPG vertical slice.

The extractors in this module deliberately model common discourse shapes
rather than domains. They convert raw text snippets into normalized relation
records that the query layer can use without knowing whether the source was a
school note, forum post, legal-style note, lab page, or project document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .extractors import identifiers, urls
from .text import clean_extracted_value, normalize


PERSON_PATTERN = (
    r"(?:(?:Dr\.|Ms\.|Mr\.|Mrs\.|Prof\.)\s+)?"
    r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}"
)

GENERIC_VERBS = {
    "accepted": "accept",
    "alleges": "allege",
    "alleged": "allege",
    "approved": "approve",
    "argued": "argue",
    "authored": "author",
    "believes": "believe",
    "believed": "believe",
    "bought": "buy",
    "caused": "cause",
    "closed": "close",
    "coached": "coach",
    "confirmed": "confirm",
    "deleted": "delete",
    "depends": "depend",
    "disagree": "disagree",
    "disagreed": "disagree",
    "drafted": "draft",
    "filed": "file",
    "fixed": "fix",
    "implemented": "implement",
    "implements": "implement",
    "inspected": "inspect",
    "managed": "manage",
    "manages": "manage",
    "merged": "merge",
    "observed": "observe",
    "opened": "open",
    "owned": "own",
    "owns": "own",
    "practiced": "practice",
    "recorded": "record",
    "reported": "report",
    "requested": "request",
    "reviewed": "review",
    "signed": "sign",
    "stated": "state",
    "tested": "test",
    "touched": "touch",
    "watered": "water",
    "wrote": "write",
}

PASSIVE_VERBS = {
    "approved": "approve",
    "authored": "author",
    "closed": "close",
    "drafted": "draft",
    "fixed": "fix",
    "merged": "merge",
    "opened": "open",
    "recorded": "record",
    "reported": "report",
    "reviewed": "review",
    "signed": "sign",
    "written": "write",
}

STATUS_TRIGGERS = {
    "no proof": "unproven",
    "no final decision": "no_final_decision",
    "not confirmed": "not_confirmed",
    "unsupported": "unsupported",
    "denied": "denied",
    "does not": "negated",
    "did not": "negated",
    "not ": "negated",
    "no crack": "negated",
}


@dataclass(frozen=True)
class ExtractedRelation:
    relation_type: str
    predicate: str
    subject: str = ""
    object: str = ""
    value: str = ""
    confidence: float = 0.75
    metadata: dict[str, object] = field(default_factory=dict)


def _append(
    relations: list[ExtractedRelation],
    relation_type: str,
    predicate: str,
    subject: str = "",
    object_: str = "",
    value: str = "",
    confidence: float = 0.75,
    **metadata: object,
) -> None:
    subject = clean_extracted_value(subject)
    object_ = clean_extracted_value(object_)
    value = clean_extracted_value(value)
    if subject or object_ or value:
        relations.append(
            ExtractedRelation(
                relation_type=relation_type,
                predicate=predicate,
                subject=subject,
                object=object_,
                value=value,
                confidence=confidence,
                metadata={key: item for key, item in metadata.items() if item not in (None, "")},
            )
        )


def extract_relations(text: str) -> list[ExtractedRelation]:
    value = str(text or "")
    lowered = normalize(value)
    relations: list[ExtractedRelation] = []

    relations.extend(_extract_label_values(value))
    relations.extend(_extract_table_row_relations(value))

    for url in urls(value):
        _append(relations, "identifier", "url", value=url, confidence=0.85)
    for identifier in identifiers(value):
        _append(relations, "identifier", "identifier", value=identifier, confidence=0.8)

    for match in re.finditer(r"([^.;:\n]+?)\s+means\s+([^.;\n]+)", value, re.I):
        _append(relations, "meaning", "mean", match.group(1).split(":")[-1], value=match.group(2), confidence=0.88)
    for match in re.finditer(r"plural\s+of\s+([^.;\n]+?)\s+is\s+([^.;\n]+)", value, re.I):
        _append(relations, "grammar", "plural", match.group(1), value=match.group(2), confidence=0.88)
    for match in re.finditer(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+is\s+also\s+called\s+([^.;\n]+)", value):
        _append(relations, "identity", "alias", match.group(1), value=match.group(2), confidence=0.82)
    for match in re.finditer(r"\bbought\s+(.+?)\s+but\s+not\s+([^.;\n]+)", value, re.I):
        _append(relations, "event", "not_buy", object_=match.group(2), value=match.group(2), confidence=0.82)
    for match in re.finditer(r"\bpracticed\s+the\s+(.+?)\s+scale\b", value, re.I):
        _append(relations, "event_detail", "practice_scale", value=match.group(1), confidence=0.82)
    for match in re.finditer(r"confirmed\s+fix\s*[:=]\s*(.+)", value, re.I):
        _append(relations, "status", "confirmed_fix", value=match.group(1), confidence=0.86)
    for match in re.finditer(r"\b(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\b", value):
        _append(relations, "temporal", "timestamp", value=match.group(1), confidence=0.8)

    for trigger, predicate in STATUS_TRIGGERS.items():
        if trigger in lowered:
            _append(relations, "status", predicate, value=value, confidence=0.75, trigger=trigger)

    verb_choices = "|".join(sorted(map(re.escape, GENERIC_VERBS), key=len, reverse=True))
    for match in re.finditer(rf"\b({PERSON_PATTERN})\s+({verb_choices})\b([^.;\n]*)", value):
        verb = match.group(2).lower()
        _append(
            relations,
            "event",
            GENERIC_VERBS[verb],
            subject=match.group(1),
            object_=match.group(3),
            confidence=0.76,
            surface_verb=verb,
        )

    passive_choices = "|".join(sorted(map(re.escape, PASSIVE_VERBS), key=len, reverse=True))
    for match in re.finditer(rf"([^.;\n]{{2,120}}?)\s+(?:was\s+)?({passive_choices})\s+by\s+({PERSON_PATTERN})", value, re.I):
        verb = match.group(2).lower()
        _append(
            relations,
            "event",
            PASSIVE_VERBS[verb],
            subject=match.group(3),
            object_=match.group(1),
            confidence=0.8,
            voice="passive",
            surface_verb=verb,
        )

    for match in re.finditer(rf"\b(?:owner|contact|researcher|vet|inspector|clinician)\b[^:;\n]{{0,30}}[:=]\s*({PERSON_PATTERN})", value, re.I):
        label = value[max(0, match.start() - 40): match.start()].split("|")[-1]
        _append(relations, "label_value", "label", subject=label, value=match.group(1), confidence=0.82)

    return _dedupe(relations)


def _extract_label_values(text: str) -> list[ExtractedRelation]:
    relations: list[ExtractedRelation] = []
    pieces = re.split(r"\s*(?:[|;,]|\n)\s*", text)
    for piece in pieces:
        match = re.match(r'\s*"?([A-Za-z][A-Za-z0-9 _/-]{1,50})"?\s*[:=]\s*"?([^"{}\[\]\n;,|]+)"?', piece)
        if not match:
            continue
        label = clean_extracted_value(match.group(1))
        value = clean_extracted_value(match.group(2))
        if label and value:
            _append(relations, "label_value", "label", subject=label, value=value, confidence=0.84)
    return relations


def _extract_table_row_relations(text: str) -> list[ExtractedRelation]:
    relations: list[ExtractedRelation] = []
    if "|" not in text and "\t" not in text:
        return relations
    cells = [clean_extracted_value(cell) for cell in re.split(r"[|\t]", text)]
    cells = [cell for cell in cells if cell]
    if len(cells) < 2:
        return relations
    first = cells[0]
    for cell in cells[1:]:
        _append(relations, "table_cell", "has_cell", subject=first, value=cell, confidence=0.72)
    return relations


def _dedupe(relations: list[ExtractedRelation]) -> list[ExtractedRelation]:
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[ExtractedRelation] = []
    for relation in relations:
        key = (
            relation.relation_type,
            normalize(relation.predicate),
            normalize(relation.subject),
            normalize(relation.object),
            normalize(relation.value),
        )
        if key not in seen:
            seen.add(key)
            unique.append(relation)
    return unique
