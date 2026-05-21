"""Generic source-grounded relation extraction for the DSPG vertical slice.

The extractors in this module deliberately model common discourse shapes
rather than domains. They convert raw text snippets into normalized relation
records that the query layer can use without knowing whether the source was a
school note, forum post, legal-style note, lab page, or project document.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

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

ACTIVE_EVENT_RE = re.compile(
    rf"\b({PERSON_PATTERN})\s+({'|'.join(sorted(map(re.escape, GENERIC_VERBS), key=len, reverse=True))})\b([^.;\n]*)"
)
PASSIVE_EVENT_RE = re.compile(
    rf"([^.;\n]{{2,120}}?)\s+(?:was\s+)?({'|'.join(sorted(map(re.escape, PASSIVE_VERBS), key=len, reverse=True))})\s+by\s+({PERSON_PATTERN})",
    re.I,
)
MEANING_RE = re.compile(r"([^.;:\n]+?)\s+means\s+([^.;\n]+)", re.I)
PLURAL_RE = re.compile(r"plural\s+of\s+([^.;\n]+?)\s+is\s+([^.;\n]+)", re.I)
ALIAS_RE = re.compile(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+is\s+also\s+called\s+([^.;\n]+)")
NEGATIVE_BUY_RE = re.compile(r"\bbought\s+(.+?)\s+but\s+not\s+([^.;\n]+)", re.I)
PRACTICE_SCALE_RE = re.compile(r"\bpracticed\s+the\s+(.+?)\s+scale\b", re.I)
CONFIRMED_FIX_RE = re.compile(r"confirmed\s+fix\s*[:=]\s*(.+)", re.I)
TIMESTAMP_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\b")
ROLE_LABEL_RE = re.compile(
    rf"\b(?:owner|contact|researcher|vet|inspector|clinician)\b[^:;\n]{{0,30}}[:=]\s*({PERSON_PATTERN})",
    re.I,
)
LABEL_VALUE_RE = re.compile(r'\s*"?([A-Za-z][A-Za-z0-9 _/-]{1,50})"?\s*[:=]\s*"?([^"{}\[\]\n;,|]+)"?')
JSON_SCALAR_PAIR_RE = re.compile(
    r'"([^"\n]{1,80})"\s*:\s*(?:"([^"\n]*)"|(-?\d+(?:\.\d+)?)|(true|false|null))',
    re.I,
)


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

    relations.extend(_extract_record_values(value))
    relations.extend(_extract_label_values(value))
    relations.extend(_extract_table_row_relations(value))

    for url in urls(value):
        _append(relations, "identifier", "url", value=url, confidence=0.85)
    for identifier in identifiers(value):
        _append(relations, "identifier", "identifier", value=identifier, confidence=0.8)

    for match in MEANING_RE.finditer(value):
        _append(relations, "meaning", "mean", match.group(1).split(":")[-1], value=match.group(2), confidence=0.88)
    for match in PLURAL_RE.finditer(value):
        _append(relations, "grammar", "plural", match.group(1), value=match.group(2), confidence=0.88)
    for match in ALIAS_RE.finditer(value):
        _append(relations, "identity", "alias", match.group(1), value=match.group(2), confidence=0.82)
    for match in NEGATIVE_BUY_RE.finditer(value):
        _append(relations, "event", "not_buy", object_=match.group(2), value=match.group(2), confidence=0.82)
    for match in PRACTICE_SCALE_RE.finditer(value):
        _append(relations, "event_detail", "practice_scale", value=match.group(1), confidence=0.82)
    for match in CONFIRMED_FIX_RE.finditer(value):
        _append(relations, "status", "confirmed_fix", value=match.group(1), confidence=0.86)
    for match in TIMESTAMP_RE.finditer(value):
        _append(relations, "temporal", "timestamp", value=match.group(1), confidence=0.8)

    for trigger, predicate in STATUS_TRIGGERS.items():
        if trigger in lowered:
            _append(relations, "status", predicate, value=value, confidence=0.75, trigger=trigger)

    for match in ACTIVE_EVENT_RE.finditer(value):
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

    for match in PASSIVE_EVENT_RE.finditer(value):
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

    for match in ROLE_LABEL_RE.finditer(value):
        label = value[max(0, match.start() - 40): match.start()].split("|")[-1]
        _append(relations, "label_value", "label", subject=label, value=match.group(1), confidence=0.82)

    return _dedupe(relations)


def _extract_label_values(text: str) -> list[ExtractedRelation]:
    relations: list[ExtractedRelation] = []
    pieces = re.split(r"\s*(?:[|;,]|\n)\s*", text)
    for piece in pieces:
        if any(marker in piece for marker in "{}[]"):
            continue
        match = LABEL_VALUE_RE.match(piece)
        if not match:
            continue
        label = clean_extracted_value(match.group(1))
        value = clean_extracted_value(match.group(2))
        if label and value:
            _append(relations, "label_value", "label", subject=label, value=value, confidence=0.84)
    return relations


def _extract_record_values(text: str) -> list[ExtractedRelation]:
    relations: list[ExtractedRelation] = []
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
        except Exception:
            parsed = None
        if parsed is not None:
            _walk_record_value(parsed, (), relations)
    for match in JSON_SCALAR_PAIR_RE.finditer(text):
        key = clean_extracted_value(match.group(1))
        value = clean_extracted_value(next(group for group in match.groups()[1:] if group is not None))
        if key and value:
            _append(
                relations,
                "record_value",
                "key_value",
                subject=key,
                value=value,
                confidence=0.83,
                record_path=key,
                surface_format="json_like",
            )
    return relations


def _walk_record_value(value: Any, path: tuple[str, ...], relations: list[ExtractedRelation]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_record_value(item, (*path, str(key)), relations)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_record_value(item, (*path, str(index)), relations)
        return
    if value is None:
        scalar = "null"
    elif isinstance(value, bool):
        scalar = "true" if value else "false"
    else:
        scalar = str(value)
    if not path or scalar == "":
        return
    _append(
        relations,
        "record_value",
        "key_value",
        subject=".".join(path),
        value=scalar,
        confidence=0.86,
        record_path=".".join(path),
        surface_format="json",
    )


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
