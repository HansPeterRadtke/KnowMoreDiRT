"""Universal surface-structure extraction for raw text.

This module intentionally avoids semantic event or role interpretation.  It only
turns broadly universal document structures into grounded records: key/value
text, JSON/object-like scalar values, delimited table cells, identifiers, URLs,
and timestamps.  Relation labels and keys are data copied from the source; they
never select bespoke answer handlers.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .extractors import identifiers, urls
from .query import term_variants
from .text import clean_extracted_value, normalize

LABEL_VALUE_RE = re.compile(r'\s*"?([A-Za-z][A-Za-z0-9 _/-]{1,80})"?\s*[:=]\s*"?([^"{}\[\]\n;,|]+)"?')
JSON_SCALAR_PAIR_RE = re.compile(
    r'"([^"\n]{1,80})"\s*:\s*(?:"([^"\n]*)"|(-?\d+(?:\.\d+)?)|(true|false|null))',
    re.I,
)
LOOSE_SCALAR_PAIR_RE = re.compile(
    r'\b"?([A-Za-z][A-Za-z0-9 _/-]{0,80})"?\s*:\s*(?:"([^"\n{}\[\]]*)"|([^,{}\[\]\n]+))',
    re.I,
)
TIMESTAMP_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2})?)\b")


@dataclass(frozen=True)
class ExtractedRelation:
    relation_type: str
    predicate: str
    subject: str = ""
    object: str = ""
    value: str = ""
    confidence: float = 0.75
    metadata: dict[str, object] = field(default_factory=dict)


def normalize_predicate(value: str) -> str:
    token = normalize(value).split()[0] if normalize(value).split() else normalize(value)
    variants = sorted(term_variants(token), key=len)
    return variants[0] if variants else token


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
    predicate = normalize_predicate(predicate) or normalize(predicate)
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
    relations: list[ExtractedRelation] = []
    relations.extend(_extract_record_values(value))
    relations.extend(_extract_label_values(value))
    relations.extend(_extract_table_row_relations(value))
    for url in urls(value):
        _append(relations, "identifier", "url", value=url, confidence=0.85)
    for identifier in identifiers(value):
        _append(relations, "identifier", "identifier", value=identifier, confidence=0.8)
    for match in TIMESTAMP_RE.finditer(value):
        _append(relations, "temporal", "timestamp", value=match.group(1), confidence=0.8)
    return _dedupe(relations)


def _extract_label_values(text: str) -> list[ExtractedRelation]:
    relations: list[ExtractedRelation] = []
    pieces = re.split(r"\s*(?:[|;,]|\n)\s*", text)
    for piece in pieces:
        if any(marker in piece for marker in "{}[]") or "://" in piece:
            continue
        match = LABEL_VALUE_RE.match(piece)
        if not match:
            without_leading_time = re.sub(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2})?\s+", "", piece)
            match = LABEL_VALUE_RE.match(without_leading_time)
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
    object_ranges: list[tuple[int, int]] = []
    for object_match in re.finditer(r"\{[^{}\n]*(?:\{[^{}\n]*\}[^{}\n]*)*\}", text):
        object_text = object_match.group(0)
        if ":" not in object_text:
            continue
        object_ranges.append((object_match.start(), object_match.end()))
        record_group = "object:" + hashlib.sha256(normalize(object_text).encode("utf-8")).hexdigest()[:16]
        for pair in LOOSE_SCALAR_PAIR_RE.finditer(object_text):
            key = clean_extracted_value(pair.group(1))
            value = clean_extracted_value(next(group for group in pair.groups()[1:] if group is not None))
            if not key or not value or "{" in value:
                continue
            _append(
                relations,
                "record_value",
                "key_value",
                subject=key,
                value=value,
                confidence=0.84,
                record_path=key,
                record_group=record_group,
                surface_format="object_like",
            )
    for match in JSON_SCALAR_PAIR_RE.finditer(text):
        if any(start <= match.start() < end for start, end in object_ranges):
            continue
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
                record_group=".".join(key.split(".")[:-1]) or key,
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
    scalar = "null" if value is None else ("true" if value is True else "false" if value is False else str(value))
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
        record_group=".".join(path[:-1]) if len(path) > 1 else ".".join(path),
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
    for index, cell in enumerate(cells[1:], start=1):
        _append(relations, "table_cell", f"cell_{index}", subject=first, value=cell, confidence=0.72, cell_index=index)
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
