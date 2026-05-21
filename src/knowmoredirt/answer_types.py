"""Generic answer type inference and validation.

The public API still returns a plain string.  Internally, KMD uses these broad
answer expectations to reject type-unsafe candidates before they leave the
grounded DSPG/query layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .extractors import identifiers, urls
from .text import clean_extracted_value, normalize


AnswerType = Literal[
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
]


@dataclass(frozen=True)
class ExpectedAnswer:
    answer_type: AnswerType
    allow_metadata_evidence: bool = False
    requires_complete_relation: bool = True


_FILE_RE = re.compile(r"(?:^|[/\\])[^/\\\s]+\.[A-Za-z0-9]{1,12}$|^[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12}$")
_DATE_TIME_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b"
)
_DURATION_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b", re.I)
_PERSON_RE = re.compile(
    r"^(?:(?:Dr\.|Ms\.|Mr\.|Mrs\.|Prof\.)\s+)?[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3}$"
)
_ORG_HINT_RE = re.compile(
    r"\b(?:association|bureau|center|centre|clinic|club|collective|committee|company|council|department|foundation|group|guild|institute|lab|laboratory|office|school|society|studio|team|union|university|workshop)\b",
    re.I,
)


def infer_expected_answer(question: str) -> ExpectedAnswer:
    q = normalize(question)
    qtokens = set(re.findall(r"[a-z0-9_-]+", q))
    metadata_terms = {
        "metadata",
        "file",
        "folder",
        "path",
        "name",
        "extension",
        "suffix",
        "size",
        "hash",
        "encoding",
        "created",
        "modified",
        "mtime",
        "ctime",
        "lines",
        "words",
    }
    asks_metadata = bool(qtokens.intersection(metadata_terms)) and any(
        term in q for term in ["file", "folder", "metadata", "path", "extension", "suffix", "size", "hash", "encoding", "line count", "word count", "created", "modified", "mtime", "ctime"]
    )
    if any(phrase in q for phrase in ["how many", "number of", "count of"]):
        return ExpectedAnswer("count")
    if re.match(r"^(did|does|do|is|are|was|were|can|could|should|has|have)\b", q):
        return ExpectedAnswer("boolean")
    if any(token in qtokens for token in ["url", "urls", "link", "links", "runbook", "manual", "guide", "endpoint", "site"]) or (
        q.startswith("where ") and any(token in qtokens for token in ["stored", "listed", "available", "published", "map"])
    ):
        return ExpectedAnswer("url")
    if any(token in qtokens for token in ["path", "paths"]) or any(phrase in q for phrase in ["which file", "what file"]):
        return ExpectedAnswer("file_path", allow_metadata_evidence=asks_metadata)
    if q.startswith("when ") or any(token in qtokens for token in ["date", "time", "timestamp", "created", "modified", "effective", "validity", "measured"]):
        return ExpectedAnswer("date_time", allow_metadata_evidence=asks_metadata)
    if any(phrase in q for phrase in ["current state", "final state", "latest state"]) or "status" in qtokens or "state" in qtokens:
        return ExpectedAnswer("state")
    if asks_metadata:
        return ExpectedAnswer("metadata_value", allow_metadata_evidence=True)
    if q.startswith("who ") or " which person" in q or " actor" in qtokens:
        return ExpectedAnswer("person")
    if any(phrase in q for phrase in ["which organization", "what organization", "which group", "what group", "which team", "what team"]):
        return ExpectedAnswer("organization")
    if any(token in qtokens for token in ["identifier", "identifiers", "reference", "references", "id", "ids", "code", "hash", "commit", "invoice", "parcel", "case", "specimen", "sample", "order"]) or (
        q.startswith(("what ", "which ")) and any(token in qtokens for token in ["raw", "json", "record"])
    ) or (
        q.startswith("which ") and any(token in qtokens for token in ["implements", "implemented", "fixed", "touches", "touched", "appears", "named"])
    ):
        return ExpectedAnswer("identifier", allow_metadata_evidence=asks_metadata)
    return ExpectedAnswer("content_phrase")


def answer_parts(value: str) -> list[str]:
    text = clean_extracted_value(value)
    if not text:
        return []
    parts = [clean_extracted_value(part) for part in re.split(r"\s*;\s*", text)]
    return [part for part in parts if part]


def classify_value(value: str) -> AnswerType:
    text = clean_extracted_value(value)
    low = normalize(text)
    if not text or low == "unknown":
        return "unknown"
    if re.match(r"^(yes|no)\b", low):
        return "boolean"
    if urls(text) and urls(text)[0].rstrip(".,;)") == text.rstrip(".,;)"):
        return "url"
    if _FILE_RE.search(text) and not urls(text):
        return "file_path"
    if re.fullmatch(r"\d+", text):
        return "count"
    if _DURATION_RE.search(text) and _DURATION_RE.search(text).group(0).strip() == text.strip():
        return "date_time"
    if _DATE_TIME_RE.search(text) and _DATE_TIME_RE.search(text).group(0).strip() == text.strip():
        return "date_time"
    extracted_ids = identifiers(text)
    if extracted_ids and any(item.rstrip(".,;)") == text.rstrip(".,;)") for item in extracted_ids):
        return "identifier"
    if _PERSON_RE.fullmatch(text):
        return "person"
    if _ORG_HINT_RE.search(text):
        return "organization"
    return "content_phrase"


def is_metadata_evidence_text(text: str) -> bool:
    low = normalize(text)
    return low.startswith("metadata ") or low.startswith("context file_")


def is_value_compatible(expected: ExpectedAnswer, value: str) -> bool:
    value_type = classify_value(value)
    if value_type == "unknown":
        return False
    expected_type = expected.answer_type
    if expected_type == "unknown":
        return False
    if expected_type in {"person", "actor"}:
        return value_type in {"person", "content_phrase"} and not _is_structural_reference(value)
    if expected_type == "organization":
        return value_type in {"organization", "content_phrase"} and not _is_structural_reference(value)
    if expected_type == "content_phrase":
        return value_type not in {"url", "file_path", "identifier"}
    if expected_type == "state":
        return value_type not in {"url", "file_path"}
    if expected_type == "metadata_value":
        return True
    return value_type == expected_type


def compatible_answer_parts(expected: ExpectedAnswer, value: str) -> list[str]:
    return [part for part in answer_parts(value) if is_value_compatible(expected, part)]


def canonicalize_answer(expected: ExpectedAnswer, value: str) -> str:
    if expected.answer_type == "boolean":
        text = str(value or "").strip()
        return text if is_value_compatible(expected, text) else ""
    if expected.answer_type in {"person", "actor", "organization", "state", "content_phrase", "metadata_value"}:
        text = str(value or "").strip().strip('"')
        return text if is_value_compatible(expected, text) else ""
    parts = compatible_answer_parts(expected, value)
    if not parts:
        return ""
    deduped = list(dict.fromkeys(_canonical_part(expected, part) for part in parts if part))
    deduped = [part for part in deduped if part]
    return "; ".join(deduped)


def _canonical_part(expected: ExpectedAnswer, value: str) -> str:
    text = str(value or "").strip().strip('"')
    cleaned = clean_extracted_value(value)
    if expected.answer_type == "url":
        found = urls(cleaned)
        return found[0].rstrip(".,;)") if found else ""
    if expected.answer_type == "identifier":
        found = identifiers(cleaned)
        return found[0].rstrip(".,;)") if found else ""
    if expected.answer_type == "file_path":
        match = _FILE_RE.search(cleaned)
        return match.group(0).rstrip(".,;)") if match else ""
    if expected.answer_type == "count":
        match = re.search(r"\b\d+\b", cleaned)
        return match.group(0) if match else ""
    if expected.answer_type == "date_time":
        duration = _DURATION_RE.search(cleaned)
        if duration:
            return duration.group(0)
        match = _DATE_TIME_RE.search(cleaned)
        return match.group(0) if match else ""
    return text


def _is_structural_reference(value: str) -> bool:
    value_type = classify_value(value)
    return value_type in {"url", "file_path", "identifier", "count", "date_time"}
