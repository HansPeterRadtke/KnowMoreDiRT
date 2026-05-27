"""Generic answer type inference and validation.

The public API still returns a plain string.  Internally, KMD uses these broad
answer expectations to reject type-unsafe candidates before they leave the
grounded DSPG/query layer.
"""

from __future__ import annotations

import ast
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
_PATH_RE = re.compile(r"\b[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+\b")
_DATE_TIME_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b"
)
_DURATION_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b", re.I)


def infer_expected_answer(question: str) -> ExpectedAnswer:
    """Return a non-semantic default expectation.

    The answer type for a natural-language question is part of the query DRS
    and must be supplied by the model.  Deterministic code may classify and
    validate candidate values once a query schema asks for a type, but it must
    not infer that type from question words.
    """

    return ExpectedAnswer("unknown")


def answer_parts(value: str) -> list[str]:
    text = clean_extracted_value(value)
    if not text:
        return []
    parts = [clean_extracted_value(part) for part in re.split(r"\s*;\s*", text)]
    return [part for part in parts if part]


def classify_value(value: str) -> AnswerType:
    """Classify deterministic value shapes, not natural-language roles.

    Person, actor, organization, state, and other content-level readings are
    query-DRS/model-owned.  Deterministic classification is limited to surface
    forms with non-semantic structure such as URLs, identifiers, files, counts,
    and dates.
    """

    text = clean_extracted_value(value)
    low = normalize(text)
    if not text or low == "unknown":
        return "unknown"
    if low in {"the", "a", "an"}:
        return "content_phrase"
    if re.match(r"^(yes|no)(?:$|[;,:.!?]\s+)", low):
        return "boolean"
    if urls(text) and urls(text)[0].rstrip(".,;)") == text.rstrip(".,;)"):
        return "url"
    found_urls = urls(text)
    if len(found_urls) == 1:
        remainder = normalize(text.replace(found_urls[0], " "))
        if not remainder or all(token in {"at", "in", "on", "from", "to"} for token in remainder.split()):
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
    if " " not in text and any(char.isupper() for char in text[1:]):
        return "content_phrase"
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
        return value_type != "unknown"
    if expected_type in {"person", "actor", "organization"}:
        return value_type not in {"url", "file_path", "identifier", "count", "date_time", "unknown"} and not _is_structural_reference(value)
    if expected_type == "content_phrase":
        return value_type not in {"url", "file_path", "identifier"}
    if expected_type == "state":
        return value_type not in {"url", "file_path", "identifier", "count", "date_time"}
    if expected_type == "metadata_value":
        return True
    return value_type == expected_type


def compatible_answer_parts(expected: ExpectedAnswer, value: str) -> list[str]:
    return [part for part in answer_parts(value) if is_value_compatible(expected, part)]


def canonicalize_answer(expected: ExpectedAnswer, value: str) -> str:
    if expected.answer_type == "boolean":
        text = str(value or "").strip()
        return text if is_value_compatible(expected, text) else ""
    if expected.answer_type in {"person", "actor", "organization"} and ";" in str(value):
        parts = compatible_answer_parts(expected, value)
        return "; ".join(dict.fromkeys(parts))
    if expected.answer_type in {"person", "actor", "organization", "state", "content_phrase", "metadata_value"}:
        text = clean_extracted_value(str(value or "").strip().strip('"')).strip(" .;:")
        text = _format_literal_list(text) or text
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
        without_urls = cleaned
        for url in urls(cleaned):
            without_urls = without_urls.replace(url, " ")
        path_match = _PATH_RE.search(without_urls)
        if path_match:
            return path_match.group(0).rstrip(".,;)")
        match = _FILE_RE.search(without_urls)
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


def _format_literal_list(value: str) -> str:
    text = str(value or "").strip()
    if not (text.startswith("[") and text.endswith("]")):
        return ""
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return ""
    if not isinstance(parsed, list) or not parsed or not all(isinstance(item, str) for item in parsed):
        return ""
    items = [item.strip() for item in parsed if item.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _is_structural_reference(value: str) -> bool:
    if "://" in str(value) or "http" in normalize(value):
        return True
    if urls(value) or identifiers(value):
        return True
    value_type = classify_value(value)
    return value_type in {"url", "file_path", "identifier", "count", "date_time"}
