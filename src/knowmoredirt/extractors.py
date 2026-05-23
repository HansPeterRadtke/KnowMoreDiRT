"""Small source-grounded extractors used by the initial KMD engine."""

from __future__ import annotations

import re

URL_RE = re.compile(r"https?://[^\s)\],\"']+")
PREFIX_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]{2,12})*-\d+[A-Z0-9-]*\b")
LOWER_UNDERSCORE_ID_RE = re.compile(r"\b[a-z][a-z0-9]{1,12}_[a-z0-9]{6,}\b")
COMMIT_RE = re.compile(r"\b[0-9a-f]{8,16}\b", re.I)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
FILE_LIKE_RE = re.compile(r"\b[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,8}\b")


def urls(text: str) -> list[str]:
    return [match.group(0).rstrip(".") for match in URL_RE.finditer(text)]


def identifiers(text: str) -> list[str]:
    values: list[str] = []
    for regex in [PREFIX_ID_RE, LOWER_UNDERSCORE_ID_RE, COMMIT_RE, EMAIL_RE, FILE_LIKE_RE]:
        values.extend(match.group(0) for match in regex.finditer(text))
    return values


def capitalized_phrases(text: str) -> list[str]:
    pattern = re.compile(
        r"\b(?:(?:Dr\.|Ms\.|Mr\.|Mrs\.|Prof\.)\s+)?"
        r"[A-Z][A-Za-z0-9_-]+(?:\s+[A-Z][A-Za-z0-9_-]+){0,3}\b"
    )
    values: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(text):
        value = match.group(0).strip()
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def after_label(text: str, labels: list[str]) -> str:
    for label in labels:
        quoted = re.compile(rf"{re.escape(label)}\s*[:=]\s*\"([^\"]+)\"", re.I)
        quoted_match = quoted.search(text)
        if quoted_match:
            return quoted_match.group(1).strip()
        pattern = re.compile(rf"{re.escape(label)}\s*[:=]\s*([^\n.;]+)", re.I)
        match = pattern.search(text)
        if match:
            return match.group(1).strip().strip('"')
    return ""
