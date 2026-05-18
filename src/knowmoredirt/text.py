"""Text utilities for raw-folder indexing and retrieval."""

from __future__ import annotations

import re


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:/#-]*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text or "")]


def content_tokens(text: str) -> list[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "can",
        "could",
        "did",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "would",
    }
    return [token for token in tokenize(text) if len(token) > 2 and token not in stop]


def split_units(text: str) -> list[tuple[int, int, str]]:
    """Split raw text into line/sentence units while keeping offsets."""
    units: list[tuple[int, int, str]] = []
    for match in SENTENCE_SPLIT_RE.finditer(text):
        pass
    cursor = 0
    for part in SENTENCE_SPLIT_RE.split(text):
        start = text.find(part, cursor)
        if start < 0:
            continue
        end = start + len(part)
        cursor = end
        value = part.strip()
        if value:
            units.append((start, end, value))
    return units


def compact_answer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).strip(" .;:")

