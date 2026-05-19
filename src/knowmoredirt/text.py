"""Text utilities for raw-folder indexing and retrieval."""

from __future__ import annotations

import re
import string


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:/#-]*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ABBREVIATION_DOTS = {
    "Dr.": "Dr§",
    "Mr.": "Mr§",
    "Ms.": "Ms§",
    "Mrs.": "Mrs§",
    "Prof.": "Prof§",
}


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
    protected = text
    for source, target in ABBREVIATION_DOTS.items():
        protected = protected.replace(source, target)
    units: list[tuple[int, int, str]] = []
    for match in SENTENCE_SPLIT_RE.finditer(protected):
        pass
    cursor = 0
    for part in SENTENCE_SPLIT_RE.split(protected):
        start = protected.find(part, cursor)
        if start < 0:
            continue
        end = start + len(part)
        cursor = end
        value = part.strip()
        for source, target in ABBREVIATION_DOTS.items():
            value = value.replace(target, source)
        if value:
            units.append((start, end, value))
    return units


def compact_answer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).strip(" .;:")


def text_quality_metrics(text: str) -> dict[str, float | int | bool]:
    """Return generic raw-text quality signals used to downweight noise.

    These metrics are intentionally structural rather than fixture-specific:
    printable ratio, alphanumeric ratio, token count, unique-token ratio, and
    long-token pressure. They do not classify topics or assume schemas.
    """

    value = str(text or "")
    length = len(value)
    if length == 0:
        return {
            "char_count": 0,
            "printable_ratio": 1.0,
            "alnum_ratio": 0.0,
            "symbol_ratio": 0.0,
            "token_count": 0,
            "unique_token_ratio": 0.0,
            "long_token_ratio": 0.0,
            "low_semantic_noise": True,
        }
    printable = sum(1 for char in value if char in string.printable or char.isprintable())
    alnum = sum(1 for char in value if char.isalnum())
    symbolic = sum(1 for char in value if not char.isalnum() and not char.isspace())
    tokens = tokenize(value)
    unique_tokens = set(tokens)
    long_tokens = [token for token in tokens if len(token) >= 24]
    printable_ratio = printable / length
    alnum_ratio = alnum / length
    symbol_ratio = symbolic / length
    token_count = len(tokens)
    unique_token_ratio = (len(unique_tokens) / token_count) if token_count else 0.0
    long_token_ratio = (len(long_tokens) / token_count) if token_count else 0.0
    low_semantic_noise = (
        printable_ratio < 0.75
        or alnum_ratio < 0.25
        or symbol_ratio > 0.35
        or (length > 80 and token_count < 4)
        or (token_count >= 20 and unique_token_ratio > 0.95 and long_token_ratio > 0.20)
    )
    return {
        "char_count": length,
        "printable_ratio": round(printable_ratio, 4),
        "alnum_ratio": round(alnum_ratio, 4),
        "symbol_ratio": round(symbol_ratio, 4),
        "token_count": token_count,
        "unique_token_ratio": round(unique_token_ratio, 4),
        "long_token_ratio": round(long_token_ratio, 4),
        "low_semantic_noise": bool(low_semantic_noise),
    }


def is_low_semantic_noise(text: str) -> bool:
    return bool(text_quality_metrics(text)["low_semantic_noise"])
