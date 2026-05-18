"""Internal data models for the KnowMoreDiRT raw-text engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """A readable raw text file discovered under the initialized folder."""

    document_id: str
    path: Path
    rel_path: str
    text: str
    size_bytes: int
    mtime: float
    ctime: float
    sha256: str


@dataclass(frozen=True)
class Sentence:
    """A source-grounded sentence or line-like text unit."""

    sentence_id: str
    document_id: str
    rel_path: str
    text: str
    order: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Evidence:
    """Source evidence used internally for scoring and diagnostics."""

    rel_path: str
    text: str
    score: float = 0.0


@dataclass
class Answer:
    """Internal answer candidate."""

    text: str
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    reason: str = ""

