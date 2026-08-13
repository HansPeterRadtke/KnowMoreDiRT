"""Internal data models for the KnowMoreDiRT raw-text engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
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
    metadata: dict[str, object] = field(default_factory=dict)


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
    span_id: str = ""
    chunk_order: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    source_kind: str = "source_span"

    def evidence_id(self) -> str:
        if self.span_id:
            return self.span_id
        material = "|".join(
            [
                self.rel_path,
                str(self.chunk_order if self.chunk_order is not None else ""),
                str(self.char_start if self.char_start is not None else ""),
                str(self.char_end if self.char_end is not None else ""),
                self.source_kind,
                self.text,
            ]
        )
        return "evidence:" + hashlib.sha256(material.encode("utf-8", errors="surrogateescape")).hexdigest()


@dataclass
class Answer:
    """Internal structured answer candidate; public API still renders ``text``."""

    text: str
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    reason: str = ""
    answer_type: str = "unknown"
    status: str = ""
    requested_scope: str = "real_world"
    direct_evidence_ids: list[str] = field(default_factory=list)
    related_evidence_ids: list[str] = field(default_factory=list)
    contradiction_ids: list[str] = field(default_factory=list)
    scope_qualifications: list[str] = field(default_factory=list)
    provenance: list[dict[str, object]] = field(default_factory=list)
    derivation: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.status:
            normalized = self.text.strip().lower()
            self.status = "unknown" if normalized == "unknown" or normalized.startswith("unknown ") else "answered"
        if not self.direct_evidence_ids and self.status == "answered":
            self.direct_evidence_ids = [item.evidence_id() for item in self.evidence]
        if not self.related_evidence_ids and self.status == "unknown":
            self.related_evidence_ids = [item.evidence_id() for item in self.evidence]
        if not self.provenance:
            self.provenance = [
                {
                    "evidence_id": item.evidence_id(),
                    "rel_path": item.rel_path,
                    "span_id": item.span_id,
                    "chunk_order": item.chunk_order,
                    "char_start": item.char_start,
                    "char_end": item.char_end,
                    "source_kind": item.source_kind,
                }
                for item in self.evidence
            ]
