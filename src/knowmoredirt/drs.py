"""Generic DRT-style discourse structures used by the DSPG layer.

These dataclasses are deliberately relation-agnostic.  Predicate and role
labels are data extracted from source text or model frames, not control-flow
intent names.  The store remains SQLite-backed, but these objects make the
Kamp-style construction target explicit in the Python package: source text
introduces discourse referents and grounded conditions inside accessible
contexts, and questions bind answer variables against those conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .text import clean_extracted_value, normalize


@dataclass(frozen=True)
class DiscourseReferent:
    surface: str
    value_type: str = "unknown"
    normalized: str = ""
    source: str = "source"

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface", clean_extracted_value(self.surface))
        object.__setattr__(self, "value_type", clean_extracted_value(self.value_type) or "unknown")
        object.__setattr__(self, "normalized", self.normalized or normalize(self.surface))


@dataclass(frozen=True)
class DiscourseArgument:
    role: str
    value: str
    value_type: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", clean_extracted_value(self.role) or "argument")
        object.__setattr__(self, "value", clean_extracted_value(self.value))
        object.__setattr__(self, "value_type", clean_extracted_value(self.value_type) or "unknown")


@dataclass(frozen=True)
class DiscourseCondition:
    predicate: str
    arguments: tuple[DiscourseArgument, ...] = ()
    frame_type: str = "relation"
    polarity: str = "positive"
    modality: str = "asserted"
    temporal_text: str = ""
    evidence_text: str = ""
    confidence: float = 0.65
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate", clean_extracted_value(self.predicate))
        object.__setattr__(self, "frame_type", clean_extracted_value(self.frame_type) or "relation")
        object.__setattr__(self, "polarity", normalize(self.polarity) or "positive")
        object.__setattr__(self, "modality", normalize(self.modality) or "asserted")
        object.__setattr__(self, "temporal_text", clean_extracted_value(self.temporal_text))
        object.__setattr__(self, "evidence_text", clean_extracted_value(self.evidence_text))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))


@dataclass(frozen=True)
class DiscourseContext:
    kind: str = "asserted"
    parent: str = ""
    holder: str = ""
    evidence_text: str = ""
    accessible_as_fact: bool = True

    def __post_init__(self) -> None:
        kind = normalize(self.kind) or "asserted"
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "parent", clean_extracted_value(self.parent))
        object.__setattr__(self, "holder", clean_extracted_value(self.holder))
        object.__setattr__(self, "evidence_text", clean_extracted_value(self.evidence_text))
        object.__setattr__(
            self,
            "accessible_as_fact",
            kind in {"asserted", "source", "metadata", "quality"},
        )


def frame_from_model_dict(frame: dict[str, Any]) -> DiscourseCondition | None:
    """Normalize one model-produced frame into a grounded condition."""

    predicate = clean_extracted_value(str(frame.get("predicate") or ""))
    evidence_text = str(frame.get("evidence_text") or "").strip()
    if not predicate or not evidence_text:
        return None
    raw_arguments = frame.get("arguments")
    if isinstance(raw_arguments, dict):
        raw_arguments = [
            {"role": str(role), "text": str(value), "value_type": "unknown"}
            for role, value in raw_arguments.items()
        ]
    arguments: list[DiscourseArgument] = []
    if isinstance(raw_arguments, list):
        for item in raw_arguments:
            if not isinstance(item, dict):
                continue
            value = clean_extracted_value(str(item.get("text") or item.get("value") or ""))
            if not value:
                continue
            arguments.append(
                DiscourseArgument(
                    role=str(item.get("role") or f"arg_{len(arguments)}"),
                    value=value,
                    value_type=str(item.get("value_type") or "unknown"),
                )
            )
    return DiscourseCondition(
        predicate=predicate,
        arguments=tuple(arguments),
        frame_type=str(frame.get("frame_type") or "relation"),
        polarity=str(frame.get("polarity") or "positive"),
        modality=str(frame.get("modality") or "asserted"),
        temporal_text=str(frame.get("temporal_text") or ""),
        evidence_text=evidence_text,
        confidence=float(frame.get("confidence") or 0.65),
        metadata={
            "source": "local_model",
            "context_holder": frame.get("context_holder") if isinstance(frame.get("context_holder"), str) else "",
            "identity_hypotheses": frame.get("identity_hypotheses") if isinstance(frame.get("identity_hypotheses"), list) else [],
        },
    )
