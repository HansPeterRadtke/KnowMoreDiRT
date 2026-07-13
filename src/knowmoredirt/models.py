"""Small data contracts for the model-owned query system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceRecord:
    record_id: str
    collection_path: str
    source_path: str
    record_index: int
    data: dict[str, Any]
    text: str

    @property
    def search_text(self) -> str:
        import json

        return (self.text + "\n" + json.dumps(self.data, ensure_ascii=False, default=str)).lower()

    def model_view(self, max_chars: int = 1800) -> dict[str, Any]:
        import json

        rendered = json.dumps(self.data, ensure_ascii=False, default=str)
        return {
            "record_id": self.record_id,
            "collection_path": self.collection_path,
            "source_path": self.source_path,
            "record_index": self.record_index,
            "data": self.data if len(rendered) <= max_chars else {},
            "excerpt": (self.text or rendered)[:max_chars],
        }


@dataclass
class ToolResult:
    step_id: str
    kind: str
    records: list[SourceRecord] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)
    scalar: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def model_view(self, max_items: int = 30, max_chars: int = 32000) -> dict[str, Any]:
        payload = {
            "step_id": self.step_id,
            "kind": self.kind,
            "records": [item.model_view() for item in self.records[:max_items]],
            "values": self.values[:max_items],
            "scalar": self.scalar,
            "diagnostics": self.diagnostics,
        }
        import json

        rendered = json.dumps(payload, ensure_ascii=False, default=str)
        if len(rendered) <= max_chars:
            return payload
        payload["records"] = [item.model_view(600) for item in self.records[:10]]
        payload["values"] = self.values[:10]
        payload["diagnostics"] = {**self.diagnostics, "truncated_for_model": True}
        return payload


@dataclass(frozen=True)
class Answer:
    text: str
    evidence: tuple[dict[str, Any], ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
