"""Two-function public API."""
from __future__ import annotations

from pathlib import Path

from .engine import KnowMoreDiRTEngine

_ENGINE: KnowMoreDiRTEngine | None = None


def initialize(folder_path: str | Path) -> None:
    global _ENGINE
    _ENGINE = KnowMoreDiRTEngine(folder_path)


def question(text: str) -> str:
    if _ENGINE is None:
        raise RuntimeError("KnowMoreDiRT is not initialized; call initialize(folder_path) first")
    return _ENGINE.answer(text).text
