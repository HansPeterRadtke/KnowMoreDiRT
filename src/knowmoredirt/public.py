"""Thread-safe two-function public API for KnowMoreDiRT."""

from __future__ import annotations

import threading
from pathlib import Path

from .engine import KnowMoreDiRTEngine

_ENGINE: KnowMoreDiRTEngine | None = None
_ENGINE_LOCK = threading.RLock()


def initialize(folder_path: str | Path) -> None:
    """Atomically replace the global KMD knowledge base."""

    global _ENGINE
    with _ENGINE_LOCK:
        previous = _ENGINE
        replacement = KnowMoreDiRTEngine(folder_path)
        try:
            if previous is not None:
                previous.close()
        except BaseException:
            replacement.close()
            raise
        _ENGINE = replacement


def question(text: str) -> str:
    """Answer one question while holding the engine lifecycle lock."""

    with _ENGINE_LOCK:
        if _ENGINE is None:
            raise RuntimeError("KnowMoreDiRT is not initialized; call initialize(folder_path) first")
        return _ENGINE.answer(text).text


def _reset() -> None:
    """Close and clear the global engine."""

    global _ENGINE
    with _ENGINE_LOCK:
        previous = _ENGINE
        _ENGINE = None
        if previous is not None:
            previous.close()
