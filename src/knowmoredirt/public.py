"""Minimal public interface stub for the test-first foundation.

The final reasoning engine is intentionally not implemented in this phase.
This module exists to pin the public contract: initialize a raw folder, then
ask a plain question and receive a plain answer string.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnowMoreDiRTSession:
    """Placeholder session over a raw text folder.

    The session records the folder and readable files so tests can validate the
    public contract without asserting real QA behavior.
    """

    folder_path: Path
    readable_files: tuple[Path, ...]
    is_stub: bool = True

    def question(self, text: str) -> str:
        """Return a placeholder answer string for a plain question string."""
        if not isinstance(text, str):
            raise TypeError("question text must be a string")
        if not text.strip():
            return "unknown"
        return "unknown"


def initialize(folder_path: str | Path) -> KnowMoreDiRTSession:
    """Initialize KnowMoreDiRT from only a folder path."""
    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    readable_files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        readable_files.append(path)

    return KnowMoreDiRTSession(folder_path=root, readable_files=tuple(readable_files))

