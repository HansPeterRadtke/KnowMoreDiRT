"""Local cache for source-grounded semantic frame extraction.

The cache stores model-derived DRT/DSPG frames by chunk hash and prompt version.
It is an optimization only; cached frames are still filtered for source
grounding before they are inserted into the internal graph.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .model_planner import CHUNK_FRAME_SCHEMA_VERSION, PROMPT_VERSION


CACHE_VERSION = "semantic-frames-v5"


def _default_cache_dir() -> Path:
    value = os.environ.get("KMD_FRAME_CACHE_DIR")
    if value:
        return Path(value)
    return Path.home() / ".cache" / "knowmoredirt" / "semantic_frames"


class SemanticFrameCache:
    """Small JSON-file cache keyed by source text and extraction version."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else _default_cache_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def key_for(self, text: str) -> str:
        material = "\x1f".join(
            [
                CACHE_VERSION,
                os.environ.get("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:14829/v1"),
                os.environ.get("KMD_LOCAL_MODEL_ID", ""),
                os.environ.get("KMD_LOCAL_MODEL_SEED", "1778779265"),
                PROMPT_VERSION,
                CHUNK_FRAME_SCHEMA_VERSION,
                os.environ.get("KMD_CHUNK_FRAME_N_PREDICT", "192"),
                os.environ.get("KMD_LOCAL_MODEL_GRAMMAR", ""),
                text,
            ]
        ).encode("utf-8", errors="replace")
        return hashlib.sha256(material).hexdigest()

    def get(self, text: str) -> dict[str, Any] | None:
        path = self.root / f"{self.key_for(text)}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("version") != CACHE_VERSION:
            return None
        frames = payload.get("frames")
        if not isinstance(frames, list):
            return None
        return payload

    def put(self, text: str, frames: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> None:
        path = self.root / f"{self.key_for(text)}.json"
        payload = {
            "version": CACHE_VERSION,
            "frames": frames,
            "metadata": metadata or {},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
