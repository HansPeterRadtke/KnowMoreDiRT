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

from kmd_runtime_config import explicit_raw as _config_explicit_raw, text as _config_text, model_cache_dir as _model_cache_dir

from .atomic_io import atomic_write_json, quarantine_corrupt_file
from .model_planner import CHUNK_FRAME_SCHEMA_VERSION, PROMPT_VERSION


CACHE_VERSION = "semantic-frames-v7-atomic-runtime-fingerprint"


def _default_cache_dir() -> Path:
    return _model_cache_dir("KMD_FRAME_CACHE_DIR")


class SemanticFrameCache:
    """Small JSON-file cache keyed by source text and extraction version."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else _default_cache_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def key_for(self, text: str, *, context: dict[str, Any] | None = None) -> str:
        material = json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "endpoint": _config_text("KMD_LOCAL_MODEL_ENDPOINT"),
                "env_model_id": str(_config_explicit_raw("KMD_LOCAL_MODEL_ID") or ""),
                "seed": _config_text("KMD_LOCAL_MODEL_SEED"),
                "prompt_version": PROMPT_VERSION,
                "schema_version": CHUNK_FRAME_SCHEMA_VERSION,
                "grammar_enabled": str(_config_explicit_raw("KMD_LOCAL_MODEL_GRAMMAR") or ""),
                "runtime_context": context or {},
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8", errors="replace")
        return hashlib.sha256(material).hexdigest()

    def get(self, text: str, *, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
        path = self.root / f"{self.key_for(text, context=context)}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            quarantine_corrupt_file(path)
            return None
        except OSError:
            return None
        if payload.get("version") != CACHE_VERSION:
            return None
        frames = payload.get("frames")
        if not isinstance(frames, list):
            return None
        return payload

    def put(
        self,
        text: str,
        frames: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        path = self.root / f"{self.key_for(text, context=context)}.json"
        payload = {
            "version": CACHE_VERSION,
            "frames": frames,
            "metadata": metadata or {},
            "context": context or {},
        }
        atomic_write_json(path, payload, ensure_ascii=False, sort_keys=True)
