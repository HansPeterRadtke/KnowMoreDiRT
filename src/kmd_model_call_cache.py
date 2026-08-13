"""Benchmark-agnostic cache for exact model-call outputs.

The cache key contains only semantic request state that can influence the model
output. Runtime bookkeeping (benchmark/run names, paths, timeouts, retry counts,
stream limits, logging, prompt-cache optimization) is deliberately excluded.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from kmd_runtime_config import model_cache_dir

_MODEL_SPLIT_RE = re.compile(r"^(?P<prefix>.+)-(?P<part>\d{5})-of-(?P<count>\d{5})\.gguf$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def semantic_request_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _model_parts(model_id: str) -> tuple[Path, ...]:
    path = Path(str(model_id or "")).expanduser()
    if not path.is_file():
        return ()
    match = _MODEL_SPLIT_RE.match(path.name)
    if not match:
        return (path.resolve(),)
    prefix = match.group("prefix")
    count = int(match.group("count"))
    parts = tuple(
        path.with_name(f"{prefix}-{index:05d}-of-{count:05d}.gguf").resolve()
        for index in range(1, count + 1)
    )
    return parts if all(part.is_file() for part in parts) else (path.resolve(),)


def _stat_signature(parts: Iterable[Path]) -> tuple[tuple[str, int, int], ...]:
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in parts)


@lru_cache(maxsize=32)
def _hash_model_signature(signature: tuple[tuple[str, int, int], ...]) -> str:
    digest = hashlib.sha256()
    for path_text, size, _mtime_ns in signature:
        path = Path(path_text)
        digest.update(path.name.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_content_fingerprint(model_id: str) -> dict[str, Any]:
    """Return an exact content identity when the configured model is a local file."""

    parts = _model_parts(model_id)
    if not parts:
        return {"kind": "logical_model_id_v1", "model_id": str(model_id or "")}
    signature = _stat_signature(parts)
    return {
        "kind": "local_model_content_sha256_v1",
        "model_id": str(model_id),
        "part_count": len(parts),
        "part_sizes": [size for _path, size, _mtime in signature],
        "content_sha256": _hash_model_signature(signature),
    }


def cache_path(request_hash: str) -> Path:
    root = model_cache_dir("KMD_MODEL_CALL_CACHE_DIR")
    return root / request_hash[:2] / f"{request_hash}.json"


def _test_cache_disabled() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) and os.environ.get("KMD_TEST_DISABLE_MODEL_CALL_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}


def read_model_call(request_hash: str) -> dict[str, Any] | None:
    if _test_cache_disabled():
        return None
    path = cache_path(request_hash)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("request_hash") != request_hash:
        return None
    response = value.get("response")
    return response if isinstance(response, dict) else None


def write_model_call(request_hash: str, response: dict[str, Any]) -> Path:
    path = cache_path(request_hash)
    if _test_cache_disabled():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cache_schema": "kmd-model-call-v1", "request_hash": request_hash, "response": response}
    data = canonical_json(payload) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return path
