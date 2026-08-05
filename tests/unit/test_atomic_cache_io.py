from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from knowmoredirt.atomic_io import atomic_write_json
from knowmoredirt.model_planner import _read_cache, _write_cache
from knowmoredirt.semantic_cache import SemanticFrameCache


def test_atomic_json_writes_never_expose_partial_payload(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    atomic_write_json(path, {"marker": -1, "blob": "x" * 100_000})
    stop = threading.Event()
    malformed: list[Exception] = []

    def reader() -> None:
        while not stop.is_set():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as error:  # pragma: no cover - assertion captures any race
                malformed.append(error)
                stop.set()

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for index in range(60):
            atomic_write_json(path, {"marker": index, "blob": str(index) * 100_000})
    finally:
        stop.set()
        thread.join(timeout=5)

    assert malformed == []
    assert json.loads(path.read_text(encoding="utf-8"))["marker"] == 59


def test_planner_cache_quarantines_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "planner.json"
    path.write_text('{"accepted": true', encoding="utf-8")

    assert _read_cache(path) is None
    assert not path.exists()
    quarantined = list(tmp_path.glob("planner.json.corrupt.*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == '{"accepted": true'


def test_planner_write_cache_is_atomic_and_readable(tmp_path: Path) -> None:
    path = tmp_path / "planner.json"

    _write_cache(path, {"accepted": True, "reason": "ok"})

    cached = _read_cache(path)
    assert cached is not None
    assert cached["accepted"] is True
    assert cached["fresh_or_cached"] == "cache"
    assert not list(tmp_path.glob("*.tmp"))


def test_semantic_cache_quarantines_malformed_json(tmp_path: Path) -> None:
    cache = SemanticFrameCache(tmp_path)
    path = tmp_path / f"{cache.key_for('text', context={'model': 'x'})}.json"
    path.write_text("not-json", encoding="utf-8")

    assert cache.get("text", context={"model": "x"}) is None
    assert not path.exists()
    assert len(list(tmp_path.glob(f"{path.name}.corrupt.*"))) == 1


def test_semantic_cache_concurrent_put_get_never_returns_malformed(tmp_path: Path) -> None:
    cache = SemanticFrameCache(tmp_path)
    stop = threading.Event()
    errors: list[Exception] = []

    def reader() -> None:
        while not stop.is_set():
            try:
                payload = cache.get("text", context={"model": "x"})
                if payload is not None:
                    assert isinstance(payload["frames"], list)
            except Exception as error:  # pragma: no cover - assertion captures any race
                errors.append(error)
                stop.set()

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for index in range(40):
            cache.put(
                "text",
                [{"marker": index, "blob": "x" * 100_000}],
                context={"model": "x"},
            )
            time.sleep(0.001)
    finally:
        stop.set()
        thread.join(timeout=5)

    assert errors == []
