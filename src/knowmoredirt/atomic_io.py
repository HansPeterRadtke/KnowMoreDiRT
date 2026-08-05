"""Durable same-directory atomic file writes for runtime artifacts and caches."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, data: bytes, *, mode: int = 0o600) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
) -> None:
    atomic_write_bytes(path, text.encode(encoding), mode=mode)


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
    trailing_newline: bool = True,
    mode: int = 0o600,
) -> None:
    text = json.dumps(payload, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
    if trailing_newline:
        text += "\n"
    atomic_write_text(path, text, mode=mode)


def quarantine_corrupt_file(path: str | Path) -> Path | None:
    source = Path(path)
    if not source.exists():
        return None
    for index in range(1000):
        destination = source.with_name(
            f"{source.name}.corrupt.{os.getpid()}.{index}"
        )
        try:
            os.replace(source, destination)
        except FileNotFoundError:
            return None
        except FileExistsError:
            continue
        _fsync_directory(source.parent)
        return destination
    raise OSError(f"unable to quarantine corrupt file: {source}")
