"""Raw folder scanning for KnowMoreDiRT."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import stat as stat_module
from pathlib import Path

from kmd_runtime_config import boolean as _config_boolean, integer as _config_int, text as _config_text

from .models import Document, Sentence
from .text import split_units, tokenize


KMD_CACHE_DIR_ENV_VARS = (
    "KMD_FRAME_CACHE_DIR",
    "KMD_CHUNK_FRAME_CACHE_DIR",
    "KMD_CHUNK_DRS_CACHE_DIR",
    "KMD_QUERY_PLAN_CACHE_DIR",
    "KMD_QUERY_DRS_CACHE_DIR",
    "KMD_QUERY_EVIDENCE_REPAIR_CACHE_DIR",
    "KMD_QUERY_EVIDENCE_CACHE_DIR",
    "KMD_EVIDENCE_ANSWER_CACHE_DIR",
    "KMD_VERIFIER_CACHE_DIR",
    "KMD_QUERY_VERIFIER_CACHE_DIR",
    "KMD_ANSWER_CANONICALIZATION_CACHE_DIR",
    "KMD_QUERY_CANONICAL_CACHE_DIR",
    "KMD_IDENTITY_CACHE_DIR",
    "KMD_IDENTITY_CANONICAL_CACHE_DIR",
    "KMD_SOURCE_RESOLUTION_CACHE_DIR",
    "KMD_SHARED_MODEL_CACHE_ROOT",
)

GENERATED_DIRECTORY_NAMES = frozenset({
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
})


def _stable_scan_id(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


def read_text_file(path: Path) -> str | None:
    """Read a file as text if possible.

    The scanner intentionally does not interpret extensions or schemas. Any
    readable text file is accepted as raw text; unreadable/binary files are
    skipped.
    """

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    except OSError:
        return None


def _read_text_file_snapshot(
    path: Path,
) -> tuple[str, dict[str, object], os.stat_result] | None:
    """Read one regular file from a stable descriptor without following symlinks."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        stat_result = os.fstat(descriptor)
        if not stat_module.S_ISREG(stat_result.st_mode):
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read()
    except OSError:
        return None
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    try:
        text = data.decode("utf-8")
        metadata = {
            "encoding": "utf-8",
            "decode_errors": False,
            "read_mode": "strict_text",
            "source_sha256": hashlib.sha256(data).hexdigest(),
        }
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        metadata = {
            "encoding": "utf-8",
            "decode_errors": True,
            "read_mode": "replacement_text",
            "source_sha256": hashlib.sha256(data).hexdigest(),
        }
    return text, metadata, stat_result


def read_text_file_with_metadata(path: Path) -> tuple[str, dict[str, object]] | None:
    """Read a regular file as text without following a symbolic link."""

    result = _read_text_file_snapshot(path)
    if result is None:
        return None
    text, metadata, _stat_result = result
    return text, metadata


def _default_max_unit_chars() -> int:
    value = _config_int("KMD_SCANNER_DEFAULT_UNIT_CHARS")
    if value <= 0:
        raise ValueError("KMD_SCANNER_DEFAULT_UNIT_CHARS must be a positive integer")
    return value

def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _include_generated_cache_content() -> bool:
    return _config_boolean("KMD_SCAN_INCLUDE_GENERATED_CACHES")


def _configured_cache_roots(root: Path) -> list[Path]:
    if _include_generated_cache_content():
        return []
    values = [
        _config_text(name).strip()
        for name in KMD_CACHE_DIR_ENV_VARS
        if _config_text(name).strip()
    ]
    values.append(str(Path.home() / ".cache" / "knowmoredirt"))
    root_resolved = root.resolve()
    cache_roots: list[Path] = []
    for value in values:
        candidate = Path(value).expanduser()
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            continue
        if resolved != root_resolved and _path_is_relative_to(resolved, root_resolved):
            cache_roots.append(resolved)
    return list(dict.fromkeys(cache_roots))


def _strip_original_span(text: str, start: int, end: int) -> tuple[int, int, str]:
    segment = text[start:end]
    leading = len(segment) - len(segment.lstrip())
    stripped = segment.strip()
    if not stripped:
        return start, start, ""
    return start + leading, start + leading + len(stripped), stripped


def _append_bounded_record_unit(
    units: list[tuple[int, int, str]],
    *,
    start: int,
    value: str,
    max_unit_chars: int,
) -> None:
    if max_unit_chars <= 0 or len(value) <= max_unit_chars:
        units.append((start, start + len(value), value))
        return
    offset = 0
    while offset < len(value):
        hard_end = min(len(value), offset + max_unit_chars)
        split_end = hard_end
        if hard_end < len(value):
            floor = offset + max(1, max_unit_chars // 2)
            whitespace = value.rfind(" ", floor, hard_end)
            if whitespace > offset:
                split_end = whitespace
        chunk = value[offset:split_end].strip()
        if chunk:
            leading = len(value[offset:split_end]) - len(value[offset:split_end].lstrip())
            units.append((start + offset + leading, start + offset + leading + len(chunk), chunk))
        offset = split_end
        while offset < len(value) and value[offset].isspace():
            offset += 1


def _json_structural_split_positions(text: str) -> list[int]:
    """Return safe lexical split positions for a valid JSON value.

    Positions are taken after commas, line endings, or completed values at the
    two shallowest container levels. This preserves every non-whitespace source
    character while preferring object fields and array records over arbitrary
    character cuts.
    """

    positions: list[int] = []
    stack: list[str] = []
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in "{[":
            stack.append(character)
            continue
        if character in "}]":
            if stack:
                stack.pop()
            if len(stack) <= 2:
                positions.append(index + 1)
            continue
        if character == "," and len(stack) <= 2:
            positions.append(index + 1)
            continue
        if character == "\n" and len(stack) <= 2:
            positions.append(index + 1)
    return sorted(set(position for position in positions if 0 < position < len(text)))


def _split_json_structure_units(
    text: str,
    *,
    max_unit_chars: int = 0,
) -> list[tuple[int, int, str]]:
    """Split a valid JSON document at structural boundaries when required."""

    if max_unit_chars <= 0 or len(text) <= max_unit_chars:
        return []
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return []
    positions = _json_structural_split_positions(text)
    if not positions:
        return []
    units: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_unit_chars)
        if hard_end >= len(text):
            split_end = len(text)
        else:
            candidates = [position for position in positions if start < position <= hard_end]
            split_end = candidates[-1] if candidates else hard_end
        out_start, out_end, value = _strip_original_span(text, start, split_end)
        if value:
            units.append((out_start, out_end, value))
        start = split_end
        while start < len(text) and text[start].isspace():
            start += 1
    return units


def _split_jsonl_units(
    text: str,
    *,
    max_unit_chars: int = 0,
) -> list[tuple[int, int, str]]:
    """Return one offset-preserving unit per valid nonblank JSONL record.

    A malformed nonblank line disables record mode for the entire file so the
    generic text splitter remains the safe fallback for ordinary files that
    merely use a .jsonl suffix.
    """

    units: list[tuple[int, int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        without_newline = line.rstrip("\r\n")
        value = without_newline.strip()
        if value:
            try:
                json.loads(value)
            except json.JSONDecodeError:
                return []
            leading = len(without_newline) - len(without_newline.lstrip())
            _append_bounded_record_unit(
                units,
                start=cursor + leading,
                value=value,
                max_unit_chars=max_unit_chars,
            )
        cursor += len(line)
    if cursor < len(text):
        tail = text[cursor:]
        value = tail.strip()
        if value:
            try:
                json.loads(value)
            except json.JSONDecodeError:
                return []
            leading = len(tail) - len(tail.lstrip())
            _append_bounded_record_unit(
                units,
                start=cursor + leading,
                value=value,
                max_unit_chars=max_unit_chars,
            )
    return units


def _pack_split_units(
    text: str,
    units: list[tuple[int, int, str]],
    *,
    max_pack_chars: int = 0,
    max_pack_units: int = 0,
) -> list[tuple[int, int, str]]:
    if (max_pack_chars <= 0 and max_pack_units <= 0) or len(units) <= 1:
        return units
    packed: list[tuple[int, int, str]] = []
    pack_start: int | None = None
    pack_end: int | None = None
    pack_count = 0
    for start, end, _unit in units:
        if pack_start is None or pack_end is None:
            pack_start, pack_end = start, end
            pack_count = 1
            continue
        would_fit_chars = max_pack_chars <= 0 or end - pack_start <= max_pack_chars
        would_fit_units = max_pack_units <= 0 or pack_count < max_pack_units
        if would_fit_chars and would_fit_units:
            pack_end = end
            pack_count += 1
            continue
        out_start, out_end, value = _strip_original_span(text, pack_start, pack_end)
        if value:
            packed.append((out_start, out_end, value))
        pack_start, pack_end = start, end
        pack_count = 1
    if pack_start is not None and pack_end is not None:
        out_start, out_end, value = _strip_original_span(text, pack_start, pack_end)
        if value:
            packed.append((out_start, out_end, value))
    return packed


def scan_folder(
    folder_path: str | Path,
    *,
    max_unit_chars: int = 0,
    pack_unit_chars: int = 0,
    pack_unit_count: int = 0,
) -> tuple[list[Document], list[Sentence]]:
    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    root_resolved = root.resolve(strict=True)
    effective_max_unit_chars = int(max_unit_chars) if int(max_unit_chars) > 0 else _default_max_unit_chars()
    cache_roots = _configured_cache_roots(root)
    documents: list[Document] = []
    sentences: list[Sentence] = []
    include_generated = _include_generated_cache_content()
    for path in sorted(root.rglob("*")):
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            rel_parts = path.parts
        if not include_generated and any(part in GENERATED_DIRECTORY_NAMES for part in rel_parts[:-1]):
            continue
        if path.is_symlink():
            continue
        try:
            resolved_path = path.resolve(strict=True)
        except OSError:
            continue
        if not _path_is_relative_to(resolved_path, root_resolved):
            continue
        if cache_roots and any(_path_is_relative_to(resolved_path, cache_root) for cache_root in cache_roots):
            continue
        read_result = _read_text_file_snapshot(path)
        if read_result is None:
            continue
        text, read_metadata, stat = read_result
        rel_path = path.relative_to(root).as_posix()
        content_hash = str(read_metadata["source_sha256"])
        document_id = _stable_scan_id("doc", rel_path, content_hash)
        suffixes = list(path.suffixes)
        metadata: dict[str, object] = {
            **read_metadata,
            "file_name": path.name,
            "stem": path.stem,
            "suffix": path.suffix,
            "suffixes": suffixes,
            "parent_rel_path": path.parent.relative_to(root).as_posix() if path.parent != root else "",
            "path_parts": list(Path(rel_path).parts),
            "directory_depth": max(0, len(Path(rel_path).parts) - 1),
            "hidden_file": any(part.startswith(".") for part in Path(rel_path).parts),
            "stat_mode": stat.st_mode,
            "permissions": stat_module.filemode(stat.st_mode),
            "uid": getattr(stat, "st_uid", None),
            "gid": getattr(stat, "st_gid", None),
            "inode": getattr(stat, "st_ino", None),
            "device": getattr(stat, "st_dev", None),
            "atime": stat.st_atime,
            "mtime": stat.st_mtime,
            "ctime": stat.st_ctime,
            "symlink": False,
            "symlink_target": "",
            "mime_type": mimetypes.guess_type(path.name)[0] or "",
            "line_count": text.count("\n") + (1 if text else 0),
            "word_count": len(tokenize(text)),
        }
        document = Document(
            document_id=document_id,
            path=path,
            rel_path=rel_path,
            text=text,
            size_bytes=stat.st_size,
            mtime=stat.st_mtime,
            ctime=stat.st_ctime,
            sha256=content_hash,
            metadata=metadata,
        )
        documents.append(document)
        record_units = (
            _split_jsonl_units(text, max_unit_chars=effective_max_unit_chars)
            if path.suffix.lower() == ".jsonl"
            else []
        )
        structured_json_units = (
            _split_json_structure_units(text, max_unit_chars=effective_max_unit_chars)
            if path.suffix.lower() == ".json"
            else []
        )
        raw_units = record_units or structured_json_units or split_units(text, max_unit_chars=effective_max_unit_chars)
        document.metadata["record_delimited_jsonl"] = bool(record_units)
        document.metadata["record_unit_count"] = len(record_units)
        document.metadata["structurally_split_json"] = bool(structured_json_units)
        document.metadata["structured_json_unit_count"] = len(structured_json_units)
        packed_units = _pack_split_units(
            text,
            raw_units,
            max_pack_chars=pack_unit_chars,
            max_pack_units=pack_unit_count,
        )
        for order, (start, end, unit) in enumerate(packed_units):
            unit_hash = hashlib.sha256(unit.encode("utf-8", errors="replace")).hexdigest()
            sentences.append(
                Sentence(
                    sentence_id=_stable_scan_id("unit", document_id, order, start, end, unit_hash),
                    document_id=document_id,
                    rel_path=rel_path,
                    text=unit,
                    order=order,
                    char_start=start,
                    char_end=end,
                )
            )
    return documents, sentences
