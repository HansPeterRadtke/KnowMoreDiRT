"""Raw folder scanning for KnowMoreDiRT."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import stat as stat_module
from pathlib import Path

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


def read_text_file_with_metadata(path: Path) -> tuple[str, dict[str, object]] | None:
    """Read a file as raw text and return structural read metadata."""

    try:
        return path.read_text(encoding="utf-8"), {
            "encoding": "utf-8",
            "decode_errors": False,
            "read_mode": "strict_text",
        }
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="replace"), {
                "encoding": "utf-8",
                "decode_errors": True,
                "read_mode": "replacement_text",
            }
        except OSError:
            return None
    except OSError:
        return None


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _include_generated_cache_content() -> bool:
    return os.environ.get("KMD_SCAN_INCLUDE_GENERATED_CACHES", "").strip().lower() in {"1", "true", "yes", "on"}


def _configured_cache_roots(root: Path) -> list[Path]:
    if _include_generated_cache_content():
        return []
    values = [
        os.environ.get(name, "").strip()
        for name in KMD_CACHE_DIR_ENV_VARS
        if os.environ.get(name, "").strip()
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
        if not path.is_file():
            continue
        if cache_roots:
            try:
                resolved_path = path.resolve(strict=False)
            except OSError:
                resolved_path = path
            if any(_path_is_relative_to(resolved_path, cache_root) for cache_root in cache_roots):
                continue
        read_result = read_text_file_with_metadata(path)
        if read_result is None:
            continue
        text, read_metadata = read_result
        stat = path.stat()
        rel_path = path.relative_to(root).as_posix()
        content_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
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
            "symlink": path.is_symlink(),
            "symlink_target": str(path.readlink()) if path.is_symlink() else "",
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
        raw_units = split_units(text, max_unit_chars=max_unit_chars)
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
