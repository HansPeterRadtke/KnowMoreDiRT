"""Raw folder scanning for KnowMoreDiRT."""

from __future__ import annotations

import hashlib
import mimetypes
import stat as stat_module
from pathlib import Path

from .models import Document, Sentence
from .text import split_units, tokenize


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


def scan_folder(folder_path: str | Path) -> tuple[list[Document], list[Sentence]]:
    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    documents: list[Document] = []
    sentences: list[Sentence] = []
    for document_index, path in enumerate(sorted(root.rglob("*"))):
        if not path.is_file():
            continue
        read_result = read_text_file_with_metadata(path)
        if read_result is None:
            continue
        text, read_metadata = read_result
        stat = path.stat()
        rel_path = path.relative_to(root).as_posix()
        document_id = f"d{document_index:05d}"
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
            sha256=hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
            metadata=metadata,
        )
        documents.append(document)
        for order, (start, end, unit) in enumerate(split_units(text)):
            sentences.append(
                Sentence(
                    sentence_id=f"{document_id}:s{order:04d}",
                    document_id=document_id,
                    rel_path=rel_path,
                    text=unit,
                    order=order,
                    char_start=start,
                    char_end=end,
                )
            )
    return documents, sentences
