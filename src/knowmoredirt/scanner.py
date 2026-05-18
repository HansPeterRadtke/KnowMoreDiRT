"""Raw folder scanning for KnowMoreDiRT."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import Document, Sentence
from .text import split_units


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
        text = read_text_file(path)
        if text is None:
            continue
        stat = path.stat()
        rel_path = path.relative_to(root).as_posix()
        document_id = f"d{document_index:05d}"
        document = Document(
            document_id=document_id,
            path=path,
            rel_path=rel_path,
            text=text,
            size_bytes=stat.st_size,
            mtime=stat.st_mtime,
            ctime=stat.st_ctime,
            sha256=hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
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

