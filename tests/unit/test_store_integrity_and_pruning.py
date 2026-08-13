from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from knowmoredirt.ingest import ingest_folder
from knowmoredirt.store import DSPGStore


def test_sqlite_foreign_keys_reject_orphan_semantic_reference_immediately(tmp_path: Path) -> None:
    store = DSPGStore(tmp_path / "store.sqlite3")
    store.start_run(tmp_path)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store.execute(
            "INSERT INTO frame_arguments(argument_id,frame_id,role,mention_id,referent_id,surface,value_type,confidence) VALUES(?,?,?,?,?,?,?,?)",
            ("orphan", "missing-frame", "value", None, None, "x", "string", 1.0),
        )
    assert store.semantic_integrity_errors() == []


def test_removed_file_rows_are_physically_pruned(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    first = source_root / "first.txt"
    removed = source_root / "removed.txt"
    first.write_text("First remains.", encoding="utf-8")
    removed.write_text("Deleted secret omega.", encoding="utf-8")
    store = DSPGStore(tmp_path / "store.sqlite3")

    store, first_run, first_documents, _ = ingest_folder(source_root, store=store)
    removed_document_id = next(
        document.document_id for document in first_documents if document.rel_path == "removed.txt"
    )
    removed.unlink()
    store, second_run, current_documents, _ = ingest_folder(source_root, store=store)

    assert first_run == second_run
    assert [document.rel_path for document in current_documents] == ["first.txt"]
    assert store.execute(
        "SELECT COUNT(*) FROM documents WHERE document_id=?",
        (removed_document_id,),
    ).fetchone()[0] == 0
    assert store.execute(
        "SELECT COUNT(*) FROM chunks WHERE text LIKE '%Deleted secret omega%'"
    ).fetchone()[0] == 0
    assert store.execute(
        "SELECT COUNT(*) FROM source_spans WHERE surface LIKE '%Deleted secret omega%'"
    ).fetchone()[0] == 0
    assert store.semantic_integrity_errors() == []


def test_changed_file_replaces_old_document_rows(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    path = source_root / "note.txt"
    path.write_text("Version one fact.", encoding="utf-8")
    store = DSPGStore(tmp_path / "store.sqlite3")

    store, run_id, first_documents, _ = ingest_folder(source_root, store=store)
    old_document_id = first_documents[0].document_id
    path.write_text("Version two fact changed.", encoding="utf-8")
    store, second_run_id, second_documents, _ = ingest_folder(source_root, store=store)

    assert run_id == second_run_id
    assert second_documents[0].document_id != old_document_id
    assert store.execute(
        "SELECT COUNT(*) FROM documents WHERE run_id=?",
        (run_id,),
    ).fetchone()[0] == 1
    assert store.execute(
        "SELECT COUNT(*) FROM chunks WHERE text='Version one fact.'"
    ).fetchone()[0] == 0
    assert store.execute(
        "SELECT COUNT(*) FROM chunks WHERE text='Version two fact changed.'"
    ).fetchone()[0] == 1
    assert store.semantic_integrity_errors() == []
