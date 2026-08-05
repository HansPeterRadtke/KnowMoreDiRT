from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from knowmoredirt import ingest
from knowmoredirt.models import Sentence
from knowmoredirt.store import DSPGStore


SOURCE = "Alice owns Widget."


def _payload(*, subject: str = "Alice", object_value: str = "Widget", evidence: str = SOURCE) -> dict[str, Any]:
    return {
        "drs": {
            "schema_version": "chunk-drs-v5",
            "source_id": "doc.txt",
            "referents": [
                {"id": "r0", "label": subject, "kind": "person", "evidence_text": subject}
            ],
            "boxes": [
                {
                    "id": "b0",
                    "kind": "asserted",
                    "parent_id": "",
                    "holder_referent_id": "",
                    "evidence_text": evidence,
                }
            ],
            "conditions": [
                {
                    "id": "c0",
                    "predicate": "owns",
                    "box_id": "b0",
                    "polarity": "positive",
                    "modality": "asserted",
                    "temporal_id": "",
                    "evidence_text": evidence,
                    "arguments": [
                        {
                            "role": "owner",
                            "target_kind": "referent",
                            "target_id": "r0",
                            "value": "",
                            "value_type": "person",
                            "evidence_text": subject,
                        },
                        {
                            "role": "object",
                            "target_kind": "literal",
                            "target_id": "",
                            "value": object_value,
                            "value_type": "string",
                            "evidence_text": object_value,
                        },
                    ],
                }
            ],
            "identity_hypotheses": [],
            "temporal_records": [],
            "evidence_spans": [],
            "semantic_notes": [],
        }
    }


def _store_with_span(path: Path) -> tuple[DSPGStore, str, str, Sentence]:
    store = DSPGStore(path)
    run_id = store.start_run("/tmp/input")
    document_id = "doc"
    chunk_id = "chunk"
    span_id = "span"
    store.execute(
        "INSERT INTO documents(document_id,run_id,path,rel_path,content_hash,size_bytes,mtime,ctime,char_count,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (document_id, run_id, "/tmp/input/doc.txt", "doc.txt", "hash", len(SOURCE), 0, 0, len(SOURCE), "{}"),
    )
    store.execute(
        "INSERT INTO chunks(chunk_id,document_id,chunk_order,char_start,char_end,text,token_estimate) VALUES(?,?,?,?,?,?,?)",
        (chunk_id, document_id, 0, 0, len(SOURCE), SOURCE, 3),
    )
    store.execute(
        "INSERT INTO source_spans(span_id,document_id,chunk_id,char_start,char_end,surface,surface_norm,span_kind) VALUES(?,?,?,?,?,?,?,?)",
        (span_id, document_id, chunk_id, 0, len(SOURCE), SOURCE, SOURCE.lower(), "sentence"),
    )
    store.commit()
    return (
        store,
        run_id,
        span_id,
        Sentence("sentence", document_id, "doc.txt", SOURCE, 0, 0, len(SOURCE)),
    )


def _semantic_counts(store: DSPGStore) -> dict[str, int]:
    return {
        table: int(store.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in (
            "drs_referents",
            "drs_boxes",
            "drs_conditions",
            "drs_condition_arguments",
            "relations",
        )
    }


def test_store_rejects_cross_record_payload_without_inserting(tmp_path: Path) -> None:
    source = '[{"name":"Alice","status":"open"},{"name":"Bob","status":"closed"}]'
    store, run_id, span_id, _sentence = _store_with_span(tmp_path / "store.sqlite3")
    false_payload = {
        "drs": {
            **_payload()["drs"],
            "source_id": "records.json",
            "referents": [{"id": "r0", "label": "Alice", "kind": "person", "evidence_text": "Alice"}],
            "boxes": [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": source}],
            "conditions": [
                {
                    "id": "c0",
                    "predicate": "status",
                    "box_id": "b0",
                    "polarity": "positive",
                    "modality": "asserted",
                    "temporal_id": "",
                    "evidence_text": source,
                    "arguments": [
                        {"role": "subject", "target_kind": "referent", "target_id": "r0", "value": "", "value_type": "person", "evidence_text": "Alice"},
                        {"role": "status", "target_kind": "literal", "target_id": "", "value": "closed", "value_type": "state", "evidence_text": "closed"},
                    ],
                }
            ],
        }
    }

    result = store.materialize_drs_payload(run_id, span_id, source, false_payload)

    assert result["accepted"] is False
    assert result["reason"] == "grounding_validation_failed"
    assert _semantic_counts(store) == {table: 0 for table in _semantic_counts(store)}


def test_materialization_exception_rolls_back_all_partial_rows(tmp_path: Path) -> None:
    store, run_id, span_id, _sentence = _store_with_span(tmp_path / "store.sqlite3")
    store.execute(
        "CREATE TRIGGER abort_condition BEFORE INSERT ON drs_conditions BEGIN SELECT RAISE(ABORT,'forced condition failure'); END"
    )

    with pytest.raises(Exception, match="forced condition failure"):
        store.materialize_drs_payload(run_id, span_id, SOURCE, _payload())

    assert _semantic_counts(store) == {table: 0 for table in _semantic_counts(store)}
    store.finish_run(run_id, {"done": True})
    assert _semantic_counts(store) == {table: 0 for table in _semantic_counts(store)}


def test_failed_replacement_model_call_preserves_previous_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, run_id, span_id, sentence = _store_with_span(tmp_path / "store.sqlite3")
    first = store.materialize_drs_payload(run_id, span_id, SOURCE, _payload())
    assert first["accepted"] is True
    before = _semantic_counts(store)

    monkeypatch.setattr(ingest, "default_chunk_drs_n_predict", lambda *_args, **_kwargs: 128)
    monkeypatch.setattr(
        ingest,
        "chunk_drs_cache_context",
        lambda *_args, **_kwargs: {
            "source_text_hash": "hash",
            "n_predict": 128,
            "schema_version": "chunk-drs-v5",
            "model_fingerprint": {"model_id": "fake"},
        },
    )

    def fail_model(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("forced model failure")

    monkeypatch.setattr(ingest, "call_model_chunk_drs", fail_model)

    with pytest.raises(RuntimeError, match="forced model failure"):
        ingest._ingest_model_drs_for_sentence(
            store,
            run_id,
            sentence,
            span_id,
            object(),
            0,
            1,
            time.monotonic(),
        )

    assert _semantic_counts(store) == before
