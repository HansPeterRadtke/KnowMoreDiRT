from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from file_system_catalog.content_pipeline import sha256_text
from file_system_catalog.content_schema import CREATE_CHUNK_TABLE_SQL
from file_system_catalog.schema import CREATE_TABLE_SQL
from knowmoredirt.models import Sentence
from knowmoredirt.query import QueryFrame
from knowmoredirt.vector_retrieval import VectorCandidateRetriever, VectorChunkCandidate, VectorRetrievalUnavailable


class FakeEmbeddingClient:
    model = "qwen3-embedding-0.6b-q8"
    revision = "370f27d7550e0def9b39c1f16d3fbaa13aa67728:Q8_0"
    expected_dimension = 3

    def embed(self, texts):
        assert texts
        return [np.asarray([1.0, 0.0, 0.0], dtype="<f4") for _ in texts]

    def health(self):
        return {"status": "ok"}


def _catalog(path: Path, root: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(CREATE_TABLE_SQL)
        connection.execute(CREATE_CHUNK_TABLE_SQL)
        # Insert only columns needed by the retriever while honoring NOT NULLs.
        columns = [row[1] for row in connection.execute("pragma table_info(filesystem_entries)")]
        values = {name: "" for name in columns}
        values.update({
            "id": 1, "scan_id": "scan", "scanner_version": "test", "scanned_at_ns": 1,
            "scan_root_display": str(root.resolve()), "scan_root_b64": "", "relative_path_display": "doc.txt",
            "relative_path_b64": "", "relative_path_hex": "", "parent_path_display": "", "parent_path_b64": "",
            "name_display": "doc.txt", "name_b64": "", "name_hex": "", "path_depth": 1,
            "name_length_bytes": 7, "extensions_json": "[]", "is_hidden": 0, "entry_type": "file",
            "is_regular": 1, "is_directory": 0, "is_symlink": 0, "is_fifo": 0, "is_socket": 0,
            "is_block_device": 0, "is_character_device": 0, "is_other": 0, "xattr_count": 0,
            "xattr_names_json": "[]", "xattrs_b64_json": "{}", "acl_access_present": 0,
            "acl_default_present": 0, "hash_status": "complete", "content_sha256": "source",
            "hash_bytes_read": 10, "embedded_metadata_status": "not_requested",
            "metadata_extraction_status": "not_requested", "metadata_parser_attempts_json": "[]",
            "raw_metadata_json": "{}", "raw_metadata_source_count": 0, "exiftool_status": "not_requested",
            "archive_metadata_json": "{}", "content_sample_b64": "", "content_sample_truncated": 0,
            "errors_json": "[]",
        })
        placeholders = ",".join("?" for _ in columns)
        connection.execute(f"insert into filesystem_entries ({','.join(columns)}) values ({placeholders})", [values[c] for c in columns])
        vec = np.asarray([1.0, 0.0, 0.0], dtype="<f4")
        ccols = [row[1] for row in connection.execute("pragma table_info(content_chunks)")]
        cvals = {
            "chunk_id": "c1", "file_id": "f1", "collection_id": "col", "filesystem_entry_id": 1,
            "content_object_id": "obj", "content_sha256": "source", "chunk_kind": "chunk", "chunk_index": 0,
            "start_char": 5, "end_char": 15, "character_count": 10, "word_count": 2, "token_count": 2,
            "text_sha256": sha256_text("target text"), "embedding_model": FakeEmbeddingClient.model,
            "embedding_model_revision": FakeEmbeddingClient.revision, "embedding_dimension": 3,
            "embedding_dtype": "float32", "embedding_norm": 1.0, "embedding_blob": sqlite3.Binary(vec.tobytes(order="C")),
            "embedding_sha256": sha256_text(vec.tobytes().hex()), "created_at_ns": 1, "updated_at_ns": 1,
        }
        connection.execute(f"insert into content_chunks ({','.join(ccols)}) values ({','.join('?' for _ in ccols)})", [cvals.get(c) for c in ccols])
        connection.commit()
    finally:
        connection.close()


def test_vector_retriever_validates_root_and_ranks_chunk(tmp_path: Path) -> None:
    root = tmp_path / "raw"; root.mkdir()
    db = tmp_path / "catalog.sqlite3"; _catalog(db, root)
    retriever = VectorCandidateRetriever(root, db, FakeEmbeddingClient(), required=True)
    hits = retriever.search("target", limit=2)
    assert [(h.rel_path, h.start_char, h.end_char) for h in hits] == [("doc.txt", 5, 15)]
    assert hits[0].score == pytest.approx(1.0)


def test_vector_retriever_rejects_catalog_for_different_root(tmp_path: Path) -> None:
    root = tmp_path / "raw"; root.mkdir()
    db = tmp_path / "catalog.sqlite3"; _catalog(db, root)
    with pytest.raises(VectorRetrievalUnavailable):
        VectorCandidateRetriever(tmp_path / "other", db, FakeEmbeddingClient(), required=True)


def test_engine_vector_mapping_uses_source_offsets() -> None:
    from knowmoredirt.engine import KnowMoreDiRTEngine
    from knowmoredirt.model_planner import ModelQueryTrace

    class FakeRetriever:
        def search(self, query, *, limit):
            assert "traffic" in query.lower()
            return [VectorChunkCandidate("doc.txt", 8, 20, 0.9, "vc1")]

    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    s1 = Sentence("s1", "d", "doc.txt", "first", 0, 0, 7)
    s2 = Sentence("s2", "d", "doc.txt", "traffic law", 1, 8, 25)
    engine._vector_retriever = FakeRetriever()
    engine._sentences_by_document = {"doc.txt": {0: s1, 1: s2}}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(question_text="traffic law", answer_type="content_phrase", answer_variables=(), target_anchors=("traffic",), requested_relation="law", relation_terms=("law",), constraints=())
    hits = engine._vector_bounded_candidates("traffic law", frame, limit=4)
    assert hits[0][0] is s2
    assert engine.model_query_trace.vector_candidate_count == 1


def test_vector_retriever_from_environment_uses_filesystem_config_names(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "raw"; root.mkdir()
    db = tmp_path / "catalog.sqlite3"; _catalog(db, root)
    monkeypatch.setenv("KMD_VECTOR_RETRIEVAL_MODE", "optional")
    monkeypatch.setenv("KMD_FILESYSTEM_DATABASE", str(db))
    monkeypatch.setenv("KMD_EMBEDDING_MODEL", FakeEmbeddingClient.model)
    monkeypatch.setenv("KMD_EMBEDDING_REVISION", FakeEmbeddingClient.revision)
    from knowmoredirt import vector_retrieval as module
    monkeypatch.setattr(module, "EmbeddingClient", lambda *args, **kwargs: FakeEmbeddingClient())
    retriever = module.VectorCandidateRetriever.from_environment(root)
    assert retriever is not None
    assert retriever.client.revision == FakeEmbeddingClient.revision


def test_optional_vector_retrieval_degrades_on_stale_catalog(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "raw"; root.mkdir()
    other = tmp_path / "other"; other.mkdir()
    db = tmp_path / "catalog.sqlite3"; _catalog(db, other)
    monkeypatch.setenv("KMD_VECTOR_RETRIEVAL_MODE", "optional")
    monkeypatch.setenv("KMD_FILESYSTEM_DATABASE", str(db))
    monkeypatch.setenv("KMD_EMBEDDING_MODEL", FakeEmbeddingClient.model)
    monkeypatch.setenv("KMD_EMBEDDING_REVISION", FakeEmbeddingClient.revision)
    from knowmoredirt import vector_retrieval as module
    monkeypatch.setattr(module, "EmbeddingClient", lambda *args, **kwargs: FakeEmbeddingClient())
    assert module.VectorCandidateRetriever.from_environment(root) is None


def test_vector_catalog_embeddings_are_loaded_once(tmp_path: Path) -> None:
    root = tmp_path / "raw"; root.mkdir()
    db = tmp_path / "catalog.sqlite3"; _catalog(db, root)
    retriever = VectorCandidateRetriever(root, db, FakeEmbeddingClient(), required=True)
    assert len(retriever._validated_rows) == 1
    db.unlink()
    # Search still works because the validated immutable candidate vectors were
    # loaded once for this engine, rather than reread for every question.
    assert retriever.search("target", limit=1)[0].chunk_id == "c1"


def test_vector_query_results_are_cached_within_engine(tmp_path: Path) -> None:
    root = tmp_path / "raw"; root.mkdir()
    db = tmp_path / "catalog.sqlite3"; _catalog(db, root)

    class CountingClient(FakeEmbeddingClient):
        def __init__(self): self.calls = 0
        def embed(self, texts):
            self.calls += 1
            return super().embed(texts)

    client = CountingClient()
    retriever = VectorCandidateRetriever(root, db, client, required=True)
    assert retriever.search("target", limit=2)
    assert retriever.search("target", limit=2)
    assert client.calls == 1
