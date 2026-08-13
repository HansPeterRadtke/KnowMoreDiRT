"""Embedding-backed candidate retrieval for the authoritative KMD DRT engine.

Vector similarity is intentionally retrieval-only.  It may add source chunks to
the bounded candidate set, but it never creates facts or bypasses DRT/DSPG
scope, temporal, identity, provenance, or answer verification.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kmd_runtime_config import floating as _config_float, integer as _config_int, text as _config_text

from file_system_catalog.content_pipeline import EmbeddingClient, cosine_similarity, vector_from_blob
from file_system_catalog.content_schema import CHUNK_TABLE_NAME
from file_system_catalog.schema import TABLE_NAME

from .filesystem import FilesystemModelConfig


class VectorRetrievalUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorChunkCandidate:
    rel_path: str
    start_char: int
    end_char: int
    score: float
    chunk_id: str


def _mode() -> str:
    value = _config_text("KMD_VECTOR_RETRIEVAL_MODE").strip().lower() or "optional"
    if value not in {"off", "optional", "required"}:
        raise ValueError("KMD_VECTOR_RETRIEVAL_MODE must be off, optional, or required")
    return value


def _minimum_similarity() -> float:
    value = _config_float("KMD_VECTOR_MIN_SIMILARITY")
    if value < -1.0 or value > 1.0:
        raise ValueError("KMD_VECTOR_MIN_SIMILARITY must be between -1 and 1")
    return value


def _result_multiplier() -> int:
    value = _config_int("KMD_VECTOR_RESULT_MULTIPLIER")
    if value < 1:
        raise ValueError("KMD_VECTOR_RESULT_MULTIPLIER must be positive")
    return value


class VectorCandidateRetriever:
    def __init__(self, root: Path, database: Path, client: EmbeddingClient, *, required: bool) -> None:
        self.root = root.resolve()
        self.database = database.resolve()
        self.client = client
        self.required = bool(required)
        self.minimum_similarity = _minimum_similarity()
        self.result_multiplier = _result_multiplier()
        self._validated_rows = self._validate_catalog_and_load()
        self._search_cache: dict[tuple[str, int], tuple[VectorChunkCandidate, ...]] = {}

    @classmethod
    def from_environment(cls, root: str | Path) -> "VectorCandidateRetriever | None":
        mode = _mode()
        if mode == "off":
            return None
        database_text = _config_text("KMD_FILESYSTEM_DATABASE").strip()
        if not database_text:
            if mode == "required":
                raise VectorRetrievalUnavailable(
                    "KMD vector retrieval is required but KMD_FILESYSTEM_DATABASE is not configured"
                )
            return None
        config = FilesystemModelConfig.from_environment()
        client = EmbeddingClient(
            config.embedding_url,
            model=config.embedding_model,
            revision=config.embedding_revision,
            expected_dimension=1024,
            batch_size=config.embedding_batch_size,
            max_batch_characters=config.embedding_max_batch_characters,
        )
        try:
            retriever = cls(Path(root), Path(database_text), client, required=mode == "required")
        except VectorRetrievalUnavailable:
            if mode == "required":
                raise
            return None
        if mode == "required":
            try:
                client.health()
            except Exception as error:
                raise VectorRetrievalUnavailable(f"required embedding endpoint is unavailable: {error}") from error
        return retriever

    def _validate_catalog_and_load(self) -> tuple[tuple[str, str, int, int, Any], ...]:
        if not self.database.is_file():
            raise VectorRetrievalUnavailable(f"filesystem vector catalog does not exist: {self.database}")
        connection = sqlite3.connect(self.database, timeout=30.0)
        try:
            roots = {
                str(row[0])
                for row in connection.execute(
                    f"SELECT DISTINCT scan_root_display FROM {TABLE_NAME} WHERE scan_root_display IS NOT NULL"
                )
            }
            if roots != {str(self.root)}:
                raise VectorRetrievalUnavailable(
                    f"filesystem vector catalog root differs: expected={self.root!s} actual={sorted(roots)!r}"
                )
            rows = connection.execute(
                f"""
                SELECT DISTINCT embedding_model, embedding_model_revision, embedding_dimension
                FROM {CHUNK_TABLE_NAME}
                WHERE chunk_kind='chunk' AND embedding_blob IS NOT NULL
                """
            ).fetchall()
            expected = (self.client.model, self.client.revision, int(self.client.expected_dimension))
            actual = {(str(row[0] or ""), str(row[1] or ""), int(row[2] or 0)) for row in rows}
            if actual != {expected}:
                raise VectorRetrievalUnavailable(
                    f"filesystem embedding contract differs: expected={expected!r} actual={sorted(actual)!r}"
                )
            loaded: list[tuple[str, str, int, int, Any]] = []
            for row in connection.execute(
                f"""
                SELECT c.chunk_id, f.relative_path_display, c.start_char, c.end_char,
                       c.embedding_dimension, c.embedding_dtype, c.embedding_blob
                FROM {CHUNK_TABLE_NAME} c
                JOIN {TABLE_NAME} f ON f.id=c.filesystem_entry_id
                WHERE c.chunk_kind='chunk' AND c.embedding_blob IS NOT NULL
                ORDER BY f.relative_path_display, c.start_char, c.end_char, c.chunk_id
                """
            ):
                loaded.append(
                    (
                        str(row[0]),
                        str(row[1]),
                        int(row[2]),
                        int(row[3]),
                        vector_from_blob(row[6], int(row[4]), str(row[5])),
                    )
                )
            return tuple(loaded)
        finally:
            connection.close()

    def search(self, query: str, *, limit: int) -> list[VectorChunkCandidate]:
        if limit <= 0 or not query.strip():
            return []
        cache_key = (query, int(limit))
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        try:
            query_vector = self.client.embed([query])[0]
        except Exception as error:
            if self.required:
                raise VectorRetrievalUnavailable(f"required query embedding failed: {error}") from error
            return []
        candidates: list[VectorChunkCandidate] = []
        for chunk_id, rel_path, start_char, end_char, vector in self._validated_rows:
            score = cosine_similarity(query_vector, vector)
            if score < self.minimum_similarity:
                continue
            candidates.append(
                VectorChunkCandidate(
                    rel_path=rel_path,
                    start_char=start_char,
                    end_char=end_char,
                    score=float(score),
                    chunk_id=chunk_id,
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.rel_path, item.start_char, item.end_char))
        result = tuple(candidates[: max(limit, limit * self.result_multiplier)])
        self._search_cache[cache_key] = result
        return list(result)
