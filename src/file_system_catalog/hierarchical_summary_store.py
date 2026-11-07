"""Append source-linked hierarchical summary vectors to the ORIGINAL KMD FileDB.

Opt-in only. Uses the existing native content_chunks/content_representations
tables, not a new database or alternate source of truth. The caller must supply
the exact original completed-source revision and the current derived-text file
already independently verified against the source-map and completion ledger.

This is a building block; the existing Infra result resolver is not yet aware
of the summary collection, so do not enable on the production FileDB before a
verified query-resolution integration is deployed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np

from .content_pipeline import (
    stable_file_id, stable_chunk_id, stable_representation_id, count_words,
)
from .content_schema import (
    CHUNK_COLUMN_NAMES, REPRESENTATION_COLUMN_NAMES,
    CHUNK_TABLE_NAME, REPRESENTATION_TABLE_NAME,
)
from .hierarchical_summary import SummaryTree, _VERSION


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def summary_rows(
    connection: sqlite3.Connection,
    *,
    original_path: str,
    original_sha256: str,
    original_bytes: int,
    derived_sha256: str,
    derived_text: str,
    summary_start_char: int,
    tree: SummaryTree,
    embedding_client,
    analysis_model: str,
    collection_id: str,
    summary_generation_seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Prepare native summary rows without writing anything to FileDB."""
    if (
        not isinstance(original_path, str)
        or not original_path
        or original_path.startswith("/")
        or original_path.startswith("./")
        or ".." in Path(original_path).parts
        or len(original_path) > 1500
        or type(original_bytes) is not int
        or original_bytes < 0
        or not isinstance(collection_id, str)
        or not collection_id.strip()
        or not isinstance(analysis_model, str)
        or not analysis_model.strip()
    ):
        raise ValueError("invalid original identity or model information")
    for val in (original_sha256, derived_sha256):
        if not isinstance(val, str) or len(val) != 64 or not all(c in "0123456789abcdef" for c in val):
            raise ValueError("invalid source revision identifier")
    if _sha(derived_text) != derived_sha256:
        raise ValueError("derived text SHA differs from source map")
    root = tree.root
    if not isinstance(derived_text, str) or type(summary_start_char) is not int:
        raise ValueError("invalid source text or original offset")
    if tree.source_sha256 != _sha(derived_text[summary_start_char:summary_start_char + tree.original_characters]):
        raise ValueError("summary tree source SHA differs from original derived span")
    if (
        type(summary_start_char) is not int
        or summary_start_char < 0
        or root.start_char != 0
        or root.end_char != tree.original_characters
        or summary_start_char + tree.original_characters > len(derived_text)
    ):
        raise ValueError("summary root is not a contiguous subset of the current derived text")
    row = connection.execute(
        "SELECT id,relative_path_b64,size_bytes FROM filesystem_entries "
        "WHERE entry_type='file' AND relative_path_display=?",
        (original_path,),
    ).fetchone()
    if row is None:
        raise ValueError("original not admitted to canonical KMD FileDB")
    native_id, path_b64, actual_bytes = row
    if actual_bytes != original_bytes:
        raise ValueError("native source size differs from original completion ledger")
    if base64.b64decode(path_b64) != original_path.encode("utf-8"):
        raise ValueError("original KMD path bytes differ from exact relative path")
    seen = set()
    source_chunks = []
    representations = []
    file_id = stable_file_id(collection_id, path_b64)
    now = time.time_ns()
    ordered = sorted(tree.nodes, key=lambda n: (n.level, n.start_char, n.end_char, n.node_id))
    if not ordered or len(ordered) > 50_000:
        raise ValueError("empty or unbounded summary nodes")
    vectors = embedding_client.embed([n.summary for n in ordered])
    if len(vectors) != len(ordered):
        raise RuntimeError("one embedding is required for every summary node")
    if getattr(embedding_client, "embedding_dimension", None) != 384:
        raise ValueError("expected verified 384-dimensional native embedding model")
    if not getattr(embedding_client, "model", "") or not getattr(embedding_client, "revision", ""):
        raise ValueError("missing embedding model identity and revision")
    for ordinal, (node, vector) in enumerate(zip(ordered, vectors, strict=True)):
        if (
            node.source_sha256 != tree.source_sha256
            or node.prompt_version != _VERSION
            or node.summary_sha256 != _sha(node.summary)
            or node.start_char < 0
            or node.end_char <= node.start_char
            or node.end_char > tree.original_characters
            or node.node_id in seen
        ):
            raise ValueError("summary node source provenance is missing or inconsistent")
        seen.add(node.node_id)
        vector = np.asarray(vector, dtype="<f4")
        if vector.shape != (384,) or not np.isfinite(vector).all():
            raise ValueError("invalid summary vector")
        norm = float(np.linalg.norm(vector))
        if norm < 0.99 or norm > 1.01:
            raise ValueError("summary embedding should already be normalized")
        source_start = summary_start_char + node.start_char
        source_end = summary_start_char + node.end_char
        direct_source = derived_text[source_start:source_end]
        source_hash = _sha(direct_source)
        chunk_index = 3_000_000 + node.level * 100_000 + ordinal
        chunk_id = stable_chunk_id(
            file_id, original_sha256, "chunk", chunk_index, source_start, source_end, source_hash,
        )
        c = {
            "chunk_id": chunk_id,
            "file_id": file_id,
            "collection_id": collection_id + ":summary",
            "filesystem_entry_id": native_id,
            "content_object_id": "sha256:" + derived_sha256,
            "content_sha256": original_sha256,
            "chunk_kind": "chunk",
            "chunk_index": chunk_index,
            "start_char": source_start,
            "end_char": source_end,
            "character_count": len(direct_source),
            "word_count": count_words(direct_source),
            "token_count": node.prompt_tokens,
            "text_sha256": source_hash,
            "embedding_model": None,
            "embedding_model_revision": None,
            "embedding_dimension": None,
            "embedding_dtype": None,
            "embedding_norm": None,
            "embedding_blob": None,
            "embedding_sha256": None,
            "created_at_ns": now,
            "updated_at_ns": now,
        }
        source_chunks.append(c)
        # Half precision is supported by KMD native vector decoding and
        # reduces stored source-summary representations to 768 vector bytes.
        compact = np.asarray(vector, dtype="<f2").tobytes()
        compact_sha = hashlib.sha256(compact).hexdigest()
        representation_id = stable_representation_id(
            chunk_id, analysis_model, _VERSION, embedding_client.model, 0,
        )
        provenance = {
            "source_start_char": source_start,
            "source_end_char": source_end,
            "original_sha256": original_sha256,
            "derived_sha256": derived_sha256,
            "source_passage_sha256": source_hash,
            "source_tree_sha256": tree.source_sha256,
            "summary_sha256": node.summary_sha256,
            "summary_level": node.level,
            "source_label": node.source_label,
            "node_id": node.node_id,
            "children": list(node.child_ids),
            "prompt_tokens": node.prompt_tokens,
            "configured_model_context": node.model_context_tokens,
            "reserved_output_tokens": node.allocated_output_tokens,
            "source_scope": "retrieval_summary_only_not_original_evidence",
        }
        representations.append({
            "representation_id": representation_id,
            "chunk_id": chunk_id,
            "representation_kind": "summary_long" if node.node_id == tree.root_id else "summary",
            "facet_label": "hierarchical_source_summary",
            "facet_strength": "moderate",
            "item_strength": "moderate",
            "facet_rank": 0,
            "item_rank": 0,
            "global_rank": 0,
            "representation_text": node.summary,
            "representation_text_sha256": node.summary_sha256,
            "analysis_model": analysis_model,
            "analysis_model_fingerprint": None,
            "prompt_version": _VERSION,
            "generation_seed": summary_generation_seed,
            "pipeline_version": _VERSION,
            "generation_json": json.dumps(provenance, ensure_ascii=False, sort_keys=True),
            "attributes_json": json.dumps({"source_label": node.source_label,
                                            "summary_node_id": node.node_id},
                                           ensure_ascii=False, sort_keys=True),
            "embedding_model": embedding_client.model,
            "embedding_model_revision": embedding_client.revision,
            "embedding_dimension": 384,
            "embedding_dtype": "float16",
            "embedding_norm": float(np.linalg.norm(np.frombuffer(compact, dtype="<f2").astype("<f4"))),
            "embedding_blob": sqlite3.Binary(compact),
            "embedding_sha256": compact_sha,
            "analysis_status": "complete",
            "analysis_error": None,
            "created_at_ns": now,
            "updated_at_ns": now,
        })
    ids = {n.node_id for n in tree.nodes}
    if any(child not in ids for node in tree.nodes for child in node.child_ids):
        raise ValueError("summary tree has unknown child node")
    return source_chunks, representations


def append_summary_rows(
    connection: sqlite3.Connection,
    *,
    chunk_rows: list[dict],
    representation_rows: list[dict],
    max_database_bytes: int = 640 * 1024 * 1024,
) -> dict:
    """Guarded, idempotent transaction. Production resolver must be upgraded first."""
    if not chunk_rows or len(chunk_rows) != len(representation_rows):
        raise ValueError("summary row count mismatch")
    if not 1024 * 1024 <= max_database_bytes <= 16 * 1024 * 1024 * 1024:
        raise ValueError("invalid FileDB capacity limit")
    if not connection.execute("PRAGMA foreign_keys").fetchone()[0]:
        raise RuntimeError("foreign keys must be enabled for native KMD writes")
    ccols = ",".join('"' + col + '"' for col in CHUNK_COLUMN_NAMES)
    rcols = ",".join('"' + col + '"' for col in REPRESENTATION_COLUMN_NAMES)
    cinsert = f"INSERT OR IGNORE INTO {CHUNK_TABLE_NAME} ({ccols}) VALUES (" + ",".join("?" for _ in CHUNK_COLUMN_NAMES) + ")"
    rinsert = f"INSERT OR IGNORE INTO {REPRESENTATION_TABLE_NAME} ({rcols}) VALUES (" + ",".join("?" for _ in REPRESENTATION_COLUMN_NAMES) + ")"
    before = connection.total_changes
    connection.execute("BEGIN IMMEDIATE")
    try:
        for chunk, representation in zip(chunk_rows, representation_rows, strict=True):
            connection.execute(cinsert, [chunk[c] for c in CHUNK_COLUMN_NAMES])
            native = connection.execute(
                f"SELECT text_sha256,content_sha256,collection_id FROM {CHUNK_TABLE_NAME} WHERE chunk_id=?",
                (chunk["chunk_id"],),
            ).fetchone()
            if native is None or tuple(native) != (chunk["text_sha256"], chunk["content_sha256"], chunk["collection_id"]):
                raise RuntimeError("conflicting native KMD source revision for summary anchor")
            connection.execute(rinsert, [representation[c] for c in REPRESENTATION_COLUMN_NAMES])
            saved = connection.execute(
                f"SELECT representation_text_sha256,embedding_sha256,prompt_version "
                f"FROM {REPRESENTATION_TABLE_NAME} WHERE representation_id=?",
                (representation["representation_id"],),
            ).fetchone()
            if saved is None or tuple(saved) != (representation["representation_text_sha256"],
                         representation["embedding_sha256"], representation["prompt_version"]):
                raise RuntimeError("conflicting native KMD summary/embedding revision")
        estimated_size = connection.execute("PRAGMA page_count").fetchone()[0] * connection.execute("PRAGMA page_size").fetchone()[0]
        if estimated_size > max_database_bytes:
            raise RuntimeError("original KMD FileDB storage checkpoint would be exceeded")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("native summary insertion violated KMD foreign key constraints")
        added = connection.total_changes - before
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return {"new_rows": added, "total_summary_nodes_requested": len(chunk_rows), "filedb_estimated_bytes": estimated_size}
