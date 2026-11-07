"""Find original documents through *all* recursively generated KMD summaries.

The original filesystem entry remains the only document identity. All summary
levels can contribute candidates, and several summary nodes per document are
returned (no one-highest-passage loss). The original source is the answer
authority; summaries only help locate it.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import sqlite3
from typing import Callable

import numpy as np
from .content_pipeline import cosine_similarity, vector_from_blob
from .hierarchical_summary import _VERSION


def search_summary_entries(
    db: sqlite3.Connection,
    query_vector: np.ndarray,
    *,
    limit: int = 10,
    nodes_per_document: int = 5,
    expected_model: str | None = None,
    require_source: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Read-only hierarchical summary search grouped by original-file identity.

    require_source is an optional caller-supplied validation callback that must
    verify current original/derived source versions for **every returned** node
    before a user-facing result is accepted. An exception from it fails closed.
    """
    if not 1 <= limit <= 100 or not 1 <= nodes_per_document <= 30:
        raise ValueError("invalid bounded summary-search results")
    vector = np.asarray(query_vector, dtype="<f4")
    if vector.shape != (384,) or not np.isfinite(vector).all():
        raise ValueError("invalid 384-dimensional query vector")
    rows = db.execute(
        """SELECT
             f.id, f.relative_path_display, f.size_bytes,
             c.chunk_id, c.start_char, c.end_char, c.text_sha256,
             c.content_sha256, c.collection_id,
             r.representation_id, r.representation_kind, r.representation_text,
             r.representation_text_sha256, r.generation_json,
             r.embedding_model, r.embedding_dimension, r.embedding_dtype,
             r.embedding_blob, r.embedding_sha256
           FROM content_representations AS r
           JOIN content_chunks AS c ON c.chunk_id = r.chunk_id
           JOIN filesystem_entries AS f ON f.id = c.filesystem_entry_id
           WHERE c.collection_id LIKE '%:summary'
             AND r.prompt_version = ?
             AND r.analysis_status = 'complete'
             AND f.entry_type = 'file'""",
        (_VERSION,),
    )
    indexed: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        (
            filesystem_entry_id, path, source_bytes,
            chunk_id, start, end, source_passage_sha,
            original_sha, collection_id,
            representation_id, kind, text, summary_sha,
            provenance_raw, model_name, dimension, dtype, blob, blob_sha,
        ) = r
        if expected_model and model_name != expected_model:
            continue
        if dimension != 384 or dtype not in {"float16", "float32"} or not blob:
            continue
        if hashlib.sha256(bytes(blob)).hexdigest() != blob_sha:
            continue
        if not isinstance(text, str) or hashlib.sha256(text.encode("utf-8")).hexdigest() != summary_sha:
            continue
        try:
            provenance = json.loads(provenance_raw)
        except (ValueError, TypeError):
            continue
        if (
            not isinstance(provenance, dict)
            or provenance.get("source_start_char") != start
            or provenance.get("source_end_char") != end
            or provenance.get("source_passage_sha256") != source_passage_sha
            or provenance.get("summary_sha256") != summary_sha
            or provenance.get("original_sha256") != original_sha
            or not isinstance(provenance.get("derived_sha256"), str)
            or len(provenance["derived_sha256"]) != 64
            or not isinstance(provenance.get("summary_level"), int)
            or not isinstance(provenance.get("node_id"), str)
            or kind not in {"summary", "summary_long"}
            or (kind == "summary_long" and provenance.get("summary_level", -1) < 0)
        ):
            continue
        try:
            value = vector_from_blob(bytes(blob), 384, dtype)
            score = cosine_similarity(vector, value)
        except (ValueError, TypeError):
            continue
        if not math.isfinite(score):
            continue
        candidate = {
            "filesystem_entry_id": filesystem_entry_id,
            "original_path": path,
            "original_bytes": source_bytes,
            "original_sha256": original_sha,
            "derived_sha256": provenance["derived_sha256"],
            "summary_level": provenance["summary_level"],
            "summary_node_id": provenance["node_id"],
            "summary_kind": kind,
            "summary": text,
            "summary_sha256": summary_sha,
            "source_section": provenance.get("source_label", ""),
            "source_start_char": start,
            "source_end_char": end,
            "source_passage_sha256": source_passage_sha,
            "source_scope": "summary_candidate_not_original_evidence",
            "similarity": float(score),
            "native_chunk_id": chunk_id,
            "native_representation_id": representation_id,
            "children": provenance.get("children", []),
        }
        if require_source is not None and not require_source(candidate):
            continue
        indexed[filesystem_entry_id].append(candidate)

    documents = []
    for filesystem_id, hits in indexed.items():
        hits.sort(key=lambda x: (-x["similarity"], -x["summary_level"], x["source_start_char"], x["summary_node_id"]))
        best = hits[0]
        documents.append({
            "filesystem_entry_id": filesystem_id,
            "original_path": best["original_path"],
            "original_bytes": best["original_bytes"],
            "original_sha256": best["original_sha256"],
            "similarity": best["similarity"],
            "best_summary": best["summary"],
            "best_level": best["summary_level"],
            "matched_summary_nodes": hits[:nodes_per_document],
            "indexed_summary_nodes_for_document": len(hits),
            "source_scope": "retrieval_candidate_only_verify_original_before_claims",
        })
    documents.sort(key=lambda x: (-x["similarity"], x["original_path"]))
    return documents[:limit]


def summary_index_status(db: sqlite3.Connection) -> dict:
    """Read-only native database summary counts and provenance dimensions."""
    row = db.execute(
        """SELECT COUNT(*), COUNT(DISTINCT c.filesystem_entry_id),
                  MIN(json_extract(r.generation_json,'$.summary_level')),
                  MAX(json_extract(r.generation_json,'$.summary_level'))
           FROM content_representations r
           JOIN content_chunks c ON c.chunk_id = r.chunk_id
           WHERE c.collection_id LIKE '%:summary'
             AND r.prompt_version = ?""",
        (_VERSION,),
    ).fetchone()
    levels = {
        str(level): count
        for level, count in db.execute(
            """SELECT json_extract(r.generation_json,'$.summary_level'), COUNT(*)
               FROM content_representations r
               JOIN content_chunks c ON c.chunk_id=r.chunk_id
               WHERE c.collection_id LIKE '%:summary'
                 AND r.prompt_version=?
               GROUP BY json_extract(r.generation_json,'$.summary_level')
               ORDER BY 1""",
            (_VERSION,),
        )
    }
    return {
        "native_summary_vectors": row[0],
        "original_documents_with_summaries": row[1],
        "minimum_summary_level": row[2],
        "maximum_summary_level": row[3],
        "vectors_by_summary_level": levels,
    }
