"""Batched, read-only native KMD vector scan for document discovery.

Only the canonical FileDB is accessed. Large source summaries / JSON
provenance are fetched lazily for the highest-scoring original documents,
rather than materialized for every embedding during every search.

A small bounded float32 matrix scan preserves full cosine-distance
semantics (including vector-length normalization) and validates each
stored raw vector SHA before it contributes to any rank. No index,
cache DB, new native table or persistent file is created.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import sqlite3
from typing import Callable

import numpy as np

SUMMARY_PROMPT = "kmd-hierarchical-source-summary-v1"


def _scan_vectors(
    db: sqlite3.Connection,
    query_vector: np.ndarray,
    *,
    expected_model: str | None,
    batch_size: int,
    include_facets: bool = False,
) -> dict[int, list[tuple[float, str, str]]]:
    if not 64 <= batch_size <= 8192:
        raise ValueError("invalid bounded vector batch")
    query = np.asarray(query_vector, dtype="<f4")
    if query.shape != (384,) or not np.isfinite(query).all():
        raise ValueError("invalid 384-dimensional query vector")
    query_norm = float(np.linalg.norm(query))
    by_doc: dict[int, list[tuple[float, str, str]]] = defaultdict(list)
    sql = """
        SELECT f.id, f.relative_path_display, r.representation_id,
               r.embedding_model, r.embedding_dimension, r.embedding_dtype,
               r.embedding_blob, r.embedding_sha256
        FROM content_representations r
        JOIN content_chunks c ON c.chunk_id=r.chunk_id
        JOIN filesystem_entries f ON f.id=c.filesystem_entry_id
        WHERE c.collection_id LIKE '%:summary'
          AND r.prompt_version=?
          AND r.analysis_status='complete'
          AND f.entry_type='file'
    """
    sql += ("" if include_facets else
            " AND r.representation_kind IN ('summary','summary_long')")
    batch = []
    embeddings = []

    def flush():
        if not batch:
            return
        # float16-decoding is performed per row, but dot products and norms
        # are computed on a single contiguous matrix instead of hundreds of
        # thousands of separate NumPy function invocations.
        values = np.stack(embeddings, axis=0)
        norms = np.sqrt(np.sum(values * values, axis=1, dtype=np.float64))
        dots = np.einsum("ij,j->i", values, query, optimize=False, dtype=np.float64)
        for item, dot, norm in zip(batch, dots, norms):
            if norm > 0 and query_norm > 0:
                score = float(dot / (norm * query_norm))
            else:
                score = 0.0
            if math.isfinite(score):
                entry_id, path, rep_id = item
                by_doc[entry_id].append((score, path, rep_id))
        batch.clear()
        embeddings.clear()

    for row in db.execute(sql, (SUMMARY_PROMPT,)):
        entry_id, path, representation_id, model, dimension, dtype, blob, raw_sha = row
        if expected_model and model != expected_model:
            continue
        if dimension != 384 or dtype not in ("float16", "float32") or blob is None:
            continue
        raw = bytes(blob)
        if hashlib.sha256(raw).hexdigest() != raw_sha:
            continue
        arr = np.frombuffer(raw, dtype="<f2" if dtype == "float16" else "<f4").astype("<f4")
        if arr.shape != (384,) or not np.isfinite(arr).all():
            continue
        batch.append((int(entry_id), str(path), str(representation_id)))
        embeddings.append(arr)
        if len(batch) == batch_size:
            flush()
    flush()
    return by_doc


def _validated_summary_node(db, rep_id, similarity: float, *,
                            include_facets: bool = False) -> dict | None:
    sql = """
        SELECT f.id, f.relative_path_display, f.size_bytes,
               c.chunk_id, c.start_char, c.end_char, c.text_sha256,
               c.content_sha256, r.representation_id,
               r.representation_kind, r.representation_text,
               r.representation_text_sha256, r.generation_json
        FROM content_representations r
        JOIN content_chunks c ON c.chunk_id=r.chunk_id
        JOIN filesystem_entries f ON f.id=c.filesystem_entry_id
        WHERE r.representation_id=?
          AND c.collection_id LIKE '%:summary'
          AND r.prompt_version=?
          AND r.analysis_status='complete'
          AND f.entry_type='file'
    """
    r = db.execute(sql, (rep_id, SUMMARY_PROMPT)).fetchone()
    if not r:
        return None
    (entry_id,path,source_bytes,chunk_id,start,end,source_passage_sha,
     original_sha,representation_id,kind,text,summary_sha,provenance_raw)=r
    if not isinstance(text, str) or hashlib.sha256(text.encode("utf-8")).hexdigest() != summary_sha:
        return None
    try:
        provenance=json.loads(provenance_raw)
    except (ValueError, TypeError):
        return None
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
        or kind not in (
            ("summary","summary_long","keyword","keyphrase","entity",
             "topic","description","sentence")
            if include_facets else ("summary","summary_long")
        )
    ):
        return None
    return {
        "filesystem_entry_id": entry_id,
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
        "similarity": float(similarity),
        "native_chunk_id": chunk_id,
        "native_representation_id": representation_id,
        "children": provenance.get("children", []),
    }


def search_summary_entries_batched(
    db: sqlite3.Connection,
    query_vector: np.ndarray,
    *,
    limit: int = 10,
    nodes_per_document: int = 5,
    expected_model: str | None = None,
    require_source: Callable[[dict], bool] | None = None,
    vector_batch_size: int = 1024,
    include_facets: bool = False,
) -> list[dict]:
    """Same source-linked output as KMD scalar search, with bounded memory."""
    if not 1 <= limit <= 100 or not 1 <= nodes_per_document <= 30:
        raise ValueError("invalid bounded summary-search results")
    documents = _scan_vectors(
        db, query_vector, expected_model=expected_model,
        batch_size=vector_batch_size,
        include_facets=include_facets,
    )
    ranked = []
    for entry_id, hits in documents.items():
        hits.sort(key=lambda t: (-t[0], t[1], t[2]))
        ranked.append((entry_id, hits))
    ranked.sort(key=lambda t: (-t[1][0][0], t[1][0][1]))
    out = []
    for entry_id, raw_hits in ranked:
        if len(out) >= limit:
            cutoff = sorted((r["similarity"] for r in out), reverse=True)[limit-1]
            if raw_hits[0][0] < cutoff:
                break
        valid=[]
        for score, path, rep_id in raw_hits:
            hit=_validated_summary_node(db, rep_id, score,
                                        include_facets=include_facets)
            if hit is None:
                continue
            if require_source is not None and not require_source(hit):
                continue
            valid.append(hit)
        if not valid:
            continue
        # Reproduce incumbent tie-breaking by the original section
        # hierarchy and character offset.
        valid.sort(key=lambda x: (
            -x["similarity"], -x["summary_level"],
            x["source_start_char"], x["summary_node_id"],
        ))
        best=valid[0]
        out.append({
            "filesystem_entry_id": entry_id,
            "original_path": best["original_path"],
            "original_bytes": best["original_bytes"],
            "original_sha256": best["original_sha256"],
            "similarity": best["similarity"],
            "best_summary": best["summary"],
            "best_kind": best["summary_kind"],
            "best_level": best["summary_level"],
            "matched_summary_nodes": valid[:nodes_per_document],
            "indexed_summary_nodes_for_document": len(valid),
            "source_scope": "retrieval_candidate_only_verify_original_before_claims",
        })
    out.sort(key=lambda x:(-x["similarity"],x["original_path"]))
    return out[:limit]
