"""Exact-result, bounded-memory native FileDB semantic scan experiment.

Uses the original KMD cosine_similarity and vector_from_blob, unchanged SQL
and source ordering, but retains ONLY the best candidate for each original
file. This preserves existing ranking and ties while avoiding a 70k-dict
all-candidates list. It does not replace the installed Infra semantic reader
or its final original-source SHA verification. No model/source/index writes.
"""
from __future__ import annotations
from typing import Any
import sqlite3
import numpy as np

from .content_pipeline import cosine_similarity, vector_from_blob
from .schema import TABLE_NAME
from .content_schema import CHUNK_TABLE_NAME, REPRESENTATION_TABLE_NAME


def stream_exact_native_semantic_entries(connection: sqlite3.Connection, query_vector: np.ndarray) -> list[dict[str,Any]]:
    """Match legacy search_semantic_entries file-level selection exactly.

    The caller controls whether `content_chunks` is a read-only scoped CTE.
    This function does not change SQL visibility, use stale cached vectors,
    or weaken any publisher-source evidence/revision checks. It intentionally
    returns the same full ranked list (not a top-k truncation), preserving
    callers that skip unverified top candidates.
    """
    best: dict[str,dict[str,Any]]={}
    for row in connection.execute(
        f"""SELECT c.chunk_id,c.file_id,c.filesystem_entry_id,f.relative_path_display,f.relative_path_b64,
        c.chunk_kind,c.chunk_index,c.start_char,c.end_char,c.embedding_dimension,c.embedding_dtype,c.embedding_blob
        FROM {CHUNK_TABLE_NAME} c JOIN {TABLE_NAME} f ON f.id=c.filesystem_entry_id
        WHERE c.embedding_blob IS NOT NULL"""
    ):
        vector=vector_from_blob(row[11],int(row[9]),row[10])
        score=cosine_similarity(query_vector,vector)
        key=str(row[1])
        prior=best.get(key)
        # Keep the legacy STRICT > comparison. Rewriting it as <= negation
        # changes the result when a corrupted native vector scores NaN:
        # NaN is neither > nor <= a finite winner.
        if prior is not None and not (score>float(prior['score'])):
            continue
        best[key]={
            'file_id':row[1],
            'filesystem_entry_id':int(row[2]),
            'relative_path_display':row[3],
            'relative_path_b64':row[4],
            'chunk_id':row[0],
            'source_level':row[5],
            'source_index':int(row[6]),
            'source_start_char':int(row[7]),
            'source_end_char':int(row[8]),
            'analysis_kind':'chunk',
            'analysis_text':'',
            'score':score,
        }
    for row in connection.execute(
        f"""SELECT r.representation_id,r.representation_kind,r.representation_text,r.embedding_dimension,
        r.embedding_dtype,r.embedding_blob,c.chunk_id,c.file_id,c.filesystem_entry_id,f.relative_path_display,
        f.relative_path_b64,c.chunk_kind,c.chunk_index,c.start_char,c.end_char
        FROM {REPRESENTATION_TABLE_NAME} r JOIN {CHUNK_TABLE_NAME} c ON c.chunk_id=r.chunk_id
        JOIN {TABLE_NAME} f ON f.id=c.filesystem_entry_id WHERE r.analysis_status='complete'"""
    ):
        vector=vector_from_blob(row[5],int(row[3]),row[4])
        score=cosine_similarity(query_vector,vector)
        key=str(row[7])
        prior=best.get(key)
        # Keep the legacy STRICT > comparison. Rewriting it as <= negation
        # changes the result when a corrupted native vector scores NaN:
        # NaN is neither > nor <= a finite winner.
        if prior is not None and not (score>float(prior['score'])):
            continue
        best[key]={
            'file_id':row[7],
            'filesystem_entry_id':int(row[8]),
            'relative_path_display':row[9],
            'relative_path_b64':row[10],
            'chunk_id':row[6],
            'representation_id':row[0],
            'source_level':row[11],
            'source_index':int(row[12]),
            'source_start_char':int(row[13]),
            'source_end_char':int(row[14]),
            'analysis_kind':row[1],
            'analysis_text':row[2],
            'score':score,
        }
    return sorted(best.values(),key=lambda item:(-float(item['score']),str(item['relative_path_display'])))
