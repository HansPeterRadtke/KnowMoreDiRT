"""Revision-aware completeness for KnowMoreDiRT's native source summary index.

A document is complete only when a full-source root summary points to the
current original SHA and current extracted-text SHA. An arbitrary leaf node,
a stale root, or a summary over only part of a source is NOT sufficient to
remove a document from the active processing backlog.

The extracted-text manifest's text_characters field deliberately excludes the
KMD provenance header. It must not be compared to source end_char directly;
compare the source-tree SHA to the derivative SHA instead.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Mapping, Sequence

PROMPT_VERSION = "kmd-hierarchical-source-summary-v1"


def revision_identity(path: str, original_sha: str, derived_sha: str) -> str:
    return hashlib.sha256("\0".join((path, original_sha, derived_sha)).encode("utf-8")).hexdigest()


def pending_failures(failed, available, processed):
    """Filter worker receipts against eligible revisions and current complete roots."""
    pending = {}
    for path, original, derived, _ in available:
        entry = failed.get(path)
        if path in processed or not isinstance(entry, dict):
            continue
        # Earlier receipts have no revision identity; retain their path-based behavior.
        if "identity" not in entry or entry["identity"] == revision_identity(path, original, derived):
            pending[path] = entry
    return pending


def current_completed_originals(
    db: sqlite3.Connection,
    sources: Mapping[str, tuple[str, str, int]],
) -> set[str]:
    """Read canonical KMD roots; return originals complete at current revisions.

    The original KMD FileDB and caller's source_map are not mutated.
    Keys are exact relative paths, never filenames or aliases.
    """
    current = set()
    query = """
        SELECT f.relative_path_display, c.content_sha256,
               c.start_char, c.end_char, r.generation_json,
               r.representation_text, r.representation_text_sha256
        FROM content_representations AS r
        JOIN content_chunks AS c ON c.chunk_id=r.chunk_id
        JOIN filesystem_entries AS f ON f.id=c.filesystem_entry_id
        WHERE c.collection_id LIKE '%:summary'
          AND r.prompt_version=?
          AND r.representation_kind='summary_long'
          AND r.analysis_status='complete'
          AND f.entry_type='file'
    """
    for path, original_sha, start, end, raw, summary, summary_sha in db.execute(
        query, (PROMPT_VERSION,)
    ):
        version = sources.get(path)
        if version is None or not isinstance(version, (tuple, list)) or len(version) != 3:
            continue
        current_original, current_derivative, text_characters = version
        if (
            original_sha != current_original
            or type(start) is not int or start != 0
            or type(end) is not int or end <= 0
            or not isinstance(raw, str)
            or not isinstance(summary, str)
            or not summary.strip()
            or not isinstance(summary_sha, str)
            or hashlib.sha256(summary.encode("utf-8")).hexdigest() != summary_sha
        ):
            continue
        try:
            metadata = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(metadata, dict):
            continue
        if (
            metadata.get("original_sha256") == current_original
            and metadata.get("derived_sha256") == current_derivative
            and metadata.get("source_tree_sha256") == current_derivative
            and metadata.get("source_start_char") == 0
            and metadata.get("source_end_char") == end
            and isinstance(metadata.get("node_id"), str)
            and bool(metadata["node_id"])
            and type(text_characters) is int
            and text_characters > 0
        ):
            current.add(path)
    return current


def summary_progress_metrics(
    sources: Mapping[str, tuple[str, str, int]],
    processed: set[str],
) -> dict:
    """Both document and content-volume progress, never conflating the two."""
    total = len(sources)
    done_paths = processed & sources.keys()
    done = len(done_paths)
    total_chars = sum(max(0, int(row[2])) for row in sources.values())
    done_chars = sum(max(0, int(sources[p][2])) for p in done_paths)
    return {
        "complete_originals": done,
        "total_eligible_originals": total,
        "remaining_originals": max(0, total-done),
        "percentage": round(100.0*done/max(1,total), 6),
        "eligible_text_characters": total_chars,
        "indexed_text_characters": done_chars,
        "text_percentage": round(100.0*done_chars/max(1,total_chars), 6),
        "content_progress_basis": "source_map.extracted_text_characters_excluding_provenance_header",
    }
