#!/usr/bin/env python3
"""Read-only, source-revision-aware 48h storage-pressure diagnostic.

Measured recent vectors, generation JSON and representation text are a
PARTIAL future-size projection: SQLite btrees, unused pages, other writers,
and a different future document mix are not predicted here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from file_system_catalog.summary_coverage import (
    current_completed_originals, summary_progress_metrics,
)
from file_system_catalog.summary_storage_policy import storage_budget_snapshot

DATABASE = Path("/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3")
SOURCES = Path("/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3")


def measure(database: Path, source_map: Path, *, window_hours: int = 48,
            now_ns: int | None = None) -> dict:
    if type(window_hours) is not int or not 1 <= window_hours <= 168:
        raise ValueError("invalid bounded recent window")
    now_ns = time.time_ns() if now_ns is None else now_ns
    if type(now_ns) is not int or now_ns <= 0:
        raise ValueError("invalid timestamp")
    if database.is_symlink() or source_map.is_symlink() or not database.is_file() or not source_map.is_file():
        raise ValueError("missing or unsafe read-only source databases")
    older = now_ns - window_hours * 3600 * 10**9
    with sqlite3.connect(f"file:{source_map.resolve()}?mode=ro", uri=True) as source:
        current = {
            row[0]: (row[1], row[2], int(row[3]))
            for row in source.execute(
                "SELECT source_path,source_sha256,derived_sha256,text_characters "
                "FROM originals WHERE status='complete' AND text_characters>100"
            )
        }
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        complete = current_completed_originals(db, current)
        progress = summary_progress_metrics(current, complete)
        recent = db.execute(
            """SELECT COUNT(*),SUM(LENGTH(r.embedding_blob)),
                      SUM(LENGTH(CAST(r.generation_json AS BLOB))),
                      SUM(LENGTH(CAST(r.representation_text AS BLOB)))
               FROM content_representations r
               JOIN content_chunks c ON c.chunk_id=r.chunk_id
               WHERE r.created_at_ns>? AND r.created_at_ns<=?
                 AND r.prompt_version='kmd-hierarchical-source-summary-v1'
                 AND c.collection_id LIKE '%:summary'
                 AND r.analysis_status='complete'""", (older, now_ns),
        ).fetchone()
        recent_sources = {
            row[0] for row in db.execute(
                """SELECT DISTINCT f.relative_path_display
                   FROM content_representations r
                   JOIN content_chunks c ON c.chunk_id=r.chunk_id
                   JOIN filesystem_entries f ON f.id=c.filesystem_entry_id
                   WHERE r.created_at_ns>? AND r.created_at_ns<=?
                     AND r.prompt_version='kmd-hierarchical-source-summary-v1'
                     AND c.collection_id LIKE '%:summary'
                     AND r.analysis_status='complete'""", (older, now_ns),
            )
        }
    in_current = [path for path in recent_sources if path in current]
    source_chars = sum(current[path][2] for path in in_current)
    component_bytes = {
        "embedding_blobs": recent[1] or 0,
        "generation_json": recent[2] or 0,
        "representation_text": recent[3] or 0,
    }
    partial_bytes = sum(component_bytes.values())
    remaining = progress["eligible_text_characters"] - progress["indexed_text_characters"]
    # No invented projection from a zero or unrepresentative denominator.
    estimated_partial = (remaining * partial_bytes // source_chars
                         if len(in_current) >= 10 and source_chars > 0 else None)
    disk = storage_budget_snapshot(database)
    return {
        "as_of_utc": datetime.fromtimestamp(now_ns / 10**9, timezone.utc).isoformat(),
        "window_hours": window_hours,
        "recent_indexed_originals": len(in_current),
        "recent_representation_rows": recent[0],
        "recent_source_text_characters": source_chars,
        "recent_measured_component_bytes": component_bytes,
        "recent_partial_payload_bytes": partial_bytes,
        "completed_originals": progress["complete_originals"],
        "remaining_originals": progress["remaining_originals"],
        "remaining_extracted_source_characters": remaining,
        "estimated_remaining_partial_payload_bytes_at_recent_density": estimated_partial,
        "projection_scope": (
            "Partial-size extrapolation for three representation fields at the recent "
            "source-character density; not a bound on future bytes. Excludes SQLite page/index overhead, "
            "other writers, page-level variance and future document mix."
        ),
        "database_storage": disk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-hours", type=int, default=48)
    args = parser.parse_args()
    report = measure(DATABASE, SOURCES, window_hours=args.window_hours)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
