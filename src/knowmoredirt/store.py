"""SQLite-backed DSPG storage for KnowMoreDiRT.

This is a cleaned vertical slice of the old DRT/DSPG store: normalized
documents, chunks, spans, mentions, referents, contexts, frames, and arguments.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .text import normalize


SCHEMA_VERSION = 1


def stable_id(prefix: str, *parts: Any) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


class DSPGStore:
    """Small SQLite persistence layer for internal DSPG records."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.initialize_schema()

    def initialize_schema(self) -> None:
        statements = [
            "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            """
            CREATE TABLE IF NOT EXISTS extraction_runs (
              run_id TEXT PRIMARY KEY,
              started_at REAL NOT NULL,
              input_root TEXT NOT NULL,
              status TEXT NOT NULL,
              metrics_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS documents (
              document_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              path TEXT NOT NULL,
              rel_path TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              size_bytes INTEGER NOT NULL,
              mtime REAL NOT NULL,
              ctime REAL NOT NULL,
              char_count INTEGER NOT NULL,
              metadata_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chunks (
              chunk_id TEXT PRIMARY KEY,
              document_id TEXT NOT NULL,
              chunk_order INTEGER NOT NULL,
              char_start INTEGER NOT NULL,
              char_end INTEGER NOT NULL,
              text TEXT NOT NULL,
              token_estimate INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS source_spans (
              span_id TEXT PRIMARY KEY,
              document_id TEXT NOT NULL,
              chunk_id TEXT NOT NULL,
              char_start INTEGER NOT NULL,
              char_end INTEGER NOT NULL,
              surface TEXT NOT NULL,
              surface_norm TEXT NOT NULL,
              span_kind TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mentions (
              mention_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              span_id TEXT NOT NULL,
              surface TEXT NOT NULL,
              surface_norm TEXT NOT NULL,
              mention_kind TEXT NOT NULL,
              entity_type TEXT NOT NULL,
              confidence REAL NOT NULL,
              source TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS referents (
              referent_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              canonical_label TEXT NOT NULL,
              canonical_label_norm TEXT NOT NULL,
              entity_type TEXT NOT NULL,
              status TEXT NOT NULL,
              attributes_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mention_referents (
              mention_id TEXT NOT NULL,
              referent_id TEXT NOT NULL,
              link_status TEXT NOT NULL,
              confidence REAL NOT NULL,
              PRIMARY KEY (mention_id, referent_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS contexts (
              context_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              kind TEXT NOT NULL,
              parent_context_id TEXT,
              holder_surface TEXT,
              evidence_surface TEXT,
              confidence REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS frames (
              frame_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              context_id TEXT NOT NULL,
              predicate TEXT NOT NULL,
              predicate_norm TEXT NOT NULL,
              trigger_surface TEXT NOT NULL,
              confidence REAL NOT NULL,
              source TEXT NOT NULL,
              span_id TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS frame_arguments (
              argument_id TEXT PRIMARY KEY,
              frame_id TEXT NOT NULL,
              role TEXT NOT NULL,
              mention_id TEXT,
              referent_id TEXT,
              surface TEXT,
              confidence REAL NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_documents_run ON documents(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_documents_rel ON documents(rel_path)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_order ON chunks(document_id, chunk_order)",
            "CREATE INDEX IF NOT EXISTS idx_spans_surface ON source_spans(surface_norm)",
            "CREATE INDEX IF NOT EXISTS idx_mentions_surface ON mentions(surface_norm)",
            "CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_referents_label ON referents(canonical_label_norm)",
            "CREATE INDEX IF NOT EXISTS idx_context_kind ON contexts(kind)",
            "CREATE INDEX IF NOT EXISTS idx_frames_predicate ON frames(predicate_norm)",
            "CREATE INDEX IF NOT EXISTS idx_frame_args_role ON frame_arguments(role)",
        ]
        for statement in statements:
            self.connection.execute(statement)
        self.connection.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()

    def start_run(self, input_root: str | Path) -> str:
        run_id = stable_id("run", str(input_root), time.time())
        self.connection.execute(
            "INSERT INTO extraction_runs(run_id, started_at, input_root, status, metrics_json) VALUES (?, ?, ?, ?, ?)",
            (run_id, time.time(), str(input_root), "running", "{}"),
        )
        return run_id

    def finish_run(self, run_id: str, metrics: dict[str, Any]) -> None:
        self.connection.execute(
            "UPDATE extraction_runs SET status=?, metrics_json=? WHERE run_id=?",
            ("completed", json.dumps(metrics, sort_keys=True), run_id),
        )
        self.connection.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self.connection.execute(sql, params)

    def commit(self) -> None:
        self.connection.commit()

    def counts(self) -> dict[str, int]:
        tables = [
            "documents",
            "chunks",
            "source_spans",
            "mentions",
            "referents",
            "mention_referents",
            "contexts",
            "frames",
            "frame_arguments",
        ]
        return {
            table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }

    def integrity_check(self) -> str:
        return str(self.connection.execute("PRAGMA integrity_check").fetchone()[0])

    def upsert_referent(self, run_id: str, label: str, entity_type: str) -> str:
        label_norm = normalize(label)
        referent_id = stable_id("ref", run_id, label_norm, entity_type)
        self.connection.execute(
            """
            INSERT OR IGNORE INTO referents(
              referent_id, run_id, canonical_label, canonical_label_norm, entity_type, status, attributes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (referent_id, run_id, label, label_norm, entity_type, "candidate", "{}"),
        )
        return referent_id

