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
            """
            CREATE TABLE IF NOT EXISTS temporal_edges (
              edge_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              source_span_id TEXT NOT NULL,
              referent_id TEXT,
              context_id TEXT,
              relation TEXT NOT NULL,
              temporal_value TEXT NOT NULL,
              state_value TEXT,
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
            "CREATE INDEX IF NOT EXISTS idx_temporal_ref ON temporal_edges(referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_temporal_relation ON temporal_edges(relation)",
            "CREATE INDEX IF NOT EXISTS idx_temporal_value ON temporal_edges(temporal_value)",
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
            "temporal_edges",
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

    def referent_candidate_chunks(self, run_id: str, anchors: list[str], limit: int = 12) -> list[sqlite3.Row]:
        scores: dict[tuple[str, int], float] = {}
        rows_by_key: dict[tuple[str, int], sqlite3.Row] = {}
        for anchor in anchors:
            anchor_norm = normalize(anchor)
            if not anchor_norm:
                continue
            rows = self.connection.execute(
                """
                SELECT d.rel_path, c.chunk_order, c.text, r.canonical_label, m.entity_type
                FROM referents r
                JOIN mention_referents mr ON mr.referent_id = r.referent_id
                JOIN mentions m ON m.mention_id = mr.mention_id
                JOIN source_spans s ON s.span_id = m.span_id
                JOIN chunks c ON c.chunk_id = s.chunk_id
                JOIN documents d ON d.document_id = c.document_id
                WHERE r.run_id = ? AND r.canonical_label_norm LIKE ?
                LIMIT ?
                """,
                (run_id, f"%{anchor_norm}%", limit),
            ).fetchall()
            for row in rows:
                key = (str(row["rel_path"]), int(row["chunk_order"]))
                rows_by_key[key] = row
                scores[key] = scores.get(key, 0.0) + 3.0
        ordered = sorted(rows_by_key.items(), key=lambda item: (-scores[item[0]], item[0][0], item[0][1]))
        return [row for _, row in ordered[:limit]]

    def frame_candidate_chunks(
        self,
        run_id: str,
        predicates: list[str],
        anchors: list[str],
        limit: int = 12,
    ) -> list[sqlite3.Row]:
        if not predicates:
            return []
        predicate_norms = [normalize(predicate) for predicate in predicates if normalize(predicate)]
        placeholders = ",".join("?" for _ in predicate_norms)
        rows = self.connection.execute(
            f"""
            SELECT d.rel_path, c.chunk_order, c.text, f.predicate_norm, f.trigger_surface, ctx.kind AS context_kind
            FROM frames f
            JOIN source_spans s ON s.span_id = f.span_id
            JOIN chunks c ON c.chunk_id = s.chunk_id
            JOIN documents d ON d.document_id = c.document_id
            JOIN contexts ctx ON ctx.context_id = f.context_id
            WHERE f.run_id = ? AND f.predicate_norm IN ({placeholders})
            LIMIT ?
            """,
            (run_id, *predicate_norms, limit * 4),
        ).fetchall()
        anchor_norms = [normalize(anchor) for anchor in anchors if normalize(anchor)]
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            text_norm = normalize(str(row["text"]))
            score = 4.0
            score += sum(2.0 for anchor in anchor_norms if anchor in text_norm)
            scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], str(item[1]["rel_path"]), int(item[1]["chunk_order"])))
        return [row for _, row in scored[:limit]]

    def latest_state(self, run_id: str, anchors: list[str]) -> sqlite3.Row | None:
        anchor_norms = [normalize(anchor) for anchor in anchors if len(normalize(anchor)) > 2]
        if not anchor_norms:
            return None
        rows = self.connection.execute(
            """
            SELECT d.rel_path, c.chunk_order, c.text, t.temporal_value, t.state_value
            FROM temporal_edges t
            JOIN source_spans s ON s.span_id = t.source_span_id
            JOIN chunks c ON c.chunk_id = s.chunk_id
            JOIN documents d ON d.document_id = c.document_id
            WHERE t.run_id = ? AND t.relation = 'state_at' AND t.state_value IS NOT NULL
            ORDER BY t.temporal_value DESC
            """,
            (run_id,),
        ).fetchall()
        for row in rows:
            text_norm = normalize(str(row["text"]))
            if all(anchor in text_norm for anchor in anchor_norms):
                return row
        for row in rows:
            text_norm = normalize(str(row["text"]))
            if any(anchor in text_norm for anchor in anchor_norms):
                return row
        return None
