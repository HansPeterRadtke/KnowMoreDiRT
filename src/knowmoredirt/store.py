"""SQLite-backed DSPG storage for KnowMoreDiRT.

This is a cleaned vertical slice of the old DRT/DSPG store: normalized
documents, chunks, spans, mentions, referents, contexts, frames, and arguments.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from .drs_validation import box_parent_cycle_errors, box_root_errors, condition_argument_cycle_errors
from .storage import StoreConfig, open_sqlite
from .text import normalize, normalize_predicate_polarity, split_units


DRS_CONTEXT_KINDS = {
    "asserted",
    "negated",
    "conditional_antecedent",
    "conditional_consequent",
    "reported",
    "quoted",
    "believed",
    "possible",
    "uncertain",
    "hypothetical",
    "fictional",
    "dreamed",
}
DRS_POLARITIES = {"positive", "negative", "unknown"}
SCHEMA_VERSION = 13
IDENTITY_EXPANSION_RELATIONS = {"accepted", "same_referent", "same_surface", "alias", "coreference", "coreferent"}


def stable_id(prefix: str, *parts: Any) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


def identity_relation_allows_expansion(relation: str) -> bool:
    return normalize(relation) in IDENTITY_EXPANSION_RELATIONS


class DSPGStore:
    """Small SQLite persistence layer for internal DSPG records."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        create_indexes: bool = True,
        config: StoreConfig | None = None,
    ) -> None:
        self.config = config or StoreConfig.sqlite(path, create_indexes=create_indexes)
        self.path = self.config.location
        self.connection = open_sqlite(self.config)
        self.initialize_schema(create_indexes=self.config.create_indexes)

    def initialize_schema(self, *, create_indexes: bool = True) -> None:
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
            CREATE TABLE IF NOT EXISTS identity_hypotheses (
              hypothesis_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              source_span_id TEXT,
              context_id TEXT,
              drs_box_id TEXT,
              box_external_id TEXT,
              left_referent_id TEXT NOT NULL,
              right_referent_id TEXT NOT NULL,
              relation TEXT NOT NULL,
              evidence TEXT NOT NULL,
              confidence REAL NOT NULL,
              source TEXT NOT NULL
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
              confidence REAL NOT NULL,
              declared_authority TEXT NOT NULL DEFAULT '',
              verified_authority TEXT NOT NULL DEFAULT '',
              authority_source_span_id TEXT REFERENCES source_spans(span_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS context_carriers (
              carrier_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              context_id TEXT NOT NULL,
              document_id TEXT,
              source_span_id TEXT,
              carrier_kind TEXT NOT NULL,
              carrier_surface TEXT NOT NULL,
              temporal_value TEXT,
              temporal_value_type TEXT,
              confidence REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS context_assignments (
              assignment_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              context_id TEXT NOT NULL,
              applies_to_type TEXT NOT NULL,
              applies_to_id TEXT NOT NULL,
              source_span_id TEXT,
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
              value_type TEXT,
              confidence REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS drs_boxes (
              drs_box_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              source_span_id TEXT NOT NULL,
              external_box_id TEXT NOT NULL,
              context_id TEXT NOT NULL,
              parent_drs_box_id TEXT,
              parent_external_box_id TEXT,
              kind TEXT NOT NULL,
              holder_referent_id TEXT,
              holder_external_referent_id TEXT,
              evidence_surface TEXT,
              confidence REAL NOT NULL,
              source TEXT NOT NULL,
              metadata_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS drs_referents (
              drs_referent_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              source_span_id TEXT NOT NULL,
              external_referent_id TEXT NOT NULL,
              referent_id TEXT NOT NULL,
              box_id TEXT,
              surface TEXT NOT NULL,
              surface_norm TEXT NOT NULL,
              value_type TEXT NOT NULL,
              evidence_surface TEXT,
              confidence REAL NOT NULL,
              source TEXT NOT NULL,
              metadata_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS drs_conditions (
              drs_condition_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              source_span_id TEXT NOT NULL,
              external_condition_id TEXT NOT NULL,
              box_id TEXT NOT NULL,
              context_id TEXT NOT NULL,
              frame_id TEXT,
              predicate TEXT NOT NULL,
              predicate_norm TEXT NOT NULL,
              polarity TEXT NOT NULL,
              modality TEXT NOT NULL,
              temporal_id TEXT,
              temporal_text TEXT,
              evidence_surface TEXT NOT NULL,
              confidence REAL NOT NULL,
              source TEXT NOT NULL,
              metadata_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS drs_condition_arguments (
              drs_argument_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              drs_condition_id TEXT NOT NULL,
              role TEXT NOT NULL,
              target_kind TEXT NOT NULL,
              target_external_id TEXT,
              referent_id TEXT,
              target_box_id TEXT,
              target_condition_id TEXT,
              value TEXT,
              value_norm TEXT,
              value_type TEXT,
              evidence_surface TEXT,
              confidence REAL NOT NULL,
              metadata_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS drs_identity_hypotheses (
              drs_hypothesis_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              source_span_id TEXT NOT NULL,
              context_id TEXT,
              box_id TEXT,
              box_external_id TEXT,
              left_external_referent_id TEXT NOT NULL,
              right_external_referent_id TEXT NOT NULL,
              left_referent_id TEXT NOT NULL,
              right_referent_id TEXT NOT NULL,
              relation TEXT NOT NULL,
              evidence_surface TEXT NOT NULL,
              confidence REAL NOT NULL,
              source TEXT NOT NULL,
              metadata_json TEXT
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
            """
            CREATE TABLE IF NOT EXISTS relations (
              relation_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              relation_type TEXT NOT NULL,
              subject TEXT,
              subject_norm TEXT,
              predicate TEXT NOT NULL,
              predicate_norm TEXT NOT NULL,
              object TEXT,
              object_norm TEXT,
              value TEXT,
              value_norm TEXT,
              source_span_id TEXT NOT NULL,
              context_id TEXT,
              confidence REAL NOT NULL,
              metadata_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS discourse_edges (
              edge_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE,
              relation_type TEXT NOT NULL,
              document_id TEXT REFERENCES documents(document_id) ON DELETE CASCADE,
              source_span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE,
              from_context_id TEXT REFERENCES contexts(context_id) ON DELETE CASCADE,
              to_context_id TEXT REFERENCES contexts(context_id) ON DELETE CASCADE,
              from_span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE,
              to_span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE,
              evidence_surface TEXT NOT NULL,
              confidence REAL NOT NULL,
              source TEXT NOT NULL,
              metadata_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS metadata_records (
              metadata_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              document_id TEXT NOT NULL,
              key TEXT NOT NULL,
              value TEXT NOT NULL,
              value_norm TEXT NOT NULL,
              source TEXT NOT NULL,
              confidence REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_attempts (
              attempt_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              source_span_id TEXT NOT NULL,
              task TEXT NOT NULL,
              source TEXT NOT NULL,
              cache_key TEXT NOT NULL,
              accepted INTEGER NOT NULL,
              materialized INTEGER NOT NULL,
              reason TEXT,
              prompt_hash TEXT,
              output_hash TEXT,
              elapsed REAL,
              metadata_json TEXT
            )
            """,
        ]
        for statement in statements:
            self.connection.execute(statement)
        current_version = self._stored_schema_version()
        self._migrate_schema(current_version)
        if create_indexes:
            self.create_indexes()
        self.connection.commit()

    def _stored_schema_version(self) -> int:
        row = self.connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            # Databases created by the historical schema had no version until
            # initialization finished.  The CREATE IF NOT EXISTS statements above
            # establish the v10 baseline before explicit migrations run.
            return 10
        try:
            version = int(row["value"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("invalid KMD DSPG schema_version metadata") from exc
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"KMD DSPG schema {version} is newer than supported schema {SCHEMA_VERSION}"
            )
        if version < 10:
            raise RuntimeError(
                f"KMD DSPG schema {version} predates the supported explicit migration baseline 10"
            )
        return version

    def _record_schema_version(self, version: int) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (str(version),),
        )

    def _migrate_schema(self, current_version: int) -> None:
        # Legacy v10 identity columns were historically added ad hoc.  Keep the
        # normalization as the explicit v10 baseline step so old databases are
        # deterministic before v11/v12 migrations.
        for table, column, definition in (
            ("identity_hypotheses", "source_span_id", "TEXT"),
            ("identity_hypotheses", "context_id", "TEXT"),
            ("identity_hypotheses", "drs_box_id", "TEXT"),
            ("identity_hypotheses", "box_external_id", "TEXT"),
            ("drs_identity_hypotheses", "context_id", "TEXT"),
            ("drs_identity_hypotheses", "box_id", "TEXT"),
            ("drs_identity_hypotheses", "box_external_id", "TEXT"),
        ):
            self._ensure_column(table, column, definition)
        version = current_version
        while version < SCHEMA_VERSION:
            target = version + 1
            if target == 13:
                self._migrate_v13_foreign_keys()
                version = target
                continue
            savepoint = f"schema_migration_{target}"
            self.connection.execute(f"SAVEPOINT {savepoint}")
            try:
                if target == 11:
                    self._ensure_column("contexts", "declared_authority", "TEXT NOT NULL DEFAULT ''")
                    self._ensure_column("contexts", "verified_authority", "TEXT NOT NULL DEFAULT ''")
                    self._ensure_column("contexts", "authority_source_span_id", "TEXT")
                elif target == 12:
                    self.connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS discourse_edges (
                          edge_id TEXT PRIMARY KEY,
                          run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE,
                          relation_type TEXT NOT NULL,
                          document_id TEXT REFERENCES documents(document_id) ON DELETE CASCADE,
                          source_span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE,
                          from_context_id TEXT REFERENCES contexts(context_id) ON DELETE CASCADE,
                          to_context_id TEXT REFERENCES contexts(context_id) ON DELETE CASCADE,
                          from_span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE,
                          to_span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE,
                          evidence_surface TEXT NOT NULL,
                          confidence REAL NOT NULL,
                          source TEXT NOT NULL,
                          metadata_json TEXT
                        )
                        """
                    )
                self._record_schema_version(target)
                self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            except BaseException:
                self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                raise
            version = target

    @staticmethod
    def _v13_fk_table_definitions() -> dict[str, str]:
        return {
            "extraction_runs": """CREATE TABLE {name} (run_id TEXT PRIMARY KEY, started_at REAL NOT NULL, input_root TEXT NOT NULL, status TEXT NOT NULL, metrics_json TEXT)""",
            "documents": """CREATE TABLE {name} (document_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, path TEXT NOT NULL, rel_path TEXT NOT NULL, content_hash TEXT NOT NULL, size_bytes INTEGER NOT NULL, mtime REAL NOT NULL, ctime REAL NOT NULL, char_count INTEGER NOT NULL, metadata_json TEXT)""",
            "chunks": """CREATE TABLE {name} (chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE, chunk_order INTEGER NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL, text TEXT NOT NULL, token_estimate INTEGER NOT NULL)""",
            "source_spans": """CREATE TABLE {name} (span_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE, chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL, surface TEXT NOT NULL, surface_norm TEXT NOT NULL, span_kind TEXT NOT NULL)""",
            "mentions": """CREATE TABLE {name} (mention_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, span_id TEXT NOT NULL REFERENCES source_spans(span_id) ON DELETE CASCADE, surface TEXT NOT NULL, surface_norm TEXT NOT NULL, mention_kind TEXT NOT NULL, entity_type TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL)""",
            "referents": """CREATE TABLE {name} (referent_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, canonical_label TEXT NOT NULL, canonical_label_norm TEXT NOT NULL, entity_type TEXT NOT NULL, status TEXT NOT NULL, attributes_json TEXT)""",
            "mention_referents": """CREATE TABLE {name} (mention_id TEXT NOT NULL REFERENCES mentions(mention_id) ON DELETE CASCADE, referent_id TEXT NOT NULL REFERENCES referents(referent_id) ON DELETE CASCADE, link_status TEXT NOT NULL, confidence REAL NOT NULL, PRIMARY KEY (mention_id, referent_id))""",
            "contexts": """CREATE TABLE {name} (context_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, kind TEXT NOT NULL, parent_context_id TEXT REFERENCES contexts(context_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED, holder_surface TEXT, evidence_surface TEXT, confidence REAL NOT NULL, declared_authority TEXT NOT NULL DEFAULT '', verified_authority TEXT NOT NULL DEFAULT '', authority_source_span_id TEXT REFERENCES source_spans(span_id) ON DELETE SET NULL)""",
            "identity_hypotheses": """CREATE TABLE {name} (hypothesis_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, source_span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE, context_id TEXT REFERENCES contexts(context_id) ON DELETE SET NULL, drs_box_id TEXT REFERENCES drs_boxes(drs_box_id) ON DELETE SET NULL, box_external_id TEXT, left_referent_id TEXT NOT NULL REFERENCES referents(referent_id) ON DELETE CASCADE, right_referent_id TEXT NOT NULL REFERENCES referents(referent_id) ON DELETE CASCADE, relation TEXT NOT NULL, evidence TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL)""",
            "context_carriers": """CREATE TABLE {name} (carrier_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, context_id TEXT NOT NULL REFERENCES contexts(context_id) ON DELETE CASCADE, document_id TEXT REFERENCES documents(document_id) ON DELETE CASCADE, source_span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE, carrier_kind TEXT NOT NULL, carrier_surface TEXT NOT NULL, temporal_value TEXT, temporal_value_type TEXT, confidence REAL NOT NULL)""",
            "context_assignments": """CREATE TABLE {name} (assignment_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, context_id TEXT NOT NULL REFERENCES contexts(context_id) ON DELETE CASCADE, applies_to_type TEXT NOT NULL, applies_to_id TEXT NOT NULL, source_span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE, confidence REAL NOT NULL)""",
            "frames": """CREATE TABLE {name} (frame_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, context_id TEXT NOT NULL REFERENCES contexts(context_id) ON DELETE CASCADE, predicate TEXT NOT NULL, predicate_norm TEXT NOT NULL, trigger_surface TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL, span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE)""",
            "frame_arguments": """CREATE TABLE {name} (argument_id TEXT PRIMARY KEY, frame_id TEXT NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE, role TEXT NOT NULL, mention_id TEXT REFERENCES mentions(mention_id) ON DELETE SET NULL, referent_id TEXT REFERENCES referents(referent_id) ON DELETE SET NULL, surface TEXT, value_type TEXT, confidence REAL NOT NULL)""",
            "drs_boxes": """CREATE TABLE {name} (drs_box_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, source_span_id TEXT NOT NULL REFERENCES source_spans(span_id) ON DELETE CASCADE, external_box_id TEXT NOT NULL, context_id TEXT NOT NULL REFERENCES contexts(context_id) ON DELETE CASCADE, parent_drs_box_id TEXT REFERENCES drs_boxes(drs_box_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED, parent_external_box_id TEXT, kind TEXT NOT NULL, holder_referent_id TEXT REFERENCES referents(referent_id) ON DELETE SET NULL, holder_external_referent_id TEXT, evidence_surface TEXT, confidence REAL NOT NULL, source TEXT NOT NULL, metadata_json TEXT)""",
            "drs_referents": """CREATE TABLE {name} (drs_referent_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, source_span_id TEXT NOT NULL REFERENCES source_spans(span_id) ON DELETE CASCADE, external_referent_id TEXT NOT NULL, referent_id TEXT NOT NULL REFERENCES referents(referent_id) ON DELETE CASCADE, box_id TEXT REFERENCES drs_boxes(drs_box_id) ON DELETE SET NULL, surface TEXT NOT NULL, surface_norm TEXT NOT NULL, value_type TEXT NOT NULL, evidence_surface TEXT, confidence REAL NOT NULL, source TEXT NOT NULL, metadata_json TEXT)""",
            "drs_conditions": """CREATE TABLE {name} (drs_condition_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, source_span_id TEXT NOT NULL REFERENCES source_spans(span_id) ON DELETE CASCADE, external_condition_id TEXT NOT NULL, box_id TEXT NOT NULL REFERENCES drs_boxes(drs_box_id) ON DELETE CASCADE, context_id TEXT NOT NULL REFERENCES contexts(context_id) ON DELETE CASCADE, frame_id TEXT REFERENCES frames(frame_id) ON DELETE SET NULL, predicate TEXT NOT NULL, predicate_norm TEXT NOT NULL, polarity TEXT NOT NULL, modality TEXT NOT NULL, temporal_id TEXT, temporal_text TEXT, evidence_surface TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL, metadata_json TEXT)""",
            "drs_condition_arguments": """CREATE TABLE {name} (drs_argument_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, drs_condition_id TEXT NOT NULL REFERENCES drs_conditions(drs_condition_id) ON DELETE CASCADE, role TEXT NOT NULL, target_kind TEXT NOT NULL, target_external_id TEXT, referent_id TEXT REFERENCES referents(referent_id) ON DELETE SET NULL, target_box_id TEXT REFERENCES drs_boxes(drs_box_id) ON DELETE SET NULL, target_condition_id TEXT REFERENCES drs_conditions(drs_condition_id) ON DELETE SET NULL, value TEXT, value_norm TEXT, value_type TEXT, evidence_surface TEXT, confidence REAL NOT NULL, metadata_json TEXT)""",
            "drs_identity_hypotheses": """CREATE TABLE {name} (drs_hypothesis_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, source_span_id TEXT NOT NULL REFERENCES source_spans(span_id) ON DELETE CASCADE, context_id TEXT REFERENCES contexts(context_id) ON DELETE SET NULL, box_id TEXT REFERENCES drs_boxes(drs_box_id) ON DELETE SET NULL, box_external_id TEXT, left_external_referent_id TEXT NOT NULL, right_external_referent_id TEXT NOT NULL, left_referent_id TEXT NOT NULL REFERENCES referents(referent_id) ON DELETE CASCADE, right_referent_id TEXT NOT NULL REFERENCES referents(referent_id) ON DELETE CASCADE, relation TEXT NOT NULL, evidence_surface TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL, metadata_json TEXT)""",
            "temporal_edges": """CREATE TABLE {name} (edge_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, source_span_id TEXT NOT NULL REFERENCES source_spans(span_id) ON DELETE CASCADE, referent_id TEXT REFERENCES referents(referent_id) ON DELETE SET NULL, context_id TEXT REFERENCES contexts(context_id) ON DELETE SET NULL, relation TEXT NOT NULL, temporal_value TEXT NOT NULL, state_value TEXT, confidence REAL NOT NULL)""",
            "relations": """CREATE TABLE {name} (relation_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, relation_type TEXT NOT NULL, subject TEXT, subject_norm TEXT, predicate TEXT NOT NULL, predicate_norm TEXT NOT NULL, object TEXT, object_norm TEXT, value TEXT, value_norm TEXT, source_span_id TEXT NOT NULL REFERENCES source_spans(span_id) ON DELETE CASCADE, context_id TEXT REFERENCES contexts(context_id) ON DELETE SET NULL, confidence REAL NOT NULL, metadata_json TEXT)""",
            "metadata_records": """CREATE TABLE {name} (metadata_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE, key TEXT NOT NULL, value TEXT NOT NULL, value_norm TEXT NOT NULL, source TEXT NOT NULL, confidence REAL NOT NULL)""",
            "model_attempts": """CREATE TABLE {name} (attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, source_span_id TEXT NOT NULL REFERENCES source_spans(span_id) ON DELETE CASCADE, task TEXT NOT NULL, source TEXT NOT NULL, cache_key TEXT NOT NULL, accepted INTEGER NOT NULL, materialized INTEGER NOT NULL, reason TEXT, prompt_hash TEXT, output_hash TEXT, elapsed REAL, metadata_json TEXT)""",
            "discourse_edges": """CREATE TABLE {name} (edge_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE, relation_type TEXT NOT NULL, document_id TEXT REFERENCES documents(document_id) ON DELETE CASCADE, source_span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE, from_context_id TEXT REFERENCES contexts(context_id) ON DELETE CASCADE, to_context_id TEXT REFERENCES contexts(context_id) ON DELETE CASCADE, from_span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE, to_span_id TEXT REFERENCES source_spans(span_id) ON DELETE CASCADE, evidence_surface TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL, metadata_json TEXT)""",
        }

    def _migrate_v13_foreign_keys(self) -> None:
        definitions = self._v13_fk_table_definitions()
        tables = list(definitions)
        integrity_errors = self.semantic_integrity_errors()
        if integrity_errors:
            raise RuntimeError(f"cannot migrate KMD DSPG schema to v13 with orphan references: {integrity_errors}")
        columns_by_table: dict[str, list[str]] = {
            table: [str(row["name"]) for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()]
            for table in tables
        }
        self.connection.commit()
        self.connection.execute("PRAGMA foreign_keys=OFF")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            for table in reversed(tables):
                self.connection.execute(f"ALTER TABLE {table} RENAME TO __v12_{table}")
            for table in tables:
                self.connection.execute(definitions[table].format(name=table))
            # Copy in dependency order. Self-references are declared DEFERRABLE.
            for table in tables:
                columns = columns_by_table[table]
                column_sql = ", ".join(columns)
                self.connection.execute(
                    f"INSERT INTO {table} ({column_sql}) SELECT {column_sql} FROM __v12_{table}"
                )
            for table in reversed(tables):
                self.connection.execute(f"DROP TABLE __v12_{table}")
            self._record_schema_version(13)
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        finally:
            self.connection.execute("PRAGMA foreign_keys=ON")
        violations = self.connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"KMD DSPG v13 foreign-key migration produced violations: {[tuple(row) for row in violations]}")

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        rows = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        if column not in {str(row["name"]) for row in rows}:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def create_indexes(self) -> None:
        statements = [
            "CREATE INDEX IF NOT EXISTS idx_documents_run ON documents(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_documents_rel ON documents(rel_path)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_order ON chunks(document_id, chunk_order)",
            "CREATE INDEX IF NOT EXISTS idx_source_spans_chunk ON source_spans(chunk_id)",
            "CREATE INDEX IF NOT EXISTS idx_source_spans_document ON source_spans(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_source_spans_doc_chunk ON source_spans(document_id, chunk_id)",
            "CREATE INDEX IF NOT EXISTS idx_spans_surface ON source_spans(surface_norm)",
            "CREATE INDEX IF NOT EXISTS idx_mentions_run_span ON mentions(run_id, span_id)",
            "CREATE INDEX IF NOT EXISTS idx_mentions_surface ON mentions(surface_norm)",
            "CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_mention_referents_ref ON mention_referents(referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_referents_label ON referents(canonical_label_norm)",
            "CREATE INDEX IF NOT EXISTS idx_identity_span ON identity_hypotheses(run_id, source_span_id)",
            "CREATE INDEX IF NOT EXISTS idx_identity_run_left ON identity_hypotheses(run_id, left_referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_identity_run_right ON identity_hypotheses(run_id, right_referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_identity_context ON identity_hypotheses(context_id)",
            "CREATE INDEX IF NOT EXISTS idx_identity_left ON identity_hypotheses(left_referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_identity_right ON identity_hypotheses(right_referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_kind ON contexts(kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_carriers_kind ON context_carriers(carrier_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_carriers_doc ON context_carriers(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_carriers_time ON context_carriers(temporal_value_type, temporal_value)",
            "CREATE INDEX IF NOT EXISTS idx_context_assignments_context ON context_assignments(context_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_assignments_applies ON context_assignments(applies_to_type, applies_to_id)",
            "CREATE INDEX IF NOT EXISTS idx_frames_run_span ON frames(run_id, span_id)",
            "CREATE INDEX IF NOT EXISTS idx_frames_span ON frames(span_id)",
            "CREATE INDEX IF NOT EXISTS idx_frames_predicate ON frames(predicate_norm)",
            "CREATE INDEX IF NOT EXISTS idx_frame_args_frame ON frame_arguments(frame_id)",
            "CREATE INDEX IF NOT EXISTS idx_frame_args_role ON frame_arguments(role)",
            "CREATE INDEX IF NOT EXISTS idx_drs_boxes_context ON drs_boxes(context_id)",
            "CREATE INDEX IF NOT EXISTS idx_drs_conditions_run_span ON drs_conditions(run_id, source_span_id)",
            "CREATE INDEX IF NOT EXISTS idx_drs_referents_surface ON drs_referents(surface_norm)",
            "CREATE INDEX IF NOT EXISTS idx_drs_conditions_predicate ON drs_conditions(predicate_norm)",
            "CREATE INDEX IF NOT EXISTS idx_drs_conditions_box ON drs_conditions(box_id)",
            "CREATE INDEX IF NOT EXISTS idx_drs_args_condition ON drs_condition_arguments(drs_condition_id)",
            "CREATE INDEX IF NOT EXISTS idx_drs_args_role ON drs_condition_arguments(role)",
            "CREATE INDEX IF NOT EXISTS idx_drs_args_ref ON drs_condition_arguments(referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_drs_identity_run_span ON drs_identity_hypotheses(run_id, source_span_id)",
            "CREATE INDEX IF NOT EXISTS idx_drs_identity_left ON drs_identity_hypotheses(left_referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_drs_identity_right ON drs_identity_hypotheses(right_referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_drs_identity_context ON drs_identity_hypotheses(context_id)",
            "CREATE INDEX IF NOT EXISTS idx_temporal_run_span ON temporal_edges(run_id, source_span_id)",
            "CREATE INDEX IF NOT EXISTS idx_temporal_span ON temporal_edges(source_span_id)",
            "CREATE INDEX IF NOT EXISTS idx_temporal_ref ON temporal_edges(referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_temporal_relation ON temporal_edges(relation)",
            "CREATE INDEX IF NOT EXISTS idx_temporal_value ON temporal_edges(temporal_value)",
            "CREATE INDEX IF NOT EXISTS idx_relations_run_span ON relations(run_id, source_span_id)",
            "CREATE INDEX IF NOT EXISTS idx_relations_span ON relations(source_span_id)",
            "CREATE INDEX IF NOT EXISTS idx_relations_predicate ON relations(predicate_norm)",
            "CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_norm)",
            "CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_norm)",
            "CREATE INDEX IF NOT EXISTS idx_relations_value ON relations(value_norm)",
            "CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type)",
            "CREATE INDEX IF NOT EXISTS idx_discourse_run_type ON discourse_edges(run_id, relation_type)",
            "CREATE INDEX IF NOT EXISTS idx_discourse_doc ON discourse_edges(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_discourse_from_context ON discourse_edges(from_context_id)",
            "CREATE INDEX IF NOT EXISTS idx_discourse_to_context ON discourse_edges(to_context_id)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_records_run_doc ON metadata_records(run_id, document_id)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_records_doc ON metadata_records(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_records_key ON metadata_records(key)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_records_value ON metadata_records(value_norm)",
            "CREATE INDEX IF NOT EXISTS idx_model_attempts_span ON model_attempts(run_id, source_span_id, task, source, cache_key)",
        ]
        for statement in statements:
            self.connection.execute(statement)
        self.connection.commit()

    def start_run(self, input_root: str | Path) -> str:
        run_id = stable_id("run", str(input_root), time.time())
        self.connection.execute(
            "INSERT INTO extraction_runs(run_id, started_at, input_root, status, metrics_json) VALUES (?, ?, ?, ?, ?)",
            (run_id, time.time(), str(input_root), "running", "{}"),
        )
        self.connection.commit()
        return run_id

    def latest_run_id(self, input_root: str | Path) -> str:
        row = self.connection.execute(
            """
            SELECT run_id
            FROM extraction_runs
            WHERE input_root=?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (str(input_root),),
        ).fetchone()
        return str(row["run_id"]) if row is not None else ""

    def semantic_integrity_errors(self) -> list[dict[str, Any]]:
        """Return orphan semantic references not covered by legacy SQLite FKs."""

        references = (
            ("chunks.document_id", "chunks", "document_id", "documents", "document_id"),
            ("source_spans.document_id", "source_spans", "document_id", "documents", "document_id"),
            ("source_spans.chunk_id", "source_spans", "chunk_id", "chunks", "chunk_id"),
            ("mentions.span_id", "mentions", "span_id", "source_spans", "span_id"),
            ("mention_referents.mention_id", "mention_referents", "mention_id", "mentions", "mention_id"),
            ("mention_referents.referent_id", "mention_referents", "referent_id", "referents", "referent_id"),
            ("metadata_records.document_id", "metadata_records", "document_id", "documents", "document_id"),
            ("model_attempts.source_span_id", "model_attempts", "source_span_id", "source_spans", "span_id"),
            ("identity_hypotheses.source_span_id", "identity_hypotheses", "source_span_id", "source_spans", "span_id"),
            ("identity_hypotheses.context_id", "identity_hypotheses", "context_id", "contexts", "context_id"),
            ("identity_hypotheses.drs_box_id", "identity_hypotheses", "drs_box_id", "drs_boxes", "drs_box_id"),
            ("identity_hypotheses.left_referent_id", "identity_hypotheses", "left_referent_id", "referents", "referent_id"),
            ("identity_hypotheses.right_referent_id", "identity_hypotheses", "right_referent_id", "referents", "referent_id"),
            ("contexts.parent_context_id", "contexts", "parent_context_id", "contexts", "context_id"),
            ("context_assignments.context_id", "context_assignments", "context_id", "contexts", "context_id"),
            ("context_assignments.source_span_id", "context_assignments", "source_span_id", "source_spans", "span_id"),
            ("context_carriers.context_id", "context_carriers", "context_id", "contexts", "context_id"),
            ("context_carriers.document_id", "context_carriers", "document_id", "documents", "document_id"),
            ("context_carriers.source_span_id", "context_carriers", "source_span_id", "source_spans", "span_id"),
            ("frames.context_id", "frames", "context_id", "contexts", "context_id"),
            ("frames.span_id", "frames", "span_id", "source_spans", "span_id"),
            ("frame_arguments.frame_id", "frame_arguments", "frame_id", "frames", "frame_id"),
            ("frame_arguments.mention_id", "frame_arguments", "mention_id", "mentions", "mention_id"),
            ("frame_arguments.referent_id", "frame_arguments", "referent_id", "referents", "referent_id"),
            ("drs_boxes.source_span_id", "drs_boxes", "source_span_id", "source_spans", "span_id"),
            ("drs_boxes.context_id", "drs_boxes", "context_id", "contexts", "context_id"),
            ("drs_boxes.parent_drs_box_id", "drs_boxes", "parent_drs_box_id", "drs_boxes", "drs_box_id"),
            ("drs_boxes.holder_referent_id", "drs_boxes", "holder_referent_id", "referents", "referent_id"),
            ("drs_referents.source_span_id", "drs_referents", "source_span_id", "source_spans", "span_id"),
            ("drs_referents.referent_id", "drs_referents", "referent_id", "referents", "referent_id"),
            ("drs_referents.box_id", "drs_referents", "box_id", "drs_boxes", "drs_box_id"),
            ("drs_conditions.source_span_id", "drs_conditions", "source_span_id", "source_spans", "span_id"),
            ("drs_conditions.box_id", "drs_conditions", "box_id", "drs_boxes", "drs_box_id"),
            ("drs_conditions.context_id", "drs_conditions", "context_id", "contexts", "context_id"),
            ("drs_conditions.frame_id", "drs_conditions", "frame_id", "frames", "frame_id"),
            ("drs_condition_arguments.drs_condition_id", "drs_condition_arguments", "drs_condition_id", "drs_conditions", "drs_condition_id"),
            ("drs_condition_arguments.referent_id", "drs_condition_arguments", "referent_id", "referents", "referent_id"),
            ("drs_condition_arguments.target_box_id", "drs_condition_arguments", "target_box_id", "drs_boxes", "drs_box_id"),
            ("drs_condition_arguments.target_condition_id", "drs_condition_arguments", "target_condition_id", "drs_conditions", "drs_condition_id"),
            ("drs_identity_hypotheses.source_span_id", "drs_identity_hypotheses", "source_span_id", "source_spans", "span_id"),
            ("drs_identity_hypotheses.context_id", "drs_identity_hypotheses", "context_id", "contexts", "context_id"),
            ("drs_identity_hypotheses.box_id", "drs_identity_hypotheses", "box_id", "drs_boxes", "drs_box_id"),
            ("drs_identity_hypotheses.left_referent_id", "drs_identity_hypotheses", "left_referent_id", "referents", "referent_id"),
            ("drs_identity_hypotheses.right_referent_id", "drs_identity_hypotheses", "right_referent_id", "referents", "referent_id"),
            ("relations.source_span_id", "relations", "source_span_id", "source_spans", "span_id"),
            ("relations.context_id", "relations", "context_id", "contexts", "context_id"),
            ("temporal_edges.source_span_id", "temporal_edges", "source_span_id", "source_spans", "span_id"),
            ("temporal_edges.referent_id", "temporal_edges", "referent_id", "referents", "referent_id"),
            ("temporal_edges.context_id", "temporal_edges", "context_id", "contexts", "context_id"),
            ("discourse_edges.document_id", "discourse_edges", "document_id", "documents", "document_id"),
            ("discourse_edges.source_span_id", "discourse_edges", "source_span_id", "source_spans", "span_id"),
            ("discourse_edges.from_context_id", "discourse_edges", "from_context_id", "contexts", "context_id"),
            ("discourse_edges.to_context_id", "discourse_edges", "to_context_id", "contexts", "context_id"),
            ("discourse_edges.from_span_id", "discourse_edges", "from_span_id", "source_spans", "span_id"),
            ("discourse_edges.to_span_id", "discourse_edges", "to_span_id", "source_spans", "span_id"),
            ("contexts.authority_source_span_id", "contexts", "authority_source_span_id", "source_spans", "span_id"),
        )
        errors: list[dict[str, Any]] = []
        for label, child_table, child_column, parent_table, parent_column in references:
            row = self.connection.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM {child_table} AS child
                LEFT JOIN {parent_table} AS parent
                  ON parent.{parent_column}=child.{child_column}
                WHERE NULLIF(CAST(child.{child_column} AS TEXT), '') IS NOT NULL
                  AND parent.{parent_column} IS NULL
                """
            ).fetchone()
            count = int(row["count"] if row is not None else 0)
            if count:
                errors.append({"reference": label, "count": count})
        return errors

    @staticmethod
    def _sql_placeholders(values: set[str]) -> str:
        return ",".join("?" for _value in values)

    def prune_stale_documents(self, run_id: str, active_document_ids: set[str]) -> dict[str, int]:
        """Physically remove source-derived rows for deleted or superseded files."""

        if active_document_ids:
            placeholders = self._sql_placeholders(active_document_ids)
            rows = self.connection.execute(
                f"SELECT document_id FROM documents WHERE run_id=? AND document_id NOT IN ({placeholders})",
                (run_id, *sorted(active_document_ids)),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT document_id FROM documents WHERE run_id=?",
                (run_id,),
            ).fetchall()
        document_ids = {str(row["document_id"]) for row in rows}
        if not document_ids:
            return {}
        document_placeholders = self._sql_placeholders(document_ids)
        chunk_ids = {
            str(row["chunk_id"])
            for row in self.connection.execute(
                f"SELECT chunk_id FROM chunks WHERE document_id IN ({document_placeholders})",
                tuple(sorted(document_ids)),
            ).fetchall()
        }
        span_ids = {
            str(row["span_id"])
            for row in self.connection.execute(
                f"SELECT span_id FROM source_spans WHERE document_id IN ({document_placeholders})",
                tuple(sorted(document_ids)),
            ).fetchall()
        }
        mention_ids: set[str] = set()
        frame_ids: set[str] = set()
        relation_ids: set[str] = set()
        condition_ids: set[str] = set()
        box_ids: set[str] = set()
        if span_ids:
            span_placeholders = self._sql_placeholders(span_ids)
            span_params = tuple(sorted(span_ids))
            mention_ids = {str(row["mention_id"]) for row in self.connection.execute(f"SELECT mention_id FROM mentions WHERE span_id IN ({span_placeholders})", span_params).fetchall()}
            frame_ids = {str(row["frame_id"]) for row in self.connection.execute(f"SELECT frame_id FROM frames WHERE span_id IN ({span_placeholders})", span_params).fetchall()}
            relation_ids = {str(row["relation_id"]) for row in self.connection.execute(f"SELECT relation_id FROM relations WHERE source_span_id IN ({span_placeholders})", span_params).fetchall()}
            condition_ids = {str(row["drs_condition_id"]) for row in self.connection.execute(f"SELECT drs_condition_id FROM drs_conditions WHERE source_span_id IN ({span_placeholders})", span_params).fetchall()}
            box_ids = {str(row["drs_box_id"]) for row in self.connection.execute(f"SELECT drs_box_id FROM drs_boxes WHERE source_span_id IN ({span_placeholders})", span_params).fetchall()}
        savepoint = stable_id("prune", run_id, *sorted(document_ids)).replace("-", "_")
        self.connection.execute(f"SAVEPOINT {savepoint}")
        deleted: dict[str, int] = {}

        def remove(table: str, where: str, params: tuple[Any, ...]) -> None:
            cursor = self.connection.execute(f"DELETE FROM {table} WHERE {where}", params)
            count = max(0, int(cursor.rowcount if cursor.rowcount is not None else 0))
            if count:
                deleted[table] = deleted.get(table, 0) + count

        try:
            if span_ids:
                sp = self._sql_placeholders(span_ids)
                params = tuple(sorted(span_ids))
                remove("context_assignments", f"source_span_id IN ({sp})", params)
                remove("discourse_edges", f"source_span_id IN ({sp}) OR from_span_id IN ({sp}) OR to_span_id IN ({sp})", params * 3)
                remove("temporal_edges", f"source_span_id IN ({sp})", params)
                remove("identity_hypotheses", f"source_span_id IN ({sp})", params)
                remove("drs_identity_hypotheses", f"source_span_id IN ({sp})", params)
                remove("relations", f"source_span_id IN ({sp})", params)
                remove("model_attempts", f"source_span_id IN ({sp})", params)
            applies_to_ids = mention_ids | frame_ids | relation_ids
            if applies_to_ids:
                ap = self._sql_placeholders(applies_to_ids)
                remove("context_assignments", f"applies_to_id IN ({ap})", tuple(sorted(applies_to_ids)))
            if condition_ids:
                cp = self._sql_placeholders(condition_ids)
                remove("drs_condition_arguments", f"drs_condition_id IN ({cp}) OR target_condition_id IN ({cp})", tuple(sorted(condition_ids)) * 2)
            if box_ids:
                bp = self._sql_placeholders(box_ids)
                remove("drs_condition_arguments", f"target_box_id IN ({bp})", tuple(sorted(box_ids)))
            if frame_ids or mention_ids:
                clauses: list[str] = []
                params_list: list[str] = []
                if frame_ids:
                    fp = self._sql_placeholders(frame_ids)
                    clauses.append(f"frame_id IN ({fp})")
                    params_list.extend(sorted(frame_ids))
                if mention_ids:
                    mp = self._sql_placeholders(mention_ids)
                    clauses.append(f"mention_id IN ({mp})")
                    params_list.extend(sorted(mention_ids))
                remove("frame_arguments", " OR ".join(clauses), tuple(params_list))
            if mention_ids:
                mp = self._sql_placeholders(mention_ids)
                remove("mention_referents", f"mention_id IN ({mp})", tuple(sorted(mention_ids)))
            if span_ids:
                sp = self._sql_placeholders(span_ids)
                params = tuple(sorted(span_ids))
                remove("drs_conditions", f"source_span_id IN ({sp})", params)
                remove("drs_referents", f"source_span_id IN ({sp})", params)
                remove("drs_boxes", f"source_span_id IN ({sp})", params)
                remove("frames", f"span_id IN ({sp})", params)
                remove("mentions", f"span_id IN ({sp})", params)
                remove("context_carriers", f"source_span_id IN ({sp})", params)
            remove("context_carriers", f"document_id IN ({document_placeholders})", tuple(sorted(document_ids)))
            remove("metadata_records", f"document_id IN ({document_placeholders})", tuple(sorted(document_ids)))
            if span_ids:
                sp = self._sql_placeholders(span_ids)
                remove("source_spans", f"span_id IN ({sp})", tuple(sorted(span_ids)))
            if chunk_ids:
                chp = self._sql_placeholders(chunk_ids)
                remove("chunks", f"chunk_id IN ({chp})", tuple(sorted(chunk_ids)))
            remove("documents", f"document_id IN ({document_placeholders})", tuple(sorted(document_ids)))
            remove(
                "referents",
                """
                run_id=?
                AND NOT EXISTS (SELECT 1 FROM mention_referents mr WHERE mr.referent_id=referents.referent_id)
                AND NOT EXISTS (SELECT 1 FROM drs_referents dr WHERE dr.referent_id=referents.referent_id)
                AND NOT EXISTS (SELECT 1 FROM frame_arguments fa WHERE fa.referent_id=referents.referent_id)
                AND NOT EXISTS (SELECT 1 FROM drs_condition_arguments da WHERE da.referent_id=referents.referent_id)
                AND NOT EXISTS (SELECT 1 FROM identity_hypotheses ih WHERE ih.left_referent_id=referents.referent_id OR ih.right_referent_id=referents.referent_id)
                AND NOT EXISTS (SELECT 1 FROM drs_identity_hypotheses dh WHERE dh.left_referent_id=referents.referent_id OR dh.right_referent_id=referents.referent_id)
                AND NOT EXISTS (SELECT 1 FROM temporal_edges te WHERE te.referent_id=referents.referent_id)
                AND NOT EXISTS (SELECT 1 FROM drs_boxes db WHERE db.holder_referent_id=referents.referent_id)
                """,
                (run_id,),
            )
            while True:
                cursor = self.connection.execute(
                    """
                    DELETE FROM contexts
                    WHERE run_id=?
                      AND NOT EXISTS (SELECT 1 FROM contexts child WHERE child.parent_context_id=contexts.context_id)
                      AND NOT EXISTS (SELECT 1 FROM context_assignments ca WHERE ca.context_id=contexts.context_id)
                      AND NOT EXISTS (SELECT 1 FROM context_carriers cc WHERE cc.context_id=contexts.context_id)
                      AND NOT EXISTS (SELECT 1 FROM frames f WHERE f.context_id=contexts.context_id)
                      AND NOT EXISTS (SELECT 1 FROM identity_hypotheses ih WHERE ih.context_id=contexts.context_id)
                      AND NOT EXISTS (SELECT 1 FROM drs_boxes db WHERE db.context_id=contexts.context_id)
                      AND NOT EXISTS (SELECT 1 FROM drs_conditions dc WHERE dc.context_id=contexts.context_id)
                      AND NOT EXISTS (SELECT 1 FROM drs_identity_hypotheses dh WHERE dh.context_id=contexts.context_id)
                      AND NOT EXISTS (SELECT 1 FROM relations r WHERE r.context_id=contexts.context_id)
                      AND NOT EXISTS (SELECT 1 FROM temporal_edges te WHERE te.context_id=contexts.context_id)
                    """,
                    (run_id,),
                )
                count = max(0, int(cursor.rowcount if cursor.rowcount is not None else 0))
                if count:
                    deleted["contexts"] = deleted.get("contexts", 0) + count
                if not count:
                    break
        except BaseException:
            self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            self.connection.commit()
        return deleted

    def finish_run(self, run_id: str, metrics: dict[str, Any]) -> None:
        integrity_errors = self.semantic_integrity_errors()
        if integrity_errors:
            self.connection.rollback()
            failed_metrics = {**metrics, "semantic_integrity_errors": integrity_errors}
            self.connection.execute(
                "UPDATE extraction_runs SET status=?, metrics_json=? WHERE run_id=?",
                ("failed", json.dumps(failed_metrics, sort_keys=True), run_id),
            )
            self.connection.commit()
            raise RuntimeError(
                "semantic integrity validation failed: "
                + json.dumps(integrity_errors, sort_keys=True)
            )
        self.connection.execute(
            "UPDATE extraction_runs SET status=?, metrics_json=? WHERE run_id=?",
            ("completed", json.dumps(metrics, sort_keys=True), run_id),
        )
        self.connection.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self.connection.execute(sql, params)

    def close(self) -> None:
        self.connection.close()

    def commit(self) -> None:
        self.connection.commit()

    def deactivate_other_model_attempt_materializations(
        self,
        run_id: str,
        source_span_id: str,
        task: str,
        source: str,
        active_cache_key: str,
    ) -> int:
        cursor = self.connection.execute(
            """
            UPDATE model_attempts
            SET materialized=0
            WHERE run_id=?
              AND source_span_id=?
              AND task=?
              AND source=?
              AND cache_key<>?
              AND materialized<>0
            """,
            (run_id, source_span_id, task, source, active_cache_key),
        )
        return max(0, int(cursor.rowcount if cursor.rowcount is not None else 0))

    def counts(self) -> dict[str, int]:
        tables = [
            "documents",
            "chunks",
            "source_spans",
            "mentions",
            "referents",
            "mention_referents",
            "identity_hypotheses",
            "contexts",
            "context_carriers",
            "context_assignments",
            "frames",
            "frame_arguments",
            "drs_boxes",
            "drs_referents",
            "drs_conditions",
            "drs_condition_arguments",
            "drs_identity_hypotheses",
            "temporal_edges",
            "relations",
            "metadata_records",
            "model_attempts",
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

    def delete_drs_materialization_for_span(
        self,
        run_id: str,
        source_span_id: str,
        *,
        source: str = "local_model_drs",
        commit: bool = True,
    ) -> dict[str, int]:
        """Remove one model DRS materialization without touching raw source rows.

        Re-ingesting a source chunk with a different model/cache fingerprint must
        not leave the old DRS live beside the new one.  The deletion is scoped to
        the exact run, source span, and DRS source, so deterministic relations,
        mentions, chunks, source spans, and unrelated model attempts remain
        intact.
        """

        def id_rows(sql: str, params: tuple[Any, ...]) -> list[str]:
            return [str(row[0]) for row in self.connection.execute(sql, params).fetchall() if str(row[0] or "")]

        evidence_span_kind = f"drs_evidence:{source_span_id}"
        evidence_span_ids = id_rows(
            "SELECT span_id FROM source_spans WHERE span_kind=?",
            (evidence_span_kind,),
        )
        evidence_chunk_ids = id_rows(
            "SELECT DISTINCT chunk_id FROM source_spans WHERE span_kind=?",
            (evidence_span_kind,),
        )

        frame_ids = id_rows(
            """
            SELECT frame_id
            FROM frames
            WHERE run_id=? AND span_id=? AND source=?
            """,
            (run_id, source_span_id, source),
        )
        condition_ids = id_rows(
            """
            SELECT drs_condition_id
            FROM drs_conditions
            WHERE run_id=? AND source_span_id=? AND source=?
            """,
            (run_id, source_span_id, source),
        )
        context_ids = id_rows(
            """
            SELECT context_id
            FROM drs_boxes
            WHERE run_id=? AND source_span_id=? AND source=?
            """,
            (run_id, source_span_id, source),
        )
        referent_ids = id_rows(
            """
            SELECT DISTINCT referent_id
            FROM drs_referents
            WHERE run_id=? AND source_span_id=? AND source=?
            """,
            (run_id, source_span_id, source),
        )
        relation_ids = [
            stable_id("rel", run_id, condition_id, "drs_condition")
            for condition_id in condition_ids
        ]

        deleted: dict[str, int] = {}

        def delete_where(table: str, where: str, params: tuple[Any, ...]) -> None:
            cursor = self.connection.execute(f"DELETE FROM {table} WHERE {where}", params)
            deleted[table] = deleted.get(table, 0) + max(0, int(cursor.rowcount if cursor.rowcount is not None else 0))

        def delete_by_ids(table: str, key: str, ids: list[str]) -> None:
            unique_ids = list(dict.fromkeys(item for item in ids if item))
            for index in range(0, len(unique_ids), 400):
                group = unique_ids[index:index + 400]
                placeholders = ",".join("?" for _ in group)
                delete_where(table, f"{key} IN ({placeholders})", tuple(group))

        def delete_orphan_referents(ids: list[str]) -> None:
            unique_ids = list(dict.fromkeys(item for item in ids if item))
            for index in range(0, len(unique_ids), 400):
                group = unique_ids[index:index + 400]
                placeholders = ",".join("?" for _ in group)
                delete_where(
                    "referents",
                    f"""
                    referent_id IN ({placeholders})
                    AND NOT EXISTS (
                      SELECT 1 FROM drs_referents dr
                      WHERE dr.referent_id=referents.referent_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM frame_arguments fa
                      WHERE fa.referent_id=referents.referent_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM mention_referents mr
                      WHERE mr.referent_id=referents.referent_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM identity_hypotheses ih
                      WHERE ih.left_referent_id=referents.referent_id
                         OR ih.right_referent_id=referents.referent_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM drs_identity_hypotheses dih
                      WHERE dih.left_referent_id=referents.referent_id
                         OR dih.right_referent_id=referents.referent_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM temporal_edges te
                      WHERE te.referent_id=referents.referent_id
                    )
                    """,
                    tuple(group),
                )

        delete_by_ids("frame_arguments", "frame_id", frame_ids)
        delete_by_ids("drs_condition_arguments", "drs_condition_id", condition_ids)
        delete_by_ids("relations", "relation_id", relation_ids)
        if context_ids:
            placeholders = ",".join("?" for _ in context_ids)
            delete_where(
                "temporal_edges",
                f"run_id=? AND source_span_id=? AND context_id IN ({placeholders})",
                (run_id, source_span_id, *context_ids),
            )
        delete_where(
            "identity_hypotheses",
            "run_id=? AND source_span_id=? AND source=?",
            (run_id, source_span_id, source),
        )
        delete_where(
            "drs_identity_hypotheses",
            "run_id=? AND source_span_id=? AND source=?",
            (run_id, source_span_id, source),
        )
        delete_by_ids("identity_hypotheses", "source_span_id", evidence_span_ids)
        delete_by_ids("drs_identity_hypotheses", "source_span_id", evidence_span_ids)
        delete_where(
            "drs_conditions",
            "run_id=? AND source_span_id=? AND source=?",
            (run_id, source_span_id, source),
        )
        delete_where(
            "frames",
            "run_id=? AND span_id=? AND source=?",
            (run_id, source_span_id, source),
        )
        delete_where(
            "drs_referents",
            "run_id=? AND source_span_id=? AND source=?",
            (run_id, source_span_id, source),
        )
        delete_where(
            "drs_boxes",
            "run_id=? AND source_span_id=? AND source=?",
            (run_id, source_span_id, source),
        )
        delete_by_ids("contexts", "context_id", context_ids)
        delete_by_ids("source_spans", "span_id", evidence_span_ids)
        delete_by_ids("chunks", "chunk_id", evidence_chunk_ids)
        delete_orphan_referents(referent_ids)
        if commit:
            self.connection.commit()
        return {table: count for table, count in deleted.items() if count}

    def delete_frame_materialization_for_span(
        self,
        run_id: str,
        source_span_id: str,
        *,
        source: str = "local_model",
    ) -> dict[str, int]:
        """Remove one model-frame materialization for a source span."""

        def id_rows(sql: str, params: tuple[Any, ...]) -> list[str]:
            return [str(row[0]) for row in self.connection.execute(sql, params).fetchall() if str(row[0] or "")]

        frame_rows = [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT frame_id, context_id
                FROM frames
                WHERE run_id=? AND span_id=? AND source=?
                """,
                (run_id, source_span_id, source),
            ).fetchall()
        ]
        frame_ids = [str(row.get("frame_id") or "") for row in frame_rows if str(row.get("frame_id") or "")]
        context_ids = [str(row.get("context_id") or "") for row in frame_rows if str(row.get("context_id") or "")]
        referent_ids: list[str] = []
        if frame_ids:
            for index in range(0, len(frame_ids), 400):
                group = frame_ids[index:index + 400]
                placeholders = ",".join("?" for _ in group)
                referent_ids.extend(
                    id_rows(
                        f"""
                        SELECT DISTINCT referent_id
                        FROM frame_arguments
                        WHERE frame_id IN ({placeholders}) AND referent_id IS NOT NULL
                        """,
                        tuple(group),
                    )
                )
        relation_ids = id_rows(
            """
            SELECT relation_id
            FROM relations
            WHERE run_id=? AND source_span_id=? AND metadata_json LIKE ?
            """,
            (run_id, source_span_id, f'%"source": "{source}"%'),
        )
        deleted: dict[str, int] = {}

        def delete_where(table: str, where: str, params: tuple[Any, ...]) -> None:
            cursor = self.connection.execute(f"DELETE FROM {table} WHERE {where}", params)
            deleted[table] = deleted.get(table, 0) + max(0, int(cursor.rowcount if cursor.rowcount is not None else 0))

        def delete_by_ids(table: str, key: str, ids: list[str]) -> None:
            unique_ids = list(dict.fromkeys(item for item in ids if item))
            for index in range(0, len(unique_ids), 400):
                group = unique_ids[index:index + 400]
                placeholders = ",".join("?" for _ in group)
                delete_where(table, f"{key} IN ({placeholders})", tuple(group))

        def delete_orphan_contexts(ids: list[str]) -> None:
            unique_ids = list(dict.fromkeys(item for item in ids if item))
            for index in range(0, len(unique_ids), 400):
                group = unique_ids[index:index + 400]
                placeholders = ",".join("?" for _ in group)
                delete_where(
                    "contexts",
                    f"""
                    context_id IN ({placeholders})
                    AND NOT EXISTS (
                      SELECT 1 FROM frames f
                      WHERE f.context_id=contexts.context_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM relations r
                      WHERE r.context_id=contexts.context_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM identity_hypotheses ih
                      WHERE ih.context_id=contexts.context_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM temporal_edges te
                      WHERE te.context_id=contexts.context_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM context_carriers cc
                      WHERE cc.context_id=contexts.context_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM context_assignments ca
                      WHERE ca.context_id=contexts.context_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM drs_boxes box
                      WHERE box.context_id=contexts.context_id
                    )
                    """,
                    tuple(group),
                )

        def delete_orphan_referents(ids: list[str]) -> None:
            unique_ids = list(dict.fromkeys(item for item in ids if item))
            for index in range(0, len(unique_ids), 400):
                group = unique_ids[index:index + 400]
                placeholders = ",".join("?" for _ in group)
                delete_where(
                    "referents",
                    f"""
                    referent_id IN ({placeholders})
                    AND NOT EXISTS (
                      SELECT 1 FROM drs_referents dr
                      WHERE dr.referent_id=referents.referent_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM frame_arguments fa
                      WHERE fa.referent_id=referents.referent_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM mention_referents mr
                      WHERE mr.referent_id=referents.referent_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM identity_hypotheses ih
                      WHERE ih.left_referent_id=referents.referent_id
                         OR ih.right_referent_id=referents.referent_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM drs_identity_hypotheses dih
                      WHERE dih.left_referent_id=referents.referent_id
                         OR dih.right_referent_id=referents.referent_id
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM temporal_edges te
                      WHERE te.referent_id=referents.referent_id
                    )
                    """,
                    tuple(group),
                )

        delete_by_ids("frame_arguments", "frame_id", frame_ids)
        delete_by_ids("relations", "relation_id", relation_ids)
        identity_source = "local_model_frame" if source == "local_model" else source
        delete_where(
            "identity_hypotheses",
            "run_id=? AND source_span_id=? AND source=?",
            (run_id, source_span_id, identity_source),
        )
        if context_ids:
            placeholders = ",".join("?" for _ in context_ids)
            delete_where(
                "temporal_edges",
                f"run_id=? AND source_span_id=? AND relation=? AND context_id IN ({placeholders})",
                (run_id, source_span_id, "frame_temporal_scope", *context_ids),
            )
        delete_where(
            "frames",
            "run_id=? AND span_id=? AND source=?",
            (run_id, source_span_id, source),
        )
        delete_orphan_contexts(context_ids)
        delete_orphan_referents(referent_ids)
        self.connection.commit()
        return {table: count for table, count in deleted.items() if count}

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

    def relation_candidate_chunks(
        self,
        run_id: str,
        predicates: list[str] | None = None,
        anchors: list[str] | None = None,
        limit: int = 20,
    ) -> list[sqlite3.Row]:
        predicate_norms = [normalize(predicate) for predicate in (predicates or []) if normalize(predicate)]
        params: list[Any] = [run_id]
        predicate_filter = ""
        if predicate_norms:
            placeholders = ",".join("?" for _ in predicate_norms)
            predicate_filter = f"AND r.predicate_norm IN ({placeholders})"
            params.extend(predicate_norms)
        params.append(limit * 8)
        rows = self.connection.execute(
            f"""
            SELECT
              d.rel_path,
              c.chunk_order,
              c.text,
              r.relation_type,
              r.subject,
              r.predicate,
              r.object,
              r.value,
              r.confidence,
              ctx.kind AS context_kind
            FROM relations r
            JOIN source_spans s ON s.span_id = r.source_span_id
            JOIN chunks c ON c.chunk_id = s.chunk_id
            JOIN documents d ON d.document_id = c.document_id
            LEFT JOIN contexts ctx ON ctx.context_id = r.context_id
            WHERE r.run_id = ? {predicate_filter}
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        anchor_norms = [normalize(anchor) for anchor in (anchors or []) if len(normalize(anchor)) > 1]
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            material = " ".join(
                str(row[key] or "")
                for key in ["text", "subject", "predicate", "object", "value", "relation_type"]
            )
            material_norm = normalize(material)
            anchor_hits = sum(1 for anchor in anchor_norms if anchor and anchor in material_norm)
            predicate_hits = sum(1 for predicate in predicate_norms if predicate and predicate in material_norm)
            if anchor_norms and not anchor_hits and not predicate_hits:
                continue
            score = float(row["confidence"] or 0.0)
            score += anchor_hits * 2.5
            score += predicate_hits * 1.5
            scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], str(item[1]["rel_path"]), int(item[1]["chunk_order"])))
        return [row for _, row in scored[:limit]]

    def materialize_drs_payload(
        self,
        run_id: str,
        source_span_id: str,
        source_text: str,
        payload: dict[str, Any],
        *,
        source: str = "local_model_drs",
    ) -> dict[str, Any]:
        """Persist one model-supplied DRS without interpreting raw language.

        The method only validates structure, exact source grounding, references,
        and provenance.  Semantic commitments must already be present as DRS
        boxes, referents, conditions, arguments, temporal records, and identity
        hypotheses in the model payload.
        """

        drs = payload.get("drs") if isinstance(payload, dict) else None
        if not isinstance(drs, dict):
            return {"accepted": False, "reason": "missing_drs_object", "inserted": {}}
        working_payload = copy.deepcopy(payload)
        working_drs = working_payload["drs"]
        normalized_conditions: list[dict[str, Any]] = []
        for condition in working_drs.get("conditions") or []:
            if not isinstance(condition, dict):
                continue
            normalized = copy.deepcopy(condition)
            predicate, polarity = normalize_predicate_polarity(
                str(normalized.get("predicate") or ""),
                str(normalized.get("polarity") or "positive"),
            )
            normalized["predicate"] = predicate
            normalized["polarity"] = polarity
            normalized_conditions.append(normalized)
        working_drs["conditions"] = normalized_conditions
        working_drs["identity_hypotheses"] = [
            copy.deepcopy(identity)
            for identity in working_drs.get("identity_hypotheses") or []
            if isinstance(identity, dict)
            and str(identity.get("left_referent_id") or "")
            != str(identity.get("right_referent_id") or "")
        ]
        payload = working_payload
        drs = working_drs
        # The store is a trust boundary too.  Re-run the authoritative source-
        # locality validator here so direct callers cannot bypass the planner's
        # grounding gate before persistence.
        from .model_planner import _validate_chunk_drs_payload

        authoritative_validation = _validate_chunk_drs_payload(payload, source_text)
        if not authoritative_validation.get("schema_valid"):
            return {
                "accepted": False,
                "reason": (
                    "grounding_validation_failed"
                    if authoritative_validation.get("grounding_failures")
                    else "schema_validation_failed"
                ),
                "errors": list(authoritative_validation.get("errors") or [])[:50],
                "grounding_failures": list(authoritative_validation.get("grounding_failures") or [])[:50],
                "inserted": {},
            }
        errors: list[str] = []
        grounding_failures: list[str] = []

        def as_list(key: str) -> list[dict[str, Any]]:
            value = drs.get(key)
            if not isinstance(value, list):
                errors.append(f"not_list:{key}")
                return []
            return [item for item in value if isinstance(item, dict)]

        referents = as_list("referents")
        boxes = as_list("boxes")
        conditions = as_list("conditions")
        identities = as_list("identity_hypotheses")
        temporals = as_list("temporal_records")
        evidence_spans = drs.get("evidence_spans", [])
        if evidence_spans is None:
            evidence_spans = []
        if not isinstance(evidence_spans, list):
            errors.append("not_list:evidence_spans")
            evidence_spans = []

        def text_value(item: dict[str, Any], key: str) -> str:
            return str(item.get(key) or "").strip()

        def check_grounding(value: Any, label: str) -> None:
            span = str(value or "").strip()
            if span and span not in source_text:
                grounding_failures.append(f"{label}:{span[:100]}")

        for span in evidence_spans:
            check_grounding(span, "evidence_spans")

        referent_labels_by_id = {text_value(item, "id"): text_value(item, "label") for item in referents}
        referent_evidence_by_id = {text_value(item, "id"): text_value(item, "evidence_text") for item in referents}

        referent_ids = {text_value(item, "id") for item in referents if text_value(item, "id")}
        box_ids = {text_value(item, "id") for item in boxes if text_value(item, "id")}
        condition_ids = {text_value(item, "id") for item in conditions if text_value(item, "id")}
        temporal_ids = {text_value(item, "id") for item in temporals if text_value(item, "id")}
        if len(referent_ids) != len([item for item in referents if text_value(item, "id")]):
            errors.append("duplicate_or_missing_referent_id")
        if len(box_ids) != len([item for item in boxes if text_value(item, "id")]):
            errors.append("duplicate_or_missing_box_id")
        if len(condition_ids) != len([item for item in conditions if text_value(item, "id")]):
            errors.append("duplicate_or_missing_condition_id")
        if not box_ids:
            errors.append("missing_box")

        for item in referents:
            if not text_value(item, "id"):
                errors.append("referent_missing_id")
            if not text_value(item, "label"):
                errors.append(f"referent_missing_label:{text_value(item, 'id')}")
            check_grounding(item.get("evidence_text"), f"referent:{text_value(item, 'id')}")
        for item in boxes:
            box_id = text_value(item, "id")
            parent_id = text_value(item, "parent_id")
            holder_id = text_value(item, "holder_referent_id")
            kind = text_value(item, "kind")
            if not box_id:
                errors.append("box_missing_id")
            if kind not in DRS_CONTEXT_KINDS:
                errors.append(f"bad_box_kind:{box_id}:{item.get('kind')}")
            if parent_id and parent_id not in box_ids:
                errors.append(f"missing_parent_box:{box_id}->{parent_id}")
            if parent_id and parent_id == box_id:
                errors.append(f"self_parent_box:{box_id}")
            if holder_id and holder_id not in referent_ids:
                errors.append(f"missing_holder_referent:{box_id}->{holder_id}")
            check_grounding(item.get("evidence_text"), f"box:{box_id}")
        errors.extend(box_root_errors(boxes))
        errors.extend(box_parent_cycle_errors(boxes))
        for item in temporals:
            temporal_id = text_value(item, "id")
            if not temporal_id:
                errors.append("temporal_missing_id")
            if not text_value(item, "value"):
                errors.append(f"temporal_missing_value:{temporal_id}")
            check_grounding(item.get("evidence_text"), f"temporal:{temporal_id}")
        for item in conditions:
            condition_id = text_value(item, "id")
            box_id = text_value(item, "box_id")
            temporal_id = text_value(item, "temporal_id")
            if not condition_id:
                errors.append("condition_missing_id")
            if not text_value(item, "predicate"):
                errors.append(f"condition_missing_predicate:{condition_id}")
            if box_id not in box_ids:
                errors.append(f"missing_condition_box:{condition_id}->{box_id}")
            if text_value(item, "polarity") not in DRS_POLARITIES:
                errors.append(f"bad_polarity:{condition_id}:{item.get('polarity')}")
            if text_value(item, "modality") not in DRS_CONTEXT_KINDS:
                errors.append(f"bad_modality:{condition_id}:{item.get('modality')}")
            if temporal_id and temporal_id not in temporal_ids:
                errors.append(f"missing_temporal:{condition_id}->{temporal_id}")
            check_grounding(item.get("evidence_text"), f"condition:{condition_id}")
            args = item.get("arguments")
            if not isinstance(args, list):
                errors.append(f"condition_arguments_not_list:{condition_id}")
                continue
            for arg in args:
                if not isinstance(arg, dict):
                    errors.append(f"condition_argument_not_object:{condition_id}")
                    continue
                target_kind = text_value(arg, "target_kind")
                target_id = text_value(arg, "target_id")
                if target_kind == "referent" and target_id and target_id not in referent_ids:
                    errors.append(f"missing_argument_referent:{condition_id}->{target_id}")
                elif target_kind == "box" and target_id and target_id not in box_ids:
                    errors.append(f"missing_argument_box:{condition_id}->{target_id}")
                elif target_kind == "box" and target_id and target_id == box_id:
                    errors.append(f"self_argument_box:{condition_id}->{target_id}")
                elif target_kind == "condition" and target_id and target_id not in condition_ids:
                    errors.append(f"missing_argument_condition:{condition_id}->{target_id}")
                elif target_kind == "condition" and target_id and target_id == condition_id:
                    errors.append(f"self_argument_condition:{condition_id}->{target_id}")
                elif target_kind in {"literal", "unknown"} and target_id:
                    errors.append(f"literal_argument_has_target_id:{condition_id}->{target_id}")
                elif target_kind not in {"referent", "box", "condition", "literal", "unknown"}:
                    errors.append(f"bad_argument_target_kind:{condition_id}:{target_kind}")
                check_grounding(arg.get("evidence_text"), f"argument:{condition_id}:{text_value(arg, 'role')}")
        errors.extend(condition_argument_cycle_errors(conditions))
        for item in identities:
            left_id = text_value(item, "left_referent_id")
            right_id = text_value(item, "right_referent_id")
            identity_box_id = text_value(item, "box_id")
            if left_id not in referent_ids:
                errors.append(f"missing_identity_left:{left_id}")
            if right_id not in referent_ids:
                errors.append(f"missing_identity_right:{right_id}")
            if identity_box_id and identity_box_id not in box_ids:
                errors.append(f"missing_identity_box:{left_id}:{right_id}->{identity_box_id}")
            evidence = text_value(item, "evidence_text")
            if left_id and right_id and left_id != right_id:
                left_surfaces = [
                    surface
                    for surface in [
                        referent_labels_by_id.get(left_id, ""),
                        referent_evidence_by_id.get(left_id, ""),
                    ]
                    if surface
                ]
                right_surfaces = [
                    surface
                    for surface in [
                        referent_labels_by_id.get(right_id, ""),
                        referent_evidence_by_id.get(right_id, ""),
                    ]
                    if surface
                ]
                if normalize(referent_labels_by_id.get(left_id, "")) == normalize(referent_labels_by_id.get(right_id, "")):
                    errors.append(f"identity_evidence_ambiguous_same_surface:{left_id}:{right_id}")
                elif not any(surface in evidence for surface in left_surfaces) or not any(
                    surface in evidence for surface in right_surfaces
                ):
                    errors.append(f"identity_evidence_missing_side:{left_id}:{right_id}")
            check_grounding(item.get("evidence_text"), f"identity:{left_id}:{right_id}")

        if errors or grounding_failures:
            return {
                "accepted": False,
                "reason": "schema_validation_failed" if errors else "grounding_validation_failed",
                "errors": errors[:50],
                "grounding_failures": grounding_failures[:50],
                "inserted": {},
            }

        def confidence(value: Any, default: float = 0.65) -> float:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return default

        savepoint_name = stable_id("savepoint", run_id, source_span_id, source).replace("-", "_")
        self.connection.execute(f"SAVEPOINT {savepoint_name}")
        replaced: dict[str, int] = {}
        try:
            replaced = self.delete_drs_materialization_for_span(
                run_id,
                source_span_id,
                source=source,
                commit=False,
            )
            parent_span = self.connection.execute(
                """
                SELECT ss.document_id, ss.chunk_id, ss.char_start, ss.char_end,
                       c.chunk_order
                FROM source_spans ss
                JOIN chunks c ON c.chunk_id=ss.chunk_id
                WHERE ss.span_id=?
                LIMIT 1
                """,
                (source_span_id,),
            ).fetchone()

            def evidence_source_span(evidence: str) -> str:
                """Create precise provenance inside a packed model-input chunk."""
                value = str(evidence or "").strip()
                if not value or value == source_text or parent_span is None:
                    return source_span_id
                offset = source_text.find(value)
                if offset < 0:
                    return source_span_id
                unit_start, unit_end, unit_text = offset, offset + len(value), value
                for candidate_start, candidate_end, candidate_text in split_units(source_text):
                    if candidate_start <= offset and offset + len(value) <= candidate_end:
                        unit_start, unit_end, unit_text = candidate_start, candidate_end, candidate_text
                        break
                if unit_start == 0 and unit_end == len(source_text):
                    return source_span_id
                absolute_start = int(parent_span["char_start"]) + unit_start
                absolute_end = int(parent_span["char_start"]) + unit_end
                child_chunk_id = stable_id(
                    "drs_evidence_chunk", source_span_id, absolute_start, absolute_end, unit_text
                )
                child_span_id = stable_id(
                    "drs_evidence_span", source_span_id, absolute_start, absolute_end, unit_text
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO chunks(
                      chunk_id, document_id, chunk_order, char_start, char_end, text, token_estimate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        child_chunk_id,
                        str(parent_span["document_id"]),
                        int(parent_span["chunk_order"]),
                        absolute_start,
                        absolute_end,
                        unit_text,
                        max(1, len(unit_text.split())),
                    ),
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO source_spans(
                      span_id, document_id, chunk_id, char_start, char_end,
                      surface, surface_norm, span_kind
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        child_span_id,
                        str(parent_span["document_id"]),
                        child_chunk_id,
                        absolute_start,
                        absolute_end,
                        unit_text,
                        normalize(unit_text),
                        f"drs_evidence:{source_span_id}",
                    ),
                )
                return child_span_id

            external_to_referent: dict[str, str] = {}
            external_to_drs_referent: dict[str, str] = {}
            external_to_referent_label: dict[str, str] = {}
            last_non_demonstrative_referent_id: str | None = None
            demonstratives = {"this", "that", "these", "those", "it"}
            for item in referents:
                external_id = text_value(item, "id")
                label = text_value(item, "label")
                value_type = text_value(item, "kind") or text_value(item, "value_type") or "unknown"
                label_norm = normalize(label)
                if label_norm in demonstratives and last_non_demonstrative_referent_id:
                    referent_id = last_non_demonstrative_referent_id
                else:
                    referent_id = stable_id("ref", run_id, source_span_id, external_id, label_norm, value_type)
                    if label_norm not in demonstratives:
                        last_non_demonstrative_referent_id = referent_id
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO referents(
                      referent_id, run_id, canonical_label, canonical_label_norm, entity_type, status, attributes_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        referent_id,
                        run_id,
                        label,
                        normalize(label),
                        value_type,
                        "candidate",
                        json.dumps(
                            {
                                "source": source,
                                "source_span_id": source_span_id,
                                "external_referent_id": external_id,
                            },
                            sort_keys=True,
                        ),
                    ),
                )
                drs_referent_id = stable_id("drsref", run_id, source_span_id, external_id, label)
                external_to_referent[external_id] = referent_id
                external_to_drs_referent[external_id] = drs_referent_id
                external_to_referent_label[external_id] = label
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO drs_referents(
                      drs_referent_id, run_id, source_span_id, external_referent_id, referent_id, box_id,
                      surface, surface_norm, value_type, evidence_surface, confidence, source, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        drs_referent_id,
                        run_id,
                        source_span_id,
                        external_id,
                        referent_id,
                        None,
                        label,
                        normalize(label),
                        value_type,
                        text_value(item, "evidence_text"),
                        confidence(item.get("confidence"), 0.65),
                        source,
                        json.dumps({"model_referent": item}, sort_keys=True),
                    ),
                )

            temporal_values: dict[str, dict[str, Any]] = {text_value(item, "id"): item for item in temporals}
            external_to_box: dict[str, str] = {}
            external_to_context: dict[str, str] = {}
            external_to_box_evidence: dict[str, str] = {}
            external_to_box_kind: dict[str, str] = {}
            root_asserted_box_ids = [
                text_value(item, "id")
                for item in boxes
                if text_value(item, "kind") == "asserted" and not text_value(item, "parent_id")
            ]
            has_contextual_boxes = any(
                text_value(item, "parent_id") or text_value(item, "kind") not in {"", "asserted"}
                for item in boxes
            )
            for item in boxes:
                external_id = text_value(item, "id")
                kind = text_value(item, "kind") or "asserted"
                evidence = text_value(item, "evidence_text")
                external_to_context[external_id] = stable_id(
                    "ctx", run_id, "drs_box", source_span_id, external_id, kind, evidence
                )
                external_to_box[external_id] = stable_id(
                    "drsbox", run_id, source_span_id, external_id, kind, evidence
                )
                external_to_box_evidence[external_id] = evidence
                external_to_box_kind[external_id] = kind

            for item in boxes:
                external_id = text_value(item, "id")
                kind = text_value(item, "kind") or "asserted"
                parent_external = text_value(item, "parent_id")
                holder_external = text_value(item, "holder_referent_id")
                evidence = text_value(item, "evidence_text")
                context_id = external_to_context[external_id]
                drs_box_id = external_to_box[external_id]
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO contexts(
                      context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        context_id,
                        run_id,
                        f"drs:{kind}",
                        external_to_context.get(parent_external),
                        external_to_referent_label.get(holder_external) or holder_external or None,
                        evidence or kind,
                        confidence(item.get("confidence"), 0.75),
                    ),
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO drs_boxes(
                      drs_box_id, run_id, source_span_id, external_box_id, context_id, parent_drs_box_id,
                      parent_external_box_id, kind, holder_referent_id, holder_external_referent_id,
                      evidence_surface, confidence, source, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        drs_box_id,
                        run_id,
                        source_span_id,
                        external_id,
                        context_id,
                        external_to_box.get(parent_external),
                        parent_external or None,
                        kind,
                        external_to_referent.get(holder_external),
                        holder_external or None,
                        evidence,
                        confidence(item.get("confidence"), 0.75),
                        source,
                        json.dumps({"model_box": item}, sort_keys=True),
                    ),
                )

            external_to_condition: dict[str, str] = {}
            external_to_condition_evidence: dict[str, str] = {}
            for item in conditions:
                external_id = text_value(item, "id")
                predicate, condition_polarity = normalize_predicate_polarity(
                    text_value(item, "predicate"),
                    text_value(item, "polarity") or "positive",
                )
                evidence = text_value(item, "evidence_text")
                if external_id:
                    external_to_condition[external_id] = stable_id(
                        "drscond", run_id, source_span_id, external_id, predicate, evidence
                    )
                    external_to_condition_evidence[external_id] = evidence

            def condition_targets_identity_sides(
                condition: dict[str, Any],
                left_external: str,
                right_external: str,
            ) -> bool:
                target_ids = {
                    text_value(arg, "target_id")
                    for arg in condition.get("arguments") or []
                    if isinstance(arg, dict) and text_value(arg, "target_kind") == "referent"
                }
                return bool(left_external and right_external and left_external in target_ids and right_external in target_ids)

            def resolved_argument_surface(arg: dict[str, Any]) -> str:
                value = text_value(arg, "value")
                if value:
                    return value
                target_kind = text_value(arg, "target_kind")
                target_external = text_value(arg, "target_id")
                if target_kind == "referent":
                    return external_to_referent_label.get(target_external, "")
                if target_kind == "box":
                    return external_to_box_evidence.get(target_external, "")
                if target_kind == "condition":
                    return external_to_condition_evidence.get(target_external, "")
                return text_value(arg, "evidence_text")

            fictional_cue_pattern = re.compile(
                r"\b(?:fiction(?:al)?|story|novel|imaginary|make[- ]?believe|made[- ]?up|pretend(?:ed|ing)?)\b",
                re.I,
            )

            def effective_condition_modality(item: dict[str, Any], box_external: str) -> str:
                raw_modality = text_value(item, "modality") or "asserted"
                if raw_modality != "fictional":
                    return raw_modality
                if external_to_box_kind.get(box_external, "asserted") != "asserted":
                    return raw_modality
                grounding_material = " ".join(
                    [
                        text_value(item, "evidence_text"),
                        external_to_box_evidence.get(box_external, ""),
                    ]
                )
                # Fictionality is a source-scoped claim. If the governing box is
                # asserted and neither the condition nor box evidence contains a
                # fiction cue, keep the model's raw label for audit but do not let
                # that unsupported label make an asserted fact inaccessible.
                return "fictional" if fictional_cue_pattern.search(grounding_material) else "asserted"

            inserted_arguments = 0
            for item in conditions:
                external_id = text_value(item, "id")
                predicate, condition_polarity = normalize_predicate_polarity(
                    text_value(item, "predicate"),
                    text_value(item, "polarity") or "positive",
                )
                box_external = text_value(item, "box_id")
                context_id = external_to_context[box_external]
                effective_modality = effective_condition_modality(item, box_external)
                temporal_id = text_value(item, "temporal_id")
                temporal_text = text_value(temporal_values.get(temporal_id, {}), "value") if temporal_id else ""
                evidence = text_value(item, "evidence_text")
                condition_id = external_to_condition[external_id]
                frame_id = stable_id("frm", run_id, source_span_id, "drs", external_id, predicate, evidence)
                condition_confidence = confidence(item.get("confidence"), 0.65)
                self.connection.execute(
                    "INSERT OR IGNORE INTO frames(frame_id, run_id, context_id, predicate, predicate_norm, trigger_surface, confidence, source, span_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        frame_id,
                        run_id,
                        context_id,
                        predicate,
                        normalize(predicate),
                        predicate,
                        condition_confidence,
                        source,
                        source_span_id,
                    ),
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO drs_conditions(
                      drs_condition_id, run_id, source_span_id, external_condition_id, box_id, context_id, frame_id,
                      predicate, predicate_norm, polarity, modality, temporal_id, temporal_text, evidence_surface,
                      confidence, source, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        condition_id,
                        run_id,
                        source_span_id,
                        external_id,
                        external_to_box[box_external],
                        context_id,
                        frame_id,
                        predicate,
                        normalize(predicate),
                        condition_polarity,
                        text_value(item, "modality") or "asserted",
                        temporal_id or None,
                        temporal_text,
                        evidence,
                        condition_confidence,
                        source,
                        json.dumps({"model_condition": item}, sort_keys=True),
                    ),
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO relations(
                      relation_id, run_id, relation_type, subject, subject_norm, predicate, predicate_norm,
                      object, object_norm, value, value_norm, source_span_id, context_id, confidence, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("rel", run_id, condition_id, "drs_condition"),
                        run_id,
                        "drs_condition",
                        effective_modality,
                        normalize(effective_modality),
                        predicate,
                        normalize(predicate),
                        condition_polarity,
                        normalize(condition_polarity),
                        evidence,
                        normalize(evidence),
                        source_span_id,
                        context_id,
                        condition_confidence,
                        json.dumps(
                            {
                                "source": source,
                                "external_condition_id": external_id,
                                "external_box_id": box_external,
                                "raw_condition_modality": text_value(item, "modality") or "asserted",
                                "effective_condition_modality": effective_modality,
                            },
                            sort_keys=True,
                        ),
                    ),
                )
                temporal_edge_values: list[str] = []
                temporal_edge_referent_ids: list[str] = []
                for arg_index, arg in enumerate(item.get("arguments") or []):
                    if not isinstance(arg, dict):
                        continue
                    role = text_value(arg, "role") or "argument"
                    target_kind = text_value(arg, "target_kind") or "unknown"
                    target_external = text_value(arg, "target_id")
                    value = text_value(arg, "value")
                    value_type = text_value(arg, "value_type") or "unknown"
                    referent_id = external_to_referent.get(target_external) if target_kind == "referent" else None
                    if temporal_text and referent_id:
                        temporal_edge_referent_ids.append(referent_id)
                    argument_surface = resolved_argument_surface(arg)
                    if temporal_text and target_kind in {"literal", "unknown"} and value:
                        temporal_edge_values.append(value)
                    argument_id = stable_id("drsarg", run_id, condition_id, arg_index, role, target_kind, target_external, value)
                    self.connection.execute(
                        """
                        INSERT OR IGNORE INTO drs_condition_arguments(
                          drs_argument_id, run_id, drs_condition_id, role, target_kind, target_external_id,
                          referent_id, target_box_id, target_condition_id, value, value_norm, value_type,
                          evidence_surface, confidence, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            argument_id,
                            run_id,
                            condition_id,
                            role,
                            target_kind,
                            target_external or None,
                            referent_id,
                            external_to_box.get(target_external) if target_kind == "box" else None,
                            external_to_condition.get(target_external) if target_kind == "condition" else None,
                            value,
                            normalize(value),
                            value_type,
                            text_value(arg, "evidence_text"),
                            condition_confidence,
                            json.dumps({"model_argument": arg}, sort_keys=True),
                        ),
                    )
                    self.connection.execute(
                        "INSERT OR IGNORE INTO frame_arguments(argument_id, frame_id, role, mention_id, referent_id, surface, value_type, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            stable_id("arg", frame_id, arg_index, role, target_kind, target_external, value),
                            frame_id,
                            role,
                            None,
                            referent_id,
                            argument_surface,
                            value_type,
                            condition_confidence,
                        ),
                    )
                    inserted_arguments += 1
                if temporal_text:
                    edge_values = list(dict.fromkeys(temporal_edge_values)) or [evidence]
                    edge_referents = list(dict.fromkeys(temporal_edge_referent_ids)) or [None]
                    for edge_index, state_value in enumerate(edge_values):
                        for edge_referent_id in edge_referents:
                            self.connection.execute(
                                """
                                INSERT OR IGNORE INTO temporal_edges(
                                  edge_id, run_id, source_span_id, referent_id, context_id, relation, temporal_value, state_value, confidence
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    stable_id(
                                        "tmp",
                                        run_id,
                                        condition_id,
                                        temporal_id,
                                        temporal_text,
                                        edge_index,
                                        state_value,
                                        edge_referent_id or "",
                                    ),
                                    run_id,
                                    source_span_id,
                                    edge_referent_id,
                                    context_id,
                                    predicate,
                                    temporal_text,
                                    state_value,
                                    condition_confidence,
                                ),
                            )

            inserted_identity = 0
            for index, item in enumerate(identities):
                left_external = text_value(item, "left_referent_id")
                right_external = text_value(item, "right_referent_id")
                left_ref = external_to_referent[left_external]
                right_ref = external_to_referent[right_external]
                relation = text_value(item, "status") or text_value(item, "relation") or "candidate"
                evidence = text_value(item, "evidence_text")
                identity_source_span_id = evidence_source_span(evidence)
                conf = confidence(item.get("confidence"), 0.65)
                identity_box_external = text_value(item, "box_id")
                if not identity_box_external and evidence:
                    matching_condition_boxes = [
                        text_value(condition, "box_id")
                        for condition in conditions
                        if (
                            text_value(condition, "evidence_text") == evidence
                            and text_value(condition, "box_id")
                            and condition_targets_identity_sides(condition, left_external, right_external)
                        )
                    ]
                    if len(set(matching_condition_boxes)) == 1:
                        identity_box_external = matching_condition_boxes[0]
                if not identity_box_external and len(root_asserted_box_ids) == 1 and not has_contextual_boxes:
                    identity_box_external = root_asserted_box_ids[0]
                identity_context_id = external_to_context.get(identity_box_external)
                identity_drs_box_id = external_to_box.get(identity_box_external)
                drs_hypothesis_id = stable_id("drsidh", run_id, identity_source_span_id, index, left_external, right_external, relation, evidence)
                can_materialize_identity = (
                    identity_context_id or left_ref == right_ref
                ) and identity_relation_allows_expansion(relation)
                identity_metadata: dict[str, Any] = {
                    "model_identity_hypothesis": item,
                    "resolved_box_external_id": identity_box_external or None,
                }
                if identity_relation_allows_expansion(relation) and not can_materialize_identity:
                    identity_metadata["expansion_blocked_reason"] = "missing_grounded_box"
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO drs_identity_hypotheses(
                      drs_hypothesis_id, run_id, source_span_id, context_id, box_id, box_external_id,
                      left_external_referent_id, right_external_referent_id,
                      left_referent_id, right_referent_id, relation, evidence_surface, confidence, source, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        drs_hypothesis_id,
                        run_id,
                        identity_source_span_id,
                        identity_context_id,
                        identity_drs_box_id,
                        identity_box_external or None,
                        left_external,
                        right_external,
                        left_ref,
                        right_ref,
                        relation,
                        evidence,
                        conf,
                        source,
                        json.dumps(identity_metadata, sort_keys=True),
                    ),
                )
                if can_materialize_identity:
                    self.connection.execute(
                        """
                        INSERT OR IGNORE INTO identity_hypotheses(
                          hypothesis_id, run_id, source_span_id, context_id, drs_box_id, box_external_id,
                          left_referent_id, right_referent_id, relation, evidence, confidence, source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            stable_id("idh", run_id, identity_source_span_id, "drs", left_external, right_external, relation, evidence),
                            run_id,
                            identity_source_span_id,
                            identity_context_id,
                            identity_drs_box_id,
                            identity_box_external or None,
                            left_ref,
                            right_ref,
                            relation,
                            evidence,
                            conf,
                            source,
                        ),
                    )
                inserted_identity += 1

        except BaseException:
            self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            self.connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            raise
        else:
            self.connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            self.connection.commit()
        return {
            "accepted": True,
            "reason": "materialized",
            "replaced": replaced,
            "inserted": {
                "drs_referents": len(referents),
                "drs_boxes": len(boxes),
                "drs_conditions": len(conditions),
                "drs_condition_arguments": inserted_arguments,
                "drs_identity_hypotheses": inserted_identity,
            },
        }
