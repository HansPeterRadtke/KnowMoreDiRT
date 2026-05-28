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


SCHEMA_VERSION = 6


def stable_id(prefix: str, *parts: Any) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


class DSPGStore:
    """Small SQLite persistence layer for internal DSPG records."""

    def __init__(self, path: str | Path = ":memory:", *, create_indexes: bool = True) -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        if self.path == ":memory:":
            self.connection.execute("PRAGMA journal_mode=MEMORY")
        self.initialize_schema(create_indexes=create_indexes)

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
              confidence REAL NOT NULL
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
        ]
        for statement in statements:
            self.connection.execute(statement)
        if create_indexes:
            self.create_indexes()
        self.connection.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()

    def create_indexes(self) -> None:
        statements = [
            "CREATE INDEX IF NOT EXISTS idx_documents_run ON documents(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_documents_rel ON documents(rel_path)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_order ON chunks(document_id, chunk_order)",
            "CREATE INDEX IF NOT EXISTS idx_spans_surface ON source_spans(surface_norm)",
            "CREATE INDEX IF NOT EXISTS idx_mentions_surface ON mentions(surface_norm)",
            "CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_referents_label ON referents(canonical_label_norm)",
            "CREATE INDEX IF NOT EXISTS idx_identity_left ON identity_hypotheses(left_referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_identity_right ON identity_hypotheses(right_referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_kind ON contexts(kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_carriers_kind ON context_carriers(carrier_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_carriers_doc ON context_carriers(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_carriers_time ON context_carriers(temporal_value_type, temporal_value)",
            "CREATE INDEX IF NOT EXISTS idx_context_assignments_context ON context_assignments(context_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_assignments_applies ON context_assignments(applies_to_type, applies_to_id)",
            "CREATE INDEX IF NOT EXISTS idx_frames_predicate ON frames(predicate_norm)",
            "CREATE INDEX IF NOT EXISTS idx_frame_args_role ON frame_arguments(role)",
            "CREATE INDEX IF NOT EXISTS idx_drs_boxes_context ON drs_boxes(context_id)",
            "CREATE INDEX IF NOT EXISTS idx_drs_referents_surface ON drs_referents(surface_norm)",
            "CREATE INDEX IF NOT EXISTS idx_drs_conditions_predicate ON drs_conditions(predicate_norm)",
            "CREATE INDEX IF NOT EXISTS idx_drs_conditions_box ON drs_conditions(box_id)",
            "CREATE INDEX IF NOT EXISTS idx_drs_args_role ON drs_condition_arguments(role)",
            "CREATE INDEX IF NOT EXISTS idx_drs_args_ref ON drs_condition_arguments(referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_drs_identity_left ON drs_identity_hypotheses(left_referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_drs_identity_right ON drs_identity_hypotheses(right_referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_temporal_ref ON temporal_edges(referent_id)",
            "CREATE INDEX IF NOT EXISTS idx_temporal_relation ON temporal_edges(relation)",
            "CREATE INDEX IF NOT EXISTS idx_temporal_value ON temporal_edges(temporal_value)",
            "CREATE INDEX IF NOT EXISTS idx_relations_predicate ON relations(predicate_norm)",
            "CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_norm)",
            "CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_norm)",
            "CREATE INDEX IF NOT EXISTS idx_relations_value ON relations(value_norm)",
            "CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_records_key ON metadata_records(key)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_records_value ON metadata_records(value_norm)",
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
            if not box_id:
                errors.append("box_missing_id")
            if parent_id and parent_id not in box_ids:
                errors.append(f"missing_parent_box:{box_id}->{parent_id}")
            if holder_id and holder_id not in referent_ids:
                errors.append(f"missing_holder_referent:{box_id}->{holder_id}")
            check_grounding(item.get("evidence_text"), f"box:{box_id}")
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
                elif target_kind == "condition" and target_id and target_id not in condition_ids:
                    errors.append(f"missing_argument_condition:{condition_id}->{target_id}")
                elif target_kind not in {"referent", "box", "condition", "literal", "unknown"}:
                    errors.append(f"bad_argument_target_kind:{condition_id}:{target_kind}")
                check_grounding(arg.get("evidence_text"), f"argument:{condition_id}:{text_value(arg, 'role')}")
        for item in identities:
            left_id = text_value(item, "left_referent_id")
            right_id = text_value(item, "right_referent_id")
            if left_id not in referent_ids:
                errors.append(f"missing_identity_left:{left_id}")
            if right_id not in referent_ids:
                errors.append(f"missing_identity_right:{right_id}")
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

        external_to_referent: dict[str, str] = {}
        external_to_drs_referent: dict[str, str] = {}
        for item in referents:
            external_id = text_value(item, "id")
            label = text_value(item, "label")
            value_type = text_value(item, "kind") or text_value(item, "value_type") or "unknown"
            referent_id = self.upsert_referent(run_id, label, value_type)
            drs_referent_id = stable_id("drsref", run_id, source_span_id, external_id, label)
            external_to_referent[external_id] = referent_id
            external_to_drs_referent[external_id] = drs_referent_id
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
        for item in boxes:
            external_id = text_value(item, "id")
            kind = text_value(item, "kind") or "asserted"
            parent_external = text_value(item, "parent_id")
            holder_external = text_value(item, "holder_referent_id")
            evidence = text_value(item, "evidence_text")
            context_id = stable_id("ctx", run_id, "drs_box", source_span_id, external_id, kind, evidence)
            drs_box_id = stable_id("drsbox", run_id, source_span_id, external_id, kind, evidence)
            external_to_context[external_id] = context_id
            external_to_box[external_id] = drs_box_id
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
                    holder_external or None,
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
        inserted_arguments = 0
        for item in conditions:
            external_id = text_value(item, "id")
            predicate = text_value(item, "predicate")
            box_external = text_value(item, "box_id")
            context_id = external_to_context[box_external]
            temporal_id = text_value(item, "temporal_id")
            temporal_text = text_value(temporal_values.get(temporal_id, {}), "value") if temporal_id else ""
            evidence = text_value(item, "evidence_text")
            condition_id = stable_id("drscond", run_id, source_span_id, external_id, predicate, evidence)
            frame_id = stable_id("frm", run_id, source_span_id, "drs", external_id, predicate, evidence)
            external_to_condition[external_id] = condition_id
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
                    text_value(item, "polarity") or "positive",
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
                    text_value(item, "modality") or "asserted",
                    normalize(text_value(item, "modality") or "asserted"),
                    predicate,
                    normalize(predicate),
                    text_value(item, "polarity") or "positive",
                    normalize(text_value(item, "polarity") or "positive"),
                    evidence,
                    normalize(evidence),
                    source_span_id,
                    context_id,
                    condition_confidence,
                    json.dumps({"source": source, "external_condition_id": external_id, "external_box_id": box_external}, sort_keys=True),
                ),
            )
            for arg_index, arg in enumerate(item.get("arguments") or []):
                if not isinstance(arg, dict):
                    continue
                role = text_value(arg, "role") or "argument"
                target_kind = text_value(arg, "target_kind") or "unknown"
                target_external = text_value(arg, "target_id")
                value = text_value(arg, "value")
                value_type = text_value(arg, "value_type") or "unknown"
                referent_id = external_to_referent.get(target_external) if target_kind == "referent" else None
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
                        value,
                        value_type,
                        condition_confidence,
                    ),
                )
                inserted_arguments += 1
            if temporal_text:
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO temporal_edges(
                      edge_id, run_id, source_span_id, referent_id, context_id, relation, temporal_value, state_value, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("tmp", run_id, condition_id, temporal_id, temporal_text),
                        run_id,
                        source_span_id,
                        None,
                        context_id,
                        predicate,
                        temporal_text,
                        evidence,
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
            conf = confidence(item.get("confidence"), 0.65)
            drs_hypothesis_id = stable_id("drsidh", run_id, source_span_id, index, left_external, right_external, relation, evidence)
            self.connection.execute(
                """
                INSERT OR IGNORE INTO drs_identity_hypotheses(
                  drs_hypothesis_id, run_id, source_span_id, left_external_referent_id, right_external_referent_id,
                  left_referent_id, right_referent_id, relation, evidence_surface, confidence, source, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    drs_hypothesis_id,
                    run_id,
                    source_span_id,
                    left_external,
                    right_external,
                    left_ref,
                    right_ref,
                    relation,
                    evidence,
                    conf,
                    source,
                    json.dumps({"model_identity_hypothesis": item}, sort_keys=True),
                ),
            )
            if relation != "rejected":
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO identity_hypotheses(
                      hypothesis_id, run_id, left_referent_id, right_referent_id, relation, evidence, confidence, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("idh", run_id, source_span_id, "drs", left_external, right_external, relation, evidence),
                        run_id,
                        left_ref,
                        right_ref,
                        relation,
                        evidence,
                        conf,
                        source,
                    ),
                )
            inserted_identity += 1

        self.connection.commit()
        return {
            "accepted": True,
            "reason": "materialized",
            "inserted": {
                "drs_referents": len(referents),
                "drs_boxes": len(boxes),
                "drs_conditions": len(conditions),
                "drs_condition_arguments": inserted_arguments,
                "drs_identity_hypotheses": inserted_identity,
            },
        }
