#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2


def stable_id(prefix: str, *parts: Any) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS extraction_runs (
      run_id TEXT PRIMARY KEY,
      started_at REAL NOT NULL,
      input_root TEXT NOT NULL,
      config_path TEXT,
      variant TEXT NOT NULL,
      model_endpoint TEXT,
      model_name TEXT,
      model_metadata_json TEXT,
      status TEXT NOT NULL DEFAULT 'running',
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
      char_count INTEGER NOT NULL,
      metadata_json TEXT,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id)
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
      token_estimate INTEGER NOT NULL,
      FOREIGN KEY(document_id) REFERENCES documents(document_id)
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
      span_kind TEXT NOT NULL,
      FOREIGN KEY(document_id) REFERENCES documents(document_id),
      FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mentions (
      mention_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      span_id TEXT NOT NULL,
      surface TEXT NOT NULL,
      surface_norm TEXT NOT NULL,
      mention_kind TEXT,
      entity_type TEXT,
      confidence REAL,
      source TEXT,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id),
      FOREIGN KEY(span_id) REFERENCES source_spans(span_id)
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
      attributes_json TEXT,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mention_referents (
      mention_id TEXT NOT NULL,
      referent_id TEXT NOT NULL,
      link_status TEXT NOT NULL DEFAULT 'member',
      confidence REAL NOT NULL DEFAULT 1.0,
      PRIMARY KEY (mention_id, referent_id),
      FOREIGN KEY(mention_id) REFERENCES mentions(mention_id),
      FOREIGN KEY(referent_id) REFERENCES referents(referent_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS contexts (
      context_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      parent_context_id TEXT,
      holder_mention_id TEXT,
      evidence_surface TEXT,
      confidence REAL,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS context_carriers (
      context_carrier_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      context_id TEXT,
      context_kind TEXT NOT NULL,
      context_subkind TEXT,
      source_document_id TEXT,
      source_chunk_id TEXT,
      source_span_id TEXT,
      carrier_surface TEXT NOT NULL,
      carrier_type TEXT NOT NULL,
      applies_to_range_start INTEGER,
      applies_to_range_end INTEGER,
      inheritance_parent_context_id TEXT,
      confidence REAL,
      status TEXT NOT NULL,
      created_by_extraction_run TEXT,
      evidence_source_span_id TEXT,
      notes TEXT,
      temporal_value TEXT,
      temporal_value_type TEXT,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id),
      FOREIGN KEY(context_id) REFERENCES contexts(context_id),
      FOREIGN KEY(source_document_id) REFERENCES documents(document_id),
      FOREIGN KEY(source_chunk_id) REFERENCES chunks(chunk_id),
      FOREIGN KEY(source_span_id) REFERENCES source_spans(span_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS context_assignments (
      context_assignment_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      context_carrier_id TEXT NOT NULL,
      context_id TEXT,
      applies_to_type TEXT NOT NULL,
      applies_to_id TEXT NOT NULL,
      applies_to_range_start INTEGER,
      applies_to_range_end INTEGER,
      confidence REAL,
      status TEXT NOT NULL,
      evidence_source_span_id TEXT,
      notes TEXT,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id),
      FOREIGN KEY(context_carrier_id) REFERENCES context_carriers(context_carrier_id),
      FOREIGN KEY(context_id) REFERENCES contexts(context_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS frames (
      frame_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      context_id TEXT,
      predicate TEXT NOT NULL,
      predicate_norm TEXT NOT NULL,
      trigger_surface TEXT,
      confidence REAL,
      source TEXT,
      span_id TEXT,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id),
      FOREIGN KEY(context_id) REFERENCES contexts(context_id),
      FOREIGN KEY(span_id) REFERENCES source_spans(span_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS frame_arguments (
      argument_id TEXT PRIMARY KEY,
      frame_id TEXT NOT NULL,
      role TEXT NOT NULL,
      mention_id TEXT,
      referent_id TEXT,
      confidence REAL,
      FOREIGN KEY(frame_id) REFERENCES frames(frame_id),
      FOREIGN KEY(mention_id) REFERENCES mentions(mention_id),
      FOREIGN KEY(referent_id) REFERENCES referents(referent_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS propositions (
      proposition_id TEXT PRIMARY KEY,
      frame_id TEXT,
      context_id TEXT,
      predicate_norm TEXT,
      source_span_id TEXT,
      confidence REAL,
      metadata_json TEXT,
      FOREIGN KEY(frame_id) REFERENCES frames(frame_id),
      FOREIGN KEY(context_id) REFERENCES contexts(context_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS relations (
      relation_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      left_id TEXT NOT NULL,
      right_id TEXT NOT NULL,
      relation_type TEXT NOT NULL,
      confidence REAL,
      source_span_id TEXT,
      metadata_json TEXT,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS identity_hypotheses (
      identity_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      left_referent_id TEXT,
      right_referent_id TEXT,
      mention_id TEXT,
      referent_id TEXT,
      status TEXT NOT NULL,
      confidence REAL,
      evidence_json TEXT,
      source TEXT,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS temporal_edges (
      temporal_edge_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      source_frame_id TEXT,
      target_frame_id TEXT,
      relation_type TEXT NOT NULL,
      time_value TEXT,
      confidence REAL,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS contradictions_or_support (
      item_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      left_frame_id TEXT,
      right_frame_id TEXT,
      relation_type TEXT NOT NULL,
      confidence REAL,
      evidence_json TEXT,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_calls (
      call_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      stage TEXT NOT NULL,
      unit_id TEXT,
      endpoint TEXT,
      prompt_hash TEXT,
      grammar_hash TEXT,
      input_hash TEXT,
      output_hash TEXT,
      n_predict INTEGER,
      timeout_seconds REAL,
      elapsed_seconds REAL,
      stop_reason TEXT,
      fresh_or_cached TEXT,
      accepted INTEGER NOT NULL,
      schema_valid INTEGER NOT NULL,
      raw_output TEXT,
      parsed_json TEXT,
      metadata_json TEXT,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rejected_model_outputs (
      rejection_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      call_id TEXT,
      stage TEXT,
      reason TEXT NOT NULL,
      raw_output TEXT,
      parsed_json TEXT,
      metadata_json TEXT,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS query_runs (
      query_run_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      question TEXT NOT NULL,
      question_hash TEXT NOT NULL,
      query_plan_json TEXT,
      used_model INTEGER NOT NULL,
      status TEXT NOT NULL,
      created_at REAL NOT NULL,
      FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS answers (
      answer_id TEXT PRIMARY KEY,
      query_run_id TEXT NOT NULL,
      answer_text TEXT NOT NULL,
      answer_norm TEXT NOT NULL,
      answer_type TEXT,
      confidence REAL,
      status TEXT NOT NULL,
      evidence_json TEXT,
      FOREIGN KEY(query_run_id) REFERENCES query_runs(query_run_id)
    )
    """,
]


INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_documents_path_hash ON documents(path, content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc_order ON chunks(document_id, chunk_order)",
    "CREATE INDEX IF NOT EXISTS idx_source_spans_document ON source_spans(document_id, chunk_id)",
    "CREATE INDEX IF NOT EXISTS idx_mentions_surface_norm ON mentions(surface_norm)",
    "CREATE INDEX IF NOT EXISTS idx_mentions_kind ON mentions(mention_kind)",
    "CREATE INDEX IF NOT EXISTS idx_mentions_span ON mentions(span_id)",
    "CREATE INDEX IF NOT EXISTS idx_mentions_run_span ON mentions(run_id, span_id)",
    "CREATE INDEX IF NOT EXISTS idx_referents_label ON referents(canonical_label_norm)",
    "CREATE INDEX IF NOT EXISTS idx_referents_type ON referents(entity_type)",
    "CREATE INDEX IF NOT EXISTS idx_frames_predicate ON frames(predicate_norm)",
    "CREATE INDEX IF NOT EXISTS idx_frames_context ON frames(context_id)",
    "CREATE INDEX IF NOT EXISTS idx_frames_span ON frames(span_id)",
    "CREATE INDEX IF NOT EXISTS idx_arguments_frame ON frame_arguments(frame_id)",
    "CREATE INDEX IF NOT EXISTS idx_arguments_role ON frame_arguments(role)",
    "CREATE INDEX IF NOT EXISTS idx_arguments_referent ON frame_arguments(referent_id)",
    "CREATE INDEX IF NOT EXISTS idx_identity_left_right_status ON identity_hypotheses(left_referent_id, right_referent_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_identity_right_left_status ON identity_hypotheses(right_referent_id, left_referent_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_spans_artifacts ON source_spans(surface_norm, span_kind)",
    "CREATE INDEX IF NOT EXISTS idx_context_kind ON contexts(kind)",
    "CREATE INDEX IF NOT EXISTS idx_context_carrier_kind ON context_carriers(context_kind, context_subkind)",
    "CREATE INDEX IF NOT EXISTS idx_context_carrier_temporal ON context_carriers(temporal_value_type, temporal_value)",
    "CREATE INDEX IF NOT EXISTS idx_context_carrier_source ON context_carriers(source_document_id, source_chunk_id, source_span_id)",
    "CREATE INDEX IF NOT EXISTS idx_context_assignment_target ON context_assignments(applies_to_type, applies_to_id)",
    "CREATE INDEX IF NOT EXISTS idx_context_assignment_carrier ON context_assignments(context_carrier_id)",
    "CREATE INDEX IF NOT EXISTS idx_temporal_relation_time ON temporal_edges(relation_type, time_value)",
    "CREATE INDEX IF NOT EXISTS idx_answers_norm ON answers(answer_norm)",
    "CREATE INDEX IF NOT EXISTS idx_query_hash ON query_runs(question_hash)",
]


REQUIRED_TABLES = {
    "documents",
    "chunks",
    "source_spans",
    "mentions",
    "referents",
    "contexts",
    "context_carriers",
    "context_assignments",
    "frames",
    "frame_arguments",
    "propositions",
    "relations",
    "identity_hypotheses",
    "temporal_edges",
    "contradictions_or_support",
    "extraction_runs",
    "model_calls",
    "rejected_model_outputs",
    "query_runs",
    "answers",
}


def init_db(con: sqlite3.Connection, create_indexes: bool = True) -> None:
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    for stmt in SCHEMA:
        con.execute(stmt)
    if create_indexes:
        for stmt in INDEXES:
            con.execute(stmt)
    con.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    con.commit()


def validate_schema(con: sqlite3.Connection) -> dict[str, Any]:
    tables = {
        row["name"]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    indexes = [
        row["name"]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]
    missing = sorted(REQUIRED_TABLES - tables)
    version = con.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()
    return {
        "valid": not missing and version is not None and version["value"] == str(SCHEMA_VERSION),
        "schema_version": version["value"] if version else None,
        "missing_tables": missing,
        "table_count": len(tables),
        "index_count": len(indexes),
        "indexes": indexes,
    }


def insert_run(
    con: sqlite3.Connection,
    input_root: str,
    config_path: str,
    variant: str,
    model_endpoint: str = "",
    model_name: str = "",
    model_metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> str:
    rid = run_id or stable_id("run", input_root, config_path, variant, time.time())
    con.execute(
        """
        INSERT OR REPLACE INTO extraction_runs
        (run_id, started_at, input_root, config_path, variant, model_endpoint, model_name, model_metadata_json, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rid,
            time.time(),
            input_root,
            config_path,
            variant,
            model_endpoint,
            model_name,
            json.dumps(model_metadata or {}, ensure_ascii=False),
            "running",
        ),
    )
    con.commit()
    return rid


def finish_run(con: sqlite3.Connection, run_id: str, status: str, metrics: dict[str, Any]) -> None:
    con.execute(
        "UPDATE extraction_runs SET status=?, metrics_json=? WHERE run_id=?",
        (status, json.dumps(metrics, ensure_ascii=False), run_id),
    )
    con.commit()


def insert_document(con: sqlite3.Connection, run_id: str, root: Path, path: Path, text: str) -> str:
    rel = str(path.relative_to(root))
    did = stable_id("doc", run_id, rel, text_hash(text))
    con.execute(
        """
        INSERT OR REPLACE INTO documents(document_id, run_id, path, rel_path, content_hash, char_count, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (did, run_id, str(path), rel, text_hash(text), len(text), "{}"),
    )
    return did


def insert_chunk(
    con: sqlite3.Connection,
    document_id: str,
    chunk_order: int,
    char_start: int,
    char_end: int,
    text: str,
) -> str:
    cid = stable_id("chunk", document_id, chunk_order, char_start, char_end, text_hash(text))
    con.execute(
        """
        INSERT OR REPLACE INTO chunks(chunk_id, document_id, chunk_order, char_start, char_end, text, token_estimate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (cid, document_id, chunk_order, char_start, char_end, text, max(1, int(len(text) / 4))),
    )
    return cid


def insert_span(
    con: sqlite3.Connection,
    document_id: str,
    chunk_id: str,
    char_start: int,
    char_end: int,
    surface: str,
    span_kind: str,
) -> str:
    sid = stable_id("span", document_id, chunk_id, char_start, char_end, surface)
    con.execute(
        """
        INSERT OR REPLACE INTO source_spans(span_id, document_id, chunk_id, char_start, char_end, surface, surface_norm, span_kind)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (sid, document_id, chunk_id, char_start, char_end, surface, norm_text(surface), span_kind),
    )
    return sid


def insert_model_records(
    con: sqlite3.Connection,
    run_id: str,
    records: Iterable[dict[str, Any] | None],
) -> dict[str, int]:
    counts = {"model_calls": 0, "accepted": 0, "rejected": 0, "request_failed": 0, "truncated": 0, "schema_invalid": 0}
    for record in records:
        if not record:
            continue
        units = list(record.get("unit_results") or [])
        if not units and record.get("stage"):
            units = [record]
        for unit in units:
            counts["model_calls"] += 1
            accepted = bool(unit.get("accepted"))
            counts["accepted" if accepted else "rejected"] += 1
            reason = unit.get("reason") or ("ok" if accepted else "rejected")
            if reason == "request_failed":
                counts["request_failed"] += 1
            if reason == "truncated_output":
                counts["truncated"] += 1
            if reason in {"schema_invalid", "malformed_relation_endpoint"}:
                counts["schema_invalid"] += 1
            call_id = stable_id("call", run_id, unit.get("stage"), unit.get("unit_id"), unit.get("prompt_hash"), unit.get("output_hash"), counts["model_calls"])
            con.execute(
                """
                INSERT OR REPLACE INTO model_calls
                (call_id, run_id, stage, unit_id, endpoint, prompt_hash, grammar_hash, input_hash, output_hash,
                 n_predict, timeout_seconds, elapsed_seconds, stop_reason, fresh_or_cached, accepted, schema_valid,
                 raw_output, parsed_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_id,
                    run_id,
                    unit.get("stage") or record.get("stage") or "unknown",
                    unit.get("unit_id"),
                    unit.get("model_endpoint"),
                    unit.get("prompt_hash"),
                    unit.get("grammar_hash"),
                    unit.get("input_hash"),
                    unit.get("output_hash"),
                    unit.get("n_predict"),
                    None,
                    unit.get("elapsed"),
                    unit.get("reason") or unit.get("stop"),
                    record.get("fresh_or_cached") or unit.get("fresh_or_cached") or "fresh",
                    int(accepted),
                    int(bool(unit.get("schema_validation_success"))),
                    unit.get("raw_text"),
                    json.dumps(unit.get("json"), ensure_ascii=False) if unit.get("json") is not None else None,
                    json.dumps({k: v for k, v in unit.items() if k not in {"raw_text", "json"}}, ensure_ascii=False),
                ),
            )
            if not accepted:
                con.execute(
                    """
                    INSERT OR REPLACE INTO rejected_model_outputs
                    (rejection_id, run_id, call_id, stage, reason, raw_output, parsed_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("rej", call_id, reason),
                        run_id,
                        call_id,
                        unit.get("stage") or record.get("stage") or "unknown",
                        reason,
                        unit.get("raw_text"),
                        json.dumps(unit.get("json"), ensure_ascii=False) if unit.get("json") is not None else None,
                        json.dumps({k: v for k, v in unit.items() if k not in {"raw_text", "json"}}, ensure_ascii=False),
                    ),
                )
    return counts


def store_staged_document_graph(
    con: sqlite3.Connection,
    run_id: str,
    root: Path,
    path: Path,
    text: str,
    chunks: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    referents: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    scopes: list[dict[str, Any]],
    identities: list[dict[str, Any]],
    context_carriers: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    counts = {"documents": 1, "chunks": 0, "mentions": 0, "referents": 0, "frames": 0, "context_carriers": 0, "context_assignments": 0}
    document_id = insert_document(con, run_id, root, path, text)
    local_chunk_to_db: dict[str, str] = {}
    for idx, chunk in enumerate(chunks):
        db_chunk_id = insert_chunk(con, document_id, idx, int(chunk["char_start"]), int(chunk["char_end"]), chunk["text"])
        local_chunk_to_db[chunk["chunk_id"]] = db_chunk_id
        counts["chunks"] += 1
    class_by_mid = {c["mention_id"]: c for c in classifications}
    local_mid_to_db: dict[str, str] = {}
    local_mid_to_span: dict[str, str] = {}
    for mention in mentions:
        chunk_id = local_chunk_to_db.get(mention["source_chunk_id"])
        if not chunk_id:
            continue
        cls = class_by_mid.get(mention["mention_id"], {})
        span_id = insert_span(
            con,
            document_id,
            chunk_id,
            int(mention["char_start"]),
            int(mention["char_end"]),
            mention["surface"],
            mention.get("mention_kind") or cls.get("entity_type") or "unknown",
        )
        db_mid = stable_id("ment", run_id, document_id, mention["mention_id"], mention["char_start"], mention["surface"])
        local_mid_to_db[mention["mention_id"]] = db_mid
        local_mid_to_span[mention["mention_id"]] = span_id
        con.execute(
            """
            INSERT OR REPLACE INTO mentions
            (mention_id, run_id, span_id, surface, surface_norm, mention_kind, entity_type, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                db_mid,
                run_id,
                span_id,
                mention["surface"],
                norm_text(mention["surface"]),
                mention.get("mention_kind"),
                cls.get("entity_type") or "unknown",
                float(cls.get("confidence", 0.0) or 0.0),
                cls.get("source") or mention.get("method") or "unknown",
            ),
        )
        counts["mentions"] += 1
    local_ref_to_db: dict[str, str] = {}
    for ref in referents:
        db_rid = stable_id("ref", run_id, document_id, ref["referent_id"], ref["entity_type"], norm_text(ref["label"]))
        local_ref_to_db[ref["referent_id"]] = db_rid
        con.execute(
            """
            INSERT OR REPLACE INTO referents
            (referent_id, run_id, canonical_label, canonical_label_norm, entity_type, status, attributes_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                db_rid,
                run_id,
                ref["label"],
                norm_text(ref["label"]),
                ref["entity_type"],
                ref.get("status") or "candidate",
                json.dumps(ref, ensure_ascii=False),
            ),
        )
        for local_mid in ref.get("mention_ids", []):
            db_mid = local_mid_to_db.get(local_mid)
            if db_mid:
                con.execute(
                    """
                    INSERT OR REPLACE INTO mention_referents(mention_id, referent_id, link_status, confidence)
                    VALUES (?, ?, ?, ?)
                    """,
                    (db_mid, db_rid, "member", 1.0),
                )
        counts["referents"] += 1
    context_ids: dict[str, str] = {}
    con.execute(
        """
        INSERT OR IGNORE INTO contexts(context_id, run_id, kind, parent_context_id, holder_mention_id, evidence_surface, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (stable_id("ctx", run_id, document_id, "asserted"), run_id, "asserted", None, None, None, 1.0),
    )
    context_ids["asserted"] = stable_id("ctx", run_id, document_id, "asserted")
    for base_kind in ["source_metadata", "document_genre", "measurement_time", "table_time", "validity", "reported", "allegation", "quoted", "fiction", "homework", "dreamed", "uncertain", "unknown_validity"]:
        cid = stable_id("ctx", run_id, document_id, base_kind)
        context_ids[base_kind] = cid
        con.execute(
            """
            INSERT OR IGNORE INTO contexts(context_id, run_id, kind, parent_context_id, holder_mention_id, evidence_surface, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (cid, run_id, base_kind, context_ids["asserted"], None, None, 1.0),
        )
    def scope_rank(scope: dict[str, Any]) -> int:
        kind = scope.get("context_kind") or "asserted"
        if kind == "asserted":
            return 0
        if kind in {"quoted", "reported", "believed", "negated", "conditional_antecedent", "conditional_consequent", "dreamed"}:
            return 3
        return 2
    scope_by_frame: dict[str, dict[str, Any]] = {}
    local_frame_to_db: dict[str, str] = {}
    for scope in scopes:
        existing = scope_by_frame.get(scope["frame_id"])
        if existing is None or scope_rank(scope) >= scope_rank(existing):
            scope_by_frame[scope["frame_id"]] = scope
    for frame in frames:
        scope = scope_by_frame.get(frame["frame_id"], {"context_kind": "asserted", "confidence": 1.0})
        kind = scope.get("context_kind") or "asserted"
        if kind not in context_ids:
            context_ids[kind] = stable_id("ctx", run_id, document_id, kind)
            holder = local_mid_to_db.get(scope.get("holder_mention_id"))
            con.execute(
                """
                INSERT OR REPLACE INTO contexts(context_id, run_id, kind, parent_context_id, holder_mention_id, evidence_surface, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    context_ids[kind],
                    run_id,
                    kind,
                    context_ids["asserted"],
                    holder,
                    scope.get("evidence_surface"),
                    float(scope.get("confidence", 0.0) or 0.0),
                ),
            )
        span_id = None
        trigger = frame.get("trigger_surface") or ""
        if trigger:
            local_pos = text.find(trigger, max(0, int(frame.get("char_start", 0)) - 5))
            if local_pos < 0:
                local_pos = text.find(trigger)
            if local_pos >= 0:
                source_chunk_id = None
                for chunk in chunks:
                    if int(chunk["char_start"]) <= local_pos < int(chunk["char_end"]):
                        source_chunk_id = local_chunk_to_db.get(chunk["chunk_id"])
                        break
                if source_chunk_id:
                    span_id = insert_span(con, document_id, source_chunk_id, local_pos, local_pos + len(trigger), trigger, "frame_trigger")
        db_frame_id = stable_id("frame", run_id, document_id, frame["frame_id"], frame.get("predicate"), frame.get("char_start"))
        local_frame_id = frame["frame_id"]
        local_frame_to_db[local_frame_id] = db_frame_id
        con.execute(
            """
            INSERT OR REPLACE INTO frames
            (frame_id, run_id, context_id, predicate, predicate_norm, trigger_surface, confidence, source, span_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                db_frame_id,
                run_id,
                context_ids[kind],
                frame.get("predicate") or "unknown",
                norm_text(frame.get("predicate") or "unknown"),
                trigger,
                float(frame.get("confidence", 0.0) or 0.0),
                frame.get("source") or "unknown",
                span_id,
            ),
        )
        con.execute(
            """
            INSERT OR REPLACE INTO propositions(proposition_id, frame_id, context_id, predicate_norm, source_span_id, confidence, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_id("prop", db_frame_id),
                db_frame_id,
                context_ids[kind],
                norm_text(frame.get("predicate") or "unknown"),
                span_id,
                float(frame.get("confidence", 0.0) or 0.0),
                json.dumps(frame, ensure_ascii=False),
            ),
        )
        for idx, arg in enumerate(frame.get("arguments", [])):
            db_mid = local_mid_to_db.get(arg.get("mention_id"))
            db_ref = None
            if db_mid:
                row = con.execute(
                    "SELECT referent_id FROM mention_referents WHERE mention_id=? LIMIT 1",
                    (db_mid,),
                ).fetchone()
                db_ref = row["referent_id"] if row else None
            con.execute(
                """
                INSERT OR REPLACE INTO frame_arguments(argument_id, frame_id, role, mention_id, referent_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (stable_id("arg", db_frame_id, idx, arg.get("role"), db_mid), db_frame_id, arg.get("role") or "theme", db_mid, db_ref, 0.8),
            )
        if norm_text(frame.get("predicate")) in {"close", "reopen", "state_open", "state_closed"}:
            con.execute(
                """
                INSERT OR REPLACE INTO temporal_edges(temporal_edge_id, run_id, source_frame_id, target_frame_id, relation_type, time_value, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (stable_id("temp", db_frame_id), run_id, db_frame_id, None, norm_text(frame.get("predicate")), None, frame.get("confidence")),
            )
        counts["frames"] += 1
    for carrier in context_carriers or []:
        surface = str(carrier.get("carrier_surface") or "")
        carrier_start = int(carrier.get("char_start", -1) if carrier.get("char_start", -1) is not None else -1)
        if carrier_start < 0 and surface:
            carrier_start = text.find(surface)
        carrier_end = carrier_start + len(surface) if carrier_start >= 0 else carrier_start
        source_chunk_id = None
        if carrier_start >= 0:
            for chunk in chunks:
                if int(chunk["char_start"]) <= carrier_start < int(chunk["char_end"]):
                    source_chunk_id = local_chunk_to_db.get(chunk["chunk_id"])
                    break
        source_span_id = None
        if source_chunk_id and surface:
            source_span_id = insert_span(con, document_id, source_chunk_id, carrier_start, carrier_end, surface, "context_carrier")
        context_kind = carrier.get("context_kind") or "unknown_validity"
        context_id = context_ids.get(context_kind) or stable_id("ctx", run_id, document_id, context_kind)
        if context_kind not in context_ids:
            context_ids[context_kind] = context_id
            con.execute(
                """
                INSERT OR IGNORE INTO contexts(context_id, run_id, kind, parent_context_id, holder_mention_id, evidence_surface, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (context_id, run_id, context_kind, context_ids["asserted"], None, surface, carrier.get("confidence", 0.7)),
            )
        carrier_id = stable_id("ctxcar", run_id, document_id, context_kind, surface, carrier.get("temporal_value"), carrier_start, carrier.get("carrier_type"))
        con.execute(
            """
            INSERT OR REPLACE INTO context_carriers
            (context_carrier_id, run_id, context_id, context_kind, context_subkind, source_document_id, source_chunk_id, source_span_id,
             carrier_surface, carrier_type, applies_to_range_start, applies_to_range_end, inheritance_parent_context_id,
             confidence, status, created_by_extraction_run, evidence_source_span_id, notes, temporal_value, temporal_value_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                carrier_id,
                run_id,
                context_id,
                context_kind,
                carrier.get("context_subkind"),
                document_id,
                source_chunk_id,
                source_span_id,
                surface or str(carrier.get("carrier_type") or context_kind),
                carrier.get("carrier_type") or "unknown",
                carrier.get("applies_to_range_start", 0),
                carrier.get("applies_to_range_end", len(text)),
                carrier.get("inheritance_parent_context_id"),
                float(carrier.get("confidence", 0.0) or 0.0),
                carrier.get("status") or "asserted",
                run_id,
                source_span_id,
                carrier.get("notes"),
                carrier.get("temporal_value"),
                carrier.get("temporal_value_type") or "unknown_time",
            ),
        )
        counts["context_carriers"] += 1
        targets = [("document", document_id)]
        c_start = int(carrier.get("applies_to_range_start", 0) or 0)
        c_end = int(carrier.get("applies_to_range_end", len(text)) or len(text))
        for frame in frames:
            f_start = int(frame.get("char_start", 0) or 0)
            if c_start <= f_start <= c_end and frame.get("frame_id") in local_frame_to_db:
                targets.append(("frame", local_frame_to_db[frame["frame_id"]]))
        for applies_to_type, applies_to_id in targets:
            assignment_id = stable_id("ctxassign", carrier_id, applies_to_type, applies_to_id)
            con.execute(
                """
                INSERT OR REPLACE INTO context_assignments
                (context_assignment_id, run_id, context_carrier_id, context_id, applies_to_type, applies_to_id,
                 applies_to_range_start, applies_to_range_end, confidence, status, evidence_source_span_id, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assignment_id,
                    run_id,
                    carrier_id,
                    context_id,
                    applies_to_type,
                    applies_to_id,
                    c_start,
                    c_end,
                    float(carrier.get("confidence", 0.0) or 0.0),
                    carrier.get("status") or "asserted",
                    source_span_id,
                    carrier.get("notes"),
                ),
            )
            counts["context_assignments"] += 1
    for identity in identities:
        left = local_ref_to_db.get(identity.get("left_referent_id"))
        right = local_ref_to_db.get(identity.get("right_referent_id"))
        db_mid = local_mid_to_db.get(identity.get("mention_id"))
        db_ref = local_ref_to_db.get(identity.get("referent_id"))
        status = identity.get("decision") or identity.get("status") or "uncertain"
        con.execute(
            """
            INSERT OR REPLACE INTO identity_hypotheses
            (identity_id, run_id, left_referent_id, right_referent_id, mention_id, referent_id, status, confidence, evidence_json, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_id("ident", run_id, left, right, db_mid, db_ref, status, identity.get("confidence")),
                run_id,
                left,
                right,
                db_mid,
                db_ref,
                status,
                float(identity.get("confidence", 0.0) or 0.0),
                json.dumps(identity.get("evidence_mention_ids", []), ensure_ascii=False),
                identity.get("source") or "unknown",
            ),
        )
    con.commit()
    return counts


def table_counts(con: sqlite3.Connection) -> dict[str, int]:
    out: dict[str, int] = {}
    for table in sorted(REQUIRED_TABLES):
        out[table] = int(con.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
    return out
