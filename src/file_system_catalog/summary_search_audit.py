"""Private, append-only audit for native KnowMoreDiRT semantic document searches.

Writes only to the private Raspi data tier, never a Git checkout, and records
the exact query, ranked original source IDs, similarity, and summary nodes.
A successful user-facing search must be written successfully first.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

AUDIT_DIR = Path("/data/var/knowmoredirt/knowledge-library/search-audit")
SCHEMA = "knowmoredirt-summary-search-audit-v1"
MAX_RECORD_BYTES = 1_000_000
MAX_DAILY_LOG_BYTES = 64 * 1024 * 1024


def audit_path(*, now: datetime | None = None, directory: Path = AUDIT_DIR) -> Path:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("audit timestamp must include timezone")
    return directory / f"summary-search-{current.astimezone(timezone.utc).date().isoformat()}.jsonl"


def append_query_audit(record: dict, *, directory: Path = AUDIT_DIR, now: datetime | None = None) -> Path:
    """Append one complete JSON line under an exclusive OS-level lock."""
    if not isinstance(record, dict) or not isinstance(record.get("query"), str):
        raise ValueError("query log requires an actual query")
    if len(record["query"]) > 2000:
        raise ValueError("query exceeds bounded log input")
    destination = audit_path(now=now, directory=directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.is_symlink():
        raise ValueError("query log directory may not be a symlink")
    os.chmod(directory, 0o700)
    item = {
        "schema": SCHEMA,
        "timestamp_utc": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(timespec="milliseconds"),
        **record,
    }
    raw = (json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("unbounded search audit record")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(destination, flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("search audit destination must be a regular file")
        if os.fstat(fd).st_size + len(raw) > MAX_DAILY_LOG_BYTES:
            raise RuntimeError("daily query log reached its size cap")
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "ab", closefd=False) as output:
            output.write(raw)
            output.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    return destination


def make_search_audit(
    *,
    query: str,
    result: dict | None,
    elapsed_seconds: float,
    error: str | None = None,
    model: str,
    database: str,
) -> dict:
    """Keep useful ranked search evidence even when results were filtered."""
    hits = result.get("documents", []) if isinstance(result, dict) else []
    ranked = []
    for rank, source in enumerate(hits, 1):
        matched = []
        for node in source.get("matched_summary_nodes", []):
            matched.append({
                "summary_node_id": node.get("summary_node_id"),
                "summary_kind": node.get("summary_kind"),
                "summary_level": node.get("summary_level"),
                "summary_sha256": node.get("summary_sha256"),
                "original_path": node.get("original_path"),
                "original_sha256": node.get("original_sha256"),
                "derived_sha256": node.get("derived_sha256"),
                "source_passage_sha256": node.get("source_passage_sha256"),
                "source_start_char": node.get("source_start_char"),
                "source_end_char": node.get("source_end_char"),
                "score": round(float(node.get("similarity", 0)), 6),
                "summary_excerpt": str(node.get("summary", ""))[:1000],
            })
        ranked.append({
            "rank": rank,
            "title": source.get("title"),
            "original_path": source.get("original_path"),
            "score": source.get("similarity"),
            "summary_level": source.get("best_level"),
            "summary": str(source.get("best_summary", ""))[:1000],
            "indexed_nodes": source.get("indexed_summary_nodes_for_document"),
            "matched_nodes": matched,
        })
    policy = None
    if result is not None and error is None:
        policy = result.get("provenance_policy", "ledger_and_current_derived_revision_checked;summary_is_not_primary_evidence")
    return {
        "query": query,
        "model": model,
        "database": database,
        "retrieval_engine": (result or {}).get("retrieval_engine", "legacy_or_failed_query"),
        "duration_seconds": round(elapsed_seconds, 3),
        "success": error is None,
        "error": error,
        "count": len(hits),
        "ranked_results": ranked,
        "index_scope": (result or {}).get("index_scope"),
        "relevance_warning": (result or {}).get("relevance_warning"),
        "provenance_policy": policy,
    }


def recent_search_audits(*, directory: Path = AUDIT_DIR, limit: int = 10) -> list[dict]:
    """Read recent exact query/rank receipts without returning whole source books."""
    if not 1 <= limit <= 100:
        raise ValueError("invalid log count")
    if not directory.is_dir() or directory.is_symlink():
        return []
    out = []
    files = sorted(directory.glob("summary-search-????-??-??.jsonl"), reverse=True)
    for path in files[:8]:
        if path.is_symlink() or not path.is_file():
            continue
        # Read at most eight MiB of each newest log for bounded read-only UI.
        with path.open("rb") as handle:
            size = path.stat().st_size
            if size > 8_000_000:
                handle.seek(size - 8_000_000)
                handle.readline()  # discard partial first record
            lines = handle.readlines()
        for raw in reversed(lines):
            try:
                entry = json.loads(raw)
            except (UnicodeError, ValueError):
                continue
            if entry.get("schema") == SCHEMA:
                out.append(entry)
                if len(out) >= limit:
                    return out
    return out
