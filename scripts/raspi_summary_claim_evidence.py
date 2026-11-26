#!/usr/bin/env python3
"""Read-only, source-revision-verified screening of one native summary root.

The output is private review candidates with exact original-extracted-text
spans, not a verdict about scientific truth or accuracy of model summaries.
No model calls, database modifications, document acquisition or worker control.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from file_system_catalog.recent_source_tree_audit import (
    audit_current_source_tree, MAX_SOURCE_BYTES, MAX_DERIVED_BYTES, PROMPT,
)
from file_system_catalog.source_review_audit import safe_file
from file_system_catalog.summary_claim_evidence import scan_summary_claims


ROOT = Path("/data/var/Knowledge Library")
DERIVED = Path("/data/var/knowmoredirt/knowledge-library/semantic/text")
SOURCE_MAP = Path("/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3")
FILEDB = Path("/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3")
LEDGER = ROOT / "_management/status.json"
HEX = re.compile("[0-9a-f]{64}\\Z")


def read_approved_revision(path: Path, original_path: str) -> str:
    """One unique completed approved receipt; no trust from the file name."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("unavailable or oversized canonical approval ledger")
    ledger = json.loads(path.read_bytes())
    if not isinstance(ledger, dict) or not isinstance(ledger.get("items"), list):
        raise ValueError("unexpected canonical approval ledger")
    rows = [x for x in ledger["items"] if isinstance(x, dict)
            and x.get("path") == original_path]
    if len(rows) != 1 or rows[0].get("status") != "complete":
        raise ValueError("original has no unique approved completed acquisition receipt")
    digest = rows[0].get("sha256")
    if not isinstance(digest, str) or not HEX.fullmatch(digest):
        raise ValueError("approved original SHA missing")
    return digest


def reviewed_native_root(
    *, original_path: str, source_map: sqlite3.Connection,
    native_db: sqlite3.Connection, original_root: Path,
    derived_root: Path, completed_receipts: dict[str, str],
    context_chars: int = 2000, all_nodes: bool = False,
    maximum_reviewed_nodes: int = 64,
    focus_subject_label: str | None = None,
) -> dict:
    """Check current original, native tree and exact root BEFORE and AFTER scan."""
    if (type(all_nodes) is not bool or type(maximum_reviewed_nodes) is not int
            or not 1 <= maximum_reviewed_nodes <= 128):
        raise ValueError("invalid bounded native summary-node review")
    structural = audit_current_source_tree(
        original_path=original_path, source_map=source_map, native_db=native_db,
        original_root=original_root, derivative_root=derived_root,
        completed_receipts=completed_receipts, max_original_bytes=MAX_SOURCE_BYTES,
        max_derived_bytes=MAX_DERIVED_BYTES,
    )
    rows = source_map.execute(
        "SELECT source_sha256,derived_sha256,derived_path,status "
        "FROM originals WHERE source_path=?", (original_path,),
    ).fetchall()
    if len(rows) != 1 or rows[0][3] != "complete":
        raise ValueError("unique current derivative is unavailable")
    original_sha, derived_sha, derived_relative, _ = rows[0]
    if (original_sha != structural["publisher_original_sha256_reverified"]
            or derived_sha != structural["derived_original_sha256_reverified"]):
        raise ValueError("source revision moved after structural verification")
    derivative = safe_file(derived_root, derived_relative)
    blob = derivative.read_bytes()
    if len(blob) > MAX_DERIVED_BYTES or hashlib.sha256(blob).hexdigest() != derived_sha:
        raise ValueError("extracted text changed during read-only claim review")
    source = blob.decode("utf-8", "strict")
    if len(source) != structural["derived_chars_checked"]:
        raise ValueError("extracted text no longer matches verified source-tree length")
    roots = native_db.execute("""
        SELECT c.start_char, c.end_char, c.content_sha256,
               r.representation_text, r.representation_text_sha256, r.generation_json
        FROM filesystem_entries f
        JOIN content_chunks c ON c.filesystem_entry_id=f.id
        JOIN content_representations r ON r.chunk_id=c.chunk_id
        WHERE f.relative_path_display=? AND f.entry_type='file'
          AND c.collection_id LIKE '%:summary'
          AND r.prompt_version=? AND r.analysis_status='complete'
          AND r.representation_kind='summary_long'
          AND c.start_char=0 AND c.end_char=?
    """, (original_path, PROMPT, len(source))).fetchall()
    selected = []
    for start, end, content_sha, summary, summary_sha, provenance_raw in roots:
        props = json.loads(provenance_raw)
        if (not isinstance(props, dict) or props.get("original_sha256") != original_sha
                or props.get("derived_sha256") != derived_sha):
            continue  # Historical revision rows cannot be current.
        if (start != 0 or end != len(source) or content_sha != original_sha
                or not isinstance(summary, str)
                or hashlib.sha256(summary.encode("utf-8")).hexdigest() != summary_sha
                or props.get("summary_sha256") != summary_sha):
            raise ValueError("current native summary root changed or is malformed")
        selected.append((summary, summary_sha, props))
    if len(selected) != 1:
        raise ValueError("unique current native summary root not found")
    summary, summary_sha, props = selected[0]
    evidence = scan_summary_claims(
        summary=summary, source=source, context_chars=context_chars,
        focus_subject_label=focus_subject_label,
    )
    if evidence["summary_sha256"] != summary_sha:
        raise ValueError("claim review does not belong to current native root")
    node_reviews = None
    if all_nodes:
        native_rows = native_db.execute("""
            SELECT c.start_char, c.end_char, c.text_sha256, c.content_sha256,
                   r.representation_kind, r.representation_text,
                   r.representation_text_sha256, r.generation_json
            FROM filesystem_entries f
            JOIN content_chunks c ON c.filesystem_entry_id=f.id
            JOIN content_representations r ON r.chunk_id=c.chunk_id
            WHERE f.relative_path_display=? AND f.entry_type='file'
              AND c.collection_id LIKE '%:summary'
              AND r.prompt_version=? AND r.analysis_status='complete'
              AND r.representation_kind IN ('summary','summary_long')
        """, (original_path, PROMPT)).fetchall()
        selected_nodes = []
        for start, end, span_sha, content_sha, kind, body, body_sha, gen_raw in native_rows:
            gen = json.loads(gen_raw)
            if (not isinstance(gen, dict)
                    or gen.get("original_sha256") != original_sha
                    or gen.get("derived_sha256") != derived_sha):
                continue
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(source)
                    or content_sha != original_sha
                    or hashlib.sha256(source[start:end].encode()).hexdigest() != span_sha
                    or gen.get("source_passage_sha256") != span_sha
                    or gen.get("source_start_char") != start
                    or gen.get("source_end_char") != end
                    or not isinstance(body, str)
                    or hashlib.sha256(body.encode()).hexdigest() != body_sha
                    or gen.get("summary_sha256") != body_sha
                    or type(gen.get("summary_level")) is not int
                    or not isinstance(gen.get("node_id"), str)):
                raise ValueError("current summary child differs from source-linked native graph")
            selected_nodes.append((gen["summary_level"], start, end, kind,
                                   gen["node_id"], span_sha, body, body_sha))
        if (len(selected_nodes) != structural["current_native_summary_nodes_fully_verified"]
                or len(selected_nodes) > maximum_reviewed_nodes):
            raise ValueError("native summary-node count changed or exceeds bounded review")
        node_reviews = []
        for level, start, end, kind, node_id, span_sha, body, body_sha in sorted(selected_nodes):
            review = scan_summary_claims(
                summary=body, source=source, context_chars=context_chars,
                source_start_char=start, source_end_char=end,
                focus_subject_label=focus_subject_label,
            )
            if review["summary_sha256"] != body_sha:
                raise ValueError("native node summary changed during claim review")
            node_reviews.append({
                "node_id": node_id, "level": level, "kind": kind,
                "original_source_start_char": start,
                "original_source_end_char": end,
                "original_source_span_sha256": span_sha,
                "native_summary_sha256": body_sha,
                "claim_evidence": review,
            })
    final = audit_current_source_tree(
        original_path=original_path, source_map=source_map, native_db=native_db,
        original_root=original_root, derivative_root=derived_root,
        completed_receipts=completed_receipts, max_original_bytes=MAX_SOURCE_BYTES,
        max_derived_bytes=MAX_DERIVED_BYTES,
    )
    if final != structural:
        raise ValueError("original/native tree changed during claim review")
    if hashlib.sha256(derivative.read_bytes()).hexdigest() != derived_sha:
        raise ValueError("extracted source changed after final structural check")
    return {
        "schema": "kmd.current_original_root_claim_evidence_review.v1",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_path": original_path,
        "original_sha256": original_sha, "derived_sha256": derived_sha,
        "current_native_root_id": props["node_id"],
        "current_native_root_summary_sha256": summary_sha,
        "root_text_retrieval_hypothesis_not_authoritative": summary,
        "verified_complete_native_source_tree": structural,
        "candidate_review": evidence,
        "all_native_node_claim_reviews": node_reviews,
        "all_native_node_review_candidate_count": (
            sum(x["claim_evidence"]["review_candidates"] for x in node_reviews)
            if node_reviews is not None else None
        ),
        "original_and_native_tree_checked_again_after_review": True,
        "source_visual_figure_or_table_reviewed": False,
        **({"reviewer_supplied_focus_subject":focus_subject_label,
            "candidate_source_table_row_labels_not_semantically_verified":True}
           if focus_subject_label is not None else {}),
        "cannot_automatically_establish_factual_truth": True,
        "side_effects": {
            "new_downloads": 0, "model_calls": 0,
            "original_or_derivative_writes": 0, "native_db_writes": 0,
            "native_summary_approvals_or_replacements": 0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original-path", required=True)
    ap.add_argument("--output", required=True, type=Path,
                    help="New private .json output path; never overwrites an existing artifact")
    ap.add_argument("--original-root", type=Path, default=ROOT)
    ap.add_argument("--derived-root", type=Path, default=DERIVED)
    ap.add_argument("--source-map", type=Path, default=SOURCE_MAP)
    ap.add_argument("--native-db", type=Path, default=FILEDB)
    ap.add_argument("--completed-ledger", type=Path, default=LEDGER)
    ap.add_argument("--context-chars", type=int, default=2000)
    ap.add_argument("--all-nodes", action="store_true",
                    help="Review every current native child and root within its exact original span")
    ap.add_argument("--maximum-reviewed-nodes", type=int, default=64)
    ap.add_argument("--focus-subject-label",
                    help="Explicit source subject/entity label, e.g. AgentMemRec. Candidate same-line percentage row review ONLY; never an automatic false-attribution verdict")
    args = ap.parse_args()
    # Refuse absolute/traversal/symlink escapes before touching the index.
    safe_file(args.original_root, args.original_path)
    output = args.output
    if (output.suffix != ".json" or output.exists() or output.is_symlink()
            or not output.parent.is_dir() or output.parent.is_symlink()):
        raise ValueError("private claim-review destination must be a fresh .json file")
    for database in (args.source_map, args.native_db):
        if not database.is_file() or database.is_symlink():
            raise ValueError("missing canonical source-map/native FileDB")
    receipt = read_approved_revision(args.completed_ledger, args.original_path)
    with sqlite3.connect(args.source_map.resolve().as_uri() + "?mode=ro", uri=True) as sm, \
         sqlite3.connect(args.native_db.resolve().as_uri() + "?mode=ro", uri=True) as db:
        sm.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA query_only=ON")
        result = reviewed_native_root(
            original_path=args.original_path, source_map=sm, native_db=db,
            original_root=args.original_root, derived_root=args.derived_root,
            completed_receipts={args.original_path: receipt},
            context_chars=args.context_chars,
            all_nodes=args.all_nodes, maximum_reviewed_nodes=args.maximum_reviewed_nodes,
            focus_subject_label=args.focus_subject_label,
        )
    # The only write is a fresh private audit artifact, never the corpus.
    old_umask = os.umask(0o077)
    try:
        with output.open("x", encoding="utf-8") as stream:
            os.chmod(output, 0o600)
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    finally:
        os.umask(old_umask)
    print(json.dumps({
        "status": "read_only_original_review_saved",
        "path": str(output),
        "review_candidates_not_error_verdicts":
            result["candidate_review"]["review_candidates"],
        "source_and_native_revision_reverified": True,
        "all_node_review_candidates": result["all_native_node_review_candidate_count"],
        "native_summary_modified": False,
    }))


if __name__ == "__main__":
    main()
