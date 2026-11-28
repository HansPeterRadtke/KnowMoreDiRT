#!/usr/bin/env python3
"""Private Raspi KnowMoreDiRT hierarchical book-summary operator.

build: create COMPLETE source-linked hierarchical summary tree using local LLM.
import: append ALL node summaries/vectors to the ORIGINAL KMD FileDB only after
        the legacy query reader excludes summary candidates.
search: document discovery across every stored summary level, with original
        and derivative revision checks; does not answer from summaries.
status: native FileDB summary-level coverage.

No alternate semantic database is created. All source text and generated
summaries remain private under /data/var/knowmoredirt.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import tempfile
import hashlib
from itertools import chain
import json
import math
import os
from contextlib import closing
from pathlib import Path
import re
import sqlite3
import stat
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
INFRA = Path("/data/infra/libexec")
sys.path.insert(0, str(INFRA))
from file_system_catalog.hierarchical_summary import (
    EmbeddingContextExceeded, HierarchicalSummarizer, SummaryActivity, SummaryNode, SummaryTree,
)
from file_system_catalog.hierarchical_summary_model import LocalSummaryModel
from file_system_catalog.hierarchical_summary_store import (
    append_summary_rows, prepare_node_facets, retain_fitting_native_fields, summary_rows, validate_complete_summary_tree,
)
from file_system_catalog.hierarchical_summary_search import summary_index_scope, summary_index_status
from file_system_catalog.summary_search_batched import (
    search_summary_entries_batched, has_native_summary_candidates,
)
from file_system_catalog.summary_claim_quarantine import (
    load_reviewed_summary_exclusions, exclude_reviewed_summary_nodes,
)
from file_system_catalog.source_evidence import source_evidence_candidates
from file_system_catalog.source_markdown_tables import (
    verified_original_markdown_table_rows,
    verified_original_markdown_table_cell,
    compare_claimed_literal_to_current_original_cell,
)
from file_system_catalog.source_passage_utility import review_source_documents,review_verified_linked_originals
from file_system_catalog.named_source import resolve_source_name
from file_system_catalog.publication_identifier import publication_receipt_matches
from file_system_catalog.publisher_ccs import PREFIX_CHARS, first_page_ccs
from file_system_catalog.publisher_pdf_page import verify_pdf_page_excerpts
from file_system_catalog.rfc_section import matching_rfc, rfc_header_matches, rfc_number, rfc_section_evidence, section_key
from file_system_catalog.source_passage_digests import source_passage_digests
from file_system_catalog.summary_search_audit import (
    append_query_audit, make_search_audit, recent_search_audits, AUDIT_DIR,
)
from file_system_catalog.summary_coverage import (
    current_completed_originals, pending_failures, revision_identity, summary_progress_metrics,
)
from file_system_catalog.summary_storage_policy import (
    DEFAULT_DATABASE_BUDGET_BYTES, require_storage_admission, storage_budget_snapshot,
)

DB = Path("/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3")
LIBRARY = Path("/data/var/Knowledge Library")
SOURCES = Path("/data/infra/etc/knowledge-library-sources.json")
LEDGER = LIBRARY / "_management/status.json"
CACHE = Path("/data/var/knowmoredirt/knowledge-library/semantic")
SOURCE_MAP = CACHE / "source-map.sqlite3"
OUT = Path("/data/var/knowmoredirt/knowledge-library/summaries")
COLLECTION = "raspi-original-books-semantic-content-v1"
DB_BUDGET = DEFAULT_DATABASE_BUDGET_BYTES
REVIEWED_SUMMARY_EXCLUSIONS = DB.parent / "summary-claim-exclusions.json"
# A live three-gigabyte native FileDB had a legitimate concurrent summary import
# hold this shared writer lock beyond ninety seconds. Retain a bounded wait
# rather than falsely marking another already-generated source as failed.
# The lock helper explicitly refuses anything above five minutes.
NATIVE_WRITER_WAIT_SECONDS = 300.0
# Two background workers can overlap a read-only source-anchor lookup with
# another worker's native FileDB commit in DELETE journal mode. The SQLite
# default five-second reader wait is too short for that legitimate workload.
# This stays read-only and bounded, separate from the exclusive writer lock.
NATIVE_READER_WAIT_SECONDS = 120.0
# This is the independent extraction/source-revision metadata database.
# It has its own legitimate writer; never mistake a transient SQLite
# exclusive lock for an absent approved original or stale extraction.
SOURCE_MAP_READER_WAIT_SECONDS = 20.0
PUBLISHER_TEXT_LIMIT = 8 * 1024 * 1024
GUARD_TEXT = "exclude_unresolvable_summary_candidates_from_native_excerpt_search_v1"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def regular_file_blocks(path: Path):
    """Yield stable regular-file bytes, checking for replacement when the reader reaches EOF."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(path, flags), "rb") as reader:
        before = os.fstat(reader.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("expected a regular file for source verification")
        yield from iter(lambda: reader.read(1024 * 1024), b"")
        after = path.stat(follow_symlinks=False)
        if (not os.path.samestat(before, after) or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns or before.st_ctime_ns != after.st_ctime_ns):
            raise ValueError("source file changed during verification")


def file_sha256(path: Path, *, capture: bytearray | None = None, max_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    for block in regular_file_blocks(path):
        if capture is not None:
            if max_bytes is None or len(capture) + len(block) > max_bytes:
                raise ValueError("publisher file exceeds the expected capture size")
            capture.extend(block)
        digest.update(block)
    return digest.hexdigest()


def publisher_path(relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith(("/", "./"))
        or "\\" in relative
        or "\x00" in relative
        or any(x in ("", ".", "..") for x in relative.split("/"))
        or len(relative) > 1500
    ):
        raise ValueError("invalid exact original path")
    path = LIBRARY / relative
    if (not path.is_file() or path.is_symlink() or
            not path.resolve().is_relative_to(LIBRARY.resolve())):
        raise ValueError("original file absent, symlink or outside library")
    return path


def verify_publisher_revision(path: Path, record: dict, *, capture: bytearray | None = None) -> int:
    size = path.stat().st_size
    if size != record.get("bytes_received"):
        raise ValueError("original file byte count differs from completed receipt")
    if file_sha256(path, capture=capture, max_bytes=size) != record.get("sha256"):
        raise ValueError("original file SHA differs from completed receipt")
    return size


def open_source_map_read_connection(
    database: Path, *, wait_seconds: float = SOURCE_MAP_READER_WAIT_SECONDS,
) -> sqlite3.Connection:
    """Read current extraction/source-map metadata through a bounded lock.

    A completed source-map writer may briefly hold SQLite's exclusive lock.
    Use the canonical source-map file without creating a second database,
    changing the journal mode, acquiring a write lock or accepting stale
    source revisions. The caller MUST close the connection after its SELECTs.
    """
    if (not isinstance(wait_seconds,(int,float)) or isinstance(wait_seconds,bool)
            or not 1 <= wait_seconds <= 120 or not math.isfinite(wait_seconds)):
        raise ValueError('source-map read-only wait must be bounded')
    database=Path(database)
    if database.is_symlink() or not database.is_file():
        raise ValueError('canonical source-map database missing or symlinked')
    conn=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',
                         uri=True,timeout=float(wait_seconds))
    try:
        conn.execute('PRAGMA query_only=ON')
    except Exception:
        conn.close()
        raise
    return conn


def checked_derivative_path(derivative_relative: str) -> Path:
    if not isinstance(derivative_relative, str) or not derivative_relative:
        raise ValueError("empty derived relative path")
    rel = Path(derivative_relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("derived path is unsafe")
    derived = CACHE / "text" / rel
    if (not derived.is_file() or derived.is_symlink() or
            not derived.resolve().is_relative_to((CACHE / "text").resolve()) or
            derived.stat().st_size > 350 * 1024 * 1024):
        raise ValueError("derived source absent/unsafe/oversized")
    return derived


def checked_original(relative: str, *, include_publisher_text: bool = False,
                     publication_record: dict | None = None) -> dict:
    path = publisher_path(relative)
    manifest = json.loads(SOURCES.read_text())["sources"]
    entries = [r for r in manifest if r["path"] == relative]
    if len(entries) != 1:
        raise ValueError("source not found uniquely in original manifest")
    download = json.loads(LEDGER.read_text())["items"]
    completed = [r for r in download if r["path"] == relative and r["status"] == "complete"]
    if len(completed) != 1:
        raise ValueError("original source not completed in the download ledger")
    record = completed[0]
    if publication_record is not None and (
            entries[0] != publication_record or not publication_receipt_matches(entries[0], record)):
        raise ValueError("publication catalog record changed or acquisition URL differs from completed receipt")
    capture = None
    if (include_publisher_text and path.suffix.casefold() in (".md", ".markdown", ".txt", ".html", ".htm")
            and type(record.get("bytes_received")) is int and record["bytes_received"] <= PUBLISHER_TEXT_LIMIT):
        capture = bytearray()
    original_bytes = verify_publisher_revision(path, record, capture=capture)
    number = None
    if include_publisher_text and entries[0].get("kind") == "text":
        number = rfc_number(entries[0].get("url") or "")
        if number and (not matching_rfc(entries[0], number)
                       or not publication_receipt_matches(entries[0], record)
                       or not rfc_header_matches(capture or b"", number)):
            raise ValueError("RFC number, publisher header or acquisition receipt does not match")
    with closing(open_source_map_read_connection(SOURCE_MAP)) as con:
        rows = con.execute(
            "SELECT source_sha256,derived_sha256,derived_path,status "
            "FROM originals WHERE source_path=?",
            (relative,),
        ).fetchall()
    if len(rows) != 1 or rows[0][3] != "complete" or rows[0][0] != record["sha256"]:
        raise ValueError("no current complete source-map derivative for this original")
    _, derivative_sha, derivative_relative, _ = rows[0]
    derived = checked_derivative_path(derivative_relative)
    original_text = derived.read_bytes()
    if sha(original_text) != derivative_sha:
        raise ValueError("derived source SHA differs from source-map receipt")
    text = original_text.decode("utf-8", errors="strict")
    return {
        "original_path": relative,
        "title": entries[0]["title"],
        "edition_version": entries[0].get("edition_version"),
        "publication_year": entries[0].get("publication_year"),
        "original_sha256": record["sha256"],
        "original_bytes": original_bytes,
        "derived_sha256": derivative_sha,
        "derived_text": text,
        **({"publisher_text_bytes": bytes(capture)} if capture is not None and
            path.suffix.casefold() in ('.md','.markdown','.txt') else {}),
        **({"publisher_html_bytes": bytes(capture)} if capture is not None and
            path.suffix.casefold() in ('.html','.htm') else {}),
        **({"verified_rfc_number": number} if number else {}),
    }



def final_original_revision_guard(source: dict, *, original_path: str, native_file_id: int) -> dict:
    """Fail closed if one live original/source-map/native revision changed during retrieval.

    Reopen fresh read-only views after source passage/page verification and
    stream both exact source files again. This detects stale outputs caused by
    concurrent writers but is NOT a cross-SQLite atomic generation or a live
    source migration permit. No changes to any native reader or service.
    """
    if (not isinstance(source, dict) or source.get("original_path") != original_path
            or type(native_file_id) is not int or native_file_id < 1):
        raise ValueError("final original revision guard requires the selected native source")
    manifest = json.loads(SOURCES.read_text(encoding="utf-8"))["sources"]
    entries = [item for item in manifest if item.get("path") == original_path]
    if (len(entries) != 1 or entries[0].get("title") != source.get("title")
            or entries[0].get("publication_year") != source.get("publication_year")
            or entries[0].get("edition_version") != source.get("edition_version")):
        raise ValueError("curated original metadata changed during evidence search")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))["items"]
    entries = [item for item in ledger if item.get("path") == original_path
               and item.get("status") == "complete"]
    if (len(entries) != 1 or entries[0].get("sha256") != source.get("original_sha256")
            or entries[0].get("bytes_received") != source.get("original_bytes")):
        raise ValueError("completed original publisher receipt changed during evidence search")
    def assert_current_views():
        # Each new SQLite connection sees the latest committed state, rather
        # than the old read transaction taken before PDF verification.
        with closing(open_source_map_read_connection(SOURCE_MAP)) as sm:
            rows=sm.execute("SELECT source_sha256,derived_sha256,derived_path,status "
                "FROM originals WHERE source_path=?",(original_path,)).fetchall()
        if (len(rows)!=1 or rows[0][0]!=source.get("original_sha256")
                or rows[0][1]!=source.get("derived_sha256") or rows[0][3]!="complete"):
            raise ValueError("current source-map original or derived revision changed during evidence search")
        with closing(open_native_summary_read_connection(DB)) as native:
            entries=native.execute("SELECT id FROM filesystem_entries "
                "WHERE entry_type='file' AND relative_path_display=?",(original_path,)).fetchall()
        if entries != [(native_file_id,)]:
            raise ValueError("current original native FileDB identity changed during evidence search")
        return rows[0][2]
    derivative_path = assert_current_views()
    # Hash source and derivative through no-follow regular-file readers with
    # device, inode, size and modification-time rechecks at EOF.
    verify_publisher_revision(publisher_path(original_path), entries[0])
    derivative=checked_derivative_path(derivative_path)
    if file_sha256(derivative, max_bytes=derivative.stat().st_size)!=source["derived_sha256"]:
        raise ValueError("derived original text changed during evidence search")
    if assert_current_views()!=derivative_path:
        raise ValueError("derived original file path changed during evidence search")
    return {"original_publisher_bytes_rechecked_after_search":True,
            "derived_source_bytes_rechecked_after_search":True,
            "latest_source_map_revision_and_native_file_id_rechecked":True,
            "cross_store_immutable_generation_pinned":False,
            "safe_to_cutover_live":False}


def payload_path(original_path: str, original_sha: str, derived_sha: str) -> Path:
    basename = Path(original_path).name[:64]
    basename = re.sub(r"[^A-Za-z0-9_.-]+", "-", basename)
    key = revision_identity(original_path, original_sha, derived_sha)[:24]
    return OUT / (basename + "." + key + ".summary-tree.json")


def source_progress_path(original_path: str, original_sha: str, derived_sha: str) -> Path:
    """Private per-original journal, independent of canonical vector FileDB."""
    return OUT / "progress" / (
        payload_path(original_path, original_sha, derived_sha).name + ".progress.json"
    )


def save_source_progress(path: Path, value: dict) -> None:
    """Atomic, owner-only progress: no partial document or summary text leaked."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise ValueError("source progress directory must not be a symlink")
    os.chmod(path.parent, 0o700)
    content = dict(value)
    content["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fd, temp = tempfile.mkstemp(prefix=".source-progress-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as writer:
            json.dump(content, writer, indent=2, ensure_ascii=False)
            writer.write(chr(10))
            writer.flush()
            os.fsync(writer.fileno())
        os.chmod(temp, 0o600)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def load_tree(payload: dict) -> SummaryTree:
    fields = payload["tree"]
    nodes = tuple(SummaryNode(**{
        **v,
        "child_ids": tuple(v["child_ids"]),
    }) for v in fields["nodes"])
    return SummaryTree(
        source_sha256=fields["source_sha256"],
        original_characters=fields["original_characters"],
        root_id=fields["root_id"],
        nodes=nodes,
        leaf_sections=fields["leaf_sections"],
    )


class E5PassageBudget:
    """Token-only preflight using the pinned Raspi encoder's files and passage prefix."""
    def __init__(self):
        from knowledge_library_semantic_model import MODEL_DIRECTORY
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(MODEL_DIRECTORY / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.limit = json.loads((MODEL_DIRECTORY / "tokenizer_config.json").read_bytes())["model_max_length"]

    def token_count(self, text):
        return len(self.tokenizer.encode("passage: " + text).ids)


def read_source_tree(path, source):
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("verified complete summary tree not generated yet")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if (payload["schema"] != "kmd-complete-hierarchical-summary-tree-v1"
            or any(payload[key] != source[key] for key in ("original_path", "original_sha256", "derived_sha256"))):
        raise ValueError("summary tree is stale or belongs to a different original")
    tree = load_tree(payload)
    if tree.source_sha256 != source["derived_sha256"] or tree.original_characters != len(source["derived_text"]):
        raise ValueError("summary tree does not cover the verified derivative")
    validate_complete_summary_tree(tree)
    for node in tree.nodes:
        if node.summary_sha256 != sha(node.summary.encode()) or (
            node.native_analysis is not None and node.native_analysis["document_summary"] != node.summary
        ):
            raise ValueError("saved summary or native analysis differs from node digest")
    return payload, tree, sha(raw)


def recovery_path(base, digest):
    return base.with_name(base.name + ".recovery-" + digest + ".json")


def import_tree(base, source):
    payload, tree, digest = read_source_tree(base, source)
    candidate = recovery_path(base, digest)
    if not candidate.exists() and not candidate.is_symlink():
        return payload, tree
    recovered, tree, _ = read_source_tree(candidate, source)
    if (recovered.get("recovered_from_tree_sha256") != digest or recovered["model"] != payload["model"]
            or not recovered.get("native_facets") or not payload.get("native_facets")):
        raise ValueError("recovered summary tree is not bound to its saved predecessor")
    return recovered, tree


def save_tree_once(output, payload):
    """Atomically expose a complete owner-only file; an existing tree wins the race."""
    fd, temporary = tempfile.mkstemp(prefix=".complete-tree-", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as writer:
            json.dump(payload, writer, ensure_ascii=False, indent=2)
            writer.write("\n")
            writer.flush()
            os.fsync(writer.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            return (not output.is_symlink() and output.is_file()
                    and json.loads(output.read_bytes()) == json.loads(Path(temporary).read_bytes()))
        return True
    finally:
        Path(temporary).unlink(missing_ok=True)


def summary_prompt_title(source: dict) -> str:
    """Prefer an original's printed edition over a stale extraction header."""
    title = source["title"][:300]
    edition = source.get("edition_version")
    if (not isinstance(edition, str) or not 1 <= len(edition) <= 100
            or not edition.isprintable() or not edition.strip()):
        return title
    text = source["derived_text"]
    body = text.find("\n[[SOURCE UNIT: ")
    printed_line = re.compile(r"^[ \t]*" + re.escape(edition) + r"[ \t]*\r?$", re.MULTILINE)
    if body < 0 or not printed_line.search(text, body):
        return title
    suffix = f" [Publisher's printed edition, authoritative over catalog dates: {edition}]"
    return title[:300 - len(suffix)] + suffix


def stale_catalog_header_length(source: dict) -> int:
    """Identify just the extractor-owned preamble for the same original."""
    text = source["derived_text"]
    prefix = "KMD source provenance (the original PDF/HTML/ZIP/XLSX, not a new publisher book):\n"
    if not text.startswith(prefix) or not source.get("publication_year"):
        return 0
    end = text.find("\n\n", len(prefix), 16_384)
    if end < 0:
        return 0
    try:
        header = json.loads(text[len(prefix):end])
    except ValueError:
        return 0
    if (not isinstance(header, dict) or header.get("source_path") != source["original_path"]
            or header.get("source_sha256") != source["original_sha256"]
            or header.get("publication_year") == source["publication_year"]):
        return 0
    marker = text.find("[[SOURCE UNIT: ", end + 2, 16_384)
    if marker < 0 or text[end:marker].strip():
        return 0
    return marker


def build(args) -> None:
    source = checked_original(args.original_path)
    # The full canonical title remains in the source manifest/payload.
    # Only the model prompt receives a bounded title, so unusually long
    # publisher titles never prevent complete document processing.
    model_title = summary_prompt_title(source)
    if getattr(args,"native_facets",False) and args.output_tokens<700:
        raise ValueError("native keyword/facet output requires at least 700 reserved generation tokens")
    output = payload_path(source["original_path"], source["original_sha256"], source["derived_sha256"])
    recovery_digest = None
    previous_tree = None
    recover = getattr(args, "recover_embedding_context", False)
    need_budget = getattr(args, "native_facets", False) and (recover or not args.plan_only)
    passage_budget = E5PassageBudget() if need_budget else None

    def validate_native(analysis, start, end):
        prepare_node_facets(
            analysis["document_summary"], analysis, text=source["derived_text"], start=start, end=end,
            embedding_client=passage_budget, limit=passage_budget.limit,
        )

    def retain_native(original, retry, start, end):
        return retain_fitting_native_fields(original, retry, text=source["derived_text"], start=start,
                                            end=end, embedding_client=passage_budget, limit=passage_budget.limit)

    if recover:
        prior, tree, recovery_digest = read_source_tree(output, source)
        if not (passage_budget and prior.get("native_facets") and args.model == prior["model"]):
            raise ValueError("recovery requires the saved native tree's generation model")
        overflows = 0
        for node in tree.nodes:
            if node.native_analysis is None:
                raise ValueError("recovery requires a complete native facet tree")
            try:
                validate_native(node.native_analysis, node.start_char, node.end_char)
            except EmbeddingContextExceeded:
                overflows += 1
        if not overflows:
            raise ValueError("saved tree fits the passage encoder; recovery is unnecessary")
        recovered = recovery_path(output, recovery_digest)
        if recovered.exists() or recovered.is_symlink():
            import_tree(output, source)
            print("EXISTING_CONTEXT_RECOVERY", str(recovered), flush=True)
            return
        output = recovered
        previous_tree = tree
    model = LocalSummaryModel(
        args.model_url, model=args.model,
        timeout_seconds=args.model_timeout,
        context_cap_tokens=getattr(args,"context_cap_tokens",None),
        shared_slot_idle_seconds=getattr(args,"shared_slot_idle_seconds",0.0),
    )
    engine = HierarchicalSummarizer(
        model, output_tokens=args.output_tokens, safety_tokens=args.safety_tokens,
        max_levels=args.max_levels, max_nodes=args.max_nodes,
        native_facets=getattr(args,"native_facets",False),
        native_facets_compact=getattr(args,"compact_native_facets",False),
        catalog_header_chars=stale_catalog_header_length(source),
    )
    sections, groups = engine.plan_source(title=model_title, text=source["derived_text"])
    plan = {
        "title": source["title"],
        "original_path": source["original_path"],
        "source_sections": len(sections),
        "first_level_model_calls": len(groups),
        "complete_source_characters": len(source["derived_text"]),
        "configured_context": model.context_tokens,
        "reserved_output": args.output_tokens,
        "reserved_safety": args.safety_tokens,
        "native_facets":getattr(args,"native_facets",False),
        "compact_native_facets":getattr(args,"compact_native_facets",False),
        "shared_model_idle_admission_seconds":getattr(args,"shared_slot_idle_seconds",0),
    }
    print("PLAN", json.dumps(plan, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    elapsed_from = time.monotonic()
    progress_file = source_progress_path(
        source["original_path"], source["original_sha256"], source["derived_sha256"]
    )
    if recovery_digest is not None:
        progress_file = progress_file.with_name(output.name + ".progress.json")
    activity = SummaryActivity()
    progress = {
        "schema": "knowmoredirt-recursive-summary-node-progress-v1",
        "status": "summarizing",
        "original_path": source["original_path"],
        "original_sha256": source["original_sha256"],
        "derived_sha256": source["derived_sha256"],
        "model": args.model,
        "source_characters": len(source["derived_text"]),
        "first_level_summary_nodes_planned": len(groups),
        "completed_first_level_nodes": 0,
        **asdict(activity),
        "completed_nodes": 0,
        "last_summary_level": None,
        "last_node_id": None,
        "elapsed_seconds": 0.0,
    }
    save_source_progress(progress_file, progress)
    def on_node(node: SummaryNode, count: int) -> None:
        progress.update(asdict(activity))
        progress["completed_nodes"] = count
        progress["last_summary_level"] = node.level
        progress["last_node_id"] = node.node_id
        progress["last_summary_sha256"] = node.summary_sha256
        progress["last_source_start_char"] = node.start_char
        progress["last_source_end_char"] = node.end_char
        progress["elapsed_seconds"] = round(time.monotonic() - elapsed_from, 1)
        if node.level == 0:
            progress["completed_first_level_nodes"] += 1
            progress["completed_source_characters_in_leaf_pass"] = node.end_char
        # First-level completion is a real known denominator. The number of
        # later parent calls is unknown until each recursive stage is planned.
        progress["first_level_percent"] = round(
            100.0 * progress["completed_first_level_nodes"] / max(1,len(groups)), 2
        )
        save_source_progress(progress_file, progress)
        print("KMD_HIERARCHICAL_SUMMARY_NODE", json.dumps({
            "original_path": source["original_path"], "level": node.level,
            "completed_calls": activity.completed_model_calls,
            "completed_nodes": count,
            "first_level_percent": progress["first_level_percent"],
            "source_end_char": node.end_char,
            "summary_sha256": node.summary_sha256,
        }, ensure_ascii=False), flush=True)
    try:
        tree = engine.summarize(
            title=model_title, text=source["derived_text"],
            max_calls=args.max_calls, on_node=on_node,
            validate_native=validate_native if passage_budget is not None else None,
            retain_native=retain_native, previous_tree=previous_tree, activity=activity,
        )
    except Exception as error:
        progress.update(asdict(activity))
        progress["status"] = "incomplete_generation"
        progress["last_error"] = f"{type(error).__name__}: {str(error)[:260]}"
        progress["elapsed_seconds"] = round(time.monotonic() - elapsed_from, 1)
        save_source_progress(progress_file, progress)
        raise
    if tree.source_sha256 != sha(source["derived_text"].encode()):
        raise RuntimeError("generated summary tree source mismatch")
    if len(tree.nodes) < 1:
        raise RuntimeError("zero summaries generated")
    payload = {
        "schema": "kmd-complete-hierarchical-summary-tree-v1",
        "original_path": source["original_path"],
        "title": source["title"],
        "original_sha256": source["original_sha256"],
        "derived_sha256": source["derived_sha256"],
        "model": args.model,
        "model_context": model.context_tokens,
        "reserved_output_tokens": args.output_tokens,
        "native_facets":getattr(args,"native_facets",False),
        "tree": asdict(tree),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    os.chmod(OUT, 0o700)
    if recovery_digest is not None:
        payload["recovered_from_tree_sha256"] = recovery_digest
    if not save_tree_once(output, payload):
        progress["status"] = "existing_complete_tree_differs"
        progress["last_error"] = "different completed tree already exists; replacement refused"
        save_source_progress(progress_file, progress)
        print("EXISTING COMPLETE SUMMARY TREE FOUND; refusing replacement", flush=True)
        return
    progress.update({
        "status": "complete_tree_saved",
        **asdict(activity),
        "completed_nodes": len(tree.nodes),
        "completed_first_level_nodes": sum(node.level == 0 for node in tree.nodes),
        "first_level_percent": 100.0,
        "completed_source_characters_in_leaf_pass": len(source["derived_text"]),
        "root_summary_sha256": tree.root.summary_sha256,
        "root_node_id": tree.root_id,
        "root_level": tree.root.level,
        "elapsed_seconds": round(time.monotonic() - elapsed_from, 1),
        "complete_tree_path": str(output),
    })
    save_source_progress(progress_file, progress)
    print("COMPLETE_TREE", json.dumps({
        "original_path": source["original_path"],
        "nodes": len(tree.nodes),
        "levels": tree.root.level + 1,
        "leaf_summary_nodes": sum(n.level == 0 for n in tree.nodes),
        "total_summary_characters": sum(len(n.summary) for n in tree.nodes),
        "source_characters": tree.original_characters,
        "root_summary_length": len(tree.root.summary),
        "private_payload": str(output),
    }, ensure_ascii=False), flush=True)


def acquire_native_filedb_content_lock(
    file_descriptor: int, *, wait_seconds: float = 90.0,
    poll_seconds: float = 0.25,
) -> None:
    """Bounded interprocess writer admission for the SAME canonical FileDB.

    Multiple source workers may generate and embed concurrently; only
    their final original-KMD SQLite transactions are serialized under
    the existing writer lock. Never silently skip another source.
    """
    import fcntl
    if not 1 <= wait_seconds <= 300 or not 0.05 <= poll_seconds <= 10:
        raise ValueError("invalid bounded FileDB lock wait")
    deadline=time.monotonic()+wait_seconds
    while True:
        try:
            fcntl.flock(file_descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return
        except BlockingIOError:
            remaining=deadline-time.monotonic()
            if remaining<=0:
                raise TimeoutError("native KMD content lock remained busy beyond bounded wait")
            time.sleep(min(poll_seconds,remaining))


def open_native_summary_read_connection(
    database: Path, *, wait_seconds: float = NATIVE_READER_WAIT_SECONDS,
) -> sqlite3.Connection:
    """Bounded source-identity read of the SAME native FileDB during commits.

    SQLite's default five-second reader busy timeout is insufficient while
    another source worker commits a large DELETE-journal transaction. This
    function never changes journal mode, acquires a writer lock, creates a
    new database, or treats delayed source evidence as a valid source.
    """
    if (not isinstance(wait_seconds,(int,float)) or isinstance(wait_seconds,bool)
            or not 1 <= wait_seconds <= 300):
        raise ValueError("native read-only source wait must be bounded")
    database = Path(database)
    if database.is_symlink() or not database.is_file():
        raise ValueError("canonical native FileDB is missing or symlinked")
    connection=sqlite3.connect(database.resolve().as_uri()+"?mode=ro",
                               uri=True,timeout=float(wait_seconds))
    try:
        connection.execute("PRAGMA query_only=ON")
    except Exception:
        connection.close()
        raise
    return connection


def import_source(args):
    source = checked_original(args.original_path)
    tree_file = payload_path(
        source["original_path"], source["original_sha256"], source["derived_sha256"],
    )
    payload, tree = import_tree(tree_file, source)
    from knowledge_library_semantic_model import LocalE5
    model = LocalE5(threads=2, batch_size=4)
    with closing(open_native_summary_read_connection(DB)) as ro:
        chunks, reps = summary_rows(
            ro,
            original_path=source["original_path"],
            original_sha256=source["original_sha256"],
            original_bytes=source["original_bytes"],
            derived_sha256=source["derived_sha256"],
            derived_text=source["derived_text"],
            summary_start_char=0,
            tree=tree,
            embedding_client=model,
            analysis_model=payload["model"],
            require_native_terms=bool(payload.get("native_facets")),
            collection_id=COLLECTION,
        )
    if (len(chunks)!=len(tree.nodes)
            or sum(r["representation_kind"] in ("summary","summary_long") for r in reps)!=len(tree.nodes)):
        raise RuntimeError("missing required source-linked native summary vector")
    print("VERIFIED_SUMMARY_VECTORS", json.dumps({
        "original_path": source["original_path"],
        "nodes": len(tree.nodes),
        "vector_count": len(reps),
        "representation_kinds":{
            kind:sum(r["representation_kind"]==kind for r in reps)
            for kind in sorted({r["representation_kind"] for r in reps})
        },
        "levels": tree.root.level + 1,
        "summary_chars": sum(len(n.summary) for n in tree.nodes),
        "bytes_halfprecision_vectors": sum(len(bytes(r["embedding_blob"])) for r in reps),
    }), flush=True)
    if not args.write:
        print("DRY_RUN_NO_WRITES: rerun import with --write after legacy reader guard is tested")
        return
    # The legacy UI's global reader must be protected before a new higher-
    # scoring summary representation can shadow its extractable raw passage.
    live_reader = (INFRA / "knowledge_library_semantic_query.py").read_text(encoding="utf-8")
    if GUARD_TEXT not in live_reader:
        raise RuntimeError("live Infra source-excerpt search not yet protected from summary-only candidates")
    lockfile = DB.parent / ".primary-kmd-content.lock"
    with lockfile.open("a+") as lock:
        acquire_native_filedb_content_lock(lock.fileno(),wait_seconds=NATIVE_WRITER_WAIT_SECONDS)
        # Check after taking the inter-process writer lock so concurrent
        # native jobs cannot race this admission. SQLite keeps its own
        # transactional size ceiling even if free disk changes afterward.
        require_storage_admission(DB, budget_bytes=DB_BUDGET)
        with sqlite3.connect(DB, timeout=60) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            result = append_summary_rows(
                conn, chunk_rows=chunks, representation_rows=reps,
                max_database_bytes=DB_BUDGET,
            )
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite integrity failure detected after summary insertion")
    print("CANONICAL_KMD_UPDATED", json.dumps(result), flush=True)


def _unique_path_records(records, path_key="path"):
    """An ambiguous manifest or completion receipt cannot identify a revision."""
    by_path = {}
    ambiguous = set()
    for record in records:
        path = record[path_key]
        if path in by_path:
            ambiguous.add(path)
        else:
            by_path[path] = record
    for path in ambiguous:
        del by_path[path]
    return by_path


def _current_versions():
    manifest = _unique_path_records(json.loads(SOURCES.read_text())["sources"])
    ledger = _unique_path_records(
        record for record in json.loads(LEDGER.read_text())["items"]
        if record.get("status") == "complete"
    )
    with closing(open_source_map_read_connection(SOURCE_MAP)) as con:
        source_map = _unique_path_records(con.execute(
            "SELECT source_path,source_sha256,derived_sha256,derived_path,status "
            "FROM originals WHERE status='complete'"
        ), path_key=0)
    return manifest, ledger, source_map


_SUMMARY_FINAL_SOURCE_PROOF_LIMIT_BYTES = 1024*1024*1024


def final_summary_discovery_revision_guard(found:list[dict], *,
                                           manifest:dict, ledger:dict, sources:dict)->dict:
    """Reject source or native summary edits occurring after original ranking.

    Reopen fresh read-only views and rehash all selected publisher/derived
    originals. This is NOT an atomic generation pin across both SQLite DBs.
    """
    if not isinstance(found,list) or len(found)>30:
        raise ValueError('unbounded summary source final proof')
    if not found:
        return {'returned_originals_rechecked_after_summary_ranking':0,
                'cross_store_immutable_generation_pinned':False,
                'safe_to_cutover_live':False}
    records={};total=0
    for item in found:
        path=item.get('original_path')
        if (not isinstance(path,str) or not path or path in records
                or path.startswith('/') or '\\' in path
                or any(part in ('','.','..') for part in path.split('/'))
                or path not in manifest or path not in ledger or path not in sources
                or type(item.get('filesystem_entry_id')) is not int
                or item['filesystem_entry_id']<1):
            raise ValueError('summary original source revision has no unique current identity')
        receipt=ledger[path];rev=sources[path]
        if (receipt.get('sha256')!=item.get('original_sha256')
                or receipt.get('bytes_received')!=item.get('original_bytes')
                or receipt.get('status')!='complete'
                or rev[1]!=item.get('original_sha256')
                or not isinstance(rev[2],str)):
            raise ValueError('original publisher or source-map revision changed after summary ranking')
        nodes=item.get('matched_summary_nodes')
        if not isinstance(nodes,list) or not 1<=len(nodes)<=30:
            raise ValueError('ranked original lacks bounded source-linked summary nodes')
        for node in nodes:
            if (not isinstance(node,dict) or node.get('original_path')!=path
                    or node.get('filesystem_entry_id')!=item['filesystem_entry_id']
                    or node.get('original_sha256')!=rev[1]
                    or node.get('derived_sha256')!=rev[2]
                    or not isinstance(node.get('native_chunk_id'),str)
                    or not isinstance(node.get('native_representation_id'),str)):
                raise ValueError('native ranked summary identity is not current-source bound')
        derivative=checked_derivative_path(rev[3]);original_size=item['original_bytes']
        # Limit total full-source rehash work even when some approved books
        # are hundreds of megabytes. Reject unverified wide searches safely.
        if (type(original_size) is not int or original_size<1
                or original_size+derivative.stat().st_size+total>_SUMMARY_FINAL_SOURCE_PROOF_LIMIT_BYTES):
            raise ValueError('summary source final proof exceeds bounded physical byte budget; narrow top or named source')
        total+=original_size+derivative.stat().st_size
        records[path]=(item,rev,receipt)

    def metadata_is_current():
        latest_manifest,latest_ledger,latest_sources=_current_versions()
        for path,(item,rev,receipt) in records.items():
            if (latest_manifest.get(path)!=manifest[path]
                    or latest_ledger.get(path)!=receipt
                    or latest_sources.get(path)!=rev):
                raise ValueError('source-map or publisher original revision changed after summary ranking')

    def native_is_current():
        with closing(open_native_summary_read_connection(DB)) as db:
            db.execute('PRAGMA query_only=ON')
            for path,(item,rev,_) in records.items():
                ids=db.execute('SELECT id,size_bytes FROM filesystem_entries '
                        'WHERE entry_type=? AND relative_path_display=?',('file',path)).fetchall()
                if ids!=[(item['filesystem_entry_id'],item['original_bytes'])]:
                    raise ValueError('native FileDB original source identity changed after summary ranking')
                for node in item['matched_summary_nodes']:
                    rows=db.execute("""SELECT c.filesystem_entry_id,c.content_sha256,c.content_object_id,
                             c.text_sha256,c.start_char,c.end_char,r.representation_text_sha256,
                             r.generation_json,r.analysis_status,r.representation_kind
                             FROM content_chunks c JOIN content_representations r
                             ON c.chunk_id=r.chunk_id
                             WHERE c.chunk_id=? AND r.representation_id=?""",
                             (node['native_chunk_id'],node['native_representation_id'])).fetchall()
                    if len(rows)!=1:
                        raise ValueError('native summary representation changed after source ranking')
                    (source_id,source_sha,source_object,passage_sha,start,end,
                     summary_sha,metadata,state,kind)=rows[0]
                    if (source_id!=item['filesystem_entry_id'] or source_sha!=rev[1]
                            or source_object!='sha256:'+rev[2]
                            or passage_sha!=node['source_passage_sha256']
                            or start!=node['source_start_char'] or end!=node['source_end_char']
                            or summary_sha!=node['summary_sha256']
                            or state!='complete' or kind!=node['summary_kind']):
                        raise ValueError('native summary source span or representation revision changed after ranking')
                    try:meta=json.loads(metadata)
                    except (ValueError,TypeError) as exc:
                        raise ValueError('native summary generation metadata malformed after ranking') from exc
                    if (not isinstance(meta,dict) or meta.get('derived_sha256')!=rev[2]
                            or meta.get('original_sha256')!=rev[1]
                            or meta.get('summary_sha256')!=node['summary_sha256']):
                        raise ValueError('native summary generation revision changed after ranking')

    metadata_is_current();native_is_current()
    for path,(item,rev,receipt) in records.items():
        record=manifest[path]
        original={'original_path':path,'title':record.get('title'),
                  'publication_year':record.get('publication_year'),
                  'edition_version':record.get('edition_version'),
                  'original_sha256':rev[1],'derived_sha256':rev[2],
                  'original_bytes':item['original_bytes']}
        proof=final_original_revision_guard(original,original_path=path,
                                          native_file_id=item['filesystem_entry_id'])
        if not proof['latest_source_map_revision_and_native_file_id_rechecked']:
            raise ValueError('incomplete summary original source final revision proof')
    metadata_is_current();native_is_current()
    return {'returned_originals_rechecked_after_summary_ranking':len(records),
            'complete_original_and_derived_source_bytes_rehashed':total,
            'current_native_summary_representations_and_source_spans_rechecked':True,
            'fresh_catalog_receipts_source_map_and_native_file_IDs_rechecked':True,
            'cross_store_immutable_generation_pinned':False,
            'safe_to_cutover_live':False}

def _search_impl(args) -> dict:
    manifest, ledger, sources = _current_versions()
    reviewed_exclusions = load_reviewed_summary_exclusions(REVIEWED_SUMMARY_EXCLUSIONS)
    excluded_reviewed = [0]
    name_match = resolve_source_name(args.query, manifest)
    preferred_path = name_match.get("original_path") if name_match else None
    publication = name_match.get("matched_identifier") if name_match else None
    checked = set()

    def verify_document(hits):
        path = hits[0]["original_path"]
        original = ledger.get(path)
        derivative = sources.get(path)
        if path not in manifest or not original or not derivative or original.get("sha256") != derivative[1]:
            return []
        if (path == preferred_path and publication
                and not publication_receipt_matches(manifest[path], original)):
            return []
        current = [
            hit for hit in hits
            if hit["original_path"] == path and hit["original_sha256"] == derivative[1]
            and hit["derived_sha256"] == derivative[2] and hit["original_bytes"] == original.get("bytes_received")
        ]
        if not current:
            return []
        intervals = {(hit["source_start_char"], hit["source_end_char"]) for hit in current}
        try:
            if path == preferred_path and name_match["kind"] == "catalog_rfc":
                if original["bytes_received"] > PUBLISHER_TEXT_LIMIT:
                    return []
                raw = bytearray()
                verify_publisher_revision(publisher_path(path), original, capture=raw)
                if not rfc_header_matches(raw, publication.removeprefix("RFC ")):
                    return []
            else:
                verify_publisher_revision(publisher_path(path), original)
            derived = checked_derivative_path(derivative[3])
            blocks = regular_file_blocks(derived)
            subjects = {}
            if any("publisher_classification" in hit for hit in current):
                prefix = next(blocks, b"")
                # The complete stream is strictly decoded and hashed below.
                subjects = first_page_ccs(prefix[:4 * PREFIX_CHARS].decode("utf-8", errors="ignore"))
                blocks = chain((prefix,), blocks)
            full_digest, passages = source_passage_digests(blocks, intervals)
        except (OSError, ValueError):
            return []
        if full_digest != derivative[2]:
            return []
        verified = [hit for hit in current
                    if passages.get((hit["source_start_char"], hit["source_end_char"])) == hit["source_passage_sha256"]
                    and ("publisher_classification" not in hit
                         or subjects.get(hit["summary"]) == hit["publisher_classification"])]
        # A separately reviewed false claim can remain in generated native
        # vectors while two genuine indexers write other source documents.
        # Exclude ONLY its exact native representation and source revision;
        # the original and all other original-linked summaries remain usable.
        verified, rejected = exclude_reviewed_summary_nodes(verified, reviewed_exclusions)
        excluded_reviewed[0] += rejected
        if verified:
            checked.add((path, derivative[1], derivative[2]))
        return verified

    missing_identifier = name_match and name_match["status"] == "no_catalog_match"
    ambiguous_reference = name_match and name_match["status"] == "ambiguous"
    named_title_lead = (name_match and name_match.get("kind") == "catalog_title_lead")
    # The native FileDB is also written by two legitimate concurrent source
    # summary workers. SQLite's default five-second reader busy budget is too
    # short in DELETE journal mode: a query can fail with "database is locked"
    # at the initial batched vector SELECT, even though source bytes are healthy.
    # Use the existing bounded 120-second read-only native helper used by
    # other source readers. closing() releases the connection promptly rather
    # than leaving a context-managed-but-unclosed native read transaction.
    with closing(open_native_summary_read_connection(DB)) as conn:
        found = []
        # No model request or whole-corpus vector scan is justified when an
        # explicitly named publisher original has zero native summary rows.
        missing_named_summary = bool(named_title_lead and not
            has_native_summary_candidates(conn, preferred_path))
        if not missing_identifier and not ambiguous_reference and not missing_named_summary:
            from knowledge_library_semantic_model import LocalE5
            model = LocalE5(threads=2, batch_size=4)
            vector = model.embed_queries([args.query])[0]
            found = search_summary_entries_batched(
                conn, vector, limit=args.top,
                include_facets=getattr(args, "with_facets", False),
                nodes_per_document=args.nodes_per_original,
                expected_model=model.model,
                filter_source_nodes=verify_document, preferred_path=preferred_path,
                restrict_original_path=preferred_path if named_title_lead else None,
            )
        snapshot = summary_index_scope(conn)
    out = []
    for row in found:
        original = row["original_path"]
        out.append({
            "original_path": original,
            "title": manifest.get(original, {}).get("title"),
            "source_url": (manifest.get(original, {}).get("source_url")
                           or manifest.get(original, {}).get("url")),
            "similarity": round(row["similarity"], 4),
            "best_summary": row["best_summary"],
            "best_level": row["best_level"],
            "indexed_summary_nodes_for_document": row["indexed_summary_nodes_for_document"],
            "matched_summary_nodes": row["matched_summary_nodes"],
            "source_scope": "summaries_for_document_discovery_not_original_factual_evidence",
        })
    warning = ("The ranked results are nearest summaries. Inspect the original "
               "to determine whether it contains the information you need.")
    if preferred_path is not None:
        verified = bool(out and out[0]["original_path"] == preferred_path)
        name_match["status"] = "verified" if verified else "no_verified_index_match"
        if verified:
            out[0]["rank_basis"] = "explicit_source_name"
            warning = ("The requested original was verified by name and revision. Cosine similarity ranks "
                       "its passages and the other documents. Check factual claims in the original.")
        else:
            if name_match["kind"] == "catalog_title_lead":
                # The user explicitly cited a unique short publisher title.
                # If its summary is missing, stale, or not yet indexed,
                # unrelated vector matches must never masquerade as answers.
                out = []
                checked.clear()
                warning = ("The explicitly cited original has no verified indexed summary; no different "
                           "publication was substituted. Use source-evidence with the exact catalog path "
                           "to inspect the existing publisher original directly.")
            else:
                warning = ("The named original has no verified summary result. Any ranked documents are other "
                           "sources. Use source-evidence with the exact catalog path to inspect the named original.")
    elif missing_identifier and name_match["kind"] == "catalog_rfc":
        warning = "The catalog has no archived RFC Editor text matching this RFC number."
    elif missing_identifier:
        warning = "The catalog has no archived PDF matching this publication identifier and requested version."
    elif ambiguous_reference:
        # Neither a source quote nor an edition can be attributed when the
        # catalog contains multiple real publications with the same name.
        warning = ("The explicitly cited source name identifies multiple catalog originals. "
                   "No document was substituted; select the full title or exact original path.")
    if excluded_reviewed[0]:
        warning += (" Reviewed, source-contradicted generated summary representations "
                    "were excluded from ranking. Consult the publisher original for figures.")
    final_source_proof=final_summary_discovery_revision_guard(
        [row for row in found if any(x['original_path']==row['original_path'] for x in out)],
        manifest=manifest,ledger=ledger,sources=sources)
    return {
        **({"source_name_match": name_match} if name_match else {}),
        "final_current_summary_source_revision_check":final_source_proof,
        "query": args.query,
        "database": str(DB),
        "retrieval_engine": ("exact_publication_catalog_lookup_v1" if missing_identifier
                             else "exact_ambiguous_catalog_lookup_v1" if ambiguous_reference
                             else "exact_catalog_unindexed_original_v1" if missing_named_summary
                             else "native_batched_facets_cosine_v1"
                             if getattr(args,"with_facets",False)
                             else "native_batched_cosine_v1"),
        "documents": out,
        "count": len(out),
        "originals_validated_for_version": len(checked),
        "reviewed_incorrect_generated_summary_representations_excluded": excluded_reviewed[0],
        "reviewed_summary_exclusion_policy": "exact_source_revision_and_native_representation_only_not_automatic_truth",
        "source_passage_validation": "exact_character_interval_sha256_in_current_derivative",
        "provenance_policy": "publisher_and_derived_sha256_checked;summary_is_not_primary_evidence",
        "index_scope": {
            "native_summary_vectors": snapshot["native_summary_vectors"],
            "summarized_originals": snapshot["original_documents_with_summaries"],
            "complete_extracted_originals": len(sources),
            "indexed_original_percent": round(100.0*snapshot["original_documents_with_summaries"]/max(1,len(sources)),6),
        },
        "relevance_warning": warning,
    }


def search(args) -> None:
    """Always retain actual query, ranked results and failure for later QA."""
    if len(args.query) > 1000:
        raise ValueError("search query exceeds bound")
    started = time.monotonic()
    try:
        result = _search_impl(args)
    except Exception as error:
        receipt = make_search_audit(
            query=args.query, result=None,
            elapsed_seconds=time.monotonic() - started,
            error=f"{type(error).__name__}: {str(error)[:300]}",
            model="intfloat/multilingual-e5-small-onnx-int8",
            database=str(DB),
        )
        append_query_audit(receipt)
        raise
    receipt = make_search_audit(
        query=args.query, result=result,
        elapsed_seconds=time.monotonic() - started,
        model="intfloat/multilingual-e5-small-onnx-int8",
        database=str(DB),
    )
    path = append_query_audit(receipt)
    result["query_audit_logged"] = True
    result["private_query_audit_file"] = str(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def review_search_log(args) -> None:
    """Inspect the latest private query/rank receipts."""
    records = recent_search_audits(directory=AUDIT_DIR, limit=args.last)
    print(json.dumps({
        "count": len(records), "audit_directory": str(AUDIT_DIR),
        "search_audits": records,
    }, indent=2, ensure_ascii=False))


def _original_passages(source, entry_id, query, top, snippet_characters, *,
                       scope_context_characters=0):
    return source_evidence_candidates(
        original_path=source["original_path"], original_title=source["title"], filesystem_entry_id=entry_id,
        original_sha256=source["original_sha256"], derived_sha256=source["derived_sha256"],
        derived_text=source["derived_text"], publisher_text_bytes=source.get("publisher_text_bytes"),
        publisher_html_bytes=source.get("publisher_html_bytes"),
        query=query, top=top, snippet_characters=snippet_characters,
        scope_context_characters=scope_context_characters,
    )


def _native_passage_search(query, top):
    # This search belongs to the existing Infra semantic reader on Raspi.
    from knowledge_library_semantic_query import search as native_search

    return native_search(query, top=top, mode="any")


def _discovered_pdf(hit, rank, args, versions, db):
    path = hit["original_path"]
    doc = {
        "rank": rank, "original_path": path, "discovery_chunk_id": hit["kmd_native_chunk_id"],
        "discovery_similarity": hit["semantic_similarity"], "passages": [],
    }
    if hit["source_kind"] != "pdf" or Path(path).suffix.casefold() != ".pdf":
        return {**doc, "status": "not_pdf"}
    manifest, ledger, derivatives = versions
    record = manifest.get(path)
    derivative = derivatives.get(path)
    original_sha = hit["original_source_sha256"]
    if (not record or record.get("kind") != "pdf" or not derivative
            or ledger.get(path, {}).get("sha256") != original_sha or derivative[1] != original_sha):
        return {**doc, "status": "source_unavailable", "reason": "discovery revision differs from catalog"}
    entry_id = hit["kmd_original_file_entry_id"]
    entry = db.execute(
        "SELECT id FROM filesystem_entries WHERE id=? AND relative_path_display=? AND entry_type='file'",
        (entry_id, path),
    ).fetchone()
    if entry is None:
        return {**doc, "status": "source_unavailable", "reason": "discovery entry differs from current FileDB"}

    try:
        source = checked_original(path, publication_record=record)
        if (source["original_sha256"], source["derived_sha256"]) != (original_sha, derivative[2]):
            raise ValueError("source revision changed during discovery")
        # Look beyond the requested count when a high-ranked excerpt lacks a physical page match.
        candidates = _original_passages(source, entry_id, args.query,
                                        args.passages + 3, args.snippet_characters,
                                        scope_context_characters=getattr(
                                            args, "scope_context_characters", 0))
        hits = candidates["passages"]
        unreadable_dropped = int(candidates.get("unreadable_pdf_excerpts_dropped", 0))
        proofs = verify_pdf_page_excerpts(publisher_path(path), original_sha, hits)
    except (ValueError, OSError) as exc:
        return {**doc, "status": "source_unavailable", "reason": str(exc)[:300]}

    verified = [
        dict(passage, publisher_pdf_page=proof, source_lexical_rank=rank,
             matched_excerpt_query_terms=len(passage.get("matched_excerpt_terms", ())))
        for rank, (passage, proof) in enumerate(zip(hits, proofs, strict=True), 1)
        if proof["status"] == "verified"
    ]
    # A high BM25 match can be generic and omit the decisive sentence.
    # Prefer physical-page-verified excerpts that cover more DISTINCT actual
    # query terms, breaking ties with the original lexical order. Retain that
    # order as source_lexical_rank; these are candidate excerpts, not answers.
    verified.sort(key=lambda passage: (-passage["matched_excerpt_query_terms"],
                                       passage["source_lexical_rank"]))
    verified = verified[:args.passages]
    return {
        **doc, "status": "verified" if verified else "no_verified_excerpt", "passages": verified,
        "source_candidates": len(hits),
        "unreadable_pdf_excerpts_dropped": unreadable_dropped,
        "physical_page_unverified_candidates": sum(p["status"] != "verified" for p in proofs),
        "original_sha256": original_sha, "derived_sha256": derivative[2], "title": source["title"],
        "publisher_url": record.get("source_url") or record["url"], "filesystem_entry_id": entry_id,
    }


_NON_PDF_SOURCE_KINDS = frozenset(("html", "text", "skill", "xlsx", "zip"))


def _discovered_non_pdf(hit, rank, args, versions, db):
    """Find exact SHA-linked derivative excerpts without claiming literal publisher text.

    The original bytes and extracted text are both independently revision
    checked. HTML/Office/archive text extraction is a transformation; physical
    original byte offsets cannot be invented from a matching derivative span.
    """
    path = hit["original_path"]
    kind = hit["source_kind"]
    doc = {
        "rank": rank, "original_path": path, "source_kind": kind,
        "discovery_chunk_id": hit["kmd_native_chunk_id"],
        "discovery_similarity": hit["semantic_similarity"], "passages": [],
    }
    if kind not in _NON_PDF_SOURCE_KINDS:
        return {**doc, "status": "unsupported_source_kind"}
    manifest, ledger, derivatives = versions
    record = manifest.get(path)
    derivative = derivatives.get(path)
    original_sha = hit["original_source_sha256"]
    if (not record or record.get("kind") != kind or not derivative
            or ledger.get(path, {}).get("sha256") != original_sha
            or derivative[1] != original_sha):
        return {**doc, "status": "source_unavailable",
                "reason": "discovery source kind or revision differs from catalog"}
    entry_id = hit["kmd_original_file_entry_id"]
    entry = db.execute(
        "SELECT id FROM filesystem_entries "
        "WHERE id=? AND relative_path_display=? AND entry_type='file'",
        (entry_id, path),
    ).fetchone()
    if entry is None:
        return {**doc, "status": "source_unavailable",
                "reason": "discovery entry differs from current FileDB"}

    try:
        # When a plain UTF-8 text source is reproduced exactly, the existing
        # publisher_excerpt machinery may prove literal original byte offsets.
        # Other formats keep only the SHA-verified derivative span claim.
        source = checked_original(path, include_publisher_text=True, publication_record=record)
        if (source["original_sha256"], source["derived_sha256"]) != (original_sha, derivative[2]):
            raise ValueError("source revision changed during discovery")
        # Exact RFC section citations are a structural request, NOT lexical
        # search terms. The publisher's table of contents and numbered body
        # boundary must agree before returning the verified section opening.
        # Never use a TOC occurrence or arbitrary nearest vector hit as a
        # replacement for a cited section.
        rfc = source.get("verified_rfc_number")
        selector = section_key(args.query) if rfc else None
        section_result = None
        if selector:
            section_result = rfc_section_evidence(
                source, entry_id, rfc, selector, args.snippet_characters,
                cursor=getattr(args, "continue_section", None),
            )
            if section_result["rfc_section_lookup"]["status"] != "verified":
                hits = []
            else:
                hits = section_result["passages"]
                if (len(hits) != 1 or
                        any(not passage.get("publisher_excerpt") or
                            passage.get("status") not in (
                                "verified_rfc_section_opening",
                                "verified_rfc_section_continuation",
                            ) for passage in hits)):
                    raise ValueError("RFC structural section lacks original publisher byte proof")
        else:
            candidates = _original_passages(
                source, entry_id, args.query, args.passages + 3, args.snippet_characters,
                scope_context_characters=getattr(args, "scope_context_characters", 0),
            )
            hits = candidates["passages"]
    except (ValueError, OSError) as exc:
        return {**doc, "status": "source_unavailable", "reason": str(exc)[:300]}

    verified = []
    for lexical_rank, passage in enumerate(hits, 1):
        # source_evidence_candidates has checked the exact derived SHA and
        # calculated excerpt offsets and SHA over those very original-derived
        # characters. The original bytes have a separate completed ledger SHA.
        pub = passage.get("publisher_excerpt")
        html_anchor = passage.get("publisher_html_dom_anchor")
        scope = ("publisher_utf8_byte_interval" if pub else
                 "publisher_html_main_article_normalized_text" if html_anchor else
                 "verified_extracted_text_only")
        verified.append(dict(
            passage, source_lexical_rank=None if selector else lexical_rank,
            matched_excerpt_query_terms=(
                None if selector else len(passage.get("matched_excerpt_terms", ()))
            ),
            evidence_provenance={
                "status": ("verified" if pub else "verified_original_html_dom_text" if html_anchor
                           else "verified_derived_text"),
                "scope": scope,
                "physical_pdf_page_verified": False,
                "literal_publisher_byte_interval_verified": bool(pub),
                "answer_factual_sufficiency_verified": False,
                "warning": (
                    "A publisher-byte range was reproduced against the exact original UTF-8 revision."
                    if pub else
                    "The exact normalized excerpt occurs uniquely in the publisher's original "
                    "HTML article/main text. HTML entities and whitespace were normalized; "
                    "no literal publisher byte interval or correct answer has been proven."
                    if html_anchor else
                    "The original file SHA and derived text SHA are verified, but extracted "
                    "text can differ from literal publisher bytes; no physical page/byte quote proven."
                ),
            },
        ))
    if not selector:
        verified.sort(key=lambda passage: (
            -passage["matched_excerpt_query_terms"], passage["source_lexical_rank"],
        ))
    verified = verified[:args.passages]
    return {
        **doc,
        "status": (
            "verified_derived_text" if verified else
            "rfc_section_not_verified" if selector else "no_verified_excerpt"
        ),
        "passages": verified, "source_candidates": len(hits),
        "passage_selection_basis": (
            "verified_rfc_body_section_boundary" if selector
            else "within_original_lexical_evidence"
        ),
        **({"rfc_section_lookup": section_result["rfc_section_lookup"],
            "rfc_context_notice": section_result["evidence_warning"]}
           if section_result else {}),
        "original_sha256": original_sha, "derived_sha256": derivative[2],
        "title": source["title"], "publisher_url": record.get("source_url") or record["url"],
        "filesystem_entry_id": entry_id,
        "evidence_scope": "SHA-linked extracted text; publisher byte proof only where explicitly present",
    }


def _audit_passage_without_text(passage):
    """Retain exact SHA, intervals and scope indicators, never extra quoted text."""
    output = {key: value for key, value in passage.items()
              if key not in ("source_excerpt", "scope_context")}
    if "scope_context" in passage:
        context = passage["scope_context"]
        output["scope_context"] = {
            key: value for key, value in context.items() if key != "spans"
        }
        output["scope_context"]["spans"] = {
            label: {key: value for key, value in entry.items() if key != "text"}
            for label, entry in context["spans"].items()
        }
    return output


_MIXED_SOURCE_FINAL_PHYSICAL_BUDGET = 1024*1024*1024


def final_mixed_document_source_revision_guard(documents:list[dict], *, versions:tuple)->dict:
    """Refuse assembled source excerpts whose original changed while composing evidence.

    The preexisting publisher page/byte proof is not an immutable generation;
    validate again after all document/overlay operations, without index writes.
    """
    if (not isinstance(documents,list) or len(documents)>5
            or not isinstance(versions,tuple) or len(versions)!=3):
        raise ValueError('mixed original-source evidence selection is unbounded')
    manifest,ledger,sources=versions
    selected={};total=0
    for doc in documents:
        if not isinstance(doc,dict):raise ValueError('invalid mixed source discovery record')
        if doc.get('status') not in ('verified','verified_derived_text'):
            continue  # No original passage was emitted for this record.
        path=doc.get('original_path');entry=doc.get('filesystem_entry_id')
        if (not isinstance(path,str) or path in selected or path not in manifest
                or path not in ledger or path not in sources
                or path.startswith('/') or '\\' in path or any(z in ('','.','..') for z in path.split('/'))
                or type(entry) is not int or entry<1
                or not isinstance(doc.get('passages'),list) or not doc['passages']):
            raise ValueError('mixed publisher original evidence lacks unique source identity')
        source_sha=doc.get('original_sha256');derived_sha=doc.get('derived_sha256')
        current=sources[path];receipt=ledger[path]
        if (source_sha!=current[1] or derived_sha!=current[2]
                or receipt.get('status')!='complete' or receipt.get('sha256')!=source_sha
                or type(receipt.get('bytes_received')) is not int):
            raise ValueError('mixed publisher source-map or completed original revision differs')
        derivative=checked_derivative_path(current[3])
        if (total+receipt['bytes_received']+derivative.stat().st_size>
                _MIXED_SOURCE_FINAL_PHYSICAL_BUDGET):
            raise ValueError('mixed original publisher full-SHA proof exceeds bounded physical read budget; narrow source selection')
        total+=receipt['bytes_received']+derivative.stat().st_size
        selected[path]=(doc,current,receipt)
    if not selected:
        return {'verified_source_originals_rechecked_after_combined_evidence':0,
                'cross_store_immutable_generation_pinned':False,
                'safe_to_cutover_live':False}

    # A later writer can change one of several selected originals while a
    # different original is being independently PDF-page validated. Reject
    # all returned citations if any source revision or native ID now differs.
    def check_metadata():
        latest_manifest,latest_ledger,latest_sources=_current_versions()
        with closing(open_native_summary_read_connection(DB)) as db:
            db.execute('pragma query_only=ON')
            for path,(doc,rev,receipt) in selected.items():
                if (latest_manifest.get(path)!=manifest[path]
                        or latest_ledger.get(path)!=receipt
                        or latest_sources.get(path)!=rev):
                    raise ValueError('mixed original source-map or publisher receipt changed after evidence ranking')
                ids=db.execute("SELECT id FROM filesystem_entries WHERE entry_type='file' "
                               "AND relative_path_display=?",(path,)).fetchall()
                if ids!=[(doc['filesystem_entry_id'],)]:
                    raise ValueError('mixed original native FileDB source identity changed during evidence search')
                chunk_id=doc.get('discovery_chunk_id')
                if chunk_id is not None:
                    rows=db.execute("SELECT f.relative_path_display,c.content_sha256 "
                       "FROM content_chunks c JOIN filesystem_entries f ON c.filesystem_entry_id=f.id "
                       "WHERE c.chunk_id=?",(chunk_id,)).fetchall()
                    if rows!=[(path,rev[1])]:
                        raise ValueError('mixed original native discovery chunk changed after evidence search')
    check_metadata()
    for path,(doc,rev,receipt) in selected.items():
        # The document-specific selector already validated any publisher
        # acquisition URL. The final fence separately compares the current
        # complete catalog and receipt to that exact pre-selection snapshot,
        # then independently rehashes the publisher original and derivative.
        source=checked_original(path)
        if (source['original_sha256']!=rev[1] or source['derived_sha256']!=rev[2]
                or source['original_bytes']!=receipt['bytes_received']):
            raise ValueError('mixed original publisher full source bytes changed during evidence search')
        proof=final_original_revision_guard(source,original_path=path,
                                            native_file_id=doc['filesystem_entry_id'])
        if not proof['latest_source_map_revision_and_native_file_id_rechecked']:
            raise ValueError('mixed original source final revision proof did not complete')
    check_metadata()
    return {'verified_source_originals_rechecked_after_combined_evidence':len(selected),
            'complete_original_and_derived_publisher_bytes_rehashed':total,
            'final_source_map_native_ID_and_original_revision_checked':True,
            'cross_store_immutable_generation_pinned':False,
            'safe_to_cutover_live':False}

def _opt_in_native_selected_archived_references(args, documents:list[dict], *,
                                                evidence_query:str, versions:tuple):
    """Opt-in publisher links of native-selected, current verified HTML originals.

    Child candidates are separate from the native search rank. Every result is
    read only, original/derived SHA verified, and ultimately candidate-only.
    At most five archived destination originals are returned across all
    selected HTML parent originals; no external browsing or downloading.
    """
    if not getattr(args,'follow_archived_links',False):
        return None
    cursor=getattr(args,'continue_section',None)
    private_stage=getattr(args,'private_staged_pdf_preflight',None)
    budget=getattr(args,'linked_originals',3)
    if (type(budget) is not int or not 1<=budget<=5
            or cursor is not None or private_stage is not None
            or not isinstance(evidence_query,str) or not 1<=len(evidence_query)<=300
            or not isinstance(documents,list) or len(documents)>5
            or not isinstance(versions,tuple) or len(versions)!=3):
        raise ValueError('archived original reference follow requires bounded HTML/Markdown evidence and no incompatible PDF overlay')
    from raspi_archived_html_follow_links import find_archived_linked_source_passages
    from raspi_archived_markdown_follow_links import follow_current_archived_markdown_references
    manifest,ledger,sources=versions
    eligible=[];unique=set()
    for doc in documents:
        kind=doc.get('source_kind')
        path=doc.get('original_path')
        is_html=(kind=='html' and isinstance(path,str)
                 and Path(path).suffix.casefold() in ('.html','.htm'))
        is_markdown=(kind=='text' and isinstance(path,str)
                     and Path(path).suffix.casefold()=='.md')
        if not (is_html or is_markdown):continue
        if doc.get('status') not in ('verified_derived_text','no_verified_excerpt'):
            continue
        expected_kind='html' if is_html else 'text'
        if (path in unique
                or path not in manifest or manifest[path].get('kind')!=expected_kind
                or path not in ledger or path not in sources
                or type(doc.get('filesystem_entry_id')) is not int
                or doc['filesystem_entry_id']<1):
            raise ValueError('native-selected original HTML/Markdown source lacks unique verified identity')
        unique.add(path)
        if (doc.get('original_sha256')!=sources[path][1]
                or doc.get('derived_sha256')!=sources[path][2]
                or ledger[path].get('sha256')!=doc['original_sha256']):
            raise ValueError('native-selected HTML/Markdown source SHA differs from current catalog original')
        eligible.append(doc)
    if not eligible:
        raise ValueError('archived source reference follow needs at least one native-selected verified HTML or Markdown original')
    # Linked article references are not necessarily NEW documents. A child
    # may already be among the native-ranked originals; preserve its link
    # provenance while reporting the overlap instead of inflating discovery.
    native_selected={doc['original_path']:doc for doc in documents
                     if isinstance(doc,dict) and isinstance(doc.get('original_path'),str)}
    used=0;parents=[];revisions={};linked_occurrences=[]
    for doc in eligible:
        if used>=budget:break
        parent_path=doc['original_path']
        markdown=Path(parent_path).suffix.casefold()=='.md'
        if markdown:
            report=follow_current_archived_markdown_references(
                parent_path,evidence_query,limit=budget-used,
                passages=min(2,args.passages),snippet_characters=args.snippet_characters)
            children=report.get('linked_approved_current_original_sources')
            parent_proof=report.get('parent_original_Markdown_literal_byte_link_proof')
            html_proofs=0
            text_proofs=report.get('independently_publisher_original_plaintext_byte_attested_passages')
            missing=parent_proof.get('missing_or_unapproved_official_reference_count') if isinstance(parent_proof,dict) else None
        else:
            report=find_archived_linked_source_passages(parent_path,evidence_query,
                limit=budget-used,passages=min(2,args.passages),
                snippet_characters=args.snippet_characters)
            children=report.get('linked_existing_original_sources')
            parent_proof=report.get('parent_original_HTML_article_link_proof')
            html_proofs=report.get('publisher_original_article_evidence_passages')
            text_proofs=0
            missing=parent_proof.get('same_origin_relevant_unarchived_or_unapproved_references') if isinstance(parent_proof,dict) else None
        if (not isinstance(report,dict) or report.get('parent_original_path')!=parent_path
                or report.get('parent_original_sha256')!=doc['original_sha256']
                or report.get('parent_derived_sha256')!=doc['derived_sha256']
                or report.get('safe_to_cutover_live') is not False
                or type(html_proofs) is not int or type(text_proofs) is not int
                or type(missing) is not int or min(html_proofs,text_proofs,missing)<0):
            raise ValueError('archived link parent source or proof scope differs from current native original')
        if (not isinstance(children,list) or len(children)>budget-used
                or report.get('linked_original_sources_checked')!=len(children)):
            raise ValueError('archived link returned unbounded or inconsistent current originals')
        annotated=[]
        for child in children:
            if (not isinstance(child,dict) or not isinstance(child.get('original_path'),str)
                    or child['original_path']==parent_path
                    or type(child.get('native_filedb_original_id')) is not int
                    or child['native_filedb_original_id']<1):
                raise ValueError('invalid linked native source identity')
            path=child['original_path']
            revision=(child['original_sha256'],child['derived_sha256'],
                      child['native_filedb_original_id'])
            existing=revisions.get(path)
            if existing is not None and existing!=revision:
                raise ValueError('linked source revision changed across native parent references')
            revisions[path]=revision
            already=native_selected.get(path)
            if already is not None and (already.get('original_sha256')!=revision[0]
                 or already.get('derived_sha256')!=revision[1]
                 or already.get('filesystem_entry_id')!=revision[2]
                 or already.get('source_kind') not in ('html','text')):
                raise ValueError('linked HTML/Markdown source identity differs from already native-ranked original')
            item=dict(child)
            item['already_in_native_selected_documents']=already is not None
            item['native_selected_original_rank_if_already_selected']=(
                already.get('rank') if already is not None else None)
            annotated.append(item)
            linked_occurrences.append((path,already is not None))
        parent_revision=(doc['original_sha256'],doc['derived_sha256'],
                         doc['filesystem_entry_id'])
        if parent_path in revisions and revisions[parent_path]!=parent_revision:
            raise ValueError('previously linked parent source revision differs from native-ranked original')
        revisions[parent_path]=parent_revision
        parents.append({'native_parent_rank':doc['rank'],'parent_original_path':parent_path,
                        'parent_source_kind':'original_Markdown' if markdown else 'original_HTML',
                        'parent_status':doc['status'],
                        'parent_direct_excerpt_count':len(doc['passages']),
                        'linked_existing_original_sources':annotated,
                        **({'parent_original_Markdown_literal_byte_link_proof':parent_proof}
                           if markdown else {'parent_original_HTML_article_link_proof':parent_proof}),
                        'linked_source_excerpt_candidates':(
                            report['linked_derived_passage_candidates'] if markdown
                            else report['linked_source_excerpt_candidates']),
                        'publisher_original_article_passage_proofs':html_proofs,
                        'publisher_original_plaintext_byte_passage_proofs':text_proofs,
                        'missing_unapproved_original_references':missing,
                        'linked_sources_are_not_semantic_ranks':True})
        used+=len(children)
    # A previous destination might be rewritten during later parent searches.
    # Reverify the complete current publisher+derived bytes and native ID for
    # ALL returned parents and children as one final bounded group. This is NOT
    # an atomic multi-database generation or an admission lock for old clients.
    for path,(orig_sha,derived_sha,native_id) in revisions.items():
        source=checked_original(path)
        if (source['original_sha256']!=orig_sha or source['derived_sha256']!=derived_sha):
            raise ValueError('linked source parent or child changed during mixed native discovery')
        state=final_original_revision_guard(source,original_path=path,native_file_id=native_id)
        if not state['latest_source_map_revision_and_native_file_id_rechecked']:
            raise ValueError('linked source no longer matches original source-map and native file ID')
    return {
        'parents':parents,
        'native_selected_HTML_parents_checked':sum(x['parent_source_kind']=='original_HTML' for x in parents),
        'native_selected_Markdown_parents_checked':sum(x['parent_source_kind']=='original_Markdown' for x in parents),
        'linked_original_sources_checked':used,
        'already_native_ranked_linked_reference_count':sum(is_native for _path,is_native in linked_occurrences),
        'distinct_new_linked_originals_not_in_native_selection':len({
            path for path,is_native in linked_occurrences if not is_native}),
        'distinct_linked_original_paths':len({path for path,_is_native in linked_occurrences}),
        'original_HTML_article_evidence_passages':sum(x['publisher_original_article_passage_proofs'] for x in parents),
        'original_Markdown_publisher_byte_evidence_passages':sum(x['publisher_original_plaintext_byte_passage_proofs'] for x in parents),
        'missing_unapproved_original_references':sum(x['missing_unapproved_original_references'] for x in parents),
        'all_returned_native_selected_parent_and_child_current_source_revisions_rechecked':True,
        'no_new_external_source_fetch_or_download':True,
        'linked_originals_do_not_replace_existing_native_document_rank':True,
        'not_a_verified_correct_answer':True,
        'cross_store_immutable_generation_pinned':False,
        'safe_to_cutover_live':False,
    }


def _optional_verified_pdf_page_windows(args, documents, *, evidence_query):
    """Existing selected originals only; caller performs final revision fence."""
    budget=getattr(args,'expand_pdf_pages',0)
    if not budget:return None
    from file_system_catalog.verified_pdf_window_bridge import expand_verified_page_evidence
    from raspi_original_pdf_page_neighbors import original_pdf_context
    return expand_verified_page_evidence(documents,evidence_query,page_budget=budget,
        snippet_characters=args.snippet_characters,
        read_context=lambda path,page:original_pdf_context(
            path,page,neighbor_radius=0,include_center_reading_order=True))


def _validate_pdf_page_expansion(args, passage_count):
    budget=getattr(args,'expand_pdf_pages',0)
    if type(budget) is int and budget==0:return
    from file_system_catalog.verified_pdf_window_bridge import validate_expansion_request
    validate_expansion_request(budget,passage_count,args.snippet_characters)
    if budget and (getattr(args,'continue_section',None) is not None
                   or getattr(args,'private_staged_pdf_preflight',None) is not None):
        raise ValueError('verified PDF page expansion cannot be combined with section continuation or private corrected-page overlay')


def _pdf_page_windows_audit(result):
    from file_system_catalog.verified_pdf_window_bridge import expansion_audit_summary
    return expansion_audit_summary(result)


def find_source_evidence(args) -> None:
    """Discover native originals or resolve one explicit named publisher source.

    An unambiguous RFC question or explicitly supplied --source-name takes the
    verified catalog path, NOT a speculative semantic similarity rank.
    """
    engine = "native_discovery_mixed_source_evidence_v1"
    receipt = {"operation": "find_source_evidence", "query": args.query, "retrieval_engine": engine}
    try:
        if (not args.query.strip() or len(args.query) > 600
                or not 1 <= args.documents <= 5
                or not 1 <= args.passages <= 5
                or not 160 <= args.snippet_characters <= 2000
                or type(getattr(args, "scope_context_characters", 0)) is not int
                or (getattr(args, "scope_context_characters", 0) != 0
                    and not 80 <= args.scope_context_characters <= 600)):
            raise ValueError("invalid bounded source evidence query")
        _validate_pdf_page_expansion(args,args.passages)
        source_name = getattr(args, "source_name", None)
        if source_name is not None and (
                not isinstance(source_name, str) or not source_name.strip()
                or len(source_name) > 1000):
            raise ValueError("invalid explicit source name")
        cursor = getattr(args, "continue_section", None)
        if cursor is not None and (
                not isinstance(cursor, str) or not 1 <= len(cursor) <= 128):
            raise ValueError("invalid RFC continuation cursor")
        literal_subject=getattr(args,'table_subject',None)
        literal_column=getattr(args,'table_column',None)
        literal_section=getattr(args,'table_exact_section',None)
        asserted_value=getattr(args,'expect_table_value',None)
        if asserted_value is not None and literal_subject is None:
            raise ValueError('--expect-table-value requires explicit original --inspect-original-tables, --table-subject and --table-column')
        if any(x is not None for x in (literal_subject,literal_column,literal_section)):
            if (not getattr(args,'inspect_original_tables',False)
                    or literal_subject is None or literal_column is None
                    or cursor is not None):
                raise ValueError('exact original table cell requires --inspect-original-tables with both --table-subject and --table-column, without section continuation')
        if getattr(args,'follow_archived_links',False):
            limit=getattr(args,'linked_originals',3)
            if (type(limit) is not int or not 1<=limit<=5
                    or getattr(args,'private_staged_pdf_preflight',None) is not None
                    or getattr(args,'continue_section',None) is not None
                    or len(args.query)>300):
                raise ValueError('archived original reference follow requires bounded HTML/Markdown evidence and no incompatible PDF overlay')
        versions = _current_versions()
        manifest, ledger, _ = versions
        # Do not choose among editions or similarly worded studies. Resolve
        # only complete catalog names, publisher identifiers and strict RFC
        # content questions. A comparison remains an ordinary vector query.
        named = resolve_source_name(source_name or args.query, manifest)
        if source_name is not None and named is None:
            raise ValueError("explicit source name not uniquely recognized in current catalog")
        if named and named["status"] == "ambiguous":
            raise ValueError("ambiguous catalog original; specify one exact source path or edition")
        if named and named["status"] == "no_catalog_match":
            raise ValueError("requested original not found in approved source catalog")
        if cursor is not None:
            if not named or named.get("status") != "unique":
                raise ValueError("RFC section continuation requires one exact named original")
            selected_record = manifest.get(named["original_path"])
            if not selected_record:
                raise ValueError("named original not found in current catalog")
            subject = named.get("subject_query") or args.query
            if selected_record.get("kind") != "text" or section_key(subject) is None:
                raise ValueError("RFC section continuation requires a verified numbered RFC citation")

        selection_basis = "native_semantic_document_ranking"
        evidence_args = args
        with closing(open_native_summary_read_connection(DB)) as db:
            if named is not None:
                path = named["original_path"]
                record = manifest[path]
                receipt_row = ledger.get(path)
                if not receipt_row or not receipt_row.get("sha256"):
                    raise ValueError("requested original is not a completed download")
                rows = db.execute(
                    "SELECT id FROM filesystem_entries "
                    "WHERE relative_path_display=? AND entry_type='file'",
                    (path,),
                ).fetchall()
                if len(rows) != 1:
                    raise ValueError("named original lacks a unique native FileDB identity")
                hits = [{
                    "original_path": path, "source_kind": record["kind"],
                    "original_source_sha256": receipt_row["sha256"],
                    "kmd_native_chunk_id": None,
                    "kmd_original_file_entry_id": rows[0][0],
                    "semantic_similarity": None,
                }]
                selection_basis = (
                    "explicit_named_catalog_original" if source_name is not None
                    else "unambiguous_named_source_question"
                )
                evidence_args = argparse.Namespace(**{
                    **vars(args), "query": named.get("subject_query") or args.query,
                })
            else:
                hits = _native_passage_search(args.query, top=args.documents)["results"]

            documents = [
                (_discovered_pdf(hit, rank, evidence_args, versions, db)
                 if hit["source_kind"] == "pdf"
                 else _discovered_non_pdf(hit, rank, evidence_args, versions, db))
                for rank, hit in enumerate(hits, 1)
            ]
    except Exception as exc:
        append_query_audit({**receipt, "success": False,
                            "error": f"{type(exc).__name__}: {exc}"[:300]})
        raise

    # Corrected publisher-page lookup is deliberately opt-in and a SEPARATE
    # original-PDF-verified evidence channel. Existing live vectors, cached
    # derivatives and native search results remain unchanged until a safe
    # coordinated cutover. Never promote private staged text to native rows.
    corrected_overlay = None
    private_stage = getattr(args, 'private_staged_pdf_preflight', None)
    if private_stage is not None:
        from file_system_catalog.staged_page_native_bridge import (
            pinned_private_corrected_sources, verified_native_selected_private_page_overlay,
        )
        audit_root = Path('/data/var/knowmoredirt/knowledge-library/search-audit')
        try:
            staged = pinned_private_corrected_sources(stage_preflight=private_stage,
                                                       private_root=audit_root)
            completed = {p:r['sha256'] for p,r in ledger.items()
                         if isinstance(r,dict) and isinstance(r.get('sha256'),str)}
            with sqlite3.connect(f'{SOURCE_MAP.resolve().as_uri()}?mode=ro',uri=True,timeout=25) as sm,\
                 sqlite3.connect(f'{DB.resolve().as_uri()}?mode=ro',uri=True,timeout=25) as native:
                sm.execute('PRAGMA query_only=ON');native.execute('PRAGMA query_only=ON')
                corrected_overlay = verified_native_selected_private_page_overlay(
                   documents=documents, query=evidence_args.query,
                   staged_records=staged,stage_root=audit_root,
                   original_root=LIBRARY,derived_root=CACHE/'text',
                   completed_receipts=completed,source_map=sm,native=native,
                   top_per_original=min(2,args.passages),candidates_per_original=20)
        except Exception as exc:
            append_query_audit({**receipt,'success':False,
                'operation_stage':'opt_in_original_pdf_corrected_page_overlay',
                'error':f'{type(exc).__name__}: {exc}'[:250]})
            raise

    # Optional publisher-original table-row evidence is a separate bounded
    # source structure channel. A wide Markdown table can contain an exact
    # answer column that a normal source excerpt crops. Do NOT promote table
    # rows to documents, replace the original ranked passages, or infer truth.
    selected_original_tables=None
    selected_unique_original_cell=None
    selected_literal_claim_check=None
    try:
        if getattr(args,'inspect_original_tables',False):
            inspected=[]
            cell_candidates=[]
            for document in documents:
                original_path=document['original_path']
                # The approved archive classifies third-party Markdown
                # agent manuals as 'skill', not 'text'. Both are exact
                # publisher UTF-8 Markdown originals and must be eligible
                # for the same original-SHA and native-ID table proof. PDFs,
                # HTML, and unrecognized source kinds remain excluded.
                if (document.get('source_kind') not in ('text','skill')
                        or Path(original_path).suffix.casefold() not in ('.md','.markdown')):
                    continue
                entry=checked_original(original_path,include_publisher_text=True)
                if (entry['original_sha256']!=document.get('original_sha256')
                        or entry['derived_sha256']!=document.get('derived_sha256')
                        or not isinstance(entry.get('publisher_text_bytes'),bytes)):
                    raise ValueError('native selected Markdown table source differs from the SHA-verified selected original')
                table=verified_original_markdown_table_rows(
                    query=evidence_args.query,original_path=original_path,
                    original_sha256=entry['original_sha256'],
                    derived_sha256=entry['derived_sha256'],
                    derived_text=entry['derived_text'],
                    publisher_text_bytes=entry['publisher_text_bytes'],
                    filesystem_entry_id=document['filesystem_entry_id'],top=2)
                inspected.append({'native_original_rank_not_changed':document['rank'],
                                  'original_path':original_path,
                                  'original_firstparty_markdown_table_evidence':table})
                if literal_subject is not None:
                    exact=verified_original_markdown_table_cell(
                        subject=literal_subject,column=literal_column,
                        exact_section=literal_section,original_path=original_path,
                        original_sha256=entry['original_sha256'],derived_sha256=entry['derived_sha256'],
                        derived_text=entry['derived_text'],
                        publisher_text_bytes=entry['publisher_text_bytes'],
                        filesystem_entry_id=document['filesystem_entry_id'])
                    cell_candidates.append((document['rank'],original_path,exact))
            if literal_subject is not None:
                ambiguous=any(c['status']=='ambiguous' for _,_,c in cell_candidates)
                resolved=[(rank,path,c) for rank,path,c in cell_candidates if c['status']=='resolved']
                selected_status=('ambiguous' if ambiguous or len(resolved)>1 else
                                 ('resolved' if len(resolved)==1 else 'not_found_in_selected_native_originals'))
                if selected_status=='resolved':
                    rank,path,c=resolved[0]
                    chosen={'original_native_rank_prior_to_cell_lookup':rank,
                            'original_path':path,
                            'original_verified_unique_table_cell':c['resolved_unique_literal_publisher_cell']}
                else:chosen=None
                selected_unique_original_cell={
                    'schema':'kmd.optin_native_selected_unique_publisher_markdown_cell.v1',
                    'status':selected_status,
                    'native_originals_searched_prior_to_cell_lookup':len(documents),
                    'original_markdown_sources_checked':len(cell_candidates),
                    'matching_original_table_rows_across_selected_documents':sum(c['matching_original_table_rows'] for _,_,c in cell_candidates),
                    'selected_original_source_cell':chosen,
                    'never_choose_a_native_rank_to_silently_resolve_cell_ambiguity':True,
                    'not_found_is_only_within_selected_native_originals_not_entire_archive':True,
                    'existing_native_document_rank_and_excerpts_unchanged':True,
                    'no_automatic_answer_entailment_or_factual_correctness_claimed':True}
                if asserted_value is not None:
                    selected_literal_claim_check=compare_claimed_literal_to_current_original_cell(
                        original_cell_result=selected_unique_original_cell,
                        claimed_literal_value=asserted_value)
            selected_original_tables={
                'schema':'kmd.optin_native_ranked_original_markdown_tables.v1',
                'original_source_documents_selected_before_table_lookup':len(documents),
                'original_markdown_documents_inspected':len(inspected),
                'original_publisher_table_row_candidates':sum(x['original_firstparty_markdown_table_evidence']['returned_original_publisher_table_row_candidates'] for x in inspected),
                'native_ranked_documents_unchanged':True,
                'publisher_table_rows_separate_from_source_passage_output':True,
                'no_additional_original_document_discovery_or_search_rerank':True,
                'not_automatic_factual_entailment_or_answer_truth':True,
                'originals':inspected}
    except Exception as exc:
        append_query_audit({**receipt,'success':False,
            'operation_stage':'opt_in_current_original_markdown_table_byte_proof',
            'error':f'{type(exc).__name__}: {exc}'[:280]})
        raise
    # Document order is the native ranking for generic queries; for explicit
    # name selection rank 1 means one catalog choice, not a measured similarity.
    for doc in documents:
        doc["selection_basis"] = selection_basis
        if doc.get("status") == "verified" and "source_kind" not in doc:
            doc["source_kind"] = "pdf"
    pdf_count = sum(len(doc["passages"]) for doc in documents if doc["status"] == "verified")
    derived_count = sum(len(doc["passages"]) for doc in documents
                        if doc["status"] == "verified_derived_text")
    publisher_byte_count = sum(
        bool(passage.get("publisher_excerpt"))
        for doc in documents if doc["status"] == "verified_derived_text"
        for passage in doc["passages"]
    )
    evidence_kind_counts = {
        "physical_pdf_pages": pdf_count,
        "publisher_utf8_byte_intervals": publisher_byte_count,
        "extracted_text_only": derived_count - publisher_byte_count,
    }
    try:
        linked_archive=_opt_in_native_selected_archived_references(
            args,documents,evidence_query=evidence_args.query,versions=versions)
        pdf_windows=_optional_verified_pdf_page_windows(args,documents,evidence_query=evidence_args.query)
        final_source_proof=final_mixed_document_source_revision_guard(documents,versions=versions)
    except Exception as exc:
        append_query_audit({**receipt,'success':False,
            'operation_stage':'final_mixed_current_original_revision_check',
            'error':f'{type(exc).__name__}: {exc}'[:290]})
        raise
    # Optional structural diagnostics never alter native rank, excerpt,
    # publisher SHA, page verification, or answer authority.
    utility_review=(review_source_documents(evidence_args.query,documents)
                    if getattr(args,'assess_passage_utility',False) else None)
    if utility_review is not None and linked_archive is not None:
        linked_children=[child for parent in linked_archive['parents']
                         for child in parent['linked_existing_original_sources']]
        utility_review['optional_linked_original_structural_review']=review_verified_linked_originals(
            evidence_args.query,linked_children)
    # Both evidence and extra source-scope text are untrusted original data:
    # keep coordinate/checksum audit receipts, never actual publisher prose.
    audited = [
        {**doc, "passages": [
            _audit_passage_without_text(passage) for passage in doc["passages"]
        ]} for doc in documents
    ]
    audit_file = append_query_audit({
        **receipt, "success": True, "selection_basis": selection_basis,
        "corrected_private_source_overlay_enabled": corrected_overlay is not None,
        "private_overlay_verified_pdf_pages":
            (corrected_overlay['publisher_verified_corrected_passages']
             if corrected_overlay is not None else 0),
        "evidence_query": evidence_args.query,
        **({'opt_in_original_markdown_table_evidence_counts':{
            'inspected_native_originals':selected_original_tables['original_markdown_documents_inspected'],
            'original_table_row_candidates':selected_original_tables['original_publisher_table_row_candidates'],
            'not_verified_answer':True}} if selected_original_tables is not None else {}),
        **({'opt_in_exact_original_table_cell_counts':{
            'status':selected_unique_original_cell['status'],
            'selected_originals_checked':selected_unique_original_cell['original_markdown_sources_checked'],
            'matching_literal_rows':selected_unique_original_cell['matching_original_table_rows_across_selected_documents'],
            'literal_cell_value_excluded_from_private_audit':True,
            'not_semantic_answer_verification':True}}
            if selected_unique_original_cell is not None else {}),
        **({'opt_in_expected_original_cell_literal_status':{
            'status':selected_literal_claim_check['status'],
            'asserted_and_source_field_text_excluded_from_private_audit':True,
            'literal_difference_is_not_semantic_contradiction':True}}
            if selected_literal_claim_check is not None else {}),
        **({"opt_in_archived_original_references": {
            "selected_native_parent_originals": [p['parent_original_path'] for p in linked_archive['parents']],
            "linked_original_sources_checked":linked_archive['linked_original_sources_checked'],
            "already_native_ranked_linked_references":linked_archive['already_native_ranked_linked_reference_count'],
            "distinct_new_linked_originals":linked_archive['distinct_new_linked_originals_not_in_native_selection'],
            "original_HTML_article_evidence_passages":linked_archive['original_HTML_article_evidence_passages'],
            "original_Markdown_publisher_byte_evidence_passages":linked_archive['original_Markdown_publisher_byte_evidence_passages'],
            "missing_unapproved_original_references":linked_archive['missing_unapproved_original_references'],
            "linked_original_paths": [x['original_path'] for p in linked_archive['parents']
                                      for x in p['linked_existing_original_sources']],
            "not_a_verified_answer":True,
        }} if linked_archive is not None else {}),
        **({'optional_source_passage_utility_counts':{
            'reviewed_candidates':utility_review['reviewed_publisher_excerpt_candidates'],
            'navigation_or_prompt_candidates':utility_review['navigation_or_prompt_template_candidates'],
            'bibliography_context_candidates':utility_review.get('bibliography_context_candidates',0),
            'linked_excerpt_candidates':(utility_review['optional_linked_original_structural_review']['reviewed_linked_excerpt_candidates']
                if 'optional_linked_original_structural_review' in utility_review else 0),
            'linked_navigation_or_prompt_candidates':(utility_review['optional_linked_original_structural_review']['navigation_or_prompt_linked_excerpt_candidates']
                if 'optional_linked_original_structural_review' in utility_review else 0),
            'not_answer_verification':True}} if utility_review is not None else {}),
        **({'optional_verified_pdf_page_window_audit':_pdf_page_windows_audit(pdf_windows)}
           if pdf_windows is not None else {}),
        "verified_pdf_excerpts": pdf_count,
        "verified_non_pdf_derived_excerpts": derived_count,
        "evidence_kind_counts": evidence_kind_counts, "documents": audited,
        "final_current_original_source_revision_check":final_source_proof,
    })
    print(json.dumps({
        "query": args.query, "evidence_query": evidence_args.query,
        "selection_basis": selection_basis, "retrieval_engine": engine,
        "documents": documents,
        **({'optional_original_markdown_table_evidence':selected_original_tables}
           if selected_original_tables is not None else {}),
        **({'optional_exact_original_markdown_table_cell':selected_unique_original_cell}
           if selected_unique_original_cell is not None else {}),
        **({'optional_expected_original_cell_literal_comparison':selected_literal_claim_check}
           if selected_literal_claim_check is not None else {}),
        **({'opt_in_archived_original_reference_navigation':linked_archive}
           if linked_archive is not None else {}),
        "final_current_original_source_revision_check":final_source_proof,
        **({'optional_source_passage_utility_review':utility_review}
           if utility_review is not None else {}),
        **({'corrected_private_publisher_page_overlay': corrected_overlay}
           if corrected_overlay is not None else {}),
        "verified_pdf_excerpts": pdf_count,
        "verified_non_pdf_derived_excerpts": derived_count,
        "evidence_kind_counts": evidence_kind_counts,
        **({'optional_verified_pdf_page_windows':pdf_windows}
           if pdf_windows is not None else {}),
        "query_audit_logged": True, "private_query_audit_file": str(audit_file),
        "evidence_scope": (
            "PDF excerpts require physical publisher-page verification. For HTML, Office, ZIP "
            "and other source types, excerpt evidence is only verified against the "
            "revision-linked extracted text unless a publisher-byte interval is explicit. "
            "An explicitly named catalog original has no semantic score. "
            "No excerpt alone proves a question's answer or real-world factual scope."
        ),
    }, indent=2, ensure_ascii=False))


def find_pdf_evidence(args) -> None:
    """Discover original documents and return only excerpts reproduced on physical PDF pages."""
    engine = "native_discovery_verified_pdf_excerpts_v1"
    receipt = {"operation": "find_pdf_evidence", "query": args.query, "retrieval_engine": engine}
    try:
        if (not args.query.strip() or len(args.query) > 600 or not 1 <= args.documents <= 5
                or not 1 <= args.passages <= 5 or not 160 <= args.snippet_characters <= 2000):
            raise ValueError("invalid PDF evidence query or bounds")
        versions = _current_versions()
        discovery = _native_passage_search(args.query, top=args.documents)
        with closing(open_native_summary_read_connection(DB)) as db:
            documents = [_discovered_pdf(hit, rank, args, versions, db)
                         for rank, hit in enumerate(discovery["results"], 1)]
    except Exception as exc:
        append_query_audit({**receipt, "success": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
        raise

    try:
        final_source_proof=final_mixed_document_source_revision_guard(documents,versions=versions)
    except Exception as exc:
        append_query_audit({**receipt,'success':False,
            'operation_stage':'final_pdf_current_original_revision_check',
            'error':f'{type(exc).__name__}: {exc}'[:290]})
        raise
    count = sum(len(doc["passages"]) for doc in documents)
    audited = [{**doc, "passages": [{k: v for k, v in hit.items() if k != "source_excerpt"}
                                     for hit in doc["passages"]]} for doc in documents]
    audit_file = append_query_audit({**receipt, "success": True, "verified_excerpts": count,
                                     "documents": audited,
                                     "final_current_original_source_revision_check":final_source_proof})
    print(json.dumps({
        "query": args.query, "retrieval_engine": engine, "documents": documents, "verified_excerpts": count,
        "final_current_original_source_revision_check":final_source_proof,
        "query_audit_logged": True, "private_query_audit_file": str(audit_file),
        "evidence_scope": "Documents retain their native semantic rank. Check whether the quoted text answers the question.",
    }, indent=2, ensure_ascii=False))


def _optional_verified_archived_links(args, *, original_path:str, evidence_query:str,
                                     continuation_cursor:str|None):
    """Explicit, optional archived first-party reference-following ONLY.

    Never invoked by the default source evidence or semantic search path.
    The separate linked-reader rechecks current parent/destination source and
    native identities after building its candidates. No external downloads.
    """
    exact_name=getattr(args,'exact_link_text',None)
    if not getattr(args, 'follow_archived_links', False):
        if exact_name is not None:
            raise ValueError('--exact-link-text requires explicit --follow-archived-links')
        return None
    limit=getattr(args,'linked_originals',3)
    suffix=Path(original_path).suffix.casefold()
    if (type(limit) is not int or not 1<=limit<=5
            or continuation_cursor is not None
            or suffix not in ('.html','.htm','.md')
            or not isinstance(evidence_query,str) or not 1<=len(evidence_query)<=300):
        raise ValueError('archived source following requires one HTML/Markdown original, no continuation, and 1-5 linked originals')
    # Import only on explicit invocation. Plain Markdown has literal publisher
    # original byte intervals, unlike HTML's independently normalized DOM.
    if suffix=='.md':
        from raspi_archived_markdown_follow_links import follow_current_archived_markdown_references
        return follow_current_archived_markdown_references(
            original_path,evidence_query,limit=limit,
            passages=min(2,args.top),snippet_characters=args.snippet_characters,
            exact_link_text=exact_name)
    if exact_name is not None:
        raise ValueError('exact first-party link text navigation is only supported for original Markdown')
    from raspi_archived_html_follow_links import find_archived_linked_source_passages
    return find_archived_linked_source_passages(
        original_path,evidence_query,limit=limit,
        passages=min(2,args.top),snippet_characters=args.snippet_characters)


def original_source_evidence(args) -> None:
    """Inspect verified original pages; never mistake lexical hits for truth."""
    started=time.monotonic()
    original_path = getattr(args, "original_path", None)
    evidence_query = args.query
    match = None
    publication_record = None
    engine = "verified_original_ephemeral_fts5_v1"
    cursor = getattr(args, "continue_section", None)
    try:
        if not 1 <= args.top <= 50:
            raise ValueError("invalid bounded requested source passage count")
        _validate_pdf_page_expansion(args,args.top)
        if getattr(args,'assess_passage_utility',False) and (args.top>5 or cursor is not None):
            raise ValueError('optional structural source review requires at most five original passages and no RFC continuation')
        if getattr(args,'inspect_original_tables',False) and (
                cursor is not None or args.top>5):
            raise ValueError('optional original table evidence requires at most five original passages and no RFC continuation')
        literal_subject=getattr(args,'table_subject',None)
        literal_column=getattr(args,'table_column',None)
        literal_section=getattr(args,'table_exact_section',None)
        asserted_value=getattr(args,'expect_table_value',None)
        if asserted_value is not None and literal_subject is None:
            raise ValueError('--expect-table-value requires explicit original --inspect-original-tables, --table-subject and --table-column')
        if any(x is not None for x in (literal_subject,literal_column,literal_section)):
            if (not getattr(args,'inspect_original_tables',False)
                    or literal_subject is None or literal_column is None):
                raise ValueError('exact publisher table cell requires --inspect-original-tables with both --table-subject and --table-column')
        if original_path is None:
            explicit_name = getattr(args, "source_name", None)
            manifest = _unique_path_records(json.loads(SOURCES.read_text())["sources"])
            match = resolve_source_name(explicit_name or args.query, manifest)
            if not match:
                raise ValueError("source name unrecognized; supply an exact --source-name or --original-path")
            if match["status"] == "no_catalog_match":
                if match["kind"] == "catalog_rfc":
                    raise ValueError("no archived RFC Editor text matches this RFC number")
                raise ValueError("no cataloged PDF matches this publication identifier and requested version")
            if match["status"] != "unique":
                choices = "\n".join(match["candidate_paths"])
                raise ValueError(f"source name matches {match['matching_catalog_paths']} catalog paths; "
                                 f"select --original-path. Available choices include\n{choices}")
            original_path = match["original_path"]
            if "matched_identifier" in match:
                publication_record = manifest[original_path]
            if explicit_name and match.get("kind") == "catalog_rfc" and match.get("subject_query"):
                if section_key(match["subject_query"]) != section_key(args.query):
                    raise ValueError("a section citation is a complete evidence query; supply it as the positional query")
            if explicit_name is None:
                evidence_query = match.get("subject_query", "")
                if not evidence_query:
                    raise ValueError("supply the topic to find after --source-name or --original-path")
        source=checked_original(original_path, include_publisher_text=True, publication_record=publication_record)
        with closing(open_native_summary_read_connection(DB)) as db:
            row=db.execute(
                "SELECT id FROM filesystem_entries WHERE entry_type='file' "
                "AND relative_path_display=?",(original_path,)
            ).fetchone()
        if row is None:
            raise ValueError("original has no canonical KnowMoreDiRT FileDB entry")
        rfc = source.get("verified_rfc_number")
        selector = section_key(evidence_query) if rfc else None
        if cursor is not None and not selector:
            raise ValueError("--continue-section requires an exact RFC section citation")
        if selector:
            engine = "verified_rfc_section_context_v1" if cursor is not None else "verified_rfc_section_opening_v1"
            result = rfc_section_evidence(source, int(row[0]), rfc, selector, args.snippet_characters, cursor=cursor)
        else:
            result = _original_passages({**source, "original_path": original_path}, int(row[0]), evidence_query,
                                        args.top, args.snippet_characters)
        if getattr(args,'inspect_original_tables',False) and Path(original_path).suffix.casefold() in ('.html','.htm'):
            if (literal_subject is None or literal_column is None or literal_section is not None
                    or not isinstance(source.get('publisher_html_bytes'),bytes)):
                raise ValueError('original HTML table inspection requires exact subject and column, bounded UTF-8 HTML, and no Markdown section selector')
            from file_system_catalog.source_html_table_cells import (
                verified_original_html_table_cell,compare_claimed_literal_to_original_html_cell)
            result['optional_exact_original_html_table_cell']=verified_original_html_table_cell(
                subject=literal_subject,column=literal_column,original_path=original_path,
                original_sha256=source['original_sha256'],derived_sha256=source['derived_sha256'],
                derived_text=source['derived_text'],publisher_html_bytes=source['publisher_html_bytes'],
                filesystem_entry_id=int(row[0]))
            if asserted_value is not None:
                result['optional_expected_original_cell_literal_comparison']=compare_claimed_literal_to_original_html_cell(
                    original_cell_result=result['optional_exact_original_html_table_cell'],
                    claimed_literal_value=asserted_value)
        elif getattr(args,'inspect_original_tables',False):
            if (Path(original_path).suffix.casefold() not in ('.md','.markdown')
                    or not isinstance(source.get('publisher_text_bytes'),bytes)):
                raise ValueError('original publisher table inspection requires one verified UTF-8 Markdown original')
            result['optional_original_markdown_table_evidence']=verified_original_markdown_table_rows(
                query=evidence_query,original_path=original_path,
                original_sha256=source['original_sha256'],
                derived_sha256=source['derived_sha256'],derived_text=source['derived_text'],
                publisher_text_bytes=source['publisher_text_bytes'],
                filesystem_entry_id=int(row[0]),top=min(3,args.top))
            if literal_subject is not None:
                result['optional_exact_original_markdown_table_cell']=verified_original_markdown_table_cell(
                    subject=literal_subject,column=literal_column,
                    exact_section=literal_section,original_path=original_path,
                    original_sha256=source['original_sha256'],derived_sha256=source['derived_sha256'],
                    derived_text=source['derived_text'],
                    publisher_text_bytes=source['publisher_text_bytes'],
                    filesystem_entry_id=int(row[0]))
                if asserted_value is not None:
                    result['optional_expected_original_cell_literal_comparison']=compare_claimed_literal_to_current_original_cell(
                        original_cell_result=result['optional_exact_original_markdown_table_cell'],
                        claimed_literal_value=asserted_value)
        if Path(original_path).suffix.casefold() == ".pdf":
            hits = result["passages"]
            proofs = verify_pdf_page_excerpts(publisher_path(original_path), source["original_sha256"], hits)
            for hit, proof in zip(hits, proofs, strict=True):
                hit["publisher_pdf_page"] = proof
            verified = sum(proof["status"] == "verified" for proof in proofs)
            result["pdf_page_verification"] = {
                "verified": verified, "unverified": len(hits) - verified,
                "coordinates": "zero_based_half_open_characters_in_whitespace_normalized_pdftotext_layout_page",
            }
            result["evidence_warning"] += (
                " The source section is the derivative label. Cite the physical PDF page only "
                "when publisher_pdf_page has verified status."
            )
        if getattr(args,'expand_pdf_pages',0):
            if Path(original_path).suffix.casefold()!='.pdf':
                raise ValueError('verified PDF page expansion requires an original PDF')
            result['optional_verified_pdf_page_windows']=_optional_verified_pdf_page_windows(
                args,[{'rank':1,'status':'verified','original_path':original_path,
                       'original_sha256':source['original_sha256'],
                       'derived_sha256':source['derived_sha256'],
                       'filesystem_entry_id':int(row[0]),'passages':result['passages']}],
                evidence_query=evidence_query)
        # The initial source SHA check precedes all retrieval and potentially
        # expensive independent original-page checks. A worker might publish
        # a newer source-map/native revision in the meantime: never return a
        # successfully verified *old* passage without this second check.
        result["final_current_source_revision_check"] = final_original_revision_guard(
            source, original_path=original_path, native_file_id=int(row[0]))
        if getattr(args,'assess_passage_utility',False):
            result['optional_source_passage_utility_review']=review_source_documents(
                evidence_query,[{'original_path':original_path,'rank':1,
                                 'passages':result['passages']}])
        linked=_optional_verified_archived_links(args,original_path=original_path,
                         evidence_query=evidence_query,continuation_cursor=cursor)
        if linked is not None:
            # This is a supplementary, current ORIGINAL-publisher reference
            # channel. The original's own zero lexical excerpts remain zero:
            # never pretend a linked document's excerpt came from the parent.
            result['linked_archived_original_references']=linked
            if 'optional_source_passage_utility_review' in result:
                children=linked.get('linked_existing_original_sources',
                                    linked.get('linked_approved_current_original_sources',[]))
                result['optional_source_passage_utility_review']['optional_linked_original_structural_review']=review_verified_linked_originals(
                    evidence_query,children)
    except Exception as error:
        append_query_audit({
            "operation":"verified_original_source_evidence",
            "query":args.query,
            **({"continuation_requested": True} if cursor is not None else {}),
            "original_path":original_path,
            **({"source_name_match":match} if match else {}),
            "success":False,
            "error":f"{type(error).__name__}: {str(error)[:300]}",
            "duration_seconds":round(time.monotonic()-started,3),
        })
        raise
    selection = {}
    if match:
        selection = {"source_name_match": {**match, "status": "verified"}, "evidence_query": evidence_query}
        result.update(selection)
    receipt={
        **selection,
        "operation":"verified_original_source_evidence",
        "query":args.query,
        "original_path":original_path,
        "original_sha256":result["original_sha256"],
        "derived_sha256":result["derived_sha256"],
        "final_current_source_revision_check":result["final_current_source_revision_check"],
        "success":True,
        "count":result["returned_passages"],
        "retrieval_engine":engine,
        **({"continuation_cursor": cursor} if cursor is not None else {}),
        **({"rfc_section_lookup":result["rfc_section_lookup"]} if "rfc_section_lookup" in result else {}),
        "duration_seconds":round(time.monotonic()-started,3),
        **({"pdf_page_verification": result["pdf_page_verification"]} if "pdf_page_verification" in result else {}),
        **({'optional_original_table_evidence_counts':{
            'publisher_table_row_candidates':result['optional_original_markdown_table_evidence']['returned_original_publisher_table_row_candidates'],
            'publisher_table_not_answer_verification':True}}
            if 'optional_original_markdown_table_evidence' in result else {}),
        **({'optional_exact_original_table_cell_diagnostics':{
            'status':result['optional_exact_original_markdown_table_cell']['status'],
            'matching_original_table_rows':result['optional_exact_original_markdown_table_cell']['matching_original_table_rows'],
            'cell_value_and_source_body_not_in_private_audit':True,
            'no_semantic_entailment_proven':True}}
            if 'optional_exact_original_markdown_table_cell' in result else {}),
        **({'optional_exact_original_html_table_cell_diagnostics':{
            'status':result['optional_exact_original_html_table_cell']['status'],
            'matching_original_table_rows':result['optional_exact_original_html_table_cell']['matching_original_table_rows'],
            'cell_value_and_markup_not_in_private_audit':True,
            'no_semantic_entailment_proven':True}}
            if 'optional_exact_original_html_table_cell' in result else {}),
        **({'optional_expected_original_cell_literal_status':{
            'status':result['optional_expected_original_cell_literal_comparison']['status'],
            'asserted_and_source_field_text_excluded_from_private_audit':True,
            'literal_difference_is_not_semantic_contradiction':True}}
            if 'optional_expected_original_cell_literal_comparison' in result else {}),
        **({'optional_source_passage_utility_counts':{
            'reviewed_candidates':result['optional_source_passage_utility_review']['reviewed_publisher_excerpt_candidates'],
            'navigation_or_prompt_candidates':result['optional_source_passage_utility_review']['navigation_or_prompt_template_candidates'],
            'bibliography_context_candidates':result['optional_source_passage_utility_review'].get('bibliography_context_candidates',0),
            'linked_excerpt_candidates':(
                result['optional_source_passage_utility_review']['optional_linked_original_structural_review']['reviewed_linked_excerpt_candidates']
                if 'optional_linked_original_structural_review' in result['optional_source_passage_utility_review'] else 0),
            'linked_navigation_or_prompt_candidates':(
                result['optional_source_passage_utility_review']['optional_linked_original_structural_review']['navigation_or_prompt_linked_excerpt_candidates']
                if 'optional_linked_original_structural_review' in result['optional_source_passage_utility_review'] else 0),
            'not_answer_verification':True}}
            if 'optional_source_passage_utility_review' in result else {}),
        **({"opt_in_archived_reference_follow": {
            "linked_originals_checked": linked['linked_original_sources_checked'],
            "verified_publisher_HTML_article_candidate_passages":linked.get('publisher_original_article_evidence_passages',0),
            "independently_publisher_original_plaintext_byte_attested_passages":
                linked.get('independently_publisher_original_plaintext_byte_attested_passages',0),
            "missing_unapproved_original_reference_count":(
                linked['parent_original_HTML_article_link_proof']['same_origin_relevant_unarchived_or_unapproved_references']
                if 'parent_original_HTML_article_link_proof' in linked else
                linked['parent_original_Markdown_literal_byte_link_proof']['missing_or_unapproved_official_reference_count']),
            "candidate_original_paths": [x['original_path'] for x in
                linked.get('linked_existing_original_sources',
                           linked.get('linked_approved_current_original_sources',[]))],
            "not_a_verified_answer": True,
        }} if linked is not None else {}),
        **({'optional_verified_pdf_page_window_audit':_pdf_page_windows_audit(result['optional_verified_pdf_page_windows'])}
           if 'optional_verified_pdf_page_windows' in result else {}),
        "ranked_source_spans":[{
            "rank":i,
            "source_section":v["source_section"],
            "source_start_char":v["source_start_char"],
            "source_end_char":v["source_end_char"],
            "source_span_sha256":v["source_span_sha256"],
            "excerpt_start_char":v["excerpt_start_char"],
            "excerpt_end_char":v["excerpt_end_char"],
            "excerpt_sha256":v["excerpt_sha256"],
            **{key: v[key] for key in ("lexical_bm25", "matched_source_terms") if key in v},
            **({"matched_word_forms": v["matched_word_forms"]} if "matched_word_forms" in v else {}),
            **({"matched_identifier_words": v["matched_identifier_words"]} if "matched_identifier_words" in v else {}),
            **({"publisher_excerpt": v["publisher_excerpt"]} if "publisher_excerpt" in v else {}),
            **({"publisher_pdf_page": v["publisher_pdf_page"]} if "publisher_pdf_page" in v else {}),
        } for i,v in enumerate(result["passages"],1)],
        "evidence_warning":result["evidence_warning"],
    }
    filename=append_query_audit(receipt)
    result["query"]=args.query
    result["retrieval_engine"]=engine
    result["query_audit_logged"]=True
    result["private_query_audit_file"]=str(filename)
    print(json.dumps(result,indent=2,ensure_ascii=False))


def status(_args):
    # Status is requested while two legitimate native summary writers run.
    # Keep both source stores read-only, close after each read, and reuse the
    # same bounded reader admission as search. The earlier bare SQLite
    # connection could fail after the default five-second native lock wait,
    # falsely making current indexing coverage unavailable.
    with closing(open_source_map_read_connection(SOURCE_MAP)) as source:
        current = {
            row[0]: (row[1], row[2], int(row[3]))
            for row in source.execute(
                "SELECT source_path,source_sha256,derived_sha256,text_characters "
                "FROM originals WHERE status='complete' AND text_characters>100"
            )
        }
    with closing(open_native_summary_read_connection(DB)) as conn:
        result = summary_index_status(conn)
        valid = current_completed_originals(conn, current)
    result.update(summary_progress_metrics(current, valid))
    # Retain physical rows, including any stale historic source revisions.
    result["all_historical_originals_with_summary_vectors"] = result["original_documents_with_summaries"]
    result["original_documents_with_summaries"] = len(valid)
    result["originals_without_recursive_summaries"] = result["remaining_originals"]
    result["complete_extracted_originals"] = len(current)
    result["canonical_database"] = str(DB)
    result["database_bytes"] = DB.stat().st_size
    result["configured_database_budget_bytes"] = DB_BUDGET
    result["native_filedb_storage_admission"] = storage_budget_snapshot(
        DB, budget_bytes=DB_BUDGET,
    )
    # Active document progress is separate from durable complete-root counts:
    # an unfinished subtree must never be presented as indexed evidence.
    worker_state = DB.parent / "hierarchical-worker-status.json"
    try:
        state = json.loads(worker_state.read_text(encoding="utf-8"))
        active = state.get("current_original")
        if isinstance(active, str) and active in current:
            result["active_original"] = active
            result["worker_phase"] = state.get("phase")
            original_sha, derivative_sha, _ = current[active]
            target = source_progress_path(active, original_sha, derivative_sha)
            if target.is_file() and not target.is_symlink() and target.stat().st_size < 65536:
                receipt = json.loads(target.read_text(encoding="utf-8"))
                if (receipt.get("original_path") == active
                        and receipt.get("original_sha256") == original_sha
                        and receipt.get("derived_sha256") == derivative_sha
                        and str(receipt.get("updated_utc", "")) >= str(
                            state.get("current_started_utc", "")
                        )):
                    result["active_source_progress"] = receipt
    except (OSError, ValueError, TypeError, KeyError):
        result["active_source_progress_unavailable"] = True
    # The original FileDB is the authoritative total across both workers.
    # Expose their separate queues without pretending either's status file
    # has sole ownership of the library.
    worker_snapshots = {}
    for worker_name,filename in (
        ("coder","hierarchical-worker-status.json"),
        ("qwen38_q4","hierarchical-worker-q4-status.json"),
    ):
        path = DB.parent/filename
        try:
            w=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,ValueError,TypeError):
            continue
        if not isinstance(w,dict):
            continue
        worker_snapshots[worker_name]={
            key:w.get(key) for key in (
                "phase","model","shard_count","assigned_shards",
                "assigned_eligible_originals","assigned_complete_originals",
                "assigned_pending_originals","current_original",
                "current_started_utc","last_completed_original",
                "last_completed_utc","updated_utc",
            )
        }
        recorded = w.get("failed") if isinstance(w.get("failed"), dict) else {}
        revisions = ((p, *current[p]) for p in recorded if p in current)
        worker_snapshots[worker_name]["recorded_failure_count"] = len(recorded)
        worker_snapshots[worker_name]["pending_failure_count"] = len(pending_failures(recorded, revisions, valid))
    if worker_snapshots:
        result["parallel_workers"]=worker_snapshots
    print(json.dumps(result, indent=2))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Generate every summary level for a complete original")
    b.add_argument("--original-path", required=True)
    b.add_argument("--model-url", default="http://10.8.0.7:14829")
    b.add_argument("--model", default="qwen38-q4-default")
    b.add_argument("--model-timeout", type=float, default=150)
    b.add_argument("--context-cap-tokens", type=int, default=None,
                   help="Quality source-group cap; actual llama.cpp context is still discovered")
    b.add_argument("--shared-slot-idle-seconds",type=float,default=0.0,
                   help="Wait for a continuously idle llama.cpp slot before each request")
    b.add_argument("--output-tokens", type=int, default=420)
    b.add_argument("--safety-tokens", type=int, default=384)
    b.add_argument("--max-levels", type=int, default=12)
    b.add_argument("--max-nodes", type=int, default=50000)
    b.add_argument("--max-calls", type=int, default=80)
    b.add_argument("--plan-only", action="store_true")
    b.add_argument("--native-facets",action="store_true",
                   help="Use existing KMD file facet schema for summary+keywords in each model call")
    b.add_argument("--compact-native-facets",action="store_true",
                   help="Use concise source-faithful KMD facet instructions for faster structured output")
    b.add_argument("--recover-embedding-context", action="store_true",
                   help="Rebuild an oversized saved native tree beside its hash-verified predecessor")
    i = sub.add_parser("import", help="Prepare or append every summary vector to the original FileDB")
    i.add_argument("--original-path", required=True)
    i.add_argument("--write", action="store_true")
    s = sub.add_parser("search", help="Find original documents by any summary level")
    s.add_argument("query")
    s.add_argument("--top", type=int, default=8)
    s.add_argument("--with-facets",action="store_true",
                   help="Include existing native KMD keyword/topic/keyphrase embeddings in candidate ranking")
    s.add_argument("--nodes-per-original", type=int, default=5)
    sub.add_parser("status")
    evidence=sub.add_parser(
        "source-evidence",
        help="Read verified source pages/sections of an explicitly chosen original",
    )
    target = evidence.add_mutually_exclusive_group()
    target.add_argument("--original-path", help="Exact catalog path of the publisher file")
    target.add_argument("--source-name", help="Exact catalog title, RFC number, DOI, arXiv reference, or publisher-qualified archived skill name")
    evidence.add_argument("query")
    evidence.add_argument("--continue-section", metavar="CURSOR",
                          help="Read the next contiguous excerpt using the previous RFC section result\u0027s context.next_cursor")
    evidence.add_argument("--top",type=int,default=5)
    evidence.add_argument("--snippet-characters",type=int,default=900)
    evidence.add_argument("--expand-pdf-pages",type=int,default=0,
        help="Opt-in up to 5 distinct physically verified pages, globally round-robin across selected originals; up to 3 reading-order windows per page, original ranks unchanged")
    evidence.add_argument("--assess-passage-utility",action="store_true",
        help="Opt-in structural warnings for navigation, quoted prompts and possible PDF bibliography context; never score factual correctness or change source rank")
    evidence.add_argument("--inspect-original-tables",action="store_true",
        help="Opt-in original Markdown rows or an explicit exact HTML table cell; separate from normal excerpts; never infer answer truth")
    evidence.add_argument("--table-subject",
        help="Exact literal first-column publisher table subject; with --table-column and --inspect-original-tables, fail closed on missing/ambiguous values")
    evidence.add_argument("--table-column",
        help="Exact publisher Markdown or normalized HTML column name for the named source row, not a guessed semantic alias")
    evidence.add_argument("--table-exact-section",
        help="Optional exact original Markdown section heading to resolve duplicate subject+column rows")
    evidence.add_argument("--expect-table-value",
        help="Opt-in literal assertion to compare with the unique original publisher cell; different text is not a semantic falsehood. Requires exact table subject and column")
    evidence.add_argument("--follow-archived-links",action="store_true",
        help="Optional official first-party links from one original HTML article or Markdown publisher file into already archived source evidence; source candidates remain separate from answers")
    evidence.add_argument("--linked-originals",type=int,default=3,
        help="At most this many independently checked linked archived originals (1-5; only with --follow-archived-links)")
    evidence.add_argument("--exact-link-text",
        help="Optional exact publisher Markdown link label, for navigation without a semantic relevance claim; requires --follow-archived-links and one .md source")
    mixed = sub.add_parser(
        "find-source-evidence",
        help="Find native originals with physically verified PDF or SHA-linked non-PDF excerpts",
    )
    mixed.add_argument("query")
    mixed.add_argument("--source-name", help="Exact original title, RFC, DOI or pinned skill name; bypasses unrelated vector ranks")
    mixed.add_argument("--private-staged-pdf-preflight",type=Path,
        help="Optional exact private 327-PDF page repair receipt; add original-verified corrected page excerpts for native-selected PDFs without changing live indexes")
    mixed.add_argument("--follow-archived-links",action="store_true",
        help="Opt-in first-party links on selected verified HTML article or Markdown originals; approved already archived publisher excerpt candidates, no native rank promotion")
    mixed.add_argument("--linked-originals",type=int,default=3,
        help="Total linked source originals across ALL native-selected HTML and Markdown parents (1-5; opt-in only)")
    mixed.add_argument("--documents", type=int, default=3,
                       help="Inspect this many ranked originals (1–5)")
    mixed.add_argument("--passages", type=int, default=2,
                       help="Candidate excerpts per original (1–5)")
    mixed.add_argument("--snippet-characters", type=int, default=1600)
    mixed.add_argument("--expand-pdf-pages",type=int,default=0,
        help="Opt-in up to 5 distinct physically verified pages, globally round-robin across selected originals; up to 3 reading-order windows per page, original ranks unchanged")
    mixed.add_argument("--assess-passage-utility",action="store_true",
        help="Opt-in conservative structural diagnostics for original passage candidates, without changing source ranking or claiming answer sufficiency")
    mixed.add_argument("--inspect-original-tables",action="store_true",
        help="Opt-in exact publisher-byte Markdown table rows and column headings for native-selected originals; retain original ranks and excerpts")
    mixed.add_argument("--table-subject",
        help="Exact publisher first-column row identifier, restricted to previously selected native original documents; requires --inspect-original-tables and --table-column")
    mixed.add_argument("--table-column",
        help="Exact publisher original Markdown column heading, without a guessed semantic alias")
    mixed.add_argument("--table-exact-section",
        help="Optional exact original source section heading to disambiguate identical row subject and field")
    mixed.add_argument("--expect-table-value",
        help="Compare supplied literal string with ONE verified original cell across all selected originals; do not infer semantic truth or silently pick a source")
    mixed.add_argument("--scope-context-characters", type=int, default=320,
                       help="Bounded original-derived opening/neighbor context per excerpt (0 or 80–600); no physical-page claim")
    mixed.add_argument("--continue-section", metavar="CURSOR",
                       help="Continue one exact RFC section using the previous rfc_section_lookup.context.next_cursor")
    discover = sub.add_parser("find-pdf-evidence", help="Find originals by meaning and quote matching physical PDF pages")
    discover.add_argument("query")
    discover.add_argument("--documents", type=int, default=3, help="Inspect this many ranked originals (1–5)")
    discover.add_argument("--passages", type=int, default=2, help="Verified excerpts per PDF (1–5)")
    discover.add_argument("--snippet-characters", type=int, default=1600)
    audit = sub.add_parser("search-log", help="Review persistent native semantic query rankings")
    audit.add_argument("--last", type=int, default=10)
    a = p.parse_args()
    {"build": build, "import": import_source, "search": search, "status": status,
     "search-log": review_search_log, "find-pdf-evidence": find_pdf_evidence,
     "find-source-evidence": find_source_evidence,
     "source-evidence": original_source_evidence}[a.cmd](a)


if __name__ == "__main__":
    main()
