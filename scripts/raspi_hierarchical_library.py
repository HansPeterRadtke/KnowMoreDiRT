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
import os
from pathlib import Path
import re
import sqlite3
import stat
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
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
from file_system_catalog.summary_search_batched import search_summary_entries_batched
from file_system_catalog.source_evidence import source_evidence_candidates
from file_system_catalog.named_source import resolve_source_name
from file_system_catalog.publication_identifier import publication_receipt_matches
from file_system_catalog.publisher_ccs import PREFIX_CHARS, first_page_ccs
from file_system_catalog.publisher_pdf_page import verify_pdf_page_excerpts
from file_system_catalog.source_passage_digests import source_passage_digests
from file_system_catalog.summary_search_audit import (
    append_query_audit, make_search_audit, recent_search_audits, AUDIT_DIR,
)
from file_system_catalog.summary_coverage import (
    current_completed_originals, pending_failures, revision_identity, summary_progress_metrics,
)

DB = Path("/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3")
LIBRARY = Path("/data/var/Knowledge Library")
SOURCES = Path("/data/infra/etc/knowledge-library-sources.json")
LEDGER = LIBRARY / "_management/status.json"
CACHE = Path("/data/var/knowmoredirt/knowledge-library/semantic")
SOURCE_MAP = CACHE / "source-map.sqlite3"
OUT = Path("/data/var/knowmoredirt/knowledge-library/summaries")
COLLECTION = "raspi-original-books-semantic-content-v1"
DB_BUDGET = 4 * 1024 * 1024 * 1024
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
    if (include_publisher_text and path.suffix.casefold() in (".md", ".markdown", ".txt")
            and type(record.get("bytes_received")) is int and record["bytes_received"] <= 8 * 1024 * 1024):
        capture = bytearray()
    original_bytes = verify_publisher_revision(path, record, capture=capture)
    with sqlite3.connect(f"file:{SOURCE_MAP}?mode=ro", uri=True) as con:
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
        **({"publisher_text_bytes": bytes(capture)} if capture is not None else {}),
    }


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


def import_source(args):
    source = checked_original(args.original_path)
    tree_file = payload_path(
        source["original_path"], source["original_sha256"], source["derived_sha256"],
    )
    payload, tree = import_tree(tree_file, source)
    from knowledge_library_semantic_model import LocalE5
    model = LocalE5(threads=2, batch_size=4)
    with sqlite3.connect(f"file:{DB.resolve()}?mode=ro", uri=True) as ro:
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
        acquire_native_filedb_content_lock(lock.fileno(),wait_seconds=90)
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
    with sqlite3.connect(f"file:{SOURCE_MAP.resolve()}?mode=ro", uri=True) as con:
        source_map = _unique_path_records(con.execute(
            "SELECT source_path,source_sha256,derived_sha256,derived_path,status "
            "FROM originals WHERE status='complete'"
        ), path_key=0)
    return manifest, ledger, source_map


def _search_impl(args) -> dict:
    manifest, ledger, sources = _current_versions()
    name_match = resolve_source_name(args.query, manifest)
    preferred_path = name_match.get("original_path") if name_match else None
    checked = set()

    def verify_document(hits):
        path = hits[0]["original_path"]
        original = ledger.get(path)
        derivative = sources.get(path)
        if path not in manifest or not original or not derivative or original.get("sha256") != derivative[1]:
            return []
        if (path == preferred_path and name_match.get("kind") in ("catalog_arxiv", "catalog_doi")
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
        if verified:
            checked.add((path, derivative[1], derivative[2]))
        return verified

    missing_identifier = name_match and name_match["status"] == "no_catalog_match"
    with sqlite3.connect(f"file:{DB.resolve()}?mode=ro", uri=True) as conn:
        found = []
        if not missing_identifier:
            from knowledge_library_semantic_model import LocalE5
            model = LocalE5(threads=2, batch_size=4)
            vector = model.embed_queries([args.query])[0]
            found = search_summary_entries_batched(
                conn, vector, limit=args.top,
                include_facets=getattr(args, "with_facets", False),
                nodes_per_document=args.nodes_per_original,
                expected_model=model.model,
                filter_source_nodes=verify_document, preferred_path=preferred_path,
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
            warning = ("The named original has no verified summary result. Any ranked documents are other "
                       "sources. Use source-evidence with the exact catalog path to inspect the named original.")
    elif missing_identifier:
        warning = "The catalog has no archived PDF matching this publication identifier and requested version."
    elif name_match:
        warning = ("The source name matches multiple catalog paths. The listed documents use semantic "
                   "ranking. Select an exact candidate path to choose an archived source.")
    return {
        **({"source_name_match": name_match} if name_match else {}),
        "query": args.query,
        "database": str(DB),
        "retrieval_engine": ("exact_publication_catalog_lookup_v1" if missing_identifier
                             else "native_batched_facets_cosine_v1"
                             if getattr(args,"with_facets",False)
                             else "native_batched_cosine_v1"),
        "documents": out,
        "count": len(out),
        "originals_validated_for_version": len(checked),
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


def original_source_evidence(args) -> None:
    """Inspect verified original pages; never mistake lexical hits for truth."""
    started=time.monotonic()
    original_path = getattr(args, "original_path", None)
    evidence_query = args.query
    match = None
    publication_record = None
    try:
        if original_path is None:
            explicit_name = getattr(args, "source_name", None)
            manifest = _unique_path_records(json.loads(SOURCES.read_text())["sources"])
            match = resolve_source_name(explicit_name or args.query, manifest)
            if not match:
                raise ValueError("source name unrecognized; supply an exact --source-name or --original-path")
            if match["status"] == "no_catalog_match":
                raise ValueError("no cataloged PDF matches this publication identifier and requested version")
            if match["status"] != "unique":
                choices = "\n".join(match["candidate_paths"])
                raise ValueError(f"source name matches {match['matching_catalog_paths']} catalog paths; "
                                 f"select --original-path. Available choices include\n{choices}")
            original_path = match["original_path"]
            if match["kind"] in ("catalog_arxiv", "catalog_doi"):
                publication_record = manifest[original_path]
            if explicit_name is None:
                evidence_query = match.get("subject_query", "")
                if not evidence_query:
                    raise ValueError("supply the topic to find after --source-name or --original-path")
        source=checked_original(original_path, include_publisher_text=True, publication_record=publication_record)
        with sqlite3.connect(f"file:{DB.resolve()}?mode=ro",uri=True) as db:
            row=db.execute(
                "SELECT id FROM filesystem_entries WHERE entry_type='file' "
                "AND relative_path_display=?",(original_path,)
            ).fetchone()
        if row is None:
            raise ValueError("original has no canonical KnowMoreDiRT FileDB entry")
        result=source_evidence_candidates(
            original_path=original_path,
            original_title=source["title"],
            filesystem_entry_id=int(row[0]),
            original_sha256=source["original_sha256"],
            derived_sha256=source["derived_sha256"],
            derived_text=source["derived_text"],
            publisher_text_bytes=source.get("publisher_text_bytes"),
            query=evidence_query,
            top=args.top,
            snippet_characters=args.snippet_characters,
        )
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
    except Exception as error:
        append_query_audit({
            "operation":"verified_original_source_evidence",
            "query":args.query,
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
        "success":True,
        "count":result["returned_passages"],
        "retrieval_engine":"verified_original_ephemeral_fts5_v1",
        "duration_seconds":round(time.monotonic()-started,3),
        **({"pdf_page_verification": result["pdf_page_verification"]} if "pdf_page_verification" in result else {}),
        "ranked_source_spans":[{
            "rank":i,
            "source_section":v["source_section"],
            "source_start_char":v["source_start_char"],
            "source_end_char":v["source_end_char"],
            "source_span_sha256":v["source_span_sha256"],
            "excerpt_start_char":v["excerpt_start_char"],
            "excerpt_end_char":v["excerpt_end_char"],
            "excerpt_sha256":v["excerpt_sha256"],
            "lexical_bm25":v["lexical_bm25"],
            "matched_source_terms":v["matched_source_terms"],
            **({"matched_word_forms": v["matched_word_forms"]} if "matched_word_forms" in v else {}),
            **({"matched_identifier_words": v["matched_identifier_words"]} if "matched_identifier_words" in v else {}),
            **({"publisher_excerpt": v["publisher_excerpt"]} if "publisher_excerpt" in v else {}),
            **({"publisher_pdf_page": v["publisher_pdf_page"]} if "publisher_pdf_page" in v else {}),
        } for i,v in enumerate(result["passages"],1)],
        "evidence_warning":result["evidence_warning"],
    }
    filename=append_query_audit(receipt)
    result["query"]=args.query
    result["retrieval_engine"]="verified_original_ephemeral_fts5_v1"
    result["query_audit_logged"]=True
    result["private_query_audit_file"]=str(filename)
    print(json.dumps(result,indent=2,ensure_ascii=False))


def status(_args):
    with sqlite3.connect(f"file:{SOURCE_MAP.resolve()}?mode=ro", uri=True) as source:
        current = {
            row[0]: (row[1], row[2], int(row[3]))
            for row in source.execute(
                "SELECT source_path,source_sha256,derived_sha256,text_characters "
                "FROM originals WHERE status='complete' AND text_characters>100"
            )
        }
    with sqlite3.connect(f"file:{DB.resolve()}?mode=ro", uri=True) as conn:
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
    target.add_argument("--source-name", help="Exact catalog title, DOI, arXiv reference, or publisher-qualified archived skill name")
    evidence.add_argument("query")
    evidence.add_argument("--top",type=int,default=5)
    evidence.add_argument("--snippet-characters",type=int,default=900)
    audit = sub.add_parser("search-log", help="Review persistent native semantic query rankings")
    audit.add_argument("--last", type=int, default=10)
    a = p.parse_args()
    {"build": build, "import": import_source, "search": search, "status": status,
     "search-log": review_search_log,
     "source-evidence": original_source_evidence}[a.cmd](a)


if __name__ == "__main__":
    main()
