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
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
INFRA = Path("/data/infra/libexec")
sys.path.insert(0, str(INFRA))
from file_system_catalog.hierarchical_summary import HierarchicalSummarizer, SummaryNode, SummaryTree
from file_system_catalog.hierarchical_summary_model import LocalSummaryModel
from file_system_catalog.hierarchical_summary_store import summary_rows, append_summary_rows
from file_system_catalog.hierarchical_summary_search import summary_index_status
from file_system_catalog.summary_search_batched import search_summary_entries_batched
from file_system_catalog.source_evidence import source_evidence_candidates
from file_system_catalog.summary_search_audit import (
    append_query_audit, make_search_audit, recent_search_audits, AUDIT_DIR,
)
from file_system_catalog.summary_coverage import (
    current_completed_originals, summary_progress_metrics,
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


def file_sha256(path: Path) -> str:
    """Hash original publisher bytes without loading a multi-gigabyte PDF."""
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_original(relative: str) -> dict:
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
    manifest = json.loads(SOURCES.read_text())["sources"]
    entries = [r for r in manifest if r["path"] == relative]
    if len(entries) != 1:
        raise ValueError("source not found uniquely in original manifest")
    download = json.loads(LEDGER.read_text())["items"]
    completed = [r for r in download if r["path"] == relative and r["status"] == "complete"]
    if len(completed) != 1:
        raise ValueError("original source not completed in the download ledger")
    record = completed[0]
    path = LIBRARY / relative
    if (not path.is_file() or path.is_symlink() or
            not path.resolve().is_relative_to(LIBRARY.resolve())):
        raise ValueError("original file absent, symlink or outside library")
    stat = path.stat()
    if stat.st_size != record.get("bytes_received"):
        raise ValueError("original file byte count differs from completed receipt")
    if file_sha256(path) != record["sha256"]:
        raise ValueError("original file SHA differs from completed receipt")
    with sqlite3.connect(f"file:{SOURCE_MAP}?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT source_sha256,derived_sha256,derived_path,status "
            "FROM originals WHERE source_path=?",
            (relative,),
        ).fetchall()
    if len(rows) != 1 or rows[0][3] != "complete" or rows[0][0] != record["sha256"]:
        raise ValueError("no current complete source-map derivative for this original")
    _, derivative_sha, derivative_relative, _ = rows[0]
    if not derivative_relative:
        raise ValueError("empty derived relative path")
    rel = Path(derivative_relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("derived path is unsafe")
    derived = CACHE / "text" / rel
    if (not derived.is_file() or derived.is_symlink() or
            not derived.resolve().is_relative_to((CACHE / "text").resolve()) or
            derived.stat().st_size > 350 * 1024 * 1024):
        raise ValueError("derived source absent/unsafe/oversized")
    original_text = derived.read_bytes()
    if sha(original_text) != derivative_sha:
        raise ValueError("derived source SHA differs from source-map receipt")
    text = original_text.decode("utf-8", errors="strict")
    return {
        "original_path": relative,
        "title": entries[0]["title"],
        "original_sha256": record["sha256"],
        "original_bytes": stat.st_size,
        "derived_sha256": derivative_sha,
        "derived_text": text,
    }


def payload_path(original_path: str, original_sha: str, derived_sha: str) -> Path:
    basename = Path(original_path).name[:64]
    basename = re.sub(r"[^A-Za-z0-9_.-]+", "-", basename)
    key = sha((original_path + "\0" + original_sha + "\0" + derived_sha).encode("utf-8"))[:24]
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


def build(args) -> None:
    source = checked_original(args.original_path)
    # The full canonical title remains in the source manifest/payload.
    # Only the model prompt receives a bounded title, so unusually long
    # publisher titles never prevent complete document processing.
    model_title = source["title"][:300]
    if getattr(args,"native_facets",False) and args.output_tokens<700:
        raise ValueError("native keyword/facet output requires at least 700 reserved generation tokens")
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
        "completed_model_calls": 0,
        "last_summary_level": None,
        "last_node_id": None,
        "elapsed_seconds": 0.0,
    }
    save_source_progress(progress_file, progress)
    def on_node(node: SummaryNode, count: int) -> None:
        progress["completed_model_calls"] = count
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
            "completed_calls": count,
            "first_level_percent": progress["first_level_percent"],
            "source_end_char": node.end_char,
            "summary_sha256": node.summary_sha256,
        }, ensure_ascii=False), flush=True)
    try:
        tree = engine.summarize(
            title=model_title, text=source["derived_text"],
            max_calls=args.max_calls, on_node=on_node,
        )
    except Exception as error:
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
    output = payload_path(source["original_path"], source["original_sha256"], source["derived_sha256"])
    if output.exists():
        # Never overwrite a different completed LLM output on another run.
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            progress["status"] = "existing_complete_tree_differs"
            progress["last_error"] = "different completed tree already exists; replacement refused"
            save_source_progress(progress_file, progress)
            print("EXISTING COMPLETE SUMMARY TREE FOUND; refusing replacement", flush=True)
            return
    else:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.chmod(output, 0o600)
    progress.update({
        "status": "complete_tree_saved",
        "completed_model_calls": len(tree.nodes),
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


def import_source(args):
    source = checked_original(args.original_path)
    tree_file = payload_path(
        source["original_path"], source["original_sha256"], source["derived_sha256"],
    )
    if not tree_file.is_file() or tree_file.is_symlink():
        raise FileNotFoundError("verified complete summary tree not generated yet")
    payload = json.loads(tree_file.read_text(encoding="utf-8"))
    if (
        payload["original_path"] != source["original_path"]
        or payload["original_sha256"] != source["original_sha256"]
        or payload["derived_sha256"] != source["derived_sha256"]
        or payload["schema"] != "kmd-complete-hierarchical-summary-tree-v1"
    ):
        raise ValueError("summary tree is stale or belongs to a different original")
    tree = load_tree(payload)
    if (
        tree.root.start_char != 0
        or tree.root.end_char != len(source["derived_text"])
        or tree.original_characters != len(source["derived_text"])
    ):
        raise RuntimeError("refusing partial-book summary insertion")
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
            collection_id=COLLECTION,
        )
    if (len(chunks)!=len(tree.nodes)
            or sum(r["representation_kind"] in ("summary","summary_long") for r in reps)!=len(tree.nodes)):
        raise RuntimeError("missing required source-linked native summary vector")
    if (payload.get("native_facets") and
            not any(r["representation_kind"] in ("keyword","keyphrase") for r in reps)):
        raise RuntimeError("requested native keyword and keyphrase generation produced no indexable terms")
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
    import fcntl
    lockfile = DB.parent / ".primary-kmd-content.lock"
    with lockfile.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sqlite3.connect(DB, timeout=60) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            result = append_summary_rows(
                conn, chunk_rows=chunks, representation_rows=reps,
                max_database_bytes=DB_BUDGET,
            )
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite integrity failure detected after summary insertion")
    print("CANONICAL_KMD_UPDATED", json.dumps(result), flush=True)


def _current_versions():
    manifest = {x["path"]: x for x in json.loads(SOURCES.read_text())["sources"]}
    ledger = {
        x["path"]: x for x in json.loads(LEDGER.read_text())["items"]
        if x.get("status") == "complete" and x.get("sha256")
    }
    with sqlite3.connect(f"file:{SOURCE_MAP.resolve()}?mode=ro", uri=True) as con:
        source_map = {
            x[0]: x for x in con.execute(
                "SELECT source_path,source_sha256,derived_sha256,derived_path,status "
                "FROM originals WHERE status='complete'"
            )
        }
    return manifest, ledger, source_map


def _search_impl(args) -> dict:
    from knowledge_library_semantic_model import LocalE5
    manifest, ledger, sources = _current_versions()
    checked: dict[tuple[str, str, str], bool] = {}

    def verify(hit):
        path = hit["original_path"]
        revision = (path, hit["original_sha256"], hit["derived_sha256"])
        if revision in checked:
            return checked[revision]
        original = ledger.get(path)
        derivative = sources.get(path)
        good = bool(
            path in manifest and original and derivative and
            hit["original_sha256"] == original["sha256"] == derivative[1] and
            hit["derived_sha256"] == derivative[2] and
            hit["original_bytes"] == original.get("bytes_received")
        )
        if good:
            # Verify actual current extracted text and native source bytes
            # once per candidate document; reject stale or tampered summaries.
            try:
                rel = Path(derivative[3])
                if rel.is_absolute() or ".." in rel.parts:
                    good = False
                else:
                    source = CACHE / "text" / rel
                    if (
                        not source.is_file() or source.is_symlink()
                        or not source.resolve().is_relative_to((CACHE / "text").resolve())
                        or source.stat().st_size > 350 * 1024 * 1024
                    ):
                        good = False
                    else:
                        good = sha(source.read_bytes()) == derivative[2]
            except OSError:
                good = False
        checked[revision] = bool(good)
        return checked[revision]

    model = LocalE5(threads=2, batch_size=4)
    vector = model.embed_queries([args.query])[0]
    with sqlite3.connect(f"file:{DB.resolve()}?mode=ro", uri=True) as conn:
        found = search_summary_entries_batched(
            conn, vector, limit=args.top,
            include_facets=getattr(args,"with_facets",False),
            nodes_per_document=args.nodes_per_original,
            expected_model=model.model,
            require_source=verify,
        )
        snapshot = summary_index_status(conn)
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
    return {
        "query": args.query,
        "database": str(DB),
        "retrieval_engine": ("native_batched_facets_cosine_v1"
                             if getattr(args,"with_facets",False)
                             else "native_batched_cosine_v1"),
        "documents": out,
        "count": len(out),
        "originals_validated_for_version": sum(checked.values()),
        "index_scope": {
            "native_summary_vectors": snapshot["native_summary_vectors"],
            "summarized_originals": snapshot["original_documents_with_summaries"],
            "complete_extracted_originals": len(sources),
            "indexed_original_percent": round(100.0*snapshot["original_documents_with_summaries"]/max(1,len(sources)),6),
        },
        "relevance_warning": (
            "The ranked results are nearest summaries, not verified answers. "
            "A nonempty response does not establish that the requested topic "
            "is contained in the currently summarized original documents."
        ),
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
    try:
        source=checked_original(args.original_path)
        with sqlite3.connect(f"file:{DB.resolve()}?mode=ro",uri=True) as db:
            row=db.execute(
                "SELECT id FROM filesystem_entries WHERE entry_type='file' "
                "AND relative_path_display=?",(args.original_path,)
            ).fetchone()
        if row is None:
            raise ValueError("original has no canonical KnowMoreDiRT FileDB entry")
        result=source_evidence_candidates(
            original_path=args.original_path,
            original_title=source["title"],
            filesystem_entry_id=int(row[0]),
            original_sha256=source["original_sha256"],
            derived_sha256=source["derived_sha256"],
            derived_text=source["derived_text"],
            query=args.query,
            top=args.top,
            snippet_characters=args.snippet_characters,
        )
    except Exception as error:
        append_query_audit({
            "operation":"verified_original_source_evidence",
            "query":args.query,
            "original_path":args.original_path,
            "success":False,
            "error":f"{type(error).__name__}: {str(error)[:300]}",
            "duration_seconds":round(time.monotonic()-started,3),
        })
        raise
    receipt={
        "operation":"verified_original_source_evidence",
        "query":args.query,
        "original_path":args.original_path,
        "original_sha256":result["original_sha256"],
        "derived_sha256":result["derived_sha256"],
        "success":True,
        "count":result["returned_passages"],
        "retrieval_engine":"verified_original_ephemeral_fts5_v1",
        "duration_seconds":round(time.monotonic()-started,3),
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
    evidence.add_argument("--original-path",required=True)
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
