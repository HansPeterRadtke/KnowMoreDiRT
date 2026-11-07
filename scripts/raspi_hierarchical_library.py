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
from file_system_catalog.hierarchical_summary_search import search_summary_entries, summary_index_status
from file_system_catalog.summary_search_audit import (
    append_query_audit, make_search_audit, recent_search_audits, AUDIT_DIR,
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
    model = LocalSummaryModel(
        args.model_url, model=args.model,
        timeout_seconds=args.model_timeout,
    )
    engine = HierarchicalSummarizer(
        model, output_tokens=args.output_tokens, safety_tokens=args.safety_tokens,
        max_levels=args.max_levels, max_nodes=args.max_nodes,
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
    }
    print("PLAN", json.dumps(plan, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    tree = engine.summarize(
        title=model_title, text=source["derived_text"],
        max_calls=args.max_calls,
    )
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
        "tree": asdict(tree),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    os.chmod(OUT, 0o700)
    output = payload_path(source["original_path"], source["original_sha256"], source["derived_sha256"])
    if output.exists():
        # Never overwrite a different completed LLM output on another run.
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            print("EXISTING COMPLETE SUMMARY TREE FOUND; refusing replacement", flush=True)
            return
    else:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.chmod(output, 0o600)
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
    if len(chunks) != len(tree.nodes) or len(reps) != len(tree.nodes):
        raise RuntimeError("missing summary vector: every tree node must be embedded")
    print("VERIFIED_SUMMARY_VECTORS", json.dumps({
        "original_path": source["original_path"],
        "nodes": len(tree.nodes),
        "vector_count": len(reps),
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
        found = search_summary_entries(
            conn, vector, limit=args.top,
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


def status(_args):
    with sqlite3.connect(f"file:{DB.resolve()}?mode=ro", uri=True) as conn:
        result = summary_index_status(conn)
    with sqlite3.connect(f"file:{SOURCE_MAP.resolve()}?mode=ro", uri=True) as source:
        extracted = source.execute(
            "SELECT COUNT(*) FROM originals WHERE status='complete'"
        ).fetchone()[0]
    result["complete_extracted_originals"] = extracted
    result["originals_without_recursive_summaries"] = max(
        0, extracted - result["original_documents_with_summaries"]
    )
    result["canonical_database"] = str(DB)
    result["database_bytes"] = DB.stat().st_size
    result["configured_database_budget_bytes"] = DB_BUDGET
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Generate every summary level for a complete original")
    b.add_argument("--original-path", required=True)
    b.add_argument("--model-url", default="http://10.8.0.7:14829")
    b.add_argument("--model", default="qwen38-q4-default")
    b.add_argument("--model-timeout", type=float, default=150)
    b.add_argument("--output-tokens", type=int, default=420)
    b.add_argument("--safety-tokens", type=int, default=384)
    b.add_argument("--max-levels", type=int, default=12)
    b.add_argument("--max-nodes", type=int, default=50000)
    b.add_argument("--max-calls", type=int, default=80)
    b.add_argument("--plan-only", action="store_true")
    i = sub.add_parser("import", help="Prepare or append every summary vector to the original FileDB")
    i.add_argument("--original-path", required=True)
    i.add_argument("--write", action="store_true")
    s = sub.add_parser("search", help="Find original documents by any summary level")
    s.add_argument("query")
    s.add_argument("--top", type=int, default=8)
    s.add_argument("--nodes-per-original", type=int, default=5)
    sub.add_parser("status")
    audit = sub.add_parser("search-log", help="Review persistent native semantic query rankings")
    audit.add_argument("--last", type=int, default=10)
    a = p.parse_args()
    {"build": build, "import": import_source, "search": search, "status": status,
     "search-log": review_search_log}[a.cmd](a)


if __name__ == "__main__":
    main()
