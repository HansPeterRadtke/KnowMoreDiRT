#!/usr/bin/env python3
"""Search approved archived HTML articles inside an explicit publisher directory.

For example, --publisher-url-prefix https://docs.blender.org/manual/en/4.5/
restricts discovery to that archived manual version. No fetching, version
inference, out-of-scope fallback, persistent index or model call occurs.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from file_system_catalog.publisher_html_article_discovery import (
    ArticleRAMIndex, PublisherScope, MAX_SELECTED_SOURCES,
    load_records, load_scoped_articles, readonly, verify_entry,
    final_metadata_fence, article_context,
)
from file_system_catalog.pdf_visual_link import safe_private_result_path
from raspi_publisher_firstunit_discovery import input_queries


def search(*, publisher_url_prefix, queries, top, source_map, native_db,
           catalog_path, completed_ledger, original_root, derivative_root,
           context_chars=0):
    scope = PublisherScope.parse(publisher_url_prefix)
    if type(context_chars) is not int or (context_chars != 0 and not 80 <= context_chars <= 2000):
        raise ValueError("article context must be zero or eighty through two thousand characters per neighbor")
    if type(top) is not int or not 1 <= top <= 10:
        raise ValueError("article result count must be between one and ten")
    if not isinstance(queries, list) or not 1 <= len(queries) <= 32:
        raise ValueError("bounded query batch required")
    seen = set()
    for q in queries:
        if (not isinstance(q, dict) or set(q) != {"id", "query"}
                or not isinstance(q["id"], str) or not 1 <= len(q["id"]) <= 80
                or q["id"] in seen):
            raise ValueError("invalid or duplicate article question")
        ArticleRAMIndex.clause(q["query"])
        seen.add(q["id"])
    start = time.monotonic()
    catalog = load_records(catalog_path, "sources")
    ledger = load_records(completed_ledger, "items")
    selected = {}
    proofs = {}
    result_queries = []
    with closing(readonly(source_map)) as sm, closing(readonly(native_db)) as native, ArticleRAMIndex() as idx:
        scan = load_scoped_articles(source_map=sm, original_root=original_root,
                                    catalog=catalog, ledger=ledger, index=idx, scope=scope)
        build_seconds = round(time.monotonic() - start, 3)
        for question in queries:
            channels = {}
            for channel in ("literal", "porter"):
                returned = []
                omitted = []
                for entry, ident, score in idx.ranked(question["query"], channel=channel,
                                                     limit=min(100, max(30, top * 6))):
                    if len(returned) == top:
                        break
                    try:
                        if entry.path not in proofs:
                            if len(selected) >= MAX_SELECTED_SOURCES:
                                raise RuntimeError("query batch selects too many distinct sources; split the batch")
                            proofs[entry.path] = verify_entry(
                                entry, source_map=sm, native_db=native, catalog=catalog, ledger=ledger,
                                original_root=original_root, derived_root=derivative_root, scope=scope)
                            selected[entry.path] = entry
                        excerpt = idx.excerpt(entry, question["query"], channel=channel)
                        if context_chars:
                            excerpt["source_context"] = article_context(entry, excerpt, characters=context_chars)
                    except (ValueError, OSError, UnicodeError) as error:
                        omitted.append({"original_path": entry.path, "reason": str(error)[:200]})
                        continue
                    public_proof = {k: v for k, v in proofs[entry.path].items()
                                    if k not in ("original_file_identity", "derivative_file_identity")}
                    returned.append({
                        "source_rank": len(returned) + 1,
                        "ephemeral_article_id": ident,
                        "rank_channel": channel + "_title_original_article_bm25",
                        "lexical_bm25_not_probability": score,
                        **public_proof, **excerpt,
                    })
                channels[channel] = {
                    "returned_current_originals": len(returned),
                    "source_originals": returned,
                    "stale_or_unverified_candidates_omitted": omitted,
                }
            result_queries.append({
                "id": question["id"], "query": question["query"],
                "query_sha256": hashlib.sha256(question["query"].encode()).hexdigest(),
                "independent_rank_channels": channels,
                "rank_fusion_applied": False,
            })

    # Reopen EVERY independent authority and rehash selected files after the
    # entire ranking batch. An earlier long-lived SQLite read view is not fresh.
    fresh_catalog = load_records(catalog_path, "sources")
    fresh_ledger = load_records(completed_ledger, "items")
    with closing(readonly(source_map)) as sm, closing(readonly(native_db)) as native:
        for path, entry in selected.items():
            fresh = verify_entry(entry, source_map=sm, native_db=native,
                                 catalog=fresh_catalog, ledger=fresh_ledger,
                                 original_root=original_root, derived_root=derivative_root, scope=scope)
            if fresh != proofs[path]:
                raise ValueError("selected publisher source changed during article search")

    # Catch metadata/native/file changes occurring while earlier selected
    # originals were being rehashed. This remains a fresh check, not an
    # atomic cross-store generation or a lock against all external writers.
    newest_catalog = load_records(catalog_path, "sources")
    newest_ledger = load_records(completed_ledger, "items")
    with closing(readonly(source_map)) as sm, closing(readonly(native_db)) as native:
        final_metadata_fence(selected, proofs, source_map=sm, native_db=native,
                             catalog=newest_catalog, ledger=newest_ledger, scope=scope,
                             original_root=original_root, derived_root=derivative_root)
    return {
        "schema": "kmd.readonly_scoped_publisher_html_article_discovery.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "publisher_url_prefix": scope.url,
        "query_count": len(queries), "source_scan": scan,
        "context_characters_per_side": context_chars,
        "index_build_seconds": build_seconds,
        "total_seconds": round(time.monotonic() - start, 3),
        "selected_original_count": len(selected),
        "all_displayed_sources_freshly_revalidated_after_batch": True,
        "cross_store_atomic_generation_pinned": False,
        "native_filedb_and_source_map_read_only": True,
        "no_persistent_second_database_created": True,
        "originals_downloaded_or_modified": 0,
        "out_of_scope_fallback_used": False,
        "independent_rank_channels_no_fusion": True,
        "warnings": [
            "This retrieves source candidates, not verified factual answers or calibrated relevance.",
            "The URL directory is an explicit archived-source scope, not proof of installed software, release status or applicability.",
            "Excerpts use normalized selected HTML container characters, not literal HTML bytes or rendered browser positions.",
            "Navigation and explicitly hidden HTML are excluded; external CSS, scripts, figures and browser visibility are not evaluated.",
            "Fresh source checks detect observed changes but do not pin an atomic generation across independently changing stores.",
            "Coverage excludes non-HTML, unapproved, malformed, ambiguous or oversized sources and must be read with source_scan.",
        ] + (["Optional context spans remain separate quotations with explicit omitted intervals; neither governing sections nor answer sufficiency are verified."] if context_chars else []),
        "queries": result_queries,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    choose = ap.add_mutually_exclusive_group(required=True)
    choose.add_argument("--query")
    choose.add_argument("--queries-file", type=Path)
    ap.add_argument("--publisher-url-prefix", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--context-chars", type=int, default=0,
                    help="Optional characters per original-text neighbor, 80..2000; zero disables context")
    ap.add_argument("--source-map", type=Path, default=Path("/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3"))
    ap.add_argument("--native-db", type=Path, default=Path("/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3"))
    ap.add_argument("--catalog", type=Path, default=Path("/data/infra/etc/knowledge-library-sources.json"))
    ap.add_argument("--completed-ledger", type=Path, default=Path("/data/var/Knowledge Library/_management/status.json"))
    ap.add_argument("--original-root", type=Path, default=Path("/data/var/Knowledge Library"))
    ap.add_argument("--derivative-root", type=Path, default=Path("/data/var/knowmoredirt/knowledge-library/semantic/text"))
    ap.add_argument("--output", type=Path, help="New private JSON file; stdout otherwise")
    args = ap.parse_args()
    if args.output is not None:
        args.output = safe_private_result_path(
            args.output, source_root=args.original_root,
            derived_root=args.derivative_root, native_filedb=args.native_db,
            source_map_db=args.source_map,
        )
    output = search(
        publisher_url_prefix=args.publisher_url_prefix,
        queries=input_queries(query=args.query, query_file=args.queries_file),
        top=args.top, source_map=args.source_map, native_db=args.native_db,
        catalog_path=args.catalog, completed_ledger=args.completed_ledger,
        original_root=args.original_root, derivative_root=args.derivative_root,
        context_chars=args.context_chars,
    )
    raw = (json.dumps(output, ensure_ascii=False, indent=2) + "\n").encode()
    if args.output is None:
        sys.stdout.buffer.write(raw)
    else:
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        print("OUTPUT", args.output, "SHA256", hashlib.sha256(raw).hexdigest(), file=sys.stderr)


if __name__ == "__main__":
    main()
