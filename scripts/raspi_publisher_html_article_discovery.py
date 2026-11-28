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
    final_metadata_fence, article_context, read_bounded, MAX_HTML_BYTES,
)
from file_system_catalog.pdf_visual_link import safe_private_result_path
from file_system_catalog.source_review_audit import safe_file
from raspi_publisher_firstunit_discovery import input_queries


def search(*, publisher_url_prefix, queries, top, source_map, native_db,
           catalog_path, completed_ledger, original_root, derivative_root,
           context_chars=0, include_local_semantic=False, semantic_model=None,
           include_approved_parent_links=False):
    scope = PublisherScope.parse(publisher_url_prefix)
    if type(context_chars) is not int or (context_chars != 0 and not 80 <= context_chars <= 2000):
        raise ValueError("article context must be zero or eighty through two thousand characters per neighbor")
    if type(top) is not int or not 1 <= top <= 10:
        raise ValueError("article result count must be between one and ten")
    if type(include_local_semantic) is not bool or (semantic_model is not None
             and not include_local_semantic):
        raise ValueError('local semantic source discovery requires explicit opt-in')
    if type(include_approved_parent_links) is not bool:
        raise ValueError('publisher parent navigation requires an explicit boolean opt-in')
    if include_approved_parent_links and isinstance(queries,list) and len(queries) > 4:
        raise ValueError('bounded parent navigation admits at most four questions')
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
    catalog_rows = list(catalog.values()) if include_approved_parent_links else None
    with closing(readonly(source_map)) as sm, closing(readonly(native_db)) as native, ArticleRAMIndex() as idx:
        scan = load_scoped_articles(source_map=sm, original_root=original_root,
                                    catalog=catalog, ledger=ledger, index=idx, scope=scope)
        build_seconds = round(time.monotonic() - start, 3)
        semantic_index = None
        if include_local_semantic:
            # Keep the default stdlib/SQLite-only article CLI independent
            # from optional NumPy/ONNX packages and their import side effects.
            from file_system_catalog.publisher_html_article_semantic import OriginalArticleSemanticIndex
            # Only the explicit opt-in loads the existing pinned LOCAL CPU
            # E5 model. Operator must obtain a broker lease before calling it.
            # All selected results reuse the same full original/source-map
            # evidence gates and final post-query source revision rechecks.
            if semantic_model is None:
                sys.path.insert(0, '/data/infra/libexec')
                from knowledge_library_semantic_model import LocalE5
                semantic_model = LocalE5(threads=1, batch_size=4)
            semantic_index = OriginalArticleSemanticIndex(list(idx.entries.values()),
                                                          semantic_model)
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
            if semantic_index is not None:
                for name in ("article", "catalog_title"):
                    candidates = semantic_index.ranked(question["query"],
                                                       channel=name, limit=top)
                    returned = []
                    for candidate in candidates:
                        path = candidate["original_path"]
                        entry = idx.entries[idx.ids[path]]
                        a, b = (candidate["normalized_original_article_span_start_char"],
                                candidate["normalized_original_article_span_end_char"])
                        excerpt = candidate["source_excerpt"]
                        if (not 0 <= a < b <= len(entry.text)
                                or entry.text[a:b] != excerpt
                                or hashlib.sha256(excerpt.encode()).hexdigest()
                                != candidate["source_excerpt_sha256"]):
                            raise ValueError("local semantic passage left verified original article span")
                        if path not in proofs:
                            if len(selected) >= MAX_SELECTED_SOURCES:
                                raise RuntimeError("query batch selects too many distinct sources; split the batch")
                            proofs[path] = verify_entry(
                                entry, source_map=sm, native_db=native, catalog=catalog,
                                ledger=ledger, original_root=original_root,
                                derived_root=derivative_root, scope=scope)
                            selected[path] = entry
                        public_proof = {k: v for k, v in proofs[path].items()
                                        if k not in ("original_file_identity", "derivative_file_identity")}
                        returned.append({"source_rank": len(returned)+1,
                                         **public_proof, **candidate})
                    channels["semantic_" + name] = {
                        "returned_current_originals":len(returned),
                        "source_originals":returned,
                        "stale_or_unverified_candidates_omitted":[],
                    }
            # Optional original-publisher ARTICLE parent navigation is
            # completely independent of the ranked candidate lists. A real
            # first-party <article> anchor and a same-origin approved catalog
            # URL that is a proper path-segment ancestor of the source are
            # both required; no word overlap or semantic model score is
            # inferred from this structural relationship.
            parent_navigation = []
            scanned_source_paths = set()
            if include_approved_parent_links:
                from file_system_catalog.archived_html_source_links import source_link_candidates
                for ranked_channel, group in channels.items():
                    for ranked in group["source_originals"][:10]:
                        path = ranked["original_path"]
                        if path in scanned_source_paths:
                            continue
                        scanned_source_paths.add(path)
                        parent_entry = idx.entries[idx.ids[path]]
                        original = safe_file(original_root,path)
                        current_original_bytes = read_bounded(original,MAX_HTML_BYTES)
                        if hashlib.sha256(current_original_bytes).hexdigest() != parent_entry.source_map_row[1]:
                            raise ValueError('linked publisher parent source SHA changed during navigation')
                        hints = source_link_candidates(
                            current_original_bytes,parent_entry.source_map_row[1],
                            parent_entry.url,catalog_rows,ledger,question["query"],
                            original_path=path,limit=5,
                            include_unmatched_ancestor_links=True,
                            only_explicit_ancestor_navigation=True)
                        for link in hints["selected_links"]:
                            if not link["explicit_archived_source_ancestor_reference"]:
                                continue
                            target_path = link["catalog_target_original_path"]
                            if target_path not in idx.ids:
                                # Only in the EXPLICIT caller-selected version
                                # / publisher topic scope; never broaden it.
                                continue
                            destination = idx.entries[idx.ids[target_path]]
                            if not parent_entry.url.startswith(destination.url.rstrip("/")+"/"):
                                raise ValueError('publisher article breadcrumb is not a proper current URL ancestor')
                            if destination.path not in proofs:
                                if len(selected) >= MAX_SELECTED_SOURCES:
                                    raise RuntimeError('bounded article navigation selects too many current source originals')
                                proofs[destination.path] = verify_entry(
                                    destination,source_map=sm,native_db=native,
                                    catalog=catalog,ledger=ledger,original_root=original_root,
                                    derived_root=derivative_root,scope=scope)
                                selected[destination.path] = destination
                            proof={k:v for k,v in proofs[destination.path].items()
                                   if k not in ("original_file_identity","derivative_file_identity")}
                            parent_navigation.append({
                                "source_ranked_original_path":path,
                                "source_rank_channels":sorted(
                                    k for k,g in channels.items()
                                    if any(z["original_path"]==path for z in g["source_originals"])),
                                "publisher_article_link_proof":link,
                                "approved_linked_parent_original":proof,
                                "original_link_sha_and_source_rechecked":True,
                                "linked_parent_has_NO_new_semantic_rank":True,
                                "linked_parent_is_not_verified_answer":True,
                                "other_versions_or_unapproved_sources_not_followed":True,
                            })
                            if len(parent_navigation)>=32:
                                break
                        if len(parent_navigation)>=32:
                            break
                    if len(parent_navigation)>=32:
                        break
            result_queries.append({
                "id": question["id"], "query": question["query"],
                "query_sha256": hashlib.sha256(question["query"].encode()).hexdigest(),
                "independent_rank_channels": channels,
                "rank_fusion_applied": False,
                **({"approved_original_article_parent_breadcrumbs":parent_navigation,
                    "selected_original_article_sources_inspected_for_parent_links":len(scanned_source_paths),
                    "linked_parent_sources_are_navigation_only_not_relevance_ranks":True}
                   if include_approved_parent_links else {}),
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
        "schema": ("kmd.readonly_scoped_publisher_html_article_semantic.v1"
                   if include_local_semantic else
                   "kmd.readonly_scoped_publisher_html_article_discovery.v1"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "publisher_url_prefix": scope.url,
        "query_count": len(queries), "source_scan": scan,
        "context_characters_per_side": context_chars,
        **({"approved_parent_breadcrumbs_opt_in":True,
            "approved_parent_breadcrumbs_are_not_rank_fusion_or_factual_answers":True,
            "only_explicit_original_article_same_version_ancestor_links":True}
           if include_approved_parent_links else {}),
        **({"local_article_semantic_opt_in":True,
            "local_semantic_model":semantic_model.model,
            "local_semantic_model_revision":semantic_model.revision,
            "local_semantic_channels_are_independent_and_never_score_fused":True,
            "semantic_article_windows_cover_each_admitted_current_original":True,
            "semantic_title_channel_is_catalog_metadata_only":True}
           if include_local_semantic else {}),
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
        ] + (["Optional context spans remain separate quotations with explicit omitted intervals; neither governing sections nor answer sufficiency are verified."] if context_chars else [])
          + (["Opt-in approved publisher breadcrumbs show only verified source-document navigation, not relevance, answer correctness, a new rank or a different edition. Every linked overview is fully SHA-rechecked against its approved original and native source identity after the batch."] if include_approved_parent_links else [])
          + (["Local semantic E5 ranks are opt-in uncalibrated discoverability only. The title-only model channel is metadata, not publisher prose; article windows are SHA-linked to the exact current original and the full read-back proof is unchanged.",
              "The original semantic model has a trained 512-token context. Every admitted article is covered by explicit non-overlapping word-aligned windows; overlarge sources fail rather than being silently truncated."] if include_local_semantic else []),
        "queries": result_queries,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    choose = ap.add_mutually_exclusive_group(required=True)
    choose.add_argument("--query")
    choose.add_argument("--queries-file", type=Path)
    ap.add_argument("--publisher-url-prefix", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--include-local-semantic",action="store_true",
                    help="Opt-in local CPU E5 article and catalog-title channels. Requires broker admission; no global model/rank change.")
    ap.add_argument("--include-approved-parent-links",action="store_true",
                    help="Opt-in SHA-verified parent overview navigation from original first-party HTML article links; separate from ranked candidates and limited to four questions.")
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
        include_local_semantic=args.include_local_semantic,
        include_approved_parent_links=args.include_approved_parent_links,
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
