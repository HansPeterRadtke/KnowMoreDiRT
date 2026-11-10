"""Compare bounded native summary and facet retrieval on a frozen Raspi FileDB.

Run on Raspi with its semantic Python environment. Cases are private JSON
records with id, group, query and expected_path. The output contains ranks
and source identifiers, never original text or generated summaries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import raspi_hierarchical_library as library
from file_system_catalog.summary_search_batched import search_summary_entries_batched

POOL = 24
TOP = 8


def revision_key(hit):
    return (hit["original_path"], hit["original_sha256"], hit["derived_sha256"])


def compact(hit):
    node = hit["matched_summary_nodes"][0]
    return {key: node[key] for key in (
        "original_path", "original_sha256", "derived_sha256", "source_start_char",
        "source_end_char", "source_passage_sha256", "native_representation_id",
        "native_chunk_id", "summary_kind", "similarity",
    )}


def supported_ranking(summaries, combined):
    """Rerank only summary candidates, with a bounded same-revision bonus."""
    corroboration = {revision_key(hit): hit for hit in combined}
    ranked = []
    for anchor in summaries:
        score = anchor["similarity"]
        support = corroboration.get(revision_key(anchor))
        if support and support["similarity"] > score:
            score += 0.25 * (support["similarity"] - score)
        else:
            support = None
        ranked.append({"anchor": anchor, "support": support, "score": score})
    return sorted(ranked, key=lambda row: (-row["score"], row["anchor"]["original_path"]))


def compare(db, cases, vectors, model, current):
    results = []
    for number, (case, vector) in enumerate(zip(cases, vectors, strict=True), 1):
        pools = []
        for facets in (False, True):
            hits = search_summary_entries_batched(
                db, vector, limit=POOL, nodes_per_document=1, expected_model=model,
                require_source=current, include_facets=facets,
            )
            pools.append([compact(hit) for hit in hits])
        summary, combined = pools
        modes = {
            "summary": [{"anchor": hit} for hit in summary[:TOP]],
            "all_facets": [{"anchor": hit} for hit in combined[:TOP]],
            "supported": supported_ranking(summary, combined)[:TOP],
        }
        ranks = {}
        for name, rows in modes.items():
            ranks[name] = next((i for i, row in enumerate(rows, 1)
                                if row["anchor"]["original_path"] == case["expected_path"]), None)
        results.append({**case, "ranks": ranks, "modes": modes})
        print(f"{number}/{len(cases)} {case['id']} {ranks}", flush=True)
    return results


def verify_returned_sources(results):
    """Check every displayed anchor and every facet used to score it."""
    by_path = defaultdict(dict)
    for case in results:
        for rows in case["modes"].values():
            for row in rows:
                for hit in (row["anchor"], row.get("support")):
                    if hit:
                        by_path[hit["original_path"]][hit["native_representation_id"]] = hit
    count = 0
    for path, hits in by_path.items():
        source = library.checked_original(path)
        text = source["derived_text"]
        for hit in hits.values():
            start, end = hit["source_start_char"], hit["source_end_char"]
            if (revision_key(hit) != revision_key(source)
                    or not 0 <= start < end <= len(text)
                    or library.sha(text[start:end].encode()) != hit["source_passage_sha256"]):
                raise ValueError(f"source revision or passage differs for {path}")
            count += 1
    return {"originals": len(by_path), "distinct_representations": count}


def load_cases(path):
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("case file exceeds one MiB")
    cases = json.loads(raw)
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        raise ValueError("supply one to one hundred cases")
    for case in cases:
        if not isinstance(case, dict) or not all(
            isinstance(case.get(key), str) and case[key].strip()
            for key in ("id", "group", "expected_path", "query")
        ):
            raise ValueError("each case needs an ID, group, expected path and query")
        if len(case["query"]) > 1000:
            raise ValueError("query exceeds the native search character limit")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    return cases, hashlib.sha256(raw).hexdigest()


def main():
    from knowledge_library_semantic_model import LocalE5

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new private evidence path")
    cases, case_digest = load_cases(args.cases)
    manifest, ledger, sources = library._current_versions()

    def current(hit):
        path = hit["original_path"]
        original, derivative = ledger.get(path), sources.get(path)
        return bool(path in manifest and original and derivative
                    and original["sha256"] == hit["original_sha256"] == derivative[1]
                    and hit["derived_sha256"] == derivative[2]
                    and hit["original_bytes"] == original.get("bytes_received"))

    model = LocalE5(threads=1, batch_size=4)
    vectors = model.embed_queries([case["query"] for case in cases])
    # SQLite owns the transient on-disk copy. Release the live read connection
    # before benchmarking so the concurrent index writers can keep committing.
    with closing(sqlite3.connect("")) as frozen:
        with closing(sqlite3.connect(library.DB.as_uri() + "?mode=ro", uri=True)) as live:
            live.backup(frozen, pages=512, sleep=0.05)
            native_writes = live.total_changes
        frozen.execute("PRAGMA query_only=ON")
        counts = dict(frozen.execute(
            "SELECT representation_kind, count(*) FROM content_representations GROUP BY 1"
        ))
        print("Frozen snapshot ready", counts, flush=True)
        results = compare(frozen, cases, vectors, model.model, current)
        if native_writes or frozen.total_changes:
            raise RuntimeError("unexpected database writes")
    verified = verify_returned_sources(results)
    report = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "cases_sha256": case_digest,
        "embedding_model": model.revision, "pool": POOL, "top": TOP,
        "summary_weight": 0.75, "representation_counts": counts,
        "native_connection_changes": native_writes,
        "verified": verified, "cases": results,
    }
    with open(args.output, "x", encoding="utf-8",
              opener=lambda path, flags: os.open(path, flags, 0o600)) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print("Source verification passed", verified, flush=True)


if __name__ == "__main__":
    main()
