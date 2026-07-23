#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from file_system_catalog.content_pipeline import (
    EmbeddingClient,
    search_literal_chunks,
    search_semantic_entries,
    stable_file_id,
    vector_from_blob,
)
from file_system_catalog.content_schema import (
    CHUNK_TABLE_NAME,
    REPRESENTATION_TABLE_NAME,
    STRENGTH_VALUES,
)


def dcg(grades: list[float], limit: int = 10) -> float:
    return sum((2.0**grade - 1.0) / math.log2(index + 2.0) for index, grade in enumerate(grades[:limit]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18139")
    parser.add_argument("--embedding-model", default="qwen3-embedding-0.6b-q8")
    parser.add_argument("--embedding-revision", default="370f27d7550e0def9b39c1f16d3fbaa13aa67728:Q8_0")
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    root = (arguments.root or arguments.manifest.parent.parent / "content-test-files").resolve()
    cases = {case["path"]: case for case in manifest["cases"]}
    failures: list[str] = []
    report: dict[str, Any] = {"failures": failures}
    connection = sqlite3.connect(arguments.database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        tables = sorted(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        expected_tables = ["content_chunks", "content_representations", "filesystem_entries"]
        if tables != expected_tables:
            failures.append(f"unexpected tables: {tables}")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if integrity != "ok":
            failures.append(f"integrity differs: {integrity}")
        if foreign_keys:
            failures.append(f"foreign key violations: {foreign_keys[:20]}")
        chunks = list(connection.execute(f"SELECT * FROM {CHUNK_TABLE_NAME}"))
        representations = list(connection.execute(f"SELECT * FROM {REPRESENTATION_TABLE_NAME}"))
        report["chunk_rows"] = len(chunks)
        report["representation_rows"] = len(representations)
        chunk_columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({CHUNK_TABLE_NAME})")}
        representation_columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({REPRESENTATION_TABLE_NAME})")
        }
        if "text" in chunk_columns or "analysis_text" in chunk_columns:
            failures.append("chunk table stores duplicate source text")
        for duplicate in ("file_id", "start_char", "end_char", "word_count", "token_count"):
            if duplicate in representation_columns:
                failures.append(f"representation table repeats chunk column: {duplicate}")
        file_paths = {
            row["relative_path_display"]
            for row in connection.execute(
                f"SELECT DISTINCT f.relative_path_display FROM {CHUNK_TABLE_NAME} c JOIN filesystem_entries f ON f.id=c.filesystem_entry_id"
            )
        }
        if file_paths != set(cases):
            failures.append(
                f"chunk file set differs: missing={sorted(set(cases)-file_paths)}, extra={sorted(file_paths-set(cases))}"
            )
        duplicate_chunks = connection.execute(
            f"""SELECT count(*) FROM (
            SELECT file_id,chunk_kind,chunk_index,content_sha256,start_char,end_char,count(*) n
            FROM {CHUNK_TABLE_NAME} GROUP BY 1,2,3,4,5,6 HAVING n>1)"""
        ).fetchone()[0]
        duplicate_representation_slots = connection.execute(
            f"""SELECT count(*) FROM (
            SELECT chunk_id,analysis_model,prompt_version,embedding_model,global_rank,count(*) n
            FROM {REPRESENTATION_TABLE_NAME} GROUP BY 1,2,3,4,5 HAVING n>1)"""
        ).fetchone()[0]
        if duplicate_chunks:
            failures.append(f"duplicate chunk slots: {duplicate_chunks}")
        if duplicate_representation_slots:
            failures.append(f"duplicate representation slots: {duplicate_representation_slots}")
        if len({row["chunk_id"] for row in chunks}) != len(chunks):
            failures.append("chunk IDs are not unique")
        if len({row["representation_id"] for row in representations}) != len(representations):
            failures.append("representation IDs are not unique")
        report["duplicate_chunks"] = duplicate_chunks
        report["duplicate_representation_slots"] = duplicate_representation_slots
        chunks_by_path: dict[str, list[sqlite3.Row]] = {}
        for row in connection.execute(
            f"""SELECT c.*,f.relative_path_display,f.relative_path_b64 FROM {CHUNK_TABLE_NAME} c
            JOIN filesystem_entries f ON f.id=c.filesystem_entry_id ORDER BY f.relative_path_display,c.chunk_kind,c.chunk_index"""
        ):
            chunks_by_path.setdefault(row["relative_path_display"], []).append(row)
        actual_chunk_count = 0
        over_context: dict[str, int] = {}
        bad_vectors = 0
        for path, case in cases.items():
            source = (root / path).read_text(encoding="utf-8")
            rows = chunks_by_path.get(path, [])
            actual = [row for row in rows if row["chunk_kind"] == "chunk"]
            aggregate = [row for row in rows if row["chunk_kind"] == "file"]
            actual_chunk_count += len(actual)
            if not actual:
                failures.append(f"no chunks for {path}")
                continue
            if actual[0]["start_char"] != 0 or actual[-1]["end_char"] != len(source):
                failures.append(f"chunk coverage differs for {path}")
            if case["must_be_multichunk"] and len(actual) < 2:
                failures.append(f"required multi-chunk case has {len(actual)} chunks: {path}")
            for row in actual:
                start, end = int(row["start_char"]), int(row["end_char"])
                text = source[start:end]
                expected_hash = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
                if row["character_count"] != end - start:
                    failures.append(f"character count differs for {path} chunk {row['chunk_index']}")
                if row["word_count"] != len(re.findall(r"\b[\w’'-]+\b", text)):
                    failures.append(f"word count differs for {path} chunk {row['chunk_index']}")
                if row["text_sha256"] != expected_hash:
                    failures.append(f"text hash differs for {path} chunk {row['chunk_index']}")
                if row["embedding_dimension"] != 1024 or row["embedding_dtype"] != "float32":
                    bad_vectors += 1
                    continue
                blob = bytes(row["embedding_blob"])
                if len(blob) != 4096 or hashlib.sha256(blob).hexdigest() != row["embedding_sha256"]:
                    bad_vectors += 1
                    continue
                vector = vector_from_blob(blob, int(row["embedding_dimension"]), row["embedding_dtype"])
                if abs(float(np.linalg.norm(vector)) - 1.0) > 2e-5:
                    bad_vectors += 1
            if aggregate:
                tokens = max(int(row["token_count"]) for row in aggregate)
                if tokens > 32768:
                    over_context[path] = tokens
                for row in aggregate:
                    if row["embedding_blob"] is not None:
                        failures.append(f"file aggregate unexpectedly stores a chunk vector: {path}")
        report["actual_chunk_rows"] = actual_chunk_count
        report["over_context_files"] = over_context
        bad_representation_vectors = 0
        strengths = set(STRENGTH_VALUES)
        by_chunk: dict[str, list[sqlite3.Row]] = {}
        for row in representations:
            by_chunk.setdefault(row["chunk_id"], []).append(row)
            if row["facet_strength"] not in strengths or row["item_strength"] not in strengths:
                failures.append(f"invalid verbal strength: {row['representation_id']}")
            blob = bytes(row["embedding_blob"])
            if (
                row["embedding_dimension"] != 1024
                or row["embedding_dtype"] != "float32"
                or len(blob) != 4096
                or hashlib.sha256(blob).hexdigest() != row["embedding_sha256"]
            ):
                bad_representation_vectors += 1
                continue
            vector = vector_from_blob(blob, int(row["embedding_dimension"]), row["embedding_dtype"])
            if abs(float(np.linalg.norm(vector)) - 1.0) > 2e-5:
                bad_representation_vectors += 1
        for chunk_id, rows in by_chunk.items():
            ranks = sorted(int(row["global_rank"]) for row in rows)
            if ranks != list(range(len(ranks))):
                failures.append(f"non-consecutive global ranks for chunk {chunk_id}: {ranks[:20]}")
            if sum(int(row["global_rank"]) == 0 and row["representation_kind"] in {"summary","summary_short"} for row in rows) != 1:
                failures.append(f"chunk does not have exactly one rank-zero summary: {chunk_id}")
        if bad_vectors:
            failures.append(f"invalid chunk vectors: {bad_vectors}")
        if bad_representation_vectors:
            failures.append(f"invalid representation vectors: {bad_representation_vectors}")
        report["bad_chunk_vectors"] = bad_vectors
        report["bad_representation_vectors"] = bad_representation_vectors
        first = connection.execute(
            f"""SELECT DISTINCT c.file_id,c.content_object_id FROM {CHUNK_TABLE_NAME} c
            JOIN filesystem_entries f ON f.id=c.filesystem_entry_id WHERE f.relative_path_display='01_single_topic_short_mountain_biking.txt'"""
        ).fetchone()
        copy = connection.execute(
            f"""SELECT DISTINCT c.file_id,c.content_object_id FROM {CHUNK_TABLE_NAME} c
            JOIN filesystem_entries f ON f.id=c.filesystem_entry_id WHERE f.relative_path_display='21_duplicate_copy_of_01.txt'"""
        ).fetchone()
        if first and copy:
            if first[0] == copy[0]:
                failures.append("duplicate paths share a file ID")
            if first[1] != copy[1]:
                failures.append("byte-identical files have different content object IDs")
        for row in connection.execute(
            f"""SELECT DISTINCT c.file_id,c.collection_id,f.relative_path_b64 FROM {CHUNK_TABLE_NAME} c
            JOIN filesystem_entries f ON f.id=c.filesystem_entry_id"""
        ):
            if row["file_id"] != stable_file_id(row["collection_id"], row["relative_path_b64"]):
                failures.append(f"file ID differs for {row['relative_path_b64']}")
                break
        marker_source = (root / "07_middle_needle_topic.txt").read_text(encoding="utf-8")
        marker_match = re.search(r"[0-9a-f]{8}", marker_source)
        if marker_match is None:
            failures.append("deterministic literal marker missing")
            marker = None
        else:
            marker = marker_match.group(0)
            literal = search_literal_chunks(connection, root, marker, case_sensitive=True, whole_word=True)
            expected_start = marker_source.index(marker)
            hit = next(
                (item for item in literal if item["relative_path_display"] == "07_middle_needle_topic.txt"),
                None,
            )
            if hit is None or hit["match_start_char"] != expected_start:
                failures.append("literal marker lookup or offset differs")
        report["literal_marker"] = marker
        embedder = EmbeddingClient(
            arguments.embedding_url,
            model=arguments.embedding_model,
            revision=arguments.embedding_revision,
        )
        bicycle_vector = embedder.embed(["show me all files about bicycles"])[0]
        bicycle_ranked = search_semantic_entries(connection, bicycle_vector)
        target = "19_single_topic_synonyms_without_common_label.txt"
        bicycle_rank = next(
            (index + 1 for index, item in enumerate(bicycle_ranked) if item["relative_path_display"] == target),
            None,
        )
        report["bicycle_paraphrase_rank"] = bicycle_rank
        if bicycle_rank is None or bicycle_rank > 10:
            failures.append(f"label-free bicycle file rank differs: {bicycle_rank}")
        topic_ids = sorted(manifest["topics"])
        query_vectors = embedder.embed([manifest["topics"][topic]["query"] for topic in topic_ids])
        ndcgs: list[float] = []
        hit5: list[float] = []
        retrieval: dict[str, Any] = {}
        for topic, vector in zip(topic_ids, query_vectors, strict=True):
            ranked = search_semantic_entries(connection, vector)
            relevance = {
                path: float(case["expected_topics"].get(topic, 0)) for path, case in cases.items()
            }
            grades = [relevance.get(item["relative_path_display"], 0.0) for item in ranked]
            ideal = sorted(relevance.values(), reverse=True)
            denominator = dcg(ideal)
            ndcg = dcg(grades) / denominator if denominator else 1.0
            primary = {path for path, case in cases.items() if topic in case["primary_topics"]}
            first_primary = next(
                (index + 1 for index, item in enumerate(ranked) if item["relative_path_display"] in primary),
                None,
            )
            ndcgs.append(ndcg)
            hit5.append(1.0 if first_primary is not None and first_primary <= 5 else 0.0)
            retrieval[topic] = {
                "ndcg_at_10": ndcg,
                "first_primary_rank": first_primary,
                "top10": [
                    {
                        "path": item["relative_path_display"],
                        "score": item["score"],
                        "kind": item["analysis_kind"],
                    }
                    for item in ranked[:10]
                ],
            }
        mean_ndcg = float(np.mean(ndcgs))
        primary_hit5 = float(np.mean(hit5))
        if mean_ndcg < 0.72:
            failures.append(f"mean nDCG@10 below threshold: {mean_ndcg}")
        if primary_hit5 < 0.90:
            failures.append(f"primary hit@5 below threshold: {primary_hit5}")
        report["retrieval"] = {
            "mean_ndcg_at_10": mean_ndcg,
            "primary_hit_at_5": primary_hit5,
            "topics": retrieval,
        }
        report.update(
            {
                "status": "ok" if not failures else "failed",
                "tables": tables,
                "integrity": integrity,
                "foreign_key_violations": len(foreign_keys),
            }
        )
    finally:
        connection.close()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
