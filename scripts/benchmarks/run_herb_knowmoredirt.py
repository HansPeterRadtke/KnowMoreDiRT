#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

from knowmoredirt.engine import KnowMoreDiRTEngine

QA_KEYS = {"answerable_questions", "unanswerable_questions"}
ID_KEY_PRIORITY = (
    {"artifactid"},
    {"sourceid"},
    {"documentid"},
    {"messageid"},
    {"transcriptid"},
    {"chatid"},
    {"prid"},
    {"id"},
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def sanitize_herb_source(raw_root: Path, source_root: Path, *, rebuild: bool = False) -> dict[str, int]:
    marker = source_root / ".prepared.json"
    if marker.exists() and not rebuild:
        return json.loads(marker.read_text(encoding="utf-8"))
    if rebuild and source_root.exists():
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    counts = {"products": 0, "metadata_files": 0, "removed_qa_fields": 0}
    product_out = source_root / "products"
    product_out.mkdir(parents=True, exist_ok=True)
    for path in sorted((raw_root / "products").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"HERB product file is not an object: {path}")
        clean = {key: value for key, value in payload.items() if key not in QA_KEYS}
        counts["removed_qa_fields"] += sum(key in payload for key in QA_KEYS)
        atomic_write_json(product_out / path.name, clean)
        counts["products"] += 1
    metadata_out = source_root / "metadata"
    metadata_out.mkdir(parents=True, exist_ok=True)
    for path in sorted((raw_root / "metadata").glob("*.json")):
        shutil.copy2(path, metadata_out / path.name)
        counts["metadata_files"] += 1
    atomic_write_json(marker, counts)
    return counts


def _normalized_key(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "").replace(" ", "")


def _local_source_ids(value: Any, *, max_depth: int = 3) -> list[str]:
    """Choose the most specific shallow identifier class for one evidence record."""
    buckets: list[list[str]] = [[] for _ in ID_KEY_PRIORITY]

    def visit(node: Any, depth: int) -> None:
        if depth > max_depth or not isinstance(node, dict):
            return
        for key, child in node.items():
            normalized = _normalized_key(str(key))
            for index, key_group in enumerate(ID_KEY_PRIORITY):
                if normalized not in key_group:
                    continue
                values = child if isinstance(child, list) else [child]
                for item in values[:20]:
                    if isinstance(item, (dict, list)):
                        continue
                    text = str(item).strip()
                    if text and text not in buckets[index]:
                        buckets[index].append(text)
                break
        for child in node.values():
            if isinstance(child, dict):
                visit(child, depth + 1)
            # Do not descend through lists of child records: those IDs belong to
            # records that were not themselves cited.

    visit(value, 0)
    for bucket in buckets:
        if bucket:
            return bucket[:4]
    return []


def evidence_outputs(answer: Any) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    source_ids: list[str] = []
    chunk_ids: list[str] = []
    chunks: list[dict[str, Any]] = []
    source_seen: set[str] = set()
    chunk_seen: set[str] = set()
    for rank, view in enumerate(answer.evidence, start=1):
        record_id = str(view.get("record_id", "")).strip()
        source_path = str(view.get("source_path", "")).strip()
        local_source_ids = _local_source_ids(view.get("data", {}))
        for candidate in local_source_ids:
            if candidate not in source_seen:
                source_seen.add(candidate)
                source_ids.append(candidate)
        if not local_source_ids and source_path:
            fallback = Path(source_path).stem
            if fallback:
                local_source_ids.append(fallback)
                if fallback not in source_seen:
                    source_seen.add(fallback)
                    source_ids.append(fallback)
        if record_id and record_id not in chunk_seen:
            chunk_seen.add(record_id)
            chunk_ids.append(record_id)
        chunks.append(
            {
                "artifact_id": local_source_ids[0] if local_source_ids else source_path,
                "chunk_id": record_id or f"evidence-{rank}",
                "source_path": source_path,
                "text": str(view.get("excerpt", ""))[:4000],
                "score": max(0.0, 1.0 - (rank - 1) * 0.05),
            }
        )
    return source_ids[:48], chunk_ids[:48], chunks[:24]


def prediction_answer(text: str, question_type: str) -> Any:
    clean = text.strip()
    if clean.lower() == "unknown":
        return []
    if question_type in {"person", "company", "url", "pr"} and ";" in clean:
        return [item.strip() for item in clean.split(";") if item.strip()]
    return clean


def bundle_path(bundle_root: Path, question_id: str) -> Path:
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
    return bundle_root / f"{digest}.json"


def load_bundles(bundle_root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if not bundle_root.exists():
        return output
    for path in bundle_root.glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        question_id = str(row.get("question_id", ""))
        if question_id:
            output[question_id] = row
    return output


def materialize(run_dir: Path, questions: list[dict[str, Any]], bundles: dict[str, dict[str, Any]]) -> None:
    ordered = [bundles[row["question_id"]] for row in questions if row["question_id"] in bundles]
    atomic_write_jsonl(run_dir / "predictions.jsonl", [row["prediction"] for row in ordered])
    atomic_write_jsonl(run_dir / "retrieved_sources.jsonl", [row["retrieved_sources"] for row in ordered])
    atomic_write_jsonl(run_dir / "evidence_packets.jsonl", [row["evidence_packet"] for row in ordered])


def select_questions(rows: list[dict[str, Any]], *, limit: int = 0, question_id: str = "") -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for row in rows:
        question = {
            "question_id": str(row["question_id"]),
            "question": str(row["question"]),
            "question_type": str(row.get("question_type", "content")),
            "product_id": row.get("product_id"),
        }
        if question_id and question["question_id"] != question_id:
            continue
        clean.append(question)
    return clean[:limit] if limit else clean


def main() -> int:
    parser = argparse.ArgumentParser(description="Run KnowMoreDiRT on the official local HERB corpus")
    parser.add_argument("--raw-root", type=Path, default=Path("/data/var/herb_benchmark/raw/herb_raw/hf_snapshot"))
    parser.add_argument("--questions-jsonl", type=Path, default=Path("/data/var/herb_benchmark/normalized/herb_normalized/questions.jsonl"))
    parser.add_argument("--run-root", type=Path, default=Path("/data/var/herb_benchmark/runs"))
    parser.add_argument("--source-root", type=Path, default=Path("/data/var/herb_benchmark/external_runtime/knowmoredirt_source"))
    parser.add_argument("--run-name", default="knowmoredirt_raw_model_owned_v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question-id", default="")
    parser.add_argument("--rebuild-source", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    source_counts = sanitize_herb_source(args.raw_root, args.source_root, rebuild=args.rebuild_source)
    if args.prepare_only:
        print(json.dumps({"source_root": str(args.source_root), **source_counts}, ensure_ascii=False))
        return 0

    questions = select_questions(read_jsonl(args.questions_jsonl), limit=args.limit, question_id=args.question_id)
    run_dir = args.run_root / args.run_name
    bundle_root = run_dir / "bundles"
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundles = load_bundles(bundle_root)
    engine = KnowMoreDiRTEngine(args.source_root)
    total = len(questions)
    completed = sum(row["question_id"] in bundles for row in questions)
    print(f"HERB start run={args.run_name} completed={completed} total={total}", flush=True)

    for index, question_row in enumerate(questions, start=1):
        question_id = question_row["question_id"]
        if question_id in bundles:
            continue
        answer = engine.answer(question_row["question"])
        source_ids, chunk_ids, chunks = evidence_outputs(answer)
        answerable = answer.text.strip().lower() != "unknown"
        review = answer.diagnostics.get("review", {}) if isinstance(answer.diagnostics, dict) else {}
        raw_confidence = review.get("confidence", 0.8 if answerable else 0.7)
        try:
            confidence = max(0.0, min(float(raw_confidence), 1.0))
        except (TypeError, ValueError):
            confidence = 0.8 if answerable else 0.7
        prediction = {
            "question_id": question_id,
            "answer": prediction_answer(answer.text, question_row["question_type"]),
            "answerable": answerable,
            "confidence": confidence,
            "supporting_source_ids": source_ids,
            "supporting_chunk_ids": chunk_ids,
            "reasoning_summary": str(review.get("reason", "KnowMoreDiRT grounded execution")),
        }
        retrieved_sources = {
            "question_id": question_id,
            "question": question_row["question"],
            "source_ids": source_ids,
            "chunk_ids": chunk_ids,
            "candidate_entities": [],
            "top_score": chunks[0]["score"] if chunks else 0.0,
        }
        evidence_packet = {
            "question_id": question_id,
            "question": question_row["question"],
            "question_type": question_row["question_type"],
            "answerable": answerable,
            "system_variant": "knowmoredirt_raw_model_owned",
            "allowed_product_ids": [question_row["product_id"]] if question_row.get("product_id") else [],
            "exact_matches": [],
            "candidate_entities": [],
            "retrieved_chunks": chunks,
            "graph_facts": [],
            "temporal_facts": [],
        }
        bundle = {
            "question_id": question_id,
            "prediction": prediction,
            "retrieved_sources": retrieved_sources,
            "evidence_packet": evidence_packet,
        }
        atomic_write_json(bundle_path(bundle_root, question_id), bundle)
        bundles[question_id] = bundle
        completed += 1
        if completed % args.progress_every == 0 or completed == total:
            materialize(run_dir, questions, bundles)
            atomic_write_json(
                run_dir / "progress.json",
                {"run_name": args.run_name, "completed": completed, "total": total, "remaining": total - completed},
            )
            print(f"HERB progress run={args.run_name} completed={completed}/{total}", flush=True)

    materialize(run_dir, questions, bundles)
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "system": "knowmoredirt_raw_model_owned",
            "run_name": args.run_name,
            "question_count": total,
            "source_root": str(args.source_root),
            "source_counts": source_counts,
            "gold_used_for_prediction": False,
            "embedded_qa_fields_removed": True,
        },
    )
    print(f"HERB complete run={args.run_name} questions={total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
