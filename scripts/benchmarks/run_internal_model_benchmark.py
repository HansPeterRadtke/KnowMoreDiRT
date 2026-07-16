#!/usr/bin/env python3
"""Resumable public-API internal benchmark."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITES = {
    name: {
        "corpus": REPO_ROOT / "tests" / "fixtures" / name,
        "qa": REPO_ROOT / "tests" / "fixtures" / f"{name}_qa.json",
    }
    for name in ["broad_raw_world", "hardcore_noise", "hard_raw_reasoning", "messy_raw_corpus", "structured_record_json"]
}


def score_answer(predicted: str, expected: str) -> tuple[bool, bool, float]:
    from knowmoredirt.evaluation import exact_match, semantic_match, token_f1
    return (
        exact_match(predicted, expected),
        semantic_match(predicted, expected),
        token_f1(predicted, expected),
    )


def acquire_output_lock(output_root: Path):
    """Hold an exclusive process lock for one benchmark output root."""
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".run.lock"
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        raise SystemExit(
            f"benchmark output root is already locked: {output_root} owner_pid={owner}"
        )
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def load_existing(path: Path) -> dict[tuple[str, str], dict]:
    output = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                output[(row["suite"], row["id"])] = row
    return output


def write_results_atomic(path: Path, rows: list[dict]) -> None:
    """Atomically checkpoint one row per suite/question key."""
    unique: dict[tuple[str, str], dict] = {}
    for row in rows:
        unique[(row["suite"], row["id"])] = row
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in unique.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/data/var/knowmoredirt/internal_benchmark")
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_lock = acquire_output_lock(output_root)
    results_path = output_root / "results.jsonl"
    if args.force and results_path.exists():
        results_path.unlink()
    existing = load_existing(results_path)
    if results_path.exists():
        write_results_atomic(results_path, list(existing.values()))
    selected = args.suite or list(SUITES)
    unknown = set(selected) - set(SUITES)
    if unknown:
        raise SystemExit(f"unknown suites: {sorted(unknown)}")
    from knowmoredirt import initialize, question
    records = list(existing.values())
    for suite_name in selected:
        suite = SUITES[suite_name]
        questions = json.loads(suite["qa"].read_text())["questions"]
        if args.question_id:
            wanted = set(args.question_id)
            questions = [item for item in questions if item["id"] in wanted]
        pending = [item for item in questions if (suite_name, item["id"]) not in existing]
        if pending:
            print(f"suite_start {suite_name} pending={len(pending)}", flush=True)
            initialize(suite["corpus"])
        for index, item in enumerate(questions, 1):
            key = (suite_name, item["id"])
            if key in existing:
                continue
            started = time.time()
            predicted = question(item["question"])
            exact_correct, semantic_correct, answer_f1 = score_answer(predicted, item["answer"])
            row = {
                "suite": suite_name,
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "expected": item["answer"],
                "predicted": predicted,
                "correct": exact_correct,
                "exact_correct": exact_correct,
                "semantic_correct": semantic_correct,
                "token_f1": round(answer_f1, 6),
                "elapsed_seconds": round(time.time() - started, 3),
            }
            existing[key] = row
            records = list(existing.values())
            write_results_atomic(results_path, records)
            print(f"answer {suite_name} {index}/{len(questions)} {item['id']} exact={row['exact_correct']} semantic={row['semantic_correct']} predicted={predicted!r}", flush=True)
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_suite[row["suite"]].append(row)
    summary = {
        "total": len(records),
        "exact_correct": sum(bool(row.get("exact_correct", row.get("correct"))) for row in records),
        "exact_score": (
            sum(bool(row.get("exact_correct", row.get("correct"))) for row in records) / len(records)
            if records else 0.0
        ),
        "semantic_correct": sum(bool(row.get("semantic_correct", row.get("correct"))) for row in records),
        "semantic_score": (
            sum(bool(row.get("semantic_correct", row.get("correct"))) for row in records) / len(records)
            if records else 0.0
        ),
        "average_token_f1": (
            sum(float(row.get("token_f1", 1.0 if row.get("correct") else 0.0)) for row in records) / len(records)
            if records else 0.0
        ),
        "suites": {
            name: {
                "total": len(rows),
                "exact_correct": sum(bool(row.get("exact_correct", row.get("correct"))) for row in rows),
                "exact_score": sum(bool(row.get("exact_correct", row.get("correct"))) for row in rows) / len(rows) if rows else 0.0,
                "semantic_correct": sum(bool(row.get("semantic_correct", row.get("correct"))) for row in rows),
                "semantic_score": sum(bool(row.get("semantic_correct", row.get("correct"))) for row in rows) / len(rows) if rows else 0.0,
                "average_token_f1": sum(float(row.get("token_f1", 1.0 if row.get("correct") else 0.0)) for row in rows) / len(rows) if rows else 0.0,
            }
            for name, rows in sorted(by_suite.items())
        },
        "endpoint": os.environ.get("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:14829/v1"),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    output_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
