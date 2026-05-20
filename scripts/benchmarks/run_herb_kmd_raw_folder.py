#!/usr/bin/env python3
"""Run KnowMoreDiRT against the local HERB benchmark with raw-folder input.

This script is benchmark glue only. It does not change KMD's public contract:
KMD is initialized once with a raw folder path and each question is sent through
``knowmoredirt.question(text)``. The script writes HERB-compatible prediction
files and then invokes the existing local HERB evaluator on those completed
predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_RAW_FOLDER = Path("/data/var/herb_benchmark/raw/herb_raw/hf_snapshot")
DEFAULT_HERB_ROOT = Path("/data/src/github/devtests/herb_benchmark")
DEFAULT_VAR_ROOT = Path("/data/var/herb_benchmark")
DEFAULT_KMD_REPORT_ROOT = Path("/data/var/knowmoredirt/reports")
DEFAULT_KMD_RUN_ROOT = Path("/data/var/knowmoredirt/herb_runs")


def log_event(log_path: Path, event: str, **payload: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **payload}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def read_official_questions(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            rows.append(
                {
                    "question_id": str(source["question_id"]),
                    "question": str(source["question"]),
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def serialize_answer(public_answer: str) -> str | list[str]:
    answer = str(public_answer or "").strip()
    if not answer or answer.lower() == "unknown":
        return ""
    values = re.findall(r"(?:https?://[^\s,;]+|[A-Z][A-Z0-9]{1,9}-\d+[A-Z0-9-]*|eid_[a-z0-9]+|CUST-\d+)", answer)
    if values and len(" ".join(values)) >= max(3, len(answer.strip()) - 4):
        return list(dict.fromkeys(value.rstrip(".") for value in values))
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description="Run KMD raw-folder public API on local HERB.")
    parser.add_argument("--raw-folder", default=str(DEFAULT_RAW_FOLDER))
    parser.add_argument("--herb-root", default=str(DEFAULT_HERB_ROOT))
    parser.add_argument("--var-root", default=str(DEFAULT_VAR_ROOT))
    parser.add_argument("--run-root", default=str(DEFAULT_KMD_RUN_ROOT))
    parser.add_argument("--report-root", default=str(DEFAULT_KMD_REPORT_ROOT))
    parser.add_argument("--run-name", default=f"kmd_public_raw_folder_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit; 0 means all questions.")
    parser.add_argument("--use-local-model", action="store_true", help="Enable KMD's optional localhost-only migrated DRT model-query planner.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run directory without deleting completed JSONL outputs.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    raw_folder = Path(args.raw_folder).resolve()
    herb_root = Path(args.herb_root).resolve()
    var_root = Path(args.var_root).resolve()
    run_dir = Path(args.run_root).resolve() / args.run_name
    report_root = Path(args.report_root).resolve()
    normalized_questions = var_root / "normalized" / "herb_normalized" / "questions.jsonl"
    log_path = run_dir / "progress.jsonl"
    checkpoint_path = run_dir / "kmd_public_answers.jsonl"
    sanitized_questions_path = run_dir / "questions_sanitized_for_kmd.jsonl"

    if not raw_folder.is_dir():
        raise FileNotFoundError(raw_folder)
    if not normalized_questions.exists():
        raise FileNotFoundError(normalized_questions)
    if not herb_root.is_dir():
        raise FileNotFoundError(herb_root)

    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(herb_root / "src"))
    if args.use_local_model:
        os.environ["KMD_USE_LOCAL_MODEL"] = "1"

    import knowmoredirt as kmd  # noqa: WPS433 - operational benchmark adapter
    from knowmoredirt import public as kmd_public  # noqa: WPS433
    from herb_kgqa.config import get_settings  # noqa: WPS433
    from herb_kgqa.evaluator import evaluate_run  # noqa: WPS433

    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for output_name in ["retrieved_sources.jsonl", "evidence_packets.jsonl", "predictions.jsonl", "kmd_public_answers.jsonl"]:
            output_path = run_dir / output_name
            if output_path.exists():
                output_path.unlink()

    all_questions = read_official_questions(normalized_questions)
    questions = all_questions[: args.limit] if args.limit else all_questions
    write_jsonl(sanitized_questions_path, questions)
    completed_ids: set[str] = set()
    answered_count = 0
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                checkpoint = json.loads(line)
                question_id = str(checkpoint.get("question_id") or "")
                if question_id:
                    completed_ids.add(question_id)
                    if checkpoint.get("answered"):
                        answered_count += 1
    log_event(
        log_path,
        "resume_start" if args.resume else "run_start",
        repo_root=str(repo_root),
        raw_folder=str(raw_folder),
        run_dir=str(run_dir),
        official_questions=str(normalized_questions),
        sanitized_questions=str(sanitized_questions_path),
        total_questions=len(questions),
        already_completed=len(completed_ids),
        already_answered=answered_count,
        limit=args.limit,
        query_input_fields=["question_id", "question"],
        model_status="optional localhost migrated DRT model-query planner enabled" if args.use_local_model else "not used; current KMD public API is deterministic",
    )

    init_started = time.time()
    init_done = threading.Event()

    def heartbeat() -> None:
        while not init_done.wait(30):
            log_event(log_path, "initialize_progress", elapsed_seconds=round(time.time() - init_started, 3))

    log_event(log_path, "initialize_start")
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        kmd.initialize(raw_folder)
    finally:
        init_done.set()
    init_elapsed = round(time.time() - init_started, 3)
    log_event(log_path, "initialize_done", elapsed_seconds=init_elapsed)

    prediction_rows: list[dict[str, Any]] = []
    retrieved_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    started = time.time()
    total = len(questions)
    for index, row in enumerate(questions, start=1):
        question_id = row["question_id"]
        question_text = row["question"]
        if question_id in completed_ids:
            continue
        question_started = time.time()
        log_event(log_path, "question_start", index=index, total=total, question_id=question_id)
        public_answer = kmd.question(question_text)
        internal_answer = getattr(getattr(kmd_public, "_ENGINE", None), "last_answer", None)
        model_trace = getattr(getattr(kmd_public, "_ENGINE", None), "model_query_trace", None)
        bounded_diagnostics = getattr(getattr(kmd_public, "_ENGINE", None), "last_bounded_diagnostics", None)
        evidence_items = []
        for evidence in getattr(internal_answer, "evidence", []) or []:
            evidence_items.append(
                {
                    "source_id": evidence.rel_path,
                    "chunk_id": evidence.rel_path,
                    "text": evidence.text,
                    "score": evidence.score,
                }
            )
        elapsed = round(time.time() - question_started, 3)
        serialized_answer = serialize_answer(public_answer)
        is_answered = bool(serialized_answer)
        if is_answered:
            answered_count += 1

        retrieved_row = {
            "question_id": question_id,
            "question": question_text,
            "source_ids": [item["source_id"] for item in evidence_items],
            "chunk_ids": [item["chunk_id"] for item in evidence_items],
            "candidate_entities": [public_answer] if is_answered else [],
            "top_score": max([item["score"] for item in evidence_items] or ([1.0] if is_answered else [0.0])),
        }
        evidence_row = {
            "question_id": question_id,
            "question": question_text,
            "question_type": "unknown_to_kmd_adapter",
            "answerable": is_answered,
            "system_variant": "knowmoredirt_public_raw_folder",
            "allowed_product_ids": [],
            "exact_matches": [],
            "candidate_entities": [public_answer] if is_answered else [],
            "retrieved_chunks": evidence_items,
            "graph_facts": [{"public_api": "initialize(folder_path); question(text) -> string"}],
            "temporal_facts": [],
        }
        prediction_row = {
            "question_id": question_id,
            "answer": serialized_answer,
            "answerable": is_answered,
            "confidence": 1.0 if is_answered else 0.0,
            "supporting_source_ids": [item["source_id"] for item in evidence_items],
            "supporting_chunk_ids": [item["chunk_id"] for item in evidence_items],
            "reasoning_summary": "KnowMoreDiRT public raw-folder answer serialized without gold labels or source conversion.",
        }
        checkpoint_row = {
            "index": index,
            "total": total,
            "question_id": question_id,
            "question": question_text,
            "public_answer": public_answer,
            "serialized_answer": serialized_answer,
            "answered": is_answered,
            "evidence_count": len(evidence_items),
            "model_query_trace": model_trace.as_dict() if model_trace and args.use_local_model else None,
            "bounded_diagnostics": bounded_diagnostics,
            "elapsed_seconds": elapsed,
        }

        append_jsonl(run_dir / "retrieved_sources.jsonl", retrieved_row)
        append_jsonl(run_dir / "evidence_packets.jsonl", evidence_row)
        append_jsonl(run_dir / "predictions.jsonl", prediction_row)
        append_jsonl(checkpoint_path, checkpoint_row)
        prediction_rows.append(prediction_row)
        retrieved_rows.append(retrieved_row)
        evidence_rows.append(evidence_row)
        log_event(
            log_path,
            "question_done",
            index=index,
            total=total,
            percent=round((index / total) * 100, 3) if total else 100.0,
            question_id=question_id,
            answered=is_answered,
            answered_count=answered_count,
            evidence_count=len(evidence_items),
            bounded_record_counts=(
                bounded_diagnostics.get("execution", {}).get("record_counts", {})
                if isinstance(bounded_diagnostics, dict)
                else {}
            ),
            elapsed_seconds=elapsed,
        )

    query_elapsed = round(time.time() - started, 3)
    completed_after = len(completed_ids)
    if checkpoint_path.exists():
        seen_after: set[str] = set()
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                checkpoint = json.loads(line)
                question_id = str(checkpoint.get("question_id") or "")
                if question_id:
                    seen_after.add(question_id)
        completed_after = len(seen_after)
    log_event(log_path, "query_done", completed=completed_after, answered_count=answered_count, elapsed_seconds=query_elapsed)

    if len(questions) != len(all_questions) or completed_after != len(all_questions):
        reason = "limit was set" if len(questions) != len(all_questions) else "run incomplete"
        log_event(log_path, "scorer_skipped", reason=reason, completed=completed_after, full_count=len(all_questions))
        scores: dict[str, Any] = {"runtime_failure": True, "error": f"scorer skipped because {reason}"}
    else:
        os.environ.setdefault("HERB_BENCHMARK_SOURCE_ROOT", str(herb_root))
        os.environ.setdefault("HERB_BENCHMARK_VAR_ROOT", str(var_root))
        os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:14829/v1")
        get_settings.cache_clear()
        settings = get_settings()
        log_event(log_path, "scorer_start", scorer="herb_kgqa.evaluator.evaluate_run", use_local_judge=False)
        scores = evaluate_run(run_dir, settings=settings, use_local_judge=False)
        log_event(log_path, "scorer_done", scores_path=str(run_dir / "scores.json"))

    report = {
        "status": "completed" if not scores.get("runtime_failure") else "failed",
        "kmd_commit": os.popen(f"cd {repo_root} && HOME=/root git rev-parse HEAD").read().strip(),
        "raw_herb_source_folder": str(raw_folder),
        "questions_count": len(questions),
        "official_questions_count": len(all_questions),
        "query_completed": completed_after == len(all_questions),
        "completed_question_count": completed_after,
        "completed_percent": round((completed_after / len(all_questions)) * 100, 3) if all_questions else 0.0,
        "answered_count": answered_count,
        "deterministic_model_status": "optional localhost migrated DRT model-query planner enabled" if args.use_local_model else "deterministic KMD public API; no LLM calls made",
        "scorer": "herb_kgqa.evaluator.evaluate_run(use_local_judge=False)",
        "run_dir": str(run_dir),
        "progress_log": str(log_path),
        "checkpoint": str(checkpoint_path),
        "predictions": str(run_dir / "predictions.jsonl"),
        "retrieved_sources": str(run_dir / "retrieved_sources.jsonl"),
        "evidence_packets": str(run_dir / "evidence_packets.jsonl"),
        "scores_path": str(run_dir / "scores.json"),
        "scores": scores,
        "model_query_trace": (
            getattr(getattr(kmd_public, "_ENGINE", None), "model_query_trace", None).as_dict()
            if getattr(getattr(kmd_public, "_ENGINE", None), "model_query_trace", None)
            else None
        ),
        "no_gold_use_audit": {
            "query_input_fields": ["question_id", "question"],
            "gold_answers_used_for_query": False,
            "answerability_labels_used_for_query": False,
            "official_question_type_used_for_query": False,
            "prepared_corpus_used": False,
            "metadata_wrappers_used": False,
            "raw_folder_only": True,
            "local_model_enabled": bool(args.use_local_model),
        },
    }
    report_root.mkdir(parents=True, exist_ok=True)
    report_json = report_root / f"{args.run_name}.json"
    report_md = report_root / f"{args.run_name}.md"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    score_lines = json.dumps(scores, ensure_ascii=False, indent=2, sort_keys=True)
    report_md.write_text(
        "\n".join(
            [
                "# KnowMoreDiRT HERB Raw-Folder Public Run",
                "",
                f"- Status: `{report['status']}`",
                f"- KMD commit: `{report['kmd_commit']}`",
                f"- Raw HERB source folder: `{raw_folder}`",
                f"- Questions: `{len(questions)}`",
                f"- Completed: `{report['completed_question_count']}/{report['official_questions_count']}`",
                f"- Answered count: `{answered_count}`",
                f"- Deterministic/model status: `{report['deterministic_model_status']}`",
                f"- Scorer: `{report['scorer']}`",
                f"- Run directory: `{run_dir}`",
                f"- Progress log: `{log_path}`",
                "",
                "## No-Gold Query Audit",
                "",
                "- Query input contained only `question_id` and `question`.",
                "- Gold answers, answerability labels, question type labels, citations, and scores were not used for querying.",
                "- KMD source input was the raw HERB folder only; no prepared DRT corpus or metadata wrapper was used.",
                "",
                "## Scores",
                "",
                "```json",
                score_lines,
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md), "run_dir": str(run_dir), "scores": scores}, indent=2))
    return 0 if not scores.get("runtime_failure") else 1


if __name__ == "__main__":
    raise SystemExit(main())
