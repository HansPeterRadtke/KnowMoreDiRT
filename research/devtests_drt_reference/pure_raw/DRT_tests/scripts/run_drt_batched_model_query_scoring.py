#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from merge_drt_query_batches import merge_batches


ROOT = Path(__file__).resolve().parents[1]
DEVTESTS_ROOT = ROOT.parent
HERB_ROOT = DEVTESTS_ROOT / "herb_benchmark"
PREP_ROOT = Path("/data/var/herb_benchmark/drt_prepared")
DEFAULT_BATCH_DIR = PREP_ROOT / "model_query_fix_batches100_20260517_141511"
DEFAULT_QUESTIONS = PREP_ROOT / "manifests" / "questions_for_drt.jsonl"
DEFAULT_MERGED = DEFAULT_BATCH_DIR / "merged_model_query_pure_raw_results.json"
DEFAULT_LOG_MD = ROOT / "logs" / "HERB_DRT_MODEL_QUERY_PURE_RAW_FINAL.md"
DEFAULT_LOG_JSON = ROOT / "logs" / "HERB_DRT_MODEL_QUERY_PURE_RAW_FINAL.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(DEVTESTS_ROOT),
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "HOME": os.environ.get("HOME", "/root")},
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def run_official_scorer(*, merged_output: Path, run_name: str, command_log: Path) -> tuple[int, Path, dict[str, Any]]:
    env = os.environ.copy()
    env.update(
        {
            "HERB_BENCHMARK_VAR_ROOT": "/data/var/herb_benchmark",
            "DRT_HERB_PREP_ROOT": str(PREP_ROOT),
            "DRT_HERB_QUERY_OUTPUT": str(merged_output),
            "DRT_ROOT": str(ROOT),
            "LLM_BASE_URL": "http://127.0.0.1:14829/v1",
            "PYTHONPATH": f"{HERB_ROOT / 'src'}:{env.get('PYTHONPATH', '')}",
        }
    )
    cmd = [
        "/data/venv/bin/python3",
        "-m",
        "herb_kgqa.run_pipeline",
        "--system",
        "drt_dspg",
        "--questions",
        "all",
        "--run-name",
        run_name,
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(HERB_ROOT), env=env, text=True, capture_output=True)
    run_dir = Path("/data/var/herb_benchmark/runs") / run_name
    command_record = {
        "cmd": cmd,
        "cwd": str(HERB_ROOT),
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "run_dir": str(run_dir),
        "env": {
            "DRT_HERB_PREP_ROOT": env["DRT_HERB_PREP_ROOT"],
            "DRT_HERB_QUERY_OUTPUT": env["DRT_HERB_QUERY_OUTPUT"],
            "LLM_BASE_URL": env["LLM_BASE_URL"],
        },
        "stdout_tail": proc.stdout[-12000:],
        "stderr_tail": proc.stderr[-12000:],
    }
    command_log.parent.mkdir(parents=True, exist_ok=True)
    command_log.write_text(json.dumps(command_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    scores_path = run_dir / "scores.json"
    scores = read_json(scores_path) if scores_path.exists() else {}
    return proc.returncode, run_dir, scores


def row_counts(run_dir: Path) -> dict[str, int]:
    return {
        "predictions": count_jsonl(run_dir / "predictions.jsonl"),
        "retrieved_sources": count_jsonl(run_dir / "retrieved_sources.jsonl"),
        "evidence_packets": count_jsonl(run_dir / "evidence_packets.jsonl"),
    }


def write_reports(report: dict[str, Any], md_path: Path, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    scores = report.get("scores", {})
    merge = report.get("merge", {})
    rows = report.get("official_artifact_counts", {})
    lines = [
        "# HERB DRT Model Query Pure Raw Final",
        "",
        "## Status",
        f"- status: `{report.get('status')}`",
        f"- git_commit: `{report.get('git_commit')}`",
        f"- run_name: `{report.get('run_name')}`",
        f"- run_dir: `{report.get('run_dir')}`",
        "",
        "## Merge Validation",
        f"- batch_dir: `{merge.get('batch_dir')}`",
        f"- partial_count: `{merge.get('partial_count')}`",
        f"- merged_count: `{merge.get('merged_count')}`",
        f"- unique_ids: `{merge.get('unique_ids')}`",
        f"- first_id: `{merge.get('first_id')}`",
        f"- last_id: `{merge.get('last_id')}`",
        f"- status_counts: `{merge.get('status_counts')}`",
        f"- use_model_query: `{report.get('use_model_query')}`",
        f"- mode: `{report.get('mode')}`",
        "",
        "## Official Scorer",
        f"- scorer_command_log: `{report.get('scorer_command_log')}`",
        f"- scores_path: `{report.get('scores_path')}`",
        f"- runtime_failure: `{scores.get('runtime_failure', False)}`",
        f"- artifact_counts: `{rows}`",
        "",
        "## Score Values",
    ]
    for key in [
        "answerable_accuracy",
        "deterministic_exact_match",
        "token_f1",
        "unanswerable_accuracy",
        "retrieval_recall",
        "source_citation_precision",
        "source_citation_recall",
    ]:
        lines.append(f"- {key}: `{scores.get(key)}`")
    lines.extend(
        [
            "",
            "## Comparability",
            "- `answerable_accuracy` is produced by the official local HERB scorer from `predictions.jsonl` and is comparable to other HERB answerable-accuracy runs.",
            "- The adapter was run after DRT query output existed and only mapped DRT question IDs back to HERB IDs for scoring.",
            "",
            "## No-Gold / No-Tuning Audit",
            "- DRT query input remained the sanitized `questions_for_drt.jsonl`.",
            "- The merge tool validates against generated DRT IDs only; it does not read gold answers, citations, answerability labels, or evaluator labels.",
            "- The wrapper passes the precomputed pure-raw model-query JSON to the official adapter/scorer and does not patch answers.",
            "",
            "## Caveats",
            "- Retrieval/citation recall remains weak.",
            "- Candidate routing still needs general improvement.",
            "- The older `38.53%` metadata-bridge result is separate from this pure-raw model-query run.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge completed DRT model-query batches and run official HERB scoring.")
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    parser.add_argument("--questions-jsonl", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--merged-output", default=str(DEFAULT_MERGED))
    parser.add_argument("--expected-count", type=int, default=1514)
    parser.add_argument("--run-name", default=f"drt_dspg_model_query_pure_raw_repro_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--report-md", default=str(DEFAULT_LOG_MD))
    parser.add_argument("--report-json", default=str(DEFAULT_LOG_JSON))
    parser.add_argument("--merge-report-json", default=str(ROOT / "logs" / "HERB_DRT_MODEL_QUERY_PURE_RAW_MERGE.json"))
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    questions_jsonl = Path(args.questions_jsonl)
    merged_output = Path(args.merged_output)
    merge_report_json = Path(args.merge_report_json)

    merge = merge_batches(
        batch_dir=batch_dir,
        questions_jsonl=questions_jsonl,
        output=merged_output,
        expected_count=args.expected_count,
        mode="pure_raw_model_query_batched",
        report_json=merge_report_json,
    )
    merged_payload = read_json(merged_output)
    if merged_payload.get("use_model_query") is not True or merged_payload.get("mode") != "pure_raw_model_query_batched":
        raise SystemExit("Refusing to score: merged output is not pure_raw_model_query_batched with use_model_query=true")

    command_log = ROOT / "logs" / "HERB_DRT_MODEL_QUERY_PURE_RAW_SCORER_COMMAND.json"
    returncode, run_dir, scores = run_official_scorer(merged_output=merged_output, run_name=args.run_name, command_log=command_log)
    counts = row_counts(run_dir)
    status = "completed" if returncode == 0 and not scores.get("runtime_failure") else "failed"
    report = {
        "status": status,
        "git_commit": git_commit(),
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "merged_output": str(merged_output),
        "merge_report_json": str(merge_report_json),
        "scorer_command_log": str(command_log),
        "scores_path": str(run_dir / "scores.json"),
        "scores": scores,
        "official_artifact_counts": counts,
        "merge": merge,
        "use_model_query": True,
        "mode": "pure_raw_model_query_batched",
    }
    write_reports(report, Path(args.report_md), Path(args.report_json))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
