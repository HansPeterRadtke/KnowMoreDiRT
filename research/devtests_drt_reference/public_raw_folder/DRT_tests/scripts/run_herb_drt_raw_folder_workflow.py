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


DRT_ROOT = Path(__file__).resolve().parents[1]
HERB_ROOT = Path("/data/src/github/devtests/herb_benchmark")
DEFAULT_RAW_FOLDER = Path("/data/var/herb_benchmark/raw/herb_raw/hf_snapshot")
DEFAULT_RUNTIME_ROOT = Path("/data/var/herb_benchmark/drt_raw_folder")


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str] | None, log_path: Path) -> dict[str, Any]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"event": "command_start", "cmd": cmd, "cwd": str(cwd)}, ensure_ascii=False) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        elapsed = round(time.time() - started, 3)
        log.write(json.dumps({"event": "command_done", "returncode": proc.returncode, "elapsed_seconds": elapsed}, ensure_ascii=False) + "\n")
    result = {
        "cmd": cmd,
        "cwd": str(cwd),
        "log": str(log_path),
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
    }
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}; see {log_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DRT/HERB using direct raw-folder DRT ingestion.")
    parser.add_argument("--raw-folder", default=str(DEFAULT_RAW_FOLDER), help="Existing raw source folder to ingest as text.")
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT), help="Runtime output root for DB/query/scorer artifacts.")
    parser.add_argument("--questions-jsonl", default=None, help="Sanitized DRT questions JSONL for scorer/query execution.")
    parser.add_argument("--question-map", default=None, help="Adapter DRT-to-HERB ID map for scorer output only.")
    parser.add_argument("--run-name", default=None, help="HERB run name when scorer is enabled.")
    parser.add_argument("--variant", default="all_model_assisted", help="DRT ingestion variant.")
    parser.add_argument("--bounded-doc-limit", type=int, default=None, help="Bounded query candidate document limit.")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    parser.add_argument("--skip-scorer", action="store_true")
    args = parser.parse_args()

    raw_folder = Path(args.raw_folder).resolve()
    if not raw_folder.exists() or not raw_folder.is_dir():
        raise FileNotFoundError(raw_folder)
    runtime_root = Path(args.runtime_root).resolve()
    questions_jsonl = Path(args.questions_jsonl).resolve() if args.questions_jsonl else None
    question_map = Path(args.question_map).resolve() if args.question_map else None
    if (not args.skip_query or not args.skip_scorer) and (questions_jsonl is None or not questions_jsonl.exists()):
        raise FileNotFoundError(questions_jsonl or "missing --questions-jsonl")
    if not args.skip_scorer and (question_map is None or not question_map.exists()):
        raise FileNotFoundError(question_map or "missing --question-map")

    run_id = args.run_name or f"drt_raw_folder_{time.strftime('%Y%m%d_%H%M%S')}"
    dspg_root = runtime_root / "dspg"
    log_root = runtime_root / "logs" / run_id
    db_path = dspg_root / "herb_drt.sqlite"
    ingest_report = dspg_root / "herb_drt_ingest_report.json"
    query_output = dspg_root / "herb_drt_query_results.json"
    progress_log = dspg_root / "herb_drt_query_progress.jsonl"
    checkpoint_jsonl = dspg_root / "herb_drt_query_checkpoint.jsonl"
    workflow_report = runtime_root / "raw_folder_workflow_report.json"
    runtime_root.mkdir(parents=True, exist_ok=True)
    dspg_root.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, Any]] = []
    if not args.skip_ingest:
        commands.append(
            run_command(
                [
                    sys.executable,
                    str(DRT_ROOT / "scripts" / "dspg_ingest_folder.py"),
                    "--input-folder",
                    str(raw_folder),
                    "--config",
                    str(DRT_ROOT / "config" / "dspg_system.yaml"),
                    "--db",
                    str(db_path),
                    "--variant",
                    args.variant,
                    "--run-id",
                    run_id,
                    "--report",
                    str(ingest_report),
                ],
                cwd=DRT_ROOT,
                env=os.environ.copy(),
                log_path=log_root / "ingest.log",
            )
        )
    if not args.skip_query:
        query_cmd = [
            sys.executable,
            str(DRT_ROOT / "scripts" / "dspg_query.py"),
            "--db",
            str(db_path),
            "--config",
            str(DRT_ROOT / "config" / "dspg_system.yaml"),
            "--questions-jsonl",
            str(questions_jsonl),
            "--use-model-query",
            "--output",
            str(query_output),
            "--progress-log",
            str(progress_log),
            "--checkpoint-jsonl",
            str(checkpoint_jsonl),
        ]
        if args.bounded_doc_limit is not None:
            query_cmd.extend(["--bounded-doc-limit", str(args.bounded_doc_limit)])
        commands.append(run_command(query_cmd, cwd=DRT_ROOT, env=os.environ.copy(), log_path=log_root / "query.log"))
    if not args.skip_scorer:
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": f"{HERB_ROOT / 'src'}:{env.get('PYTHONPATH', '')}",
                "DRT_ROOT": str(DRT_ROOT),
                "DRT_HERB_RUNTIME_ROOT": str(runtime_root),
                "DRT_HERB_QUESTIONS": str(questions_jsonl),
                "DRT_HERB_QUESTION_MAP": str(question_map),
                "DRT_HERB_DB": str(db_path),
                "DRT_HERB_QUERY_OUTPUT": str(query_output),
                "LLM_BASE_URL": env.get("LLM_BASE_URL", "http://127.0.0.1:14829/v1"),
            }
        )
        commands.append(
            run_command(
                [
                    "/data/venv/bin/python3",
                    "-m",
                    "herb_kgqa.run_pipeline",
                    "--system",
                    "drt_dspg",
                    "--questions",
                    "all",
                    "--run-name",
                    run_id,
                ],
                cwd=HERB_ROOT,
                env=env,
                log_path=log_root / "scorer.log",
            )
        )

    report = {
        "status": "completed",
        "run_id": run_id,
        "raw_folder": str(raw_folder),
        "runtime_root": str(runtime_root),
        "questions_jsonl": str(questions_jsonl) if questions_jsonl else None,
        "question_map": str(question_map) if question_map else None,
        "db_path": str(db_path),
        "ingest_report": str(ingest_report),
        "query_output": str(query_output),
        "progress_log": str(progress_log),
        "checkpoint_jsonl": str(checkpoint_jsonl),
        "skip_ingest": args.skip_ingest,
        "skip_query": args.skip_query,
        "skip_scorer": args.skip_scorer,
        "commands": commands,
        "input_contract": "existing raw folder; every readable file is treated as raw text by DRT ingestion",
        "adapter_glue": "sanitized question IDs and HERB ID mapping only; source files stay untouched",
    }
    workflow_report.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
