#!/usr/bin/env python3
"""Run the internal KMD fixture benchmark through the public API with a local model.

This runner intentionally calls only ``initialize(folder_path)`` and
``question(text)`` for benchmark answers. It persists each completed answer so a
long model-backed run can resume without losing successful work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/data/src/github/devtests/kmd_model_benchmark")
CACHE_ENV_VARS = (
    "KMD_FRAME_CACHE_DIR",
    "KMD_CHUNK_DRS_CACHE_DIR",
    "KMD_QUERY_PLAN_CACHE_DIR",
    "KMD_QUERY_DRS_CACHE_DIR",
    "KMD_QUERY_EVIDENCE_REPAIR_CACHE_DIR",
    "KMD_QUERY_EVIDENCE_CACHE_DIR",
    "KMD_EVIDENCE_ANSWER_CACHE_DIR",
    "KMD_VERIFIER_CACHE_DIR",
    "KMD_ANSWER_CANONICALIZATION_CACHE_DIR",
    "KMD_IDENTITY_CACHE_DIR",
)
MODEL_ENV_KEYS = (
    "KMD_LOCAL_MODEL_ENDPOINT",
    "KMD_LOCAL_MODEL_TIMEOUT",
    "KMD_CHUNK_MODEL_TIMEOUT_SECONDS",
    "KMD_QUESTION_MODEL_TIMEOUT_SECONDS",
    "KMD_LOCAL_MODEL_API",
    "KMD_LOCAL_MODEL_STREAM",
    "KMD_LOCAL_MODEL_CACHE_PROMPT",
    "KMD_LOCAL_MODEL_JSON_SCHEMA",
    "KMD_LOCAL_MODEL_GRAMMAR",
    "KMD_LOCAL_MODEL_SEED",
    "KMD_LOCAL_MODEL_TEMPERATURE",
    "KMD_LOCAL_MODEL_TOP_P",
    "KMD_LOCAL_MODEL_TOP_K",
    "KMD_LOCAL_MODEL_MIN_P",
    "KMD_LOCAL_MODEL_REPEAT_PENALTY",
    "KMD_VERIFIER_DISCOURSE_FRAME_LIMIT",
    "KMD_LLM_DRS_INGEST",
    "KMD_LLM_INGEST",
    "KMD_QUERY_DRS_PLAN",
    "KMD_LAZY_LLM_FRAMES",
    "KMD_CHUNK_DRS_COMPACT_FIRST",
    "KMD_QUERY_DRS_COMPACT_FIRST",
)
SUITES = {
    "broad_raw_world": {
        "corpus": REPO_ROOT / "tests" / "fixtures" / "broad_raw_world",
        "qa": REPO_ROOT / "tests" / "fixtures" / "broad_raw_world_qa.json",
    },
    "hardcore_noise": {
        "corpus": REPO_ROOT / "tests" / "fixtures" / "hardcore_noise",
        "qa": REPO_ROOT / "tests" / "fixtures" / "hardcore_noise_qa.json",
    },
    "hard_raw_reasoning": {
        "corpus": REPO_ROOT / "tests" / "fixtures" / "hard_raw_reasoning",
        "qa": REPO_ROOT / "tests" / "fixtures" / "hard_raw_reasoning_qa.json",
    },
    "messy_raw_corpus": {
        "corpus": REPO_ROOT / "tests" / "fixtures" / "messy_raw_corpus",
        "qa": REPO_ROOT / "tests" / "fixtures" / "messy_raw_corpus_qa.json",
    },
}


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel_path = path.relative_to(root).as_posix()
        digest.update(rel_path.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _fetch_json(url: str, timeout: float = 8.0) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _endpoint_root(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    for suffix in (
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/models",
        "/models",
        "/completion",
        "/v1",
    ):
        if value.endswith(suffix):
            return value[: -len(suffix)] or value
    return value


def _git_revision() -> dict[str, str]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return ""

    return {
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def _configure_environment(output_root: Path) -> None:
    os.environ.setdefault("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:14829/v1")
    os.environ.setdefault("KMD_LOCAL_MODEL_TIMEOUT", "240")
    os.environ.setdefault("KMD_CHUNK_MODEL_TIMEOUT_SECONDS", "420")
    os.environ.setdefault("KMD_QUESTION_MODEL_TIMEOUT_SECONDS", "420")
    os.environ.setdefault("KMD_LOCAL_MODEL_API", "chat")
    os.environ.setdefault("KMD_LOCAL_MODEL_STREAM", "1")
    os.environ.setdefault("KMD_LOCAL_MODEL_CACHE_PROMPT", "1")
    os.environ.setdefault("KMD_LOCAL_MODEL_JSON_SCHEMA", "1")
    os.environ.setdefault("KMD_LOCAL_MODEL_GRAMMAR", "1")
    os.environ.setdefault("KMD_LOCAL_MODEL_SEED", "1778779265")
    os.environ.setdefault("KMD_LOCAL_MODEL_TEMPERATURE", "0.0")
    os.environ.setdefault("KMD_LOCAL_MODEL_TOP_P", "1.0")
    os.environ.setdefault("KMD_VERIFIER_DISCOURSE_FRAME_LIMIT", "0")
    os.environ.setdefault("KMD_LLM_DRS_INGEST", "1")
    os.environ.setdefault("KMD_QUERY_DRS_PLAN", "1")
    os.environ.setdefault("KMD_CHUNK_DRS_COMPACT_FIRST", "1")
    os.environ.setdefault("KMD_QUERY_DRS_COMPACT_FIRST", "1")
    os.environ.setdefault("KMD_TEST_ALLOW_NO_MODEL", "0")
    os.environ.setdefault("KMD_PROGRESS", "1")
    os.environ.setdefault("KMD_EVAL_PROGRESS", "1")
    for name in CACHE_ENV_VARS:
        cache_name = name.lower()
        if cache_name.startswith("kmd_"):
            cache_name = cache_name[4:]
        if cache_name.endswith("_dir"):
            cache_name = cache_name[:-4]
        os.environ.setdefault(name, str(output_root / "caches" / cache_name))
    for name in CACHE_ENV_VARS:
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


def _cache_stats() -> dict[str, dict[str, int | str]]:
    stats: dict[str, dict[str, int | str]] = {}
    for name in CACHE_ENV_VARS:
        root = Path(os.environ.get(name, ""))
        files = list(root.glob("*.json")) if root.exists() else []
        stats[name] = {
            "path": str(root),
            "json_files": len(files),
            "bytes": sum(path.stat().st_size for path in files if path.exists()),
        }
    return stats


def _load_existing_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return existing
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        suite = str(record.get("suite") or "")
        question_id = str(record.get("id") or "")
        if suite and question_id:
            existing[(suite, question_id)] = record
    return existing


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
        handle.flush()


def _engine_trace() -> dict[str, Any]:
    try:
        from knowmoredirt import public

        engine = getattr(public, "_ENGINE", None)
        trace = getattr(engine, "model_query_trace", None)
        payload = trace.as_dict() if trace is not None and hasattr(trace, "as_dict") else {}
        counts = engine.dspg_counts() if engine is not None and hasattr(engine, "dspg_counts") else {}
        integrity = engine.dspg_integrity() if engine is not None and hasattr(engine, "dspg_integrity") else ""
        return {"trace": payload, "dspg_counts": counts, "dspg_integrity": integrity}
    except Exception as exc:
        return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}


def _score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[bool]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    for record in records:
        correct = bool(record.get("correct"))
        category = str(record.get("category") or "")
        by_category[category].append(correct)
        if not correct:
            failures.append(
                {
                    "id": record.get("id"),
                    "category": category,
                    "question": record.get("question"),
                    "expected": record.get("expected"),
                    "predicted": record.get("predicted"),
                }
            )
    correct_total = sum(1 for record in records if record.get("correct"))
    total = len(records)
    return {
        "total": total,
        "correct": correct_total,
        "score": (correct_total / total) if total else 0.0,
        "by_category": {
            category: {
                "total": len(values),
                "correct": sum(1 for value in values if value),
                "score": (sum(1 for value in values if value) / len(values)) if values else 0.0,
            }
            for category, values in sorted(by_category.items())
        },
        "failures": failures,
    }


def _load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"{path} does not contain a questions list")
    return [item for item in questions if isinstance(item, dict)]


def _selected_suites(names: list[str]) -> list[str]:
    if not names or names == ["all"]:
        return list(SUITES)
    unknown = [name for name in names if name not in SUITES]
    if unknown:
        raise ValueError(f"unknown suite(s): {', '.join(unknown)}")
    return names


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _configure_environment(output_root)
    from knowmoredirt import initialize as kmd_initialize
    from knowmoredirt import question as kmd_question
    from knowmoredirt.evaluation import answer_matches

    results_path = output_root / "results.jsonl"
    summary_path = output_root / "summary.json"
    selected = _selected_suites(args.suite)
    if args.force:
        for path in (results_path, summary_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    existing = {} if args.force else _load_existing_results(results_path)
    started = time.time()
    endpoint = os.environ["KMD_LOCAL_MODEL_ENDPOINT"].rstrip("/")
    root = _endpoint_root(endpoint)
    metadata = {
        "endpoint": endpoint,
        "models": _fetch_json(root + "/v1/models"),
        "props": _fetch_json(root + "/props"),
        "slots": _fetch_json(root + "/slots"),
    }
    run_metadata: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "repo": _git_revision(),
        "model": metadata,
        "env": {key: os.environ.get(key, "") for key in MODEL_ENV_KEYS},
        "cache_dirs": {key: os.environ.get(key, "") for key in CACHE_ENV_VARS},
        "cache_stats_before": _cache_stats(),
        "suites": {},
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

    completed_records: list[dict[str, Any]] = []
    for suite_name in selected:
        suite = SUITES[suite_name]
        corpus = Path(suite["corpus"])
        qa_path = Path(suite["qa"])
        questions = _load_questions(qa_path)
        selected_question_ids = {str(value) for value in getattr(args, "question_id", []) or []}
        if selected_question_ids:
            questions = [item for item in questions if str(item.get("id") or "") in selected_question_ids]
        suite_started = time.time()
        print(
            f"kmd-model-benchmark suite_start {suite_name} "
            f"questions={len(questions)} corpus={corpus}",
            flush=True,
        )
        suite_records: list[dict[str, Any]] = [
            existing[(suite_name, str(item.get("id") or ""))]
            for item in questions
            if (suite_name, str(item.get("id") or "")) in existing
        ]
        missing_questions = [
            item for item in questions if (suite_name, str(item.get("id") or "")) not in existing
        ]
        initialized = False
        if missing_questions:
            init_started = time.time()
            kmd_initialize(corpus)
            initialized = True
            init_seconds = round(time.time() - init_started, 3)
            init_diagnostics = _engine_trace()
            print(
                f"kmd-model-benchmark suite_initialized {suite_name} "
                f"seconds={init_seconds:.3f} pending_questions={len(missing_questions)}",
                flush=True,
            )
        else:
            init_seconds = 0.0
            init_diagnostics = {"resumed_complete": True}

        for index, item in enumerate(questions, start=1):
            question_id = str(item.get("id") or "")
            key = (suite_name, question_id)
            if key in existing:
                print(
                    f"kmd-model-benchmark answer_cached {suite_name} "
                    f"{index}/{len(questions)} id={question_id}",
                    flush=True,
                )
                continue
            question_text = str(item.get("question") or "")
            expected = str(item.get("answer") or "")
            answer_started = time.time()
            predicted = kmd_question(question_text)
            elapsed = round(time.time() - answer_started, 3)
            correct = answer_matches(predicted, expected)
            diagnostics = _engine_trace()
            record = {
                "suite": suite_name,
                "id": question_id,
                "category": str(item.get("category") or ""),
                "question": question_text,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "elapsed_seconds": elapsed,
                "answered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_trace": diagnostics.get("trace", {}),
                "dspg_counts": diagnostics.get("dspg_counts", {}),
            }
            _append_jsonl(results_path, record)
            existing[key] = record
            suite_records.append(record)
            print(
                f"kmd-model-benchmark answer_done {suite_name} "
                f"{index}/{len(questions)} id={question_id} correct={correct} "
                f"predicted={predicted!r} seconds={elapsed:.3f}",
                flush=True,
            )

        ordered_suite_records = [
            existing[(suite_name, str(item.get("id") or ""))]
            for item in questions
            if (suite_name, str(item.get("id") or "")) in existing
        ]
        suite_score = _score_records(ordered_suite_records)
        suite_score.update(
            {
                "corpus": str(corpus),
                "qa": str(qa_path),
                "qa_hash": _sha256_file(qa_path),
                "corpus_tree_hash": _tree_hash(corpus),
                "wall_time_seconds": round(time.time() - suite_started, 3),
                "initialize_seconds": init_seconds,
                "initialized_this_run": initialized,
                "initialize_diagnostics": init_diagnostics,
            }
        )
        run_metadata["suites"][suite_name] = suite_score
        completed_records.extend(ordered_suite_records)
        summary = {
            **run_metadata,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_time_seconds": round(time.time() - started, 3),
            "cache_stats_after": _cache_stats(),
            "overall": _score_records(completed_records),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        print(
            f"kmd-model-benchmark suite_done {suite_name} "
            f"score={suite_score['correct']}/{suite_score['total']} "
            f"percent={suite_score['score'] * 100:.2f}",
            flush=True,
        )

    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        nargs="+",
        default=["all"],
        help="Suite names to run, or all.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for resumable results, metadata, logs, and model caches.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore existing results.jsonl entries.")
    parser.add_argument(
        "--question-id",
        nargs="*",
        default=[],
        help="Optional question ids to run within the selected suite(s).",
    )
    args = parser.parse_args()
    summary = run_benchmark(args)
    overall = summary["overall"]
    print(
        f"kmd-model-benchmark overall score={overall['correct']}/{overall['total']} "
        f"percent={overall['score'] * 100:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
