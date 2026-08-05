#!/usr/bin/env python3
"""Run the internal KMD fixture benchmark through the public API with a local model.

This runner intentionally calls only ``initialize(folder_path)`` and
``question(text)`` for benchmark answers. It persists each completed answer so a
long model-backed run can resume without losing successful work.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/data/src/github/devtests/kmd_model_benchmark")
RUN_COMPATIBILITY_SCHEMA = "kmd-internal-benchmark-resume-v2"
CACHE_ENV_VARS = (
    "KMD_FRAME_CACHE_DIR",
    "KMD_CHUNK_FRAME_CACHE_DIR",
    "KMD_CHUNK_DRS_CACHE_DIR",
    "KMD_QUERY_PLAN_CACHE_DIR",
    "KMD_QUERY_DRS_CACHE_DIR",
    "KMD_QUERY_EVIDENCE_REPAIR_CACHE_DIR",
    "KMD_QUERY_EVIDENCE_CACHE_DIR",
    "KMD_EVIDENCE_ANSWER_CACHE_DIR",
    "KMD_VERIFIER_CACHE_DIR",
    "KMD_QUERY_VERIFIER_CACHE_DIR",
    "KMD_ANSWER_CANONICALIZATION_CACHE_DIR",
    "KMD_QUERY_CANONICAL_CACHE_DIR",
    "KMD_IDENTITY_CACHE_DIR",
    "KMD_IDENTITY_CANONICAL_CACHE_DIR",
    "KMD_SOURCE_RESOLUTION_CACHE_DIR",
)
MODEL_ENV_KEYS = (
    "KMD_LOCAL_MODEL_ENDPOINT",
    "KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS",
    "KMD_LOCAL_MODEL_API",
    "KMD_LOCAL_MODEL_CONSTRAINT_MODE",
    "KMD_LOCAL_MODEL_CACHE_PROMPT",
    "KMD_LOCAL_MODEL_JSON_SCHEMA",
    "KMD_LOCAL_MODEL_GRAMMAR",
    "KMD_LOCAL_MODEL_SEED",
    "KMD_LOCAL_MODEL_TEMPERATURE",
    "KMD_LOCAL_MODEL_TOP_P",
    "KMD_LOCAL_MODEL_TOP_K",
    "KMD_LOCAL_MODEL_MIN_P",
    "KMD_LOCAL_MODEL_REPEAT_PENALTY",
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
    "structured_record_json": {
        "corpus": REPO_ROOT / "tests" / "fixtures" / "structured_record_json",
        "qa": REPO_ROOT / "tests" / "fixtures" / "structured_record_json_qa.json",
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


def _fetch_json(url: str) -> Any:
    raw_timeout = os.environ.get("KMD_LOCAL_MODEL_CONTROL_TIMEOUT_SECONDS", "30").strip()
    try:
        timeout = float(raw_timeout)
    except ValueError:
        timeout = 30.0
    if timeout <= 0:
        timeout = 30.0
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
    os.environ.setdefault("KMD_LOCAL_MODEL_EXPECTED_ID", "Qwen3.5-27B-Q8_0.gguf")
    os.environ.setdefault("KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS", "420")
    os.environ.setdefault("KMD_LOCAL_MODEL_API", "chat")
    os.environ.setdefault("KMD_LOCAL_MODEL_CONSTRAINT_MODE", "native")
    os.environ.setdefault("KMD_LOCAL_MODEL_CACHE_PROMPT", "1")
    os.environ.setdefault("KMD_LOCAL_MODEL_JSON_SCHEMA", "1")
    os.environ.setdefault("KMD_LOCAL_MODEL_GRAMMAR", "1")
    os.environ.setdefault("KMD_LOCAL_MODEL_SEED", "1778779265")
    os.environ.setdefault("KMD_LOCAL_MODEL_TEMPERATURE", "0.0")
    os.environ.setdefault("KMD_LOCAL_MODEL_TOP_P", "1.0")
    os.environ.setdefault("KMD_LLM_DRS_INGEST", "1")
    os.environ.setdefault("KMD_QUERY_DRS_PLAN", "1")
    os.environ.setdefault("KMD_CHUNK_DRS_COMPACT_FIRST", "1")
    os.environ.setdefault("KMD_QUERY_DRS_COMPACT_FIRST", "1")
    os.environ.setdefault("KMD_TEST_ALLOW_NO_MODEL", "0")
    os.environ.setdefault("KMD_PROGRESS", "1")
    os.environ.setdefault("KMD_EVAL_PROGRESS", "1")
    shared_cache_root = Path(os.environ.get("KMD_SHARED_MODEL_CACHE_ROOT", str(output_root.parent / ".kmd_model_cache_shared")))
    for name in CACHE_ENV_VARS:
        cache_name = name.lower()
        if cache_name.startswith("kmd_"):
            cache_name = cache_name[4:]
        if cache_name.endswith("_dir"):
            cache_name = cache_name[:-4]
        os.environ.setdefault(name, str(shared_cache_root / cache_name))
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


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_existing_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"malformed benchmark JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise RuntimeError(f"non-object benchmark JSONL row at {path}:{line_number}")
            suite = str(record.get("suite") or "")
            question_id = str(record.get("id") or "")
            if not suite or not question_id:
                raise RuntimeError(f"benchmark row missing suite/id at {path}:{line_number}")
            key = (suite, question_id)
            if key in existing:
                raise RuntimeError(f"duplicate benchmark result row for {suite}/{question_id}")
            existing[key] = record
    return existing


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _manifest_digest(manifest: dict[str, Any]) -> str:
    material = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _source_policy_hashes() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "context_capacity": REPO_ROOT / "src" / "context_capacity.py",
        "model_planner": REPO_ROOT / "src" / "knowmoredirt" / "model_planner.py",
        "engine": REPO_ROOT / "src" / "knowmoredirt" / "engine.py",
        "bounded_dspg": REPO_ROOT / "src" / "knowmoredirt" / "bounded_dspg.py",
        "store": REPO_ROOT / "src" / "knowmoredirt" / "store.py",
        "scanner": REPO_ROOT / "src" / "knowmoredirt" / "scanner.py",
        "evaluation": REPO_ROOT / "src" / "knowmoredirt" / "evaluation.py",
    }
    return {name: _sha256_file(path) for name, path in files.items()}


def _build_run_compatibility_manifest(
    selected: list[str],
    args: argparse.Namespace,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    selected_question_ids = sorted(str(value) for value in getattr(args, "question_id", []) or [])
    suite_inputs: dict[str, Any] = {}
    for suite_name in selected:
        suite = SUITES[suite_name]
        corpus = Path(args.corpus_override) if getattr(args, "corpus_override", None) else Path(suite["corpus"])
        qa_path = Path(suite["qa"])
        questions = _load_questions(qa_path)
        if selected_question_ids:
            allowed = set(selected_question_ids)
            questions = [item for item in questions if str(item.get("id") or "") in allowed]
        suite_inputs[suite_name] = {
            "corpus": str(corpus.resolve()),
            "corpus_tree_hash": _tree_hash(corpus),
            "qa": str(qa_path.resolve()),
            "qa_hash": _sha256_file(qa_path),
            "questions": [
                {
                    "id": str(item.get("id") or ""),
                    "question": str(item.get("question") or ""),
                    "answer": str(item.get("answer") or ""),
                    "category": str(item.get("category") or ""),
                }
                for item in questions
            ],
        }
    repo = _git_revision()
    return {
        "schema": RUN_COMPATIBILITY_SCHEMA,
        "repo_commit": repo.get("commit", ""),
        "repo_dirty_status": repo.get("status_short", ""),
        "source_policy_hashes": _source_policy_hashes(),
        "model": {
            "endpoint": model_metadata.get("endpoint"),
            "models": model_metadata.get("models"),
            "props": model_metadata.get("props"),
        },
        "model_env": {key: os.environ.get(key, "") for key in MODEL_ENV_KEYS},
        "selected_suites": selected,
        "selected_question_ids": selected_question_ids,
        "continue_on_failure": bool(getattr(args, "continue_on_failure", False)),
        "suite_inputs": suite_inputs,
    }


def _prepare_resume_manifest(
    manifest_path: Path,
    results_path: Path,
    manifest: dict[str, Any],
    *,
    force: bool,
) -> str:
    digest = _manifest_digest(manifest)
    if force:
        manifest_path.unlink(missing_ok=True)
    elif results_path.exists():
        if not manifest_path.exists():
            raise RuntimeError("refusing benchmark resume: compatibility manifest is missing")
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"refusing benchmark resume: invalid compatibility manifest: {error}") from error
        if previous != manifest:
            raise RuntimeError("refusing benchmark resume: code, data, model, policy, or selection changed")
    _atomic_write_json(manifest_path, manifest)
    return digest


def _validate_existing_record(
    record: dict[str, Any],
    *,
    suite_name: str,
    item: dict[str, Any],
    manifest_digest: str,
) -> None:
    expected = {
        "suite": suite_name,
        "id": str(item.get("id") or ""),
        "category": str(item.get("category") or ""),
        "question": str(item.get("question") or ""),
        "expected": str(item.get("answer") or ""),
        "run_compatibility_sha256": manifest_digest,
    }
    mismatches = {
        key: {"cached": record.get(key), "current": value}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "refusing benchmark resume: cached row is incompatible: "
            + json.dumps(mismatches, sort_keys=True, default=_json_default)
        )

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



def _filesystem_catalog_diagnostics(database: Path) -> dict[str, Any]:
    if not database.exists():
        return {"exists": False, "path": str(database)}
    diagnostics: dict[str, Any] = {
        "exists": True,
        "path": str(database),
        "bytes": database.stat().st_size,
        "tables": {},
    }
    with sqlite3.connect(database) as connection:
        table_names = [
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        for table in table_names:
            safe = table.replace('"', '""')
            try:
                count = int(connection.execute(f'SELECT COUNT(*) FROM "{safe}"').fetchone()[0])
            except sqlite3.Error:
                continue
            diagnostics["tables"][table] = count
    return diagnostics


def _write_failure_artifact(output_root: Path, record: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    root = output_root / "failures" / str(record.get("suite") or "unknown")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{str(record.get('id') or 'unknown')}.json"
    payload = dict(record)
    payload["full_diagnostics"] = diagnostics
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return str(path)

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
    compatibility_path = output_root / "run_compatibility.json"
    selected = _selected_suites(args.suite)
    if args.force:
        for path in (results_path, summary_path, compatibility_path):
            path.unlink(missing_ok=True)
    started = time.time()
    endpoint = os.environ["KMD_LOCAL_MODEL_ENDPOINT"].rstrip("/")
    root = _endpoint_root(endpoint)
    metadata = {
        "endpoint": endpoint,
        "models": _fetch_json(root + "/v1/models"),
        "props": _fetch_json(root + "/props"),
        "slots": _fetch_json(root + "/slots"),
    }
    compatibility_manifest = _build_run_compatibility_manifest(selected, args, metadata)
    compatibility_digest = _prepare_resume_manifest(
        compatibility_path,
        results_path,
        compatibility_manifest,
        force=bool(args.force),
    )
    existing = {} if args.force else _load_existing_results(results_path)
    run_metadata: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "repo": _git_revision(),
        "model": metadata,
        "env": {key: os.environ.get(key, "") for key in MODEL_ENV_KEYS},
        "cache_dirs": {key: os.environ.get(key, "") for key in CACHE_ENV_VARS},
        "cache_stats_before": _cache_stats(),
        "run_compatibility_path": str(compatibility_path),
        "run_compatibility_sha256": compatibility_digest,
        "suites": {},
    }
    run_metadata_path = output_root / "run_metadata.json"
    _atomic_write_json(run_metadata_path, run_metadata)
    print(
        "kmd-model-benchmark run_start "
        f"output_root={output_root} results={results_path} summary={summary_path} "
        f"metadata={run_metadata_path} endpoint={endpoint}",
        flush=True,
    )

    completed_records: list[dict[str, Any]] = []
    for suite_name in selected:
        suite = SUITES[suite_name]
        corpus = Path(args.corpus_override) if getattr(args, "corpus_override", None) else Path(suite["corpus"])
        qa_path = Path(suite["qa"])
        questions = _load_questions(qa_path)
        selected_question_ids = {str(value) for value in getattr(args, "question_id", []) or []}
        if selected_question_ids:
            questions = [item for item in questions if str(item.get("id") or "") in selected_question_ids]
        for item in questions:
            key = (suite_name, str(item.get("id") or ""))
            if key in existing:
                _validate_existing_record(
                    existing[key],
                    suite_name=suite_name,
                    item=item,
                    manifest_digest=compatibility_digest,
                )
                if not existing[key].get("correct") and not getattr(args, "continue_on_failure", False):
                    raise RuntimeError(f"refusing benchmark resume after cached incorrect answer: {suite_name}/{key[1]}")
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
        filesystem_init_seconds = 0.0
        filesystem_init_result: dict[str, Any] = {}
        filesystem_database = output_root / "filesystem_catalogs" / f"{suite_name}.sqlite3"
        filesystem_diagnostics: dict[str, Any] = {}
        if missing_questions:
            from knowmoredirt.filesystem import initialize_filesystem_database

            filesystem_database.parent.mkdir(parents=True, exist_ok=True)
            if filesystem_database.exists() and not args.force:
                filesystem_init_result = {"reused_existing": True}
                filesystem_init_seconds = 0.0
            else:
                fs_started = time.time()
                filesystem_init_result = initialize_filesystem_database(
                    corpus,
                    filesystem_database,
                    replace=filesystem_database.exists(),
                    chunks_only=False,
                    collection_id=f"internal-benchmark:{suite_name}",
                    progress_every=25,
                )
                filesystem_init_seconds = round(time.time() - fs_started, 3)
            filesystem_diagnostics = _filesystem_catalog_diagnostics(filesystem_database)
            if not filesystem_diagnostics.get("exists") or not filesystem_diagnostics.get("tables"):
                raise RuntimeError(f"filesystem catalog initialization failed for {suite_name}: {filesystem_diagnostics}")
            print(
                f"kmd-model-benchmark filesystem_initialized {suite_name} "
                f"seconds={filesystem_init_seconds:.3f} database={filesystem_database} "
                f"tables={filesystem_diagnostics.get('tables', {})}",
                flush=True,
            )

            init_started = time.time()
            kmd_initialize(corpus)
            initialized = True
            init_seconds = round(time.time() - init_started, 3)
            init_diagnostics = _engine_trace()
            init_diagnostics["filesystem_database"] = str(filesystem_database)
            init_diagnostics["filesystem_init_seconds"] = filesystem_init_seconds
            init_diagnostics["filesystem_init_result"] = filesystem_init_result
            init_diagnostics["filesystem_diagnostics"] = filesystem_diagnostics
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
            print(
                f"kmd-model-benchmark answer_start {suite_name} "
                f"{index}/{len(questions)} id={question_id} "
                f"category={str(item.get('category') or '')!r} question={question_text!r}",
                flush=True,
            )
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
                "run_compatibility_sha256": compatibility_digest,
                "model_trace": diagnostics.get("trace", {}),
                "dspg_counts": diagnostics.get("dspg_counts", {}),
            }
            if not correct:
                record["failure_artifact"] = _write_failure_artifact(output_root, record, diagnostics)
            _append_jsonl(results_path, record)
            existing[key] = record
            suite_records.append(record)
            print(
                f"kmd-model-benchmark answer_done {suite_name} "
                f"{index}/{len(questions)} id={question_id} correct={correct} "
                f"predicted={predicted!r} seconds={elapsed:.3f}",
                flush=True,
            )
            if not correct and not getattr(args, "continue_on_failure", False):
                partial_summary = {
                    **run_metadata,
                    "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "stop_reason": "first_incorrect_answer",
                    "stopped_suite": suite_name,
                    "stopped_question_id": question_id,
                    "cache_stats_after": _cache_stats(),
                    "overall": _score_records(completed_records + suite_records),
                }
                _atomic_write_json(summary_path, partial_summary)
                raise RuntimeError(
                    f"benchmark stopped after first incorrect answer: suite={suite_name} id={question_id}"
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
                "filesystem_initialize_seconds": filesystem_init_seconds,
                "filesystem_database": str(filesystem_database),
                "filesystem_diagnostics": filesystem_diagnostics,
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
        _atomic_write_json(summary_path, summary)
        print(
            f"kmd-model-benchmark summary_written {suite_name} "
            f"summary={summary_path} results={results_path}",
            flush=True,
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
        "--continue-on-failure",
        action="store_true",
        help="Continue after an incorrect answer. The default is to stop after the first failure.",
    )
    parser.add_argument(
        "--question-id",
        nargs="*",
        default=[],
        help="Optional question ids to run within the selected suite(s).",
    )
    parser.add_argument(
        "--corpus-override",
        type=Path,
        default=None,
        help="Optional corpus path override for targeted integration validation.",
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
