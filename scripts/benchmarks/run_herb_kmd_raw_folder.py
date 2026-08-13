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
import atexit
import faulthandler
import fcntl
import hashlib
import json
import os
import re
import signal
import sys
import threading
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


DEFAULT_PREPARED_ROOT = Path("/data/var/herb_benchmark/prepared/kmd_official_rag_v1")
DEFAULT_RAW_FOLDER = DEFAULT_PREPARED_ROOT / "source"
DEFAULT_QUESTIONS_FILE = DEFAULT_PREPARED_ROOT / "questions.jsonl"
DEFAULT_PREPARED_MANIFEST = DEFAULT_PREPARED_ROOT / "manifest.json"
DEFAULT_HERB_ROOT = Path("/data/src/github/devtests/herb_benchmark")
DEFAULT_VAR_ROOT = Path("/data/var/herb_benchmark")
DEFAULT_KMD_REPORT_ROOT = Path("/data/var/knowmoredirt/reports")
DEFAULT_KMD_RUN_ROOT = Path("/data/var/knowmoredirt/herb_runs")
HERB_RESUME_SCHEMA = "kmd-herb-resume-v3"
HERB_ANSWER_ENV_KEYS = (
    "KMD_USE_LOCAL_MODEL",
    "KMD_LOCAL_MODEL_ENDPOINT",
    "KMD_LOCAL_MODEL_NAME",
    "KMD_LOCAL_MODEL_ID",
    "KMD_LOCAL_MODEL_EXPECTED_ID",
    "KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS",
    "KMD_LOCAL_MODEL_API",
    "KMD_LOCAL_MODEL_CONSTRAINT_MODE",
    "KMD_LOCAL_MODEL_CACHE_PROMPT",
    "KMD_LOCAL_MODEL_JSON_SCHEMA",
    "KMD_LOCAL_MODEL_GRAMMAR",
    "KMD_LOCAL_MODEL_STREAM_BYTES_PER_TOKEN",
    "KMD_LOCAL_MODEL_STREAM_EVENT_MULTIPLIER",
    "KMD_LOCAL_MODEL_STREAM_TOTAL_TIMEOUT_SECONDS",
    "KMD_LOCAL_MODEL_SEND_THINKING_CONTROLS",
    "KMD_LOCAL_MODEL_SEED",
    "KMD_LOCAL_MODEL_TEMPERATURE",
    "KMD_LOCAL_MODEL_TOP_P",
    "KMD_LOCAL_MODEL_TOP_K",
    "KMD_LOCAL_MODEL_MIN_P",
    "KMD_LOCAL_MODEL_REPEAT_PENALTY",
    "KMD_LLM_DRS_INGEST",
    "KMD_QUERY_DRS_PLAN",
    "KMD_CHUNK_DRS_COMPACT_FIRST",
    "KMD_QUERY_DRS_COMPACT_FIRST",
    "KMD_LAZY_LLM_FRAMES",
    "KMD_DRS_RETRY_FAILED_ATTEMPTS",
    "KMD_SCAN_UNIT_MAX_CHARS",
    "KMD_SCAN_PACK_UNITS",
    "KMD_SCAN_PACK_MAX_CHARS",
    "KMD_SCAN_PACK_MAX_UNITS",
    "KMD_VECTOR_RETRIEVAL_MODE",
    "KMD_VECTOR_MIN_SIMILARITY",
    "KMD_VECTOR_RESULT_MULTIPLIER",
    "KMD_DOCUMENT_CONTEXT_ENVELOPES",
    "KMD_DOCUMENT_CONTEXT_MIN_CONFIDENCE",
    "KMD_DOCUMENT_CONTEXT_BOUNDARY_RATIO",
    "KMD_EMBEDDING_MODEL",
    "KMD_EMBEDDING_REVISION",
    "KMD_FILESYSTEM_CHUNK_INPUT_RATIO",
    "KMD_FILESYSTEM_CHUNK_CHARS_PER_TOKEN",
    "KMD_FILESYSTEM_CHUNK_TARGET_RATIO",
    "KMD_FILESYSTEM_CHUNK_OVERLAP_RATIO",
)


_RUN_STATE: dict[str, Any] = {"stage": "startup"}
_TERMINATION_LOGGED = False


def _normalize_embedding_base_url() -> str:
    """Normalize KMD_EMBEDDING_ENDPOINT to the server base URL.

    KMD's EmbeddingClient appends /v1/models and /v1/embeddings itself.  A full
    request endpoint here would produce invalid doubled paths.
    """

    value = os.environ.get("KMD_EMBEDDING_ENDPOINT", "").strip().rstrip("/")
    if not value:
        return value
    for suffix in ("/v1/embeddings", "/embeddings", "/v1"):
        if value.endswith(suffix):
            value = value[: -len(suffix)].rstrip("/")
            break
    if not value:
        raise RuntimeError("KMD_EMBEDDING_ENDPOINT normalized to an empty base URL")
    os.environ["KMD_EMBEDDING_ENDPOINT"] = value
    return value


def _pin_kmd_model_identity() -> str:
    """Pin every KMD model-identity setting to the benchmark judge/model id.

    HERB uses one local model endpoint for KMD semantic analysis and for the
    evaluator-model substitution.  Refuse conflicting explicit identities so a
    run cannot mix models across subsystems or reuse a cache under the wrong id.
    """

    model_id = os.environ.get("LLM_MODEL", "").strip()
    if not model_id:
        model_id = os.environ.get("KMD_LOCAL_MODEL_NAME", "").strip()
    if not model_id:
        raise RuntimeError(
            "HERB local-model run requires LLM_MODEL or KMD_LOCAL_MODEL_NAME to pin the live model identity"
        )
    for key in ("KMD_LOCAL_MODEL_NAME", "KMD_LOCAL_MODEL_ID", "KMD_LOCAL_MODEL_EXPECTED_ID"):
        configured = os.environ.get(key, "").strip()
        if configured and configured != model_id:
            raise RuntimeError(
                f"conflicting HERB/KMD model identity: LLM_MODEL={model_id!r} {key}={configured!r}"
            )
        os.environ[key] = model_id
    return model_id


def configure_process_io() -> None:
    """Make benchmark logs useful even when stdout is redirected."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except Exception:
                pass
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        pass


def _flush_file(handle: Any) -> None:
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass


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


def _append_json_line(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            _flush_file(handle)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def log_event(log_path: Path, event: str, **payload: Any) -> None:
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(), "event": event, **payload}
    _append_json_line(log_path, row)
    print("kmd-herb " + json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)

def register_process_diagnostics(log_path: Path) -> None:
    def record_exit() -> None:
        global _TERMINATION_LOGGED
        if _TERMINATION_LOGGED:
            return
        _TERMINATION_LOGGED = True
        log_event(log_path, "process_exit", state=dict(_RUN_STATE))

    def record_signal(signum: int, _frame: Any) -> None:
        global _TERMINATION_LOGGED
        name = signal.Signals(signum).name if signum in {item.value for item in signal.Signals} else str(signum)
        _TERMINATION_LOGGED = True
        log_event(log_path, "process_signal", signal=signum, signal_name=name, state=dict(_RUN_STATE))
        raise SystemExit(128 + signum)

    atexit.register(record_exit)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(signum, record_signal)
        except (OSError, RuntimeError, ValueError):
            pass


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            _flush_file(handle)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    _append_json_line(path, row)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.is_symlink()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            _flush_file(handle)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


HERB_FILESYSTEM_CACHE_SCHEMA = "kmd-herb-filesystem-vector-cache-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _herb_answer_source_hashes(repo_root: Path) -> dict[str, str]:
    """Hash KMD source that can affect public answer outputs.

    Evaluation/scoring code is intentionally excluded so corrected scoring can
    be rerun against unchanged expensive KMD answers. Filesystem vector-builder
    source is covered separately by the catalog manifest embedded below.
    """

    source_root = repo_root / "src" / "knowmoredirt"
    files = {
        path.relative_to(repo_root).as_posix(): path
        for path in source_root.glob("*.py")
        if path.name != "evaluation.py"
    }
    files["src/context_capacity.py"] = repo_root / "src" / "context_capacity.py"
    return {name: _sha256_file(path) for name, path in sorted(files.items())}


def _filesystem_catalog_source_hashes(repo_root: Path) -> dict[str, str]:
    files = {
        "filesystem_facade": repo_root / "src" / "knowmoredirt" / "filesystem.py",
        "content_pipeline": repo_root / "src" / "file_system_catalog" / "content_pipeline.py",
        "content_schema": repo_root / "src" / "file_system_catalog" / "content_schema.py",
        "filesystem_schema": repo_root / "src" / "file_system_catalog" / "schema.py",
        "filesystem_scanner": repo_root / "src" / "file_system_catalog" / "scanner.py",
        "context_capacity": repo_root / "src" / "context_capacity.py",
    }
    return {name: _sha256_file(path) for name, path in files.items()}


def _herb_filesystem_catalog_manifest(repo_root: Path, raw_folder: Path) -> dict[str, Any]:
    from knowmoredirt.filesystem import FilesystemModelConfig

    config = FilesystemModelConfig.from_environment()
    env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("KMD_EMBEDDING_")
        or key.startswith("KMD_FILESYSTEM_CHUNK_")
        or key in {"KMD_LOCAL_MODEL_ENDPOINT", "KMD_LOCAL_MODEL_NAME"}
    }
    return {
        "schema": HERB_FILESYSTEM_CACHE_SCHEMA,
        "raw_folder": str(raw_folder.resolve()),
        "raw_tree_hash": _tree_hash(raw_folder),
        "analysis_url": config.analysis_url,
        "analysis_model": config.analysis_model,
        "embedding_url": config.embedding_url,
        "embedding_model": config.embedding_model,
        "embedding_revision": config.embedding_revision,
        "embedding_batch_size": config.embedding_batch_size,
        "embedding_max_batch_characters": config.embedding_max_batch_characters,
        "environment": env,
        "source_hashes": _filesystem_catalog_source_hashes(repo_root),
        "chunks_only": True,
    }


def _prepare_herb_filesystem_catalog(
    repo_root: Path,
    raw_folder: Path,
    var_root: Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    from knowmoredirt.filesystem import initialize_filesystem_database

    manifest = manifest or _herb_filesystem_catalog_manifest(repo_root, raw_folder)
    digest = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    cache_root = Path(
        os.environ.get(
            "KMD_HERB_FILESYSTEM_CACHE_ROOT",
            str(var_root / "kmd_filesystem_catalog_cache"),
        )
    )
    root = cache_root / digest
    database = root / "catalog.sqlite3"
    manifest_path = root / "manifest.json"
    if database.is_file() and manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if cached == manifest:
            return database, {"reused_existing": True, "cache_key": digest, "manifest": manifest}
    root.mkdir(parents=True, exist_ok=True)
    result = initialize_filesystem_database(
        raw_folder,
        database,
        replace=database.exists(),
        chunks_only=True,
        collection_id=f"herb-kmd:{digest[:16]}",
        progress_every=25,
    )
    _atomic_write_json(manifest_path, manifest)
    payload = dict(result) if isinstance(result, dict) else {"result": result}
    payload.update({"reused_existing": False, "cache_key": digest, "manifest": manifest})
    return database, payload


def _herb_resume_manifest(
    *,
    repo_root: Path,
    herb_root: Path,
    raw_folder: Path,
    questions_path: Path,
    prepared_manifest: Path,
    questions: list[dict[str, str]],
    args: argparse.Namespace,
    filesystem_catalog_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env = {key: os.environ.get(key, "") for key in HERB_ANSWER_ENV_KEYS}
    return {
        "schema": HERB_RESUME_SCHEMA,
        "kmd_answer_source_hashes": _herb_answer_source_hashes(repo_root),
        "runner_hash": _sha256_file(Path(__file__).resolve()),
        "raw_folder": str(raw_folder),
        "raw_tree_hash": _tree_hash(raw_folder),
        "questions_path": str(questions_path),
        "questions_hash": _sha256_file(questions_path),
        "prepared_manifest": str(prepared_manifest),
        "prepared_manifest_hash": _sha256_file(prepared_manifest),
        "filesystem_catalog_manifest": filesystem_catalog_manifest or {},
        "questions": questions,
        "use_local_model": bool(args.use_local_model),
        "limit": int(args.limit),
        "question_ids": sorted(str(value) for value in args.question_id or []),
        "environment": env,
    }


def _prepare_herb_resume_manifest(path: Path, manifest: dict[str, Any], *, resume: bool) -> None:
    if resume:
        if not path.exists():
            raise RuntimeError("refusing HERB resume: compatibility manifest is missing")
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"refusing HERB resume: invalid compatibility manifest: {error}") from error
        if previous != manifest:
            raise RuntimeError("refusing HERB resume: code, data, model configuration, or question selection changed")
    _atomic_write_json(path, manifest)


def _load_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"malformed HERB JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object HERB JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def _reconcile_resume_outputs(run_dir: Path) -> tuple[set[str], int]:
    names = (
        "retrieved_sources.jsonl",
        "evidence_packets.jsonl",
        "predictions.jsonl",
        "kmd_public_answers.jsonl",
    )
    latest: dict[str, dict[str, dict[str, Any]]] = {}
    order: list[str] = []
    for name in names:
        per_id: dict[str, dict[str, Any]] = {}
        for row in _load_jsonl_strict(run_dir / name):
            question_id = str(row.get("question_id") or "")
            if not question_id:
                raise RuntimeError(f"HERB row missing question_id in {name}")
            per_id[question_id] = row
            if question_id not in order:
                order.append(question_id)
        latest[name] = per_id
    complete = set.intersection(*(set(rows) for rows in latest.values())) if latest else set()
    canonical_order = [question_id for question_id in order if question_id in complete]
    for name in names:
        write_jsonl(run_dir / name, [latest[name][question_id] for question_id in canonical_order])
    checkpoints = latest["kmd_public_answers.jsonl"]
    answered = sum(bool(checkpoints[question_id].get("answered")) for question_id in complete)
    return complete, answered

def serialize_answer(public_answer: str) -> str | list[str]:
    answer = str(public_answer or "").strip()
    normalized = answer.lower()
    if not answer or normalized == "unknown" or re.match(r"^unknown(?:\s|$|[—–:;,-])", normalized):
        return ""
    values = re.findall(r"(?:https?://[^\s,;]+|[A-Z][A-Z0-9]{1,9}-\d+[A-Z0-9-]*|eid_[a-z0-9]+|CUST-\d+)", answer)
    if values and len(" ".join(values)) >= max(3, len(answer.strip()) - 4):
        return list(dict.fromkeys(value.rstrip(".") for value in values))
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description="Run KMD raw-folder public API on local HERB.")
    parser.add_argument("--raw-folder", default=str(DEFAULT_RAW_FOLDER))
    parser.add_argument("--herb-root", default=str(DEFAULT_HERB_ROOT))
    parser.add_argument("--questions-file", default=str(DEFAULT_QUESTIONS_FILE))
    parser.add_argument("--prepared-manifest", default=str(DEFAULT_PREPARED_MANIFEST))
    parser.add_argument("--var-root", default=str(DEFAULT_VAR_ROOT))
    parser.add_argument("--run-root", default=str(DEFAULT_KMD_RUN_ROOT))
    parser.add_argument("--report-root", default=str(DEFAULT_KMD_REPORT_ROOT))
    parser.add_argument("--run-name", default=f"kmd_public_raw_folder_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit; 0 means all questions.")
    parser.add_argument("--question-id", action="append", default=[], help="Run only the given HERB public question id; may be repeated.")
    parser.add_argument("--use-local-model", action="store_true", help="Enable KMD's optional localhost-only migrated DRT model-query planner.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run directory without deleting completed JSONL outputs.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    raw_folder = Path(args.raw_folder).resolve()
    herb_root = Path(args.herb_root).resolve()
    var_root = Path(args.var_root).resolve()
    run_dir = Path(args.run_root).resolve() / args.run_name
    report_root = Path(args.report_root).resolve()
    normalized_questions = Path(args.questions_file).resolve()
    prepared_manifest = Path(args.prepared_manifest).resolve()
    log_path = run_dir / "progress.jsonl"
    checkpoint_path = run_dir / "kmd_public_answers.jsonl"
    compatibility_path = run_dir / "run_compatibility.json"
    sanitized_questions_path = run_dir / "questions_sanitized_for_kmd.jsonl"

    configure_process_io()
    register_process_diagnostics(log_path)
    _RUN_STATE.update({"stage": "preflight", "run_dir": str(run_dir), "raw_folder": str(raw_folder)})
    log_event(log_path, "process_start", argv=sys.argv, run_dir=str(run_dir), raw_folder=str(raw_folder))

    if not raw_folder.is_dir():
        raise FileNotFoundError(raw_folder)
    if not normalized_questions.exists():
        raise FileNotFoundError(normalized_questions)
    if not prepared_manifest.is_file():
        raise FileNotFoundError(prepared_manifest)
    if not herb_root.is_dir():
        raise FileNotFoundError(herb_root)

    from prepare_herb_kmd_bundle import validate_prepared_bundle

    prepared = validate_prepared_bundle(prepared_manifest)
    artifact_index_path = prepared["root"] / str(prepared["manifest"]["artifact_index_file"])
    source_to_artifact = {
        str(row["source_file"]): str(row["artifact_id"])
        for row in read_jsonl(artifact_index_path)
    }
    if len(source_to_artifact) != int(prepared["manifest"]["artifact_count"]):
        raise ValueError("prepared HERB artifact/source map is incomplete")
    if prepared["source_root"].resolve() != raw_folder:
        raise ValueError(
            f"raw folder does not match prepared manifest: {raw_folder} != {prepared['source_root']}"
        )
    if prepared["questions_path"].resolve() != normalized_questions:
        raise ValueError(
            "questions file does not match prepared manifest: "
            f"{normalized_questions} != {prepared['questions_path']}"
        )

    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(herb_root / "src"))
    if args.use_local_model:
        os.environ["KMD_USE_LOCAL_MODEL"] = "1"
        _pin_kmd_model_identity()
        _normalize_embedding_base_url()
        from kmd_runtime_config import configure_model_cache_environment

        configure_model_cache_environment()
        os.environ.setdefault("KMD_LOCAL_MODEL_CACHE_PROMPT", "1")
        os.environ["KMD_SCAN_PACK_UNITS"] = "1"
        os.environ["KMD_SCAN_PACK_MAX_UNITS"] = "0"


    os.environ.setdefault("KMD_VECTOR_RETRIEVAL_MODE", "required")
    os.environ.setdefault("KMD_EVALUATION_USE_LOCAL_JUDGE", "1")
    os.environ.setdefault("KMD_DOCUMENT_CONTEXT_ENVELOPES", "1")

    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for output_name in [
            "retrieved_sources.jsonl",
            "evidence_packets.jsonl",
            "predictions.jsonl",
            "kmd_public_answers.jsonl",
            "run_compatibility.json",
            "scores.json",
            "question_details.jsonl",
            "error_report.md",
        ]:
            (run_dir / output_name).unlink(missing_ok=True)

    all_questions = read_official_questions(normalized_questions)
    selected_ids = {str(value) for value in args.question_id or []}
    if selected_ids:
        all_questions = [row for row in all_questions if str(row.get("question_id") or "") in selected_ids]
    questions = all_questions[: args.limit] if args.limit else all_questions

    # Computing the catalog manifest is cheap and does not contact embedding/model
    # endpoints. Validate resume compatibility before any expensive vector build.
    filesystem_catalog_manifest = _herb_filesystem_catalog_manifest(repo_root, raw_folder)
    compatibility_manifest = _herb_resume_manifest(
        repo_root=repo_root,
        herb_root=herb_root,
        raw_folder=raw_folder,
        questions_path=normalized_questions,
        prepared_manifest=prepared_manifest,
        questions=questions,
        args=args,
        filesystem_catalog_manifest=filesystem_catalog_manifest,
    )
    _prepare_herb_resume_manifest(compatibility_path, compatibility_manifest, resume=bool(args.resume))

    filesystem_database, filesystem_catalog_result = _prepare_herb_filesystem_catalog(
        repo_root, raw_folder, var_root, manifest=filesystem_catalog_manifest
    )
    os.environ["KMD_FILESYSTEM_DATABASE"] = str(filesystem_database)

    import knowmoredirt as kmd  # noqa: WPS433 - operational benchmark adapter
    from knowmoredirt import public as kmd_public  # noqa: WPS433
    from herb_kgqa.config import get_settings  # noqa: WPS433
    from evaluate_herb_official_local import evaluate_run_official_local  # noqa: WPS433

    write_jsonl(sanitized_questions_path, questions)
    if args.resume:
        completed_ids, answered_count = _reconcile_resume_outputs(run_dir)
    else:
        completed_ids, answered_count = set(), 0
    _RUN_STATE.update({"stage": "run_start", "total_questions": len(questions)})
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
        model_status="localhost KMD model pipeline enabled" if args.use_local_model else "legacy no-model path",
        prepared_manifest=str(prepared_manifest),
        prepared_format=prepared["manifest"]["format"],
        prepared_artifact_count=prepared["manifest"]["artifact_count"],
        prepared_source_file_count=prepared["manifest"]["source_file_count"],
        prepared_source_tree_sha256=prepared["manifest"]["output_hashes"]["source_tree"],
        filesystem_database=str(filesystem_database),
        filesystem_catalog_reused=bool(filesystem_catalog_result.get("reused_existing")),
        filesystem_catalog_cache_key=str(filesystem_catalog_result.get("cache_key") or ""),
    )

    init_started = time.time()
    init_done = threading.Event()

    def heartbeat() -> None:
        while not init_done.wait(30):
            log_event(log_path, "initialize_progress", elapsed_seconds=round(time.time() - init_started, 3), state=dict(_RUN_STATE))

    _RUN_STATE.update({"stage": "initialize", "initialized": False})
    log_event(log_path, "initialize_start")
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        kmd.initialize(raw_folder)
    finally:
        init_done.set()
    init_elapsed = round(time.time() - init_started, 3)
    _RUN_STATE.update({"stage": "questions", "initialized": True})
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
        _RUN_STATE.update({"stage": "question", "question_index": index, "question_total": total, "question_id": question_id})
        log_event(
            log_path,
            "question_start",
            index=index,
            total=total,
            question_id=question_id,
            question=question_text,
            percent=round((index / total) * 100, 3) if total else 100.0,
        )
        public_answer = kmd.question(question_text)
        internal_answer = getattr(getattr(kmd_public, "_ENGINE", None), "last_answer", None)
        model_trace = getattr(getattr(kmd_public, "_ENGINE", None), "model_query_trace", None)
        bounded_diagnostics = getattr(getattr(kmd_public, "_ENGINE", None), "last_bounded_diagnostics", None)
        evidence_items = []
        for evidence in getattr(internal_answer, "evidence", []) or []:
            herb_source_id = source_to_artifact.get(evidence.rel_path, evidence.rel_path)
            evidence_items.append(
                {
                    "source_id": herb_source_id,
                    "source_rel_path": evidence.rel_path,
                    "chunk_id": (
                        evidence.span_id
                        or (f"{herb_source_id}#chunk-{evidence.chunk_order}" if evidence.chunk_order is not None else herb_source_id)
                    ),
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
    _RUN_STATE.update({"stage": "scoring", "completed_questions": completed_after, "question_total": len(all_questions)})
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
        log_event(log_path, "scorer_start", scorer="Salesforce HERB official evaluate.py semantics; localhost judge substitution", use_local_judge=True)
        scores = evaluate_run_official_local(run_dir, settings=settings)
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
        "deterministic_model_status": "localhost KMD model pipeline enabled" if args.use_local_model else "legacy no-model path",
        "scorer": "Salesforce HERB official evaluate.py semantics; get_gpt4_response replaced only by localhost judge",
        "run_dir": str(run_dir),
        "progress_log": str(log_path),
        "checkpoint": str(checkpoint_path),
        "predictions": str(run_dir / "predictions.jsonl"),
        "retrieved_sources": str(run_dir / "retrieved_sources.jsonl"),
        "evidence_packets": str(run_dir / "evidence_packets.jsonl"),
        "scores_path": str(run_dir / "scores.json"),
        "question_details_path": str(run_dir / "question_details.jsonl"),
        "filesystem_database": str(filesystem_database),
        "filesystem_catalog": filesystem_catalog_result,
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
            "prepared_corpus_used": True,
            "metadata_wrappers_used": False,
            "prepared_source_folder_only": True,
            "raw_hf_snapshot_ingested": False,
            "local_model_enabled": bool(args.use_local_model),
        },
    }
    report_root.mkdir(parents=True, exist_ok=True)
    report_json = report_root / f"{args.run_name}.json"
    report_md = report_root / f"{args.run_name}.md"
    _atomic_write_json(report_json, report)
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
                "- KMD source input was the validated leakage-free prepared HERB source folder; the mixed raw HF snapshot was not ingested.",
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
    log_event(log_path, "report_written", report_json=str(report_json), report_md=str(report_md), status=report["status"])
    _RUN_STATE.update({"stage": "complete", "status": report["status"]})
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md), "run_dir": str(run_dir), "scores": scores}, indent=2), flush=True)
    return 0 if not scores.get("runtime_failure") else 1


if __name__ == "__main__":
    raise SystemExit(main())
