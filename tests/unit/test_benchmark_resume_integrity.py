from __future__ import annotations

import importlib.util
import json
import os
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def internal_runner() -> ModuleType:
    return _load(ROOT / "scripts" / "benchmarks" / "run_internal_model_benchmark.py", "internal_resume_test")


@pytest.fixture
def herb_runner() -> ModuleType:
    return _load(ROOT / "scripts" / "benchmarks" / "run_herb_kmd_raw_folder.py", "herb_resume_test")


def test_internal_resume_requires_exact_manifest(internal_runner: ModuleType, tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    manifest_path = tmp_path / "run_compatibility.json"
    manifest = {"schema": "v", "source": "one"}
    results.write_text('{"suite":"s","id":"q"}\n', encoding="utf-8")
    internal_runner._atomic_write_json(manifest_path, manifest)

    digest = internal_runner._prepare_resume_manifest(
        manifest_path,
        results,
        manifest,
        force=False,
    )
    assert digest == internal_runner._manifest_digest(manifest)

    with pytest.raises(RuntimeError, match="code, data, model, policy, or selection changed"):
        internal_runner._prepare_resume_manifest(
            manifest_path,
            results,
            {"schema": "v", "source": "changed"},
            force=False,
        )


def test_internal_resume_rejects_old_question_or_answer(internal_runner: ModuleType) -> None:
    cached = {
        "suite": "s",
        "id": "q1",
        "category": "c",
        "question": "Old question?",
        "expected": "OLD",
        "run_compatibility_sha256": "digest",
    }
    current = {"id": "q1", "category": "c", "question": "New question?", "answer": "NEW"}

    with pytest.raises(RuntimeError, match="cached row is incompatible"):
        internal_runner._validate_existing_record(
            cached,
            suite_name="s",
            item=current,
            manifest_digest="digest",
        )


def test_internal_jsonl_append_is_locked_and_durable(internal_runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"

    def writer(prefix: int) -> None:
        for index in range(40):
            internal_runner._append_jsonl(
                path,
                {"suite": "s", "id": f"{prefix}-{index}", "value": "x" * 1000},
            )

    threads = [threading.Thread(target=writer, args=(prefix,)) for prefix in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 240
    assert len({row["id"] for row in rows}) == 240


def test_internal_loader_rejects_malformed_and_duplicate_rows(internal_runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text('{"suite":"s","id":"q"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="malformed benchmark JSONL"):
        internal_runner._load_existing_results(path)

    path.write_text(
        '{"suite":"s","id":"q"}\n{"suite":"s","id":"q"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate benchmark result"):
        internal_runner._load_existing_results(path)


def test_herb_reconcile_removes_torn_question_and_deduplicates_complete_rows(
    herb_runner: ModuleType,
    tmp_path: Path,
) -> None:
    complete = {
        "retrieved_sources.jsonl": {"question_id": "complete", "source_ids": []},
        "evidence_packets.jsonl": {"question_id": "complete", "retrieved_chunks": []},
        "predictions.jsonl": {"question_id": "complete", "answer": "A"},
        "kmd_public_answers.jsonl": {"question_id": "complete", "answered": True},
    }
    for name, row in complete.items():
        herb_runner.append_jsonl(tmp_path / name, row)
        herb_runner.append_jsonl(tmp_path / name, {**row, "latest": True})
    for name in ("retrieved_sources.jsonl", "evidence_packets.jsonl", "predictions.jsonl"):
        herb_runner.append_jsonl(tmp_path / name, {"question_id": "torn", "partial": True})

    completed, answered = herb_runner._reconcile_resume_outputs(tmp_path)

    assert completed == {"complete"}
    assert answered == 1
    for name in complete:
        rows = herb_runner._load_jsonl_strict(tmp_path / name)
        assert rows == [{**complete[name], "latest": True}]


def test_herb_resume_requires_exact_manifest(herb_runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "run_compatibility.json"
    herb_runner._prepare_herb_resume_manifest(path, {"schema": "one"}, resume=False)

    herb_runner._prepare_herb_resume_manifest(path, {"schema": "one"}, resume=True)
    with pytest.raises(RuntimeError, match="code, data, model configuration, or question selection changed"):
        herb_runner._prepare_herb_resume_manifest(path, {"schema": "two"}, resume=True)


def test_internal_filesystem_catalog_cache_reuses_only_identical_inputs(
    internal_runner: ModuleType, tmp_path: Path, monkeypatch
) -> None:
    import knowmoredirt.filesystem as filesystem

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("alpha", encoding="utf-8")
    cache_root = tmp_path / "catalog-cache"
    monkeypatch.setattr(internal_runner, "FILESYSTEM_CATALOG_CACHE_ROOT", cache_root)
    monkeypatch.setattr(internal_runner, "_filesystem_catalog_source_hashes", lambda: {"policy": "same"})
    calls: list[Path] = []

    def fake_initialize(folder, database, **kwargs):
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sqlite-placeholder")
        calls.append(path)
        return {"status": "ok"}

    monkeypatch.setattr(filesystem, "initialize_filesystem_database", fake_initialize)
    first, first_result, _ = internal_runner._prepare_shared_filesystem_catalog(corpus, "suite")
    second, second_result, _ = internal_runner._prepare_shared_filesystem_catalog(corpus, "suite")
    assert first == second
    assert len(calls) == 1
    assert first_result["reused_existing"] is False
    assert second_result["reused_existing"] is True

    (corpus / "a.txt").write_text("changed", encoding="utf-8")
    third, third_result, _ = internal_runner._prepare_shared_filesystem_catalog(corpus, "suite")
    assert third != first
    assert len(calls) == 2
    assert third_result["reused_existing"] is False


def test_internal_manifest_tracks_vector_and_transport_policy(
    internal_runner: ModuleType, tmp_path: Path, monkeypatch
) -> None:
    corpus = tmp_path / "corpus"; corpus.mkdir(); (corpus / "a.txt").write_text("alpha", encoding="utf-8")
    qa = tmp_path / "qa.json"; qa.write_text('{"questions":[{"id":"q","question":"Q?","answer":"A","category":"c"}]}', encoding="utf-8")
    monkeypatch.setattr(internal_runner, "SUITES", {"s": {"corpus": corpus, "qa": qa}})
    monkeypatch.setattr(internal_runner, "_source_policy_hashes", lambda: {"policy": "hash"})
    monkeypatch.setattr(internal_runner, "_filesystem_catalog_source_hashes", lambda: {"filesystem": "hash"})
    monkeypatch.setattr(internal_runner, "_git_revision", lambda: {"commit": "c", "status_short": ""})
    args = type("Args", (), {"corpus_override": None, "question_id": [], "stop_on_failure": False})()
    metadata = {"endpoint": "http://127.0.0.1:14829/v1", "models": {}, "props": {}}
    monkeypatch.setenv("KMD_VECTOR_MIN_SIMILARITY", "0.50")
    first = internal_runner._build_run_compatibility_manifest(["s"], args, metadata)
    monkeypatch.setenv("KMD_VECTOR_MIN_SIMILARITY", "0.60")
    second = internal_runner._build_run_compatibility_manifest(["s"], args, metadata)
    assert first != second
    assert "KMD_LOCAL_MODEL_STREAM_BYTES_PER_TOKEN" not in first["model_env"]
    assert first["model_env"]["KMD_VECTOR_MIN_SIMILARITY"]["value"] == "0.50"
    assert second["model_env"]["KMD_VECTOR_MIN_SIMILARITY"]["value"] == "0.60"
    assert first["model_env"]["KMD_VECTOR_MIN_SIMILARITY"]["source"] == "environment"


def test_internal_policy_hashes_cover_model_vector_and_filesystem_paths(internal_runner: ModuleType) -> None:
    hashes = internal_runner._source_policy_hashes()
    for key in (
        "runtime_config",
        "default_config_xml",
        "runtime_logging",
        "model",
        "ingest",
        "vector_retrieval",
        "filesystem_facade",
        "filesystem_content_pipeline",
        "filesystem_content_schema",
    ):
        assert key in hashes


def test_herb_normalizes_embedding_request_url_to_server_base(herb_runner: ModuleType, monkeypatch) -> None:
    monkeypatch.setenv("KMD_EMBEDDING_ENDPOINT", "http://127.0.0.1:18139/v1/embeddings")
    assert herb_runner._normalize_embedding_base_url() == "http://127.0.0.1:18139"
    assert os.environ["KMD_EMBEDDING_ENDPOINT"] == "http://127.0.0.1:18139"


def test_herb_keeps_embedding_server_base_unchanged(herb_runner: ModuleType, monkeypatch) -> None:
    monkeypatch.setenv("KMD_EMBEDDING_ENDPOINT", "http://127.0.0.1:18139")
    assert herb_runner._normalize_embedding_base_url() == "http://127.0.0.1:18139"


def test_herb_pins_kmd_model_identity_from_llm_model(herb_runner: ModuleType, monkeypatch) -> None:
    model = "/models/live-120b.gguf"
    monkeypatch.setenv("LLM_MODEL", model)
    # Seed empty values through monkeypatch so direct os.environ assignments
    # performed by the helper are guaranteed to be removed at teardown.
    monkeypatch.setenv("KMD_LOCAL_MODEL_NAME", "")
    monkeypatch.setenv("KMD_LOCAL_MODEL_ID", "")
    monkeypatch.setenv("KMD_LOCAL_MODEL_EXPECTED_ID", "")
    assert herb_runner._pin_kmd_model_identity() == model
    assert os.environ["KMD_LOCAL_MODEL_NAME"] == model
    assert os.environ["KMD_LOCAL_MODEL_ID"] == model
    assert os.environ["KMD_LOCAL_MODEL_EXPECTED_ID"] == model


def test_herb_rejects_conflicting_kmd_model_identity(herb_runner: ModuleType, monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "/models/live-120b.gguf")
    monkeypatch.setenv("KMD_LOCAL_MODEL_NAME", "/models/old-27b.gguf")
    try:
        herb_runner._pin_kmd_model_identity()
    except RuntimeError as error:
        assert "conflicting HERB/KMD model identity" in str(error)
    else:
        raise AssertionError("expected conflicting model identity failure")


def test_herb_read_jsonl_for_prepared_artifact_index(herb_runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "artifact_index.jsonl"
    path.write_text('{"artifact_id":"a","source_file":"x.txt"}\n', encoding="utf-8")
    assert herb_runner.read_jsonl(path) == [{"artifact_id": "a", "source_file": "x.txt"}]


def test_herb_qualified_unknown_serializes_as_unanswerable(herb_runner: ModuleType) -> None:
    assert herb_runner.serialize_answer("unknown") == ""
    assert herb_runner.serialize_answer("unknown — relevant dreamed evidence in story.txt: blue lamps") == ""
    assert herb_runner.serialize_answer("UNKNOWN: only reported evidence exists") == ""
    assert herb_runner.serialize_answer("RPT-91") == ["RPT-91"]


def test_herb_resume_manifest_tracks_filesystem_catalog(herb_runner: ModuleType, tmp_path: Path) -> None:
    repo = tmp_path / "repo"; (repo / "src").mkdir(parents=True); (repo / "src" / "x.py").write_text("x"); (repo / "src" / "context_capacity.py").write_text("capacity")
    herb = tmp_path / "herb"; (herb / "src" / "herb_kgqa").mkdir(parents=True); (herb / "src" / "herb_kgqa" / "evaluator.py").write_text("e")
    raw = tmp_path / "raw"; raw.mkdir(); (raw / "a.txt").write_text("a")
    questions = tmp_path / "questions.jsonl"; questions.write_text('{"question_id":"q","question":"Q?"}\n')
    prepared = tmp_path / "manifest.json"; prepared.write_text('{}')
    args = type("Args", (), {"use_local_model": True, "limit": 0, "question_id": []})()
    manifest = herb_runner._herb_resume_manifest(
        repo_root=repo, herb_root=herb, raw_folder=raw, questions_path=questions,
        prepared_manifest=prepared, questions=[{"question_id":"q","question":"Q?"}], args=args,
        filesystem_catalog_manifest={"cache_key":"abc"},
    )
    assert manifest["filesystem_catalog_manifest"] == {"cache_key":"abc"}
    assert manifest["schema"] == "kmd-herb-resume-v3"


def test_internal_force_removes_stale_failure_artifacts(internal_runner: ModuleType, tmp_path: Path) -> None:
    failures = tmp_path / "failures" / "suite"
    failures.mkdir(parents=True)
    stale = failures / "old.json"; stale.write_text("{}")
    results = tmp_path / "results.jsonl"; results.write_text("old")
    summary = tmp_path / "summary.json"; summary.write_text("old")
    internal_runner._clear_forced_run_outputs(tmp_path, results, summary)
    assert not stale.exists()
    assert not results.exists()
    assert not summary.exists()


def test_herb_resume_manifest_ignores_scorer_only_changes(herb_runner: ModuleType, tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"; (repo / "src").mkdir(parents=True); (repo / "src" / "x.py").write_text("x"); (repo / "src" / "context_capacity.py").write_text("capacity")
    herb = tmp_path / "herb"; (herb / "src" / "herb_kgqa").mkdir(parents=True); (herb / "src" / "herb_kgqa" / "evaluator.py").write_text("e1")
    raw = tmp_path / "raw"; raw.mkdir(); (raw / "a.txt").write_text("a")
    questions = tmp_path / "questions.jsonl"; questions.write_text('{"question_id":"q","question":"Q?"}\n')
    prepared = tmp_path / "manifest.json"; prepared.write_text('{}')
    args = type("Args", (), {"use_local_model": True, "limit": 0, "question_id": []})()
    monkeypatch.setenv("KMD_EVALUATION_USE_LOCAL_JUDGE", "0")
    first = herb_runner._herb_resume_manifest(
        repo_root=repo, herb_root=herb, raw_folder=raw, questions_path=questions,
        prepared_manifest=prepared, questions=[{"question_id":"q","question":"Q?"}], args=args,
        filesystem_catalog_manifest={"cache_key":"abc"},
    )
    (herb / "src" / "herb_kgqa" / "evaluator.py").write_text("e2")
    monkeypatch.setenv("KMD_EVALUATION_USE_LOCAL_JUDGE", "1")
    second = herb_runner._herb_resume_manifest(
        repo_root=repo, herb_root=herb, raw_folder=raw, questions_path=questions,
        prepared_manifest=prepared, questions=[{"question_id":"q","question":"Q?"}], args=args,
        filesystem_catalog_manifest={"cache_key":"abc"},
    )
    assert first == second


def test_herb_answer_source_hashes_exclude_scorer_only_evaluation(herb_runner: ModuleType, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "src" / "knowmoredirt"; source.mkdir(parents=True)
    (repo / "src" / "context_capacity.py").write_text("capacity")
    (source / "engine.py").write_text("engine")
    (source / "evaluation.py").write_text("score-v1")
    first = herb_runner._herb_answer_source_hashes(repo)
    (source / "evaluation.py").write_text("score-v2")
    second = herb_runner._herb_answer_source_hashes(repo)
    assert first == second
    (source / "engine.py").write_text("engine-v2")
    assert herb_runner._herb_answer_source_hashes(repo) != first


def test_internal_benchmark_continues_on_failure_by_default(internal_runner: ModuleType) -> None:
    default_args = type("Args", (), {})()
    explicit_continue_args = type("Args", (), {"continue_on_failure": True})()
    stop_args = type("Args", (), {"stop_on_failure": True})()
    assert internal_runner._stop_on_failure(default_args) is False
    assert internal_runner._stop_on_failure(explicit_continue_args) is False
    assert internal_runner._stop_on_failure(stop_args) is True


def test_internal_manifest_tracks_user_xml_config(
    internal_runner: ModuleType, tmp_path: Path, monkeypatch
) -> None:
    corpus = tmp_path / "corpus"; corpus.mkdir(); (corpus / "a.txt").write_text("alpha", encoding="utf-8")
    qa = tmp_path / "qa.json"; qa.write_text('{"questions":[{"id":"q","question":"Q?","answer":"A","category":"c"}]}', encoding="utf-8")
    monkeypatch.setattr(internal_runner, "SUITES", {"s": {"corpus": corpus, "qa": qa}})
    monkeypatch.setattr(internal_runner, "_source_policy_hashes", lambda: {"policy": "hash"})
    monkeypatch.setattr(internal_runner, "_filesystem_catalog_source_hashes", lambda: {"filesystem": "hash"})
    monkeypatch.setattr(internal_runner, "_git_revision", lambda: {"commit": "c", "status_short": ""})
    args = type("Args", (), {"corpus_override": None, "question_id": [], "stop_on_failure": False})()
    metadata = {"endpoint": "http://127.0.0.1:14829/v1", "models": {}, "props": {}}
    config_path = tmp_path / "kmd.xml"
    config_path.write_text(
        '<knowmoredirt-config version="1"><settings>'
        '<setting name="KMD_VECTOR_MIN_SIMILARITY" value="0.61" />'
        '</settings></knowmoredirt-config>',
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_CONFIG_FILE", str(config_path))
    import kmd_runtime_config as runtime_config
    runtime_config._USER_CACHE = None
    first = internal_runner._build_run_compatibility_manifest(["s"], args, metadata)
    config_path.write_text(
        '<knowmoredirt-config version="1"><settings>'
        '<setting name="KMD_VECTOR_MIN_SIMILARITY" value="0.62" />'
        '</settings></knowmoredirt-config>',
        encoding="utf-8",
    )
    runtime_config._USER_CACHE = None
    second = internal_runner._build_run_compatibility_manifest(["s"], args, metadata)
    assert first != second
    assert first["runtime_config"]["user_hash"] != second["runtime_config"]["user_hash"]
