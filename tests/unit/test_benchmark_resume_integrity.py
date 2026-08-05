from __future__ import annotations

import importlib.util
import json
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
