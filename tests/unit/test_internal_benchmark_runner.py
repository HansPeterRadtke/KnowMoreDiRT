from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "run_internal_model_benchmark.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("internal_benchmark_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_output_root_lock_rejects_concurrent_runner(tmp_path):
    runner = load_runner()
    first = runner.acquire_output_lock(tmp_path)
    try:
        with pytest.raises(SystemExit, match="already locked"):
            runner.acquire_output_lock(tmp_path)
    finally:
        first.close()
    second = runner.acquire_output_lock(tmp_path)
    second.close()


def test_atomic_checkpoint_keeps_one_row_per_suite_question(tmp_path):
    runner = load_runner()
    path = tmp_path / "results.jsonl"
    rows = [
        {"suite": "s", "id": "q1", "predicted": "old"},
        {"suite": "s", "id": "q2", "predicted": "two"},
        {"suite": "s", "id": "q1", "predicted": "new"},
    ]
    runner.write_results_atomic(path, rows)
    loaded = runner.load_existing(path)
    assert len(path.read_text().splitlines()) == 2
    assert loaded[("s", "q1")]["predicted"] == "new"
    assert loaded[("s", "q2")]["predicted"] == "two"
