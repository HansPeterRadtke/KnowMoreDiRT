from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


def _runner_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "run_internal_model_benchmark.py"
    spec = importlib.util.spec_from_file_location("kmd_internal_runner_suite_env", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compatibility_manifest_records_suite_env_without_mutating_process_env(monkeypatch) -> None:
    runner = _runner_module()
    monkeypatch.setenv("KMD_SCAN_PACK_UNITS", "0")
    args = argparse.Namespace(question_id=[], corpus_override=None)
    model_metadata = {"endpoint": "http://127.0.0.1:14829/v1", "model_id": "test", "context_size": 65536}
    manifest = runner._build_run_compatibility_manifest(
        ["recording_context_requirements"],
        args,
        model_metadata,
    )
    assert os.environ["KMD_SCAN_PACK_UNITS"] == "0"
    assert manifest["suite_inputs"]["recording_context_requirements"]["suite_env"] == {
        "KMD_SCAN_PACK_UNITS": "0"
    }
