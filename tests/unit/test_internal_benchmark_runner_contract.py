from __future__ import annotations

import importlib.util
import os
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "run_internal_model_benchmark.py"


def _runner_module():
    spec = importlib.util.spec_from_file_location("kmd_internal_benchmark_runner_contract", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_does_not_hardcode_model_identity(monkeypatch, tmp_path: Path) -> None:
    module = _runner_module()
    before = dict(os.environ)
    try:
        os.environ.pop("KMD_LOCAL_MODEL_EXPECTED_ID", None)
        os.environ.pop("KMD_LOCAL_MODEL_ID", None)
        module._configure_environment(tmp_path)
        assert "KMD_LOCAL_MODEL_EXPECTED_ID" not in os.environ
        assert "KMD_LOCAL_MODEL_ID" not in os.environ
    finally:
        os.environ.clear()
        os.environ.update(before)


def test_runner_tracks_central_cache_namespaces_and_new_semantic_settings() -> None:
    module = _runner_module()
    assert "KMD_MODEL_CALL_CACHE_DIR" in module.CACHE_ENV_VARS
    assert "KMD_QUERY_EXPANSION_CACHE_DIR" in module.CACHE_ENV_VARS
    assert "KMD_RRF_K" in module.MODEL_ENV_KEYS
    assert "KMD_QUERY_EXPANSION_MAX_TERMS" not in module.MODEL_ENV_KEYS
    assert "KMD_DOCUMENT_CONTEXT_COVERAGE_RATIO" in module.MODEL_ENV_KEYS
    assert "KMD_MODEL_CALL_CACHE_DIR" not in module.MODEL_ENV_KEYS


def test_runner_cache_stats_count_sharded_raw_call_entries(monkeypatch, tmp_path: Path) -> None:
    module = _runner_module()
    root = tmp_path / "model_call"
    (root / "ab").mkdir(parents=True)
    (root / "ab" / "request.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("KMD_MODEL_CALL_CACHE_DIR", str(root))
    monkeypatch.setattr(module, "CACHE_ENV_VARS", ("KMD_MODEL_CALL_CACHE_DIR",))
    stats = module._cache_stats()["KMD_MODEL_CALL_CACHE_DIR"]
    assert stats["path"] == str(root)
    assert stats["json_files"] == 1
