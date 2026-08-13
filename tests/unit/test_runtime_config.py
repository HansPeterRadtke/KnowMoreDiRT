from __future__ import annotations

import json
from pathlib import Path

import pytest

import kmd_runtime_config as config
from context_capacity import context_ratio
from knowmoredirt.semantic_cache import CACHE_VERSION, SemanticFrameCache
from knowmoredirt.model_planner import CHUNK_FRAME_SCHEMA_VERSION, PROMPT_VERSION


def _reset_config_caches() -> None:
    config._USER_CACHE = None


def _write_config(path: Path, *settings: tuple[str, str]) -> None:
    rows = "\n".join(
        f'    <setting name="{name}" value="{value}" />' for name, value in settings
    )
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<knowmoredirt-config version="1">\n'
        '  <settings>\n'
        f'{rows}\n'
        '  </settings>\n'
        '</knowmoredirt-config>\n',
        encoding="utf-8",
    )


def test_packaged_config_is_complete_and_metadata_rich() -> None:
    specs = config.default_specs()
    assert len(specs) >= 170
    assert config.DEFAULT_CONFIG_PATH.is_file()
    for name, spec in specs.items():
        assert name.startswith("KMD_")
        assert spec.group
        assert spec.risk in {"low", "medium", "high"}
        assert spec.change_frequency
        assert spec.description
    assert specs["KMD_LOCAL_MODEL_CONTROL_TIMEOUT_SECONDS"].unit == "seconds"
    assert specs["KMD_VECTOR_MIN_SIMILARITY"].minimum == -1
    assert specs["KMD_VECTOR_MIN_SIMILARITY"].maximum == 1


def test_config_precedence_environment_over_user_xml_over_packaged_default(tmp_path: Path, monkeypatch) -> None:
    user = tmp_path / "kmd.xml"
    _write_config(user, ("KMD_VECTOR_MIN_SIMILARITY", "0.61"))
    monkeypatch.setenv("KMD_CONFIG_FILE", str(user))
    monkeypatch.delenv("KMD_VECTOR_MIN_SIMILARITY", raising=False)
    _reset_config_caches()
    assert config.raw("KMD_VECTOR_MIN_SIMILARITY") == "0.61"
    assert config.source("KMD_VECTOR_MIN_SIMILARITY") == str(user)
    monkeypatch.setenv("KMD_VECTOR_MIN_SIMILARITY", "0.72")
    assert config.raw("KMD_VECTOR_MIN_SIMILARITY") == "0.72"
    assert config.source("KMD_VECTOR_MIN_SIMILARITY") == "environment"


def test_user_xml_range_validation_is_centralized(tmp_path: Path, monkeypatch) -> None:
    user = tmp_path / "bad.xml"
    _write_config(user, ("KMD_VECTOR_MIN_SIMILARITY", "2.0"))
    monkeypatch.setenv("KMD_CONFIG_FILE", str(user))
    _reset_config_caches()
    with pytest.raises(ValueError, match="KMD_VECTOR_MIN_SIMILARITY"):
        config.validate_all()


def test_packaged_default_is_not_an_explicit_override(monkeypatch) -> None:
    monkeypatch.delenv("KMD_CONFIG_FILE", raising=False)
    monkeypatch.delenv("KMD_LOCAL_MODEL_ID", raising=False)
    monkeypatch.delenv("KMD_LOCAL_MODEL_GRAMMAR", raising=False)
    _reset_config_caches()
    assert config.raw("KMD_LOCAL_MODEL_GRAMMAR") == "1"
    assert config.explicit_raw("KMD_LOCAL_MODEL_GRAMMAR") is None
    assert config.explicit_raw("KMD_LOCAL_MODEL_ID") is None


def test_context_capacity_uses_user_xml_without_environment_mutation(tmp_path: Path, monkeypatch) -> None:
    user = tmp_path / "context.xml"
    _write_config(user, ("KMD_DOCUMENT_CONTEXT_BOUNDARY_RATIO", "0.33"))
    monkeypatch.setenv("KMD_CONFIG_FILE", str(user))
    monkeypatch.delenv("KMD_DOCUMENT_CONTEXT_BOUNDARY_RATIO", raising=False)
    _reset_config_caches()
    assert context_ratio(("KMD_DOCUMENT_CONTEXT_BOUNDARY_RATIO",), 0.99) == pytest.approx(0.33)
    assert "KMD_DOCUMENT_CONTEXT_BOUNDARY_RATIO" not in __import__("os").environ


def test_semantic_cache_default_key_preserves_historical_explicit_override_shape(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("KMD_CONFIG_FILE", raising=False)
    for key in ("KMD_LOCAL_MODEL_ID", "KMD_LOCAL_MODEL_GRAMMAR"):
        monkeypatch.delenv(key, raising=False)
    _reset_config_caches()
    cache = SemanticFrameCache(tmp_path)
    text = "source text"
    context = {"context_size": 65536}
    expected_material = json.dumps(
        {
            "cache_version": CACHE_VERSION,
            "endpoint": "http://127.0.0.1:14829/v1",
            "env_model_id": "",
            "seed": "1778779265",
            "prompt_version": PROMPT_VERSION,
            "schema_version": CHUNK_FRAME_SCHEMA_VERSION,
            "grammar_enabled": "",
            "runtime_context": context,
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8", errors="replace")
    import hashlib
    assert cache.key_for(text, context=context) == hashlib.sha256(expected_material).hexdigest()


def test_model_caches_default_to_one_kmd_wide_root(monkeypatch) -> None:
    monkeypatch.delenv("KMD_CONFIG_FILE", raising=False)
    monkeypatch.delenv("KMD_SHARED_MODEL_CACHE_ROOT", raising=False)
    for name in config.MODEL_CACHE_NAMESPACES:
        monkeypatch.delenv(name, raising=False)
    _reset_config_caches()
    assert config.model_cache_root() == Path("/data/var/knowmoredirt/model_cache")
    resolved = {config.model_cache_dir(name) for name in config.MODEL_CACHE_NAMESPACES}
    assert all(path.parent == config.model_cache_root() for path in resolved)
    assert config.model_cache_dir("KMD_VERIFIER_CACHE_DIR") == config.model_cache_dir("KMD_QUERY_VERIFIER_CACHE_DIR")
    assert config.model_cache_dir("KMD_IDENTITY_CACHE_DIR") == config.model_cache_dir("KMD_IDENTITY_CANONICAL_CACHE_DIR")


def test_model_cache_specific_override_remains_available_for_tests(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "chunk-drs"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(override))
    assert config.model_cache_dir("KMD_CHUNK_DRS_CACHE_DIR") == override


def test_packaged_filesystem_analysis_model_default_is_discovery_sentinel(monkeypatch) -> None:
    from kmd_runtime_config import text
    monkeypatch.delenv("KMD_LOCAL_MODEL_NAME", raising=False)
    assert text("KMD_LOCAL_MODEL_NAME") == ""
