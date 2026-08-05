from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCANNED_ROOTS = [ROOT / "src", ROOT / "scripts", ROOT / "research", ROOT / "build" / "lib"]


def _python_files():
    for root in SCANNED_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" not in path.parts:
                yield path


def test_all_model_completion_payloads_enable_streaming() -> None:
    violations: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "stream":
                    if not (isinstance(value, ast.Constant) and value.value is True):
                        violations.append(f"{path.relative_to(ROOT)}:{getattr(value, 'lineno', 0)}")
    assert not violations, "non-streaming model payloads: " + ", ".join(violations)


def test_only_shared_per_token_timeout_setting_controls_generation() -> None:
    forbidden = {
        "KMD_LOCAL_MODEL_TIMEOUT_SECONDS",
        "KMD_LOCAL_MODEL_METADATA_TIMEOUT",
        "KMD_MODEL_PROBE_TIMEOUT",
        "KMD_FALLBACK_MODEL_PER_TOKEN_TIMEOUT_SECONDS",
        "request_json_without_read_deadline",
        "connect_timeout",
    }
    violations: list[str] = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)}:{token}")
    assert not violations, "forbidden model timeout contracts: " + ", ".join(violations)


def test_model_metadata_and_streams_have_explicit_finite_client_limits() -> None:
    content = (ROOT / "src" / "file_system_catalog" / "content_pipeline.py").read_text(encoding="utf-8")
    model = (ROOT / "src" / "knowmoredirt" / "model.py").read_text(encoding="utf-8")
    benchmark = (ROOT / "scripts" / "benchmarks" / "run_internal_model_benchmark.py").read_text(encoding="utf-8")

    combined = "\n".join((content, model, benchmark))
    assert "timeout=None" not in combined
    assert "KMD_LOCAL_MODEL_CONTROL_TIMEOUT_SECONDS" in combined
    assert "KMD_LOCAL_MODEL_STREAM_TOTAL_TIMEOUT_SECONDS" in content
    assert "KMD_LOCAL_MODEL_STREAM_TOTAL_TIMEOUT_SECONDS" in model
    assert "KMD_LOCAL_MODEL_STREAM_EVENT_MULTIPLIER" in content
    assert "KMD_LOCAL_MODEL_STREAM_EVENT_MULTIPLIER" in model
    assert "KMD_LOCAL_MODEL_STREAM_BYTES_PER_TOKEN" in content
    assert "KMD_LOCAL_MODEL_STREAM_BYTES_PER_TOKEN" in model
