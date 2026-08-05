from __future__ import annotations

import json
from typing import Any

import pytest

from file_system_catalog import content_pipeline
from knowmoredirt import model


class _Response:
    def __init__(self, lines: list[bytes] | None = None, payload: bytes = b"{}") -> None:
        self.lines = lines or []
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)

    def read(self) -> bytes:
        return self.payload


def test_model_control_fetch_uses_finite_environment_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float | None] = []

    def fake_urlopen(_request: Any, timeout: float | None = None) -> _Response:
        seen.append(timeout)
        return _Response(payload=b'{"ok":true}')

    monkeypatch.setenv("KMD_LOCAL_MODEL_CONTROL_TIMEOUT_SECONDS", "1.25")
    monkeypatch.setattr(model.urllib.request, "urlopen", fake_urlopen)

    assert model._fetch_json("http://127.0.0.1:9999/health") == {"ok": True}
    assert seen == [1.25]


def test_catalog_request_json_replaces_none_with_finite_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float | None] = []

    def fake_urlopen(_request: Any, timeout: float | None = None) -> _Response:
        seen.append(timeout)
        return _Response(payload=b'{"ok":true}')

    monkeypatch.setenv("KMD_LOCAL_MODEL_CONTROL_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setattr(content_pipeline.urllib.request, "urlopen", fake_urlopen)

    assert content_pipeline.request_json("http://127.0.0.1:9999/health", timeout=None) == {"ok": True}
    assert seen == [2.5]


def test_analysis_client_uses_control_timeout_for_tokenize_and_template(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, float | None]] = []

    def fake_request(url: str, payload: dict[str, Any] | None = None, *, timeout: float | None = None) -> dict[str, Any]:
        calls.append((url, timeout))
        if url.endswith("/tokenize"):
            return {"tokens": [1, 2, 3]}
        if url.endswith("/apply-template"):
            return {"prompt": "rendered"}
        return {"status": "ok"}

    monkeypatch.setattr(content_pipeline, "request_json", fake_request)
    client = content_pipeline.AnalysisClient(
        "http://127.0.0.1:9999",
        model="fake",
        control_timeout_seconds=3.5,
    )

    assert client.token_count("abc") == 3
    assert client._render_prompt(system="s", user="u") == "rendered"
    assert calls == [
        ("http://127.0.0.1:9999/tokenize", 3.5),
        ("http://127.0.0.1:9999/apply-template", 3.5),
    ]


def test_catalog_stream_stops_fast_endless_emitter_at_event_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    event = b'data: {"choices":[{"delta":{"content":"x"}}]}\n'
    monkeypatch.setenv("KMD_LOCAL_MODEL_STREAM_EVENT_MULTIPLIER", "1")
    event_limit = content_pipeline._stream_event_limit(1)
    lines = [event for _ in range(event_limit + 1)]

    monkeypatch.setattr(
        content_pipeline.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(lines=lines),
    )

    with pytest.raises(RuntimeError, match=rf"event limit {event_limit}"):
        content_pipeline.stream_chat_completion_json(
            "http://127.0.0.1:9999/v1/chat/completions",
            {"model": "fake", "max_tokens": 1, "messages": []},
            per_token_timeout_seconds=0.2,
        )


def test_stream_limits_scale_from_requested_output() -> None:
    assert content_pipeline._stream_event_limit(10) == 104
    assert content_pipeline._stream_byte_limit(10) >= 65536
    assert model._stream_event_limit(10) == 104
    assert model._stream_byte_limit(10) >= 65536
