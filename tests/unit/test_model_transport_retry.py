from __future__ import annotations

import io
import json
import urllib.error

import pytest

from knowmoredirt.model import (
    LocalModelJSONError,
    _fetch_json,
    complete_json_with_transport_retry,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_control_get_retries_transient_url_error(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(_request, timeout):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise urllib.error.URLError("temporary disconnect")
        return _Response({"ok": True})

    monkeypatch.setenv("KMD_LOCAL_MODEL_CONTROL_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("KMD_LOCAL_MODEL_CONTROL_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setattr("knowmoredirt.model.urllib.request.urlopen", fake_urlopen)
    assert _fetch_json("http://127.0.0.1:14829/health", timeout=1) == {"ok": True}
    assert calls == 3


def test_control_get_does_not_retry_nontransient_http_error(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            str(request),
            400,
            "bad request",
            hdrs=None,
            fp=io.BytesIO(b"bad"),
        )

    monkeypatch.setenv("KMD_LOCAL_MODEL_CONTROL_RETRY_ATTEMPTS", "5")
    monkeypatch.setenv("KMD_LOCAL_MODEL_CONTROL_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setattr("knowmoredirt.model.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        _fetch_json("http://127.0.0.1:14829/health", timeout=1)
    assert calls == 1


def test_direct_semantic_retry_retries_transport_disconnect(monkeypatch) -> None:
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls < 3:
                raise urllib.error.URLError("connection reset")
            return {"value": "ok"}

    client = Client()
    monkeypatch.setenv("KMD_LOCAL_MODEL_DIRECT_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("KMD_LOCAL_MODEL_DIRECT_RETRY_BACKOFF_SECONDS", "0")
    assert complete_json_with_transport_retry(
        client,
        "prompt",
        n_predict=32,
        json_schema={"type": "object"},
    ) == {"value": "ok"}
    assert client.calls == 3


def test_direct_semantic_retry_does_not_retry_schema_failure(monkeypatch) -> None:
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            raise LocalModelJSONError(
                "bad schema",
                raw_text="{}",
                snippet="{}",
                reason="schema_validation_failed",
            )

    client = Client()
    monkeypatch.setenv("KMD_LOCAL_MODEL_DIRECT_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("KMD_LOCAL_MODEL_DIRECT_RETRY_BACKOFF_SECONDS", "0")
    with pytest.raises(LocalModelJSONError):
        complete_json_with_transport_retry(
            client,
            "prompt",
            n_predict=32,
            json_schema={"type": "object"},
        )
    assert client.calls == 1
