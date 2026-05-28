from __future__ import annotations

import json
from typing import Any

from knowmoredirt.model import LocalModelClient


class FakeHTTPResponse:
    def __init__(self, payload: Any | None = None, lines: list[bytes] | None = None) -> None:
        self.payload = payload
        self.lines = lines or []

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __iter__(self):
        return iter(self.lines)


def test_local_model_client_discovers_runtime_metadata(monkeypatch) -> None:
    def fake_urlopen(request, timeout: float = 0) -> FakeHTTPResponse:
        url = getattr(request, "full_url", request)
        if str(url).endswith("/v1/models"):
            return FakeHTTPResponse(
                {
                    "data": [
                        {
                            "id": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
                            "meta": {"n_ctx_train": 32768, "n_params": 14770033664},
                        }
                    ]
                }
            )
        if str(url).endswith("/slots"):
            return FakeHTTPResponse(
                [
                    {
                        "n_ctx": 24576,
                        "params": {
                            "top_k": 17,
                            "min_p": 0.03,
                            "repeat_penalty": 1.05,
                        },
                    }
                ]
            )
        if str(url).endswith("/props"):
            return FakeHTTPResponse(
                {
                    "model_alias": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
                    "default_generation_settings": {
                        "n_ctx": 32768,
                        "params": {"top_k": 40, "min_p": 0.05, "repeat_penalty": 1.0},
                    },
                }
            )
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = LocalModelClient(endpoint="http://127.0.0.1:14829/v1", timeout_seconds=30)

    assert client.model_id() == "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    assert client.context_size() == 24576
    assert client.context_source() == "/slots[0].n_ctx"
    assert client.request_settings()["top_k"] == 17
    assert client.request_settings()["min_p"] == 0.03
    assert client.cache_fingerprint()["context_size"] == 24576


def test_local_model_client_uses_completion_stream_and_json_schema(monkeypatch) -> None:
    requests: list[dict[str, Any]] = []

    def fake_urlopen(request, timeout: float = 0) -> FakeHTTPResponse:
        url = getattr(request, "full_url", request)
        if str(url).endswith("/v1/models"):
            return FakeHTTPResponse({"data": [{"id": "test-model", "meta": {"n_ctx_train": 4096}}]})
        if str(url).endswith("/slots"):
            return FakeHTTPResponse([{"n_ctx": 4096, "params": {"top_k": 40, "min_p": 0.05, "repeat_penalty": 1.0}}])
        if str(url).endswith("/props"):
            return FakeHTTPResponse({"default_generation_settings": {"n_ctx": 4096, "params": {}}})
        if str(url).endswith("/completion"):
            body = json.loads(request.data.decode("utf-8"))
            requests.append({"url": str(url), "body": body})
            return FakeHTTPResponse(
                lines=[
                    b'data: {"content":"{\\"ok\\":true"}\n\n',
                    b'data: {"content":"} trailing text"}\n\n',
                ]
            )
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = LocalModelClient(endpoint="http://127.0.0.1:14829/v1", timeout_seconds=30)
    client.server_metadata()

    parsed = client.complete_json(
        "return ok",
        n_predict=64,
        grammar='root ::= "{" "\\"ok\\"" ":" "true" "}"',
        json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
    )

    assert parsed["ok"] is True
    assert parsed["_model_endpoint"] == "http://127.0.0.1:14829/completion"
    assert parsed["_model_stream_closed_after_json"] is True
    assert requests[0]["body"]["stream"] is True
    assert requests[0]["body"]["json_schema"]["type"] == "object"
    assert "grammar" in requests[0]["body"]
