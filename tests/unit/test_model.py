from __future__ import annotations
import json
from contextlib import contextmanager
import pytest
from knowmoredirt.model import ModelError, StrictModelClient, _chat_endpoint, _validate


def test_chat_endpoint_normalization():
    assert _chat_endpoint("http://host:1/v1") == "http://host:1/v1/chat/completions"
    assert _chat_endpoint("http://host:1/v1/chat/completions") == "http://host:1/v1/chat/completions"


def test_client_sends_strict_json_schema(monkeypatch, tmp_path):
    captured = {}
    schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({"value": "ok"})}}]}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = StrictModelClient("http://localhost:9/v1", tmp_path)
    assert client.complete_json("unit", "prompt", schema, 123) == {"value": "ok"}
    body = captured["body"]
    assert body["max_tokens"] == 123
    assert body["reasoning_effort"] == "low"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "grammar" not in body


def test_client_retries_empty_length_response_with_larger_budget(monkeypatch, tmp_path):
    bodies = []
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout):
        bodies.append(json.loads(request.data))
        if len(bodies) == 1:
            return Response(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "", "reasoning_content": "bounded reasoning"},
                        }
                    ]
                }
            )
        return Response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"value": 12})},
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = StrictModelClient("http://localhost:9/v1", tmp_path)
    assert client.complete_json("unit", "prompt", schema, 123) == {"value": 12}
    assert [body["max_tokens"] for body in bodies] == [123, 2048]


def test_local_schema_validation_enforces_array_and_numeric_bounds():
    with pytest.raises(ModelError):
        _validate(["a", "b"], {"type": "array", "items": {"type": "string"}, "maxItems": 1})
    with pytest.raises(ModelError):
        _validate(1.5, {"type": "number", "minimum": 0, "maximum": 1})
    _validate(["a"], {"type": "array", "items": {"type": "string"}, "maxItems": 1})
    _validate(1, {"type": "number", "minimum": 0, "maximum": 1})
