from __future__ import annotations
import json
from contextlib import contextmanager
from knowmoredirt.model import StrictModelClient, _chat_endpoint


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
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "grammar" not in body
