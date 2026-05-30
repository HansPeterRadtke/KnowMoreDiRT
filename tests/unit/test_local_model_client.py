from __future__ import annotations

import json
from typing import Any

from knowmoredirt.model import LocalModelClient, LocalModelJSONError
from knowmoredirt.model_planner import (
    call_model_chunk_drs,
    call_model_chunk_frames,
    call_model_query_drs,
    chunk_drs_json_schema,
    query_frame_from_query_drs,
)


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


def test_chunk_frame_planner_prefers_json_schema_for_capable_clients(monkeypatch) -> None:
    class JsonSchemaCapableModel:
        def __init__(self) -> None:
            self.json_schema: dict[str, Any] | None = None
            self.grammar: str | None = None

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake", "context_size": 4096}

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, Any] | None = None,
        ) -> dict[str, object]:
            self.grammar = grammar
            self.json_schema = json_schema
            assert "Extract generic DRT/DSPG discourse frames" in prompt
            return {
                "frames": [
                    {
                        "frame_type": "state",
                        "predicate": "ready",
                        "arguments": [{"role": "entity", "text": "Aero Gate", "value_type": "entity"}],
                        "identity_hypotheses": [],
                        "polarity": "positive",
                        "modality": "asserted",
                        "context_holder": "",
                        "temporal_text": "",
                        "evidence_text": "Aero Gate is ready",
                        "confidence": 0.9,
                    }
                ],
                "_model_raw": "{}",
            }

    monkeypatch.delenv("KMD_LOCAL_MODEL_JSON_SCHEMA", raising=False)
    model = JsonSchemaCapableModel()

    result = call_model_chunk_frames("Aero Gate is ready.", model)  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert model.grammar is None
    assert model.json_schema is not None
    assert "frames" in model.json_schema["properties"]


def test_chunk_drs_planner_uses_json_schema_and_validates_grounding(monkeypatch, tmp_path) -> None:
    class JsonSchemaCapableModel:
        def __init__(self) -> None:
            self.json_schema: dict[str, Any] | None = None
            self.prompt = ""

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-drs", "context_size": 8192}

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, Any] | None = None,
        ) -> dict[str, object]:
            self.prompt = prompt
            self.json_schema = json_schema
            assert grammar is None
            return {
                "drs": {
                    "schema_version": "chunk-drs-v1",
                    "source_id": "note.txt",
                    "referents": [
                        {"id": "r1", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"},
                        {"id": "r2", "label": "Mira Chen", "kind": "person", "evidence_text": "Mira Chen"},
                    ],
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": "Aero Gate is ready"},
                    ],
                    "conditions": [
                        {
                            "id": "c1",
                            "predicate": "ready",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "entity",
                                    "target_kind": "referent",
                                    "target_id": "r1",
                                    "value": "Aero Gate",
                                    "value_type": "entity",
                                    "evidence_text": "Aero Gate",
                                }
                            ],
                            "evidence_text": "Aero Gate is ready",
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                    "evidence_spans": ["Aero Gate is ready"],
                    "semantic_notes": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.delenv("KMD_LOCAL_MODEL_JSON_SCHEMA", raising=False)
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "drs-cache"))
    model = JsonSchemaCapableModel()

    result = call_model_chunk_drs("Aero Gate is ready. Mira Chen signed.", model, rel_path="note.txt")  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["validation"]["condition_count"] == 1
    assert result["context_budget"]["runtime_context_size"] == 8192
    assert model.json_schema is not None
    assert "drs" in model.json_schema["properties"]
    assert "source-grounded DRS" in model.prompt


def test_query_drs_planner_uses_json_schema(monkeypatch, tmp_path) -> None:
    class JsonSchemaCapableModel:
        def __init__(self) -> None:
            self.json_schema: dict[str, Any] | None = None
            self.prompt = ""

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-query-drs", "context_size": 8192}

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, Any] | None = None,
        ) -> dict[str, object]:
            self.prompt = prompt
            self.json_schema = json_schema
            assert grammar is None
            return {
                "query_drs": {
                    "schema_version": "query-drs-v3",
                    "question": "Who reviewed Aero Gate?",
                    "answer_variables": ["reviewer"],
                    "target_referents": [
                        {"id": "qr0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"}
                    ],
                    "requested_conditions": [
                        {
                            "id": "qc0",
                            "predicate": "reviewed",
                            "box_id": "",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "object",
                                    "target_kind": "referent",
                                    "target_id": "qr0",
                                    "value": "Aero Gate",
                                    "value_type": "entity",
                                    "evidence_text": "Aero Gate",
                                }
                            ],
                            "evidence_text": "reviewed Aero Gate",
                        }
                    ],
                    "constraints": [],
                    "box_requirements": [],
                    "temporal_scope": "",
                    "aggregation": "",
                    "answer_type": "person",
                    "requires_evidence": True,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.delenv("KMD_LOCAL_MODEL_JSON_SCHEMA", raising=False)
    monkeypatch.setenv("KMD_QUERY_DRS_CACHE_DIR", str(tmp_path / "query-drs-cache"))
    model = JsonSchemaCapableModel()

    result = call_model_query_drs("Who reviewed Aero Gate?", model)  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["validation"]["condition_count"] == 1
    assert model.json_schema is not None
    assert "query_drs" in model.json_schema["properties"]
    query_schema = model.json_schema["properties"]["query_drs"]
    assert query_schema["properties"]["question"]["enum"] == ["Who reviewed Aero Gate?"]
    assert query_schema["properties"]["schema_version"]["enum"] == ["query-drs-v3"]
    assert "generic DRT query DRS" in model.prompt


def test_query_drs_request_failure_does_not_poison_cache(monkeypatch, tmp_path) -> None:
    class FailsThenSucceedsModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-query-drs-retry", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary local model failure")
            return {
                "query_drs": {
                    "schema_version": "query-drs-v3",
                    "question": "Who reviewed Aero Gate?",
                    "answer_variables": [
                        {
                            "id": "qv0",
                            "label": "reviewer",
                            "answer_type": "person",
                            "evidence_text": "Who",
                        }
                    ],
                    "target_referents": [
                        {"id": "qr0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"}
                    ],
                    "temporal_records": [],
                    "requested_conditions": [
                        {
                            "id": "qc0",
                            "predicate": "reviewed",
                            "box_id": "",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "agent",
                                    "target_kind": "answer_variable",
                                    "target_id": "qv0",
                                    "value": "",
                                    "value_type": "person",
                                    "evidence_text": "Who",
                                },
                                {
                                    "role": "theme",
                                    "target_kind": "referent",
                                    "target_id": "qr0",
                                    "value": "Aero Gate",
                                    "value_type": "entity",
                                    "evidence_text": "Aero Gate",
                                },
                            ],
                            "evidence_text": "reviewed Aero Gate",
                        }
                    ],
                    "constraints": [],
                    "box_requirements": [],
                    "temporal_scope": "",
                    "aggregation": "",
                    "answer_type": "person",
                    "requires_evidence": True,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_QUERY_DRS_CACHE_DIR", str(tmp_path / "query-drs-cache"))
    model = FailsThenSucceedsModel()

    first = call_model_query_drs("Who reviewed Aero Gate?", model)  # type: ignore[arg-type]
    second = call_model_query_drs("Who reviewed Aero Gate?", model)  # type: ignore[arg-type]

    assert first["accepted"] is False
    assert first["reason"] == "request_failed"
    assert second["accepted"] is True
    assert model.calls == 2


def test_query_drs_invalid_json_is_not_request_failure(monkeypatch, tmp_path) -> None:
    class InvalidJSONModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-query-drs-invalid-json", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            raise LocalModelJSONError("bad json", raw_text="not json", snippet="not json")

    monkeypatch.setenv("KMD_QUERY_DRS_CACHE_DIR", str(tmp_path / "query-drs-cache"))

    result = call_model_query_drs("Who reviewed Aero Gate?", InvalidJSONModel())  # type: ignore[arg-type]

    assert result["accepted"] is False
    assert result["reason"] == "invalid_json"
    assert result["raw_text"] == "not json"
    assert result["raw_snippet"] == "not json"


def test_chunk_drs_schema_caps_evidence_strings_to_chunk_length() -> None:
    schema = chunk_drs_json_schema(19)
    drs_schema = schema["properties"]["drs"]
    referent_schema = drs_schema["properties"]["referents"]["items"]
    box_schema = drs_schema["properties"]["boxes"]["items"]
    condition_schema = drs_schema["properties"]["conditions"]["items"]
    argument_schema = condition_schema["properties"]["arguments"]["items"]

    assert referent_schema["properties"]["evidence_text"]["maxLength"] == 19
    assert box_schema["properties"]["evidence_text"]["maxLength"] == 19
    assert condition_schema["properties"]["evidence_text"]["maxLength"] == 19
    assert argument_schema["properties"]["evidence_text"]["maxLength"] == 19
    assert drs_schema["properties"]["evidence_spans"]["items"]["maxLength"] == 19


def test_chunk_drs_planner_repairs_model_referent_argument_records(monkeypatch, tmp_path) -> None:
    class MissingReferentModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-missing-ref", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            return {
                "drs": {
                    "schema_version": "chunk-drs-v1",
                    "source_id": "note.txt",
                    "referents": [],
                    "boxes": [
                        {
                            "id": "b0",
                            "kind": "asserted",
                            "parent_id": "",
                            "holder_referent_id": "",
                            "evidence_text": "Aero Gate is ready.",
                        }
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "ready",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "theme",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "Aero Gate",
                                    "value_type": "entity",
                                    "evidence_text": "Aero Gate",
                                }
                            ],
                            "evidence_text": "Aero Gate is ready.",
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                    "evidence_spans": ["Aero Gate is ready."],
                    "semantic_notes": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "missing-ref-cache"))
    result = call_model_chunk_drs("Aero Gate is ready.", MissingReferentModel(), rel_path="note.txt")  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["validation"]["referent_count"] == 1
    assert result["drs"]["referents"][0]["id"] == "r0"
    assert result["drs"]["referents"][0]["label"] == "Aero Gate"


def test_query_drs_projects_to_query_frame_without_language_handlers() -> None:
    query_drs = {
        "answer_variables": ["reviewer"],
        "target_referents": [
            {"id": "r0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"}
        ],
        "requested_conditions": [
            {
                "id": "c0",
                "predicate": "review",
                "box_id": "",
                "polarity": "positive",
                "modality": "asserted",
                "temporal_id": "",
                "arguments": [
                    {
                        "role": "theme",
                        "target_kind": "referent",
                        "target_id": "r0",
                        "value": "Aero Gate",
                        "value_type": "entity",
                        "evidence_text": "Aero Gate",
                    }
                ],
                "evidence_text": "reviewed Aero Gate",
            }
        ],
        "constraints": ["release"],
        "box_requirements": [
            {"id": "b1", "kind": "reported", "parent_id": "", "holder_referent_id": "", "evidence_text": "reported"}
        ],
        "temporal_scope": "latest",
        "aggregation": "",
        "answer_type": "person",
        "requires_evidence": True,
    }

    frame = query_frame_from_query_drs("Who reviewed Aero Gate?", query_drs)

    assert frame is not None
    assert frame["source"] == "model_query_drs"
    assert frame["target_anchors"] == ("Aero Gate",)
    assert frame["answer_variables"] == ("reviewer",)
    assert frame["requested_relation"] == "review"
    assert "theme" in frame["relation_terms"]
    assert frame["scope_requirements"] == ("reported",)
    assert frame["temporal_scope"] == "latest"
    assert frame["answer_type"] == "person"
