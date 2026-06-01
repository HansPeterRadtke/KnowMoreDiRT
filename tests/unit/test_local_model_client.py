from __future__ import annotations

import json
from typing import Any

import pytest

from knowmoredirt.model import LocalModelClient, LocalModelJSONError
from knowmoredirt.model_planner import (
    CHUNK_FRAME_CONTEXT_BUDGET_POLICY,
    CHUNK_DRS_IDENTITY_PROVENANCE_POLICY,
    CHUNK_DRS_TEMPORAL_PROVENANCE_POLICY,
    QUERY_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY,
    QUERY_OPERATOR_SCHEMA_POLICY,
    build_answer_verification_prompt,
    call_model_chunk_drs,
    call_model_chunk_frames,
    call_model_query_evidence_answer,
    call_model_query_plan,
    call_model_query_drs,
    chunk_drs_cache_context,
    chunk_drs_json_schema,
    chunk_frame_cache_context,
    default_query_drs_n_predict,
    query_drs_array_max_items,
    query_frame_from_query_drs,
)


def test_verifier_prompt_allows_scoped_embedded_bindings() -> None:
    prompt = build_answer_verification_prompt(
        "What does Kalo Reed believe?",
        {
            "target_anchors": ["Kalo Reed"],
            "requested_relation": "believe",
            "relation_terms": ["believe", "content"],
            "scope_requirements": ["reported"],
            "answer_type": "content_phrase",
        },
        "Mira Stone archived the Slate Quill",
        [{"source": "belief.txt", "text": "Kalo Reed believes that Mira Stone archived the Slate Quill."}],
        [{"record_kind": "condition", "subject": "reported", "predicate": "archive"}],
    )

    assert "embedded proposition or scoped value" in prompt
    assert "instead of requiring the candidate text itself to repeat the target anchor" in prompt


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
    monkeypatch.setenv("KMD_LOCAL_MODEL_API", "chat")
    monkeypatch.setenv("KMD_LOCAL_MODEL_STREAM", "0")

    client = LocalModelClient(endpoint="http://127.0.0.1:14829/v1", timeout_seconds=30)

    assert client.model_id() == "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    assert client.context_size() == 24576
    assert client.context_source() == "/slots[0].n_ctx"
    assert client.request_settings()["top_k"] == 17
    assert client.request_settings()["min_p"] == 0.03
    fingerprint = client.cache_fingerprint()
    assert fingerprint["context_size"] == 24576
    assert fingerprint["transport_settings"] == {"api": "chat", "stream": False}
    assert chunk_drs_cache_context(client)["model_fingerprint"]["transport_settings"] == {
        "api": "chat",
        "stream": False,
    }


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


def test_local_model_client_stream_enforces_wall_timeout(monkeypatch) -> None:
    def fake_urlopen(request, timeout: float = 0) -> FakeHTTPResponse:
        url = getattr(request, "full_url", request)
        if str(url).endswith("/v1/models"):
            return FakeHTTPResponse({"data": [{"id": "test-model", "meta": {"n_ctx_train": 4096}}]})
        if str(url).endswith("/slots"):
            return FakeHTTPResponse([{"n_ctx": 4096, "params": {"top_k": 40, "min_p": 0.05, "repeat_penalty": 1.0}}])
        if str(url).endswith("/props"):
            return FakeHTTPResponse({"default_generation_settings": {"n_ctx": 4096, "params": {}}})
        if str(url).endswith("/completion"):
            return FakeHTTPResponse(
                lines=[
                    b'data: {"content":"{\\"ok\\":"}\n\n',
                    b'data: {"content":"true"}\n\n',
                ]
            )
        raise AssertionError(f"unexpected URL {url}")

    ticks = iter([0.0, 0.5, 3.0])

    def fake_time() -> float:
        return next(ticks, 3.0)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("knowmoredirt.model.time.time", fake_time)
    client = LocalModelClient(endpoint="http://127.0.0.1:14829/v1", timeout_seconds=2)
    client.server_metadata()

    with pytest.raises(TimeoutError):
        client.complete_json("return ok", n_predict=64)


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
    assert result["context_budget"]["context_budget_policy"] == CHUNK_FRAME_CONTEXT_BUDGET_POLICY
    cache_context = chunk_frame_cache_context(model)  # type: ignore[arg-type]
    assert cache_context["context_budget_policy"] == CHUNK_FRAME_CONTEXT_BUDGET_POLICY
    assert model.grammar is None
    assert model.json_schema is not None
    assert "frames" in model.json_schema["properties"]


def test_chunk_frame_invalid_json_keeps_context_budget() -> None:
    class InvalidFrameModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-invalid-frame-budget", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            assert "Extract generic DRT/DSPG discourse frames" in prompt
            return {"unexpected": [], "_model_raw": "{}"}

    result = call_model_chunk_frames("Aero Gate is ready.", InvalidFrameModel())  # type: ignore[arg-type]

    assert result["accepted"] is False
    assert result["reason"] == "invalid_json"
    assert result["context_budget"]["runtime_context_size"] == 4096
    assert result["context_budget"]["context_budget_policy"] == CHUNK_FRAME_CONTEXT_BUDGET_POLICY


def test_query_evidence_invalid_json_after_failed_repair_is_bounded(monkeypatch, tmp_path) -> None:
    class InvalidEvidenceAnswerModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-query-evidence-invalid-repair", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            if "Repair the previous local-model output" in prompt:
                return {"still_unexpected": True, "_model_raw": '{"still_unexpected":true}'}
            assert "bounded DRT/DSPG question analysis" in prompt
            return {"unexpected": True, "_model_raw": '{"unexpected":true}'}

    monkeypatch.setenv("KMD_QUERY_EVIDENCE_CACHE_DIR", str(tmp_path / "query-evidence-cache"))
    monkeypatch.setenv("KMD_QUERY_EVIDENCE_REPAIR_CACHE_DIR", str(tmp_path / "query-evidence-repair-cache"))
    model = InvalidEvidenceAnswerModel()

    result = call_model_query_evidence_answer(
        "What is the state of Aero Gate?",
        [{"rel_path": "note.txt", "text": "Aero Gate is ready."}],
        model,  # type: ignore[arg-type]
    )

    assert result["accepted"] is False
    assert result["reason"] == "invalid_json"
    assert result["repair_failure_reason"] == "invalid_json"
    assert result["repair_prompt_hash"]
    assert model.calls == 2


def test_query_evidence_repair_request_failure_does_not_poison_cache(monkeypatch, tmp_path) -> None:
    class RepairFailsThenSucceedsModel:
        def __init__(self) -> None:
            self.primary_calls = 0
            self.repair_calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-query-evidence-repair-retry", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Repair the previous local-model output" in prompt:
                self.repair_calls += 1
                if self.repair_calls == 1:
                    raise RuntimeError("temporary repair failure")
                return {
                    "result": {
                        "query_frame": {
                            "target_anchors": ["Aero Gate"],
                            "answer_variables": ["state"],
                            "requested_relation": "state",
                            "relation_terms": ["state"],
                            "constraints": [],
                            "scope_requirements": [],
                            "modality_requirements": [],
                            "answer_type": "state",
                            "temporal_scope": "",
                            "negated": False,
                            "aggregation": "",
                            "requires_evidence": True,
                        },
                        "sufficient_evidence": False,
                        "answer_type": "state",
                        "answer": "unknown",
                        "evidence_span": "",
                        "reason": "insufficient evidence",
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            self.primary_calls += 1
            assert "bounded DRT/DSPG question analysis" in prompt
            return {"unexpected": True, "_model_raw": '{"unexpected":true}'}

    monkeypatch.setenv("KMD_QUERY_EVIDENCE_CACHE_DIR", str(tmp_path / "query-evidence-cache"))
    monkeypatch.setenv("KMD_QUERY_EVIDENCE_REPAIR_CACHE_DIR", str(tmp_path / "query-evidence-repair-cache"))
    model = RepairFailsThenSucceedsModel()
    evidence = [{"rel_path": "note.txt", "text": "Aero Gate is ready."}]

    first = call_model_query_evidence_answer("What is the state of Aero Gate?", evidence, model)  # type: ignore[arg-type]
    second = call_model_query_evidence_answer("What is the state of Aero Gate?", evidence, model)  # type: ignore[arg-type]

    assert first["accepted"] is False
    assert first["repair_failure_reason"] == "request_failed"
    assert second["accepted"] is True
    assert second["fresh_or_cached"] == "fresh_repair"
    assert model.primary_calls == 2
    assert model.repair_calls == 2


def test_query_frame_schema_constrains_temporal_scope_operator(monkeypatch, tmp_path) -> None:
    class QueryFrameModel:
        def __init__(self) -> None:
            self.json_schema: dict[str, Any] | None = None
            self.prompt = ""

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-query-frame-temporal", "context_size": 4096}

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
                "query_frame": {
                    "target_anchors": ["Delta Well"],
                    "answer_variables": ["state"],
                    "requested_relation": "state",
                    "relation_terms": ["state"],
                    "constraints": [],
                    "scope_requirements": [],
                    "modality_requirements": [],
                    "answer_type": "state",
                    "temporal_scope": "latest",
                    "negated": False,
                    "aggregation": "",
                    "requires_evidence": True,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.delenv("KMD_LOCAL_MODEL_JSON_SCHEMA", raising=False)
    monkeypatch.setenv("KMD_QUERY_PLAN_CACHE_DIR", str(tmp_path / "query-frame-cache"))
    model = QueryFrameModel()

    result = call_model_query_plan("What is the current state of Delta Well?", model, n_predict=64)  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["temporal_scope"] == "latest"
    assert "temporal_scope must be" in model.prompt
    assert "aggregation must be" in model.prompt
    assert result["operator_schema_policy"] == QUERY_OPERATOR_SCHEMA_POLICY
    assert model.json_schema is not None
    query_schema = model.json_schema["properties"]["query_frame"]
    assert query_schema["properties"]["temporal_scope"]["enum"] == ["", "earliest", "latest"]
    assert query_schema["properties"]["aggregation"]["enum"] == ["", "count", "list", "set"]


def test_query_frame_invalid_json_failure_is_cached(monkeypatch, tmp_path) -> None:
    class InvalidQueryFrameModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-query-frame-invalid-cache", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            return {"_model_raw": "{}"}

    monkeypatch.setenv("KMD_QUERY_PLAN_CACHE_DIR", str(tmp_path / "query-frame-cache"))
    model = InvalidQueryFrameModel()

    first = call_model_query_plan("What state is Delta Well in?", model, n_predict=64)  # type: ignore[arg-type]
    second = call_model_query_plan("What state is Delta Well in?", model, n_predict=64)  # type: ignore[arg-type]

    assert first["accepted"] is False
    assert first["reason"] == "invalid_json"
    assert second["accepted"] is False
    assert second["reason"] == "invalid_json"
    assert model.calls == 1


def test_query_frame_request_failure_does_not_poison_cache(monkeypatch, tmp_path) -> None:
    class FailsThenSucceedsQueryFrameModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-query-frame-request-retry", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary local model failure")
            return {
                "query_frame": {
                    "target_anchors": ["Delta Well"],
                    "answer_variables": ["state"],
                    "requested_relation": "state",
                    "relation_terms": ["state"],
                    "constraints": [],
                    "scope_requirements": [],
                    "modality_requirements": [],
                    "answer_type": "state",
                    "temporal_scope": "latest",
                    "negated": False,
                    "aggregation": "",
                    "requires_evidence": True,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_QUERY_PLAN_CACHE_DIR", str(tmp_path / "query-frame-cache"))
    model = FailsThenSucceedsQueryFrameModel()

    first = call_model_query_plan("What is the current state of Delta Well?", model, n_predict=64)  # type: ignore[arg-type]
    second = call_model_query_plan("What is the current state of Delta Well?", model, n_predict=64)  # type: ignore[arg-type]

    assert first["accepted"] is False
    assert first["reason"] == "request_failed"
    assert second["accepted"] is True
    assert second["temporal_scope"] == "latest"
    assert model.calls == 2


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


def test_chunk_drs_request_failure_keeps_context_budget_and_retries(monkeypatch, tmp_path) -> None:
    class FailsThenSucceedsChunkDRSModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-chunk-drs-request-retry", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary chunk DRS failure")
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "note.txt",
                    "referents": [
                        {"id": "r0", "label": "Aero Gate", "kind": "artifact", "evidence_text": "Aero Gate"},
                    ],
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": "Aero Gate is ready"},
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
                                    "role": "subject",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "artifact",
                                    "evidence_text": "Aero Gate",
                                }
                            ],
                            "evidence_text": "Aero Gate is ready",
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "drs-request-cache"))
    model = FailsThenSucceedsChunkDRSModel()

    first = call_model_chunk_drs("Aero Gate is ready.", model, rel_path="note.txt")  # type: ignore[arg-type]
    second = call_model_chunk_drs("Aero Gate is ready.", model, rel_path="note.txt")  # type: ignore[arg-type]

    assert first["accepted"] is False
    assert first["reason"] == "request_failed"
    assert first["context_budget"]["runtime_context_size"] == 8192
    assert second["accepted"] is True
    assert model.calls == 2


def test_chunk_drs_filters_identity_without_bilateral_evidence(monkeypatch, tmp_path) -> None:
    class IdentityModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-identity-provenance", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "note.txt",
                    "referents": [
                        {"id": "r0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"},
                        {"id": "r1", "label": "AG-1", "kind": "identifier", "evidence_text": "AG-1"},
                        {"id": "r2", "label": "Mira Chen", "kind": "person", "evidence_text": "Mira Chen"},
                    ],
                    "boxes": [
                        {
                            "id": "b0",
                            "kind": "asserted",
                            "parent_id": "",
                            "holder_referent_id": "",
                            "evidence_text": "Aero Gate alias AG-1.",
                        }
                    ],
                    "conditions": [],
                    "identity_hypotheses": [
                        {
                            "left_referent_id": "r0",
                            "right_referent_id": "r2",
                            "status": "candidate",
                            "evidence_text": "Aero Gate alias AG-1",
                            "confidence": 0.4,
                        },
                        {
                            "left_referent_id": "r0",
                            "right_referent_id": "r1",
                            "status": "candidate",
                            "evidence_text": "Aero Gate alias AG-1",
                            "confidence": 0.7,
                        },
                    ],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "drs-cache"))
    model = IdentityModel()

    result = call_model_chunk_drs(
        "Aero Gate alias AG-1. Mira Chen signed.",
        model,  # type: ignore[arg-type]
        rel_path="note.txt",
    )

    assert result["accepted"] is True
    assert result["drs"]["identity_hypotheses"] == [
        {
            "left_referent_id": "r0",
            "right_referent_id": "r1",
            "status": "candidate",
            "evidence_text": "Aero Gate alias AG-1",
            "confidence": 0.7,
        }
    ]
    assert (
        chunk_drs_cache_context(model, n_predict=384)["identity_provenance_policy"]
        == CHUNK_DRS_IDENTITY_PROVENANCE_POLICY
    )


def test_chunk_drs_prunes_unreferenced_temporal_records(monkeypatch, tmp_path) -> None:
    class TemporalModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-temporal-provenance", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "note.txt",
                    "referents": [
                        {"id": "r0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"}
                    ],
                    "boxes": [
                        {
                            "id": "b0",
                            "kind": "asserted",
                            "parent_id": "",
                            "holder_referent_id": "",
                            "evidence_text": "On 2026-01-03, Aero Gate is ready.",
                        }
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "ready",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "t0",
                            "arguments": [],
                            "evidence_text": "Aero Gate is ready.",
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [
                        {
                            "id": "t0",
                            "value": "2026-01-03",
                            "value_type": "date_time",
                            "evidence_text": "2026-01-03",
                        },
                        {"id": "t1", "value": "ready", "value_type": "state", "evidence_text": "ready"},
                    ],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "drs-cache"))
    model = TemporalModel()

    result = call_model_chunk_drs(
        "On 2026-01-03, Aero Gate is ready.",
        model,  # type: ignore[arg-type]
        rel_path="note.txt",
    )

    assert result["accepted"] is True
    assert result["drs"]["temporal_records"] == [
        {"id": "t0", "value": "2026-01-03", "value_type": "date_time", "evidence_text": "2026-01-03"}
    ]
    assert (
        chunk_drs_cache_context(model, n_predict=384)["temporal_provenance_policy"]
        == CHUNK_DRS_TEMPORAL_PROVENANCE_POLICY
    )


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
    assert query_schema["properties"]["temporal_scope"]["enum"] == ["", "earliest", "latest"]
    assert query_schema["properties"]["aggregation"]["enum"] == ["", "count", "list", "set"]
    assert query_schema["properties"]["requested_conditions"]["maxItems"] == query_drs_array_max_items(256)
    assert (
        query_schema["properties"]["requested_conditions"]["items"]["properties"]["arguments"]["maxItems"]
        == query_drs_array_max_items(256)
    )
    assert "generic DRT query DRS" in model.prompt


def test_short_query_drs_uses_smaller_surface_budget(monkeypatch, tmp_path) -> None:
    class LargeContextQueryModel:
        def __init__(self) -> None:
            self.n_predict = 0
            self.json_schema: dict[str, Any] | None = None

        def context_size(self) -> int:
            return 32768

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-large-context-query-drs", "context_size": 32768}

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, Any] | None = None,
        ) -> dict[str, object]:
            self.n_predict = n_predict
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
    model = LargeContextQueryModel()

    assert default_query_drs_n_predict(model, "Who reviewed Aero Gate?") == 384  # type: ignore[arg-type]
    result = call_model_query_drs("Who reviewed Aero Gate?", model)  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["output_budget_policy"] == QUERY_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY
    assert result["operator_schema_policy"] == QUERY_OPERATOR_SCHEMA_POLICY
    assert model.n_predict == 384
    assert model.json_schema is not None
    query_schema = model.json_schema["properties"]["query_drs"]
    assert query_schema["properties"]["requested_conditions"]["maxItems"] == query_drs_array_max_items(384)


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
