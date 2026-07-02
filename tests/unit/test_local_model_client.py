from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.model import (
    LocalModelClient,
    LocalModelJSONError,
    LocalModelUnavailableError,
    _append_missing_json_closers,
)
from knowmoredirt.models import Answer, Evidence
from knowmoredirt import model_planner
from knowmoredirt.model_planner import (
    CHUNK_FRAME_CONTEXT_BUDGET_POLICY,
    CHUNK_DRS_IDENTITY_PROVENANCE_POLICY,
    CHUNK_DRS_TEMPORAL_PROVENANCE_POLICY,
    QUERY_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY,
    QUERY_OPERATOR_SCHEMA_POLICY,
    call_model_answer_verification,
    call_model_answer_canonicalization,
    call_model_source_resolved_answer,
    build_answer_verification_prompt,
    build_compact_chunk_drs_prompt,
    call_model_chunk_drs,
    call_model_chunk_drs_compact,
    call_model_chunk_frames,
    call_model_identity_canonicalization,
    call_model_query_evidence_answer,
    call_model_query_plan_test_only,
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


def test_json_fragment_repair_appends_missing_trailing_closer() -> None:
    repaired = _append_missing_json_closers(
        '{"a":"content_phrase","targets":["Mist Vale"],"predicates":["say"]'
    )

    assert repaired == '{"a":"content_phrase","targets":["Mist Vale"],"predicates":["say"]}'
    assert _append_missing_json_closers("not json") is None


def test_engine_chunk_stage_uses_per_token_timeout(monkeypatch) -> None:
    created: list[tuple[str, float]] = []

    class FakeClient:
        def __init__(self, endpoint: str, timeout_seconds: float) -> None:
            self.endpoint = endpoint
            self.timeout_seconds = timeout_seconds

    def fake_local_model_client(endpoint: str, timeout_seconds: float) -> FakeClient:
        created.append((endpoint, timeout_seconds))
        return FakeClient(endpoint, timeout_seconds)

    monkeypatch.setenv("KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS", "420")
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", fake_local_model_client)
    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    default_client = FakeClient("http://127.0.0.1:14829/v1", 240)

    chunk_client = engine._chunk_stage_model_client(default_client)

    assert chunk_client is not default_client
    assert chunk_client.timeout_seconds == 420
    assert default_client.timeout_seconds == 240
    assert created == [("http://127.0.0.1:14829/v1", 420.0)]


def test_engine_question_stage_per_token_timeout_uses_shared_positive_validation(monkeypatch) -> None:
    class FakeClient:
        endpoint = "http://127.0.0.1:14829/v1"
        timeout_seconds = 240

    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    monkeypatch.setenv("KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS", "not-a-number")

    with pytest.raises(LocalModelUnavailableError, match="KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS"):
        engine._question_stage_model_client(FakeClient())  # type: ignore[arg-type]


def test_engine_chunk_stage_per_token_timeout_rejects_non_positive_values(monkeypatch) -> None:
    class FakeClient:
        endpoint = "http://127.0.0.1:14829/v1"
        timeout_seconds = 240

    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    monkeypatch.setenv("KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS", "0")

    with pytest.raises(LocalModelUnavailableError, match="KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS"):
        engine._chunk_stage_model_client(FakeClient())  # type: ignore[arg-type]


def test_answer_verification_old_request_failure_cache_is_ignored(monkeypatch) -> None:
    class VerifierModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-verifier-old-request-cache", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            assert "Verify whether the candidate answer" in prompt
            return {
                "verification": {
                    "entailed": True,
                    "answer_type": "state",
                    "answer": "ready",
                    "evidence_span": "Aero Gate is ready.",
                    "reason": "grounded verifier answer",
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setattr(
        model_planner,
        "_read_cache",
        lambda path: {
            "accepted": False,
            "reason": "request_failed",
            "fresh_or_cached": "cache",
        },
    )
    monkeypatch.setattr(model_planner, "_write_cache", lambda path, payload: None)
    model = VerifierModel()

    result = call_model_answer_verification(
        "What is the state of Aero Gate?",
        {"answer_type": "state", "target_anchors": ["Aero Gate"], "requested_relation": "state"},
        "ready",
        [{"rel_path": "note.txt", "text": "Aero Gate is ready."}],
        [{"record_kind": "condition", "predicate": "state"}],
        model,  # type: ignore[arg-type]
    )

    assert result["accepted"] is True
    assert result["entailed"] is True
    assert result["answer"] == "ready"
    assert result["cache_context"]["n_predict"] == 128
    assert result["cache_context"]["expected_answer_type"] == "state"
    assert result["cache_context"]["evidence_count"] == 1
    assert result["cache_context"]["discourse_frame_count"] == 1
    assert result["cache_context"]["model_fingerprint"]["model_id"] == "fake-verifier-old-request-cache"
    assert model.calls == 1


def test_answer_verification_invalid_json_is_not_request_failure(monkeypatch) -> None:
    class VerifierModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-verifier-invalid-json", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            raise LocalModelJSONError("bad json", raw_text='{"verification":"unterminated', snippet='{"verification"')

    monkeypatch.setattr(model_planner, "_read_cache", lambda path: None)
    monkeypatch.setattr(model_planner, "_write_cache", lambda path, payload: None)
    model = VerifierModel()

    result = call_model_answer_verification(
        "What is the state of Aero Gate?",
        {"answer_type": "state", "target_anchors": ["Aero Gate"], "requested_relation": "state"},
        "ready",
        [{"rel_path": "note.txt", "text": "Aero Gate is ready."}],
        [{"record_kind": "condition", "predicate": "state"}],
        model,  # type: ignore[arg-type]
    )

    assert result["accepted"] is False
    assert result["reason"] == "invalid_json"
    assert result["raw_text"].startswith('{"verification"')
    assert result["cache_context"]["model_fingerprint"]["model_id"] == "fake-verifier-invalid-json"
    assert model.calls == 1


def test_drs_ingest_scan_unit_default_does_not_scale_to_full_large_context(monkeypatch) -> None:
    from knowmoredirt.ingest import _scan_unit_max_chars

    class LargeContextModel:
        def context_size(self) -> int:
            return 131072

    monkeypatch.delenv("KMD_SCAN_UNIT_MAX_CHARS", raising=False)
    monkeypatch.delenv("KMD_SCAN_UNIT_CONTEXT_MAX_CHARS", raising=False)

    assert _scan_unit_max_chars(LargeContextModel()) == 6000


def test_drs_ingest_scan_unit_context_cap_is_still_configurable(monkeypatch) -> None:
    from knowmoredirt.ingest import _scan_unit_max_chars

    class LargeContextModel:
        def context_size(self) -> int:
            return 131072

    monkeypatch.setenv("KMD_SCAN_UNIT_CONTEXT_MAX_CHARS", "8000")

    assert _scan_unit_max_chars(LargeContextModel()) == 8000


def test_local_model_drs_ingest_sends_noise_like_chunks_to_model_by_default(monkeypatch, tmp_path) -> None:
    from knowmoredirt import ingest as ingest_module

    class FakeClient:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-local-model", "context_size": 4096}

    text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    (tmp_path / "noise.txt").write_text(text, encoding="utf-8")
    calls: list[str] = []

    def fake_chunk_drs(chunk_text, client, **kwargs):
        calls.append(chunk_text)
        evidence = chunk_text[:16]
        return {
            "accepted": True,
            "reason": "compact_drs",
            "elapsed": 0.0,
            "cache_context": {},
            "drs": {
                "schema_version": "chunk-drs-v2",
                "source_id": "noise.txt",
                "referents": [{"id": "r0", "label": evidence, "kind": "unknown", "evidence_text": evidence}],
                "boxes": [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": ""}],
                "conditions": [
                    {
                        "id": "c0",
                        "predicate": "contains",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "text",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": evidence,
                                "value_type": "unknown",
                                "evidence_text": evidence,
                            }
                        ],
                        "evidence_text": evidence,
                    }
                ],
                "identity_hypotheses": [],
                "temporal_records": [],
                "evidence_spans": [],
            },
        }

    monkeypatch.delenv("KMD_ALLOW_PREMODEL_SEMANTIC_SKIP", raising=False)
    monkeypatch.setattr(ingest_module, "call_model_chunk_drs", fake_chunk_drs)

    store, _run_id, _docs, _sentences = ingest_module.ingest_folder(
        tmp_path,
        semantic_client=FakeClient(),
        use_drs_semantics=True,
    )

    assert calls
    assert store.execute("SELECT COUNT(*) FROM model_attempts WHERE task='chunk_drs'").fetchone()[0] >= 1


def test_local_model_drs_ingest_does_not_materialize_deterministic_semantics(monkeypatch, tmp_path) -> None:
    from knowmoredirt import ingest as ingest_module

    class FakeClient:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-local-model", "context_size": 4096}

    (tmp_path / "record.txt").write_text("Alice: owner\nBob: reviewer", encoding="utf-8")

    def fake_chunk_drs(chunk_text, client, **kwargs):
        evidence = "Alice" if "Alice" in chunk_text else chunk_text[:5]
        return {
            "accepted": True,
            "reason": "compact_drs",
            "elapsed": 0.0,
            "cache_context": {},
            "drs": {
                "schema_version": "chunk-drs-v2",
                "source_id": "record.txt",
                "referents": [{"id": "r0", "label": evidence, "kind": "person", "evidence_text": evidence}],
                "boxes": [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": ""}],
                "conditions": [
                    {
                        "id": "c0",
                        "predicate": "mentions",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "person",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": evidence,
                                "value_type": "person",
                                "evidence_text": evidence,
                            }
                        ],
                        "evidence_text": evidence,
                    }
                ],
                "identity_hypotheses": [],
                "temporal_records": [],
                "evidence_spans": [],
            },
        }

    monkeypatch.delenv("KMD_ALLOW_DETERMINISTIC_SEMANTICS_WITH_LOCAL_MODEL", raising=False)
    monkeypatch.setattr(ingest_module, "call_model_chunk_drs", fake_chunk_drs)

    store, _run_id, _docs, _sentences = ingest_module.ingest_folder(
        tmp_path,
        semantic_client=FakeClient(),
        use_drs_semantics=True,
    )

    assert store.execute("SELECT COUNT(*) FROM mentions WHERE source='deterministic'").fetchone()[0] == 0
    assert store.execute("SELECT COUNT(*) FROM frames WHERE source='deterministic_relation'").fetchone()[0] == 0
    assert store.execute("SELECT COUNT(*) FROM relations WHERE relation_type IN ('label_value', 'record_value', 'table_cell', 'temporal')").fetchone()[0] == 0
    assert store.execute("SELECT COUNT(*) FROM drs_conditions WHERE source='local_model_drs'").fetchone()[0] >= 1


def test_chunk_drs_repair_clears_ungrounded_optional_box_evidence() -> None:
    payload = {
        "drs": {
            "schema_version": "chunk-drs-v2",
            "source_id": "metadata/team.json",
            "referents": [
                {"id": "r0", "label": "Emma Miller", "kind": "person", "evidence_text": '"name": "Emma Miller"'},
            ],
            "boxes": [
                {
                    "id": "b0",
                    "kind": "asserted",
                    "parent_id": "",
                    "holder_referent_id": "",
                    "evidence_text": "Employee data",
                }
            ],
            "conditions": [
                {
                    "id": "c0",
                    "predicate": "employee_record",
                    "box_id": "b0",
                    "polarity": "positive",
                    "modality": "asserted",
                    "temporal_id": "",
                    "arguments": [
                        {
                            "role": "employee",
                            "target_kind": "referent",
                            "target_id": "r0",
                            "value": "Emma Miller",
                            "value_type": "person",
                            "evidence_text": '"name": "Emma Miller"',
                        }
                    ],
                    "evidence_text": '"name": "Emma Miller"',
                }
            ],
            "identity_hypotheses": [],
            "temporal_records": [],
            "evidence_spans": [],
        }
    }
    source_text = '{"name": "Emma Miller", "role": "Software Engineer"}'

    repaired = model_planner._repair_chunk_drs_payload(payload, source_text)
    validation = model_planner._validate_chunk_drs_payload(repaired, source_text)

    assert repaired["drs"]["boxes"][0]["evidence_text"] == ""
    assert validation["schema_valid"] is True


def test_query_drs_projection_does_not_inherit_deterministic_question_plan_defaults() -> None:
    from knowmoredirt.model_planner import query_frame_from_query_drs

    frame = query_frame_from_query_drs(
        "Which owner closed ticket BUG-123 yesterday?",
        {
            "schema_version": "query-drs-v2",
            "answer_type": "unknown",
            "answer_variables": [],
            "target_referents": [],
            "box_requirements": [],
            "requested_conditions": [],
            "constraints": [],
            "temporal_records": [],
            "temporal_scope": "",
            "aggregation": "",
            "requires_evidence": True,
        },
    )

    assert frame is not None
    assert frame["target_anchors"] == ()
    assert frame["answer_variables"] == ()
    assert frame["relation_terms"] == ()
    assert frame["constraints"] == ()
    assert frame["requested_relation"] == ""
    assert frame["temporal_scope"] == ""
    assert frame["answer_type"] == "unknown"
    assert frame["source"] == "model_query_drs"


def test_engine_model_mode_answer_calls_local_model_before_deterministic_source_handlers() -> None:
    from knowmoredirt.engine import KnowMoreDiRTEngine
    from knowmoredirt.models import Answer

    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    engine._use_local_model = True
    engine.last_answer = None
    calls: list[str] = []

    def model_answer(question: str) -> Answer:
        calls.append("model")
        return Answer("model-owned answer")

    engine._answer_with_local_model = model_answer
    engine._cleanup_public_answer = lambda answer, question=None: answer

    deterministic_handlers = [
        "_answer_with_labeled_attribute_source",
        "_answer_with_actor_role_ids_source",
        "_answer_with_reference_role_chain_source",
        "_answer_with_review_or_approval_source",
        "_answer_with_clause_table_message_source",
        "_answer_with_missing_organization_owner_source",
        "_answer_with_discussion_belief_source",
        "_answer_with_correction_owner_source",
        "_answer_with_precise_source_content",
        "_answer_with_table_field_source",
        "_answer_with_discourse_clause_source",
        "_answer_with_structured_object_source",
        "_answer_with_commit_hash_source",
        "_answer_with_exact_source_field",
        "_answer_with_arithmetic_source",
        "_answer_with_definition_source_explanation",
        "_answer_with_generic_sentence_source",
        "_answer_with_generic_labeled_field_source",
        "_answer_with_action_holder_source",
        "_answer_with_negated_action_source",
        "_answer_with_temporal_source_records",
        "_answer_with_source_rows",
        "_answer_with_row_field_source",
        "_answer_with_boolean_source_explanation",
        "_answer_with_bounded_dspg",
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError("deterministic source-answer handler ran before model-owned answer returned")

    for name in deterministic_handlers:
        setattr(engine, name, forbidden)

    result = engine.answer("Which owner is listed for the artifact?")

    assert result.text == "model-owned answer"
    assert calls == ["model"]


def test_engine_model_mode_unknown_does_not_fall_back_to_deterministic_source_handlers() -> None:
    from knowmoredirt.engine import KnowMoreDiRTEngine
    from knowmoredirt.models import Answer

    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    engine._use_local_model = True
    engine.last_answer = None
    engine._answer_with_local_model = lambda question: Answer("unknown", reason="model decided unknown")
    engine._cleanup_public_answer = lambda answer, question=None: answer

    def forbidden(*args, **kwargs):
        raise AssertionError("deterministic source-answer fallback ran after model-owned unknown")

    for name in [
        "_answer_with_generic_sentence_source",
        "_answer_with_labeled_attribute_source",
        "_answer_with_table_field_source",
        "_answer_with_source_rows",
        "_answer_with_bounded_dspg",
    ]:
        setattr(engine, name, forbidden)

    result = engine.answer("Who owns the artifact?")

    assert result.text == "unknown"
    assert result.reason == "model decided unknown"


def test_ingest_model_structured_failures_abort_initialize_boundary() -> None:
    from knowmoredirt.ingest import _raise_model_request_failed

    for reason in ["request_failed", "invalid_json", "schema_validation_failed", "grounding_validation_failed"]:
        with pytest.raises(LocalModelUnavailableError, match=reason):
            _raise_model_request_failed({"reason": reason, "error": reason, "cache_context": {"source_rel_path": "x"}}, "chunk DRS ingest")


def test_ingest_model_materialization_failure_aborts_initialize_boundary() -> None:
    from knowmoredirt.ingest import _raise_model_materialization_failed

    with pytest.raises(LocalModelUnavailableError, match="duplicate_or_missing_condition_id"):
        _raise_model_materialization_failed(
            {"accepted": True, "reason": "staged_fallback", "cache_context": {"source_rel_path": "x"}},
            {"accepted": False, "reason": "schema_validation_failed", "errors": ["duplicate_or_missing_condition_id"]},
            "chunk DRS ingest",
        )


def test_ingest_model_materialization_success_does_not_abort_initialize_boundary() -> None:
    from knowmoredirt.ingest import _raise_model_materialization_failed

    _raise_model_materialization_failed(
        {"accepted": True, "reason": "staged_fallback"},
        {"accepted": True, "reason": "materialized", "inserted": {"drs_conditions": 1}},
        "chunk DRS ingest",
    )


def test_ingest_model_materialized_result_does_not_abort_initialize_boundary() -> None:
    from knowmoredirt.ingest import _raise_model_request_failed

    _raise_model_request_failed({"accepted": True, "reason": "compact_drs"}, "chunk DRS ingest")


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

    client = LocalModelClient(endpoint="http://127.0.0.1:14829/v1", timeout_seconds=30)

    assert client.model_id() == "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    assert client.context_size() == 24576
    assert client.context_source() == "/slots[0].n_ctx"
    assert client.request_settings()["top_k"] == 17
    assert client.request_settings()["min_p"] == 0.03
    fingerprint = client.cache_fingerprint()
    assert fingerprint["context_size"] == 24576
    transport = fingerprint["transport_settings"]
    assert transport["api"] == "chat"
    assert transport["cache_prompt"] is True
    assert transport["min_constrained_json_tokens"] == 16384
    assert transport["constraint_mode"] == "auto"
    assert transport["native_constraints"] is True
    assert transport["reasoning_control_token_model"] is False
    chunk_transport = chunk_drs_cache_context(client)["model_fingerprint"]["transport_settings"]
    assert chunk_transport["api"] == "chat"
    assert chunk_transport["cache_prompt"] is True
    assert chunk_transport["min_constrained_json_tokens"] == 16384
    assert chunk_transport["constraint_mode"] == "auto"
    assert chunk_transport["native_constraints"] is True


def test_local_model_auto_constraints_stay_native_for_reasoning_control_models(monkeypatch) -> None:
    def fake_urlopen(request, timeout: float = 0) -> FakeHTTPResponse:
        url = str(getattr(request, "full_url", request))
        if url.endswith("/v1/models"):
            return FakeHTTPResponse({"data": [{"id": "/models/gpt-oss-120b.gguf", "meta": {"n_ctx_train": 131072}}]})
        if url.endswith("/slots"):
            return FakeHTTPResponse([{"n_ctx": 131072, "params": {}}])
        if url.endswith("/props"):
            return FakeHTTPResponse({"model_alias": "gpt-oss-120b.gguf", "default_generation_settings": {}})
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.delenv("KMD_LOCAL_MODEL_CONSTRAINT_MODE", raising=False)

    client = LocalModelClient(endpoint="http://127.0.0.1:14829/v1", timeout_seconds=30)
    transport = client.transport_settings()

    assert transport["constraint_mode"] == "auto"
    assert transport["reasoning_control_token_model"] is True
    assert transport["native_constraints"] is True


def test_local_model_prompt_constraint_mode_is_diagnostic_only_without_override(monkeypatch) -> None:
    def fake_urlopen(request, timeout: float = 0) -> FakeHTTPResponse:
        url = str(getattr(request, "full_url", request))
        if url.endswith("/v1/models"):
            return FakeHTTPResponse({"data": [{"id": "/models/gpt-oss-120b.gguf", "meta": {"n_ctx_train": 131072}}]})
        if url.endswith("/slots"):
            return FakeHTTPResponse([{"n_ctx": 131072, "params": {}}])
        if url.endswith("/props"):
            return FakeHTTPResponse({"model_alias": "gpt-oss-120b.gguf", "default_generation_settings": {}})
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("KMD_LOCAL_MODEL_CONSTRAINT_MODE", "prompt")

    client = LocalModelClient(endpoint="http://127.0.0.1:14829/v1", timeout_seconds=30)
    transport = client.transport_settings()

    assert transport["constraint_mode"] == "prompt"
    assert transport["reasoning_control_token_model"] is True
    assert transport["native_constraints"] is False
    with pytest.raises(LocalModelUnavailableError):
        client.complete_json(
            "Return JSON.",
            json_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
            },
        )


def test_local_model_prompt_constraint_mode_requires_explicit_soft_json_override(monkeypatch) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout: float = 0) -> FakeHTTPResponse:
        url = str(getattr(request, "full_url", request))
        calls.append(url)
        if url.endswith("/v1/models"):
            return FakeHTTPResponse({"data": [{"id": "/models/gpt-oss-120b.gguf", "meta": {"n_ctx_train": 131072}}]})
        if url.endswith("/slots"):
            return FakeHTTPResponse([{"n_ctx": 131072, "params": {}}])
        if url.endswith("/props"):
            return FakeHTTPResponse({"model_alias": "gpt-oss-120b.gguf", "default_generation_settings": {}})
        if url.endswith("/v1/chat/completions"):
            payload = json.loads(getattr(request, "data", b"{}").decode("utf-8"))
            assert "response_format" not in payload
            return FakeHTTPResponse(
                lines=[
                    b'data: {"choices":[{"delta":{"content":"{\\"answer\\":\\"ok\\"}"}}]}',
                    b"data: [DONE]",
                ]
            )
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("KMD_LOCAL_MODEL_CONSTRAINT_MODE", "prompt")
    monkeypatch.setenv("KMD_LOCAL_MODEL_ALLOW_PROMPT_CONSTRAINTS", "1")

    client = LocalModelClient(endpoint="http://127.0.0.1:14829/v1", timeout_seconds=30)
    result = client.complete_json(
        "Return JSON.",
        json_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        },
    )

    assert result["answer"] == "ok"
    assert result["_model_constraint_settings"]["mode"] == "prompt_json_schema"
    assert result["_model_constraint_settings"]["native_constraints_applied"] is False


def test_engine_required_probe_uses_client_endpoint_normalization(monkeypatch) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout: float = 0) -> FakeHTTPResponse:
        url = str(getattr(request, "full_url", request))
        calls.append(url)
        if url == "http://127.0.0.1:14829/v1/models":
            return FakeHTTPResponse({"data": [{"id": "test-model", "meta": {"n_ctx_train": 4096}}]})
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:14829/completion")

    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)

    client = engine._required_local_model_client()
    assert client.endpoint == "http://127.0.0.1:14829/completion"
    assert calls == ["http://127.0.0.1:14829/v1/models"]


def test_verifier_diagnostic_frames_are_capped(monkeypatch) -> None:
    class FakeCursor:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self.rows = rows

        def fetchall(self) -> list[dict[str, object]]:
            return self.rows

    class FakeStore:
        def __init__(self) -> None:
            self.params: tuple[object, ...] | None = None

        def execute(self, query: str, params: tuple[object, ...]) -> FakeCursor:
            self.params = params
            limit = int(params[-1])
            rows = [
                {
                    "rel_path": "note.txt",
                    "predicate": f"predicate_{index}",
                    "trigger_surface": f"trigger_{index}",
                    "source": "model",
                    "kind": "asserted",
                }
                for index in range(20)
            ]
            return FakeCursor(rows[:limit])

    monkeypatch.setenv("KMD_VERIFIER_DISCOURSE_FRAME_LIMIT", "8")
    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    store = FakeStore()
    engine.store = store

    frames = engine._diagnostic_frames_for_answer(Answer("ready", evidence=[Evidence("note.txt", "Aero Gate is ready.")]))

    assert len(frames) == 8
    assert store.params == ("note.txt", 8)


def test_engine_requires_reachable_local_model_in_normal_runtime(monkeypatch, tmp_path) -> None:
    def fake_urlopen(request, timeout: float = 0) -> object:
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.delenv("KMD_TEST_ALLOW_NO_MODEL", raising=False)
    monkeypatch.setenv("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:9/v1")
    (tmp_path / "note.txt").write_text("Aster Lock is guarded by Mira Vale.\n", encoding="utf-8")

    with pytest.raises(LocalModelUnavailableError, match="requires a reachable localhost llama.cpp"):
        KnowMoreDiRTEngine(tmp_path)


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
    monkeypatch.setenv("KMD_LOCAL_MODEL_API", "completion")
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
    assert requests[0]["body"]["n_predict"] == 16384
    assert requests[0]["body"]["json_schema"]["type"] == "object"
    assert "grammar" not in requests[0]["body"]
    assert parsed["_model_constraint_settings"]["mode"] == "completion_json_schema"
    assert parsed["_model_constraint_settings"]["requested_n_predict"] == 64
    assert parsed["_model_constraint_settings"]["effective_n_predict"] == 16384



def test_local_model_client_chat_json_schema_uses_response_format(monkeypatch) -> None:
    requests: list[dict[str, Any]] = []

    def fake_urlopen(request, timeout: float = 0) -> FakeHTTPResponse:
        url = getattr(request, "full_url", request)
        if str(url).endswith("/v1/models"):
            return FakeHTTPResponse({"data": [{"id": "test-model", "meta": {"n_ctx_train": 4096}}]})
        if str(url).endswith("/slots"):
            return FakeHTTPResponse([{"n_ctx": 4096, "params": {"top_k": 40, "min_p": 0.05, "repeat_penalty": 1.0}}])
        if str(url).endswith("/props"):
            return FakeHTTPResponse({"default_generation_settings": {"n_ctx": 4096, "params": {}}})
        if str(url).endswith("/v1/chat/completions"):
            body = json.loads(request.data.decode("utf-8"))
            requests.append({"url": str(url), "body": body})
            return FakeHTTPResponse(lines=[('data: ' + json.dumps({"choices": [{"delta": {"content": '{\"ok\":true}'}}]}) + '\n\n').encode()])
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("KMD_LOCAL_MODEL_API", "chat")
    client = LocalModelClient(endpoint="http://127.0.0.1:14829/v1", timeout_seconds=30)

    parsed = client.complete_json("return ok", n_predict=64, json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}})

    assert parsed["ok"] is True
    body = requests[0]["body"]
    assert "json_schema" not in body
    assert body["max_tokens"] == 16384
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["properties"]["ok"]["type"] == "boolean"
    assert parsed["_model_constraint_settings"]["mode"] == "chat_response_format_json_schema"
    assert parsed["_model_constraint_settings"]["requested_n_predict"] == 64
    assert parsed["_model_constraint_settings"]["effective_n_predict"] == 16384


def test_local_model_client_stream_uses_per_token_read_timeout_only(monkeypatch) -> None:
    def fake_urlopen(request, timeout: float = 0) -> FakeHTTPResponse:
        url = getattr(request, "full_url", request)
        if str(url).endswith("/v1/models"):
            return FakeHTTPResponse({"data": [{"id": "test-model", "meta": {"n_ctx_train": 4096}}]})
        if str(url).endswith("/slots"):
            return FakeHTTPResponse([{"n_ctx": 4096, "params": {"top_k": 40, "min_p": 0.05, "repeat_penalty": 1.0}}])
        if str(url).endswith("/props"):
            return FakeHTTPResponse({"default_generation_settings": {"n_ctx": 4096, "params": {}}})
        if str(url).endswith("/completion"):
            assert timeout == 2
            return FakeHTTPResponse(
                lines=[
                    b'data: {"content": "{\\"ok\\":"}\n\n',
                    b'data: {"content": "true}"}\n\n',
                ]
            )
        raise AssertionError(f"unexpected URL {url}")

    ticks = iter([0.0, 0.5, 3.0])

    def fake_time() -> float:
        return next(ticks, 3.0)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("knowmoredirt.model.time.time", fake_time)
    monkeypatch.setenv("KMD_LOCAL_MODEL_API", "completion")
    client = LocalModelClient(endpoint="http://127.0.0.1:14829/v1", timeout_seconds=2)
    client.server_metadata()

    parsed = client.complete_json("return ok", n_predict=64)

    assert parsed["ok"] is True


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
    assert result["cache_context"]["context_budget"]["context_budget_policy"] == CHUNK_FRAME_CONTEXT_BUDGET_POLICY
    assert result["cache_context"]["model_fingerprint"]["model_id"] == "fake"
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
    assert result["cache_context"]["context_budget"]["runtime_context_size"] == 4096
    assert result["cache_context"]["model_fingerprint"]["model_id"] == "fake-invalid-frame-budget"


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
    assert result["cache_context"]["repair"] is False
    assert result["cache_context"]["n_predict"] == 128
    assert result["cache_context"]["model_fingerprint"]["model_id"] == "fake-query-evidence-invalid-repair"
    assert result["repair_cache_context"]["repair"] is True
    assert result["repair_cache_context"]["model_fingerprint"]["model_id"] == "fake-query-evidence-invalid-repair"
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
    assert first["cache_context"]["repair"] is False
    assert first["repair_cache_context"]["repair"] is True
    assert second["accepted"] is True
    assert second["fresh_or_cached"] == "fresh_repair"
    assert second["cache_context"]["repair"] is True
    assert second["cache_context"]["model_fingerprint"]["model_id"] == "fake-query-evidence-repair-retry"
    assert model.primary_calls == 2
    assert model.repair_calls == 2


def test_query_evidence_old_repair_request_failure_cache_is_ignored(monkeypatch) -> None:
    class EvidenceAnswerModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-query-evidence-old-request-cache", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            assert "bounded DRT/DSPG question analysis" in prompt
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
                    "sufficient_evidence": True,
                    "answer_type": "state",
                    "answer": "ready",
                    "evidence_span": "Aero Gate is ready.",
                    "reason": "directly supported",
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setattr(
        model_planner,
        "_read_cache",
        lambda path: {
            "accepted": False,
            "reason": "invalid_json",
            "repair_failure_reason": "request_failed",
            "fresh_or_cached": "cache",
        },
    )
    monkeypatch.setattr(model_planner, "_write_cache", lambda path, payload: None)
    model = EvidenceAnswerModel()

    result = call_model_query_evidence_answer(
        "What is the state of Aero Gate?",
        [{"rel_path": "note.txt", "text": "Aero Gate is ready."}],
        model,  # type: ignore[arg-type]
    )

    assert result["accepted"] is True
    assert result["answer"] == "ready"
    assert result["fresh_or_cached"] == "fresh"
    assert model.calls == 1


def test_answer_canonicalization_old_request_failure_cache_is_ignored(monkeypatch) -> None:
    class CanonicalizationModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-canonicalization-old-request-cache", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            assert "Canonicalize a model-selected final answer" in prompt
            return {
                "canonical_answer": {
                    "answer": "ready",
                    "evidence_span": "Aero Gate is ready.",
                    "reason": "already grounded",
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setattr(
        model_planner,
        "_read_cache",
        lambda path: {
            "accepted": False,
            "reason": "request_failed",
            "fresh_or_cached": "cache",
        },
    )
    monkeypatch.setattr(model_planner, "_write_cache", lambda path, payload: None)
    model = CanonicalizationModel()

    result = call_model_answer_canonicalization(
        "What is the state of Aero Gate?",
        "ready",
        "state",
        [{"rel_path": "note.txt", "text": "Aero Gate is ready."}],
        model,  # type: ignore[arg-type]
    )

    assert result["accepted"] is True
    assert result["answer"] == "ready"
    assert result["fresh_or_cached"] == "fresh"
    assert result["cache_context"]["n_predict"] == 96
    assert result["cache_context"]["answer_type"] == "state"
    assert result["cache_context"]["evidence_count"] == 1
    assert result["cache_context"]["model_fingerprint"]["model_id"] == "fake-canonicalization-old-request-cache"
    assert model.calls == 1


def test_answer_canonicalization_accepts_grounded_unknown(monkeypatch) -> None:
    class CanonicalizationModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-canonicalization-grounded-unknown", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            assert "return answer='unknown'" in prompt
            return {
                "canonical_answer": {
                    "answer": "unknown",
                    "evidence_span": "The note says no complete binding is available.",
                    "reason": "candidate is an absence statement",
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setattr(model_planner, "_read_cache", lambda path: None)
    monkeypatch.setattr(model_planner, "_write_cache", lambda path, payload: None)

    result = call_model_answer_canonicalization(
        "Which bound value is requested?",
        "The note says no complete binding is available.",
        "content_phrase",
        [{"rel_path": "note.txt", "text": "The note says no complete binding is available."}],
        CanonicalizationModel(),  # type: ignore[arg-type]
    )

    assert result["accepted"] is True
    assert result["answer"] == "unknown"
    assert result["evidence_span"] == "The note says no complete binding is available."


def test_answer_canonicalization_accepts_source_grounded_deictic_rewrite(monkeypatch) -> None:
    class CanonicalizationModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-canonicalization-deictic-rewrite", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            assert "speaker/source identity" in prompt
            assert "source-resolved answer" in prompt
            return {
                "canonical_answer": {
                    "answer": "Drew planned to repair valve.py tomorrow, not today.",
                    "evidence_span": "I plan to repair valve.py tomorrow, not today.",
                    "reason": "speaker evidence grounds the deictic rewrite",
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setattr(model_planner, "_read_cache", lambda path: None)
    monkeypatch.setattr(model_planner, "_write_cache", lambda path, payload: None)

    result = call_model_answer_canonicalization(
        "What did the forwarded Drew message say about repairing valve.py?",
        "I plan to repair valve.py tomorrow, not today",
        "content_phrase",
        [{"rel_path": "thread.eml", "text": "From: Drew Lane\nI plan to repair valve.py tomorrow, not today."}],
        CanonicalizationModel(),  # type: ignore[arg-type]
    )

    assert result["accepted"] is True
    assert result["answer"] == "Drew planned to repair valve.py tomorrow, not today."


def test_source_resolved_answer_rewrites_deictic_reported_content(monkeypatch) -> None:
    class SourceResolutionModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-source-resolution-rewrite", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            assert "public reported answer" in prompt
            assert "past reporting auxiliaries" in prompt
            assert json_schema is None
            return {
                "source_resolved_answer": {
                    "answer": "Taylor expected patch.py to land tomorrow.",
                    "evidence_span": "I expect patch.py to land tomorrow.",
                    "reason": "source identity grounds the deictic paraphrase",
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setattr(model_planner, "_read_cache", lambda path: None)
    monkeypatch.setattr(model_planner, "_write_cache", lambda path, payload: None)
    monkeypatch.setenv("KMD_LOCAL_MODEL_GRAMMAR", "1")

    result = call_model_source_resolved_answer(
        "What did the forwarded Taylor message say about patch.py?",
        "I expect patch.py to land tomorrow.",
        "content_phrase",
        [{"rel_path": "thread.eml", "text": "From: Taylor Quinn\nI expect patch.py to land tomorrow."}],
        SourceResolutionModel(),  # type: ignore[arg-type]
    )

    assert result["accepted"] is True
    assert result["answer"] == "Taylor expected patch.py to land tomorrow."
    assert result["cache_context"]["n_predict"] == 160
    assert result["cache_context"]["constraint_mode"] == "gbnf"


def test_answer_canonicalization_invalid_json_is_not_request_failure(monkeypatch) -> None:
    class CanonicalizationModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-canonicalization-invalid-json", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            raise LocalModelJSONError(
                "bad json",
                raw_text='{"canonical_answer":{"answer":"ready"',
                snippet='{"canonical_answer"',
            )

    monkeypatch.setattr(model_planner, "_read_cache", lambda path: None)
    monkeypatch.setattr(model_planner, "_write_cache", lambda path, payload: None)
    model = CanonicalizationModel()

    result = call_model_answer_canonicalization(
        "What is the state of Aero Gate?",
        "ready now",
        "state",
        [{"rel_path": "note.txt", "text": "Aero Gate is ready now."}],
        model,  # type: ignore[arg-type]
    )

    assert result["accepted"] is False
    assert result["reason"] == "invalid_json"
    assert result["raw_text"].startswith('{"canonical_answer"')
    assert result["cache_context"]["model_fingerprint"]["model_id"] == "fake-canonicalization-invalid-json"
    assert model.calls == 1


def test_identity_canonicalization_old_invalid_cache_is_ignored(monkeypatch) -> None:
    class IdentityModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-identity-old-invalid-cache", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            assert "same relevant DRS context" in prompt
            return {
                "canonicalization": {
                    "same_referent": True,
                    "answer": "Aero Gate",
                    "evidence_span": "Aero Gate",
                    "reason": "identity grounded in evidence",
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setattr(
        model_planner,
        "_read_cache",
        lambda path: {
            "accepted": False,
            "reason": "invalid_json",
            "fresh_or_cached": "cache",
        },
    )
    monkeypatch.setattr(model_planner, "_write_cache", lambda path, payload: None)
    model = IdentityModel()

    result = call_model_identity_canonicalization(
        "Who is ready?",
        "Aero",
        ["Aero Gate"],
        [{"rel_path": "note.txt", "text": "Aero Gate is ready."}],
        model,  # type: ignore[arg-type]
    )

    assert result["accepted"] is True
    assert result["same_referent"] is True
    assert result["answer"] == "Aero Gate"
    assert result["fresh_or_cached"] == "fresh"
    assert result["cache_context"]["n_predict"] == 96
    assert result["cache_context"]["fuller_candidate_count"] == 1
    assert result["cache_context"]["evidence_count"] == 1
    assert result["cache_context"]["model_fingerprint"]["model_id"] == "fake-identity-old-invalid-cache"
    assert model.calls == 1


def test_identity_canonicalization_invalid_json_is_not_request_failure(monkeypatch) -> None:
    class IdentityModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-identity-invalid-json", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            raise LocalModelJSONError(
                "bad json",
                raw_text='{"canonicalization":{"same_referent":true',
                snippet='{"canonicalization"',
            )

    monkeypatch.setattr(model_planner, "_read_cache", lambda path: None)
    monkeypatch.setattr(model_planner, "_write_cache", lambda path, payload: None)
    model = IdentityModel()

    result = call_model_identity_canonicalization(
        "Who is ready?",
        "Aero",
        ["Aero Gate"],
        [{"rel_path": "note.txt", "text": "Aero Gate is ready."}],
        model,  # type: ignore[arg-type]
    )

    assert result["accepted"] is False
    assert result["reason"] == "invalid_json"
    assert result["raw_text"].startswith('{"canonicalization"')
    assert result["cache_context"]["model_fingerprint"]["model_id"] == "fake-identity-invalid-json"
    assert model.calls == 1


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

    result = call_model_query_plan_test_only("What is the current state of Delta Well?", model, n_predict=64)  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["temporal_scope"] == "latest"
    assert "temporal_scope must be" in model.prompt
    assert "aggregation must be" in model.prompt
    assert result["operator_schema_policy"] == QUERY_OPERATOR_SCHEMA_POLICY
    assert result["cache_context"]["n_predict"] == 64
    assert result["cache_context"]["operator_schema_policy"] == QUERY_OPERATOR_SCHEMA_POLICY
    assert result["cache_context"]["model_fingerprint"]["model_id"] == "fake-query-frame-temporal"
    assert model.json_schema is not None
    query_schema = model.json_schema["properties"]["query_frame"]
    assert query_schema["properties"]["temporal_scope"]["enum"] == ["", "earliest", "latest"]
    assert query_schema["properties"]["aggregation"]["enum"] == ["", "count", "list", "set"]


def test_query_frame_invalid_json_failure_is_retryable(monkeypatch, tmp_path) -> None:
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

    first = call_model_query_plan_test_only("What state is Delta Well in?", model, n_predict=64)  # type: ignore[arg-type]
    second = call_model_query_plan_test_only("What state is Delta Well in?", model, n_predict=64)  # type: ignore[arg-type]

    assert first["accepted"] is False
    assert first["reason"] == "invalid_json"
    assert first["cache_context"]["n_predict"] == 64
    assert second["accepted"] is False
    assert second["reason"] == "invalid_json"
    assert second["cache_context"]["model_fingerprint"]["model_id"] == "fake-query-frame-invalid-cache"
    assert model.calls == 2


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

    first = call_model_query_plan_test_only("What is the current state of Delta Well?", model, n_predict=64)  # type: ignore[arg-type]
    second = call_model_query_plan_test_only("What is the current state of Delta Well?", model, n_predict=64)  # type: ignore[arg-type]

    assert first["accepted"] is False
    assert first["reason"] == "request_failed"
    assert first["cache_context"]["n_predict"] == 64
    assert second["accepted"] is True
    assert second["temporal_scope"] == "latest"
    assert second["cache_context"]["model_fingerprint"]["model_id"] == "fake-query-frame-request-retry"
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
    assert result["cache_context"]["context_budget"]["runtime_context_size"] == 8192
    assert result["cache_context"]["source_rel_path"] == "note.txt"
    assert result["cache_context"]["model_fingerprint"]["model_id"] == "fake-drs"
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
    assert first["cache_context"]["context_budget"]["runtime_context_size"] == 8192
    assert first["cache_context"]["model_fingerprint"]["model_id"] == "fake-chunk-drs-request-retry"
    assert second["accepted"] is True
    assert second["cache_context"]["source_rel_path"] == "note.txt"
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


def test_compact_query_drs_undercovered_slot_falls_back_to_full_model(monkeypatch, tmp_path) -> None:
    class CompactUndercoverageModel:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-compact-undercoverage", "context_size": 8192}

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, Any] | None = None,
        ) -> dict[str, object]:
            self.prompts.append(prompt)
            if "compact DRS query data" in prompt:
                return {
                    "a": "identifier",
                    "answer": "code",
                    "targets": ["Alpha Node"],
                    "predicates": ["belongs to"],
                    "constraints": [],
                    "temporal_scope": "",
                    "aggregation": "",
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            return {
                "query_drs": {
                    "schema_version": "query-drs-v3",
                    "question": "Which priority code belongs to Alpha Node?",
                    "answer_variables": [
                        {
                            "id": "qv0",
                            "label": "priority code",
                            "answer_type": "identifier",
                            "evidence_text": "priority code",
                        }
                    ],
                    "target_referents": [
                        {"id": "qr0", "label": "Alpha Node", "kind": "entity", "evidence_text": "Alpha Node"}
                    ],
                    "requested_conditions": [
                        {
                            "id": "qc0",
                            "predicate": "belongs to",
                            "box_id": "",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "answer",
                                    "target_kind": "answer_variable",
                                    "target_id": "qv0",
                                    "value": "",
                                    "value_type": "identifier",
                                    "evidence_text": "priority code",
                                },
                                {
                                    "role": "argument",
                                    "target_kind": "referent",
                                    "target_id": "qr0",
                                    "value": "",
                                    "value_type": "entity",
                                    "evidence_text": "Alpha Node",
                                },
                            ],
                            "evidence_text": "Which priority code belongs to Alpha Node?",
                        }
                    ],
                    "constraints": [],
                    "box_requirements": [],
                    "temporal_scope": "",
                    "aggregation": "",
                    "answer_type": "identifier",
                    "requires_evidence": True,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_FORCE_COMPACT_MODEL_PATH", "1")
    monkeypatch.setenv("KMD_QUERY_DRS_CACHE_DIR", str(tmp_path / "query-drs-cache"))
    model = CompactUndercoverageModel()

    result = call_model_query_drs("Which priority code belongs to Alpha Node?", model)  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["query_drs"]["answer_variables"][0]["label"] == "priority code"
    assert result["compact_fallback_attempt"]["full_accepted"] is True
    assert len(model.prompts) == 2


def test_compact_query_drs_missing_relation_is_repaired_from_uncovered_tokens(monkeypatch, tmp_path) -> None:
    class MissingRelationCompactModel:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-compact-missing-relation", "context_size": 8192}

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, Any] | None = None,
        ) -> dict[str, object]:
            self.prompts.append(prompt)
            if "compact DRS query data" in prompt:
                return {
                    "a": "content_phrase",
                    "answer": "lumo",
                    "targets": ["lumo"],
                    "predicates": [],
                    "constraints": [],
                    "temporal_scope": "",
                    "aggregation": "",
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
            }
            raise AssertionError("compact query repair should avoid full query call")

    monkeypatch.setenv("KMD_FORCE_COMPACT_MODEL_PATH", "1")
    monkeypatch.setenv("KMD_QUERY_DRS_CACHE_DIR", str(tmp_path / "query-drs-cache"))
    model = MissingRelationCompactModel()

    result = call_model_query_drs("What does lumo mean?", model)  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["query_drs"]["requested_conditions"][0]["predicate"] == "mean"
    assert "compact_fallback_attempt" not in result
    assert len(model.prompts) == 1


def test_compact_query_drs_repairs_cached_missing_relation(monkeypatch, tmp_path) -> None:
    class CachedMissingRelationModel:
        def __init__(self) -> None:
            self.calls = 0
            self.timeout_seconds = 420

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-cached-missing-query-relation", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            raise AssertionError("cached query repair should avoid live model call")

    question = "What does mave mean?"
    model = CachedMissingRelationModel()
    monkeypatch.setenv("KMD_FORCE_COMPACT_MODEL_PATH", "1")
    monkeypatch.setenv("KMD_QUERY_DRS_CACHE_DIR", str(tmp_path / "query-drs-cache"))
    settings = {
        "n_predict": 64,
        "schema": model_planner.QUERY_DRS_SCHEMA_VERSION,
        "compact_plan_policy": model_planner.QUERY_DRS_COMPACT_PLAN_POLICY,
        **model_planner._constraint_settings(
            model_planner.QUERY_DRS_GRAMMAR,
            model_planner.COMPACT_QUERY_DRS_JSON_SCHEMA,
            model_planner.QUERY_DRS_SCHEMA_VERSION,
        ),
    }
    prompt_hash = model_planner._cache_hash(
        "query_drs_compact",
        model_planner.build_compact_query_drs_prompt(question),
        model,  # type: ignore[arg-type]
        settings,
    )
    cache_path = model_planner._cache_path("KMD_QUERY_DRS_CACHE_DIR", prompt_hash)
    assert cache_path is not None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "prompt_hash": prompt_hash,
                "query_drs": {
                    "schema_version": "query-drs-v3",
                    "question": question,
                    "answer_variables": [
                        {
                            "id": "qv0",
                            "label": "mave",
                            "answer_type": "content_phrase",
                            "evidence_text": "mave",
                        }
                    ],
                    "target_referents": [
                        {"id": "qr0", "label": "mave", "kind": "unknown", "evidence_text": "mave"}
                    ],
                    "requested_conditions": [],
                    "constraints": [],
                    "box_requirements": [],
                    "temporal_records": [],
                    "temporal_scope": "",
                    "aggregation": "",
                    "answer_type": "content_phrase",
                    "requires_evidence": True,
                },
                "raw_text": "{}",
            }
        ),
        encoding="utf-8",
    )

    result = call_model_query_drs(question, model)  # type: ignore[arg-type]

    assert model.calls == 0
    assert result["query_drs"]["requested_conditions"][0]["predicate"] == "mean"
    assert result["query_drs"]["requested_conditions"][0]["arguments"][0]["target_kind"] == "answer_variable"


def test_compact_query_drs_repairs_cached_answer_argument_kind(monkeypatch, tmp_path) -> None:
    class CachedWrongAnswerArgumentModel:
        def __init__(self) -> None:
            self.calls = 0
            self.timeout_seconds = 420

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-cached-wrong-answer-argument", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            raise AssertionError("cached query repair should avoid live model call")

    question = "What does mave mean?"
    model = CachedWrongAnswerArgumentModel()
    monkeypatch.setenv("KMD_FORCE_COMPACT_MODEL_PATH", "1")
    monkeypatch.setenv("KMD_QUERY_DRS_CACHE_DIR", str(tmp_path / "query-drs-cache"))
    settings = {
        "n_predict": 64,
        "schema": model_planner.QUERY_DRS_SCHEMA_VERSION,
        "compact_plan_policy": model_planner.QUERY_DRS_COMPACT_PLAN_POLICY,
        **model_planner._constraint_settings(
            model_planner.QUERY_DRS_GRAMMAR,
            model_planner.COMPACT_QUERY_DRS_JSON_SCHEMA,
            model_planner.QUERY_DRS_SCHEMA_VERSION,
        ),
    }
    prompt_hash = model_planner._cache_hash(
        "query_drs_compact",
        model_planner.build_compact_query_drs_prompt(question),
        model,  # type: ignore[arg-type]
        settings,
    )
    cache_path = model_planner._cache_path("KMD_QUERY_DRS_CACHE_DIR", prompt_hash)
    assert cache_path is not None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "prompt_hash": prompt_hash,
                "query_drs": {
                    "schema_version": "query-drs-v3",
                    "question": question,
                    "answer_variables": [
                        {
                            "id": "qv0",
                            "label": "mave",
                            "answer_type": "content_phrase",
                            "evidence_text": "mave",
                        }
                    ],
                    "target_referents": [
                        {"id": "qr0", "label": "mave", "kind": "unknown", "evidence_text": "mave"}
                    ],
                    "requested_conditions": [
                        {
                            "id": "qc0",
                            "predicate": "mean",
                            "box_id": "",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "answer",
                                    "target_kind": "referent",
                                    "target_id": "qr0",
                                    "value": "",
                                    "value_type": "content_phrase",
                                    "evidence_text": "mave",
                                },
                                {
                                    "role": "argument",
                                    "target_kind": "referent",
                                    "target_id": "qr0",
                                    "value": "",
                                    "value_type": "unknown",
                                    "evidence_text": "mave",
                                },
                            ],
                            "evidence_text": question,
                        }
                    ],
                    "constraints": [],
                    "box_requirements": [],
                    "temporal_records": [],
                    "temporal_scope": "",
                    "aggregation": "",
                    "answer_type": "content_phrase",
                    "requires_evidence": True,
                },
                "raw_text": "{}",
            }
        ),
        encoding="utf-8",
    )

    result = call_model_query_drs(question, model)  # type: ignore[arg-type]

    assert model.calls == 0
    arguments = result["query_drs"]["requested_conditions"][0]["arguments"]
    assert arguments[0]["role"] == "answer"
    assert arguments[0]["target_kind"] == "answer_variable"
    assert arguments[0]["target_id"] == "qv0"
    assert arguments[1]["target_kind"] == "referent"


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
    assert result["cache_context"]["n_predict"] == 384
    assert result["cache_context"]["output_budget_policy"] == QUERY_DRS_DYNAMIC_OUTPUT_BUDGET_POLICY
    assert result["cache_context"]["operator_schema_policy"] == QUERY_OPERATOR_SCHEMA_POLICY
    assert result["cache_context"]["max_array_items"] == query_drs_array_max_items(384)
    assert result["cache_context"]["model_fingerprint"]["model_id"] == "fake-large-context-query-drs"
    assert result["cache_context"]["model_fingerprint"]["context_size"] == 32768
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
    assert first["cache_context"]["n_predict"] == 256
    assert first["cache_context"]["model_fingerprint"]["model_id"] == "fake-query-drs-retry"
    assert second["accepted"] is True
    assert second["cache_context"]["n_predict"] == 256
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


def test_query_drs_invalid_json_cache_is_retried(monkeypatch, tmp_path) -> None:
    class InvalidThenValidModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-query-drs-invalid-json-retry", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            if self.calls == 1:
                raise LocalModelJSONError("bad json", raw_text="not json", snippet="not json")
            return {
                "query_drs": {
                    "schema_version": "query-drs-v3",
                    "question": "Who reviewed Aero Gate?",
                    "answer_variables": [
                        {"id": "qv0", "label": "reviewer", "answer_type": "person", "evidence_text": "Who"}
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
    model = InvalidThenValidModel()

    first = call_model_query_drs("Who reviewed Aero Gate?", model)  # type: ignore[arg-type]
    second = call_model_query_drs("Who reviewed Aero Gate?", model)  # type: ignore[arg-type]

    assert first["accepted"] is False
    assert first["reason"] == "invalid_json"
    assert second["accepted"] is True
    assert model.calls == 2


def test_query_drs_request_failure_retries_smaller_budget(monkeypatch, tmp_path) -> None:
    class TimeoutThenSucceedsModel:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def context_size(self) -> int:
            return 32768

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-query-drs-staged-retry", "context_size": 32768}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls.append(n_predict)
            if len(self.calls) == 1:
                raise TimeoutError("local model request timed out waiting for the next streamed chunk")
            return {
                "query_drs": {
                    "schema_version": "query-drs-v3",
                    "question": "Who reviewed Aero Gate?",
                    "answer_variables": [
                        {"id": "qv0", "label": "reviewer", "answer_type": "person", "evidence_text": "Who"}
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
    model = TimeoutThenSucceedsModel()

    result = call_model_query_drs("Who reviewed Aero Gate?", model)  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert model.calls == [384, 256]
    assert result["request_failure_retry_index"] == 1
    assert result["query_drs_retry_attempts"][0]["reason"] == "request_failed"


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


def test_compact_chunk_drs_prompt_keeps_source_stated_definitions() -> None:
    prompt = build_compact_chunk_drs_prompt(
        'Glossary: "kave" means bright river.',
        rel_path="notes/terms.txt",
    )

    assert "definitions, meanings, names, aliases, and terminology" in prompt
    assert "bright river" in prompt
    assert "notes/terms.txt" in prompt


def test_compact_chunk_drs_regenerates_empty_legacy_definition_cache(monkeypatch, tmp_path) -> None:
    class DefinitionCompactModel:
        def __init__(self) -> None:
            self.calls = 0
            self.timeout_seconds = 240

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-definition-compact", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            return {
                "facts": [
                    {
                        "p": "means",
                        "agent": "kave",
                        "value": "bright river",
                        "e": '"kave" means bright river.',
                    }
                ],
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    text = 'Glossary: "kave" means bright river.'
    rel_path = "notes/terms.txt"
    model = DefinitionCompactModel()
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    source_text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    settings = {
        "n_predict": 72,
        "schema": model_planner.CHUNK_DRS_SCHEMA_VERSION,
        "compact_fact_policy": model_planner.CHUNK_DRS_COMPACT_FACT_POLICY_PREVIOUS,
        **model_planner._constraint_settings(
            model_planner.CHUNK_DRS_GRAMMAR,
            model_planner.COMPACT_CHUNK_DRS_JSON_SCHEMA,
            model_planner.CHUNK_DRS_SCHEMA_VERSION,
        ),
        "source_text_hash": source_text_hash,
    }
    legacy_hash = model_planner._cache_hash(
        "chunk_drs_compact",
        model_planner._build_compact_chunk_drs_prompt_v2(text, rel_path=rel_path),
        model,  # type: ignore[arg-type]
        settings,
    )
    legacy_path = model_planner._cache_path("KMD_CHUNK_DRS_CACHE_DIR", legacy_hash)
    assert legacy_path is not None
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": rel_path,
                    "referents": [],
                    "boxes": [{"id": "b0", "kind": "asserted", "parent_id": ""}],
                    "conditions": [],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                    "evidence_spans": [],
                },
                "raw_text": '{"facts":[]}',
            }
        ),
        encoding="utf-8",
    )

    result = call_model_chunk_drs_compact(  # type: ignore[arg-type]
        text,
        model,
        rel_path=rel_path,
        n_predict=72,
        refresh_empty_legacy=True,
    )

    assert model.calls == 1
    assert result["accepted"] is True
    assert result["drs"]["conditions"][0]["predicate"] == "means"


def test_compact_chunk_drs_retries_empty_current_cache(monkeypatch, tmp_path) -> None:
    class EmptyCacheRetryCompactModel:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.timeout_seconds = 240

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-current-empty-compact", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls.append(n_predict)
            return {
                "facts": [
                    {
                        "p": "means",
                        "agent": "luro",
                        "patient": "silver path",
                        "e": '"luro" means silver path.',
                    }
                ],
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    text = 'Glossary: "luro" means silver path.'
    rel_path = "notes/terms.txt"
    model = EmptyCacheRetryCompactModel()
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    source_text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    settings = {
        "n_predict": 72,
        "schema": model_planner.CHUNK_DRS_SCHEMA_VERSION,
        "compact_fact_policy": model_planner.CHUNK_DRS_COMPACT_FACT_POLICY,
        **model_planner._constraint_settings(
            model_planner.CHUNK_DRS_GRAMMAR,
            model_planner.COMPACT_CHUNK_DRS_JSON_SCHEMA,
            model_planner.CHUNK_DRS_SCHEMA_VERSION,
        ),
        "source_text_hash": source_text_hash,
    }
    current_hash = model_planner._cache_hash(
        "chunk_drs_compact",
        build_compact_chunk_drs_prompt(text, rel_path=rel_path),
        model,  # type: ignore[arg-type]
        settings,
    )
    current_path = model_planner._cache_path("KMD_CHUNK_DRS_CACHE_DIR", current_hash)
    assert current_path is not None
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": rel_path,
                    "referents": [],
                    "boxes": [{"id": "b0", "kind": "asserted", "parent_id": ""}],
                    "conditions": [],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                    "evidence_spans": [],
                },
                "raw_text": '{"facts":[]}',
            }
        ),
        encoding="utf-8",
    )

    result = call_model_chunk_drs_compact(  # type: ignore[arg-type]
        text,
        model,
        rel_path=rel_path,
        n_predict=72,
        refresh_empty_legacy=True,
    )

    assert model.calls == [160]
    assert result["accepted"] is True
    assert result["drs"]["conditions"][0]["predicate"] == "means"
    assert result["compact_retry_attempts"][0]["reason"] == "empty_compact_drs_cache"





def test_structured_cache_failures_are_retryable_for_all_json_calls() -> None:
    for reason in ["request_failed", "invalid_json", "schema_validation_failed", "grounding_validation_failed"]:
        assert model_planner._cached_structured_failure_retryable({"reason": reason}) is True
        assert model_planner._cached_request_failed({"reason": reason}) is True
    assert model_planner._cached_structured_failure_retryable({"reason": "compact_drs", "accepted": True}) is False


def test_cached_schema_and_grounding_failures_are_retryable() -> None:
    assert model_planner._query_drs_cached_retryable_failure({"accepted": False, "reason": "schema_validation_failed"}) is True
    assert model_planner._query_drs_cached_retryable_failure({"accepted": False, "reason": "grounding_validation_failed"}) is True
    assert model_planner._query_drs_cached_retryable_failure({"accepted": True, "reason": "compact_drs"}) is False


def test_compact_chunk_drs_does_not_reuse_source_cache_from_old_constraint_policy(monkeypatch, tmp_path) -> None:
    class PolicyAwareCompactModel:
        def __init__(self) -> None:
            self.calls = 0
            self.timeout_seconds = 240

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-policy-aware-compact", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            assert json_schema is not None
            return {
                "facts": [
                    {
                        "p": "means",
                        "e": "Glossary: zeno means red stone.",
                        "arguments": [{"role": "term", "value": "zeno"}, {"role": "meaning", "value": "red stone"}],
                        "temporal_text": "",
                        "scope": "asserted",
                    }
                ],
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.1,
            }

    text = "Glossary: zeno means red stone."
    rel_path = "notes/terms.txt"
    model = PolicyAwareCompactModel()
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    source_text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    cache_dir = tmp_path / "chunk-drs-cache"
    cache_dir.mkdir(parents=True)
    stale_path = cache_dir / "stale-old-policy.json"
    stale_path.write_text(json.dumps({
        "accepted": True,
        "reason": "compact_drs",
        "compact_fact_policy": model_planner.CHUNK_DRS_COMPACT_FACT_POLICY,
        "cache_context": {
            "schema": model_planner.CHUNK_DRS_SCHEMA_VERSION,
            "source_rel_path": rel_path,
            "source_text_hash": source_text_hash,
            "constraint_mode": "validated_json_no_schema",
        },
        "drs": {
            "schema_version": "chunk-drs-v2",
            "source_id": rel_path,
            "referents": [{"id": "r0", "label": "stale", "kind": "unknown", "evidence_text": "zeno"}],
            "boxes": [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": ""}],
            "conditions": [{"id": "c0", "predicate": "stale", "box_id": "b0", "polarity": "positive", "modality": "asserted", "temporal_id": "", "arguments": [], "evidence_text": "zeno"}],
            "identity_hypotheses": [],
            "temporal_records": [],
            "evidence_spans": [],
        },
    }), encoding="utf-8")

    result = call_model_chunk_drs_compact(text, model, rel_path=rel_path, n_predict=72)  # type: ignore[arg-type]

    assert model.calls == 1
    assert result["accepted"] is True
    assert result["drs"]["conditions"][0]["predicate"] == "means"



def test_chunk_drs_does_not_skip_structured_json_records_before_model(monkeypatch, tmp_path) -> None:
    class StructuredRecordModel:
        def __init__(self) -> None:
            self.calls = 0
            self.timeout_seconds = 240

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-structured-record-call", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            assert "products/ActionGenie.json" in prompt
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "products/ActionGenie.json",
                    "referents": [
                        {"id": "r0", "label": "sales", "kind": "unknown", "evidence_text": '"channel":"sales"'}
                    ],
                    "boxes": [
                        {
                            "id": "b0",
                            "kind": "asserted",
                            "parent_id": "",
                            "holder_referent_id": "",
                            "evidence_text": '"channel":"sales"',
                        }
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "channel",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "value",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "sales",
                                    "value_type": "unknown",
                                    "evidence_text": '"channel":"sales"',
                                }
                            ],
                            "evidence_text": '"channel":"sales"',
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.delenv("KMD_MODEL_DRS_FOR_STRUCTURED_JSON_RECORDS", raising=False)
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    text = '{"messages":[{"user":"alice","text":"hello team","ts":"2026-01-01T00:00:00"},{"user":"bob","text":"ack","ts":"2026-01-01T00:01:00"}],"channel":"sales"}'
    model = StructuredRecordModel()

    result = call_model_chunk_drs(text, model, rel_path="products/ActionGenie.json")  # type: ignore[arg-type]

    assert model.calls >= 1
    assert result["accepted"] is True
    assert str(result.get("reason") or "") != "skipped_structured_record"
    assert "structured_json_skip" not in result["context_budget"]


def test_compact_chunk_schema_is_bounded() -> None:
    facts_schema = model_planner.COMPACT_CHUNK_DRS_JSON_SCHEMA["properties"]["facts"]
    assert facts_schema["maxItems"] == 8
    arg_schema = facts_schema["items"]["properties"]["arguments"]
    assert arg_schema["maxItems"] == 6
    assert arg_schema["items"]["properties"]["value"]["maxLength"] == 160


def test_compact_chunk_drs_uses_json_schema_and_explicit_arguments(monkeypatch, tmp_path) -> None:
    class CompactSchemaModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.timeout_seconds = 240

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-compact-schema", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls.append({"grammar": grammar, "json_schema": json_schema, "n_predict": n_predict})
            assert json_schema is not None
            assert json_schema["required"] == ["facts"]
            return {
                "facts": [
                    {
                        "p": "means",
                        "e": "Glossary: mave means quiet hill.",
                        "arguments": [{"role": "term", "value": "mave"}, {"role": "meaning", "value": "quiet hill"}],
                        "temporal_text": "",
                        "scope": "asserted",
                    }
                ],
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.1,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    model = CompactSchemaModel()
    result = call_model_chunk_drs_compact("Glossary: mave means quiet hill.", model, rel_path="notes.txt", n_predict=72)  # type: ignore[arg-type]

    assert model.calls and model.calls[0]["json_schema"] is not None
    assert result["accepted"] is True
    assert result["cache_context"]["constraint_mode"] == "json_schema"
    assert result["drs"]["conditions"][0]["predicate"] == "means"
    assert result["drs"]["conditions"][0]["arguments"][0]["role"] == "term"


def test_chunk_drs_validation_rejects_duplicate_referent_ids() -> None:
    payload = {
        "drs": {
            "schema_version": "chunk-drs-v2",
            "source_id": "notes.txt",
            "referents": [
                {"id": "r0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"},
                {"id": "r0", "label": "ready", "kind": "state", "evidence_text": "ready"},
            ],
            "boxes": [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": ""}],
            "conditions": [
                {
                    "id": "c0",
                    "predicate": "state",
                    "box_id": "b0",
                    "polarity": "positive",
                    "modality": "asserted",
                    "temporal_id": "",
                    "arguments": [],
                    "evidence_text": "ready",
                }
            ],
            "identity_hypotheses": [],
            "temporal_records": [],
            "evidence_spans": [],
        }
    }

    validation = model_planner._validate_chunk_drs_payload(payload, "Aero Gate is ready.")

    assert validation["schema_valid"] is False
    assert "duplicate_or_missing_referent_id" in validation["errors"]


def test_chunk_drs_validation_rejects_duplicate_condition_ids() -> None:
    payload = {
        "drs": {
            "schema_version": "chunk-drs-v2",
            "source_id": "notes.txt",
            "referents": [{"id": "r0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"}],
            "boxes": [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": ""}],
            "conditions": [
                {
                    "id": "c0",
                    "predicate": "state",
                    "box_id": "b0",
                    "polarity": "positive",
                    "modality": "asserted",
                    "temporal_id": "",
                    "arguments": [],
                    "evidence_text": "ready",
                },
                {
                    "id": "c0",
                    "predicate": "status",
                    "box_id": "b0",
                    "polarity": "positive",
                    "modality": "asserted",
                    "temporal_id": "",
                    "arguments": [],
                    "evidence_text": "ready",
                },
            ],
            "identity_hypotheses": [],
            "temporal_records": [],
            "evidence_spans": [],
        }
    }

    validation = model_planner._validate_chunk_drs_payload(payload, "Aero Gate is ready.")

    assert validation["schema_valid"] is False
    assert "duplicate_or_missing_condition_id" in validation["errors"]


def test_repair_chunk_drs_adds_only_safe_empty_auxiliary_lists() -> None:
    payload = {
        "drs": {
            "schema_version": "chunk-drs-v2",
            "source_id": "notes.txt",
            "referents": [],
            "boxes": [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": ""}],
            "conditions": [],
        }
    }

    repaired = model_planner._repair_chunk_drs_payload(payload, "")
    validation = model_planner._validate_chunk_drs_payload(repaired, "")

    assert repaired["drs"]["identity_hypotheses"] == []
    assert repaired["drs"]["temporal_records"] == []
    assert repaired["drs"]["evidence_spans"] == []
    assert repaired["drs"]["semantic_notes"] == []
    assert validation["schema_valid"] is True


def test_compact_chunk_drs_reuses_condition_retry_cache_after_empty_current_cache(monkeypatch, tmp_path) -> None:
    class CachedRetryCompactModel:
        def __init__(self) -> None:
            self.calls = 0
            self.timeout_seconds = 240

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-retry-cache-compact", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            raise AssertionError("retry cache should avoid live model call")

    text = 'Glossary: "mave" means quiet hill.'
    rel_path = "notes/terms.txt"
    model = CachedRetryCompactModel()
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    source_text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    base_settings = {
        "n_predict": 72,
        "schema": model_planner.CHUNK_DRS_SCHEMA_VERSION,
        "compact_fact_policy": model_planner.CHUNK_DRS_COMPACT_FACT_POLICY,
        **model_planner._constraint_settings(
            model_planner.CHUNK_DRS_GRAMMAR,
            model_planner.COMPACT_CHUNK_DRS_JSON_SCHEMA,
            model_planner.CHUNK_DRS_SCHEMA_VERSION,
        ),
        "source_text_hash": source_text_hash,
    }
    prompt = build_compact_chunk_drs_prompt(text, rel_path=rel_path)
    current_hash = model_planner._cache_hash("chunk_drs_compact", prompt, model, base_settings)  # type: ignore[arg-type]
    current_path = model_planner._cache_path("KMD_CHUNK_DRS_CACHE_DIR", current_hash)
    assert current_path is not None
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "elapsed": None,
                "prompt_hash": current_hash,
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": rel_path,
                    "referents": [],
                    "boxes": [{"id": "b0", "kind": "asserted", "parent_id": ""}],
                    "conditions": [],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                    "evidence_spans": [],
                },
                "raw_text": '{"facts":[]}',
            }
        ),
        encoding="utf-8",
    )
    retry_settings = {
        **base_settings,
        "n_predict": 160,
        "compact_retry_policy": model_planner.CHUNK_DRS_COMPACT_RETRY_POLICY,
        "compact_retry_index": 1,
        "compact_retry_after": {
            "n_predict": 72,
            "reason": "empty_compact_drs_cache",
            "elapsed": None,
            "prompt_hash": current_hash,
        },
    }
    retry_hash = model_planner._cache_hash("chunk_drs_compact", prompt, model, retry_settings)  # type: ignore[arg-type]
    retry_path = model_planner._cache_path("KMD_CHUNK_DRS_CACHE_DIR", retry_hash)
    assert retry_path is not None
    retry_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "prompt_hash": retry_hash,
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": rel_path,
                    "referents": [
                        {"id": "r0", "label": "mave", "kind": "unknown", "evidence_text": "mave"},
                        {"id": "r1", "label": "quiet hill", "kind": "unknown", "evidence_text": "quiet hill"},
                    ],
                    "boxes": [{"id": "b0", "kind": "asserted", "parent_id": ""}],
                    "conditions": [
                        {
                            "id": "c0",
                            "box_id": "b0",
                            "predicate": "means",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "evidence_text": '"mave" means quiet hill.',
                            "arguments": [
                                {
                                    "role": "agent",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "unknown",
                                    "evidence_text": "mave",
                                },
                                {
                                    "role": "patient",
                                    "target_kind": "referent",
                                    "target_id": "r1",
                                    "value": "",
                                    "value_type": "unknown",
                                    "evidence_text": "quiet hill",
                                },
                            ],
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                    "evidence_spans": ['"mave" means quiet hill.'],
                },
                "raw_text": "{}",
            }
        ),
        encoding="utf-8",
    )

    result = call_model_chunk_drs_compact(  # type: ignore[arg-type]
        text,
        model,
        rel_path=rel_path,
        n_predict=72,
    )

    assert model.calls == 0
    assert result["prompt_hash"] == retry_hash
    assert result["drs"]["conditions"][0]["predicate"] == "means"


def test_compact_chunk_drs_reuses_equivalent_condition_cache_after_empty_current_cache(monkeypatch, tmp_path) -> None:
    class EquivalentCacheCompactModel:
        def __init__(self) -> None:
            self.calls = 0
            self.timeout_seconds = 240.0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-equivalent-cache-compact", "context_size": 4096, "timeout_seconds": 240.0}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            raise AssertionError("equivalent condition cache should avoid live model call")

    text = 'Glossary: "tavil" means clear meadow.'
    rel_path = "notes/terms.txt"
    model = EquivalentCacheCompactModel()
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    source_text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    base_settings = {
        "n_predict": 72,
        "schema": model_planner.CHUNK_DRS_SCHEMA_VERSION,
        "compact_fact_policy": model_planner.CHUNK_DRS_COMPACT_FACT_POLICY,
        **model_planner._constraint_settings(
            model_planner.CHUNK_DRS_GRAMMAR,
            model_planner.COMPACT_CHUNK_DRS_JSON_SCHEMA,
            model_planner.CHUNK_DRS_SCHEMA_VERSION,
        ),
        "source_text_hash": source_text_hash,
    }
    prompt = build_compact_chunk_drs_prompt(text, rel_path=rel_path)
    current_hash = model_planner._cache_hash("chunk_drs_compact", prompt, model, base_settings)  # type: ignore[arg-type]
    current_path = model_planner._cache_path("KMD_CHUNK_DRS_CACHE_DIR", current_hash)
    assert current_path is not None
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "elapsed": 0.01,
                "prompt_hash": current_hash,
                "cache_context": {
                    **base_settings,
                    "model_fingerprint": model.cache_fingerprint(),
                    "source_rel_path": rel_path,
                },
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": rel_path,
                    "referents": [],
                    "boxes": [{"id": "b0", "kind": "asserted", "parent_id": ""}],
                    "conditions": [],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                    "evidence_spans": [],
                },
                "raw_text": '{"facts":[]}',
            }
        ),
        encoding="utf-8",
    )
    equivalent_path = current_path.parent / "equivalent-condition-cache.json"
    equivalent_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "prompt_hash": "different-equivalent-cache-key",
                "compact_fact_policy": model_planner.CHUNK_DRS_COMPACT_FACT_POLICY,
                "cache_context": {
                    **base_settings,
                    "model_fingerprint": {
                        "model_id": "fake-equivalent-cache-compact",
                        "context_size": 4096,
                        "timeout_seconds": 240,
                    },
                    "source_rel_path": rel_path,
                },
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": rel_path,
                    "referents": [
                        {"id": "r0", "label": "tavil", "kind": "unknown", "evidence_text": "tavil"},
                        {"id": "r1", "label": "clear meadow", "kind": "unknown", "evidence_text": "clear meadow"},
                    ],
                    "boxes": [{"id": "b0", "kind": "asserted", "parent_id": ""}],
                    "conditions": [
                        {
                            "id": "c0",
                            "box_id": "b0",
                            "predicate": "means",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "evidence_text": '"tavil" means clear meadow.',
                            "arguments": [
                                {
                                    "role": "agent",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "unknown",
                                    "evidence_text": "tavil",
                                },
                                {
                                    "role": "patient",
                                    "target_kind": "referent",
                                    "target_id": "r1",
                                    "value": "",
                                    "value_type": "unknown",
                                    "evidence_text": "clear meadow",
                                },
                            ],
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                    "evidence_spans": ['"tavil" means clear meadow.'],
                },
                "raw_text": "{}",
            }
        ),
        encoding="utf-8",
    )

    result = call_model_chunk_drs_compact(  # type: ignore[arg-type]
        text,
        model,
        rel_path=rel_path,
        n_predict=72,
    )

    assert model.calls == 0
    assert result["drs"]["conditions"][0]["predicate"] == "means"
    assert result["compact_source_cache_reuse"]["from_prompt_hash"] == "different-equivalent-cache-key"


def test_compact_chunk_drs_reuses_equivalent_condition_cache_before_live_call(monkeypatch, tmp_path) -> None:
    class EquivalentCacheOnlyCompactModel:
        def __init__(self) -> None:
            self.calls = 0
            self.timeout_seconds = 240.0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-equivalent-cache-only-compact", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            raise AssertionError("same-source condition cache should avoid live model call")

    text = 'Glossary: "pavin" means bright harbor.'
    rel_path = "notes/terms.txt"
    model = EquivalentCacheOnlyCompactModel()
    cache_dir = tmp_path / "chunk-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    source_text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "accepted-source-cache.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "prompt_hash": "accepted-source-cache",
                "compact_fact_policy": model_planner.CHUNK_DRS_COMPACT_FACT_POLICY,
                "cache_context": {
                    "n_predict": 160,
                    "schema": model_planner.CHUNK_DRS_SCHEMA_VERSION,
                    "compact_fact_policy": model_planner.CHUNK_DRS_COMPACT_FACT_POLICY,
                    **model_planner._constraint_settings(
                        model_planner.CHUNK_DRS_GRAMMAR,
                        model_planner.COMPACT_CHUNK_DRS_JSON_SCHEMA,
                        model_planner.CHUNK_DRS_SCHEMA_VERSION,
                    ),
                    "source_text_hash": source_text_hash,
                    "source_rel_path": rel_path,
                },
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": rel_path,
                    "referents": [
                        {"id": "r0", "label": "pavin", "kind": "unknown", "evidence_text": "pavin"},
                        {"id": "r1", "label": "bright harbor", "kind": "unknown", "evidence_text": "bright harbor"},
                    ],
                    "boxes": [{"id": "b0", "kind": "asserted", "parent_id": ""}],
                    "conditions": [
                        {
                            "id": "c0",
                            "box_id": "b0",
                            "predicate": "means",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "evidence_text": '"pavin" means bright harbor.',
                            "arguments": [
                                {
                                    "role": "agent",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "unknown",
                                    "evidence_text": "pavin",
                                },
                                {
                                    "role": "patient",
                                    "target_kind": "referent",
                                    "target_id": "r1",
                                    "value": "",
                                    "value_type": "unknown",
                                    "evidence_text": "bright harbor",
                                },
                            ],
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                    "evidence_spans": ['"pavin" means bright harbor.'],
                },
                "raw_text": "{}",
            }
        ),
        encoding="utf-8",
    )

    result = call_model_chunk_drs_compact(  # type: ignore[arg-type]
        text,
        model,
        rel_path=rel_path,
        n_predict=72,
    )

    assert model.calls == 0
    assert result["drs"]["conditions"][0]["predicate"] == "means"
    assert result["compact_source_cache_reuse"]["from_prompt_hash"] == "accepted-source-cache"


def test_compact_chunk_drs_retries_truncated_json_with_larger_budget(monkeypatch, tmp_path) -> None:
    class TruncatedThenValidCompactModel:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-compact-chunk-retry", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls.append(n_predict)
            if len(self.calls) == 1:
                raise LocalModelJSONError(
                    "unterminated string",
                    raw_text='{"facts":[{"p":"depends on","agent":"Sample Project","patient":"',
                    snippet='{"facts":[{"p":"depends on","agent":"Sample Project","patient":"',
                )
            return {
                "facts": [
                    {
                        "p": "depends on",
                        "agent": "Sample Project",
                        "patient": "ITEM-1",
                        "e": "Sample Project depends on ITEM-1.",
                    }
                ],
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    result = call_model_chunk_drs_compact(
        "Sample Project depends on ITEM-1.",
        TruncatedThenValidCompactModel(),  # type: ignore[arg-type]
        rel_path="samples/project-plan.note",
        n_predict=72,
    )

    assert result["accepted"] is True
    assert result["reason"] == "compact_drs"
    assert result["compact_retry_index"] == 1
    assert result["compact_retry_attempts"][0]["reason"] == "invalid_json"


def test_compact_chunk_drs_materializes_model_emitted_temporal_values(monkeypatch, tmp_path) -> None:
    class TimestampedCompactModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-compact-timestamp", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            return {
                "facts": [
                    {
                        "p": "status",
                        "patient": "Aster Well",
                        "value": "closed",
                        "time": "2026-03-09",
                        "e": "2026-03-09 status: closed for Aster Well.",
                    }
                ],
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    result = call_model_chunk_drs_compact(
        "2026-03-09 status: closed for Aster Well.",
        TimestampedCompactModel(),  # type: ignore[arg-type]
        rel_path="logs/state.log",
        n_predict=72,
    )

    assert result["accepted"] is True
    drs = result["drs"]
    assert drs["temporal_records"] == [
        {"id": "t0", "value": "2026-03-09", "value_type": "timestamp", "evidence_text": "2026-03-09"}
    ]
    condition = drs["conditions"][0]
    assert condition["temporal_id"] == "t0"
    assert {
        "role": "value",
        "target_kind": "literal",
        "target_id": "",
        "value": "closed",
        "value_type": "value",
        "evidence_text": "closed",
    } in condition["arguments"]


def test_compact_chunk_drs_attaches_source_span_temporal_prefix(monkeypatch, tmp_path) -> None:
    class UntimedCompactModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-compact-source-time", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            return {
                "facts": [
                    {
                        "p": "status",
                        "patient": "Beryl Well",
                        "value": "stable",
                        "e": "status: stable for Beryl Well.",
                    }
                ],
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    result = call_model_chunk_drs_compact(
        "2026-04-11 status: stable for Beryl Well.",
        UntimedCompactModel(),  # type: ignore[arg-type]
        rel_path="logs/state.log",
        n_predict=72,
    )

    assert result["accepted"] is True
    drs = result["drs"]
    assert drs["temporal_records"][0]["value"] == "2026-04-11"
    assert drs["conditions"][0]["temporal_id"] == drs["temporal_records"][0]["id"]
    assert any(
        argument["target_kind"] == "literal" and argument["value"] == "stable"
        for argument in drs["conditions"][0]["arguments"]
    )


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


def test_verifier_request_failure_does_not_abort_model_answer(monkeypatch) -> None:
    from knowmoredirt.models import Answer, Evidence
    from knowmoredirt.engine import ExpectedAnswer, KnowMoreDiRTEngine
    from knowmoredirt.query import QueryFrame

    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine.model_query_trace = __import__("knowmoredirt.engine", fromlist=["ModelQueryTrace"]).ModelQueryTrace()
    engine._evidence_payload = lambda evidence, limit=8: [{"source_id": "note.txt", "text": "Aero Gate is ready."}]
    engine._diagnostic_frames_for_answer = lambda answer: []
    engine._canonicalize_model_answer_with_local_model = lambda question, text, expected, evidence: text
    engine._log_progress = lambda message: None

    monkeypatch.setattr(
        "knowmoredirt.engine.call_model_answer_verification",
        lambda *args, **kwargs: {"accepted": False, "reason": "request_failed", "error": "HTTP Error 400: Bad Request"},
    )

    frame = QueryFrame(
        question_text="Is Aero Gate ready?",
        target_anchors=("Aero Gate",),
        requested_relation="ready",
        relation_terms=("ready",),
        answer_variables=(),
        constraints=(),
        answer_type="boolean",
    )
    answer = Answer("yes", evidence=[Evidence("note.txt", "Aero Gate is ready.")])

    assert engine._verify_with_local_model("Is Aero Gate ready?", frame, answer, ExpectedAnswer("boolean")) is False
    assert engine.model_query_trace.verifier_rejected_count == 1
