from __future__ import annotations

from knowmoredirt.model_planner import (
    QUERY_DRS_VALIDATION_POLICY,
    call_model_chunk_drs,
    call_model_query_drs,
    chunk_drs_array_max_items,
    chunk_drs_evidence_max_chars,
    chunk_drs_json_schema,
    query_frame_from_query_drs,
)


def test_strict_query_drs_uses_answer_variable_namespace(monkeypatch, tmp_path) -> None:
    class StrictQueryModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-strict-query", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            assert "target_kind='answer_variable'" in prompt
            assert json_schema["properties"]["query_drs"]["properties"]["answer_variables"]["items"]["properties"]["id"]
            assert "temporal_records" in json_schema["properties"]["query_drs"]["properties"]
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
    result = call_model_query_drs("Who reviewed Aero Gate?", StrictQueryModel())  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["validation"]["answer_variable_count"] == 1
    assert result["validation"]["grounding_failure_count"] == 0

    frame = query_frame_from_query_drs("Who reviewed Aero Gate?", result["query_drs"])

    assert frame is not None
    assert frame["answer_variables"] == ("reviewer",)
    assert frame["target_anchors"] == ("Aero Gate",)
    assert "agent" in frame["relation_terms"]
    assert "reviewed" in frame["relation_terms"]


def test_query_drs_temporal_namespace_reaches_query_frame(monkeypatch, tmp_path) -> None:
    class TemporalQueryModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-temporal-query", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            assert "temporal_records" in prompt
            return {
                "query_drs": {
                    "schema_version": "query-drs-v3",
                    "question": "Which state was current for Aurora Loom after the latest update?",
                    "answer_variables": [
                        {
                            "id": "qv0",
                            "label": "state",
                            "answer_type": "state",
                            "evidence_text": "Which state",
                        }
                    ],
                    "target_referents": [
                        {"id": "qr0", "label": "Aurora Loom", "kind": "entity", "evidence_text": "Aurora Loom"}
                    ],
                    "temporal_records": [
                        {
                            "id": "qt0",
                            "value": "after the latest update",
                            "value_type": "temporal",
                            "evidence_text": "after the latest update",
                        }
                    ],
                    "requested_conditions": [
                        {
                            "id": "qc0",
                            "predicate": "current",
                            "box_id": "",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "qt0",
                            "arguments": [
                                {
                                    "role": "target",
                                    "target_kind": "answer_variable",
                                    "target_id": "qv0",
                                    "value": "",
                                    "value_type": "state",
                                    "evidence_text": "state",
                                },
                                {
                                    "role": "theme",
                                    "target_kind": "referent",
                                    "target_id": "qr0",
                                    "value": "Aurora Loom",
                                    "value_type": "entity",
                                    "evidence_text": "Aurora Loom",
                                },
                            ],
                            "evidence_text": "current for Aurora Loom after the latest update",
                        }
                    ],
                    "constraints": [],
                    "box_requirements": [],
                    "temporal_scope": "",
                    "aggregation": "",
                    "answer_type": "state",
                    "requires_evidence": True,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_QUERY_DRS_CACHE_DIR", str(tmp_path / "query-drs-cache"))
    result = call_model_query_drs(
        "Which state was current for Aurora Loom after the latest update?",
        TemporalQueryModel(),  # type: ignore[arg-type]
    )

    assert result["accepted"] is True
    assert result["validation"]["temporal_record_count"] == 1

    frame = query_frame_from_query_drs(
        "Which state was current for Aurora Loom after the latest update?",
        result["query_drs"],
    )

    assert frame is not None
    assert frame["temporal_scope"] == "after the latest update"
    assert "after the latest update" in frame["relation_terms"]


def test_query_drs_repairs_condition_evidence_to_full_question(monkeypatch, tmp_path) -> None:
    class ConditionEvidenceRepairModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-query-condition-repair", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
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
                            "evidence_text": "the review request for Aero Gate",
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
    result = call_model_query_drs("Who reviewed Aero Gate?", ConditionEvidenceRepairModel())  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["validation_policy"] == QUERY_DRS_VALIDATION_POLICY
    assert result["validation"]["grounding_failure_count"] == 0
    assert result["query_drs"]["requested_conditions"][0]["evidence_text"] == "Who reviewed Aero Gate?"


def test_query_drs_repairs_answer_variable_label_variant(monkeypatch, tmp_path) -> None:
    class AnswerVariableLabelRepairModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-query-answer-variable-repair", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            return {
                "query_drs": {
                    "schema_version": "query-drs-v3",
                    "question": "What report link is listed for Orchid Gamma?",
                    "answer_variables": [
                        {
                            "id": "qv0",
                            "label": "report_link",
                            "answer_type": "url",
                            "evidence_text": "report link listed for Orchid Gamma",
                        }
                    ],
                    "target_referents": [
                        {"id": "qr0", "label": "Orchid Gamma", "kind": "entity", "evidence_text": "Orchid Gamma"}
                    ],
                    "temporal_records": [],
                    "requested_conditions": [
                        {
                            "id": "qc0",
                            "predicate": "listed",
                            "box_id": "",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "report_link",
                                    "target_kind": "answer_variable",
                                    "target_id": "qv0",
                                    "value": ">",
                                    "value_type": "url",
                                    "evidence_text": "report link listed for Orchid Gamma",
                                },
                                {
                                    "role": "for",
                                    "target_kind": "referent",
                                    "target_id": "qr0",
                                    "value": ">",
                                    "value_type": "entity",
                                    "evidence_text": "Orchid Gamma",
                                },
                            ],
                            "evidence_text": "report link listed for Orchid Gamma",
                        }
                    ],
                    "constraints": [],
                    "box_requirements": [],
                    "temporal_scope": "",
                    "aggregation": "",
                    "answer_type": "url",
                    "requires_evidence": True,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_QUERY_DRS_CACHE_DIR", str(tmp_path / "query-drs-cache"))
    result = call_model_query_drs(
        "What report link is listed for Orchid Gamma?",
        AnswerVariableLabelRepairModel(),  # type: ignore[arg-type]
    )

    assert result["accepted"] is True
    assert result["validation"]["grounding_failure_count"] == 0
    assert result["query_drs"]["answer_variables"][0]["evidence_text"] == "report link"
    assert result["query_drs"]["requested_conditions"][0]["arguments"][0]["evidence_text"] == "report link"
    assert result["query_drs"]["requested_conditions"][0]["arguments"][0]["value"] == ""
    assert result["query_drs"]["requested_conditions"][0]["arguments"][1]["value"] == ""
    assert result["query_drs"]["requested_conditions"][0]["evidence_text"] == (
        "What report link is listed for Orchid Gamma?"
    )
    frame = query_frame_from_query_drs("What report link is listed for Orchid Gamma?", result["query_drs"])
    assert frame is not None
    assert ">" not in frame["relation_terms"]


def test_query_drs_keeps_ungrounded_temporal_rejection(monkeypatch, tmp_path) -> None:
    class UngroundedTemporalQueryModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-query-ungrounded-temporal", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            return {
                "query_drs": {
                    "schema_version": "query-drs-v3",
                    "question": "Which state was current for Aero Gate?",
                    "answer_variables": [
                        {"id": "qv0", "label": "state", "answer_type": "state", "evidence_text": "Which state"}
                    ],
                    "target_referents": [
                        {"id": "qr0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"}
                    ],
                    "temporal_records": [
                        {"id": "qt0", "value": "now", "value_type": "time", "evidence_text": "now"}
                    ],
                    "requested_conditions": [
                        {
                            "id": "qc0",
                            "predicate": "current",
                            "box_id": "",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "qt0",
                            "arguments": [
                                {
                                    "role": "target",
                                    "target_kind": "answer_variable",
                                    "target_id": "qv0",
                                    "value": "",
                                    "value_type": "state",
                                    "evidence_text": "state",
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
                            "evidence_text": "Which state was current for Aero Gate?",
                        }
                    ],
                    "constraints": [],
                    "box_requirements": [],
                    "temporal_scope": "",
                    "aggregation": "",
                    "answer_type": "state",
                    "requires_evidence": True,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_QUERY_DRS_CACHE_DIR", str(tmp_path / "query-drs-cache"))
    result = call_model_query_drs(
        "Which state was current for Aero Gate?",
        UngroundedTemporalQueryModel(),  # type: ignore[arg-type]
    )

    assert result["accepted"] is False
    assert result["reason"] == "schema_validation_failed"
    assert result["validation"]["grounding_failures"] == ["temporal:qt0:now"]


def test_chunk_drs_removes_tautological_self_identity_hypotheses(monkeypatch, tmp_path) -> None:
    class SelfIdentityChunkModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-self-identity", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            assert "do not include self identity hypotheses" in prompt
            return {
                "drs": {
                    "schema_version": "chunk-drs-v1",
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
                    "identity_hypotheses": [
                        {
                            "left_referent_id": "r0",
                            "right_referent_id": "r0",
                            "status": "accepted",
                            "evidence_text": "Aero Gate",
                            "confidence": 1.0,
                        }
                    ],
                    "temporal_records": [],
                    "evidence_spans": ["Aero Gate is ready."],
                    "semantic_notes": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    result = call_model_chunk_drs("Aero Gate is ready.", SelfIdentityChunkModel(), rel_path="note.txt")  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["validation"]["identity_hypothesis_count"] == 0
    assert result["drs"]["identity_hypotheses"] == []


def test_chunk_drs_evidence_cap_uses_reserved_output_budget(monkeypatch) -> None:
    monkeypatch.delenv("KMD_CHUNK_DRS_MAX_EVIDENCE_CHARS", raising=False)
    monkeypatch.delenv("KMD_CHUNK_DRS_MAX_ARRAY_ITEMS", raising=False)

    assert chunk_drs_evidence_max_chars("x" * 1000, 512) == 128
    assert chunk_drs_evidence_max_chars("x" * 50, 512) == 50
    assert chunk_drs_array_max_items(768) == 8

    monkeypatch.setenv("KMD_CHUNK_DRS_MAX_EVIDENCE_CHARS", "77")
    monkeypatch.setenv("KMD_CHUNK_DRS_MAX_ARRAY_ITEMS", "6")

    assert chunk_drs_evidence_max_chars("x" * 1000, 512) == 77
    assert chunk_drs_array_max_items(768) == 6


def test_chunk_drs_schema_caps_arrays_from_output_budget() -> None:
    schema = chunk_drs_json_schema(31, 7)
    drs_schema = schema["properties"]["drs"]
    condition_schema = drs_schema["properties"]["conditions"]["items"]

    assert drs_schema["properties"]["referents"]["maxItems"] == 7
    assert drs_schema["properties"]["boxes"]["maxItems"] == 7
    assert drs_schema["properties"]["conditions"]["maxItems"] == 7
    assert condition_schema["properties"]["arguments"]["maxItems"] == 7
    assert drs_schema["properties"]["evidence_spans"]["maxItems"] == 7
    assert drs_schema["properties"]["evidence_spans"]["items"]["maxLength"] == 31


def test_chunk_drs_production_schema_omits_auxiliary_note_arrays(monkeypatch, tmp_path) -> None:
    class ProductionSchemaModel:
        def __init__(self) -> None:
            self.json_schema = None

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-production-lean-drs", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.json_schema = json_schema
            drs_schema = json_schema["properties"]["drs"]
            assert "evidence_spans" not in drs_schema["properties"]
            assert "semantic_notes" not in drs_schema["properties"]
            assert "evidence_spans" not in drs_schema["required"]
            assert "semantic_notes" not in drs_schema["required"]
            assert "evidence_spans" not in prompt
            assert "semantic_notes" not in prompt
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
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    model = ProductionSchemaModel()
    result = call_model_chunk_drs("Aero Gate is ready.", model, rel_path="note.txt")  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["validation"]["schema_valid"] is True
    assert model.json_schema is not None


def test_chunk_drs_rejects_ungrounded_temporal_records(monkeypatch, tmp_path) -> None:
    class UngroundedTemporalModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-ungrounded-temporal", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            return {
                "drs": {
                    "schema_version": "chunk-drs-v1",
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
                            "temporal_id": "t0",
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
                    "temporal_records": [
                        {"id": "t0", "value": "now", "value_type": "time", "evidence_text": "now"}
                    ],
                    "evidence_spans": ["Aero Gate is ready."],
                    "semantic_notes": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    result = call_model_chunk_drs("Aero Gate is ready.", UngroundedTemporalModel(), rel_path="note.txt")  # type: ignore[arg-type]

    assert result["accepted"] is False
    assert result["reason"] == "grounding_validation_failed"
    assert result["validation"]["schema_valid"] is False
    assert result["validation"]["grounding_failure_count"] == 1
    assert result["validation"]["grounding_failures"] == ["temporal:t0:now"]
