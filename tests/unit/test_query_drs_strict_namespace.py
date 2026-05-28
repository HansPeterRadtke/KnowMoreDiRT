from __future__ import annotations

from knowmoredirt.model_planner import call_model_chunk_drs, call_model_query_drs, query_frame_from_query_drs


def test_strict_query_drs_uses_answer_variable_namespace(monkeypatch, tmp_path) -> None:
    class StrictQueryModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-strict-query", "context_size": 8192}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            assert "target_kind='answer_variable'" in prompt
            assert json_schema["properties"]["query_drs"]["properties"]["answer_variables"]["items"]["properties"]["id"]
            return {
                "query_drs": {
                    "schema_version": "query-drs-v2",
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
