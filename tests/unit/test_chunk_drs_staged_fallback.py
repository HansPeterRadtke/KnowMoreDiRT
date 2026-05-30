from __future__ import annotations

from typing import Any

from knowmoredirt.model_planner import call_model_chunk_drs


def test_chunk_drs_staged_fallback_constrains_condition_targets(monkeypatch, tmp_path) -> None:
    class StagedFallbackModel:
        def __init__(self) -> None:
            self.condition_schema: dict[str, Any] | None = None

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-staged-drs", "context_size": 8192}

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, Any] | None = None,
        ) -> dict[str, object]:
            if "one source-grounded DRS object" in prompt:
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
                                "box_id": "b9",
                                "polarity": "positive",
                                "modality": "asserted",
                                "temporal_id": "",
                                "arguments": [],
                                "evidence_text": "Aero Gate is ready.",
                            }
                        ],
                        "identity_hypotheses": [],
                        "temporal_records": [],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            if "Stage 1 of source-grounded DRS extraction" in prompt:
                return {
                    "drs_skeleton": {
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
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            assert "Stage 2 of source-grounded DRS extraction" in prompt
            self.condition_schema = json_schema
            condition_schema = json_schema["properties"]["condition_stage"]["properties"]["conditions"]["items"]
            argument_schema = condition_schema["properties"]["arguments"]["items"]
            assert condition_schema["properties"]["box_id"]["enum"] == ["b0"]
            assert argument_schema["properties"]["target_id"]["enum"] == ["", "b0", "r0"]
            return {
                "condition_stage": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "note.txt",
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
                                    "role": "scope",
                                    "target_kind": "referent",
                                    "target_id": "b0",
                                    "value": "",
                                    "value_type": "box",
                                    "evidence_text": "Aero Gate is ready.",
                                }
                            ],
                            "evidence_text": "Aero Gate is ready.",
                        }
                    ],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.delenv("KMD_LOCAL_MODEL_JSON_SCHEMA", raising=False)
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "drs-cache"))
    model = StagedFallbackModel()

    result = call_model_chunk_drs("Aero Gate is ready.", model, rel_path="note.txt", n_predict=384)  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["reason"] == "staged_fallback"
    assert result["fallback_from_reason"] == "schema_validation_failed"
    assert result["validation"]["condition_count"] == 1
    assert result["drs"]["conditions"][0]["arguments"][0]["target_kind"] == "box"
    assert model.condition_schema is not None
