from __future__ import annotations

from typing import Any

from knowmoredirt.model_planner import (
    CHUNK_DRS_GROUNDING_REPAIR_POLICY,
    CHUNK_DRS_STAGED_FALLBACK_POLICY,
    call_model_chunk_drs,
    chunk_drs_cache_context,
)


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
    assert result["context_budget"]["staged_fallback_policy"] == CHUNK_DRS_STAGED_FALLBACK_POLICY
    assert chunk_drs_cache_context(model, n_predict=384)["staged_fallback_policy"] == CHUNK_DRS_STAGED_FALLBACK_POLICY
    assert model.condition_schema is not None


def test_chunk_drs_staged_fallback_runs_after_grounding_failure(monkeypatch, tmp_path) -> None:
    class GroundingFallbackModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-grounding-drs", "context_size": 8192}

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
                                "box_id": "b0",
                                "polarity": "positive",
                                "modality": "asserted",
                                "temporal_id": "",
                                "arguments": [
                                    {
                                        "role": "theme",
                                        "target_kind": "referent",
                                        "target_id": "r0",
                                        "value": "",
                                        "value_type": "entity",
                                        "evidence_text": "Aero Gate",
                                    }
                                ],
                                "evidence_text": "Aero Gate ready",
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
                                    "role": "theme",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "entity",
                                    "evidence_text": "Aero Gate",
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

    result = call_model_chunk_drs("Aero Gate is ready.", GroundingFallbackModel(), rel_path="note.txt", n_predict=384)  # type: ignore[arg-type]

    assert result["accepted"] is True
    assert result["reason"] == "staged_fallback"
    assert result["fallback_from_reason"] == "grounding_validation_failed"
    assert result["validation"]["grounding_failure_count"] == 0


def test_chunk_drs_staged_fallback_preserves_temporal_records(monkeypatch, tmp_path) -> None:
    class TemporalStagedFallbackModel:
        def __init__(self) -> None:
            self.condition_schema: dict[str, Any] | None = None

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-staged-temporal-drs", "context_size": 8192}

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, Any] | None = None,
        ) -> dict[str, object]:
            if "one source-grounded DRS object" in prompt:
                return {"not_drs": {}, "_model_raw": "{}", "_model_elapsed_seconds": 0.01}
            if "Stage 1 of source-grounded DRS extraction" in prompt:
                assert "temporal records" in prompt
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
                                "evidence_text": "On 2026-01-03, Aero Gate is ready.",
                            }
                        ],
                        "temporal_records": [
                            {
                                "id": "t0",
                                "value": "2026-01-03",
                                "value_type": "date_time",
                                "evidence_text": "2026-01-03",
                            }
                        ],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            assert "Stage 2 of source-grounded DRS extraction" in prompt
            self.condition_schema = json_schema
            condition_schema = json_schema["properties"]["condition_stage"]["properties"]["conditions"]["items"]
            assert condition_schema["properties"]["temporal_id"]["enum"] == ["", "t0"]
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
                            "temporal_id": "t0",
                            "arguments": [
                                {
                                    "role": "theme",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "entity",
                                    "evidence_text": "Aero Gate",
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
    model = TemporalStagedFallbackModel()

    result = call_model_chunk_drs(
        "On 2026-01-03, Aero Gate is ready.",
        model,  # type: ignore[arg-type]
        rel_path="note.txt",
        n_predict=384,
    )

    assert result["accepted"] is True
    assert result["reason"] == "staged_fallback"
    assert result["fallback_from_reason"] == "schema_validation_failed"
    assert result["drs"]["temporal_records"] == [
        {"id": "t0", "value": "2026-01-03", "value_type": "date_time", "evidence_text": "2026-01-03"}
    ]
    assert result["drs"]["conditions"][0]["temporal_id"] == "t0"
    assert model.condition_schema is not None


def test_chunk_drs_staged_fallback_repairs_declared_label_evidence(monkeypatch, tmp_path) -> None:
    class LabelEvidenceRepairModel:
        def __init__(self) -> None:
            self.condition_prompt = ""

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-staged-grounding-repair-drs", "context_size": 8192}

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, Any] | None = None,
        ) -> dict[str, object]:
            if "one source-grounded DRS object" in prompt:
                return {"not_drs": {}, "_model_raw": "{}", "_model_elapsed_seconds": 0.01}
            if "Stage 1 of source-grounded DRS extraction" in prompt:
                return {
                    "drs_skeleton": {
                        "schema_version": "chunk-drs-v2",
                        "source_id": "object.raw",
                        "referents": [
                            {
                                "id": "r0",
                                "label": "OG-7003",
                                "kind": "identifier",
                                "evidence_text": 'ids.asset: "OG-7003"',
                            }
                        ],
                        "boxes": [
                            {
                                "id": "b0",
                                "kind": "asserted",
                                "parent_id": "",
                                "holder_referent_id": "",
                                "evidence_text": '{ ids: { asset: \\"OG-7003\\" } }',
                            }
                        ],
                        "temporal_records": [],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            assert "Stage 2 of source-grounded DRS extraction" in prompt
            self.condition_prompt = prompt
            assert '"evidence_text": "OG-7003"' in prompt
            return {
                "condition_stage": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "object.raw",
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "asset",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "value",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "OG-7003",
                                    "value_type": "identifier",
                                    "evidence_text": "OG-7003",
                                },
                                {
                                    "role": "field",
                                    "target_kind": "literal",
                                    "target_id": "asset",
                                    "value": "asset",
                                    "value_type": "literal",
                                    "evidence_text": "asset",
                                }
                            ],
                            "evidence_text": 'asset: "OG-7003"',
                        }
                    ],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.delenv("KMD_LOCAL_MODEL_JSON_SCHEMA", raising=False)
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "drs-cache"))
    model = LabelEvidenceRepairModel()

    result = call_model_chunk_drs(
        '{ ids: { asset: "OG-7003" } }',
        model,  # type: ignore[arg-type]
        rel_path="object.raw",
        n_predict=384,
    )

    assert result["accepted"] is True
    assert result["reason"] == "staged_fallback"
    assert result["drs"]["referents"][0]["evidence_text"] == "OG-7003"
    assert result["drs"]["boxes"][0]["evidence_text"] == '{ ids: { asset: "OG-7003" } }'
    assert result["drs"]["conditions"][0]["arguments"][1]["target_id"] == ""
    assert result["context_budget"]["grounding_repair_policy"] == CHUNK_DRS_GROUNDING_REPAIR_POLICY
    assert chunk_drs_cache_context(model, n_predict=384)["grounding_repair_policy"] == CHUNK_DRS_GROUNDING_REPAIR_POLICY
    assert model.condition_prompt
