from __future__ import annotations

import json
from typing import Any

from knowmoredirt.model import LocalModelJSONError
from knowmoredirt.model_planner import (
    CHUNK_DRS_BOX_COMPLETION_POLICY,
    CHUNK_DRS_COMPACT_UNDERCOVERAGE_POLICY,
    CHUNK_DRS_GROUNDING_REPAIR_POLICY,
    CHUNK_DRS_MONOLITHIC_ID_POLICY,
    CHUNK_DRS_SPARSE_RETRY_POLICY,
    CHUNK_DRS_SKELETON_ID_POLICY,
    CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY,
    CHUNK_DRS_SOURCE_SPAN_POLICY,
    CHUNK_DRS_STAGED_FALLBACK_POLICY,
    CHUNK_DRS_STAGED_RETRY_DIAGNOSTICS_POLICY,
    call_model_chunk_drs,
    chunk_drs_cache_context,
    chunk_drs_skeleton_json_schema,
    chunk_drs_source_span_candidates,
)


def test_chunk_drs_staged_fallback_constrains_condition_targets(monkeypatch, tmp_path) -> None:
    class StagedFallbackModel:
        def __init__(self) -> None:
            self.skeleton_schema: dict[str, Any] | None = None
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
                assert "declare a distinct subordinate box" in prompt
                assert "source_span_candidates" in prompt
                self.skeleton_schema = json_schema
                skeleton_schema = json_schema["properties"]["drs_skeleton"]["properties"]
                assert skeleton_schema["referents"]["items"]["properties"]["evidence_text"]["enum"] == [
                    "",
                    "Aero Gate is ready.",
                ]
                assert skeleton_schema["boxes"]["items"]["properties"]["evidence_text"]["enum"] == [
                    "",
                    "Aero Gate is ready.",
                ]
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
                            },
                            {
                                "id": "b1",
                                "kind": "reported",
                                "parent_id": "b0",
                                "holder_referent_id": "",
                                "evidence_text": "Aero Gate is ready.",
                            }
                        ],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            assert "Stage 2 of source-grounded DRS extraction" in prompt
            assert "distinct declared subordinate box for scoped content" in prompt
            self.condition_schema = json_schema
            condition_schema = json_schema["properties"]["condition_stage"]["properties"]["conditions"]["items"]
            argument_schema = condition_schema["properties"]["arguments"]["items"]
            assert condition_schema["properties"]["box_id"]["enum"] == ["b0", "b1"]
            assert argument_schema["properties"]["target_id"]["enum"] == ["", "b0", "b1", "r0"]
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
                                    "target_id": "b1",
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
    assert result["context_budget"]["skeleton_id_policy"] == CHUNK_DRS_SKELETON_ID_POLICY
    assert result["context_budget"]["skeleton_source_span_policy"] == CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY
    cache_context = chunk_drs_cache_context(model, n_predict=384)
    assert cache_context["staged_fallback_policy"] == CHUNK_DRS_STAGED_FALLBACK_POLICY
    assert cache_context["skeleton_id_policy"] == CHUNK_DRS_SKELETON_ID_POLICY
    assert cache_context["skeleton_source_span_policy"] == CHUNK_DRS_SKELETON_SOURCE_SPAN_POLICY
    assert model.skeleton_schema is not None
    assert model.condition_schema is not None


def test_chunk_drs_missing_box_completion_after_staged_failure(monkeypatch, tmp_path) -> None:
    class MissingBoxCompletionModel:
        def __init__(self) -> None:
            self.box_completion_schema: dict[str, Any] | None = None
            self.box_completion_prompt = ""

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-box-completion-drs", "context_size": 8192}

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
                            {"id": "r0", "label": "Kalo Reed", "kind": "person", "evidence_text": "Kalo Reed"}
                        ],
                        "boxes": [
                            {
                                "id": "b0",
                                "kind": "asserted",
                                "parent_id": "",
                                "holder_referent_id": "r0",
                                "evidence_text": "Kalo Reed believes the lantern should be painted blue.",
                            }
                        ],
                        "conditions": [
                            {
                                "id": "c0",
                                "predicate": "believe",
                                "box_id": "b0",
                                "polarity": "positive",
                                "modality": "asserted",
                                "temporal_id": "",
                                "arguments": [
                                    {
                                        "role": "holder",
                                        "target_kind": "referent",
                                        "target_id": "r0",
                                        "value": "",
                                        "value_type": "",
                                        "evidence_text": "Kalo Reed",
                                    },
                                    {
                                        "role": "content",
                                        "target_kind": "box",
                                        "target_id": "b1",
                                        "value": "",
                                        "value_type": "",
                                        "evidence_text": "the lantern should be painted blue",
                                    },
                                ],
                                "evidence_text": "Kalo Reed believes the lantern should be painted blue.",
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
                            {"id": "r0", "label": "Kalo Reed", "kind": "person", "evidence_text": "Kalo Reed"}
                        ],
                        "boxes": [
                            {
                                "id": "b0",
                                "kind": "asserted",
                                "parent_id": "",
                                "holder_referent_id": "r0",
                                "evidence_text": "Kalo Reed believes the lantern should be painted blue.",
                            }
                        ],
                        "temporal_records": [],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            if "Stage 2 of source-grounded DRS extraction" in prompt:
                return {
                    "condition_stage": {
                        "schema_version": "chunk-drs-v2",
                        "source_id": "note.txt",
                        "conditions": [
                            {
                                "id": "c0",
                                "predicate": "believe",
                                "box_id": "b0",
                                "polarity": "positive",
                                "modality": "asserted",
                                "temporal_id": "",
                                "arguments": [
                                    {
                                        "role": "content",
                                        "target_kind": "box",
                                        "target_id": "b0",
                                        "value": "",
                                        "value_type": "",
                                        "evidence_text": "the lantern should be painted blue",
                                    }
                                ],
                                "evidence_text": "Kalo Reed believes the lantern should be painted blue.",
                            }
                        ],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            assert "Complete missing source-grounded DRS box declarations" in prompt
            assert '"missing_box_ids": ["b1"]' in prompt
            self.box_completion_prompt = prompt
            self.box_completion_schema = json_schema
            box_schema = json_schema["properties"]["box_completion"]["properties"]["boxes"]["items"]
            assert box_schema["properties"]["id"]["enum"] == ["b1"]
            assert box_schema["properties"]["parent_id"]["enum"] == ["", "b0"]
            assert box_schema["properties"]["holder_referent_id"]["enum"] == ["", "r0"]
            return {
                "box_completion": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "note.txt",
                    "boxes": [
                        {
                            "id": "b1",
                            "kind": "believed",
                            "parent_id": "b0",
                            "holder_referent_id": "r0",
                            "evidence_text": "the lantern should be painted blue",
                        }
                    ],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.delenv("KMD_LOCAL_MODEL_JSON_SCHEMA", raising=False)
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "drs-cache"))
    model = MissingBoxCompletionModel()

    result = call_model_chunk_drs(
        "Kalo Reed believes the lantern should be painted blue.",
        model,  # type: ignore[arg-type]
        rel_path="note.txt",
        n_predict=384,
    )

    assert result["accepted"] is True
    assert result["reason"] == "box_completion_repair"
    assert result["fallback_from_reason"] == "schema_validation_failed"
    assert result["validation"]["box_count"] == 2
    assert result["drs"]["boxes"][1]["id"] == "b1"
    assert result["elapsed"] == 0.04
    assert result["context_budget"]["box_completion_policy"] == CHUNK_DRS_BOX_COMPLETION_POLICY
    assert chunk_drs_cache_context(model, n_predict=384)["box_completion_policy"] == CHUNK_DRS_BOX_COMPLETION_POLICY
    assert model.box_completion_schema is not None
    assert model.box_completion_prompt


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


def test_chunk_drs_staged_fallback_runs_for_structurally_sparse_drs(monkeypatch, tmp_path) -> None:
    class SparseFallbackModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-sparse-drs", "context_size": 8192}

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
                                "holder_referent_id": "r0",
                                "evidence_text": "Aero Gate is ready.",
                            }
                        ],
                        "conditions": [],
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
                                "holder_referent_id": "r0",
                                "evidence_text": "Aero Gate is ready.",
                            }
                        ],
                        "temporal_records": [],
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

    result = call_model_chunk_drs(
        "Aero Gate is ready.",
        SparseFallbackModel(),  # type: ignore[arg-type]
        rel_path="note.txt",
        n_predict=384,
    )

    assert result["accepted"] is True
    assert result["reason"] == "staged_fallback"
    assert result["fallback_from_reason"] == "structural_sparsity"
    assert result["validation"]["condition_count"] == 1
    assert result["context_budget"]["sparse_retry_policy"] == CHUNK_DRS_SPARSE_RETRY_POLICY
    cache_context = chunk_drs_cache_context(SparseFallbackModel(), n_predict=384)  # type: ignore[arg-type]
    assert cache_context["sparse_retry_policy"] == CHUNK_DRS_SPARSE_RETRY_POLICY


def test_chunk_drs_staged_fallback_runs_for_compact_record_undercoverage(monkeypatch, tmp_path) -> None:
    class CompactUndercoverageModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-compact-undercoverage-drs", "context_size": 8192}

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, Any] | None = None,
        ) -> dict[str, object]:
            if "one source-grounded DRS object" in prompt:
                assert "JSON schema constrains condition and argument evidence_text" in prompt
                return {
                    "drs": {
                        "schema_version": "chunk-drs-v2",
                        "source_id": "records.txt",
                        "referents": [
                            {"id": "r0", "label": "Aster Ridge", "kind": "asset", "evidence_text": "Aster Ridge"}
                        ],
                        "boxes": [
                            {
                                "id": "b0",
                                "kind": "asserted",
                                "parent_id": "",
                                "holder_referent_id": "r0",
                                "evidence_text": "record: Aster Ridge | steward: Lina Sol | state: active",
                            }
                        ],
                        "conditions": [
                            {
                                "id": "c0",
                                "predicate": "state",
                                "box_id": "b0",
                                "polarity": "positive",
                                "modality": "asserted",
                                "temporal_id": "",
                                "arguments": [
                                    {
                                        "role": "value",
                                        "target_kind": "literal",
                                        "target_id": "",
                                        "value": "active",
                                        "value_type": "state",
                                        "evidence_text": "state: active",
                                    }
                                ],
                                "evidence_text": "state: active",
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
                        "source_id": "records.txt",
                        "referents": [
                            {"id": "r0", "label": "Aster Ridge", "kind": "asset", "evidence_text": "Aster Ridge"}
                        ],
                        "boxes": [
                            {
                                "id": "b0",
                                "kind": "asserted",
                                "parent_id": "",
                                "holder_referent_id": "r0",
                                "evidence_text": "record: Aster Ridge | steward: Lina Sol | state: active",
                            }
                        ],
                        "temporal_records": [],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            assert "Stage 2 of source-grounded DRS extraction" in prompt
            condition_schema = json_schema["properties"]["condition_stage"]["properties"]["conditions"]["items"]
            assert "steward: Lina Sol" in condition_schema["properties"]["evidence_text"]["enum"]
            return {
                "condition_stage": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "records.txt",
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "steward",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "value",
                                    "target_kind": "literal",
                                    "target_id": "",
                                    "value": "Lina Sol",
                                    "value_type": "person",
                                    "evidence_text": "steward: Lina Sol",
                                }
                            ],
                            "evidence_text": "steward: Lina Sol",
                        },
                        {
                            "id": "c1",
                            "predicate": "state",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "value",
                                    "target_kind": "literal",
                                    "target_id": "",
                                    "value": "active",
                                    "value_type": "state",
                                    "evidence_text": "state: active",
                                }
                            ],
                            "evidence_text": "state: active",
                        },
                    ],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.delenv("KMD_LOCAL_MODEL_JSON_SCHEMA", raising=False)
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "drs-cache"))

    result = call_model_chunk_drs(
        "record: Aster Ridge | steward: Lina Sol | state: active",
        CompactUndercoverageModel(),  # type: ignore[arg-type]
        rel_path="records.txt",
        n_predict=384,
    )

    assert result["accepted"] is True
    assert result["reason"] == "staged_fallback"
    assert result["fallback_from_reason"] == "structural_undercoverage"
    assert result["validation"]["condition_count"] == 2
    assert result["context_budget"]["compact_undercoverage_policy"] == CHUNK_DRS_COMPACT_UNDERCOVERAGE_POLICY
    assert (
        result["context_budget"]["staged_retry_diagnostics_policy"] == CHUNK_DRS_STAGED_RETRY_DIAGNOSTICS_POLICY
    )
    cache_context = chunk_drs_cache_context(CompactUndercoverageModel(), n_predict=384)  # type: ignore[arg-type]
    assert cache_context["compact_undercoverage_policy"] == CHUNK_DRS_COMPACT_UNDERCOVERAGE_POLICY
    assert cache_context["staged_retry_diagnostics_policy"] == CHUNK_DRS_STAGED_RETRY_DIAGNOSTICS_POLICY


def test_chunk_drs_compact_undercoverage_records_non_improving_retry(monkeypatch, tmp_path) -> None:
    class NonImprovingRetryModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-non-improving-retry-drs", "context_size": 8192}

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
                        "source_id": "records.txt",
                        "referents": [
                            {"id": "r0", "label": "Aster Ridge", "kind": "asset", "evidence_text": "Aster Ridge"}
                        ],
                        "boxes": [
                            {
                                "id": "b0",
                                "kind": "asserted",
                                "parent_id": "",
                                "holder_referent_id": "r0",
                                "evidence_text": "record: Aster Ridge | steward: Lina Sol | state: active",
                            }
                        ],
                        "conditions": [
                            {
                                "id": "c0",
                                "predicate": "state",
                                "box_id": "b0",
                                "polarity": "positive",
                                "modality": "asserted",
                                "temporal_id": "",
                                "arguments": [],
                                "evidence_text": "state: active",
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
                        "source_id": "records.txt",
                        "referents": [
                            {"id": "r0", "label": "Aster Ridge", "kind": "asset", "evidence_text": "Aster Ridge"}
                        ],
                        "boxes": [
                            {
                                "id": "b0",
                                "kind": "asserted",
                                "parent_id": "",
                                "holder_referent_id": "r0",
                                "evidence_text": "record: Aster Ridge | steward: Lina Sol | state: active",
                            }
                        ],
                        "temporal_records": [],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            return {
                "condition_stage": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "records.txt",
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "state",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [],
                            "evidence_text": "state: active",
                        }
                    ],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.delenv("KMD_LOCAL_MODEL_JSON_SCHEMA", raising=False)
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "drs-cache"))

    result = call_model_chunk_drs(
        "record: Aster Ridge | steward: Lina Sol | state: active",
        NonImprovingRetryModel(),  # type: ignore[arg-type]
        rel_path="records.txt",
        n_predict=384,
    )

    assert result["accepted"] is True
    assert result["validation"]["condition_count"] == 1
    assert result["staged_retry"]["accepted"] is True
    assert result["staged_retry"]["fallback_from_reason"] == "structural_undercoverage"
    assert result["staged_retry"]["monolithic_condition_count"] == 1
    assert result["staged_retry"]["fallback_condition_count"] == 1
    assert result["context_budget"]["staged_retry_diagnostics_policy"] == CHUNK_DRS_STAGED_RETRY_DIAGNOSTICS_POLICY


def test_chunk_drs_source_span_candidates_skip_field_headers() -> None:
    spans = chunk_drs_source_span_candidates(
        '{ name: "Orchid Gamma", ids: [asset: "OG-7003", audit: "AUD-3003"] }',
        max_evidence_chars=96,
    )

    assert "" in spans
    assert 'name: "Orchid Gamma"' in spans
    assert "Orchid Gamma" in spans
    assert 'asset: "OG-7003"' in spans
    assert "OG-7003" in spans
    assert "ids:" not in spans


def test_chunk_drs_monolithic_schema_constrains_ids_and_condition_spans(monkeypatch, tmp_path) -> None:
    class MonolithicSchemaModel:
        def __init__(self) -> None:
            self.prompt = ""
            self.schema: dict[str, Any] | None = None

        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-monolithic-span-id-drs", "context_size": 8192}

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, Any] | None = None,
        ) -> dict[str, object]:
            self.prompt = prompt
            self.schema = json_schema
            drs_schema = json_schema["properties"]["drs"]
            condition_schema = drs_schema["properties"]["conditions"]["items"]
            argument_schema = condition_schema["properties"]["arguments"]["items"]
            referent_schema = drs_schema["properties"]["referents"]["items"]
            box_schema = drs_schema["properties"]["boxes"]["items"]
            temporal_schema = drs_schema["properties"]["temporal_records"]["items"]
            assert drs_schema["properties"]["source_id"]["enum"] == ["records.txt"]
            assert referent_schema["properties"]["id"]["enum"] == ["r0", "r1", "r2", "r3"]
            assert box_schema["properties"]["id"]["enum"] == ["b0", "b1", "b2", "b3"]
            assert condition_schema["properties"]["id"]["enum"] == ["c0", "c1", "c2", "c3"]
            assert condition_schema["properties"]["box_id"]["enum"] == ["b0", "b1", "b2", "b3"]
            assert temporal_schema["properties"]["id"]["enum"] == ["t0", "t1", "t2", "t3"]
            assert "steward: Lina Sol" in condition_schema["properties"]["evidence_text"]["enum"]
            assert "Lina Sol" in condition_schema["properties"]["evidence_text"]["enum"]
            assert "r0" in argument_schema["properties"]["target_id"]["enum"]
            assert "b0" in argument_schema["properties"]["target_id"]["enum"]
            assert "c0" in argument_schema["properties"]["target_id"]["enum"]
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "records.txt",
                    "referents": [
                        {"id": "r0", "label": "Aster Ridge", "kind": "asset", "evidence_text": "Aster Ridge"}
                    ],
                    "boxes": [
                        {
                            "id": "b0",
                            "kind": "asserted",
                            "parent_id": "",
                            "holder_referent_id": "r0",
                            "evidence_text": "record: Aster Ridge | steward: Lina Sol | state: active",
                        }
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "steward",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "value",
                                    "target_kind": "literal",
                                    "target_id": "",
                                    "value": "Lina Sol",
                                    "value_type": "person",
                                    "evidence_text": "steward: Lina Sol",
                                }
                            ],
                            "evidence_text": "steward: Lina Sol",
                        },
                        {
                            "id": "c1",
                            "predicate": "state",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "value",
                                    "target_kind": "literal",
                                    "target_id": "",
                                    "value": "active",
                                    "value_type": "state",
                                    "evidence_text": "state: active",
                                }
                            ],
                            "evidence_text": "state: active",
                        },
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.delenv("KMD_LOCAL_MODEL_JSON_SCHEMA", raising=False)
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "drs-cache"))
    model = MonolithicSchemaModel()

    result = call_model_chunk_drs(
        "record: Aster Ridge | steward: Lina Sol | state: active",
        model,  # type: ignore[arg-type]
        rel_path="records.txt",
        n_predict=384,
    )

    assert result["accepted"] is True
    assert result["context_budget"]["source_span_policy"] == CHUNK_DRS_SOURCE_SPAN_POLICY
    assert result["context_budget"]["monolithic_id_policy"] == CHUNK_DRS_MONOLITHIC_ID_POLICY
    assert result["context_budget"]["source_span_candidate_count"] >= 6
    assert chunk_drs_cache_context(model, n_predict=384)["monolithic_id_policy"] == CHUNK_DRS_MONOLITHIC_ID_POLICY
    assert model.schema is not None
    assert "JSON schema constrains condition and argument evidence_text" in model.prompt


def test_chunk_drs_skeleton_schema_uses_stable_id_namespaces() -> None:
    schema = chunk_drs_skeleton_json_schema("note.txt", max_array_items=4)
    skeleton_schema = schema["properties"]["drs_skeleton"]["properties"]
    referent_item = skeleton_schema["referents"]["items"]
    box_item = skeleton_schema["boxes"]["items"]
    temporal_item = skeleton_schema["temporal_records"]["items"]

    assert referent_item["properties"]["id"]["enum"] == ["r0", "r1", "r2", "r3"]
    assert box_item["properties"]["id"]["enum"] == ["b0", "b1", "b2", "b3"]
    assert box_item["properties"]["parent_id"]["enum"] == ["", "b0", "b1", "b2", "b3"]
    assert box_item["properties"]["holder_referent_id"]["enum"] == ["", "r0", "r1", "r2", "r3"]
    assert temporal_item["properties"]["id"]["enum"] == ["t0", "t1", "t2", "t3"]


def test_chunk_drs_failed_staged_fallback_keeps_stage_diagnostics(monkeypatch, tmp_path) -> None:
    class FailedStagedFallbackModel:
        def context_size(self) -> int:
            return 8192

        def cache_fingerprint(self) -> dict[str, Any]:
            return {"model_id": "fake-failed-staged-drs", "context_size": 8192}

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, Any] | None = None,
        ) -> dict[str, object]:
            if "one source-grounded DRS object" in prompt:
                raise LocalModelJSONError(
                    "bad monolithic json",
                    raw_text='{"drs":',
                    snippet='{"drs":',
                )
            if "Stage 1 of source-grounded DRS extraction" in prompt:
                raise LocalModelJSONError(
                    "bad skeleton json",
                    raw_text='{"drs_skeleton":',
                    snippet='{"drs_skeleton":',
                )
            raise AssertionError("condition stage should not run after skeleton JSON failure")

    monkeypatch.delenv("KMD_LOCAL_MODEL_JSON_SCHEMA", raising=False)
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "drs-cache"))
    model = FailedStagedFallbackModel()

    result = call_model_chunk_drs(
        "Aero Gate is ready.",
        model,  # type: ignore[arg-type]
        rel_path="note.txt",
        n_predict=384,
    )

    assert result["accepted"] is False
    assert result["reason"] == "invalid_json"
    assert result["staged_fallback"]["accepted"] is False
    assert result["staged_fallback"]["reason"] == "invalid_json"
    assert result["staged_fallback"]["stage"] == "skeleton"
    assert result["staged_fallback"]["error"] == "bad skeleton json"
    assert result["staged_fallback"]["raw_snippet"] == '{"drs_skeleton":'


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
            assert '"declared_temporal_records": [{"id": "t0"' in prompt
            assert "set that condition's temporal_id" in prompt
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
            self.condition_schema: dict[str, Any] | None = None

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
            self.condition_schema = json_schema
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
                            "evidence_text": 'asset: \\"OG-7003\\"',
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
    assert result["drs"]["conditions"][0]["evidence_text"] == 'asset: "OG-7003"'
    assert result["drs"]["conditions"][0]["arguments"][1]["target_id"] == ""
    assert result["context_budget"]["grounding_repair_policy"] == CHUNK_DRS_GROUNDING_REPAIR_POLICY
    assert chunk_drs_cache_context(model, n_predict=384)["grounding_repair_policy"] == CHUNK_DRS_GROUNDING_REPAIR_POLICY
    assert model.condition_prompt
    assert "source_span_candidates" in model.condition_prompt
    assert model.condition_schema is not None
    condition_schema = model.condition_schema["properties"]["condition_stage"]["properties"]["conditions"]["items"]
    evidence_values = condition_schema["properties"]["evidence_text"]["enum"]
    assert 'asset: "OG-7003"' in evidence_values
    assert "ids:" not in evidence_values
    assert condition_schema["properties"]["id"]["enum"] == ["c0", "c1", "c2", "c3"]
