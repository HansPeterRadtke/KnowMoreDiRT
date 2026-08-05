from __future__ import annotations

from typing import Any

from knowmoredirt.model_planner import (
    _repair_chunk_drs_payload,
    _structured_source_record_surfaces,
    _validate_chunk_drs_payload,
)


def _status_payload(source: str, *, status: str, condition_evidence: str) -> dict[str, Any]:
    return {
        "drs": {
            "schema_version": "chunk-drs-v5",
            "source_id": "records",
            "referents": [
                {"id": "r0", "label": "Alice", "kind": "person", "evidence_text": "Alice"}
            ],
            "boxes": [
                {
                    "id": "b0",
                    "kind": "asserted",
                    "parent_id": "",
                    "holder_referent_id": "",
                    "evidence_text": source,
                }
            ],
            "conditions": [
                {
                    "id": "c0",
                    "predicate": "status",
                    "box_id": "b0",
                    "polarity": "positive",
                    "modality": "asserted",
                    "temporal_id": "",
                    "evidence_text": condition_evidence,
                    "arguments": [
                        {
                            "role": "subject",
                            "target_kind": "referent",
                            "target_id": "r0",
                            "value": "",
                            "value_type": "person",
                            "evidence_text": "Alice",
                        },
                        {
                            "role": "status",
                            "target_kind": "literal",
                            "target_id": "",
                            "value": status,
                            "value_type": "state",
                            "evidence_text": status,
                        },
                    ],
                }
            ],
            "identity_hypotheses": [],
            "temporal_records": [],
            "evidence_spans": [],
            "semantic_notes": [],
        }
    }


def _validation(source: str, *, status: str, evidence: str) -> dict[str, Any]:
    repaired = _repair_chunk_drs_payload(
        _status_payload(source, status=status, condition_evidence=evidence),
        source,
    )
    return _validate_chunk_drs_payload(repaired, source)


def test_json_array_cross_record_claim_is_rejected() -> None:
    source = '[{"name":"Alice","status":"open"},{"name":"Bob","status":"closed"}]'

    validation = _validation(source, status="closed", evidence=source)

    assert len(_structured_source_record_surfaces(source)) == 2
    assert validation["schema_valid"] is False
    assert "condition_evidence_not_localized:c0" in validation["grounding_failures"]


def test_json_array_correct_record_claim_is_accepted() -> None:
    source = '[{"name":"Alice","status":"open"},{"name":"Bob","status":"closed"}]'

    validation = _validation(
        source,
        status="open",
        evidence='{"name":"Alice","status":"open"}',
    )

    assert validation["schema_valid"] is True
    assert validation["grounding_failures"] == []


def test_plain_text_cross_sentence_claim_is_rejected() -> None:
    source = "Alice is open. Bob is closed."

    validation = _validation(source, status="closed", evidence=source)

    assert len(_structured_source_record_surfaces(source)) == 2
    assert validation["schema_valid"] is False
    assert "condition_evidence_not_localized:c0" in validation["grounding_failures"]


def test_plain_text_correct_sentence_claim_is_accepted() -> None:
    source = "Alice is open. Bob is closed."

    validation = _validation(source, status="open", evidence="Alice is open.")

    assert validation["schema_valid"] is True
    assert validation["grounding_failures"] == []


def test_jsonl_cross_record_target_is_rejected() -> None:
    source = '{"name":"Alice","status":"open"}\n{"name":"Bob","status":"closed"}'

    validation = _validation(
        source,
        status="closed",
        evidence='{"name":"Bob","status":"closed"}',
    )

    assert validation["schema_valid"] is False
    assert "condition_argument_record_mismatch:c0:subject" in validation["grounding_failures"]


def test_nested_records_inherit_only_ancestor_scalar_context() -> None:
    source = (
        '{"product":"BeaconForce","items":['
        '{"component":"retry scheduler","status":"open"},'
        '{"component":"callback","status":"closed"}]}'
    )
    units = _structured_source_record_surfaces(source)

    assert len(units) == 2
    assert all("BeaconForce" in unit for unit in units)
    assert not any("retry scheduler" in unit and "callback" in unit for unit in units)
