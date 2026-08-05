from __future__ import annotations

from knowmoredirt.model_planner import (
    _query_evidence_payload_from_result,
    query_frame_from_query_drs,
)
from knowmoredirt.query import QueryFrame


def _frame(question: str, answer_type: str, variable: str, target: str, relation: str) -> dict[str, object]:
    return QueryFrame(
        question_text=question,
        answer_type=answer_type,
        answer_variables=(variable,),
        target_anchors=(target,),
        requested_relation=relation,
        relation_terms=(relation,),
        constraints=(),
        source="model_query_drs",
    ).as_dict()


def _payload(
    *,
    question: str,
    result: dict[str, object],
    evidence: str,
    authoritative_frame: dict[str, object],
    authoritative_answer_type: str,
) -> dict[str, object]:
    return _query_evidence_payload_from_result(
        question,
        result,
        [{"rel_path": "source.txt", "text": evidence}],
        "{}",
        0.01,
        "prompt",
        "grammar",
        fresh_or_cached="test",
        authoritative_query_frame=authoritative_frame,
        authoritative_answer_type=authoritative_answer_type,
    )


def test_authoritative_content_frame_overrides_wrong_count_label() -> None:
    question = "What alloy is named in the fabrication note?"
    authoritative = _frame(question, "content_phrase", "alloy", "fabrication note", "named")
    result = {
        "query_frame": _frame(question, "count", "alloy_name", "alloy", "named"),
        "sufficient_evidence": True,
        "answer_type": "count",
        "answer": "titanium alloy",
        "evidence_span": "Alloy: titanium alloy.",
        "reason": "The labeled source line binds the catalyst value.",
    }

    payload = _payload(
        question=question,
        result=result,
        evidence="Alloy: titanium alloy.",
        authoritative_frame=authoritative,
        authoritative_answer_type="content_phrase",
    )

    assert payload["accepted"] is True
    assert payload["answer"] == "titanium alloy"
    assert payload["answer_type"] == "content_phrase"
    assert payload["model_answer_type"] == "count"
    assert payload["query_frame"]["answer_type"] == "content_phrase"


def test_grounded_value_shape_overrides_wrong_redundant_type_when_authoritative_type_unknown() -> None:
    question = "Where is the bronze statue?"
    authoritative = _frame(question, "unknown", "Where", "bronze statue", "is")
    result = {
        "query_frame": _frame(question, "date_time", "location", "bronze statue", "location"),
        "sufficient_evidence": True,
        "answer_type": "date_time",
        "answer": "on the green shelf",
        "evidence_span": "The bronze statue is on the green shelf.",
        "reason": "The spatial source line states the location.",
    }

    payload = _payload(
        question=question,
        result=result,
        evidence="The bronze statue is on the green shelf.",
        authoritative_frame=authoritative,
        authoritative_answer_type="unknown",
    )

    assert payload["accepted"] is True
    assert payload["answer"] == "on the green shelf"
    assert payload["answer_type"] == "content_phrase"
    assert payload["model_answer_type"] == "date_time"
    assert payload["query_frame"]["answer_type"] == "unknown"



def test_explicit_denial_of_requested_positive_slot_normalizes_to_unknown() -> None:
    question = "Which manager was assigned to the project?"
    authoritative = _frame(question, "person", "manager", "project", "assigned")
    result = {
        "query_frame": authoritative,
        "sufficient_evidence": True,
        "answer_type": "person",
        "answer": "No manager was assigned",
        "evidence_span": "No manager was assigned to the project.",
        "reason": "The source states that no manager was assigned.",
    }

    payload = _payload(
        question=question,
        result=result,
        evidence="No manager was assigned to the project.",
        authoritative_frame=authoritative,
        authoritative_answer_type="person",
    )

    assert payload["accepted"] is True
    assert payload["sufficient_evidence"] is False
    assert payload["answer"] == "unknown"
    assert payload["answer_type"] == "unknown"
    assert payload["model_answer"] == "No manager was assigned"
    assert payload["explicit_denial_normalized"] is True


def test_negative_status_value_is_not_mistaken_for_absence_of_status_slot() -> None:
    question = "What status was recorded for the request?"
    authoritative = _frame(question, "state", "status", "request", "recorded")
    result = {
        "query_frame": authoritative,
        "sufficient_evidence": True,
        "answer_type": "state",
        "answer": "not approved",
        "evidence_span": "Status: not approved.",
        "reason": "The source records the negative status value.",
    }

    payload = _payload(
        question=question,
        result=result,
        evidence="Status: not approved.",
        authoritative_frame=authoritative,
        authoritative_answer_type="state",
    )

    assert payload["accepted"] is True
    assert payload["sufficient_evidence"] is True
    assert payload["answer"] == "not approved"
    assert payload["explicit_denial_normalized"] is False


def test_quoted_denial_remains_valid_content_when_denial_is_not_the_requested_slot() -> None:
    question = "What did the notice say?"
    authoritative = _frame(question, "content_phrase", "notice content", "notice", "said")
    result = {
        "query_frame": authoritative,
        "sufficient_evidence": True,
        "answer_type": "content_phrase",
        "answer": "No manager was assigned",
        "evidence_span": "The notice said: No manager was assigned.",
        "reason": "The question requests the notice content itself.",
    }

    payload = _payload(
        question=question,
        result=result,
        evidence="The notice said: No manager was assigned.",
        authoritative_frame=authoritative,
        authoritative_answer_type="content_phrase",
    )

    assert payload["accepted"] is True
    assert payload["sufficient_evidence"] is True
    assert payload["answer"] == "No manager was assigned"
    assert payload["explicit_denial_normalized"] is False

def _query_drs(question: str, answer_type: str, target: str) -> dict[str, object]:
    return {
        "schema_version": "query-drs-v3",
        "question": question,
        "answer_variables": [
            {"id": "qv0", "label": "Where", "answer_type": answer_type, "evidence_text": "Where"}
        ],
        "target_referents": [
            {"id": "qr0", "label": target, "kind": "entity", "evidence_text": target}
        ],
        "temporal_records": [],
        "requested_conditions": [
            {
                "id": "qc0",
                "predicate": "is",
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
                        "value_type": answer_type,
                        "evidence_text": "Where",
                    },
                    {
                        "role": "theme",
                        "target_kind": "referent",
                        "target_id": "qr0",
                        "value": target,
                        "value_type": "entity",
                        "evidence_text": target,
                    },
                ],
                "evidence_text": question,
            }
        ],
        "constraints": [],
        "box_requirements": [],
        "temporal_scope": "",
        "aggregation": "",
        "answer_type": answer_type,
        "requires_evidence": True,
    }


def test_spatial_where_question_rejects_unsupported_file_path_type() -> None:
    frame = query_frame_from_query_drs(
        "Where is the bronze statue?",
        _query_drs("Where is the bronze statue?", "file_path", "bronze statue"),
    )
    assert frame is not None
    assert frame["answer_type"] == "unknown"



def test_authoritative_url_type_survives_location_wording() -> None:
    frame = query_frame_from_query_drs(
        "Where is the design document stored?",
        _query_drs("Where is the design document stored?", "url", "design document"),
    )
    assert frame is not None
    assert frame["answer_type"] == "url"

def test_explicit_file_question_preserves_file_path_type() -> None:
    frame = query_frame_from_query_drs(
        "Where is report.txt stored?",
        _query_drs("Where is report.txt stored?", "file_path", "report.txt"),
    )
    assert frame is not None
    assert frame["answer_type"] == "file_path"


def test_explicit_denial_of_decision_relation_normalizes_to_unknown() -> None:
    question = "Which retention period did the group choose?"
    authoritative = _frame(question, "metadata_value", "final decision", "group", "choose")
    result = {
        "query_frame": authoritative,
        "sufficient_evidence": True,
        "answer_type": "metadata_value",
        "answer": "The group reached no final decision",
        "evidence_span": "The group reached no final decision during the meeting.",
        "reason": "The source denies a completed choice.",
    }

    payload = _payload(
        question=question,
        result=result,
        evidence="The group reached no final decision during the meeting.",
        authoritative_frame=authoritative,
        authoritative_answer_type="metadata_value",
    )

    assert payload["accepted"] is True
    assert payload["answer"] == "unknown"
    assert payload["explicit_denial_normalized"] is True
