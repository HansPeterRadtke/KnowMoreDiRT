from __future__ import annotations

from knowmoredirt.answer_types import ExpectedAnswer
from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.query import QueryFrame


def _engine() -> KnowMoreDiRTEngine:
    return KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)


def _frame(question: str, *, targets: tuple[str, ...], relation: str, variables: tuple[str, ...] = ("What",)) -> QueryFrame:
    return QueryFrame(
        question_text=question,
        answer_type="content_phrase",
        answer_variables=variables,
        target_anchors=targets,
        requested_relation=relation,
        relation_terms=(relation,),
        constraints=(),
        source="model_query_drs",
    )


def test_selects_unique_authoritative_clause_without_using_expected_answer() -> None:
    frame = _frame(
        "What did the correction say about Delta shipping the green case?",
        targets=("Delta", "green case"),
        relation="shipping",
    )
    value = "Delta did not ship the green case; the corrected priority was high"
    assert _engine()._select_authoritative_answer_clause(value, frame) == "Delta did not ship the green case"


def test_extracts_reported_subject_binding_from_question_complement() -> None:
    question = "What did Inez say cracked during transit?"
    value = "the ceramic seal cracked during transit"
    assert _engine()._extract_reported_subject_binding(question, value) == "the ceramic seal"


def test_collapses_generic_reporting_wrapper_for_reported_content() -> None:
    question = "What did the forwarded note say about repairing module.py?"
    value = "Ari said they will repair module.py tomorrow"
    assert _engine()._collapse_reported_content_wrapper(
        question,
        value,
        ExpectedAnswer("content_phrase"),
    ) == "Ari will repair module.py tomorrow"


def test_does_not_collapse_reporting_wrapper_for_non_reported_question() -> None:
    question = "Who said they will repair module.py tomorrow?"
    value = "Ari said they will repair module.py tomorrow"
    assert _engine()._collapse_reported_content_wrapper(
        question,
        value,
        ExpectedAnswer("person"),
    ) == value


def test_generic_what_slot_does_not_trigger_residual_clause_stripping() -> None:
    engine = _engine()
    question = "What did the forwarded note say about repairing module.py?"
    frame = _frame(
        question,
        targets=("forwarded note",),
        relation="say repairing",
        variables=("What",),
    )
    value = "Ari said he plans to repair module.py tomorrow"

    cleaned = engine._cleanup_authoritative_surface_answer(
        question,
        value,
        ExpectedAnswer("content_phrase"),
        frame,
        [],
    )

    assert cleaned == "Ari plans to repair module.py tomorrow"
