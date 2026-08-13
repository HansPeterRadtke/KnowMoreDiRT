from __future__ import annotations

from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.models import Evidence


def _engine_with_evidence(kind: str | None):
    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    evidence = [Evidence("timmy.txt", "The flying-car law requires blue lights.", span_id="span-1")]
    engine._diagnostic_unknown_evidence = lambda **_kwargs: evidence
    engine._evidence_context_kinds = lambda _item: (() if kind is None else (kind,))
    engine._context_char_capacity = lambda *_args, **kwargs: int(kwargs.get("available") or 500)
    return engine


def test_unknown_surfaces_dream_evidence_without_promoting_it_to_fact() -> None:
    answer = _engine_with_evidence("drs:dreamed")._unknown_answer("no asserted answer")
    assert answer.answer_type == "unknown"
    assert answer.text.startswith("unknown — relevant dreamed evidence")
    assert "timmy.txt" in answer.text
    assert "flying-car law" in answer.text


def test_unknown_without_subordinate_evidence_remains_plain_unknown() -> None:
    answer = _engine_with_evidence("drs:asserted")._unknown_answer("no answer")
    assert answer.text == "unknown"
    assert answer.answer_type == "unknown"


def test_qualified_unknown_remains_incomplete_and_unknown_typed() -> None:
    from knowmoredirt.answer_types import classify_value, is_unknown_text

    text = "unknown — relevant dreamed evidence in story.txt: blue lamps"
    assert is_unknown_text(text)
    assert classify_value(text) == "unknown"
    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    from knowmoredirt.models import Answer
    assert engine._complete_answer(Answer(text, answer_type="unknown")) is False
    assert engine._answer_has_source_grounding(Answer(text, answer_type="unknown")) is True


def test_qualified_unknown_sees_subordinate_ancestor_context() -> None:
    from knowmoredirt.store import DSPGStore

    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    engine.store = DSPGStore()
    engine.run_id = "run"
    engine._sentences_by_document = {}
    engine._context_char_capacity = lambda *_args, **kwargs: int(kwargs.get("available") or 500)
    engine.store.execute("INSERT INTO extraction_runs VALUES (?, ?, ?, ?, ?)", ("run", 1.0, "/tmp", "running", "{}"))
    engine.store.execute(
        "INSERT INTO documents(document_id, run_id, path, rel_path, content_hash, size_bytes, mtime, ctime, char_count, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("doc-1", "run", "doc-1", "doc-1", "hash", 10, 0.0, 0.0, 10, "{}"),
    )
    engine.store.execute(
        "INSERT INTO chunks(chunk_id, document_id, chunk_order, char_start, char_end, text, token_estimate) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("chunk-1", "doc-1", 0, 0, 10, "dream fact", 2),
    )
    engine.store.execute(
        "INSERT INTO source_spans(span_id, document_id, chunk_id, char_start, char_end, surface, surface_norm, span_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("span-1", "doc-1", "chunk-1", 0, 10, "dream fact", "dream fact", "sentence"),
    )
    engine.store.execute("INSERT INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, ?, NULL, ?, ?)", ("root", "run", "drs:asserted", "dream", "asserted", 1.0))
    engine.store.execute("INSERT INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, NULL, NULL, ?, ?)", ("dream", "run", "drs:dreamed", "all a dream", 0.99))
    engine.store.execute(
        "INSERT INTO context_assignments VALUES (?, ?, ?, 'source_span', ?, ?, ?)",
        ("ca", "run", "root", "span-1", "span-1", 1.0),
    )
    evidence = Evidence("story.txt", "Flying cars use blue lamps.", span_id="span-1")
    engine._diagnostic_unknown_evidence = lambda **_kwargs: [evidence]
    answer = engine._unknown_answer("no asserted answer")
    assert "relevant dreamed evidence" in answer.text


def test_qualified_unknown_does_not_attach_irrelevant_subordinate_evidence() -> None:
    from knowmoredirt.model_planner import ModelQueryTrace

    engine = _engine_with_evidence("drs:reported")
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine.model_query_trace.last_plan = {
        "target_anchors": ["France"],
        "requested_relation": "capital",
        "relation_terms": ["capital"],
        "constraints": [],
    }
    answer = engine._unknown_answer("no asserted answer")
    assert answer.text == "unknown"


def test_qualified_unknown_keeps_relevant_subordinate_evidence_for_query_target() -> None:
    from knowmoredirt.model_planner import ModelQueryTrace

    engine = _engine_with_evidence("drs:dreamed")
    engine._diagnostic_unknown_evidence = lambda **_kwargs: [
        Evidence("story.txt", "Flying cars must display two blue lamps after sunset.", span_id="span-1")
    ]
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine.model_query_trace.last_plan = {
        "target_anchors": ["flying cars"],
        "requested_relation": "applies",
        "relation_terms": ["law"],
        "constraints": [],
    }
    answer = engine._unknown_answer("no asserted answer")
    assert "relevant dreamed evidence" in answer.text


def test_qualified_unknown_multi_anchor_requires_every_target_anchor() -> None:
    from knowmoredirt.model_planner import ModelQueryTrace

    engine = _engine_with_evidence("drs:negated")
    engine._diagnostic_unknown_evidence = lambda **_kwargs: [
        Evidence("product.json", "BeaconForce must not merge PR-2814 until the callback replay test passes.", span_id="span-1")
    ]
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine.model_query_trace.last_plan = {
        "target_anchors": ["billing export redesign", "BeaconForce"],
        "requested_relation": "approved",
        "relation_terms": ["approved"],
        "constraints": [],
    }
    answer = engine._unknown_answer("no asserted answer")
    assert answer.text == "unknown"


def test_qualified_unknown_multi_anchor_keeps_evidence_matching_all_targets() -> None:
    from knowmoredirt.model_planner import ModelQueryTrace

    engine = _engine_with_evidence("drs:reported")
    engine._diagnostic_unknown_evidence = lambda **_kwargs: [
        Evidence("report.txt", "The report mentioned the BeaconForce billing export redesign, but gave no approval decision.", span_id="span-1")
    ]
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine.model_query_trace.last_plan = {
        "target_anchors": ["billing export redesign", "BeaconForce"],
        "requested_relation": "approved",
        "relation_terms": ["approved"],
        "constraints": [],
    }
    answer = engine._unknown_answer("no asserted answer")
    assert "relevant reported evidence" in answer.text


def test_qualified_unknown_prefers_specific_answer_slot_when_target_is_scope_shaped() -> None:
    from knowmoredirt.model_planner import ModelQueryTrace

    engine = _engine_with_evidence("drs:dreamed")
    engine._diagnostic_unknown_evidence = lambda **_kwargs: [
        Evidence(
            "sleep.txt",
            "A clerk said canal scooters must carry a white pennant after midnight.",
            span_id="span-1",
        )
    ]
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine.model_query_trace.last_plan = {
        "answer_variables": ["canal-scooter rule"],
        "target_anchors": ["waking life"],
        "requested_relation": "applies",
        "relation_terms": ["applies", "canal-scooter rule"],
        "constraints": [],
    }
    answer = engine._unknown_answer("no asserted answer")
    assert "relevant dreamed evidence" in answer.text
    assert "canal scooters" in answer.text


def test_qualified_unknown_specific_answer_slot_rejects_unrelated_subordinate_topic() -> None:
    from knowmoredirt.model_planner import ModelQueryTrace

    engine = _engine_with_evidence("drs:hypothetical")
    engine._diagnostic_unknown_evidence = lambda **_kwargs: [
        Evidence(
            "other.txt",
            "Hypothetical safety exercise: cargo walkers carry two red flags.",
            span_id="span-1",
        )
    ]
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine.model_query_trace.last_plan = {
        "answer_variables": ["canal-scooter rule"],
        "target_anchors": [],
        "requested_relation": "applies",
        "relation_terms": ["applies", "canal-scooter rule"],
        "constraints": ["in the real world"],
    }
    answer = engine._unknown_answer("no asserted answer")
    assert answer.text == "unknown"
