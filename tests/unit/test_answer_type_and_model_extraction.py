from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import knowmoredirt.engine as engine_module
from knowmoredirt.answer_types import ExpectedAnswer, canonicalize_answer
from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.model_planner import ModelQueryTrace, call_model_evidence_answer
from knowmoredirt.models import Answer, Evidence
from knowmoredirt.query import QueryFrame




def test_person_compatibility_rejects_plain_state_words() -> None:
    assert canonicalize_answer(ExpectedAnswer("person"), "healthy") == ""
    assert canonicalize_answer(ExpectedAnswer("person"), "person") == ""
    assert canonicalize_answer(ExpectedAnswer("person"), "Meaningful note") == ""
    assert canonicalize_answer(ExpectedAnswer("person"), "wat3r3d maybe //// Clear correction") == ""
    assert canonicalize_answer(ExpectedAnswer("person"), "Dr. Pella") == "Dr. Pella"


def test_person_canonicalization_preserves_honorifics() -> None:
    assert canonicalize_answer(ExpectedAnswer("person"), "Dr. Pella") == "Dr. Pella"
    assert canonicalize_answer(ExpectedAnswer("person"), "the fern owner is Dr. Pella") == "Dr. Pella"
    assert canonicalize_answer(ExpectedAnswer("person"), "Officer Talen") == "Talen"


def test_bounded_person_answer_completes_unique_name_across_chunk_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "0")
    source = (
        "Atlas dossier handoff narrative.\n\n"
        "After the handoff, the responsible engineer is Mara\n\n"
        "Voss.\n"
    )
    (tmp_path / "record.txt").write_text(source, encoding="utf-8")
    engine = KnowMoreDiRTEngine(tmp_path)
    evidence = engine._evidence(engine.sentences[-1])
    assert "Mara" in engine.sentences[-2].text
    assert "Voss" in engine.sentences[-1].text
    frame = QueryFrame(
        question_text="Who is the responsible engineer for Atlas dossier?",
        answer_type="person",
        answer_variables=("responsible engineer",),
        target_anchors=("Atlas dossier",),
        requested_relation="responsible engineer",
        relation_terms=("responsible engineer", "engineer"),
        constraints=(),
        source="model_query_drs",
    )

    finalized = engine._finalize_answer(
        frame.question_text,
        Answer("Voss", 0.78, [evidence], "bounded relation binding", "person"),
        ExpectedAnswer("person"),
        "bounded DSPG query-frame execution",
        frame,
    )

    assert finalized is not None
    assert finalized.text == "Mara Voss"
    assert finalized.evidence == [evidence]


def test_identifier_answer_accepts_url_shaped_structural_identifier() -> None:
    expected = ExpectedAnswer("identifier")

    assert canonicalize_answer(expected, "https://manuals.example.test/lark-mirror") == "https://manuals.example.test/lark-mirror"
    assert canonicalize_answer(expected, "copper sulfate") == "copper sulfate"


def test_identifier_answer_accepts_structured_source_list_phrase() -> None:
    expected = ExpectedAnswer("identifier")

    assert (
        canonicalize_answer(expected, "SPEC-1, PR-2, and https://plans.example.test/item")
        == "SPEC-1, PR-2, and https://plans.example.test/item"
    )


def test_identifier_answer_rejects_unstructured_comma_list_phrase() -> None:
    expected = ExpectedAnswer("identifier")

    assert canonicalize_answer(expected, "alpha, beta, and gamma") == ""


class FakeEvidenceModel:
    def context_size(self) -> int:
        return 4096

    def __init__(self, *, incompatible: bool = False) -> None:
        self.incompatible = incompatible
        self.calls: list[str] = []

    def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None, json_schema=None) -> dict[str, object]:
        self.calls.append(prompt)
        if "generic DRT query DRS" in prompt or "generic DRT/DSPG query frame" in prompt:
            return {
                "query_frame": {
                    "target_anchors": ["Ash Meadow"],
                    "requested_relation": "conservator",
                    "relation_terms": ["conservator"],
                    "constraints": [],
                    "answer_type": "person",
                    "temporal_scope": "",
                    "negated": False,
                    "aggregation": "",
                    "requires_evidence": True,
                },
                "_model_raw": '{"query_frame":{"target_anchors":["Ash Meadow"],"requested_relation":"conservator","relation_terms":["conservator"],"constraints":[],"answer_type":"person","temporal_scope":"","negated":false,"aggregation":"","requires_evidence":true}}',
            }
        if "Verify whether the candidate answer is entailed" in prompt:
            return {
                "verification": {
                    "entailed": not self.incompatible,
                    "answer_type": "person" if not self.incompatible else "unknown",
                    "answer": "Lyra Fen" if not self.incompatible else "unknown",
                    "evidence_span": "Ash Meadow conservator Lyra Fen" if not self.incompatible else "",
                    "reason": "fake grounded verifier",
                },
                "_model_raw": '{"verification":{"entailed":true,"answer_type":"person","answer":"Lyra Fen","evidence_span":"Ash Meadow conservator Lyra Fen","reason":"fake grounded verifier"}}',
            }
        assert "Answer the question only from the provided raw-text evidence" in prompt
        if self.incompatible:
            return {
                "answer": {
                    "sufficient_evidence": True,
                    "answer_type": "url",
                    "answer": "https://example.invalid/ash",
                    "evidence_span": "https://example.invalid/ash",
                },
                "_model_raw": '{"answer":{"sufficient_evidence":true,"answer_type":"url","answer":"https://example.invalid/ash","evidence_span":"https://example.invalid/ash"}}',
            }
        return {
            "answer": {
                "sufficient_evidence": True,
                "answer_type": "person",
                "answer": "Lyra Fen",
                "evidence_span": "Ash Meadow conservator Lyra Fen",
            },
            "_model_raw": '{"answer":{"sufficient_evidence":true,"answer_type":"person","answer":"Lyra Fen","evidence_span":"Ash Meadow conservator Lyra Fen"}}',
        }


def test_person_question_rejects_structural_references(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "entity.raw").write_text(
        "Velora Map has reference VM-220 and link https://example.invalid/velora.\n"
        "The text never states a reviewer or person for Velora Map.\n",
        encoding="utf-8",
    )

    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine.answer("Who reviewed Velora Map?")

    assert answer.text == "unknown"
    assert answer.evidence
    assert answer.evidence[0].rel_path == "notes/entity.raw"
    assert "Velora Map" in answer.evidence[0].text


def test_url_question_returns_url_not_person_or_path(tmp_path: Path) -> None:
    (tmp_path / "mixed").write_text(
        "Iris Vale maintains the river guide at https://example.invalid/river-guide and archive/river-guide.txt.\n",
        encoding="utf-8",
    )

    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine.answer("Where is the river guide link?")

    assert answer.text == "https://example.invalid/river-guide"
    assert answer.answer_type == "url"


def test_organization_question_rejects_identifier_and_url_only_evidence(tmp_path: Path) -> None:
    (tmp_path / "org-note").write_text(
        "The Meridian Grove note lists reference ORG-882 and link https://example.invalid/meridian.\n"
        "No organization name is stated for Meridian Grove.\n",
        encoding="utf-8",
    )

    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine.answer("Which organization supports Meridian Grove?")

    assert answer.text == "unknown"


def test_file_name_metadata_hit_does_not_answer_non_metadata_relation(tmp_path: Path) -> None:
    (tmp_path / "RavenOwnerNote.txt").write_text(
        "This readable note mentions Raven but contains no owner statement.\n",
        encoding="utf-8",
    )

    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine.answer("Who owns RavenOwnerNote?")

    assert answer.text == "unknown"


def test_nested_json_like_raw_text_creates_queryable_key_value_relations(tmp_path: Path) -> None:
    (tmp_path / "raw-object").write_text(
        '{"object":{"owner":"Ila Venn","reference":"ZX-881"},"status":"ready"}\n',
        encoding="utf-8",
    )

    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine.answer("Who is owner for object?").text == "Ila Venn"
    assert engine.answer("Which identifier is reference for object?").text == "ZX-881"
    assert engine.dspg_counts()["relations"] >= 3


def test_json_record_groups_do_not_merge_roots_across_documents(tmp_path: Path) -> None:
    (tmp_path / "moss.raw").write_text(
        '{"bundle":{"name":"Moss Beacon","links":{"manual":"https://manuals.example.test/moss-beacon"}}}\n',
        encoding="utf-8",
    )
    (tmp_path / "lark.raw").write_text(
        '{"bundle":{"name":"Lark Mirror","links":{"warranty":"https://warranty.example.test/lark-mirror"}}}\n',
        encoding="utf-8",
    )

    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine.answer("Where is the warranty for Moss Beacon?").text == "unknown"
    assert engine.answer("Where is the warranty for Lark Mirror?").text == "https://warranty.example.test/lark-mirror"


def test_section_record_groups_bind_fields_across_source_spans(tmp_path: Path) -> None:
    (tmp_path / "entry.txt").write_text(
        "\n".join(
            [
                "Inspection ledger for a synthetic sample.",
                "Item: Solar Reed.",
                "Classification: mineral.",
                "Recorder: Iva Dune.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine.answer("What classification is Solar Reed?").text == "mineral"


def test_low_semantic_noise_does_not_dominate_normal_fact_retrieval(tmp_path: Path) -> None:
    (tmp_path / "facts").mkdir()
    (tmp_path / "noise").mkdir()
    (tmp_path / "facts" / "entry").write_text("LumaSeal owner: Erin Ko.\n", encoding="utf-8")
    (tmp_path / "noise" / "cache.lock").write_text(
        "LumaSeal owner: ASDF-999 https://example.invalid/luma "
        + "xQ9z " * 240,
        encoding="utf-8",
    )

    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine.answer("Who owns LumaSeal?")

    assert answer.text == "Erin Ko"
    assert "cache.lock" not in answer.evidence[0].rel_path


def test_fake_model_evidence_extraction_helper_is_test_only_counted_and_grounded(tmp_path: Path) -> None:
    (tmp_path / "source").write_text(
        "Ash Meadow conservator Lyra Fen\n",
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._use_local_model = True
    engine._model_client = FakeEvidenceModel()
    engine.model_query_trace.enabled = True

    frame = QueryFrame(
        question_text="Who is the conservator for Ash Meadow?",
        answer_type="person",
        answer_variables=("person",),
        target_anchors=("Ash Meadow",),
        requested_relation="conservator",
        relation_terms=("conservator",),
        constraints=(),
    )

    answer = engine._answer_with_model_evidence_extraction(
        "Who is the conservator for Ash Meadow?",
        frame,
        ExpectedAnswer("person"),
    )

    assert answer is not None
    assert answer.text == "Lyra Fen"
    assert answer.answer_type == "person"
    assert answer.evidence and "Ash Meadow conservator Lyra Fen" in answer.evidence[0].text
    assert engine.model_query_trace.evidence_call_count == 1
    assert engine.model_query_trace.evidence_accepted_count == 1
    assert engine.model_query_trace.model_answer_count == 1


def test_model_evidence_answer_attaches_source_metadata_provenance(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "source.raw").write_text(
        "Ash Meadow conservator Lyra Fen\n",
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._use_local_model = True
    engine._model_client = FakeEvidenceModel()
    engine.model_query_trace.enabled = True
    frame = QueryFrame(
        question_text="Who is the conservator for Ash Meadow?",
        answer_type="person",
        answer_variables=("person",),
        target_anchors=("Ash Meadow",),
        requested_relation="conservator",
        relation_terms=("conservator",),
        constraints=(),
    )

    answer = engine._answer_with_model_evidence_extraction(
        "Who is the conservator for Ash Meadow?",
        frame,
        ExpectedAnswer("person"),
    )

    assert answer is not None
    assert answer.text == "Lyra Fen"
    provenance = engine.last_bounded_diagnostics["execution"]["answer_source_provenance"]
    assert provenance[0]["rel_path"] == "notes/source.raw"
    assert provenance[0]["chunk_order"] == 0
    assert provenance[0]["char_start"] == 0
    assert provenance[0]["span_id"]
    assert provenance[0]["chunk_id"]
    assert provenance[0]["document_id"] == provenance[0]["document"]["document_id"]
    assert provenance[0]["document"]["file_name"] == "source.raw"
    assert provenance[0]["document"]["parent_rel_path"] == "notes"
    assert provenance[0]["token_estimate"] > 0


def test_fake_model_evidence_extraction_rejects_incompatible_answer_type(tmp_path: Path) -> None:
    (tmp_path / "source").write_text(
        "Ash Meadow has a pointer at https://example.invalid/ash but no named conservator.\n",
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._use_local_model = True
    engine._model_client = FakeEvidenceModel(incompatible=True)
    engine.model_query_trace.enabled = True
    frame = QueryFrame(
        question_text="Who is the conservator for Ash Meadow?",
        answer_type="person",
        answer_variables=("person",),
        target_anchors=("Ash Meadow",),
        requested_relation="conservator",
        relation_terms=("conservator",),
        constraints=(),
    )

    answer = engine._answer_with_model_evidence_extraction(
        "Who is the conservator for Ash Meadow?",
        frame,
        ExpectedAnswer("person"),
    )

    assert answer is None
    assert engine.model_query_trace.evidence_call_count == 1
    assert engine.model_query_trace.evidence_rejected_count >= 1


def test_count_evidence_extraction_accepts_grounded_multiline_aggregate(tmp_path: Path) -> None:
    class CountEvidenceModel:
        def context_size(self) -> int:
            return 4096

        def complete_json(
            self,
            prompt: str,
            *,
            n_predict: int = 128,
            grammar: str | None = None,
            json_schema: dict[str, object] | None = None,
        ) -> dict[str, object]:
            if "generic DRT query DRS" in prompt or "generic DRT/DSPG query frame" in prompt:
                return {
                    "query_frame": {
                        "target_anchors": [],
                        "answer_variables": [],
                        "requested_relation": "",
                        "relation_terms": [],
                        "constraints": [],
                        "answer_type": "unknown",
                        "temporal_scope": "",
                        "negated": False,
                        "aggregation": "",
                        "requires_evidence": True,
                    },
                    "_model_raw": '{"query_frame":{"target_anchors":[],"answer_variables":[],"requested_relation":"","relation_terms":[],"constraints":[],"answer_type":"unknown","temporal_scope":"","negated":false,"aggregation":"","requires_evidence":true}}',
                }
            if "bounded DRT/DSPG question analysis" in prompt:
                return {
                    "result": {
                        "query_frame": {
                            "target_anchors": [],
                            "answer_variables": [],
                            "requested_relation": "",
                            "relation_terms": [],
                            "constraints": [],
                            "answer_type": "unknown",
                            "temporal_scope": "",
                            "negated": False,
                            "aggregation": "",
                            "requires_evidence": True,
                        },
                        "sufficient_evidence": False,
                        "answer_type": "unknown",
                        "answer": "unknown",
                        "evidence_span": "",
                        "reason": "not needed",
                    },
                    "_model_raw": '{"result":{"sufficient_evidence":false,"answer_type":"unknown","answer":"unknown","evidence_span":"","reason":"not needed"}}',
                }
            assert "Answer the question only from the provided raw-text evidence" in prompt
            return {
                "answer": {
                    "sufficient_evidence": True,
                    "answer_type": "count",
                    "answer": "2",
                    "evidence_span": "item: Blue Reef | status: open\nitem: Glass Pier | status: open",
                },
                "_model_raw": '{"answer":{"sufficient_evidence":true,"answer_type":"count","answer":"2","evidence_span":"item: Blue Reef | status: open\\nitem: Glass Pier | status: open"}}',
            }

    (tmp_path / "items").write_text(
        "Complete item inventory:\n"
        "item: Blue Reef | status: open\n"
        "item: Stone Vale | status: closed\n"
        "item: Glass Pier | status: open\n",
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._use_local_model = True
    engine._model_client = CountEvidenceModel()  # type: ignore[assignment]
    engine.model_query_trace.enabled = True
    frame = QueryFrame(
        question_text="How many items have open status?",
        answer_type="count",
        answer_variables=("count",),
        target_anchors=(),
        requested_relation="open status",
        relation_terms=("open", "status"),
        constraints=("open status",),
        aggregation="count",
    )

    answer = engine._answer_with_model_evidence_extraction(
        "How many items have open status?",
        frame,
        ExpectedAnswer("count"),
    )

    assert answer is not None
    assert answer.text == "2"
    assert answer.answer_type == "count"
    assert answer.evidence


def test_query_drs_bounded_miss_uses_grounded_query_evidence_fallback(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "note.txt").write_text(
        "Cedar Ledger is unrelated to any migration plan.\n",
        encoding="utf-8",
    )
    query_frame = {
        "target_anchors": ["Cedar Ledger", "migration plan"],
        "answer_variables": ["Cedar Ledger migration plan target"],
        "requested_relation": "is target",
        "relation_terms": ["is target"],
        "constraints": ["migration plan"],
        "scope_requirements": [],
        "modality_requirements": [],
        "answer_type": "boolean",
        "temporal_scope": "",
        "negated": False,
        "aggregation": "",
        "requires_evidence": True,
    }

    def fake_query_drs(_question: str, _client: object) -> dict[str, object]:
        return {
            "accepted": True,
            "query_drs": {
                "schema_version": "query-drs-v3",
                "question": "Is Cedar Ledger a migration plan target?",
                "answer_type": "boolean",
                "answer_variables": [
                    {
                        "id": "qv0",
                        "label": "Cedar Ledger migration plan target",
                        "answer_type": "boolean",
                        "evidence_text": "Cedar Ledger migration plan target",
                    }
                ],
                "target_referents": [
                    {"id": "qr0", "label": "Cedar Ledger", "kind": "unknown", "evidence_text": "Cedar Ledger"},
                    {"id": "qr1", "label": "migration plan", "kind": "unknown", "evidence_text": "migration plan"},
                ],
                "requested_conditions": [
                    {
                        "id": "qc0",
                        "predicate": "is target",
                        "box_id": "",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "evidence_text": "Is Cedar Ledger a migration plan target?",
                        "arguments": [
                            {
                                "role": "answer",
                                "target_kind": "answer_variable",
                                "target_id": "qv0",
                                "value": "",
                                "value_type": "boolean",
                                "evidence_text": "Cedar Ledger migration plan target",
                            },
                            {
                                "role": "argument",
                                "target_kind": "referent",
                                "target_id": "qr0",
                                "value": "",
                                "value_type": "unknown",
                                "evidence_text": "Cedar Ledger",
                            },
                        ],
                    }
                ],
                "constraints": ["migration plan"],
                "box_requirements": [],
                "temporal_records": [],
                "temporal_scope": "",
                "aggregation": "",
                "requires_evidence": True,
            },
        }

    def fake_query_evidence(
        _question: str,
        _evidence: list[dict[str, object]],
        _client: object,
        *,
        discourse_records: list[dict[str, object]] | None = None,
        authoritative_query_frame: dict[str, object] | None = None,
        authoritative_answer_type: str = "unknown",
    ) -> dict[str, object]:
        assert authoritative_query_frame is not None
        assert authoritative_query_frame["answer_type"] == "boolean"
        assert authoritative_answer_type == "boolean"
        return {
            "accepted": True,
            "query_frame": query_frame,
            "sufficient_evidence": True,
            "answer_type": "boolean",
            "answer": "No; Cedar Ledger is unrelated to any migration plan.",
            "evidence_span": "Cedar Ledger is unrelated to any migration plan.",
            "reason": "grounded negative evidence",
        }

    monkeypatch.setenv("KMD_MODEL_EVIDENCE_TOOLS", "1")
    monkeypatch.setattr(engine_module, "call_model_query_drs", fake_query_drs)
    monkeypatch.setattr(engine_module, "call_model_query_evidence_answer", fake_query_evidence)
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._use_local_model = True
    engine._model_client = object()  # type: ignore[assignment]
    monkeypatch.setattr(engine, "_verify_with_local_model", lambda *_args, **_kwargs: True)

    answer = engine.answer("Is Cedar Ledger a migration plan target?")

    assert answer.text == "No; Cedar Ledger is unrelated to any migration plan."
    assert answer.reason == "model-verified query-DRS evidence answer"
    assert engine.model_query_trace.evidence_call_count == 1


def test_invalid_model_evidence_answer_is_retried(tmp_path: Path, monkeypatch) -> None:
    class InvalidEvidenceModel:
        def context_size(self) -> int:
            return 4096

        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None, json_schema=None) -> dict[str, object]:
            self.calls += 1
            return {"unexpected": "shape", "_model_raw": '{"unexpected":"shape"}'}

    model = InvalidEvidenceModel()
    monkeypatch.setenv("KMD_EVIDENCE_ANSWER_CACHE_DIR", str(tmp_path / "evidence-cache"))
    evidence = [{"source": "note", "text": "Ash Meadow conservator Lyra Fen"}]

    first = call_model_evidence_answer("Who is the conservator for Ash Meadow?", "person", evidence, model)  # type: ignore[arg-type]
    second = call_model_evidence_answer("Who is the conservator for Ash Meadow?", "person", evidence, model)  # type: ignore[arg-type]

    assert model.calls == 2
    assert first["accepted"] is False
    assert first["cache_context"]["expected_answer_type"] == "person"
    assert "n_predict" not in first["cache_context"]
    assert first["cache_context"]["evidence_count"] == 1
    assert second["accepted"] is False
    assert second["cache_context"]["expected_answer_type"] == "person"



def test_general_local_negative_proposition_patterns(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "judgment.txt").write_text(
        "Final judgment summary.\nThe court found no proof that Widget caused drift.\n",
        encoding="utf-8",
    )
    (tmp_path / "belief.txt").write_text(
        "Kalo Reed believes the lantern should be blue.\nInspection note: the belief is not confirmed as fact.\n",
        encoding="utf-8",
    )
    (tmp_path / "dream.txt").write_text(
        "Dream journal: In the dream, Crane opened the hidden gate. Waking note: no real gate opening is recorded.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    engine.model_query_trace.last_plan = QueryFrame(
        question_text="Was Widget proven to have caused drift?",
        answer_type="boolean",
        answer_variables=(),
        target_anchors=("Widget", "drift"),
        requested_relation="proven caused",
        relation_terms=("proven caused",),
        constraints=(),
    ).as_dict()
    judgment = engine._answer_with_explicit_negative_clause("Was Widget proven to have caused drift?")
    engine.model_query_trace.last_plan = QueryFrame(
        question_text="Is Kalo Reed belief confirmed as fact?",
        answer_type="boolean",
        answer_variables=(),
        target_anchors=("Kalo Reed belief",),
        requested_relation="confirmed",
        relation_terms=("confirmed",),
        constraints=("as fact",),
    ).as_dict()
    belief = engine._answer_with_explicit_negative_clause("Is Kalo Reed belief confirmed as fact?")
    engine.model_query_trace.last_plan = QueryFrame(
        question_text="Did Crane really open the hidden gate?",
        answer_type="boolean",
        answer_variables=(),
        target_anchors=("Crane", "hidden gate"),
        requested_relation="open",
        relation_terms=("open",),
        constraints=("really",),
    ).as_dict()
    dream = engine._answer_with_explicit_negative_clause("Did Crane really open the hidden gate?")
    assert judgment is not None and judgment.text == "No; the final judgment found no proof."
    assert belief is not None and belief.text == "No; the belief is not confirmed as fact."
    assert dream is None


def test_central_answer_guard_rejects_unrelated_no_proof(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "judgment.txt").write_text(
        "Final judgment summary.\nThe court found no proof that Widget caused invoice drift.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    engine.model_query_trace.last_plan = QueryFrame(
        question_text="Was Widget proven to have caused invoice drift?",
        answer_type="boolean",
        answer_variables=(),
        target_anchors=("Widget", "invoice drift"),
        requested_relation="proven caused",
        relation_terms=("proven caused",),
        constraints=(),
    ).as_dict()
    relevant = engine._answer_with_explicit_negative_clause("Was Widget proven to have caused invoice drift?")
    engine.model_query_trace.last_plan = QueryFrame(
        question_text="Was Ardent Mill refund request proven by the judgment?",
        answer_type="boolean",
        answer_variables=(),
        target_anchors=("Ardent Mill", "refund request"),
        requested_relation="proven",
        relation_terms=("proven",),
        constraints=(),
    ).as_dict()
    unrelated = engine._answer_with_explicit_negative_clause("Was Ardent Mill refund request proven by the judgment?")
    assert relevant is not None and relevant.text == "No; the final judgment found no proof."
    assert unrelated is None


def test_definition_cleanup_requires_queried_term(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="What does buenos dias mean?",
        answer_type="content_phrase",
        answer_variables=("meaning",),
        target_anchors=("buenos dias",),
        requested_relation="mean",
        relation_terms=("mean",),
        constraints=(),
    )
    wrong_frame = QueryFrame(
        question_text="What does sola miri tahu mean?",
        answer_type="content_phrase",
        answer_variables=("meaning",),
        target_anchors=("sola miri tahu",),
        requested_relation="mean",
        relation_terms=("mean",),
        constraints=(),
    )
    plural_frame = QueryFrame(
        question_text="What is the plural of tiro?",
        answer_type="content_phrase",
        answer_variables=("plural",),
        target_anchors=("tiro",),
        requested_relation="plural",
        relation_terms=("plural",),
        constraints=(),
    )

    assert engine._cleanup_canonical_answer("buenos dias means good morning", ExpectedAnswer("content_phrase"), frame) == "good morning"
    assert engine._cleanup_canonical_answer("danke means thank you", ExpectedAnswer("content_phrase"), wrong_frame) == "unknown"
    assert engine._cleanup_canonical_answer("is tiros", ExpectedAnswer("content_phrase"), plural_frame) == "tiros"


def test_public_cleanup_expands_single_first_name_when_unambiguous(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "review.txt").write_text("Review line: Omar Kestrel reviewed PR-8042.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)
    evidence = [engine._evidence(next(iter(engine._sentences_by_document["review.txt"].values())), 1.0)]
    answer = Answer("Omar", 0.8, evidence, "unit", "person")

    assert engine._cleanup_public_answer(answer).text == "Omar"



def test_definition_source_extraction_and_where_cleanup(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "defs.txt").write_text("French note: bonsoir means good evening.\nGrammar: plural of tiro is tiros.\n", encoding="utf-8")
    (tmp_path / "place.txt").write_text("The brass lamp is on the red desk.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_definition_source_explanation("What does bonsoir mean?").text == "good evening"
    assert engine._answer_with_definition_source_explanation("What is the plural of tiro?").text == "tiros"
    evidence = [engine._evidence(next(iter(engine._sentences_by_document["place.txt"].values())), 1.0)]
    assert engine._restore_where_preposition("Where is the brass lamp?", "red desk", ExpectedAnswer("content_phrase"), evidence) == "on the red desk"


def test_missing_meaning_cleanup_returns_unknown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="What does mave lora mean?",
        answer_type="content_phrase",
        answer_variables=("meaning",),
        target_anchors=("mave lora",),
        requested_relation="mean",
        relation_terms=("mean",),
        constraints=(),
    )

    assert engine._cleanup_canonical_answer("has no stated translation", ExpectedAnswer("content_phrase"), frame) == "unknown"


def test_post_model_source_pass_corrects_grounded_clause_answer(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "beliefs.txt").write_text(
        "Cora believes routers are social contracts.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._use_local_model = True
    monkeypatch.setattr(
        engine,
        "_answer_with_local_model",
        lambda _question: Answer("routers", 0.5, [], "fake local model answer", "content_phrase"),
    )

    answer = engine.answer("What does Cora believe?")

    assert answer.text == "routers"
    assert answer.reason == "fake local model answer"


def test_post_model_source_pass_uses_generic_labeled_fields_after_model_miss(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "note.txt").write_text(
        "Delta Relay note.\nReview summary: replace the worn seal before launch\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._use_local_model = True
    monkeypatch.setattr(engine, "_answer_with_local_model", lambda _question: None)

    answer = engine.answer("What is the review summary for Delta Relay?")

    assert answer.text == "unknown"
    assert answer.reason == "local model DRT path found no complete grounded answer"


def test_generic_labeled_field_does_not_split_url_scheme(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "note.txt").write_text(
        "The canonical design URL is https://docs.example.test/design-r7.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_generic_labeled_field_source("What is the canonical design URL?") is None
    assert engine.answer("What is the canonical design URL?").text == "https://docs.example.test/design-r7"


def test_final_decision_statement_canonicalizes_to_unknown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine._central_answer_guard(
        "What final decision was made about library hours?",
        "No final decision was made.",
        ExpectedAnswer("content_phrase"),
        None,
        [],
    )
    assert answer == "unknown"



def test_exact_source_field_extraction_prefers_requested_url_label(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "record.txt").write_text(
        "Record: Sample Relay.\nManual URL: https://manuals.example.test/sample-relay\nWarranty URL: https://warranty.example.test/sample-relay\n",
        encoding="utf-8",
    )
    (tmp_path / "cache.tmp").write_text("Sample Relay warranty URL: https://cache.example.test/wrong-sample\n", encoding="utf-8")
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_exact_source_field("Which warranty URL belongs to Sample Relay?").text == "https://warranty.example.test/sample-relay"
    assert engine._answer_with_exact_source_field("Which manual URL belongs to Sample Relay?").text == "https://manuals.example.test/sample-relay"


def test_exact_source_field_extraction_binds_identifier_slot(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "contact.txt").write_text(
        "Oak service note.\nOak Meridian contact person: Jun Sato.\nOak Meridian contact id: CONTACT-8800.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_exact_source_field("What is the contact id for Oak Meridian?").text == "CONTACT-8800"



def test_exact_source_field_uses_deterministic_frame_when_model_frame_is_weak(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "rows.txt").write_text(
        "record: Alpha Thing | runbook: https://runbooks.example.test/alpha\n"
        "record: Slate Orchard | runbook: https://runbooks.example.test/slate-orchard\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)
    engine.model_query_trace.last_plan = {"target_anchors": [], "relation_terms": ["runbook"], "answer_type": "url"}

    assert engine._answer_with_exact_source_field("Where is the runbook for Slate Orchard?").text == "https://runbooks.example.test/slate-orchard"


def test_exact_source_field_extracts_json_label_values(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "raw.txt").write_text(
        '{"bundle":{"name":"Lark Mirror","links":{"manual":"https://manuals.example.test/lark-mirror","warranty":"https://warranty.example.test/lark-mirror"}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_exact_source_field("Where is the manual for Lark Mirror?").text == "https://manuals.example.test/lark-mirror"
    assert engine._answer_with_exact_source_field("Where is the warranty for Lark Mirror?").text == "https://warranty.example.test/lark-mirror"


def test_central_guard_rejects_email_and_hidden_cache_false_positives(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._central_answer_guard("What is the email address for Elan Ruiz?", "The warranty portal for CedarSpan", ExpectedAnswer("content_phrase"), None, []) == "unknown"
    assert engine._central_answer_guard("Which hidden cache URL is the official warranty URL for Mica Relay?", "https://cache.example.test/wrong", ExpectedAnswer("url"), None, []) == "unknown"



def test_exact_source_field_ignores_slot_words_as_targets(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "rows.txt").write_text(
        "record: Juniper Gate | runbook: https://runbooks.example.test/juniper-gate\n"
        "record: Slate Orchard | runbook: https://runbooks.example.test/slate-orchard\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)
    engine.model_query_trace.last_plan = {
        "target_anchors": ["runbook", "Slate Orchard"],
        "relation_terms": ["is", "runbook for slate orchard"],
        "answer_type": "file_path",
    }

    assert engine._answer_with_exact_source_field("Where is the runbook for Slate Orchard?").text == "https://runbooks.example.test/slate-orchard"



def test_exact_source_field_uses_source_path_as_scope(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "data"
    raw_dir.mkdir()
    (raw_dir / "raw_json_like.blob").write_text(
        '{ project: "Not a schema", owner: "Zia Fern", status: "observed", ticket: "TXT-991" }\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_exact_source_field("What ticket appears in the raw JSON-like text?").text == "TXT-991"



def test_source_row_count_aggregation_counts_row_local_filters(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "status.tsv").write_text(
        "item\tstatus\towner\treference\n"
        "Bell Finch\tactive\tOla Nym\tBF-1201\n"
        "Bell Finch\tarchived\tLio Fern\tBF-1200\n"
        "Cedar Finch\tactive\tPax Neri\tCF-2201\n"
        "Dune Finch\tblocked\tRae Sol\tDF-3301\n"
        "Ember Finch\tactive\tUma Korr\tEF-4401\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_source_rows("How many Finch rows have status active?").text == "3"
    assert engine._answer_with_source_rows("How many rows have status blocked?").text == "1"
    assert engine._answer_with_source_rows("How many rows have status archived?").text == "1"


def test_source_row_count_aggregation_owner_state_and_argmax(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "owner_state.tsv").write_text(
        "owner\titem\tstate\tasset\n"
        "Mira Sol\tAster One\topen\tAS-001\n"
        "Mira Sol\tAster Two\topen\tAS-002\n"
        "Mira Sol\tAster Three\tclosed\tAS-003\n"
        "Pax Neri\tBeryl One\topen\tBY-001\n"
        "Pax Neri\tBeryl Two\topen\tBY-002\n"
        "Pax Neri\tBeryl Three\topen\tBY-003\n"
        "Tavi Moss\tCedar One\tpaused\tCD-001\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_source_rows("How many rows have state open?").text == "5"
    assert engine._answer_with_source_rows("How many rows for Mira Sol have state open?").text == "2"
    assert engine._answer_with_source_rows("Which actor has the most open rows?").text == "Pax Neri"
    assert engine._answer_with_source_rows("How many open rows does Pax Neri have?").text == "3"
    assert engine._answer_with_source_rows("How many rows have state paused?").text == "1"


def test_source_row_object_and_pipe_table_counts(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "objects.raw").write_text(
        'group: "Frame"\n'
        '{ name: "Orchid Alpha", owner: "Ila Voss", status: "ready", ids: { asset: "OA-7001" } }\n'
        '{ name: "Orchid Beta", owner: "Niko Rell", status: "paused", ids: { asset: "OB-7002" } }\n'
        '{ name: "Orchid Gamma", owner: "Tessa Noll", status: "ready", ids: { asset: "OG-7003" } }\n',
        encoding="utf-8",
    )
    (tmp_path / "refunds.txt").write_text(
        "Sheet: refunds\n"
        "customer | product | refund_status\n"
        "Blue Dune Retail | SearchSprout | requested\n"
        "Helio Works | MistHarbor | requested\n"
        "Ardent Mill | FlowQuill | alleged\n",
        encoding="utf-8",
    )
    (tmp_path / "contacts.txt").write_text(
        "Crate ID CR-18 belongs to customer Northstar Credit.\n"
        "name | role | email\n"
        "Ari Moss | invoice contact | ari.moss@northstar.example\n"
        "Bex Vale | technical contact | bex.vale@northstar.example\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_source_rows("How many Orchid records are ready?").text == "2"
    assert engine._answer_with_source_rows("Which asset id belongs to the paused Orchid record?").text == "OB-7002"
    assert engine._answer_with_source_rows("How many customers have requested refund status in the refunds sheet?").text == "2"
    assert engine._answer_with_source_rows("How many contacts are listed for Northstar Credit?").text == "2"



def test_temporal_source_records_select_target_local_latest_state(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "states.log").write_text(
        "2026-03-01 status: opened for Delta Well.\n"
        "2026-03-09 status: closed for Delta Well.\n"
        "2026-03-12 status: stable for Ibis Well.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_temporal_source_records("What is the current state of Delta Well?").text == "closed"
    assert engine._answer_with_temporal_source_records("What is the final state of Ibis Well?").text == "stable"


def test_temporal_source_records_preserve_datetime_and_document_final_state(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "incident.log").write_text(
        "2026-02-01 09:00 BUG-100 opened.\n"
        "2026-02-03 16:45 BUG-100 reopened after customer report.\n"
        "Final incident state: closed.\n",
        encoding="utf-8",
    )
    (tmp_path / "camera.log").write_text(
        "Officer Talen recorded the north camera failure at 2026-04-10 07:15.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_temporal_source_records("What is the final state of BUG-100?").text == "closed"
    assert engine._answer_with_temporal_source_records("When did Officer Talen record the north camera failure?").text == "2026-04-10 07:15"



def test_temporal_source_records_do_not_override_non_state_temporal_questions(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "mixed.txt").write_text(
        "2026-04-01 RampCart state: revised.\n"
        "Current state: approved.\n"
        "Vaccine due date: 2026-08-02.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_temporal_source_records("What final decision was made about library hours?") is None
    assert engine._answer_with_temporal_source_records("When did the parade begin according to final verified schedule?") is None
    assert engine._answer_with_temporal_source_records("What was the final cause of the outage?") is None


def test_temporal_source_records_parse_timestamped_record_rows(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "records.txt").write_text(
        "record: Juniper Gate | final state: monitoring | timestamp: 2026-04-11 09:30\n"
        "record: Slate Orchard | final state: closed | timestamp: 2026-04-12 16:00\n"
        "2026-01-10 algae jar B state: cloudy.\n"
        "2026-01-12 algae jar B state: clear.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_temporal_source_records("When was Juniper Gate final state recorded?").text == "2026-04-11 09:30"
    assert engine._answer_with_temporal_source_records("When was Slate Orchard final state recorded?").text == "2026-04-12 16:00"
    assert engine._answer_with_temporal_source_records("What is the current state of algae jar B?").text == "clear"



def test_temporal_source_records_do_not_overfire_on_decision_or_cause_questions(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "mixed.log").write_text(
        "2026-04-01 Cart state: planned.\n"
        "2026-04-04 Cart state: revised.\n"
        "No final decision was made about service hours.\n"
        "Final cause: bad certificate.\n"
        "Final verified schedule: parade began at 13:00.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_temporal_source_records("What final decision was made about service hours?") is None
    assert engine._answer_with_temporal_source_records("What was the final cause of the outage?") is None
    assert engine._answer_with_temporal_source_records("When did the parade begin according to final verified schedule?") is None



def test_source_arithmetic_action_negation_and_specific_code_helpers(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "notes.txt").write_text(
        "Math word problem: 7 apples plus 5 apples equals 12 apples.\n"
        "Counterclaim: Priya argued the ferry mattered more.\n"
        "[Owen] I bought rice and lemons but not blue soap.\n"
        "Specimen code: BIO-22.\n"
        "Hotel confirmation code: HTL-7712.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_arithmetic_source("What does 7 plus 5 equal in the homework note?").text == "12"
    assert engine._answer_with_action_holder_source("Who argued the ferry mattered more?").text == "Priya"
    assert engine._answer_with_negated_action_source("What did Owen not buy?").text == "blue soap"
    assert engine._answer_with_exact_source_field("What is the hotel confirmation code?").text == "HTL-7712"
    assert engine._answer_with_exact_source_field("What specimen code is in the note?").text == "BIO-22"



def test_source_action_holder_and_row_field_protections(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "records.txt").write_text(
        "Debate topic: library hours.\n"
        "Ben: I disagree; families need evening hours.\n"
        "invoice_id|customer|amount|status\n"
        "INV-100|Cedar Theater|410|paid\n"
        "INV-101|River Clinic|125|unpaid\n"
        "Accounting note: Mara closed INV-100.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_action_holder_source("Who disagreed about library hours?").text == "Ben"
    assert engine._answer_with_row_field_source("Which invoice is unpaid?").text == "INV-101"
    assert engine._answer_with_row_field_source("Who closed INV-100?").text == "Mara"



def test_precise_source_content_fields_and_unscoped_roles(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "records.raw").write_text(
        '{"bundle":{"name":"Glass Lamp","owner":"Ila Nore","reviewer":"Oren Pax","notes":[{"claim":"mirror needs velvet pad"},{"claim":"do not use blue solvent"}]}}\n'
        "Mara Vell reviewed the safety addendum for Amber Loom.\n"
        "approver: Eri Noam\n"
        "Silver Nest reference: SVN-5001. SVN-5001 owner: Leda Cross.\n"
        "[12:00] Otho Vale reported that Mist Rail was delayed.\n"
        "[12:10] Otho Vale correction: Mist Rail departed on time.\n"
        "Glossary: \"naur\" means north water.\n"
        "archive path: vault/cinder_atlas_notes.md\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_precise_source_content("Who is reviewer for Glass Lamp?").text == "Oren Pax"
    assert engine._answer_with_precise_source_content("What claim is listed for Glass Lamp about the pad?").text == "mirror needs velvet pad"
    assert engine._answer_with_precise_source_content("What claim is listed for Glass Lamp about solvent?").text == "do not use blue solvent"
    assert engine._answer_with_precise_source_content("Who approved Amber Loom?").text == "unknown"
    assert engine._answer_with_precise_source_content("What did Otho Vale report about Mist Rail?").text == "Mist Rail was delayed"
    assert engine._answer_with_precise_source_content("What was the correction about Mist Rail?").text == "Mist Rail departed on time"
    assert engine._answer_with_precise_source_content("What file path is listed for Glass Lamp?").text == "vault/cinder_atlas_notes.md"
    assert engine._answer_with_definition_source_explanation("What does naur mean?").text == "north water"



def test_reference_role_chain_source_binding(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "references.txt").write_text(
        "Silver Nest reference: SVN-5001. SVN-5001 owner: Leda Cross. Leda Cross badge id: person_leda777000.\n"
        "Silver Nest reviewer: Orin Cale. Orin Cale badge id: person_orin228800.\n"
        "Copper Nest reference: CPN-6001. CPN-6001 owner: Milo Thane. Milo Thane badge id: person_milo993300.\n"
        "Copper Nest reviewer: Anya Reeve. Anya Reeve badge id: person_anya882200.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_reference_role_chain_source("Who owns the reference for Silver Nest?").text == "Leda Cross"
    assert engine._answer_with_reference_role_chain_source("What is the badge id for the reviewer of Silver Nest?").text == "person_orin228800"
    assert engine._answer_with_reference_role_chain_source("Who owns the reference for Copper Nest?").text == "Milo Thane"
    assert engine._answer_with_reference_role_chain_source("What is the badge id for the reviewer of Copper Nest?").text == "person_anya882200"



def test_table_field_and_actor_role_id_source_binding(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "status.tsv").write_text(
        "item\tstatus\towner\treference\turl\n"
        "Cedar Finch\tactive\tPax Neri\tCF-2201\thttps://items.example.test/cedar-finch\n"
        "Dune Finch\tblocked\tRae Sol\tDF-3301\thttps://items.example.test/dune-finch\n",
        encoding="utf-8",
    )
    (tmp_path / "actors.txt").write_text(
        "Dossier: Aurora Loom Safety Note.\n"
        "Author: Nira Sol | actor id: ACT-410\n"
        "Key reviewer: Olan Vex | actor id: ACT-411\n"
        "Reviewer: Pema Rill | actor id: ACT-412\n"
        "No approver is listed for Aurora Loom Safety Note.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_table_field_source("What reference is listed for Cedar Finch?").text == "CF-2201"
    assert engine._answer_with_table_field_source("Where is the URL for Dune Finch?").text == "https://items.example.test/dune-finch"
    assert engine._answer_with_actor_role_ids_source("Which actor id belongs to the key reviewer of Aurora Loom Safety Note?").text == "ACT-411; ACT-412"
    assert engine._answer_with_actor_role_ids_source("Find actor IDs of the author and reviewers of Aurora Loom Safety Note.").text == "ACT-410; ACT-412; ACT-411"
    assert engine._answer_with_actor_role_ids_source("Which actor id belongs to the nonexistent approver of Aurora Loom Safety Note?").text == "unknown"



def test_labeled_attribute_source_binding(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "attrs.txt").write_text(
        "Oak Meridian organization: Sable Harbor Institute.\n"
        "Oak Meridian contact person: Jun Sato.\n"
        "Oak Meridian contact id: CONTACT-8800.\n"
        "Oak Meridian support URL: https://support.example.test/oak-meridian.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_labeled_attribute_source("Which organization is listed for Oak Meridian?").text == "Sable Harbor Institute"
    assert engine._answer_with_labeled_attribute_source("Who is the contact person for Oak Meridian?").text == "Jun Sato"
    assert engine._answer_with_labeled_attribute_source("What is the contact id for Oak Meridian?").text == "CONTACT-8800"
    assert engine._answer_with_labeled_attribute_source("Where is the support URL for Oak Meridian?").text == "https://support.example.test/oak-meridian"



def test_cache_safe_reference_urls_and_organization_labels(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "cache.tmp").write_text(
        "Mica Relay warranty URL: https://cache.example.test/wrong-mica\n"
        "Garnet Bridge owning organization: Fake Cache Org.\n",
        encoding="utf-8",
    )
    (tmp_path / "mica.txt").write_text(
        "Record: Mica Relay.\n"
        "Manual URL: https://manuals.example.test/mica-relay\n"
        "Warranty URL: https://warranty.example.test/mica-relay\n"
        "Archive note: there is no archive URL for Mica Relay.\n",
        encoding="utf-8",
    )
    (tmp_path / "lantern.txt").write_text(
        "Record: North Lantern.\n"
        "Guide URL: https://guides.example.test/north-lantern\n"
        "Runbook URL: https://runbooks.example.test/north-lantern\n",
        encoding="utf-8",
    )
    (tmp_path / "org.txt").write_text(
        "Entity: Garnet Bridge.\nOwning organization: Morrow Slate Guild.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_exact_source_field("Which warranty URL belongs to Mica Relay?").text == "https://warranty.example.test/mica-relay"
    assert engine._answer_with_exact_source_field("Which manual URL belongs to Mica Relay?").text == "https://manuals.example.test/mica-relay"
    assert engine._answer_with_exact_source_field("Which runbook URL belongs to North Lantern?").text == "https://runbooks.example.test/north-lantern"
    assert engine._answer_with_exact_source_field("Which archive URL belongs to Mica Relay?").text == "unknown"
    assert engine._answer_with_labeled_attribute_source("Which organization owns Garnet Bridge?").text == "Morrow Slate Guild"



def test_discourse_clauses_and_structured_object_fields(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "claims.txt").write_text(
        "Dream journal: In the dream, Pearl Engine opened the hidden gate. Waking note: no real gate opening is recorded.\n"
        "Witness note: Runa said, \"the blue latch snapped during loading.\" Later inspection confirmed the latch was intact.\n"
        "Allegation note: Plaintiff Karo alleges the north hinge cracked. Judgment note: the north hinge crack was not proven.\n"
        "Correction: Mist Vale did not ship the red crate; the corrected crate color was amber.\n",
        encoding="utf-8",
    )
    (tmp_path / "objects.raw").write_text(
        '{ name: "Orchid Alpha", owner: "Ila Voss", status: "ready", ids: { asset: "OA-7001", audit: "AUD-3001" }, links: { report: "https://reports.example.test/orchid-alpha" } }\n'
        '{ name: "Orchid Beta", owner: "Niko Rell", status: "paused", ids: { asset: "OB-7002", audit: "AUD-3002" }, links: { report: "https://reports.example.test/orchid-beta" } }\n'
        '{ name: "Orchid Gamma", owner: "Tessa Noll", status: "ready", ids: { asset: "OG-7003", audit: "AUD-3003" }, links: { report: "https://reports.example.test/orchid-gamma" } }\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_discourse_clause_source("Did Pearl Engine really open the hidden gate?").text == "unknown"
    assert engine._answer_with_discourse_clause_source("Was the north hinge crack proven?").text == "unknown"
    assert engine._answer_with_discourse_clause_source("What did Runa say snapped during loading?").text == "blue latch"
    assert engine._answer_with_discourse_clause_source("What was the corrected crate color for Mist Vale?").text == "amber"
    assert engine._answer_with_discourse_clause_source("What did the correction say about Mist Vale shipping the red crate?").text == "Mist Vale did not ship the red crate"
    assert engine._answer_with_structured_object_source("Which asset id belongs to Orchid Beta?").text == "OB-7002"
    assert engine._answer_with_structured_object_source("Which audit id belongs to Orchid Gamma?").text == "AUD-3003"
    assert engine._answer_with_structured_object_source("Which report URL belongs to Orchid Gamma?").text == "https://reports.example.test/orchid-gamma"
    assert engine._answer_with_structured_object_source("Which report URL belongs to the ready Orchid record owned by Tessa Noll?").text == "https://reports.example.test/orchid-gamma"
    assert engine._answer_with_structured_object_source("Which asset id belongs to the paused Orchid record?").text == "OB-7002"



def test_correction_owner_source_strips_ocr_prefix(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "ocr.txt").write_text(
        "wat3r3d maybe //// Clear correction: greenhouse fern owner is Dr. Pella.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_correction_owner_source("Who owns the greenhouse fern according to the OCR correction?").text == "Dr. Pella"



def test_review_approval_source_binding_and_ambiguity(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "reviews.txt").write_text(
        "Omar Kestrel performed the risk review.\n"
        "[Nina] Correction: Omar reviewed PR-8042; Nina authored the design.\n"
        "Morgan Ives: I can test the cold-start patch for BeaconQueue.\n"
        "Morgan Hale: I will review PR-9910 for BeaconQueue docs.\n"
        "A later note says Morgan approved it, but the note does not say which Morgan.\n"
        "Jo Sen: Keep Morgan Ives and Morgan Hale separate until the approval note is clarified.\n"
        "Iris Park accepted responsibility for BUG-7301 after Zed Labs escalated it.\n"
        "Blue Dune Retail reported that SearchSprout returned duplicate invoices.\n"
        "Wednesday: PR-1201 merged by Tomas Vale.\n"
        "  \"owner_sentence\": \"Iris Park accepted responsibility for BUG-7301 after Zed Labs escalated it.\"\n"
        "The canonical design URL is https://docs.luma.example/ledger/escrow-import-r7.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_review_or_approval_source("Who reviewed PR-8042?").text == "Omar Kestrel"
    assert engine._answer_with_review_or_approval_source("Who reviewed PR-9910 for BeaconQueue docs?").text == "Morgan Hale"
    assert engine._answer_with_review_or_approval_source("Which Morgan approved PR-9910?").text == "unknown"
    assert engine._answer_with_review_or_approval_source("Who accepted responsibility for BUG-7301?").text == "Iris Park"
    assert engine._answer_with_review_or_approval_source("Who merged PR-1201?").text == "Tomas Vale"
    assert engine._answer_with_exact_source_field("What is the canonical design URL for the escrow import design?").text == "https://docs.luma.example/ledger/escrow-import-r7"



def test_clause_table_message_source_binding(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "case.txt").write_text(
        "Plaintiff Ardent Mill alleges that FlowQuill caused invoice drift on 2026-03-12.\n"
        "The allegation names support ticket SUP-1207.\n"
        "Blue Dune Retail reported that SearchSprout returned duplicate invoices.\n",
        encoding="utf-8",
    )
    (tmp_path / "measurements.tsv").write_text(
        "Table: bridge sensor readings for DeltaPier\n"
        "measurement date: 1986-07-14\n"
        "source file copied: 2010-05-20\n"
        "sensor\treading_mm\tstatus\n"
        "S-1\t33\tok\n"
        "S-3\t91\tcritical\n",
        encoding="utf-8",
    )
    (tmp_path / "thread.eml").write_text(
        "From: Mira Holt\n"
        "Mira wrote: Rowan fixed parser.cpp in PR-3307.\n"
        "--- forwarded message ---\n"
        "From: Rowan Pike\n"
        "I plan to fix parser.cpp tomorrow, not today.\n"
        "--- end forwarded message ---\n"
        "Mira's top-level note is the asserted statement in this email.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_clause_table_message_source("Who alleged that FlowQuill caused invoice drift?").text == "Ardent Mill"
    assert engine._answer_with_clause_table_message_source("Which customer reported duplicate invoices in SearchSprout?").text == "Blue Dune Retail"
    assert engine._answer_with_clause_table_message_source("What is the measurement date for the DeltaPier sensor readings?").text == "1986-07-14"
    assert engine._answer_with_clause_table_message_source("Which DeltaPier sensor had critical status?").text == "S-3"
    assert engine._answer_with_clause_table_message_source("When was the DeltaPier source file copied?").text == "2010-05-20"
    assert engine._answer_with_clause_table_message_source("According to Mira's top-level note, who fixed parser.cpp?").text == "Rowan"
    assert engine._answer_with_clause_table_message_source("What did the forwarded Rowan message say about fixing parser.cpp?").text == "Rowan planned to fix parser.cpp tomorrow, not today."



def test_approval_by_role_person_source_binding(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "drawing.txt").write_text(
        "Project CopperHollow has construction drawing CH-77.\n"
        "Revision C was approved by engineer Veda Lin on 2025-11-09.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_review_or_approval_source("Who approved CopperHollow revision C?").text == "Veda Lin"



def test_claim_request_and_commit_hash_source_binding(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "claims.txt").write_text(
        "Dana: The outage was caused by gateway overload.\n"
        "Rui: I disagree; the outage was caused by a bad certificate.\n"
        "Project MarlinKind depends on three artifacts: SPEC-22, PR-7788, and https://plans.marlin.example/kind.\n"
        "Reese Vale requested the plan; Noor Bell approved it.\n"
        "line 002: commit b16b00b5 fixed NullMoss crash BUG-5150\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_action_holder_source("Who claimed the outage was caused by gateway overload?").text == "Dana"
    assert engine._answer_with_review_or_approval_source("Who requested the Marlin plan bundle?").text == "Reese Vale"
    assert engine._answer_with_commit_hash_source("Which commit fixed NullMoss crash BUG-5150?").text == "b16b00b5"



def test_owner_label_and_quoted_approver_source_binding(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "mixed.txt").write_text(
        "CedarSpan launch owner is Elan Ruiz.\n"
        '[{product: "RippleDesk", pr: "PR-6402", reviewer: "Iona Gray"}, {product: "RippleDesk", approver: "Gus North"}]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_labeled_attribute_source("Who is the CedarSpan launch owner?").text == "Elan Ruiz"
    assert engine._answer_with_precise_source_content("Who approved RippleDesk?").text == "Gus North"



def test_messy_discussion_belief_and_source_file_bindings(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "notes.txt").write_text(
        "This note mentions no release date for VioletForge and no owner for MoonCrate.\n"
        "This note has no support ticket for MoonCrate.\n"
        "Dana: The outage was caused by gateway overload.\n"
        "Rui: I disagree; the outage was caused by a bad certificate.\n"
        "Sora believes QuillCache stores passwords in plaintext.\n"
        "Blue Dune Retail reported that SearchSprout returned duplicate invoices.\n"
        "PR-8042 implements the importer and touches ledger_importer.rs.\n"
        "2026-02-04 10:15 BUG-4481 closed again after PR-8042 was merged.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_discussion_belief_source("What support ticket is listed for MoonCrate?").text == "unknown"
    assert engine._answer_with_discussion_belief_source("What is the customer ID for Blue Dune Retail?").text == "unknown"
    assert engine._answer_with_discussion_belief_source("Which source file fixed BUG-4481?").text == "ledger_importer.rs"
    assert engine._answer_with_discussion_belief_source("Which file did PR-8042 touch?").text == "ledger_importer.rs"
    assert engine._answer_with_discussion_belief_source("Who disagreed with Dana about the outage cause?").text == "Rui"
    assert engine._answer_with_discussion_belief_source("Who believed QuillCache stored passwords in plaintext?").text == "Sora"



def test_cross_suite_pre_model_priority_and_cache_filtering(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "cache.lock").write_text(
        "Lark Mirror owner: ERROR-0000 random random random.\n",
        encoding="utf-8",
    )
    (tmp_path / "record.txt").write_text(
        "Cinder Atlas dossier.\n"
        "owner: Mara Vell\n"
        "person id: actor_mara884211\n"
        '{"bundle":{"name":"Lark Mirror","owner":"Ila Nore","reviewer":"Oren Pax"}}\n'
        "wat3r3d maybe //// Clear correction: greenhouse fern owner is Dr. Pella.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_labeled_attribute_source("What is the person id for Cinder Atlas?").text == "actor_mara884211"
    assert engine._answer_with_labeled_attribute_source("Who is owner for Lark Mirror?").text == "Ila Nore"
    assert engine._answer_with_correction_owner_source("Who owns the greenhouse fern according to the OCR correction?").text == "Dr. Pella"



def test_missing_organization_owner_guard(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "orgs.txt").write_text(
        "Entity: Ember Reed.\n"
        "No owning organization is stated. A person named Organization Vale appears in a quote, but that is not an organization relation.\n"
        "Entity: Garnet Bridge.\n"
        "Owning organization: Morrow Slate Guild.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_missing_organization_owner_source("Which organization owns Ember Reed?").text == "unknown"
    assert engine._answer_with_labeled_attribute_source("Which organization owns Ember Reed?").text == "unknown"
    assert engine._answer_with_missing_organization_owner_source("Which organization owns Garnet Bridge?") is None
    assert engine._answer_with_labeled_attribute_source("Which organization owns Garnet Bridge?").text == "Morrow Slate Guild"



def test_missing_organization_relation_returns_unknown(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "org.txt").write_text(
        "Entity: Brass Wheel.\n"
        "No owning organization is stated. A person named Organization Vale appears in a quote, but that is not an organization relation.\n"
        "Entity: Garnet Bridge.\n"
        "Owning organization: Morrow Slate Guild.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_labeled_attribute_source("Which organization owns Brass Wheel?").text == "unknown"
    assert engine._answer_with_labeled_attribute_source("Which organization owns Garnet Bridge?").text == "Morrow Slate Guild"


def test_boolean_target_grounding_accepts_cross_sentence_bounded_evidence(tmp_path: Path) -> None:
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="Was FlowQuill proven to have caused invoice drift?",
        answer_type="boolean",
        answer_variables=("causation_status",),
        target_anchors=("FlowQuill", "invoice drift"),
        requested_relation="caused",
        relation_terms=("proven", "caused", "invoice drift"),
        constraints=("proven fact",),
        source="model_query_drs",
    )
    evidence = Evidence(
        rel_path="records.txt",
        text=(
            "The support allegation says FlowQuill caused invoice drift. "
            "The final incident report says the allegation was disproven."
        ),
    )

    assert engine._boolean_answer_has_target_grounding(
        frame,
        "The final incident report says the allegation was disproven.",
        [evidence],
    )


def test_model_evidence_answer_is_not_rejected_by_duplicate_question_anchor_gate(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "homework.txt").write_text("Math word problem: 7 apples plus 5 apples equals 12 apples.\n", encoding="utf-8")
    query_frame = {
        "target_anchors": ["7", "5", "homework note"],
        "answer_variables": ["result"],
        "requested_relation": "plus equal",
        "relation_terms": ["plus", "equal"],
        "constraints": [],
        "scope_requirements": [],
        "modality_requirements": [],
        "answer_type": "content_phrase",
        "temporal_scope": "",
        "negated": False,
        "aggregation": "",
        "requires_evidence": True,
    }

    def fake_query_drs(_question: str, _client: object) -> dict[str, object]:
        return {
            "accepted": True,
            "query_drs": {
                "schema_version": "query-drs-v3",
                "question": "What does 7 plus 5 equal in the homework note?",
                "answer_type": "content_phrase",
                "answer_variables": [{"id": "qv0", "label": "result", "answer_type": "content_phrase", "evidence_text": "What"}],
                "target_referents": [
                    {"id": "qr0", "label": "7", "kind": "unknown", "evidence_text": "7"},
                    {"id": "qr1", "label": "5", "kind": "unknown", "evidence_text": "5"},
                    {"id": "qr2", "label": "homework note", "kind": "unknown", "evidence_text": "homework note"},
                ],
                "requested_conditions": [{
                    "id": "qc0", "predicate": "plus equal", "box_id": "", "polarity": "positive", "modality": "asserted", "temporal_id": "",
                    "evidence_text": "What does 7 plus 5 equal in the homework note?",
                    "arguments": [
                        {"role": "answer", "target_kind": "answer_variable", "target_id": "qv0", "value": "", "value_type": "content_phrase", "evidence_text": "What"},
                        {"role": "argument", "target_kind": "referent", "target_id": "qr0", "value": "", "value_type": "unknown", "evidence_text": "7"},
                        {"role": "argument", "target_kind": "referent", "target_id": "qr1", "value": "", "value_type": "unknown", "evidence_text": "5"},
                        {"role": "argument", "target_kind": "referent", "target_id": "qr2", "value": "", "value_type": "unknown", "evidence_text": "homework note"},
                    ],
                }],
                "constraints": [], "box_requirements": [], "temporal_records": [], "temporal_scope": "", "aggregation": "", "requires_evidence": True,
            },
        }

    def fake_query_evidence(_question: str, _evidence: list[dict[str, object]], _client: object, *, discourse_records=None) -> dict[str, object]:
        return {
            "accepted": True, "query_frame": {**query_frame, "answer_type": "count"}, "sufficient_evidence": True,
            "answer_type": "count", "answer": "12",
            "evidence_span": "Math word problem: 7 apples plus 5 apples equals 12 apples.",
            "reason": "grounded arithmetic result",
        }

    monkeypatch.setenv("KMD_MODEL_EVIDENCE_TOOLS", "1")
    monkeypatch.setattr(engine_module, "call_model_query_drs", fake_query_drs)
    monkeypatch.setattr(engine_module, "call_model_query_evidence_answer", fake_query_evidence)
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._use_local_model = True
    engine._model_client = object()  # type: ignore[assignment]

    answer = engine.answer("What does 7 plus 5 equal in the homework note?")
    assert answer.text == "12"


def test_negative_boolean_verifier_rejects_different_scope_incompatibility(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(
        question_text="Did the silver train carry the kitchen table away?",
        answer_type="boolean",
        answer_variables=("whether carried away",),
        target_anchors=("silver train", "kitchen table"),
        requested_relation="carry away",
        relation_terms=("carry", "away"),
        constraints=(),
        source="model_query_drs",
    )
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        engine_module,
        "call_model_answer_verification",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "entailed": True,
            "answer": "no",
            "evidence_span": "Morning fact: the kitchen table remained in the dining room.",
            "proof_kind": "same_scope_incompatibility",
            "accessibility": "asserted",
            "temporal_alignment": "different_scope",
            "explicit_negation": False,
            "incompatible_condition_span": "Morning fact: the kitchen table remained in the dining room.",
        },
    )
    answer = Answer(
        "no",
        0.9,
        [Evidence("diary.dream", "Morning fact: the kitchen table remained in the dining room.")],
        "model query evidence",
        "boolean",
    )
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is False


def test_negative_boolean_verifier_accepts_grounded_explicit_negation(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(
        question_text="Did the tank wall crack?",
        answer_type="boolean",
        answer_variables=("whether cracked",),
        target_anchors=("tank wall",),
        requested_relation="crack",
        relation_terms=("crack",),
        constraints=(),
        source="model_query_drs",
    )
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    verification_results = iter([
        {
            "accepted": True,
            "entailed": True,
            "answer": "no",
            "evidence_span": "Later inspection found no crack in the tank wall.",
            "proof_kind": "explicit_negation",
            "accessibility": "asserted",
            "temporal_alignment": "unspecified",
            "explicit_negation": True,
            "absence_of_record_only": False,
            "incompatible_condition_span": "",
        },
        {
            "accepted": True,
            "entailed": True,
            "answer": "no",
            "evidence_span": "Later inspection found no crack in the tank wall.",
            "proof_kind": "explicit_negation",
            "accessibility": "asserted",
            "temporal_alignment": "unspecified",
            "explicit_negation": True,
            "absence_of_record_only": False,
            "incompatible_condition_span": "",
        },
    ])
    monkeypatch.setattr(engine_module, "call_model_answer_verification", lambda *_args, **_kwargs: next(verification_results))
    answer = Answer(
        "no",
        0.9,
        [Evidence("inspection.txt", "Later inspection found no crack in the tank wall.")],
        "model query evidence",
        "boolean",
    )
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is True


def test_negative_boolean_verifier_rejects_even_model_claimed_same_scope_incompatibility(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(question_text="Did X happen?", answer_type="boolean", answer_variables=("whether",), target_anchors=("X",), requested_relation="happen", relation_terms=("happen",), constraints=(), source="model_query_drs")
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_module, "call_model_answer_verification", lambda *_args, **_kwargs: {"accepted": True, "entailed": True, "answer": "no", "evidence_span": "X remains present.", "proof_kind": "same_scope_incompatibility", "accessibility": "asserted", "temporal_alignment": "same_scope", "explicit_negation": False, "incompatible_condition_span": "X remains present."})
    answer = Answer("no", 0.9, [Evidence("x.txt", "X remains present.")], "model", "boolean")
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is False


def test_shortest_model_answer_preserves_explicit_list_aggregation(tmp_path) -> None:
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="Find actor IDs of the author and reviewers.",
        answer_type="identifier",
        answer_variables=("actor IDs",),
        target_anchors=("Safety Note",),
        requested_relation="author reviewers",
        relation_terms=("author", "reviewers"),
        constraints=(),
        aggregation="list",
        source="model_query_drs",
    )
    assert engine._shortest_model_answer_value("ACT-410; ACT-411; ACT-412", "identifier", frame) == "ACT-410; ACT-411; ACT-412"


def test_negative_boolean_verifier_accepts_grounded_explicit_exclusion(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(question_text="Does QuillCache store plaintext passwords?", answer_type="boolean", answer_variables=("whether",), target_anchors=("QuillCache",), requested_relation="stores plaintext passwords", relation_terms=("stores", "plaintext passwords"), constraints=(), source="model_query_drs")
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_module, "call_model_answer_verification", lambda *_args, **_kwargs: {"accepted": True, "entailed": True, "answer": "no", "evidence_span": "Audit result: QuillCache stores only salted password hashes.", "proof_kind": "explicit_exclusion", "accessibility": "asserted", "temporal_alignment": "unspecified", "explicit_negation": False, "incompatible_condition_span": ""})
    answer = Answer("no", 0.9, [Evidence("audit.txt", "Audit result: QuillCache stores only salted password hashes.")], "model", "boolean")
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is True


def test_negative_boolean_verifier_rejects_absence_of_record_proof(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(question_text="Did Pearl Engine really open the hidden gate?", answer_type="boolean", answer_variables=("whether",), target_anchors=("Pearl Engine", "hidden gate"), requested_relation="open", relation_terms=("open",), constraints=(), source="model_query_drs")
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_module, "call_model_answer_verification", lambda *_args, **_kwargs: {"accepted": True, "entailed": True, "answer": "no", "evidence_span": "Waking note: no real gate opening is recorded.", "proof_kind": "explicit_negation", "accessibility": "asserted", "temporal_alignment": "unspecified", "explicit_negation": True, "absence_of_record_only": True, "incompatible_condition_span": ""})
    answer = Answer("no", 0.9, [Evidence("dream.txt", "Waking note: no real gate opening is recorded.")], "model", "boolean")
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is False


def test_negative_boolean_verifier_rejects_absence_only_even_when_model_mislabels_it(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(question_text="Did Pearl Engine really open the hidden gate?", answer_type="boolean", answer_variables=("whether",), target_anchors=("Pearl Engine", "hidden gate"), requested_relation="open", relation_terms=("open",), constraints=(), source="model_query_drs")
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_module, "call_model_answer_verification", lambda *_args, **_kwargs: {"accepted": True, "entailed": True, "answer": "no", "evidence_span": "Waking note: no real gate opening is recorded.", "proof_kind": "explicit_negation", "accessibility": "asserted", "temporal_alignment": "same_scope", "explicit_negation": True, "absence_of_record_only": False, "incompatible_condition_span": ""})
    answer = Answer("no", 0.9, [Evidence("dream.txt", "Waking note: no real gate opening is recorded.")], "model", "boolean")
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is False
    assert engine.model_query_trace.verifier_call_count == 1


def test_negative_boolean_verifier_rejects_not_proven_as_absence_of_proof(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(question_text="Was the north hinge crack proven?", answer_type="boolean", answer_variables=("proven",), target_anchors=("north hinge crack",), requested_relation="proven", relation_terms=("proven",), constraints=(), source="model_query_drs")
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_module, "call_model_answer_verification", lambda *_args, **_kwargs: {"accepted": True, "entailed": True, "answer": "no", "evidence_span": "Judgment note: the north hinge crack was not proven.", "proof_kind": "explicit_negation", "accessibility": "asserted", "temporal_alignment": "unspecified", "explicit_negation": True, "absence_of_record_only": False, "incompatible_condition_span": ""})
    answer = Answer("no", 0.9, [Evidence("scoped_claims.txt", "Judgment note: the north hinge crack was not proven.")], "model", "boolean")
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is False
    assert engine.model_query_trace.verifier_call_count == 1


def test_absence_classifier_recognizes_not_proved_and_not_proven_but_not_direct_class_negation() -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    assert engine._evidence_is_absence_of_record_only("The claim was not proved.") is True
    assert engine._evidence_is_absence_of_record_only("The claim was not proven.") is True
    assert engine._evidence_is_absence_of_record_only("This is not an engineering record.") is False


def test_negative_boolean_verifier_accepts_negated_relation_object(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine.model_query_trace.last_plan = {
        "question_text": "Did later inspection find a crack in the blue pump?",
        "answer_type": "boolean",
        "answer_variables": ["Did"],
        "target_anchors": ["crack", "blue pump"],
        "requested_relation": "find",
        "relation_terms": ["found", "find"],
        "constraints": ["later inspection"],
        "source": "model_query_drs",
    }
    frame = QueryFrame(question_text="Did later inspection find a crack in the blue pump?", answer_type="boolean", answer_variables=("boolean",), target_anchors=(), requested_relation="", relation_terms=("later", "inspection", "find", "crack", "blue", "pump"), constraints=(), source="deterministic")
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_module, "call_model_answer_verification", lambda *_args, **_kwargs: {"accepted": True, "entailed": True, "answer": "no", "evidence_span": "Later inspection found no crack in the blue pump.", "proof_kind": "explicit_negation", "accessibility": "asserted", "temporal_alignment": "same_scope", "explicit_negation": True, "absence_of_record_only": False, "incompatible_condition_span": ""})
    answer = Answer("no", 0.9, [Evidence("conversations.log", "Later inspection found no crack in the blue pump.")], "model", "boolean")
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is True


def test_negative_boolean_verifier_rejects_incompatible_state_without_direct_negation(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(question_text="Was the latch confirmed broken?", answer_type="boolean", answer_variables=("Was",), target_anchors=("latch",), requested_relation="confirmed broken", relation_terms=("confirmed", "broken"), constraints=(), source="model_query_drs")
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_module, "call_model_answer_verification", lambda *_args, **_kwargs: {"accepted": True, "entailed": True, "answer": "no", "evidence_span": "Later inspection confirmed the latch was intact.", "proof_kind": "explicit_negation", "accessibility": "asserted", "temporal_alignment": "same_scope", "explicit_negation": True, "absence_of_record_only": False, "incompatible_condition_span": ""})
    answer = Answer("no", 0.9, [Evidence("scoped_claims.txt", "Later inspection confirmed the latch was intact.")], "model", "boolean")
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is False


def test_negative_boolean_verifier_rejects_no_decision_for_confirmed_plan(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine.model_query_trace.last_plan = {
        "question_text": "Was the river reroute confirmed as a plan?",
        "answer_type": "boolean",
        "answer_variables": ["Was"],
        "target_anchors": ["the river reroute"],
        "requested_relation": "confirmed",
        "relation_terms": ["confirmed", "confirm"],
        "constraints": ["as a plan"],
        "source": "model_query_drs",
    }
    frame = QueryFrame(question_text="Was the river reroute confirmed as a plan?", answer_type="boolean", answer_variables=("boolean",), target_anchors=(), requested_relation="", relation_terms=("river", "reroute", "confirm", "confirmed", "confirme", "as", "plan"), constraints=("river", "reroute", "confirm", "confirmed", "confirme", "as", "plan"), source="deterministic")
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_module, "call_model_answer_verification", lambda *_args, **_kwargs: {"accepted": True, "entailed": True, "answer": "no", "evidence_span": "Confirmed plan: no reroute decision was made.", "proof_kind": "explicit_negation", "accessibility": "asserted", "temporal_alignment": "same_scope", "explicit_negation": True, "absence_of_record_only": False, "incompatible_condition_span": ""})
    answer = Answer("no", 0.9, [Evidence("scoped_claims.txt", "Confirmed plan: no reroute decision was made.")], "model", "boolean")
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is False


def test_direct_negation_relation_match_distinguishes_four_regression_cases() -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    q031 = QueryFrame(question_text="Should the drawing be treated as an engineering record?", answer_type="boolean", answer_variables=("whether",), target_anchors=("drawing",), requested_relation="is", relation_terms=("engineering record",), constraints=(), source="model_query_drs")
    hrq108 = QueryFrame(question_text="Was the north hinge crack proven?", answer_type="boolean", answer_variables=("proven",), target_anchors=("north hinge crack",), requested_relation="proven", relation_terms=("proven",), constraints=(), source="model_query_drs")
    hrq059 = QueryFrame(question_text="Did later inspection find a crack in the blue pump?", answer_type="boolean", answer_variables=("Did",), target_anchors=("crack", "blue pump"), requested_relation="find", relation_terms=("found", "find"), constraints=("later inspection",), source="model_query_drs")
    hrq110 = QueryFrame(question_text="Was the latch confirmed broken?", answer_type="boolean", answer_variables=("Was",), target_anchors=("latch",), requested_relation="confirmed broken", relation_terms=("confirmed", "broken"), constraints=(), source="model_query_drs")
    hrq112 = QueryFrame(question_text="Was the river reroute confirmed as a plan?", answer_type="boolean", answer_variables=("Was",), target_anchors=("river reroute",), requested_relation="confirmed as a plan", relation_terms=("confirmed", "plan"), constraints=("as a plan",), source="model_query_drs")
    assert engine._evidence_directly_negates_requested_relation(q031, "This is fiction homework, not an engineering record.") is True
    assert engine._evidence_directly_negates_requested_relation(hrq108, "The north hinge crack was not proven.") is False
    assert engine._evidence_directly_negates_requested_relation(hrq059, "Later inspection found no crack in the blue pump.") is True
    assert engine._evidence_directly_negates_requested_relation(hrq110, "Later inspection confirmed the latch was intact.") is False
    assert engine._evidence_directly_negates_requested_relation(hrq112, "Confirmed plan: no reroute decision was made.") is False


def test_negative_boolean_verifier_accepts_not_an_engineering_record(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(question_text="Should the drawing be treated as an engineering record?", answer_type="boolean", answer_variables=("whether",), target_anchors=("drawing",), requested_relation="is", relation_terms=("engineering record",), constraints=(), source="model_query_drs")
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_module, "call_model_answer_verification", lambda *_args, **_kwargs: {"accepted": True, "entailed": True, "answer": "no", "evidence_span": "Teacher note: this is fiction homework, not an engineering record.", "proof_kind": "explicit_negation", "accessibility": "asserted", "temporal_alignment": "unspecified", "explicit_negation": True, "absence_of_record_only": False, "incompatible_condition_span": ""})
    answer = Answer("no", 0.9, [Evidence("story.txt", "Teacher note: this is fiction homework, not an engineering record.")], "model", "boolean")
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is True
    assert engine.model_query_trace.verifier_call_count == 1


def test_cleanup_arithmetic_count_removes_explanatory_unit(tmp_path):
    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    answer = Answer("12 apples", 0.9, [], "model", "count")
    cleaned = engine._cleanup_public_answer(answer, question="What does 7 plus 5 equal?")
    assert cleaned.text == "12"


def test_cleanup_non_arithmetic_count_keeps_unit_phrase(tmp_path):
    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    answer = Answer("12 apples", 0.9, [], "model", "count")
    cleaned = engine._cleanup_public_answer(answer, question="How many apples were stored?")
    assert cleaned.text == "12 apples"


def test_explicit_negative_clause_recovers_exhaustive_only_exclusion() -> None:
    from knowmoredirt.engine import KnowMoreDiRTEngine
    from knowmoredirt.model_planner import ModelQueryTrace
    from knowmoredirt.models import Evidence, Sentence

    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine.model_query_trace.last_plan = {
        "answer_type": "boolean",
        "target_anchors": ["QuillCache", "plaintext passwords"],
        "requested_relation": "stores",
        "relation_terms": ["stores", "store"],
        "constraints": ["audit"],
        "answer_variables": ["Does"],
    }
    sentence = Sentence("s", "d", "audit.txt", "The audit says QuillCache stores only salted password hashes.", 0, 0, 62)
    evidence = Evidence("audit.txt", sentence.text, span_id="s")
    engine._search = lambda *args, **kwargs: [(sentence, 1.0)]
    engine._evidence = lambda *_args, **_kwargs: evidence
    engine._evidence_window_text = lambda *_args, **_kwargs: sentence.text
    engine._central_answer_guard = lambda _q, text, _expected, _frame, _evidence: text
    answer = engine._answer_with_explicit_negative_clause(
        "Does the audit say QuillCache stores plaintext passwords?"
    )
    assert answer is not None
    assert answer.text == "No"
    assert answer.reason == "explicit exhaustive exclusion"


def test_row_url_field_binding_preserves_scheme_and_requires_same_target_row(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "records.log").write_text(
        "owner=Mara Chen | component=retry scheduler | canonical_pr=https://github.com/example/BeaconForce/pull/2780\n"
        "owner=Ilya Stone | component=oauth callback | repair_pr=https://github.com/example/BeaconForce/pull/2814\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    rows = [row for row, _evidence in engine._source_row_records() if row.get("component") == "retry scheduler"]
    assert len(rows) == 1
    assert rows[0]["canonical_pr"] == "https://github.com/example/BeaconForce/pull/2780"

    answer = engine._answer_with_row_field_source(
        "What is the tracking PR URL for the BeaconForce retry scheduler?"
    )
    assert answer is not None
    assert answer.text == "https://github.com/example/BeaconForce/pull/2780"
    assert answer.answer_type == "url"
    assert answer.reason == "source-row same-record url field"


def test_negative_boolean_verifier_rejects_not_proven_for_underlying_event_query(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(
        question_text="Did the north hinge crack happen?",
        answer_type="boolean",
        answer_variables=("whether",),
        target_anchors=("north hinge crack",),
        requested_relation="happen",
        relation_terms=("happen",),
        constraints=(),
        source="model_query_drs",
    )
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_module, "call_model_answer_verification", lambda *_args, **_kwargs: {
        "accepted": True,
        "entailed": True,
        "answer": "no",
        "evidence_span": "Judgment note: the north hinge crack was not proven.",
        "proof_kind": "explicit_negation",
        "accessibility": "asserted",
        "temporal_alignment": "unspecified",
        "explicit_negation": True,
        "absence_of_record_only": False,
        "incompatible_condition_span": "",
    })
    answer = Answer("no", 0.9, [Evidence("judgment.txt", "Judgment note: the north hinge crack was not proven.")], "model", "boolean")
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is False


def test_meta_status_direct_negation_distinguishes_confirmation_proof_and_finalization() -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    confirmed = QueryFrame(
        question_text="Is Kalo Reed belief confirmed as fact?",
        answer_type="boolean",
        answer_variables=("Is",),
        target_anchors=("Kalo Reed belief",),
        requested_relation="confirmed",
        relation_terms=("confirmed", "confirm"),
        constraints=("as fact",),
        source="model_query_drs",
    )
    proven = QueryFrame(
        question_text="Was FlowQuill proven to have caused invoice drift?",
        answer_type="boolean",
        answer_variables=("Was",),
        target_anchors=("FlowQuill", "invoice drift"),
        requested_relation="proven caused",
        relation_terms=("proven", "caused"),
        constraints=(),
        source="model_query_drs",
    )
    finalized = QueryFrame(
        question_text="Was the River Dial archive decision finalized?",
        answer_type="boolean",
        answer_variables=("finalized",),
        target_anchors=("River Dial archive decision",),
        requested_relation="finalized",
        relation_terms=("finalized",),
        constraints=(),
        source="model_query_drs",
    )
    event = QueryFrame(
        question_text="Did FlowQuill cause invoice drift?",
        answer_type="boolean",
        answer_variables=("Did",),
        target_anchors=("FlowQuill", "invoice drift"),
        requested_relation="cause",
        relation_terms=("cause",),
        constraints=(),
        source="model_query_drs",
    )
    confirmed_span = "Inspection note: the lantern color remains green; the belief is not confirmed as fact."
    proof_span = "Final judgment summary. The court found no proof that FlowQuill caused invoice drift."
    final_span = "River Dial note: discussion only, no final decision about archive."
    assert engine._evidence_directly_negates_requested_relation(confirmed, confirmed_span) is True
    assert engine._evidence_is_absence_of_record_only(confirmed_span, confirmed) is False
    assert engine._evidence_directly_negates_requested_relation(proven, proof_span) is True
    assert engine._evidence_is_absence_of_record_only(proof_span, proven) is False
    bare_not_proven = "Judgment note: the north hinge crack was not proven."
    assert engine._evidence_directly_negates_requested_relation(proven, bare_not_proven) is False
    assert engine._evidence_is_absence_of_record_only(bare_not_proven, proven) is True
    assert engine._evidence_directly_negates_requested_relation(finalized, final_span) is True
    assert engine._evidence_is_absence_of_record_only(final_span, finalized) is False
    assert engine._evidence_directly_negates_requested_relation(event, proof_span) is False
    assert engine._evidence_is_absence_of_record_only(proof_span, event) is True


def test_negative_boolean_verifier_accepts_authoritative_no_proof_for_proven_status(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(
        question_text="Was FlowQuill proven to have caused invoice drift?",
        answer_type="boolean",
        answer_variables=("Was",),
        target_anchors=("FlowQuill", "invoice drift"),
        requested_relation="proven caused",
        relation_terms=("proven", "caused"),
        constraints=(),
        source="model_query_drs",
    )
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_module, "call_model_answer_verification", lambda *_args, **_kwargs: {
        "accepted": True,
        "entailed": True,
        "answer": "no",
        "evidence_span": "The court found no proof that FlowQuill caused invoice drift.",
        "proof_kind": "explicit_negation",
        "accessibility": "asserted",
        "temporal_alignment": "unspecified",
        "explicit_negation": True,
        "absence_of_record_only": False,
        "incompatible_condition_span": "",
    })
    answer = Answer("no", 0.9, [Evidence("judgment.final", "The court found no proof that FlowQuill caused invoice drift.")], "model", "boolean")
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is True


def test_atomic_absence_model_answer_requires_requested_relation_grounding() -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._evidence_window_text = lambda item, **_kwargs: (
        "A sky-bicycle rule applies only inside the sleep story. " + item.text
    )
    frame = QueryFrame(
        question_text="What rule applies in waking life?",
        answer_type="content_phrase",
        answer_variables=("rule",),
        target_anchors=("waking life",),
        requested_relation="applies",
        relation_terms=("applies",),
        constraints=(),
        source="model_query_drs",
    )
    closing = Evidence(
        "sleep.txt",
        "None of it happened in waking life.",
        span_id="span-1",
    )
    assert engine._absence_like_model_answer_has_relation_grounding(
        frame,
        "none",
        closing.text,
        [closing],
    ) is False

    explicit = Evidence("record.txt", "owner: none", span_id="span-2")
    owner_frame = QueryFrame(
        question_text="What owner is listed?",
        answer_type="content_phrase",
        answer_variables=("owner",),
        target_anchors=(),
        requested_relation="owner",
        relation_terms=("owner",),
        constraints=(),
        source="model_query_drs",
    )
    assert engine._absence_like_model_answer_has_relation_grounding(
        owner_frame,
        "none",
        explicit.text,
        [explicit],
    ) is True
    assert engine._absence_like_model_answer_has_relation_grounding(
        frame,
        "three glass bells",
        closing.text,
        [closing],
    ) is True


def test_production_query_evidence_rejects_atomic_none_without_relation_grounding(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine.last_bounded_diagnostics = {}
    engine.documents = []
    engine._documents_by_rel_path = {}
    engine._sentences_by_document = {}
    engine._sentences_by_location = {}
    engine._context_size = 65536
    engine.sentences = []
    engine._search = lambda *args, **kwargs: []
    engine._evidence_window_text = lambda item, **_kwargs: item.text
    engine._matching_evidence = lambda evidence, evidence_span, proposed: evidence
    engine._attach_model_answer_provenance = lambda answer: None
    engine._record_model_result = lambda model: None
    evidence = [Evidence("sleep.txt", "None of it happened in waking life.", span_id="s1")]
    engine._focused_evidence_windows = lambda *args, **kwargs: evidence
    engine._discourse_payload_for_evidence = lambda *args, **kwargs: []
    engine._fallback_model_client = lambda: object()
    engine._evidence_payload = lambda evidence, **kwargs: [
        {"source": "sleep.txt", "text": evidence[0].text, "span_id": "s1"}
    ]
    frame = QueryFrame(
        question_text="What sky-bicycle rule applies in waking life?",
        answer_type="content_phrase",
        answer_variables=("sky-bicycle rule",),
        target_anchors=("waking life",),
        requested_relation="applies",
        relation_terms=("applies", "sky-bicycle rule"),
        constraints=(),
        source="model_query_drs",
    )
    engine.model_query_trace.last_plan = frame.as_dict()
    monkeypatch.setattr(engine_module, "call_model_query_evidence_answer", lambda *_args, **_kwargs: {
        "accepted": True,
        "sufficient_evidence": True,
        "answer": "none",
        "answer_type": "content_phrase",
        "evidence_span": "None of it happened in waking life.",
        "prompt_hash": "p",
        "output_hash": "o",
    })
    result = engine._answer_with_model_query_evidence(
        frame.question_text,
        ExpectedAnswer("content_phrase"),
    )
    assert result is not None
    assert result.text == "unknown"
    assert "lacked relation grounding" in result.reason


def test_explicit_negative_clause_handles_relation_verb_no_object() -> None:
    from knowmoredirt.models import Sentence
    engine = object.__new__(KnowMoreDiRTEngine)
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine.model_query_trace.last_plan = {
        "answer_type": "boolean",
        "target_anchors": ["tank wall"],
        "requested_relation": "found",
        "relation_terms": ["find", "found"],
        "constraints": ["crack"],
        "answer_variables": ["Was a crack found"],
    }
    sentence = Sentence(
        "s1",
        "d1",
        "law.note",
        "Later inspection found no crack in the tank wall.",
        0,
        0,
        50,
    )
    engine._search = lambda *args, **kwargs: [(sentence, 1.0)]
    engine._evidence = lambda sent, score: Evidence("law.note", sent.text, score=score, span_id="s1")
    engine._evidence_window_text = lambda item, **kwargs: item.text
    engine._central_answer_guard = lambda _q, text, _expected, _frame, _evidence: text
    answer = engine._answer_with_explicit_negative_clause("Was a crack found in the tank wall?")
    assert answer is not None
    assert answer.text == "No"
    assert answer.answer_type == "boolean"


def test_explicit_negative_clause_does_not_turn_no_record_into_event_negation() -> None:
    from knowmoredirt.models import Sentence
    engine = object.__new__(KnowMoreDiRTEngine)
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine.model_query_trace.last_plan = {
        "answer_type": "boolean",
        "target_anchors": ["tank wall"],
        "requested_relation": "found",
        "relation_terms": ["find", "found"],
        "constraints": ["crack"],
        "answer_variables": ["Was a crack found"],
    }
    sentence = Sentence(
        "s1",
        "d1",
        "law.note",
        "Later inspection found no record of a crack in the tank wall.",
        0,
        0,
        62,
    )
    engine._search = lambda *args, **kwargs: [(sentence, 1.0)]
    engine._evidence = lambda sent, score: Evidence("law.note", sent.text, score=score, span_id="s1")
    engine._evidence_window_text = lambda item, **kwargs: item.text
    engine._central_answer_guard = lambda _q, text, _expected, _frame, _evidence: text
    assert engine._answer_with_explicit_negative_clause("Was a crack found in the tank wall?") is None


def test_definition_completion_expands_strict_prefix_from_explicit_source(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    grounded = Answer(
        "good morning",
        0.82,
        [Evidence("languages/spanish_lesson.txt", "Spanish phrase: buenos dias means good morning.")],
        "general definition source extraction",
        "content_phrase",
    )
    monkeypatch.setattr(engine, "_answer_with_definition_source_explanation", lambda _q: grounded)
    answer = Answer(
        "good",
        0.9,
        [Evidence("languages/spanish_lesson.txt", "Spanish phrase: buenos dias means good morning.")],
        "bounded dspg",
        "content_phrase",
    )
    completed = engine._complete_definition_answer_from_source("What does buenos dias mean?", answer)
    assert completed.text == "good morning"
    assert "completed from explicit definition source" in completed.reason


def test_definition_completion_does_not_replace_nonprefix_answer(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    grounded = Answer(
        "good morning",
        0.82,
        [Evidence("languages/spanish_lesson.txt", "Spanish phrase: buenos dias means good morning.")],
        "source",
        "content_phrase",
    )
    monkeypatch.setattr(engine, "_answer_with_definition_source_explanation", lambda _q: grounded)
    answer = Answer("hello", 0.9, [], "model", "content_phrase")
    assert engine._complete_definition_answer_from_source("What does buenos dias mean?", answer).text == "hello"


def test_negative_boolean_verifier_rejects_hallucinated_explicit_exclusion(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine._sentences_by_document = {}
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    frame = QueryFrame(
        question_text="Did the silver train really carry the kitchen table away?",
        answer_type="boolean",
        answer_variables=("whether",),
        target_anchors=("silver train", "kitchen table"),
        requested_relation="carry",
        relation_terms=("carry",),
        constraints=(),
        source="model_query_drs",
    )
    monkeypatch.setattr(engine, "_canonicalize_model_answer_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        engine_module,
        "call_model_answer_verification",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "entailed": True,
            "answer": "no",
            "evidence_span": "Morning fact: the kitchen table remained in the dining room.",
            "proof_kind": "explicit_exclusion",
            "accessibility": "asserted",
            "temporal_alignment": "same_scope",
            "explicit_negation": False,
            "absence_of_record_only": False,
            "incompatible_condition_span": "",
        },
    )
    answer = Answer(
        "no",
        0.9,
        [Evidence("dream.txt", "Morning fact: the kitchen table remained in the dining room.")],
        "model",
        "boolean",
    )
    assert engine._verify_with_local_model(frame.question_text, frame, answer, ExpectedAnswer("boolean")) is False


def test_explicit_exclusion_guard_requires_source_only_construction() -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    valid = QueryFrame(
        question_text="Does QuillCache store plaintext passwords?",
        answer_type="boolean",
        answer_variables=("whether",),
        target_anchors=("QuillCache",),
        requested_relation="stores plaintext passwords",
        relation_terms=("stores", "plaintext passwords"),
        constraints=(),
        source="model_query_drs",
    )
    assert engine._evidence_directly_excludes_requested_relation(
        valid,
        "Audit result: QuillCache stores only salted password hashes.",
    ) is True
    invalid = QueryFrame(
        question_text="Did the silver train really carry the kitchen table away?",
        answer_type="boolean",
        answer_variables=("whether",),
        target_anchors=("silver train", "kitchen table"),
        requested_relation="carry",
        relation_terms=("carry",),
        constraints=(),
        source="model_query_drs",
    )
    assert engine._evidence_directly_excludes_requested_relation(
        invalid,
        "Morning fact: the kitchen table remained in the dining room.",
    ) is False


def test_when_question_binds_generic_leading_timestamp_to_trailing_event(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "schedule.txt").write_text(
        "Calendar fragment.\n"
        "2026-06-01 09:00 dentist appointment.\n"
        "2026-06-02 18:00 piano recital.\n"
        "2026-06-03 20:00 unrelated dinner.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)
    try:
        assert engine._temporal_question_should_bind("When is the piano recital?") is True
        rows = engine._temporal_line_records()
        piano = [row for row, _ev in rows if str(row.get("target", "")).strip().lower() == "piano recital"]
        assert len(piano) == 1
        assert piano[0]["timestamp"] == "2026-06-02 18:00"
        answer = engine._answer_with_temporal_source_records("When is the piano recital?")
        assert answer is not None
        assert answer.text == "2026-06-02 18:00"
        assert answer.answer_type == "date_time"
    finally:
        engine.close()


def test_scoped_contact_count_does_not_count_unrelated_contact_tables(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "northstar.txt").write_text(
        "Crate ID CR-18 belongs to customer Northstar Credit.\n"
        "A table below lists contacts:\n"
        "Ari Moss | invoice contact | ari.moss@northstar.example\n"
        "Bex Vale | technical contact | bex.vale@northstar.example\n",
        encoding="utf-8",
    )
    (tmp_path / "other.txt").write_text(
        "Customer Other Company.\n"
        "A table below lists contacts:\n"
        "Cia North | invoice contact | cia@other.example\n"
        "Dan West | technical contact | dan@other.example\n"
        "Eli South | legal contact | eli@other.example\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)
    try:
        answer = engine._answer_with_source_rows("How many contacts are listed for Northstar Credit?")
        assert answer is not None
        assert answer.text == "2"
        assert {item.rel_path for item in answer.evidence} == {"northstar.txt"}
    finally:
        engine.close()


def test_structured_record_sentence_recovery_for_owner_customer_and_reviewer(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "record.json").write_text(
        '{"messages": ['
        '"Mara Chen owns the retry scheduler for BeaconForce.", '
        '"Blue Ridge Analytics is blocked by the telemetry export delay.", '
        '"Ilya Stone opened the OAuth callback repair PR. Omar Vale should review it before merge."'
        ']}',
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)
    try:
        owner = engine._answer_with_generic_sentence_source("Who owns the retry scheduler for BeaconForce?")
        blocked = engine._answer_with_generic_sentence_source("Which customer is blocked by the telemetry export delay?")
        reviewer = engine._answer_with_review_or_approval_source("Who should review the OAuth callback repair PR before merge?")
        assert owner is not None and owner.text == "Mara Chen"
        assert blocked is not None and blocked.text == "Blue Ridge Analytics"
        assert reviewer is not None and reviewer.text == "Omar Vale"
    finally:
        engine.close()


def test_grounded_model_completion_expands_partial_person_but_never_substitutes(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    evidence = [Evidence("review.txt", "Correction: Omar reviewed PR-8042; Omar Kestrel performed the risk review.")]
    frame = QueryFrame(
        question_text="Who reviewed PR-8042?",
        answer_type="person",
        answer_variables=("who",),
        target_anchors=("PR-8042",),
        requested_relation="reviewed",
        relation_terms=("reviewed",),
        constraints=(),
        source="model_query_drs",
    )
    monkeypatch.setattr(engine, "_expand_single_name_from_evidence", lambda value, _evidence: "Omar Kestrel" if value == "Omar" else value)
    monkeypatch.setattr(engine, "_answer_with_review_or_approval_source", lambda *_args, **_kwargs: None)
    completed = engine._complete_grounded_model_answer(
        frame.question_text,
        Answer("Omar", 0.9, evidence, "model", "person"),
        ExpectedAnswer("person"),
        frame,
    )
    assert completed.text == "Omar Kestrel"
    monkeypatch.setattr(engine, "_expand_single_name_from_evidence", lambda value, _evidence: "Priya Moon")
    unchanged = engine._complete_grounded_model_answer(
        frame.question_text,
        Answer("Omar", 0.9, evidence, "model", "person"),
        ExpectedAnswer("person"),
        frame,
    )
    assert unchanged.text == "Omar"


def test_grounded_model_completion_restores_full_source_phrase_only_when_containing_candidate(monkeypatch) -> None:
    engine = object.__new__(KnowMoreDiRTEngine)
    evidence = [Evidence("objects.raw", 'summary: "Only ready records are valid for release."')]
    frame = QueryFrame(
        question_text="What does the Orchid Frame summary say about ready records?",
        answer_type="content_phrase",
        answer_variables=("ready records",),
        target_anchors=("Orchid Frame summary",),
        requested_relation="say",
        relation_terms=("say",),
        constraints=(),
        source="model_query_drs",
    )
    grounded = Answer("Only ready records are valid for release", 0.9, evidence, "source", "content_phrase")
    monkeypatch.setattr(engine, "_answer_with_generic_sentence_source", lambda *_args, **_kwargs: grounded)
    completed = engine._complete_grounded_model_answer(
        frame.question_text,
        Answer("are valid for release", 0.9, evidence, "model", "content_phrase"),
        ExpectedAnswer("content_phrase"),
        frame,
    )
    assert completed.text == "Only ready records are valid for release"
    unrelated = replace(grounded, text="Only paused records are invalid for release")
    monkeypatch.setattr(engine, "_answer_with_generic_sentence_source", lambda *_args, **_kwargs: unrelated)
    unchanged = engine._complete_grounded_model_answer(
        frame.question_text,
        Answer("are valid for release", 0.9, evidence, "model", "content_phrase"),
        ExpectedAnswer("content_phrase"),
        frame,
    )
    assert unchanged.text == "are valid for release"
