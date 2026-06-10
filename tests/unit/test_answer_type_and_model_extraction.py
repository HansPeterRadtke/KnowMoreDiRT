from __future__ import annotations

from pathlib import Path

import knowmoredirt.engine as engine_module
from knowmoredirt.answer_types import ExpectedAnswer, canonicalize_answer
from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.model_planner import call_model_evidence_answer
from knowmoredirt.models import Answer
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
    def __init__(self, *, incompatible: bool = False) -> None:
        self.incompatible = incompatible
        self.calls: list[str] = []

    def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
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
        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
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
    ) -> dict[str, object]:
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

    answer = engine.answer("Is Cedar Ledger a migration plan target?")

    assert answer.text == "No; Cedar Ledger is unrelated to any migration plan."
    assert answer.reason == "local model query-DRS evidence verification"
    assert engine.model_query_trace.evidence_call_count == 1


def test_invalid_model_evidence_answer_is_cached(tmp_path: Path, monkeypatch) -> None:
    class InvalidEvidenceModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
            self.calls += 1
            return {"unexpected": "shape", "_model_raw": '{"unexpected":"shape"}'}

    model = InvalidEvidenceModel()
    monkeypatch.setenv("KMD_EVIDENCE_ANSWER_CACHE_DIR", str(tmp_path / "evidence-cache"))
    evidence = [{"source": "note", "text": "Ash Meadow conservator Lyra Fen"}]

    first = call_model_evidence_answer("Who is the conservator for Ash Meadow?", "person", evidence, model)  # type: ignore[arg-type]
    second = call_model_evidence_answer("Who is the conservator for Ash Meadow?", "person", evidence, model)  # type: ignore[arg-type]

    assert model.calls == 1
    assert first["accepted"] is False
    assert first["cache_context"]["expected_answer_type"] == "person"
    assert first["cache_context"]["n_predict"] == 128
    assert first["cache_context"]["evidence_count"] == 1
    assert second["accepted"] is False
    assert second["fresh_or_cached"] == "cache"
    assert second["cache_context"]["expected_answer_type"] == "person"



def test_general_boolean_source_explanation_patterns(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "dream.txt").write_text(
        "I had a dream that Crane deleted lock.key.\nWhen I woke up, the repository still contained lock.key.\n",
        encoding="utf-8",
    )
    (tmp_path / "judgment.txt").write_text("Final judgment summary.\nThe court found no proof that Widget caused drift.\n", encoding="utf-8")
    (tmp_path / "runtime.txt").write_text("Runtime note: the code flags stale rows for human review; it does not delete them.\n", encoding="utf-8")
    (tmp_path / "fiction.txt").write_text("School story: The candy bridge drawing floated.\nTeacher note: this is fiction homework, not an engineering record.\n", encoding="utf-8")
    (tmp_path / "audit.txt").write_text("Sora believes CacheBox stores plaintext secrets.\nAudit result: CacheBox stores only salted secret hashes.\n", encoding="utf-8")
    (tmp_path / "garden.txt").write_text("Market sketch for PlantBoard.\nThis unrelated gardening note mentions market research but has no relation to any product roadmap.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_boolean_source_explanation("Did Crane really delete lock.key?").text == "No; the deletion occurred only in a dream and the repository still contained lock.key."
    assert engine._answer_with_boolean_source_explanation("Was Widget proven to have caused drift?").text == "No; the final judgment found no proof."
    assert engine._answer_with_boolean_source_explanation("Does the runtime delete stale rows?").text == "No; runtime flags stale rows for human review."
    assert engine._answer_with_boolean_source_explanation("Should the candy bridge drawing be treated as an engineering record?").text == "No; it is fiction homework."
    assert engine._answer_with_boolean_source_explanation("Does the audit say CacheBox stores plaintext secrets?").text == "No; it stores only salted secret hashes."
    assert engine._answer_with_boolean_source_explanation("Is PlantBoard a product roadmap target?").text == "No; it is an unrelated gardening note."



def test_central_answer_guard_rejects_unrelated_no_proof(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "judgment.txt").write_text(
        "Final judgment summary.\nThe court found no proof that Widget caused invoice drift.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KMD_TEST_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine._answer_with_boolean_source_explanation("Was Widget proven to have caused invoice drift?").text == "No; the final judgment found no proof."
    assert engine._answer_with_boolean_source_explanation("Was Ardent Mill refund request proven by the judgment?") is None


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

    assert engine._cleanup_public_answer(answer).text == "Omar Kestrel"



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
    assert engine._answer_with_source_rows("How many contacts are listed for Northstar Credit?") is None



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
    assert engine._answer_with_actor_role_ids_source("Which actor id belongs to the key reviewer of Aurora Loom Safety Note?").text == "ACT-411"
    assert engine._answer_with_actor_role_ids_source("Find actor IDs of the author and reviewers of Aurora Loom Safety Note.").text == "ACT-410; ACT-411; ACT-412"
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
