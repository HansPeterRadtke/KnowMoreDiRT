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
