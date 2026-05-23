from __future__ import annotations

from pathlib import Path

from knowmoredirt.engine import KnowMoreDiRTEngine


class FakeEvidenceModel:
    def __init__(self, *, incompatible: bool = False) -> None:
        self.incompatible = incompatible
        self.calls: list[str] = []

    def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
        self.calls.append(prompt)
        if "generic DRT/DSPG query frame" in prompt:
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
    assert not answer.evidence


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


def test_fake_model_evidence_extraction_is_invoked_counted_and_grounded(tmp_path: Path) -> None:
    (tmp_path / "source").write_text(
        "Ash Meadow conservator Lyra Fen\n",
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._use_local_model = True
    engine._model_client = FakeEvidenceModel()
    engine.model_query_trace.enabled = True

    answer = engine.answer("Who is the conservator for Ash Meadow?")

    assert answer.text == "Lyra Fen"
    assert answer.answer_type == "person"
    assert answer.evidence and "Ash Meadow conservator Lyra Fen" in answer.evidence[0].text
    assert engine.model_query_trace.call_count == 1
    assert engine.model_query_trace.accepted_count == 1
    assert engine.model_query_trace.model_answer_count == 1


def test_fake_model_evidence_extraction_rejects_incompatible_answer_type(tmp_path: Path) -> None:
    (tmp_path / "source").write_text(
        "Ash Meadow has a pointer at https://example.invalid/ash but no named conservator.\n",
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._use_local_model = True
    engine._model_client = FakeEvidenceModel(incompatible=True)
    engine.model_query_trace.enabled = True

    answer = engine.answer("Who is the conservator for Ash Meadow?")

    assert answer.text == "unknown"
    assert engine.model_query_trace.evidence_call_count == 1
    assert engine.model_query_trace.evidence_rejected_count >= 1
