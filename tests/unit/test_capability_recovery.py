from __future__ import annotations

from pathlib import Path

from knowmoredirt.answer_types import ExpectedAnswer
from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.model_planner import call_model_chunk_frames
from knowmoredirt.query import QueryFrame


class FakeLocalModel:
    def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
        if "Verify whether the candidate answer is entailed" in prompt:
            return {
                "verification": {
                    "entailed": True,
                    "answer_type": "person",
                    "answer": "Nia Vale",
                    "evidence_span": "Owner: Nia Vale",
                    "reason": "fake grounded verifier",
                },
                "_model_raw": '{"verification":{"entailed":true,"answer_type":"person","answer":"Nia Vale","evidence_span":"Owner: Nia Vale","reason":"fake grounded verifier"}}',
        }
        assert "generic DRT/DSPG query frame" in prompt
        return {
            "query_frame": {
                "target_anchors": ["SequoiaLens"],
                "requested_relation": "owns",
                "relation_terms": ["owns"],
                "constraints": [],
                "answer_type": "person",
                "temporal_scope": "",
                "negated": False,
                "aggregation": "",
                "requires_evidence": True,
            },
            "_model_raw": '{"query_frame":{"target_anchors":["SequoiaLens"],"requested_relation":"owns","relation_terms":["owns"],"constraints":[],"answer_type":"person","temporal_scope":"","negated":false,"aggregation":"","requires_evidence":true}}',
            "_model_elapsed_seconds": 0.01,
        }


class FakeFrameModel(FakeLocalModel):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
        self.prompts.append(prompt)
        if "Extract generic DRT/DSPG discourse frames" in prompt:
            return {
                "frames": [
                    {
                        "frame_type": "relation",
                        "predicate": "guards",
                        "arguments": [
                            {"role": "entity", "text": "Marble Gate", "value_type": "entity"},
                            {"role": "participant", "text": "Sena Rill", "value_type": "person"},
                        ],
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_text": "",
                        "evidence_text": "Marble Gate is guarded by Sena Rill",
                        "confidence": 0.91,
                    }
                ],
                "_model_raw": '{"frames":[{"frame_type":"relation","predicate":"guards","arguments":[{"role":"entity","text":"Marble Gate","value_type":"entity"},{"role":"participant","text":"Sena Rill","value_type":"person"}],"polarity":"positive","modality":"asserted","temporal_text":"","evidence_text":"Marble Gate is guarded by Sena Rill","confidence":0.91}]}',
            }
        if "generic DRT/DSPG query frame" in prompt and "Marble Gate" in prompt:
            return {
                "query_frame": {
                    "target_anchors": ["Marble Gate"],
                    "requested_relation": "guards",
                    "relation_terms": ["guards"],
                    "constraints": [],
                    "answer_type": "person",
                    "temporal_scope": "",
                    "negated": False,
                    "aggregation": "",
                    "requires_evidence": True,
                },
                "_model_raw": '{"query_frame":{"target_anchors":["Marble Gate"],"requested_relation":"guards","relation_terms":["guards"],"constraints":[],"answer_type":"person","temporal_scope":"","negated":false,"aggregation":"","requires_evidence":true}}',
            }
        if "Verify whether the candidate answer is entailed" in prompt and "Marble Gate" in prompt:
            return {
                "verification": {
                    "entailed": True,
                    "answer_type": "person",
                    "answer": "Sena Rill",
                    "evidence_span": "Marble Gate is guarded by Sena Rill",
                    "reason": "fake grounded verifier",
                },
                "_model_raw": '{"verification":{"entailed":true,"answer_type":"person","answer":"Sena Rill","evidence_span":"Marble Gate is guarded by Sena Rill","reason":"fake grounded verifier"}}',
            }
        return super().complete_json(prompt, n_predict=n_predict, grammar=grammar)


def test_document_metadata_is_retrieval_prior_not_answer_source(tmp_path: Path) -> None:
    (tmp_path / "random_a").mkdir()
    (tmp_path / "random_b").mkdir()
    (tmp_path / "random_a" / "SequoiaLens.notes").write_text(
        "Owner: Nia Vale\nThe project uses a plain notebook entry.\n",
        encoding="utf-8",
    )
    (tmp_path / "random_b" / "distractor.txt").write_text(
        "Owner: Rho Kit\nThis unrelated note describes another object.\n",
        encoding="utf-8",
    )

    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine.answer("Who is the owner for SequoiaLens?")

    assert answer.text == "Nia Vale"
    assert answer.evidence
    assert "Owner: Nia Vale" in answer.evidence[0].text


def test_optional_local_model_invokes_generic_query_plan_path(tmp_path: Path) -> None:
    (tmp_path / "odd").mkdir()
    (tmp_path / "odd" / "SequoiaLens.raw").write_text(
        "Owner: Nia Vale\nThe delivery motto for this note is blue lantern.\n",
        encoding="utf-8",
    )
    (tmp_path / "other.txt").write_text("The delivery motto elsewhere is red comet.\n", encoding="utf-8")

    engine = KnowMoreDiRTEngine(tmp_path)
    engine._use_local_model = True
    engine._model_client = FakeLocalModel()
    engine.model_query_trace.enabled = True
    answer = engine.answer("Who owns SequoiaLens?")

    assert answer.text == "Nia Vale"
    assert answer.reason == "local model query-frame execution"
    assert answer.evidence
    assert "Owner: Nia Vale" in answer.evidence[0].text
    assert engine.last_bounded_diagnostics["ranking"]["selected_chunk_count"] > 0
    assert engine.last_bounded_diagnostics["execution"]["record_counts"]["relations"] > 0
    assert engine.model_query_trace.call_count == 1
    assert engine.model_query_trace.accepted_count == 1
    assert engine.model_query_trace.model_answer_count == 1


def test_local_model_ingest_builds_grounded_generic_frames(tmp_path: Path, monkeypatch) -> None:
    fake = FakeFrameModel()
    (tmp_path / "frame.raw").write_text("Marble Gate is guarded by Sena Rill.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "1")
    monkeypatch.setenv("KMD_LLM_INGEST", "1")
    monkeypatch.setenv("KMD_FRAME_CACHE_DIR", str(tmp_path / ".frame-cache"))
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", lambda: fake)

    engine = KnowMoreDiRTEngine(tmp_path)

    counts = engine.dspg_counts()
    semantic_rows = engine.store.execute("SELECT COUNT(*) FROM frames WHERE source='local_model'").fetchone()[0]
    assert semantic_rows >= 1
    assert counts["relations"] >= 2
    assert any("Extract generic DRT/DSPG discourse frames" in prompt for prompt in fake.prompts)
    assert engine.model_query_trace.chunk_frame_call_count >= 1
    assert engine.model_query_trace.chunk_frame_parsed_count >= 1
    assert engine.model_query_trace.chunk_frame_accepted_count >= 1


def test_local_model_ingest_logs_chunk_progress(tmp_path: Path, monkeypatch, capsys) -> None:
    fake = FakeFrameModel()
    (tmp_path / "frame.raw").write_text("Marble Gate is guarded by Sena Rill.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "1")
    monkeypatch.setenv("KMD_LLM_INGEST", "1")
    monkeypatch.setenv("KMD_PROGRESS", "1")
    monkeypatch.setenv("KMD_FRAME_CACHE_DIR", str(tmp_path / ".frame-cache"))
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", lambda: fake)

    KnowMoreDiRTEngine(tmp_path)
    output = capsys.readouterr().out

    assert "kmd-ingest llm_start chunk=1/1 source=frame.raw:0" in output
    assert "kmd-ingest llm_done chunk=1/1 source=frame.raw:0" in output
    assert "frames=1" in output


def test_local_model_ingest_caches_rejected_grounding_results(tmp_path: Path, monkeypatch) -> None:
    class RejectingFrameModel(FakeLocalModel):
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
            self.prompts.append(prompt)
            assert "Extract generic DRT/DSPG discourse frames" in prompt
            return {
                "frames": [
                    {
                        "frame_type": "relation",
                        "predicate": "guards",
                        "arguments": [{"role": "participant", "text": "Ungrounded Name", "value_type": "person"}],
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_text": "",
                        "evidence_text": "Ungrounded evidence",
                        "confidence": 0.9,
                    }
                ],
                "_model_raw": "{}",
            }

    fake = RejectingFrameModel()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "frame.raw").write_text("Marble Gate is guarded by Sena Rill.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "1")
    monkeypatch.setenv("KMD_LLM_INGEST", "1")
    monkeypatch.setenv("KMD_FRAME_CACHE_DIR", str(tmp_path / ".frame-cache"))
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", lambda: fake)

    first = KnowMoreDiRTEngine(corpus)
    second = KnowMoreDiRTEngine(corpus)

    assert sum("Extract generic DRT/DSPG discourse frames" in prompt for prompt in fake.prompts) == 1
    assert first.store.execute("SELECT COUNT(*) FROM frames WHERE source='local_model'").fetchone()[0] == 0
    assert second.store.execute("SELECT COUNT(*) FROM frames WHERE source='local_model'").fetchone()[0] == 0


def test_local_model_frame_arguments_bind_answer_variables_generically(tmp_path: Path, monkeypatch) -> None:
    fake = FakeFrameModel()
    (tmp_path / "frame.raw").write_text("Marble Gate is guarded by Sena Rill.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "1")
    monkeypatch.setenv("KMD_LLM_INGEST", "1")
    monkeypatch.setenv("KMD_FRAME_CACHE_DIR", str(tmp_path / ".frame-cache"))
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", lambda: fake)

    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine.answer("Who guards Marble Gate?")

    assert answer.text == "Sena Rill"
    assert answer.evidence
    assert answer.reason in {"local model query-frame execution", "bounded DSPG query-frame execution"}
    assert engine.last_bounded_diagnostics["execution"]["record_counts"]["frame_arguments"] >= 2
    assert engine.dspg_counts()["identity_hypotheses"] >= 0


def test_query_drs_answer_variable_selects_model_frame_role(tmp_path: Path, monkeypatch) -> None:
    class FakeRoleFrameModel(FakeLocalModel):
        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
            if "Extract generic DRT/DSPG discourse frames" in prompt:
                return {
                    "frames": [
                        {
                            "frame_type": "event",
                            "predicate": "gave",
                            "arguments": [
                                {"role": "agent", "text": "Ana", "value_type": "person"},
                                {"role": "theme", "text": "blue key", "value_type": "entity"},
                                {"role": "recipient", "text": "Zachary Vale", "value_type": "person"},
                            ],
                            "polarity": "positive",
                            "modality": "asserted",
                            "context_holder": "",
                            "temporal_text": "",
                            "evidence_text": "Ana gave the blue key to Zachary Vale",
                            "confidence": 0.9,
                        }
                    ],
                    "_model_raw": "{}",
                }
            return super().complete_json(prompt, n_predict=n_predict, grammar=grammar)

    (tmp_path / "event.txt").write_text("Ana gave the blue key to Zachary Vale.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "1")
    monkeypatch.setenv("KMD_LLM_INGEST", "1")
    monkeypatch.setenv("KMD_FRAME_CACHE_DIR", str(tmp_path / ".frame-cache"))
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", lambda: FakeRoleFrameModel())
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="model query DRS with answer role variable",
        answer_type="person",
        answer_variables=("recipient",),
        target_anchors=("blue key",),
        requested_relation="gave",
        relation_terms=("gave",),
        constraints=(),
    )

    answer = engine._answer_with_bounded_dspg("role variable DRS", frame, ExpectedAnswer("person"))

    assert answer is not None
    assert answer.text == "Zachary Vale"


def test_bounded_graph_execution_uses_model_frames_for_context_lookup(tmp_path: Path, monkeypatch) -> None:
    class FakeContextModel(FakeLocalModel):
        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
            if "Extract generic DRT/DSPG discourse frames" in prompt:
                return {
                    "frames": [
                        {
                            "frame_type": "context",
                            "predicate": "context",
                            "arguments": [
                                {"role": "entity", "text": "DreamBridge", "value_type": "entity"},
                                {"role": "value", "text": "dreamed", "value_type": "state"},
                            ],
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_text": "",
                            "evidence_text": "DreamBridge was only a dream about a silver hinge",
                            "confidence": 0.89,
                        }
                    ],
                    "_model_raw": "{}",
                }
            if "generic DRT/DSPG query frame" in prompt:
                return {
                    "query_frame": {
                        "target_anchors": ["DreamBridge"],
                        "requested_relation": "context",
                        "relation_terms": ["context"],
                        "constraints": [],
                        "answer_type": "state",
                        "temporal_scope": "",
                        "negated": False,
                        "aggregation": "",
                        "requires_evidence": True,
                    },
                    "_model_raw": "{}",
                }
            if "Verify whether the candidate answer is entailed" in prompt:
                return {
                    "verification": {
                        "entailed": True,
                        "answer_type": "state",
                        "answer": "dreamed",
                        "evidence_span": "DreamBridge was only a dream about a silver hinge",
                        "reason": "fake grounded verifier",
                    },
                    "_model_raw": "{}",
                }
            return super().complete_json(prompt, n_predict=n_predict, grammar=grammar)

    (tmp_path / "loose").mkdir()
    (tmp_path / "loose" / "dream-note").write_text(
        "DreamBridge was only a dream about a silver hinge.\nNo waking record asserts the hinge.",
        encoding="utf-8",
    )

    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "1")
    monkeypatch.setenv("KMD_LLM_INGEST", "1")
    monkeypatch.setenv("KMD_FRAME_CACHE_DIR", str(tmp_path / ".frame-cache"))
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", lambda: FakeContextModel())
    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine.answer("What dream context is asserted for DreamBridge?")

    assert answer.text == "dreamed"
    assert answer.evidence
    assert "DreamBridge" in answer.evidence[0].text
    assert engine.last_bounded_diagnostics["execution"]["record_counts"]["context_carriers"] > 0


def test_modal_context_requires_query_drs_scope(tmp_path: Path, monkeypatch) -> None:
    class FakeModalModel(FakeLocalModel):
        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
            if "Extract generic DRT/DSPG discourse frames" in prompt:
                return {
                    "frames": [
                        {
                            "frame_type": "state",
                            "predicate": "state",
                            "arguments": [
                                {"role": "entity", "text": "Violet Rack", "value_type": "entity"},
                                {"role": "value", "text": "sealed", "value_type": "state"},
                            ],
                            "polarity": "positive",
                            "modality": "reported",
                            "context_holder": "Report",
                            "temporal_text": "",
                            "evidence_text": "Report: Violet Rack was sealed",
                            "confidence": 0.9,
                        }
                    ],
                    "_model_raw": "{}",
                }
            return super().complete_json(prompt, n_predict=n_predict, grammar=grammar)

    (tmp_path / "report.txt").write_text("Report: Violet Rack was sealed.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "1")
    monkeypatch.setenv("KMD_LLM_INGEST", "1")
    monkeypatch.setenv("KMD_FRAME_CACHE_DIR", str(tmp_path / ".frame-cache"))
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", lambda: FakeModalModel())
    engine = KnowMoreDiRTEngine(tmp_path)
    expected = ExpectedAnswer("state")
    asserted_frame = QueryFrame(
        question_text="model query DRS without modal requirement",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Violet Rack",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
    )
    scoped_frame = QueryFrame(
        question_text="model query DRS with modal requirement",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Violet Rack",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        modality_requirements=("reported",),
    )
    relation_scoped_frame = QueryFrame(
        question_text="model query DRS with requested relation matching modal context",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Violet Rack",),
        requested_relation="reported",
        relation_terms=("state",),
        constraints=(),
    )

    asserted_answer = engine._answer_with_bounded_dspg("asserted DRS", asserted_frame, expected)
    scoped_answer = engine._answer_with_bounded_dspg("reported DRS", scoped_frame, expected)
    relation_scoped_answer = engine._answer_with_bounded_dspg("relation-scoped DRS", relation_scoped_frame, expected)

    assert asserted_answer is None
    assert scoped_answer is not None
    assert scoped_answer.text == "sealed"
    assert relation_scoped_answer is not None
    assert relation_scoped_answer.text == "sealed"


def test_unary_model_predicate_can_bind_nonstructural_answer_value(tmp_path: Path, monkeypatch) -> None:
    class FakeUnaryPredicateModel(FakeLocalModel):
        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
            if "Extract generic DRT/DSPG discourse frames" in prompt:
                return {
                    "frames": [
                        {
                            "frame_type": "state",
                            "predicate": "was sealed",
                            "arguments": [
                                {"role": "entity", "text": "Violet Rack", "value_type": "entity"},
                            ],
                            "polarity": "positive",
                            "modality": "asserted",
                            "context_holder": "",
                            "temporal_text": "",
                            "evidence_text": "Violet Rack was sealed",
                            "confidence": 0.9,
                        }
                    ],
                    "_model_raw": "{}",
                }
            return super().complete_json(prompt, n_predict=n_predict, grammar=grammar)

    (tmp_path / "state.txt").write_text("Violet Rack was sealed.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "1")
    monkeypatch.setenv("KMD_LLM_INGEST", "1")
    monkeypatch.setenv("KMD_FRAME_CACHE_DIR", str(tmp_path / ".frame-cache"))
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", lambda: FakeUnaryPredicateModel())
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="model query DRS for unary condition",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Violet Rack",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
    )

    answer = engine._answer_with_bounded_dspg("unary predicate DRS", frame, ExpectedAnswer("state"))

    assert answer is not None
    assert answer.text == "was sealed"


def test_model_polarity_context_blocks_unnegated_query_drs(tmp_path: Path, monkeypatch) -> None:
    class FakeNegativePredicateModel(FakeLocalModel):
        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
            if "Extract generic DRT/DSPG discourse frames" in prompt:
                return {
                    "frames": [
                        {
                            "frame_type": "state",
                            "predicate": "sealed",
                            "arguments": [
                                {"role": "entity", "text": "Violet Rack", "value_type": "entity"},
                                {"role": "value", "text": "sealed", "value_type": "state"},
                            ],
                            "polarity": "negative",
                            "modality": "asserted",
                            "context_holder": "",
                            "temporal_text": "",
                            "evidence_text": "Violet Rack was not sealed",
                            "confidence": 0.9,
                        }
                    ],
                    "_model_raw": "{}",
                }
            return super().complete_json(prompt, n_predict=n_predict, grammar=grammar)

    (tmp_path / "state.txt").write_text("Violet Rack was not sealed.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "1")
    monkeypatch.setenv("KMD_LLM_INGEST", "1")
    monkeypatch.setenv("KMD_FRAME_CACHE_DIR", str(tmp_path / ".frame-cache"))
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", lambda: FakeNegativePredicateModel())
    engine = KnowMoreDiRTEngine(tmp_path)
    expected = ExpectedAnswer("state")
    asserted_frame = QueryFrame(
        question_text="model query DRS without negated scope",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Violet Rack",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
    )
    negated_frame = QueryFrame(
        question_text="model query DRS with negated scope",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Violet Rack",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        negated=True,
    )

    asserted_answer = engine._answer_with_bounded_dspg("asserted DRS", asserted_frame, expected)
    negated_answer = engine._answer_with_bounded_dspg("negated DRS", negated_frame, expected)

    assert asserted_answer is None
    assert negated_answer is not None
    assert negated_answer.text == "sealed"


def test_chunk_frame_temporal_text_must_be_source_grounded() -> None:
    class FakeUngroundedTemporalModel(FakeLocalModel):
        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
            assert "Extract generic DRT/DSPG discourse frames" in prompt
            return {
                "frames": [
                    {
                        "frame_type": "state",
                        "predicate": "ready",
                        "arguments": [{"role": "entity", "text": "Aero Gate", "value_type": "entity"}],
                        "identity_hypotheses": [],
                        "polarity": "positive",
                        "modality": "asserted",
                        "context_holder": "",
                        "temporal_text": "tomorrow",
                        "evidence_text": "Aero Gate is ready",
                        "confidence": 0.9,
                    }
                ],
                "_model_raw": "{}",
            }

    result = call_model_chunk_frames(
        "Aero Gate is ready.",
        FakeUngroundedTemporalModel(),
        rel_path="state.txt",
    )

    assert result["accepted"] is False
    assert result["reason"] == "grounding_validation_failed"
    assert result["rejected_for_grounding"] >= 1


def test_file_metadata_answers_require_metadata_question(tmp_path: Path) -> None:
    target = tmp_path / "AtlasNote.txt"
    target.write_text("AtlasNote says the lamp state: steady.\n", encoding="utf-8")
    expected_size = str(target.stat().st_size)

    engine = KnowMoreDiRTEngine(tmp_path)
    metadata_answer = engine.answer("What size is AtlasNote.txt?")
    fact_answer = engine.answer("What is the lamp state?")

    assert metadata_answer.text == expected_size
    assert metadata_answer.evidence
    assert metadata_answer.evidence[0].text.startswith("metadata size_bytes:")
    assert fact_answer.text == "steady"
    assert not fact_answer.evidence[0].text.startswith("metadata ")


def test_missing_source_evidence_returns_unknown(tmp_path: Path) -> None:
    (tmp_path / "plain").write_text("OrionLeaf has no visible reference value.\n", encoding="utf-8")

    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine.answer("Which reference identifies OrionLeaf?")

    assert answer.text == "unknown"
    assert not answer.evidence


def test_core_has_no_prepared_or_herb_marker_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    forbidden = [
        "HERB RAW ARTIFACT",
        "allow_prepared_metadata",
        "prepared corpus",
        "question_id_map",
        "gold_answer",
    ]
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in (repo_root / "src" / "knowmoredirt").glob("*.py"))

    for marker in forbidden:
        assert marker not in source_text
