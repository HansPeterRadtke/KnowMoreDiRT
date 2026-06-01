from __future__ import annotations

import json
from pathlib import Path

from knowmoredirt.answer_types import ExpectedAnswer
from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.model_planner import ModelQueryTrace, call_model_chunk_frames
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


def test_cached_model_results_do_not_add_fresh_model_time() -> None:
    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])

    engine._record_model_result({"fresh_or_cached": "cache", "elapsed": 83.5})
    engine._record_model_result({"fresh_or_cached": "fresh", "elapsed": 1.25})

    assert engine.model_query_trace.cache_hit_count == 1
    assert engine.model_query_trace.time_spent_seconds == 1.25


def test_unknown_diagnostic_evidence_dedupes_same_chunk_with_span_upgrade() -> None:
    engine = KnowMoreDiRTEngine.__new__(KnowMoreDiRTEngine)
    engine.last_bounded_diagnostics = {
        "execution": {
            "candidate_evidence_sample": [
                {
                    "evidence": {
                        "rel_path": "logs/state.txt",
                        "chunk_order": 2,
                        "text": "2026-03-09 status: closed for Delta Well.",
                    }
                }
            ],
            "source_provenance_sample": [
                {
                    "rel_path": "logs/state.txt",
                    "span_id": "span-final",
                    "chunk_order": 2,
                    "char_start": 80,
                    "char_end": 121,
                    "text": "2026-03-09 status: closed for Delta Well.",
                }
            ],
        }
    }

    evidence = engine._diagnostic_unknown_evidence()

    assert len(evidence) == 1
    assert evidence[0].span_id == "span-final"
    assert evidence[0].char_start == 80


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


def test_unknown_answer_retains_bounded_source_provenance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "0")
    (tmp_path / "registry.txt").write_text(
        "Registry introduces North Lantern as the sealed device.",
        encoding="utf-8",
    )

    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine.answer("What status is recorded for North Lantern?")

    assert answer.text == "unknown"
    assert answer.evidence
    assert answer.evidence[0].rel_path == "registry.txt"
    assert "North Lantern" in answer.evidence[0].text
    assert engine.last_answer is answer
    assert engine.last_bounded_diagnostics["execution"]["source_provenance_sample"]


def test_unknown_answer_surfaces_blocked_identity_provenance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "0")
    (tmp_path / "note.txt").write_text("Mira report says NC-1 is Nova Case.\n", encoding="utf-8")
    engine = KnowMoreDiRTEngine(tmp_path)
    engine.last_bounded_diagnostics = {
        "execution": {
            "blocked_identity_source_provenance": [
                {
                    "rel_path": "note.txt",
                    "text": "Mira report says NC-1 is Nova Case.",
                    "span_id": "span-demo",
                    "chunk_order": 0,
                    "char_start": 0,
                    "char_end": 36,
                    "source_kind": "sentence",
                    "expansion_blocked_reason": "missing_grounded_box",
                }
            ]
        }
    }

    answer = engine._unknown_answer("missing grounded DRS identity scope")

    assert answer.text == "unknown"
    assert answer.evidence
    assert answer.evidence[0].rel_path == "note.txt"
    assert answer.evidence[0].span_id == "span-demo"


def test_local_model_does_not_evidence_fallback_over_bounded_conflict(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "blue.txt").write_text("Blue note says Jade Pin state blue.", encoding="utf-8")
    (tmp_path / "green.txt").write_text("Green note says Jade Pin state green.", encoding="utf-8")

    class ConflictModel:
        def __init__(self) -> None:
            self.evidence_calls = 0
            self.query_evidence_calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-conflict-no-evidence-fallback", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Blue note" in prompt or "Green note" in prompt:
                state = "blue" if "Blue note" in prompt else "green"
                text = f"{state.title()} note says Jade Pin state {state}."
                return {
                    "drs": {
                        "schema_version": "chunk-drs-v2",
                        "source_id": f"{state}.txt",
                        "referents": [
                            {"id": "r0", "label": "Jade Pin", "kind": "artifact", "evidence_text": "Jade Pin"}
                        ],
                        "boxes": [
                            {
                                "id": "b0",
                                "kind": "asserted",
                                "parent_id": "",
                                "holder_referent_id": "",
                                "evidence_text": text,
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
                                        "role": "entity",
                                        "target_kind": "referent",
                                        "target_id": "r0",
                                        "value": "",
                                        "value_type": "artifact",
                                        "evidence_text": "Jade Pin",
                                    },
                                    {
                                        "role": "state",
                                        "target_kind": "literal",
                                        "target_id": "",
                                        "value": state,
                                        "value_type": "state",
                                        "evidence_text": state,
                                    },
                                ],
                                "evidence_text": f"Jade Pin state {state}",
                            }
                        ],
                        "identity_hypotheses": [],
                        "temporal_records": [],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            if "generic DRT/DSPG query frame" in prompt:
                return {
                    "query_frame": {
                        "target_anchors": ["Jade Pin"],
                        "answer_variables": ["state"],
                        "requested_relation": "state",
                        "relation_terms": ["state"],
                        "constraints": [],
                        "scope_requirements": [],
                        "modality_requirements": [],
                        "answer_type": "state",
                        "temporal_scope": "",
                        "negated": False,
                        "aggregation": "",
                        "requires_evidence": True,
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            if "Answer the question only from the provided raw-text evidence" in prompt:
                self.evidence_calls += 1
                return {
                    "answer": {
                        "sufficient_evidence": True,
                        "answer_type": "state",
                        "answer": "blue",
                        "evidence_span": "Jade Pin state blue",
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            if "bounded DRT/DSPG question analysis" in prompt:
                self.query_evidence_calls += 1
                return {
                    "result": {
                        "query_frame": {
                            "target_anchors": ["Jade Pin"],
                            "answer_variables": ["state"],
                            "requested_relation": "state",
                            "relation_terms": ["state"],
                            "constraints": [],
                            "scope_requirements": [],
                            "modality_requirements": [],
                            "answer_type": "state",
                            "temporal_scope": "",
                            "negated": False,
                            "aggregation": "",
                            "requires_evidence": True,
                        },
                        "sufficient_evidence": True,
                        "answer_type": "state",
                        "answer": "blue",
                        "evidence_span": "Jade Pin state blue",
                        "reason": "fake fallback should be blocked",
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            raise AssertionError(prompt[:200])

    fake = ConflictModel()
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "1")
    monkeypatch.setenv("KMD_LLM_INGEST", "0")
    monkeypatch.setenv("KMD_LLM_DRS_INGEST", "1")
    monkeypatch.setenv("KMD_QUERY_DRS_PLAN", "0")
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    monkeypatch.setenv("KMD_QUERY_PLAN_CACHE_DIR", str(tmp_path / ".query-cache"))
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", lambda: fake)

    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine.answer("What state is recorded for Jade Pin?")

    assert answer.text == "unknown"
    assert engine.last_bounded_diagnostics["execution"]["answer_conflict_without_query_scope"]
    assert (
        engine.last_bounded_diagnostics["execution"]["model_evidence_fallback_blocked_reason"]
        == "answer_conflict_without_query_scope"
    )
    assert answer.evidence
    assert {item.rel_path for item in answer.evidence} == {"blue.txt", "green.txt"}
    assert fake.evidence_calls == 0
    assert fake.query_evidence_calls == 0


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


def test_frame_cache_context_separates_identical_text_by_source_path(tmp_path: Path, monkeypatch) -> None:
    fake = FakeFrameModel()
    corpus = tmp_path / "corpus"
    (corpus / "alpha").mkdir(parents=True)
    (corpus / "beta").mkdir()
    text = "Marble Gate is guarded by Sena Rill.\n"
    (corpus / "alpha" / "frame.raw").write_text(text, encoding="utf-8")
    (corpus / "beta" / "frame.raw").write_text(text, encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "1")
    monkeypatch.setenv("KMD_LLM_INGEST", "1")
    monkeypatch.setenv("KMD_FRAME_CACHE_DIR", str(tmp_path / ".frame-cache"))
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", lambda: fake)

    KnowMoreDiRTEngine(corpus)

    extraction_prompts = [prompt for prompt in fake.prompts if "Extract generic DRT/DSPG discourse frames" in prompt]
    assert len(extraction_prompts) == 2
    assert any('"source": "alpha/frame.raw"' in prompt for prompt in extraction_prompts)
    assert any('"source": "beta/frame.raw"' in prompt for prompt in extraction_prompts)


def test_drs_attempt_cache_context_separates_identical_text_by_source_path(tmp_path: Path, monkeypatch) -> None:
    class FakeDrsModel:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.n_predicts: list[int] = []

        def context_size(self) -> int:
            return 32768

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-drs-source-cache-context", "context_size": 32768}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.prompts.append(prompt)
            self.n_predicts.append(int(n_predict))
            text = "Aero Gate is ready."
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "fake",
                    "referents": [
                        {"id": "r0", "label": "Aero Gate", "kind": "artifact", "evidence_text": "Aero Gate"},
                    ],
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
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
                                    "role": "subject",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "artifact",
                                    "evidence_text": "Aero Gate",
                                }
                            ],
                            "evidence_text": "Aero Gate is ready",
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    fake = FakeDrsModel()
    corpus = tmp_path / "corpus"
    (corpus / "alpha").mkdir(parents=True)
    (corpus / "beta").mkdir()
    text = "Aero Gate is ready.\n"
    (corpus / "alpha" / "note.raw").write_text(text, encoding="utf-8")
    (corpus / "beta" / "note.raw").write_text(text, encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "1")
    monkeypatch.setenv("KMD_LLM_INGEST", "0")
    monkeypatch.setenv("KMD_LLM_DRS_INGEST", "1")
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    monkeypatch.setattr("knowmoredirt.engine.LocalModelClient", lambda: fake)

    engine = KnowMoreDiRTEngine(corpus)

    rows = engine.store.execute(
        """
        SELECT cache_key, metadata_json
        FROM model_attempts
        WHERE task='chunk_drs'
        ORDER BY source_span_id
        """
    ).fetchall()
    contexts = [json.loads(row["metadata_json"])["cache_context"] for row in rows]
    assert len(rows) == 2
    assert len({row["cache_key"] for row in rows}) == 2
    assert {context["n_predict"] for context in contexts} == set(fake.n_predicts)
    assert set(fake.n_predicts) == {544}
    assert all(context["context_budget"]["input_chars"] == len("Aero Gate is ready.") for context in contexts)
    assert all(context["context_budget"]["source_span_candidate_count"] >= 1 for context in contexts)
    assert {context["source_rel_path"] for context in contexts} == {
        "alpha/note.raw",
        "beta/note.raw",
    }


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


def test_lazy_frame_materialization_skips_previous_failed_attempts(tmp_path: Path, monkeypatch) -> None:
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
                        "identity_hypotheses": [],
                        "polarity": "positive",
                        "modality": "asserted",
                        "context_holder": "",
                        "temporal_text": "",
                        "evidence_text": "Ungrounded evidence",
                        "confidence": 0.9,
                    }
                ],
                "_model_raw": "{}",
            }

    fake = RejectingFrameModel()
    (tmp_path / "frame.raw").write_text("Marble Gate is guarded by Sena Rill.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "0")
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._model_client = fake
    engine._semantic_cache = None
    sentence = engine.sentences[0]

    first = engine._materialize_sentence_semantics(sentence)
    second = engine._materialize_sentence_semantics(sentence)

    assert first == 0
    assert second == 0
    assert sum("Extract generic DRT/DSPG discourse frames" in prompt for prompt in fake.prompts) == 1
    assert engine.store.execute("SELECT COUNT(*) FROM frames WHERE source='local_model'").fetchone()[0] == 0
    attempt = engine.store.execute(
        "SELECT accepted, materialized, reason FROM model_attempts WHERE task='chunk_frames'"
    ).fetchone()
    assert attempt is not None
    assert bool(attempt["accepted"]) is False
    assert bool(attempt["materialized"]) is False
    assert attempt["reason"] == "grounding_validation_failed"


def test_lazy_frame_materialization_retries_previous_request_failures(tmp_path: Path, monkeypatch) -> None:
    class TransientFrameModel(FakeLocalModel):
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-lazy-frame-request-retry", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            assert "Extract generic DRT/DSPG discourse frames" in prompt
            if self.calls == 1:
                raise RuntimeError("temporary lazy frame request failure")
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
                        "temporal_text": "",
                        "evidence_text": "Aero Gate is ready.",
                        "confidence": 0.9,
                    }
                ],
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    fake = TransientFrameModel()
    (tmp_path / "frame.raw").write_text("Aero Gate is ready.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "0")
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._model_client = fake
    engine._semantic_cache = None
    sentence = engine.sentences[0]

    first = engine._materialize_sentence_semantics(sentence)
    second = engine._materialize_sentence_semantics(sentence)

    assert first == 0
    assert second == 1
    assert fake.calls == 2
    attempt = engine.store.execute(
        "SELECT accepted, materialized, reason, metadata_json FROM model_attempts WHERE task='chunk_frames'"
    ).fetchone()
    assert attempt is not None
    assert bool(attempt["accepted"]) is True
    assert bool(attempt["materialized"]) is True
    assert attempt["reason"] == ""
    metadata = json.loads(attempt["metadata_json"])
    assert metadata["context_budget"]["runtime_context_size"] == 4096


def test_lazy_frame_materialization_replaces_stale_cache_context_rows(tmp_path: Path, monkeypatch) -> None:
    class VersionedFrameModel(FakeLocalModel):
        def __init__(self, version: str, predicate: str) -> None:
            self.version = version
            self.predicate = predicate
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": f"fake-lazy-frame-{self.version}", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            assert "Extract generic DRT/DSPG discourse frames" in prompt
            return {
                "frames": [
                    {
                        "frame_type": "state",
                        "predicate": self.predicate,
                        "arguments": [{"role": "entity", "text": "Aero Gate", "value_type": "entity"}],
                        "identity_hypotheses": [],
                        "polarity": "positive",
                        "modality": "asserted",
                        "context_holder": "",
                        "temporal_text": "",
                        "evidence_text": "Aero Gate is ready.",
                        "confidence": 0.9,
                    }
                ],
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    first_model = VersionedFrameModel("v1", "ready_v1")
    second_model = VersionedFrameModel("v2", "ready_v2")
    (tmp_path / "frame.raw").write_text("Aero Gate is ready.\n", encoding="utf-8")
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "0")
    engine = KnowMoreDiRTEngine(tmp_path)
    engine._semantic_cache = None
    engine._model_client = first_model
    sentence = engine.sentences[0]

    assert engine._materialize_sentence_semantics(sentence) == 1
    initial_predicates = [
        row["predicate"]
        for row in engine.store.execute("SELECT predicate FROM frames WHERE source='local_model'")
    ]
    assert initial_predicates == ["ready_v1"]

    engine._model_client = second_model
    assert engine._materialize_sentence_semantics(sentence) == 1

    predicates = [
        row["predicate"]
        for row in engine.store.execute("SELECT predicate FROM frames WHERE source='local_model' ORDER BY predicate")
    ]
    assert predicates == ["ready_v2"]
    assert first_model.calls == 1
    assert second_model.calls == 1
    attempt_rows = engine.store.execute(
        "SELECT metadata_json FROM model_attempts WHERE task='chunk_frames'"
    ).fetchall()
    attempt_metadata = [json.loads(row["metadata_json"]) for row in attempt_rows]
    assert any(metadata.get("replaced_prior_rows", {}).get("frames") == 1 for metadata in attempt_metadata)
    assert engine.store.execute(
        "SELECT COUNT(*) FROM model_attempts WHERE task='chunk_frames' AND materialized=1"
    ).fetchone()[0] == 1

    engine._model_client = first_model
    assert engine._materialize_sentence_semantics(sentence) == 1
    predicates = [
        row["predicate"]
        for row in engine.store.execute("SELECT predicate FROM frames WHERE source='local_model' ORDER BY predicate")
    ]
    assert predicates == ["ready_v1"]
    assert first_model.calls == 2
    assert engine.store.execute(
        "SELECT COUNT(*) FROM model_attempts WHERE task='chunk_frames' AND materialized=1"
    ).fetchone()[0] == 1


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
    arg_types = {
        str(row["value_type"])
        for row in engine.store.execute("SELECT value_type FROM frame_arguments").fetchall()
    }
    assert "person" in arg_types
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
    assert answer.evidence
    assert answer.evidence[0].rel_path == "plain"
    assert "OrionLeaf" in answer.evidence[0].text


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
