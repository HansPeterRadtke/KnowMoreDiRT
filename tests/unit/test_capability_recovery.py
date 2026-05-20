from __future__ import annotations

from pathlib import Path

from knowmoredirt.engine import KnowMoreDiRTEngine


class FakeLocalModel:
    def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, object]:
        assert "generic raw-text knowledge query plan" in prompt
        assert grammar and "query_plan" in grammar
        return {
            "query_plan": {
                "intent": "role_lookup",
                "target_surface": "SequoiaLens",
                "answer_role": "owner",
                "requires_asserted": True,
            },
            "_model_raw": '{"query_plan":{"intent":"role_lookup","target_surface":"SequoiaLens","answer_role":"owner","requires_asserted":true}}',
            "_model_elapsed_seconds": 0.01,
        }


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
    assert answer.reason == "local model query plan: role_lookup"
    assert answer.evidence
    assert "Owner: Nia Vale" in answer.evidence[0].text
    assert engine.last_bounded_diagnostics["ranking"]["selected_chunk_count"] > 0
    assert engine.last_bounded_diagnostics["execution"]["record_counts"]["relations"] > 0
    assert engine.model_query_trace.call_count == 1
    assert engine.model_query_trace.accepted_count == 1
    assert engine.model_query_trace.model_answer_count == 1


def test_bounded_graph_execution_runs_without_model_for_context_lookup(tmp_path: Path) -> None:
    (tmp_path / "loose").mkdir()
    (tmp_path / "loose" / "dream-note").write_text(
        "DreamBridge was only a dream about a silver hinge.\nNo waking record asserts the hinge.",
        encoding="utf-8",
    )

    engine = KnowMoreDiRTEngine(tmp_path)
    answer = engine.answer("What dream context is asserted for DreamBridge?")

    assert answer.text == "dreamed"
    assert answer.evidence
    assert "DreamBridge" in answer.evidence[0].text
    assert engine.last_bounded_diagnostics["execution"]["record_counts"]["context_carriers"] > 0


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
