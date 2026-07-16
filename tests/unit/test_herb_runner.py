from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "scripts" / "benchmarks" / "run_herb_knowmoredirt.py"
spec = importlib.util.spec_from_file_location("run_herb_knowmoredirt", MODULE_PATH)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_sanitize_herb_source_removes_questions_and_preserves_artifacts(tmp_path):
    raw = tmp_path / "raw"
    (raw / "products").mkdir(parents=True)
    (raw / "metadata").mkdir()
    (raw / "products" / "Product.json").write_text(json.dumps({
        "documents": [{"id": "doc-1", "content": "useful"}],
        "answerable_questions": [{"question": "leak", "ground_truth": ["secret"]}],
        "unanswerable_questions": ["leak two"],
    }))
    (raw / "metadata" / "employee.json").write_text("{}")
    output = tmp_path / "source"
    counts = runner.sanitize_herb_source(raw, output)
    clean = json.loads((output / "products" / "Product.json").read_text())
    assert clean == {"documents": [{"id": "doc-1", "content": "useful"}]}
    assert counts["removed_qa_fields"] == 2


def test_prediction_answer_and_evidence_output_shapes():
    assert runner.prediction_answer("A; B", "company") == ["A", "B"]
    assert runner.prediction_answer("unknown", "content") == []

    class Answer:
        evidence = ({
            "record_id": "chunk-1",
            "source_path": "products/X.json",
            "data": {"id": "artifact-7"},
            "excerpt": "evidence",
        },)

    source_ids, chunk_ids, chunks = runner.evidence_outputs(Answer())
    assert source_ids == ["artifact-7"]
    assert chunk_ids == ["chunk-1"]
    assert chunks[0]["artifact_id"] == "artifact-7"


def test_select_questions_drops_ground_truth_metadata():
    rows = [{
        "question_id": "q1",
        "question": "Who owns it?",
        "question_type": "person",
        "product_id": "p1",
        "answerable": True,
        "metadata": {"ground_truth": ["secret"], "citations": ["doc"]},
    }]
    assert runner.select_questions(rows) == [{
        "question_id": "q1",
        "question": "Who owns it?",
        "question_type": "person",
        "product_id": "p1",
    }]


def test_evidence_output_ignores_ids_inside_uncited_child_record_lists():
    class Answer:
        evidence = ({
            "record_id": "chunk-2",
            "source_path": "products/Y.json",
            "data": {
                "id": "container-id",
                "metadata": {"artifact_id": "artifact-primary"},
                "messages": [
                    {"message_id": "uncited-message-1"},
                    {"message_id": "uncited-message-2"},
                ],
            },
            "excerpt": "evidence",
        },)

    source_ids, _, chunks = runner.evidence_outputs(Answer())
    assert source_ids == ["artifact-primary"]
    assert chunks[0]["artifact_id"] == "artifact-primary"
