from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "prepare_herb_kmd_bundle.py"
SPEC = importlib.util.spec_from_file_location("prepare_herb_kmd_bundle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_bundle_separates_source_questions_and_gold(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    artifacts = [
        {
            "artifact_id": "doc-1",
            "artifact_type": "document",
            "product_id": "alpha",
            "raw_text": "Alpha owns REF-1. Another sentence.",
            "metadata": {"id": "doc-1"},
        },
        {
            "artifact_id": "slack-1",
            "artifact_type": "slack",
            "product_id": "alpha",
            "raw_text": "Iris reviewed REF-1.",
            "metadata": {"id": "slack-1"},
        },
    ]
    questions = [
        {
            "question_id": "q1",
            "question": "Who owns REF-1?",
            "answerable": True,
            "metadata": {"ground_truth": ["Alpha"], "citations": ["doc-1"]},
        }
    ]
    gold = [
        {
            "question_id": "q1",
            "question": "Who owns REF-1?",
            "answerable": True,
            "gold_answer": ["Alpha"],
            "gold_source_ids": ["doc-1"],
        }
    ]
    write_jsonl(normalized / "artifacts.jsonl", artifacts)
    write_jsonl(normalized / "questions.jsonl", questions)
    write_jsonl(normalized / "gold.jsonl", gold)
    output = tmp_path / "prepared"

    manifest = MODULE.prepare_bundle(normalized, output)
    validated = MODULE.validate_prepared_bundle(output / "manifest.json")

    assert manifest["artifact_count"] == 2
    assert manifest["question_count"] == 1
    assert manifest["forbidden_key_hits"] == {}
    assert manifest["official_question_text_hits"] == 0
    source_text = "\n".join(path.read_text() for path in (output / "source").rglob("*.jsonl"))
    assert "Who owns REF-1?" not in source_text
    assert "ground_truth" not in source_text
    assert "gold_answer" not in source_text
    prepared_questions = [json.loads(line) for line in (output / "questions.jsonl").read_text().splitlines()]
    assert prepared_questions == [{"question_id": "q1", "question": "Who owns REF-1?"}]
    assert validated["validation"]["source_record_count"] == 2


def test_prepare_bundle_rejects_benchmark_fields_in_source(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    write_jsonl(
        normalized / "artifacts.jsonl",
        [
            {
                "artifact_id": "bad",
                "artifact_type": "document",
                "product_id": "alpha",
                "raw_text": "fact",
                "ground_truth": ["leak"],
            }
        ],
    )
    write_jsonl(normalized / "questions.jsonl", [{"question_id": "q1", "question": "Question?"}])
    write_jsonl(normalized / "gold.jsonl", [{"question_id": "q1", "gold_answer": ["answer"]}])

    try:
        MODULE.prepare_bundle(normalized, tmp_path / "prepared")
    except ValueError as error:
        assert "forbidden benchmark keys" in str(error)
    else:
        raise AssertionError("expected forbidden source key failure")
