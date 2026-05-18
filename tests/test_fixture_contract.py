from __future__ import annotations

import json
from pathlib import Path

from conftest import FIXTURE_ROOT, QA_PATH


BANNED_STRUCTURE_MARKERS = {
    "HERB RAW ARTIFACT",
    "artifact_type:",
    "product_id:",
    "source_title:",
    "customer_id:",
    "prepared corpus",
    "DRT_HERB",
}

REQUIRED_CATEGORIES = {
    "direct_fact",
    "multi_hop",
    "temporal",
    "contradiction_resolution",
    "source_grounded",
    "unanswerable",
    "belief_vs_fact",
    "dream_vs_fact",
    "claim_vs_fact",
    "discussion_disagreement",
    "distractor_avoidance",
    "exact_id",
    "exact_url",
    "aggregation",
    "raw_json_text",
    "table_context",
}


def test_messy_raw_fixture_has_nested_unstructured_shape() -> None:
    assert FIXTURE_ROOT.is_dir()
    files = [path for path in FIXTURE_ROOT.rglob("*") if path.is_file()]
    directories = [path for path in FIXTURE_ROOT.rglob("*") if path.is_dir()]
    suffixes = {path.suffix for path in files}
    no_extension = [path for path in files if path.suffix == ""]

    assert len(files) >= 25
    assert len(directories) >= 12
    assert len(no_extension) >= 3
    assert {".log", ".md", ".eml", ".chat", ".jsonish", ".tsv"}.issubset(suffixes)
    assert max(len(path.relative_to(FIXTURE_ROOT).parts) for path in files) >= 3


def test_every_fixture_file_is_plain_readable_text_without_required_wrappers() -> None:
    files = [path for path in FIXTURE_ROOT.rglob("*") if path.is_file()]
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert text.strip(), path
        for marker in BANNED_STRUCTURE_MARKERS:
            assert marker not in text, f"{marker!r} found in {path}"


def test_qa_file_is_valid_and_category_rich() -> None:
    payload = json.loads(QA_PATH.read_text(encoding="utf-8"))
    questions = payload["questions"]
    categories = {entry["category"] for entry in questions}
    ids = [entry["id"] for entry in questions]

    assert payload["corpus_root"] == "messy_raw_corpus"
    assert len(questions) >= 50
    assert len(ids) == len(set(ids))
    assert REQUIRED_CATEGORIES.issubset(categories)
    assert sum(1 for entry in questions if entry["answer"] == "unknown") >= 8


def test_qa_evidence_points_only_to_raw_corpus_text() -> None:
    payload = json.loads(QA_PATH.read_text(encoding="utf-8"))
    for entry in payload["questions"]:
        assert isinstance(entry["question"], str) and entry["question"].strip()
        assert isinstance(entry["answer"], str) and entry["answer"].strip()
        assert isinstance(entry["evidence"], list)

        if entry["answer"] == "unknown":
            assert entry["evidence"] == []
            continue

        assert entry["evidence"], entry["id"]
        for evidence in entry["evidence"]:
            rel_path = Path(evidence["file"])
            assert not rel_path.is_absolute()
            source_path = FIXTURE_ROOT / rel_path
            assert source_path.is_file(), f"missing evidence file {rel_path}"
            source_text = source_path.read_text(encoding="utf-8")
            assert evidence["snippet"] in source_text, f"snippet missing for {entry['id']}"

