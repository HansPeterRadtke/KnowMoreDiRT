from __future__ import annotations

import json
from pathlib import Path

from conftest import BROAD_FIXTURE_ROOT, BROAD_QA_PATH, NOISE_FIXTURE_ROOT, NOISE_QA_PATH


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_broad_world_fixture_has_heterogeneous_raw_text_coverage() -> None:
    files = [path for path in BROAD_FIXTURE_ROOT.rglob("*") if path.is_file()]
    payload = _load(BROAD_QA_PATH)
    categories = {entry["category"] for entry in payload["questions"]}

    assert len(files) >= 35
    assert len(payload["questions"]) >= 60
    assert len(categories) >= 45
    assert {"school_homework", "recipe", "legal_claim", "veterinary", "multilingual", "raw_json_text"}.issubset(categories)
    assert any(path.suffix == "" for path in files)
    assert any(path.suffix in {".csvish", ".blob", ".mix"} for path in files)


def test_broad_and_noise_qa_entries_are_source_grounded() -> None:
    for root, qa_path in [(BROAD_FIXTURE_ROOT, BROAD_QA_PATH), (NOISE_FIXTURE_ROOT, NOISE_QA_PATH)]:
        payload = _load(qa_path)
        for entry in payload["questions"]:
            assert entry["question"]
            assert entry["answer"]
            evidence_paths = entry.get("evidence", [])
            assert evidence_paths
            for rel_path in evidence_paths:
                evidence_file = root / rel_path
                assert evidence_file.exists(), f"missing evidence file: {evidence_file}"
                assert evidence_file.read_text(encoding="utf-8", errors="replace").strip()


def test_hardcore_noise_fixture_contains_distinct_noise_styles() -> None:
    files = {path.relative_to(NOISE_FIXTURE_ROOT).as_posix() for path in NOISE_FIXTURE_ROOT.rglob("*") if path.is_file()}

    assert "chars/x00_noise.txt" in files
    assert "chars/base64ish.blob" in files
    assert "words/salad.noext" in files
    assert "words/multilingual_nonsense.mix" in files
    assert "mixed/adversarial_names.txt" in files
    assert "ocr/garbage_scan.txt" in files
