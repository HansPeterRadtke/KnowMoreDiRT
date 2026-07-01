from __future__ import annotations

import json
from pathlib import Path

from knowmoredirt.scanner import scan_folder
from knowmoredirt.text import text_quality_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
STRUCTURED_RECORD_ROOT = REPO_ROOT / "tests" / "fixtures" / "structured_record_json"
STRUCTURED_RECORD_QA = REPO_ROOT / "tests" / "fixtures" / "structured_record_json_qa.json"


def test_structured_record_json_fixture_is_in_internal_benchmark() -> None:
    payload = json.loads(STRUCTURED_RECORD_QA.read_text(encoding="utf-8"))
    categories = {question["category"] for question in payload["questions"]}
    product_text = (STRUCTURED_RECORD_ROOT / "products" / "BeaconForce.json").read_text(encoding="utf-8")

    assert STRUCTURED_RECORD_ROOT.exists()
    assert len(payload["questions"]) == 6
    assert "structured_json_product" in categories
    assert "structured_json_pr_url" in categories
    assert "structured_json_customer" in categories
    assert "structured_json_unanswerable" in categories
    assert "github.com/salesforce/BeaconForce/pull/2780" in product_text
    assert "forbestechcouncil" in product_text
    assert len(product_text) > 16000


def test_structured_record_json_fixture_contains_herb_like_large_record_units() -> None:
    _documents, units = scan_folder(STRUCTURED_RECORD_ROOT, max_unit_chars=16000)
    product_units = [unit for unit in units if unit.rel_path == "products/BeaconForce.json"]
    product_quality = [text_quality_metrics(unit.text) for unit in product_units]

    assert len(product_units) >= 3
    assert any("github.com/salesforce/BeaconForce/pull/2780" in unit.text for unit in product_units)
    assert any("Blue Ridge Analytics" in unit.text for unit in product_units)
    assert any("HOLD" in unit.text for unit in product_units)
    assert all(quality["semantic_quality"] != "base64_or_hex_blob" for quality in product_quality)
    assert all(quality["low_semantic_noise"] is False for quality in product_quality)
