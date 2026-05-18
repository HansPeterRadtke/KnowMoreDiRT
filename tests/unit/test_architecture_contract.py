from __future__ import annotations

from pathlib import Path

from conftest import REPO_ROOT


FORBIDDEN_CORE_MARKERS = [
    "HERB RAW ARTIFACT",
    "allow_prepared_metadata",
    "DRT_HERB_PREP_ROOT",
    "artifact_manifest_by_rel_path",
    "source_corpus",
    "product_id",
    "source_title",
]


def test_core_package_has_no_benchmark_or_prepared_input_markers() -> None:
    source_files = list((REPO_ROOT / "src" / "knowmoredirt").glob("*.py"))
    assert source_files
    findings: list[str] = []
    for path in source_files:
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_CORE_MARKERS:
            if marker in text:
                findings.append(f"{path.relative_to(REPO_ROOT)}:{marker}")
    assert findings == []


def test_public_api_exports_only_two_user_functions() -> None:
    init_file = REPO_ROOT / "src" / "knowmoredirt" / "__init__.py"
    text = init_file.read_text(encoding="utf-8")
    assert '__all__ = ["initialize", "question"]' in text

