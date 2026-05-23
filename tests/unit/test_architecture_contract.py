from __future__ import annotations

import re
from pathlib import Path

from conftest import REPO_ROOT


FORBIDDEN_CORE_MARKERS = [
    "HERB",
    "HELP",
    "benchmark",
    "benchmark family",
    "question family",
    "benchmark routing",
    "benchmark intents",
    "parity",
    "scorer",
    "gold",
    "answerability",
    "question_id",
    "official question",
    "prepared",
    "HERB RAW ARTIFACT",
    "allow_prepared_metadata",
    "DRT_HERB_PREP_ROOT",
    "artifact_manifest_by_rel_path",
    "source_corpus",
    "product_id",
    "source_title",
    "product_name",
    "employee_ids",
    "customer_id",
    "which_pr",
    "which_customer",
    "which_ticket",
    "which_issue",
    "max-PR",
    "unresolved-bug",
    "employee-ID",
    "artifact search",
    "role_lookup",
    "reference_lookup",
    "url_lookup",
    "file_lookup",
    "state_lookup",
    "answer_role",
    "_answer_who_role",
    "_answer_identifier_or_url",
    "_answer_final_state",
]

FORBIDDEN_CORE_REGEXES = [
    r"\bPR\b",
    r"\bpr\b",
    r"\bticket\b",
    r"\bissue\b",
    r"\bcustomer\b",
    r"\bemployee\b",
    r"\bartifact\b",
    r"\bif\s+.*['\"](?:owner|reviewer|approver|reporter|author)['\"]",
    r"\belif\s+.*['\"](?:owner|reviewer|approver|reporter|author)['\"]",
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
        for pattern in FORBIDDEN_CORE_REGEXES:
            if re.search(pattern, text):
                findings.append(f"{path.relative_to(REPO_ROOT)}:{pattern}")
    assert findings == []


def test_public_api_exports_only_two_user_functions() -> None:
    init_file = REPO_ROOT / "src" / "knowmoredirt" / "__init__.py"
    text = init_file.read_text(encoding="utf-8")
    assert '__all__ = ["initialize", "question"]' in text


def test_core_package_has_no_fixture_or_domain_shaped_literals() -> None:
    forbidden = [
        "FlowQuill",
        "ActionGarden",
        "vault.key",
        "stale ledgers",
        "plaintext",
        "cache expiration",
        "parser.cpp",
        "MarlinKind",
        "RippleDesk",
        "Blue Dune",
        "Northstar Credit",
    ]
    findings: list[str] = []
    for path in (REPO_ROOT / "src" / "knowmoredirt").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker.lower() in text.lower():
                findings.append(f"{path.relative_to(REPO_ROOT)}:{marker}")
    assert findings == []
