from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "src" / "knowmoredirt"
FIXTURES = ROOT / "tests" / "fixtures"


def _words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def test_core_has_no_benchmark_markers_or_domain_routes():
    forbidden = [
        "herb",
        "ground_truth",
        "gold_answer",
        "answerable_questions",
        "unanswerable_questions",
        "document_authors_reviewers",
        "artifact_metadata_search",
    ]
    findings = []
    for path in CORE.glob("*.py"):
        text = path.read_text().lower()
        for marker in forbidden:
            if marker in text:
                findings.append(f"{path.name}:{marker}")
    assert findings == []


def test_core_has_no_internal_fixture_phrases_or_names():
    fixture_text = "\n".join(
        path.read_text(errors="replace")
        for path in FIXTURES.rglob("*")
        if path.is_file()
    )
    words = _words(fixture_text)
    fixture_ngrams = {
        size: {
            " ".join(words[index : index + size])
            for index in range(len(words) - size + 1)
        }
        for size in range(4, 9)
    }
    fixture_names = {
        name
        for name in re.findall(
            r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b",
            fixture_text,
        )
        if len(name) >= 6
    }
    findings = []
    for path in CORE.glob("*.py"):
        text = path.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            literal_words = _words(node.value)
            for size in range(8, 3, -1):
                overlap = {
                    " ".join(literal_words[index : index + size])
                    for index in range(len(literal_words) - size + 1)
                } & fixture_ngrams[size]
                if overlap:
                    findings.append(f"{path.name}:{node.lineno}:{sorted(overlap)[0]}")
                    break
            for name in fixture_names:
                if name in node.value:
                    findings.append(f"{path.name}:{node.lineno}:{name}")
    assert findings == []


def test_engine_has_no_python_owned_lexical_semantics():
    path = CORE / "engine.py"
    text = path.read_text()
    tree = ast.parse(text)
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and any(alias.name == "re" for alias in node.names)
        for node in ast.walk(tree)
    )
    semantic_keys = {
        "question",
        "intent_summary",
        "answer_slot",
        "semantic_kind",
        "world_scope",
        "source_scope",
        "authority_mode",
        "target_phrases",
        "scope_phrases",
        "relation_phrases",
        "constraint_phrases",
        "polarity",
        "temporal_mode",
        "epistemic_mode",
        "reporting_tense",
        "requires_explicit_evidence",
        "compound_request",
    }
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.IfExp, ast.While)):
            continue
        source = ast.unparse(node.test)
        lexical_operation = any(
            token in source
            for token in (".lower(", ".casefold(", ".startswith(", ".endswith(", " in {", " in [", ".intersection(")
        )
        if lexical_operation and any(
            re.search(rf"['\"]{re.escape(key)}['\"]", source)
            for key in semantic_keys
        ):
            findings.append(f"{node.lineno}:{source}")
    assert findings == []


def test_engine_unit_tests_do_not_lock_private_semantic_heuristics():
    tree = ast.parse((ROOT / "tests" / "unit" / "test_engine.py").read_text())
    allowed = {
        "_bind_root_searches_to_contract",
        "_normalize_program",
        "_normalize_review",
        "_validate_program",
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("_")
        and not node.func.attr.startswith("__")
    }
    assert calls <= allowed


def test_public_exports_are_minimal():
    import knowmoredirt

    assert knowmoredirt.__all__ == ["initialize", "question"]
