from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "src" / "knowmoredirt"


def test_core_has_no_benchmark_markers_or_domain_routes():
    forbidden = ["herb", "ground_truth", "gold_answer", "employee ids", "pull request", "document_authors_reviewers", "artifact_metadata_search"]
    findings = []
    for path in CORE.glob("*.py"):
        text = path.read_text().lower()
        for marker in forbidden:
            if marker in text:
                findings.append(f"{path.name}:{marker}")
    assert findings == []


def test_core_has_no_string_triggered_question_semantics():
    findings = []
    for path in CORE.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.If, ast.IfExp)):
                continue
            source = ast.unparse(node.test)
            if any(name in source for name in ["question", "question.lower", "question.casefold"]):
                literal_comparison = False
                for comparison in (child for child in ast.walk(node.test) if isinstance(child, ast.Compare)):
                    compared = [comparison.left, *comparison.comparators]
                    if any(isinstance(value, ast.Constant) and isinstance(value.value, str) for value in compared):
                        literal_comparison = True
                if literal_comparison:
                    findings.append(f"{path.name}:{node.lineno}:{source}")
    assert findings == []


def test_public_exports_are_minimal():
    import knowmoredirt
    assert knowmoredirt.__all__ == ["initialize", "question"]
