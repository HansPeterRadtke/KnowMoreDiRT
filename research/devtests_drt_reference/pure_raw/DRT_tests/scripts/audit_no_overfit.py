#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"

SOLVER_FILES = {
    "run_staged_tests.py",
    "run_tests.py",
    "extract.py",
    "contracts.py",
    "dspg.py",
    "merge.py",
}
ALLOWED_FIXTURE_FILES = {
    "scripts/run_staged_model_isolated_tests.py",
    "scripts/prove_model_stages.py",
    "scripts/generate_dspg_surrogate.py",
}
GENERIC_REGEX_ALLOW = {
    "PR-", "BUG-", "ISSUE-", "SUP-", "TICKET-", "http", "https", ".cpp", ".tmp", ".py", ".md"
}
FIXED_SEMANTIC_THRESHOLD_RE = re.compile(r"(?:score|threshold|pass|passed)[^\n]{0,80}(?:0\.9(?!\d)|90(?:\D|$))", re.I)


def load_test_strings() -> set[str]:
    values: set[str] = set()
    for base in [ROOT / "tests" / "inputs", ROOT / "tests" / "expected", ROOT / "tests" / "realworld_stage2_inputs", ROOT / "tests" / "realworld_stage2_expected"]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore")
                values.update(re.findall(r"\b(?:PR|BUG|SUP|TICKET|ISSUE)-\d+\b", text))
                values.update(re.findall(r"https?://[^\s\]\)\"']+", text))
                values.update(re.findall(r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9]*(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9]*){1,3}\b", text))
    filtered = set()
    for value in values:
        low = value.lower()
        if len(value) < 4:
            continue
        if low in {"the operational", "source grounded", "when", "final decision", "customer impact", "known issue", "requested reviewer", "reviewer comment"}:
            continue
        filtered.add(value.strip().strip(".,;:"))
    return filtered


def iter_source_files():
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("logs/", "tests/", ".venv/", "__pycache__/")):
            continue
        yield path, rel
    for path in ROOT.rglob("*.sh"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("logs/", "tests/", ".venv/")):
            continue
        yield path, rel


def string_literals(path: Path) -> list[tuple[int, str]]:
    if path.suffix != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append((getattr(node, "lineno", 0), node.value))
    return out


def main() -> int:
    test_values = load_test_strings()
    findings: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    for path, rel in iter_source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        is_solver = path.name in SOLVER_FILES or rel in SOLVER_FILES
        is_allowed_fixture = rel in ALLOWED_FIXTURE_FILES
        for value in sorted(test_values, key=len, reverse=True):
            if not value or value not in text:
                continue
            if any(token in value for token in GENERIC_REGEX_ALLOW) and not is_solver:
                bucket = allowed
                status = "allowed_fixture_or_generic"
            elif is_allowed_fixture:
                bucket = allowed
                status = "allowed_test_fixture"
            elif is_solver:
                bucket = findings
                status = "disallowed_solver_literal"
            else:
                bucket = allowed
                status = "allowed_non_solver_reference"
            lineno = next((i for i, line in enumerate(text.splitlines(), 1) if value in line), 0)
            bucket.append({"file": rel, "line": lineno, "literal": value, "status": status})
        for match in FIXED_SEMANTIC_THRESHOLD_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            entry = {"file": rel, "line": line, "literal": match.group(0)[:120], "status": "fixed_threshold_check"}
            if is_solver:
                findings.append(entry)
            else:
                allowed.append(entry)
    result = {
        "passed": not findings,
        "disallowed_findings": findings,
        "allowed_or_justified_findings": allowed,
        "scanned_test_value_count": len(test_values),
        "scanned_source_files": [rel for _, rel in iter_source_files()],
        "policy": "Solver logic must not contain current diagnostic entity/ID/URL/query answer literals. Generic regexes and explicitly isolated test fixtures are allowed but reported.",
    }
    (LOGS / "NO_OVERFIT_AUDIT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# No-Overfit Audit", "", f"- passed: `{result['passed']}`", f"- disallowed findings: `{len(findings)}`", f"- allowed/justified findings: `{len(allowed)}`", "", "## Disallowed Findings"]
    lines += [f"- `{f['file']}:{f['line']}` `{f['literal']}` ({f['status']})" for f in findings] or ["- none"]
    lines += ["", "## Allowed / Justified Findings"]
    lines += [f"- `{f['file']}:{f['line']}` `{f['literal']}` ({f['status']})" for f in allowed[:200]] or ["- none"]
    (LOGS / "NO_OVERFIT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "disallowed": len(findings), "allowed": len(allowed)}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
