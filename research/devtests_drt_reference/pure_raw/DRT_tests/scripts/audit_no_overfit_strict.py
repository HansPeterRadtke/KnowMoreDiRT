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
    "config/dspg_system.yaml",
    "dspg_store.py",
    "run_staged_tests.py",
    "run_tests.py",
    "extract.py",
    "contracts.py",
    "dspg.py",
    "merge.py",
    "scripts/evaluate_surrogate.py",
    "scripts/run_query_paraphrase_tests.py",
    "scripts/build_herb_raw_artifact_probe.py",
    "scripts/run_folder_dspg_demo.py",
    "scripts/dspg_ingest_folder.py",
    "scripts/dspg_query.py",
}

FIXTURE_OR_REPORT_PREFIXES = (
    "tests/",
    "logs/",
    "scripts/generate_dspg_surrogate.py",
    "scripts/generate_blackbox_corpora.py",
    "scripts/run_blackbox_evaluation.py",
    "scripts/verify_phase3_complete.py",
    "scripts/write_phase3_readiness_report.py",
    "scripts/package_phase3_results.py",
    "scripts/prove_model_stages.py",
    "scripts/run_staged_model_isolated_tests.py",
    "scripts/audit_no_overfit.py",
    "scripts/audit_no_overfit_strict.py",
)

GENERIC_ALLOWED_SUBSTRINGS = {
    "PR-",
    "BUG-",
    "ISSUE-",
    "SUP-",
    "TICKET-",
    "http",
    ".cpp",
    ".tmp",
    ".py",
    ".js",
    ".yaml",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def collect_fixture_values() -> dict[str, set[str]]:
    buckets: dict[str, set[str]] = {
        "names": set(),
        "companies": set(),
        "ids": set(),
        "urls": set(),
        "files": set(),
        "queries": set(),
        "answers": set(),
        "case_ids": set(),
        "family_ids": set(),
    }
    fixture_roots = [
        ROOT / "tests" / "inputs",
        ROOT / "tests" / "expected",
        ROOT / "tests" / "realworld_inputs",
        ROOT / "tests" / "realworld_expected",
        ROOT / "tests" / "realworld_stage2_inputs",
        ROOT / "tests" / "realworld_stage2_expected",
        ROOT / "tests" / "generated_surrogate",
        ROOT / "tests" / "blackbox",
        ROOT / "tests" / "blackbox_oracles",
    ]
    for base in fixture_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            text = read_text(path)
            buckets["ids"].update(re.findall(r"\b(?:PR|BUG|ISSUE|SUP|TICKET)-\d+\b", text))
            buckets["urls"].update(re.findall(r"https?://[^\s\]\)\"']+", text))
            buckets["files"].update(re.findall(r"\b[A-Za-z0-9_./-]+\.(?:cpp|tmp|py|js|yaml|yml|md|txt)\b", text))
            buckets["names"].update(re.findall(r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]{2,}\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]{2,}\b", text))
            if path.name.endswith((".json", ".yaml", ".yml")):
                try:
                    data = json.loads(text) if path.suffix == ".json" else None
                except Exception:
                    data = None
                if data is not None:
                    collect_json_values(data, buckets)
            buckets["queries"].update(re.findall(r'"question"\s*:\s*"([^"]+)"', text))
    for key in list(buckets):
        buckets[key] = {clean(v) for v in buckets[key] if keep_value(v)}
    return buckets


def collect_json_values(obj: Any, buckets: dict[str, set[str]]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"id", "case_id"} and isinstance(value, str):
                buckets["case_ids"].add(value)
            elif key == "family" and isinstance(value, str):
                buckets["family_ids"].add(value)
            elif key in {"question"} and isinstance(value, str):
                buckets["queries"].add(value)
            elif key in {"expected", "expected_answer_refs"}:
                for item in value if isinstance(value, list) else [value]:
                    if isinstance(item, str):
                        buckets["answers"].add(item)
            collect_json_values(value, buckets)
    elif isinstance(obj, list):
        for item in obj:
            collect_json_values(item, buckets)


def clean(value: str) -> str:
    return str(value).strip().strip(".,;:?!)]}\"'")


def keep_value(value: str) -> bool:
    value = clean(value)
    if len(value) < 4:
        return False
    low = value.lower()
    generic_words = {
        "source grounded",
        "customer impact",
        "final decision",
        "requested reviewer",
        "reviewer comment",
        "support brief",
        "change note",
        "review digest",
        "unknown",
        "open",
        "closed",
        "quoted",
        "reported",
        "asserted",
        "different",
        "same",
        "person",
        "company",
        "customer",
        "artifact",
        "content",
        "people",
    }
    return low not in generic_words


def iter_code_files() -> list[Path]:
    out = []
    for pattern in ("*.py", "*.sh", "*.yaml", "*.yml"):
        for path in ROOT.rglob(pattern):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(("__pycache__/", ".venv/", "_runtime/")):
                continue
            if rel.startswith(FIXTURE_OR_REPORT_PREFIXES):
                continue
            out.append(path)
    return sorted(out)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def literal_findings(code_files: list[Path], fixture_values: dict[str, set[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    disallowed: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    all_values: list[tuple[str, str]] = []
    for bucket, values in fixture_values.items():
        for value in values:
            all_values.append((bucket, value))
    all_values.sort(key=lambda item: len(item[1]), reverse=True)
    for path in code_files:
        text = read_text(path)
        path_rel = rel(path)
        is_solver = path_rel in SOLVER_FILES or path.name in SOLVER_FILES
        for bucket, value in all_values:
            if not value or value not in text:
                continue
            line = text.count("\n", 0, text.find(value)) + 1
            finding = {"file": path_rel, "line": line, "bucket": bucket, "literal": value}
            if not is_solver:
                allowed.append({**finding, "status": "non_solver_code_reference"})
            elif bucket in {"ids", "urls", "files"} and any(token in value for token in GENERIC_ALLOWED_SUBSTRINGS) and value in {"PR-", "BUG-", "SUP-"}:
                allowed.append({**finding, "status": "generic_regex_reference"})
            else:
                disallowed.append({**finding, "status": "disallowed_solver_literal"})
    return disallowed, allowed


def threshold_findings(code_files: list[Path]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in code_files:
        path_rel = rel(path)
        if path_rel not in SOLVER_FILES and path.name not in SOLVER_FILES:
            continue
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(read_text(path))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                constants = []
                for side in [node.left, *node.comparators]:
                    if isinstance(side, ast.Constant) and isinstance(side.value, (int, float)):
                        constants.append(side.value)
                if any(value in {0.9, 0.90, 90} for value in constants):
                    findings.append(
                        {
                            "file": path_rel,
                            "line": getattr(node, "lineno", 0),
                            "literal": ast.unparse(node) if hasattr(ast, "unparse") else "compare",
                            "status": "disallowed_fixed_pass_threshold_candidate",
                        }
                    )
    return findings


def branch_findings(code_files: list[Path], fixture_values: dict[str, set[str]]) -> list[dict[str, Any]]:
    ids = fixture_values["case_ids"] | fixture_values["family_ids"] | fixture_values["queries"]
    findings: list[dict[str, Any]] = []
    for path in code_files:
        path_rel = rel(path)
        if path_rel not in SOLVER_FILES and path.name not in SOLVER_FILES:
            continue
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(read_text(path))
        except Exception:
            continue
        branch_types = (ast.If,) + ((ast.Match,) if hasattr(ast, "Match") else ())
        for node in ast.walk(tree):
            if isinstance(node, branch_types):
                src = ast.get_source_segment(read_text(path), node) or ""
                for value in ids:
                    if value and value in src:
                        findings.append(
                            {
                                "file": path_rel,
                                "line": getattr(node, "lineno", 0),
                                "literal": value,
                                "status": "disallowed_branch_on_case_family_or_query",
                            }
                        )
    return findings


def main() -> int:
    LOGS.mkdir(exist_ok=True)
    fixture_values = collect_fixture_values()
    code_files = iter_code_files()
    literal_bad, literal_allowed = literal_findings(code_files, fixture_values)
    thresholds = threshold_findings(code_files)
    branches = branch_findings(code_files, fixture_values)
    disallowed = literal_bad + thresholds + branches
    result = {
        "passed": not disallowed,
        "disallowed_findings": disallowed,
        "allowed_or_justified_findings": literal_allowed[:500],
        "fixture_value_counts": {key: len(value) for key, value in fixture_values.items()},
        "scanned_solver_files": sorted(str(path) for path in SOLVER_FILES),
        "scanned_code_files": [rel(path) for path in code_files],
        "policy": "Solver code must not contain fixture/generated literals, exact query text branches, case/family branches, or fixed pass-threshold comparisons. Generators, tests, logs, and reports are fixture/report space.",
    }
    (LOGS / "NO_OVERFIT_STRICT_AUDIT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Strict No-Overfit Audit",
        "",
        f"- passed: `{result['passed']}`",
        f"- disallowed findings: `{len(disallowed)}`",
        f"- allowed/justified findings shown: `{len(result['allowed_or_justified_findings'])}`",
        f"- fixture values scanned: `{result['fixture_value_counts']}`",
        "",
        "## Disallowed Findings",
    ]
    lines += [f"- `{f['file']}:{f['line']}` `{f['literal']}` ({f['status']})" for f in disallowed] or ["- none"]
    lines += ["", "## Allowed / Justified Findings"]
    lines += [f"- `{f['file']}:{f['line']}` `{f['literal']}` ({f['status']})" for f in result["allowed_or_justified_findings"]] or ["- none"]
    (LOGS / "NO_OVERFIT_STRICT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "disallowed": len(disallowed)}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
