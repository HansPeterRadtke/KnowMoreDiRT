#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BLACKBOX = ROOT / "tests" / "blackbox"
ORACLES = ROOT / "tests" / "blackbox_oracles"
LOGS = ROOT / "logs" / "phase3" / "blackbox"

VARIANTS = [
    "deterministic_only",
    "model_mention_type_only",
    "model_frame_only",
    "model_scope_only",
    "model_identity_only",
    "model_query_plan_only",
    "all_model_assisted",
    "one_shot_baseline",
]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def run_cmd(cmd: list[str], timeout: int = 240) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return proc.returncode, proc.stdout


def load_cases(suite: str) -> list[dict[str, Any]]:
    suite_root = BLACKBOX / suite
    cases = []
    for case_dir in sorted(p for p in suite_root.iterdir() if p.is_dir()):
        oracle = json.loads((ORACLES / suite / case_dir.name / "oracle.json").read_text(encoding="utf-8"))
        questions = [json.loads(line) for line in (case_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        cases.append({"suite": suite, "case_id": case_dir.name, "corpus": case_dir / "corpus", "questions": questions, "oracle": oracle})
    return cases


def model_endpoint() -> str:
    return os.environ.get("DRT_PHASE3_MODEL_ENDPOINT", "http://127.0.0.1:14829")


def one_shot_answer(case: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    texts = []
    for path in sorted(case["corpus"].rglob("*")):
        if path.is_file():
            texts.append(f"FILE {path.relative_to(case['corpus'])}\n{path.read_text(encoding='utf-8', errors='replace')}")
    prompt = (
        "Answer the question from the source text. Return JSON only as {\"answer\":\"...\"}. "
        "Use unknown if evidence is insufficient.\n"
        + json.dumps({"sources": texts, "question": question["question"]}, ensure_ascii=False)
    )
    grammar = 'root ::= "{" ws "\\"answer\\"" ws ":" ws string ws "}"\nstring ::= "\\"" chars "\\""\nchars ::= ([^"\\\\] | "\\\\" ["\\\\/bfnrt])*\nws ::= [ \\t\\n\\r]*'
    body = {"prompt": prompt, "n_predict": 160, "temperature": 0.0, "top_p": 1.0, "stream": False, "grammar": grammar}
    try:
        req = urllib.request.Request(model_endpoint() + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        raw = data.get("content", "")
        start = raw.find("{")
        obj = json.loads(raw[start: raw.rfind("}") + 1]) if start >= 0 and "}" in raw else {}
        return {"answer": obj.get("answer", "unknown"), "raw": raw, "accepted": bool(obj), "evidence": []}
    except Exception as exc:
        return {"answer": "unknown", "raw": "", "accepted": False, "error": str(exc), "evidence": []}


def query_case(case: dict[str, Any], variant: str, config: Path, out_root: Path) -> dict[str, Any]:
    if variant == "one_shot_baseline":
        query_results = []
        for q in case["questions"]:
            answer = one_shot_answer(case, q)
            query_results.append({"id": q["id"], "question": q["question"], "answers": [{"answer": answer["answer"], "evidence": []}], "plan": {"source": "one_shot"}, "status": "answered" if answer["answer"] != "unknown" else "unknown", "one_shot": answer})
        return {"ingest": {"returncode": 0, "totals": {"model_calls": len(case["questions"]), "request_failed": 0, "truncated": 0, "schema_invalid": 0}}, "query_results": query_results}
    ingest_variant = "deterministic_only" if variant == "model_query_plan_only" else variant
    db_path = out_root / case["suite"] / variant / case["case_id"] / "dspg.sqlite"
    report_path = out_root / case["suite"] / variant / case["case_id"] / "ingest_report.json"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = out_root / case["suite"] / variant / case["case_id"] / "query_results.json"
    if os.environ.get("DRT_REUSE_BLACKBOX_OUTPUTS", "").lower() in {"1", "true", "yes"} and report_path.exists() and out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        query_results = payload.get("results") if isinstance(payload, dict) and "results" in payload else [payload]
        return {"ingest": json.loads(report_path.read_text(encoding="utf-8")), "query_results": query_results}
    code, stdout = run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "dspg_ingest_folder.py"),
            "--input-folder",
            str(case["corpus"]),
            "--config",
            str(config),
            "--db",
            str(db_path),
            "--variant",
            ingest_variant,
            "--report",
            str(report_path),
        ],
        timeout=360,
    )
    if code != 0:
        return {"ingest": {"returncode": code, "stdout": stdout}, "query_results": []}
    query_path = case["corpus"].parent / "questions.jsonl"
    query_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "dspg_query.py"),
        "--db",
        str(db_path),
        "--questions-jsonl",
        str(query_path),
        "--config",
        str(config),
        "--output",
        str(out_path),
    ]
    if variant in {"model_query_plan_only", "all_model_assisted"}:
        query_cmd.append("--use-model-query")
    else:
        query_cmd.append("--no-model-query")
    qcode, qstdout = run_cmd(query_cmd, timeout=240)
    if qcode != 0:
        return {"ingest": json.loads(report_path.read_text(encoding="utf-8")), "query": {"returncode": qcode, "stdout": qstdout}, "query_results": []}
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    query_results = payload.get("results") if isinstance(payload, dict) and "results" in payload else [payload]
    return {"ingest": json.loads(report_path.read_text(encoding="utf-8")), "query_results": query_results}


def score_case(case: dict[str, Any], variant: str, result: dict[str, Any]) -> dict[str, Any]:
    oracle_answers = case["oracle"]["answers"]
    query_scores = []
    by_id = {r.get("id"): r for r in result.get("query_results", [])}
    for q in case["questions"]:
        expected = [norm(x) for x in oracle_answers.get(q["id"], [])]
        got_item = by_id.get(q["id"], {})
        answers = got_item.get("answers", [])
        actual = [norm(a.get("answer", "")) for a in answers]
        exact = set(actual) == set(expected)
        source_grounded = all(a.get("answer") == "unknown" or bool(a.get("evidence")) for a in answers)
        query_scores.append({"id": q["id"], "question": q["question"], "expected": expected, "actual": actual, "exact": exact, "source_grounded": source_grounded, "plan": got_item.get("plan")})
    ingest = result.get("ingest", {})
    totals = ingest.get("totals", {}) if isinstance(ingest, dict) else {}
    return {
        "suite": case["suite"],
        "case_id": case["case_id"],
        "category": case["oracle"]["category"],
        "component": case["oracle"]["component"],
        "variant": variant,
        "query_scores": query_scores,
        "query_exact": all(q["exact"] for q in query_scores),
        "source_grounded": all(q["source_grounded"] for q in query_scores),
        "request_failed": int(totals.get("request_failed", 0) or 0),
        "truncated": int(totals.get("truncated", 0) or 0),
        "schema_invalid": int(totals.get("schema_invalid", 0) or 0),
        "model_calls": int(totals.get("model_calls", 0) or 0),
    }


def evaluate(suites: list[str], variants: list[str], config: Path) -> dict[str, Any]:
    LOGS.mkdir(parents=True, exist_ok=True)
    results = []
    for suite in suites:
        for case in load_cases(suite):
            for variant in variants:
                result = query_case(case, variant, config, LOGS)
                score = score_case(case, variant, result)
                results.append(score)
    summary: dict[str, Any] = {"variants": {}, "suites": suites, "results": results}
    for variant in variants:
        rows = [r for r in results if r["variant"] == variant]
        summary["variants"][variant] = {
            "cases": len(rows),
            "exact_cases": sum(1 for r in rows if r["query_exact"]),
            "all_exact": all(r["query_exact"] for r in rows) if rows else False,
            "all_source_grounded": all(r["source_grounded"] for r in rows) if rows else False,
            "request_failed": sum(r["request_failed"] for r in rows),
            "truncated": sum(r["truncated"] for r in rows),
            "schema_invalid": sum(r["schema_invalid"] for r in rows),
            "model_calls": sum(r["model_calls"] for r in rows),
        }
    deterministic = {(r["suite"], r["case_id"]): r for r in results if r["variant"] == "deterministic_only"}
    model_wins = []
    component_wins = defaultdict(list)
    category_wins = defaultdict(list)
    for row in results:
        if row["variant"] in {"deterministic_only", "one_shot_baseline"}:
            continue
        det = deterministic.get((row["suite"], row["case_id"]))
        if det and not det["query_exact"] and row["query_exact"] and row["source_grounded"]:
            win = {"suite": row["suite"], "case_id": row["case_id"], "category": row["category"], "component": row["component"], "variant": row["variant"]}
            model_wins.append(win)
            category_wins[row["category"]].append(win)
            component_wins[row["component"]].append(win)
    required_components = {"frame", "scope", "identity", "query"}
    required_categories = {"people", "customer", "artifact", "content"}
    summary["model_wins"] = model_wins
    summary["component_wins"] = dict(component_wins)
    summary["category_wins"] = dict(category_wins)
    summary["component_value_passed"] = all(component_wins.get(c) for c in required_components)
    summary["category_value_passed"] = all(category_wins.get(c) for c in required_categories)
    return summary


def write_reports(summary: dict[str, Any]) -> None:
    (ROOT / "logs" / "PHASE3_BLACKBOX_RESULTS.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Phase 3 Black-Box Results", ""]
    for variant, data in summary["variants"].items():
        lines.append(f"- `{variant}`: exact `{data['exact_cases']}/{data['cases']}`, source_grounded=`{data['all_source_grounded']}`, request_failed=`{data['request_failed']}`, truncated=`{data['truncated']}`, schema_invalid=`{data['schema_invalid']}`")
    lines += ["", "## Component Model Wins"]
    for component, wins in sorted(summary["component_wins"].items()):
        lines.append(f"- `{component}`: {len(wins)} wins")
    lines += ["", "## Category Model Wins"]
    for category, wins in sorted(summary["category_wins"].items()):
        lines.append(f"- `{category}`: {len(wins)} wins")
    (ROOT / "logs" / "PHASE3_BLACKBOX_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def guard_solver_imports() -> dict[str, Any]:
    solver_files = [
        ROOT / "dspg_store.py",
        ROOT / "scripts" / "dspg_ingest_folder.py",
        ROOT / "scripts" / "dspg_query.py",
    ]
    banned = ["generate_blackbox_corpora", "run_blackbox_evaluation", "blackbox_oracles", "oracle.json"]
    findings = []
    for path in solver_files:
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                findings.append({"path": str(path.relative_to(ROOT)), "token": token})
    return {"passed": not findings, "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "config" / "dspg_system.yaml"))
    args = parser.parse_args()
    suites = args.suite or ["dev"]
    variants = args.variant or (VARIANTS if args.all else ["deterministic_only", "all_model_assisted"])
    guard = guard_solver_imports()
    if not guard["passed"]:
        (ROOT / "logs" / "PHASE3_BLACKBOX_RESULTS.json").write_text(json.dumps({"guard": guard}, indent=2), encoding="utf-8")
        print(json.dumps({"passed": False, "guard": guard}, indent=2))
        return 1
    summary = evaluate(suites, variants, Path(args.config))
    summary["guard"] = guard
    write_reports(summary)
    all_model = summary["variants"].get("all_model_assisted", {})
    one_shot = summary["variants"].get("one_shot_baseline", {"exact_cases": 0})
    passed = (
        bool(all_model.get("all_exact"))
        and bool(all_model.get("all_source_grounded"))
        and all_model.get("request_failed", 0) == 0
        and all_model.get("truncated", 0) == 0
        and all_model.get("schema_invalid", 0) == 0
        and summary["component_value_passed"]
        and summary["category_value_passed"]
        and one_shot.get("exact_cases", 0) < all_model.get("exact_cases", 0)
    )
    print(json.dumps({"passed": passed, "suites": suites, "variants": variants, "component_value": summary["component_value_passed"], "category_value": summary["category_value_passed"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
