from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [
    "deterministic_only",
    "model_mention_type_only",
    "model_frame_only",
    "model_scope_only",
    "model_identity_only",
    "model_query_plan_only",
    "model_context_only",
    "all_model_assisted",
    "one_shot_baseline",
]
COMPONENT_VARIANT = {
    "mention_type": "model_mention_type_only",
    "frame": "model_frame_only",
    "scope": "model_scope_only",
    "identity": "model_identity_only",
    "query": "model_query_plan_only",
    "context": "model_context_only",
}
REQUIRED_COMPONENTS = set(COMPONENT_VARIANT)
REQUIRED_CATEGORIES = {"people", "customer", "artifact", "content"}

FIRST = ["Ari", "Blair", "Casey", "Devon", "Elliot", "Finley", "Gray", "Harper", "Indra", "Jules", "Kai", "Logan", "Morgan", "Quinn", "Remy", "Sage", "Tobin", "Vale"]
LAST = ["Aster", "Bryn", "Cato", "Dane", "Ellis", "Fenn", "Gale", "Holt", "Iver", "Joss", "Kerr", "Lane", "Moss", "Noll", "Pike", "Rune", "Shaw", "Vale"]
ROOTS = ["Aster", "Boreal", "Cedar", "Delta", "Ember", "Fjord", "Granite", "Harbor", "Iris", "Juniper", "Keystone", "Lumen", "Meridian", "Northstar"]
SUFFIX = ["Travel", "Freight", "Health", "Systems", "Foods", "Logistics", "Rail", "Media"]
MODULES = ["auth", "session", "ledger", "billing", "cache", "baggage", "portal", "profile"]
DOMAINS = ["docs.example", "review.example", "support.example", "kb.example"]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def person_like_customer(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def company(rng: random.Random) -> str:
    return f"{rng.choice(ROOTS)} {rng.choice(SUFFIX)}"


def ids(rng: random.Random) -> dict[str, str]:
    module = rng.choice(MODULES)
    return {
        "pr": f"PR-{rng.randint(1200, 9999)}",
        "bug": f"BUG-{rng.randint(1200, 9999)}",
        "ticket": f"SUP-{rng.randint(1200, 9999)}",
        "file": f"{module}_{rng.choice(['guard', 'router', 'worker', 'ledger', 'cache'])}.cpp",
        "url": f"https://{rng.choice(DOMAINS)}/{module}/{rng.randint(1000,9999)}/guide",
        "title": f"{module} recovery field manual",
    }


def base_cases(seed: int, suite: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []

    a = ids(rng); cust = person_like_customer(rng); observer = person(rng)
    cases.append({
        "case_id": f"{suite}_mention_type_customer",
        "category": "customer",
        "component": "mention_type",
        "files": {
            "support/escalation.txt": f"""
            Support desk note:
            The customer account named {cust} escalated {a['bug']} through {a['ticket']}.
            Employee {observer} only watched the support queue and was not the customer.
            """,
        },
        "questions": [{"id": "q1", "question": f"Which customer escalated {a['bug']}?"}],
        "answers": {"q1": [cust]},
    })

    b = ids(rng); author = person(rng); watcher = person(rng)
    cases.append({
        "case_id": f"{suite}_frame_implicit_author",
        "category": "people",
        "component": "frame",
        "files": {
            "engineering/change_packet.md": f"""
            Change packet:
            {author} was the engineer behind the repair package for {b['pr']} and {b['bug']}.
            {watcher} attended the review call but did not own the change.
            """,
        },
        "questions": [{"id": "q1", "question": f"Who authored {b['pr']}?"}],
        "answers": {"q1": [author]},
    })

    c = ids(rng); actor = person(rng)
    cases.append({
        "case_id": f"{suite}_scope_legal_allegation",
        "category": "content",
        "component": "scope",
        "files": {
            "legal/allegation.txt": f"""
            Legal intake:
            The allegation claims that {actor} deleted {c['file']} during the outage.
            The operations journal has no independent asserted deletion for {c['file']}.
            """,
        },
        "questions": [{"id": "q1", "question": f"Is the deletion of {c['file']} asserted or alleged?"}],
        "answers": {"q1": ["reported"]},
    })

    d = ids(rng); owner = person(rng); other = person(rng)
    role = rng.choice(["release captain", "incident owner", "deployment lead"])
    cases.append({
        "case_id": f"{suite}_identity_role_name",
        "category": "people",
        "component": "identity",
        "files": {
            "meetings/ownership.txt": f"""
            Meeting note:
            The {role} repaired {d['bug']} before the handoff.
            Later, the notes explicitly say that {owner} is the {role}.
            {other} asked a scheduling question only.
            """,
        },
        "questions": [{"id": "q1", "question": f"Are the {role} and {owner} the same person?"}],
        "answers": {"q1": ["same"]},
    })

    e = ids(rng); distractor = ids(rng)
    cases.append({
        "case_id": f"{suite}_query_artifact_title",
        "category": "artifact",
        "component": "query",
        "files": {
            "docs/index.md": f"""
            Documentation index:
            The field manual titled '{e['title']}' points to {e['url']}.
            A child story mentions a pretend manual and this unrelated URL: {distractor['url']}.
            """,
        },
        "questions": [{"id": "q1", "question": f"Where is {e['title']} cataloged?"}],
        "answers": {"q1": [e['url']]},
    })

    f = ids(rng)
    cases.append({
        "case_id": f"{suite}_content_final_state",
        "category": "content",
        "component": "query",
        "files": {
            "incidents/timeline.txt": f"""
            Incident timeline:
            Monday: {f['bug']} was closed after the temporary fix.
            Tuesday: a quoted training example said {f['bug']} reopened, but that quote was fictional.
            Wednesday: {f['bug']} was put back into service after verification.
            """,
        },
        "questions": [{"id": "q1", "question": f"After the timeline, how is {f['bug']} left?"}],
        "answers": {"q1": ["open"]},
    })

    g = ids(rng); builder = person(rng)
    cases.append({
        "case_id": f"{suite}_context_separation_fiction",
        "category": "content",
        "component": "scope",
        "files": {
            "engineering/drawing_note.txt": f"Real engineering note: {builder} filed the construction drawing at {g['url']} for {g['bug']}.",
            "stories/homework.txt": f"Child story: a wizard drew a fantasy construction drawing for {g['bug']} but it was only fiction and had no real URL.",
        },
        "questions": [{"id": "q1", "question": f"Which URL contains the real construction drawing for {g['bug']}?"}],
        "answers": {"q1": [g['url']]},
    })
    h = ids(rng)
    measure_year = str(rng.randint(1980, 1995))
    modified_year = "2010"
    cases.append({
        "case_id": f"{suite}_context_table_measurement",
        "category": "content",
        "component": "context",
        "file_mtimes": {"tables/clearance_table.txt": f"{modified_year}-06-01T12:00:00"},
        "files": {
            "tables/clearance_table.txt": f"""
            Source metadata note: this exported file was modified by the filesystem in {modified_year}.
            Pump clearance table.
            Table caption: clearance measurements from {measure_year}.
            Row A: {h['file']} clearance = 4.2 mm.
            Unknown validity after export; do not treat file modified time as measurement time.
            """,
        },
        "questions": [
            {"id": "q1", "question": "What measurement year governs the clearance table?"},
            {"id": "q2", "question": "What year does the source modified time show?"},
        ],
        "answers": {"q1": [measure_year], "q2": [modified_year]},
    })

    i = ids(rng); author = person(rng); reviewer = person(rng); approver = person(rng)
    cases.append({
        "case_id": f"{suite}_reviewer_author_approver_separation",
        "category": "people",
        "component": "frame",
        "files": {
            "reviews/separation.md": f"""
            Review board:
            {author} assembled {i['pr']} for {i['bug']}.
            {reviewer} looked over {i['pr']} and left review notes.
            {approver} endorsed {i['pr']} after the test window.
            """,
        },
        "questions": [{"id": "q1", "question": f"Who looked over {i['pr']}?"}],
        "answers": {"q1": [reviewer]},
    })

    j = ids(rng); cust = company(rng); observer = person(rng)
    cases.append({
        "case_id": f"{suite}_customer_refund_request",
        "category": "customer",
        "component": "frame",
        "files": {
            "support/refund_note.txt": f"""
            Support note:
            The customer {cust} requested a refund after {j['bug']} impacted the checkout lane.
            {observer} observed the thread but did not request the refund.
            """,
        },
        "questions": [{"id": "q1", "question": f"Which customer requested a refund after {j['bug']}?"}],
        "answers": {"q1": [cust]},
    })

    k = ids(rng); other = ids(rng)
    cases.append({
        "case_id": f"{suite}_ticket_issue_crosswalk",
        "category": "artifact",
        "component": "query",
        "files": {
            "support/crosswalk.tsv": f"""
            ticket\tissue\tnote
            {k['ticket']}\t{k['bug']}\tprimary escalation mapping
            {other['ticket']}\t{other['bug']}\tdistractor mapping
            """,
        },
        "questions": [{"id": "q1", "question": f"Which ticket links to {k['bug']}?"}],
        "answers": {"q1": [k['ticket']]},
    })

    l = ids(rng); doc_author = person(rng); doc_reviewer = person(rng)
    doc_title = f"{rng.choice(ROOTS)} Relay Stability Memo"
    cases.append({
        "case_id": f"{suite}_document_role_metadata",
        "category": "people",
        "component": "query",
        "files": {
            "documents/stability_memo.md": f"""
            Document control:
            Author for {doc_title}: {doc_author}
            Reviewers for {doc_title}: {doc_reviewer}
            The memo references {l['url']} for appendices.
            """,
        },
        "questions": [
            {"id": "q1", "question": f"Who authored the {doc_title}?"},
            {"id": "q2", "question": f"Who reviewed the {doc_title}?"},
        ],
        "answers": {"q1": [doc_author], "q2": [doc_reviewer]},
    })

    m = ids(rng); cust = company(rng); engineer = person(rng)
    cases.append({
        "case_id": f"{suite}_multihop_customer_to_fix_author",
        "category": "people",
        "component": "query",
        "files": {
            "support/customer_thread.txt": f"The account {cust} raised {m['bug']} during a billing import.",
            "engineering/fix_link.txt": f"{m['pr']} repaired {m['bug']} and restored the billing import path.",
            "reviews/ownership.txt": f"{engineer} carried {m['pr']} through implementation.",
        },
        "questions": [{"id": "q1", "question": f"Which engineer carried the fix for customer {cust}?"}],
        "answers": {"q1": [engineer]},
    })

    n = ids(rng); live_url = n["url"]; old_url = ids(rng)["url"]
    cases.append({
        "case_id": f"{suite}_corrected_artifact_link",
        "category": "artifact",
        "component": "query",
        "files": {
            "docs/correction.md": f"""
            Initial note: a stale draft referenced {old_url}.
            Correction: the live recovery guide for {n['bug']} points to {live_url}.
            """,
        },
        "questions": [{"id": "q1", "question": f"Which URL is the live recovery guide for {n['bug']}?"}],
        "answers": {"q1": [live_url]},
    })

    nr = ids(rng)
    cases.append({
        "case_id": f"{suite}_artifact_indirect_guide_request",
        "category": "artifact",
        "component": "query",
        "files": {
            "docs/guide_locator.md": f"The live recovery guide for {nr['bug']} points to {nr['url']}.",
        },
        "questions": [{"id": "q1", "question": f"Point me to the live recovery guide for {nr['bug']}."}],
        "answers": {"q1": [nr["url"]]},
    })

    o = ids(rng)
    cases.append({
        "case_id": f"{suite}_temporal_quote_distractor_final_state",
        "category": "content",
        "component": "scope",
        "files": {
            "incidents/state_with_quote.txt": f"""
            Monday: {o['bug']} reopened after the first patch.
            Tuesday: {o['bug']} closed after verification.
            Wednesday training fiction says "{o['bug']} reopened again", but the quoted fiction is not operational evidence.
            """,
        },
        "questions": [{"id": "q1", "question": f"At the end of the timeline, what state is {o['bug']} left in?"}],
        "answers": {"q1": ["closed"]},
    })

    p = ids(rng); standby = person(rng)
    cases.append({
        "case_id": f"{suite}_unresolved_owner_unknown",
        "category": "content",
        "component": "query",
        "files": {
            "triage/unowned.txt": f"""
            Triage board:
            {p['bug']} remains open.
            No owner has been assigned for {p['bug']} yet.
            {standby} is only watching the queue.
            """,
        },
        "questions": [{"id": "q1", "question": f"Who owns {p['bug']}?"}],
        "answers": {"q1": ["unknown"]},
    })

    q = ids(rng); first = person(rng); same_first = first.split()[0] + " " + rng.choice([last for last in LAST if last != first.split()[-1]])
    cases.append({
        "case_id": f"{suite}_same_first_name_reviewers",
        "category": "people",
        "component": "identity",
        "files": {
            "reviews/same_first_names.txt": f"""
            Review roster:
            {first} reviewed {q['pr']}.
            {same_first} commented on an unrelated change and did not review {q['pr']}.
            """,
        },
        "questions": [{"id": "q1", "question": f"Who reviewed {q['pr']}?"}],
        "answers": {"q1": [first]},
    })
    return cases


def replace_case(case: dict[str, Any], rng: random.Random, suite: str) -> dict[str, Any]:
    text = json.dumps(case, ensure_ascii=False)
    replacements: dict[str, str] = {}
    for token in sorted(set(re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text))):
        replacements[token] = person(rng)
    for token in sorted(set(re.findall(r"\b(?:PR|BUG|SUP)-\d+\b", text))):
        prefix = token.split("-")[0]
        replacements[token] = f"{prefix}-{rng.randint(2000, 9999)}"
    for token in sorted(set(re.findall(r"https://[^\s\"']+", text))):
        replacements[token] = f"https://{rng.choice(DOMAINS)}/{rng.choice(MODULES)}/{rng.randint(2000,9999)}/guide"
    for token in sorted(set(re.findall(r"\b[A-Za-z0-9_]+\.cpp\b", text))):
        replacements[token] = f"{rng.choice(MODULES)}_{rng.choice(['guard','router','worker'])}.cpp"
    for old, new in replacements.items():
        text = text.replace(old, new)
    out = json.loads(text)
    out["case_id"] = re.sub(r"^[^_]+_", f"{suite}_", out["case_id"])
    return out


def permute_case(case: dict[str, Any], rng: random.Random, suite: str) -> dict[str, Any]:
    out = json.loads(json.dumps(case, ensure_ascii=False))
    out["case_id"] = re.sub(r"^[^_]+_", f"{suite}_", out["case_id"])
    new_files = {}
    for key, text in out["files"].items():
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        rng.shuffle(lines)
        prefix = rng.choice([
            "Distractor: quoted training text is not asserted evidence.",
            "Distractor: a child story may reuse engineering words without becoming a source artifact.",
            "Distractor: unrelated support examples are ignored unless tied to the target identifier.",
        ])
        new_files[key] = "\n".join([prefix] + lines)
    out["files"] = new_files
    return out


def paraphrase_case(case: dict[str, Any], suite: str) -> dict[str, Any]:
    out = json.loads(json.dumps(case, ensure_ascii=False))
    out["case_id"] = re.sub(r"^[^_]+_", f"{suite}_", out["case_id"])
    for q in out["questions"]:
        text = q["question"]
        text = text.replace("Which customer escalated", "Which account raised")
        text = text.replace("Who authored", "Which engineer carried")
        text = text.replace("Is the deletion of", "What is the assertion status for deleting")
        text = text.replace("Are the", "Do records identify the")
        text = text.replace("Where is", "Which URL has")
        text = text.replace("After the timeline, how is", "At the end of the timeline, what state is")
        text = text.replace("Which URL contains the real", "Where is the asserted technical")
        q["question"] = text
    return out


def write_case(root: Path, oracle_root: Path, suite: str, case: dict[str, Any]) -> None:
    case_dir = root / suite / case["case_id"]
    corpus = case_dir / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for rel, text in case["files"].items():
        path = corpus / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.strip() + "\n", encoding="utf-8")
        mtime = (case.get("file_mtimes") or {}).get(rel)
        if mtime:
            ts = time.mktime(time.strptime(mtime, "%Y-%m-%dT%H:%M:%S"))
            os.utime(path, (ts, ts))
    (case_dir / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in case["questions"]), encoding="utf-8")
    odir = oracle_root / suite / case["case_id"]
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "oracle.json").write_text(json.dumps({"case_id": case["case_id"], "category": case["category"], "component": case["component"], "answers": case["answers"]}, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_waves(run_dir: Path, seed: int) -> dict[str, Any]:
    root = run_dir / "blackbox"
    oracle_root = run_dir / "blackbox_oracles"
    if root.exists():
        shutil.rmtree(root)
    if oracle_root.exists():
        shutil.rmtree(oracle_root)
    rng = random.Random(seed)
    base = base_cases(rng.randint(1, 10_000_000), "hidden")
    required_base = [c for c in base if c["component"] in REQUIRED_COMPONENTS]
    context_cases = [c for c in base_cases(rng.randint(1, 10_000_000), "context") if "context_separation" in c["case_id"]]
    waves = {
        "hidden": base,
        "mutation": [replace_case(c, random.Random(rng.randint(1, 10_000_000)), "mutation") for c in required_base],
        "permutation": [permute_case(c, random.Random(rng.randint(1, 10_000_000)), "permutation") for c in required_base],
        "paraphrase": [paraphrase_case(c, "paraphrase") for c in required_base],
        "context": context_cases,
    }
    manifest = {"seed": seed, "root": str(root), "oracle_root": str(oracle_root), "suites": {}}
    for suite, cases in waves.items():
        manifest["suites"][suite] = []
        for case in cases:
            write_case(root, oracle_root, suite, case)
            manifest["suites"][suite].append(case["case_id"])
    templates = {
        "semantic_cue_templates": [
            "customer account named {customer} escalated {bug}",
            "engineer behind the repair package for {pr}",
            "allegation claims that {person} deleted {file}",
            "notes explicitly say that {person} is the role description",
            "field manual titled '{title}' points to {url}",
            "put back into service after verification",
            "fictional child story distractor",
            "table caption says measurements from {year}",
            "file modified time differs from measurement time",
            "reviewer looked over {pr} while approver endorsed it",
            "customer requested a refund after {bug}",
            "ticket issue crosswalk maps {ticket} to {bug}",
            "document control lists author and reviewers for a memo",
            "customer raised {bug}, {pr} repaired it, engineer carried {pr}",
            "correction replaces a stale guide URL with a live guide URL",
            "indirect guide location query asks to point to a URL",
            "quoted fiction mentions a later state that must not become asserted",
            "no owner assigned yet means ownership is unknown",
        ]
    }
    (run_dir / "phase4_generator_templates.json").write_text(json.dumps(templates, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def run_cmd(cmd: list[str], timeout: int = 420) -> tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("DRT_DISABLE_CACHE", "1")
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, env=env)
    return proc.returncode, proc.stdout


def model_endpoint() -> str:
    return os.environ.get("DRT_PHASE4_MODEL_ENDPOINT", "http://127.0.0.1:14829")


def one_shot_answer(corpus: Path, question: str) -> dict[str, Any]:
    texts = []
    for path in sorted(corpus.rglob("*")):
        if path.is_file():
            texts.append(f"FILE {path.relative_to(corpus)}\n{path.read_text(encoding='utf-8', errors='replace')}")
    prompt = "Answer from source text. JSON only {\"answer\":\"...\"}. Use unknown if insufficient.\n" + json.dumps({"sources": texts, "question": question}, ensure_ascii=False)
    grammar = 'root ::= "{" ws "\\"answer\\"" ws ":" ws string ws "}"\nstring ::= "\\"" chars "\\""\nchars ::= ([^"\\\\] | "\\\\" ["\\\\/bfnrt])*\nws ::= [ \\t\\n\\r]*'
    body = {"prompt": prompt, "n_predict": 160, "temperature": 0.0, "top_p": 1.0, "stream": False, "grammar": grammar}
    try:
        req = urllib.request.Request(model_endpoint() + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        raw = data.get("content", "")
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1]) if "{" in raw and "}" in raw else {}
        return {"answer": obj.get("answer", "unknown"), "raw": raw, "accepted": bool(obj), "evidence": []}
    except Exception as exc:
        return {"answer": "unknown", "accepted": False, "error": str(exc), "evidence": []}


def load_cases(manifest: dict[str, Any], suites: list[str]) -> list[dict[str, Any]]:
    root = Path(manifest["root"])
    oracle_root = Path(manifest["oracle_root"])
    cases = []
    for suite in suites:
        for case_id in manifest["suites"].get(suite, []):
            case_dir = root / suite / case_id
            oracle = json.loads((oracle_root / suite / case_id / "oracle.json").read_text(encoding="utf-8"))
            questions = [json.loads(line) for line in (case_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            cases.append({"suite": suite, "case_id": case_id, "corpus": case_dir / "corpus", "questions": questions, "oracle": oracle})
    return cases


def run_case(case: dict[str, Any], variant: str, config: Path, out_root: Path) -> dict[str, Any]:
    case_out = out_root / case["suite"] / variant / case["case_id"]
    case_out.mkdir(parents=True, exist_ok=True)
    if variant == "one_shot_baseline":
        results = []
        for q in case["questions"]:
            ans = one_shot_answer(case["corpus"], q["question"])
            results.append({"id": q["id"], "question": q["question"], "answers": [{"answer": ans["answer"], "evidence": []}], "plan": {"source": "one_shot"}, "status": "answered" if ans["answer"] != "unknown" else "unknown", "one_shot": ans})
        return {"ingest": {"totals": {"model_calls": len(results), "request_failed": 0, "truncated": 0, "schema_invalid": 0}}, "query_results": results}
    ingest_variant = "deterministic_only" if variant == "model_query_plan_only" else variant
    db = case_out / "dspg.sqlite"
    ingest_report = case_out / "ingest_report.json"
    query_out = case_out / "query_results.json"
    code, stdout = run_cmd([sys.executable, str(ROOT / "scripts" / "dspg_ingest_folder.py"), "--input-folder", str(case["corpus"]), "--config", str(config), "--db", str(db), "--variant", ingest_variant, "--report", str(ingest_report)], timeout=480)
    if code != 0:
        return {"ingest": {"returncode": code, "stdout": stdout}, "query_results": []}
    query_cmd = [sys.executable, str(ROOT / "scripts" / "dspg_query.py"), "--db", str(db), "--questions-jsonl", str(case["corpus"].parent / "questions.jsonl"), "--config", str(config), "--output", str(query_out)]
    query_cmd.append("--use-model-query" if variant in {"model_query_plan_only", "all_model_assisted"} else "--no-model-query")
    qcode, qstdout = run_cmd(query_cmd, timeout=360)
    if qcode != 0:
        return {"ingest": json.loads(ingest_report.read_text(encoding="utf-8")), "query": {"returncode": qcode, "stdout": qstdout}, "query_results": []}
    payload = json.loads(query_out.read_text(encoding="utf-8"))
    return {"ingest": json.loads(ingest_report.read_text(encoding="utf-8")), "query_results": payload.get("results", [payload])}


def score_case(case: dict[str, Any], variant: str, result: dict[str, Any]) -> dict[str, Any]:
    oracle = case["oracle"]
    by_id = {r.get("id"): r for r in result.get("query_results", [])}
    q_scores = []
    for q in case["questions"]:
        expected = [norm(x) for x in oracle["answers"].get(q["id"], [])]
        item = by_id.get(q["id"], {})
        answers = item.get("answers", [])
        actual = [norm(a.get("answer", "")) for a in answers]
        exact = set(actual) == set(expected)
        source_grounded = all(a.get("answer") == "unknown" or bool(a.get("evidence")) for a in answers)
        q_scores.append({"id": q["id"], "question": q["question"], "expected": expected, "actual": actual, "exact": exact, "source_grounded": source_grounded, "plan": item.get("plan")})
    ingest = result.get("ingest", {})
    totals = ingest.get("totals", {}) if isinstance(ingest, dict) else {}
    table_counts = ingest.get("table_counts", {}) if isinstance(ingest, dict) else {}
    return {
        "suite": case["suite"],
        "case_id": case["case_id"],
        "category": oracle["category"],
        "component": oracle["component"],
        "variant": variant,
        "query_scores": q_scores,
        "query_exact": all(q["exact"] for q in q_scores),
        "source_grounded": all(q["source_grounded"] for q in q_scores),
        "request_failed": int(totals.get("request_failed", 0) or 0),
        "truncated": int(totals.get("truncated", 0) or 0),
        "schema_invalid": int(totals.get("schema_invalid", 0) or 0),
        "model_calls": int(totals.get("model_calls", 0) or 0),
        "db_path": ingest.get("db_path"),
        "schema_valid": bool((ingest.get("schema") or {}).get("valid", False)) if variant != "one_shot_baseline" else True,
        "table_counts": table_counts,
    }


def evaluate(manifest: dict[str, Any], suites: list[str], config: Path, out_root: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    cases = load_cases(manifest, suites)
    results = []
    for case in cases:
        needed_variants = {"deterministic_only", "all_model_assisted"}
        if case["suite"] == "hidden":
            needed_variants.add("one_shot_baseline")
        component_variant = COMPONENT_VARIANT.get(case["oracle"]["component"])
        if component_variant:
            needed_variants.add(component_variant)
        for variant in [v for v in VARIANTS if v in needed_variants]:
            result = run_case(case, variant, config, out_root)
            results.append(score_case(case, variant, result))
    summary: dict[str, Any] = {"suites": suites, "variants": {}, "results": results, "manifest": manifest}
    for variant in VARIANTS:
        rows = [r for r in results if r["variant"] == variant]
        summary["variants"][variant] = {
            "cases": len(rows),
            "exact_cases": sum(1 for r in rows if r["query_exact"]),
            "source_grounded_cases": sum(1 for r in rows if r["source_grounded"]),
            "all_exact": all(r["query_exact"] for r in rows) if rows else False,
            "all_source_grounded": all(r["source_grounded"] for r in rows) if rows else False,
            "request_failed": sum(r["request_failed"] for r in rows),
            "truncated": sum(r["truncated"] for r in rows),
            "schema_invalid": sum(r["schema_invalid"] for r in rows),
            "model_calls": sum(r["model_calls"] for r in rows),
            "schema_valid": all(r["schema_valid"] for r in rows) if rows else False,
        }
    det = {(r["suite"], r["case_id"]): r for r in results if r["variant"] == "deterministic_only"}
    allm = {(r["suite"], r["case_id"]): r for r in results if r["variant"] == "all_model_assisted"}
    comp_wins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cat_wins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        component = row["component"]
        wanted_variant = COMPONENT_VARIANT.get(component)
        if row["variant"] != wanted_variant:
            continue
        drow = det.get((row["suite"], row["case_id"]))
        arow = allm.get((row["suite"], row["case_id"]))
        if drow and arow and not drow["query_exact"] and row["query_exact"] and row["source_grounded"] and arow["query_exact"] and arow["source_grounded"]:
            win = {"suite": row["suite"], "case_id": row["case_id"], "category": row["category"], "component": component, "component_variant": row["variant"]}
            comp_wins[component].append(win)
            cat_wins[row["category"]].append(win)
    summary["component_wins"] = dict(comp_wins)
    summary["category_wins"] = dict(cat_wins)
    summary["component_value_passed"] = all(comp_wins.get(c) for c in REQUIRED_COMPONENTS)
    summary["category_value_passed"] = all(cat_wins.get(c) for c in REQUIRED_CATEGORIES)
    summary["coverage"] = {
        "required_components": sorted(REQUIRED_COMPONENTS),
        "required_categories": sorted(REQUIRED_CATEGORIES),
        "covered_components": sorted(comp_wins),
        "covered_categories": sorted(cat_wins),
    }
    one = summary["variants"].get("one_shot_baseline", {})
    all_variant = summary["variants"].get("all_model_assisted", {})
    summary["one_shot_inferior"] = int(one.get("source_grounded_cases", 0)) < int(all_variant.get("source_grounded_cases", -1))
    return summary


def write_blackbox_reports(summary: dict[str, Any]) -> None:
    (ROOT / "logs" / "PHASE4_BLACKBOX_RESULTS.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Phase 4 Black-Box Results", ""]
    for variant, data in summary["variants"].items():
        lines.append(f"- `{variant}`: exact `{data['exact_cases']}/{data['cases']}`, source_grounded `{data['source_grounded_cases']}/{data['cases']}`, model_calls `{data['model_calls']}`, request_failed `{data['request_failed']}`, truncated `{data['truncated']}`, schema_invalid `{data['schema_invalid']}`")
    lines += ["", "## Component-Only Wins"]
    for comp in sorted(REQUIRED_COMPONENTS):
        lines.append(f"- `{comp}`: `{len(summary['component_wins'].get(comp, []))}`")
    lines += ["", "## HERB-Like Category Wins"]
    for cat in sorted(REQUIRED_CATEGORIES):
        lines.append(f"- `{cat}`: `{len(summary['category_wins'].get(cat, []))}`")
    (ROOT / "logs" / "PHASE4_BLACKBOX_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    component = {"passed": summary["component_value_passed"], "wins": summary["component_wins"], "coverage": summary["coverage"]}
    category = {"passed": summary["category_value_passed"], "wins": summary["category_wins"], "coverage": summary["coverage"]}
    (ROOT / "logs" / "PHASE4_COMPONENT_VALUE_PROOF.json").write_text(json.dumps(component, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "logs" / "PHASE4_CATEGORY_VALUE_PROOF.json").write_text(json.dumps(category, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "logs" / "PHASE4_COMPONENT_VALUE_PROOF.md").write_text("# Phase 4 Component Value Proof\n\n" + "\n".join(f"- `{k}`: `{len(component['wins'].get(k, []))}` component-only wins" for k in sorted(REQUIRED_COMPONENTS)) + "\n", encoding="utf-8")
    (ROOT / "logs" / "PHASE4_CATEGORY_VALUE_PROOF.md").write_text("# Phase 4 Category Value Proof\n\n" + "\n".join(f"- `{k}`: `{len(category['wins'].get(k, []))}` source-grounded wins" for k in sorted(REQUIRED_CATEGORIES)) + "\n", encoding="utf-8")
