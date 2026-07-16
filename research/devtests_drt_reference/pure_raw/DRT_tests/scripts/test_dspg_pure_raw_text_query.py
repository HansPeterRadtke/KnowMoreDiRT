#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dspg_store as store
import scripts.dspg_query as query


def add_doc(con, run_id: str, root: Path, rel: str, text: str) -> tuple[str, str]:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    document_id = store.insert_document(con, run_id, root, path, text)
    chunk_id = store.insert_chunk(con, document_id, 0, 0, len(text), text)
    return document_id, chunk_id


def add_mention(con, run_id: str, document_id: str, chunk_id: str, text: str, surface: str, entity_type: str) -> str:
    start = text.index(surface)
    span_id = store.insert_span(con, document_id, chunk_id, start, start + len(surface), surface, entity_type)
    mention_id = store.stable_id("ment", run_id, document_id, surface, start)
    referent_id = store.stable_id("ref", run_id, document_id, surface, entity_type)
    con.execute(
        """
        INSERT OR REPLACE INTO mentions
        (mention_id, run_id, span_id, surface, surface_norm, mention_kind, entity_type, confidence, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (mention_id, run_id, span_id, surface, store.norm_text(surface), entity_type, entity_type, 1.0, "raw_text_test"),
    )
    con.execute(
        """
        INSERT OR REPLACE INTO referents
        (referent_id, run_id, canonical_label, canonical_label_norm, entity_type, status, attributes_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (referent_id, run_id, surface, store.norm_text(surface), entity_type, "accepted", "{}"),
    )
    con.execute(
        "INSERT OR REPLACE INTO mention_referents(mention_id, referent_id, link_status, confidence) VALUES (?, ?, ?, ?)",
        (mention_id, referent_id, "member", 1.0),
    )
    return mention_id


def add_frame(con, run_id: str, document_id: str, chunk_id: str, text: str, trigger: str, predicate_norm: str, args: list[tuple[str, str]]) -> None:
    start = text.index(trigger)
    span_id = store.insert_span(con, document_id, chunk_id, start, start + len(trigger), trigger, "event")
    context_id = store.stable_id("ctx", run_id, document_id, "asserted")
    con.execute(
        "INSERT OR REPLACE INTO contexts(context_id, run_id, kind, parent_context_id, holder_mention_id, evidence_surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (context_id, run_id, "asserted", None, None, None, 1.0),
    )
    frame_id = store.stable_id("frame", run_id, document_id, trigger, start)
    con.execute(
        """
        INSERT OR REPLACE INTO frames(frame_id, run_id, context_id, predicate, predicate_norm, trigger_surface, confidence, source, span_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (frame_id, run_id, context_id, trigger, predicate_norm, trigger, 1.0, "raw_text_test", span_id),
    )
    for idx, (role, mention_id) in enumerate(args):
        con.execute(
            "INSERT OR REPLACE INTO frame_arguments(argument_id, frame_id, role, mention_id, referent_id, confidence) VALUES (?, ?, ?, ?, ?, ?)",
            (store.stable_id("arg", frame_id, idx, role, mention_id), frame_id, role, mention_id, None, 1.0),
        )


def build_raw_db(base: Path) -> tuple[Path, str]:
    root = base / "raw_folder"
    db_path = base / "pure_raw.sqlite"
    con = store.connect(db_path)
    store.init_db(con)
    run_id = store.insert_run(con, str(root), "raw-test-config", "pure-raw-text", run_id="run_pure_raw_text")
    good_text = (
        "Release log for Heliotrope.\n"
        "PR-8842 was carried by Mira Sol and reviewed by Ivo Renn.\n"
        "Customer Blue Finch reported BUG-521 in Heliotrope.\n"
        "The Heliotrope runbook is https://docs.example/heliotrope/runbook.\n"
        "On Monday PR-8842 was open. On Tuesday PR-8842 was closed.\n"
        "Table: Product | Customer | Issue\nHeliotrope | Blue Finch | BUG-521\n"
    )
    bad_text = (
        "Release log for Lumen.\n"
        "PR-8842 was carried by Wrong Person.\n"
        "The Lumen runbook is https://docs.example/lumen/runbook.\n"
    )
    good_doc, good_chunk = add_doc(con, run_id, root, "engineering/heliotrope_release.txt", good_text)
    bad_doc, bad_chunk = add_doc(con, run_id, root, "engineering/lumen_release.txt", bad_text)
    pr_good = add_mention(con, run_id, good_doc, good_chunk, good_text, "PR-8842", "pr")
    mira = add_mention(con, run_id, good_doc, good_chunk, good_text, "Mira Sol", "person")
    ivo = add_mention(con, run_id, good_doc, good_chunk, good_text, "Ivo Renn", "person")
    customer = add_mention(con, run_id, good_doc, good_chunk, good_text, "Blue Finch", "customer")
    bug = add_mention(con, run_id, good_doc, good_chunk, good_text, "BUG-521", "issue")
    url = add_mention(con, run_id, good_doc, good_chunk, good_text, "https://docs.example/heliotrope/runbook", "url")
    add_frame(con, run_id, good_doc, good_chunk, good_text, "carried", "carried", [("theme", pr_good), ("author", mira)])
    add_frame(con, run_id, good_doc, good_chunk, good_text, "reviewed", "reviewed", [("theme", pr_good), ("reviewer", ivo)])
    add_frame(con, run_id, good_doc, good_chunk, good_text, "reported", "reported", [("customer", customer), ("theme", bug)])
    add_frame(con, run_id, good_doc, good_chunk, good_text, "runbook", "linked", [("theme", url)])
    add_frame(con, run_id, good_doc, good_chunk, good_text, "open", "open", [("theme", pr_good)])
    add_frame(con, run_id, good_doc, good_chunk, good_text, "closed", "closed", [("theme", pr_good)])
    pr_bad = add_mention(con, run_id, bad_doc, bad_chunk, bad_text, "PR-8842", "pr")
    wrong = add_mention(con, run_id, bad_doc, bad_chunk, bad_text, "Wrong Person", "person")
    add_frame(con, run_id, bad_doc, bad_chunk, bad_text, "carried", "carried", [("theme", pr_bad), ("author", wrong)])
    store.finish_run(con, run_id, "completed", {"pure_raw_text": True})
    con.commit()
    con.close()
    return db_path, run_id


def assert_answer(result: dict, expected: str) -> None:
    answers = [item.get("answer") for item in result.get("answers", [])]
    assert expected in answers, {"expected": expected, "answers": answers, "result": result}
    assert all("Wrong Person" != answer for answer in answers), answers


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="dspg_pure_raw_text_") as temp:
        base = Path(temp)
        db_path, run_id = build_raw_db(base)
        config = query.load_config(ROOT / "config" / "dspg_system.yaml")
        con = store.connect(db_path)
        index = query.load_metadata_index(con, run_id, allow_prepared_metadata=False)

        def planned(question: str, _config: dict) -> dict:
            low = question.lower()
            if "carried" in low:
                intent, role, target = "who_author", "author", "PR-8842"
            elif "customer" in low:
                intent, role, target = "which_customer", "customer", "BUG-521"
            elif "runbook" in low:
                intent, role, target = "which_url", "artifact", "runbook Heliotrope"
            else:
                intent, role, target = "final_state", "state", "PR-8842"
            return {"intent": intent, "target_surface": target, "answer_role": role, "requires_asserted": True, "source": "model", "accepted": True, "elapsed": 0.0}

        query.call_model_query_plan = planned
        questions = [
            ("Which engineer carried PR-8842 in Heliotrope?", "Mira Sol"),
            ("Which customer reported BUG-521 in Heliotrope?", "Blue Finch"),
            ("Where is the runbook in Heliotrope?", "https://docs.example/heliotrope/runbook"),
            ("What was the final state of PR-8842 in Heliotrope?", "closed"),
        ]
        for text, expected in questions:
            result = query.run_query(con, text, config, True, run_id, metadata_index=index, item={"question": text}, bounded_doc_limit=5, allow_prepared_metadata=False, commit=False)
            assert result["diagnostics"]["records_scope"] == "bounded", result["diagnostics"]
            assert result["diagnostics"]["model_used"] is True, result["diagnostics"]
            assert_answer(result, expected)
        con.close()

        qpath = base / "questions.jsonl"
        qpath.write_text("\n".join(json.dumps({"id": f"raw_q_{idx}", "question": q}) for idx, (q, _) in enumerate(questions)) + "\n", encoding="utf-8")
        out = base / "out.json"
        progress = base / "progress.jsonl"
        checkpoint = base / "checkpoint.jsonl"
        subprocess.run(
            [
                "python3",
                "-u",
                str(ROOT / "scripts" / "dspg_query.py"),
                "--db",
                str(db_path),
                "--config",
                str(ROOT / "config" / "dspg_system.yaml"),
                "--questions-jsonl",
                str(qpath),
                "--use-model-query",
                "--output",
                str(out),
                "--progress-log",
                str(progress),
                "--checkpoint-jsonl",
                str(checkpoint),
                "--bounded-doc-limit",
                "5",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        assert sum(1 for _ in checkpoint.open(encoding="utf-8")) == 4
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert len(payload["results"]) == 4
        events = [json.loads(line)["event"] for line in progress.read_text(encoding="utf-8").splitlines()]
        assert events.count("checkpoint_write") == 4, events
        print("pure raw-text query tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
