#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dspg_store as store
import scripts.dspg_query as query


def write_doc(root: Path, rel: str, metadata: dict[str, object], body: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False) + "\n--- HERB RAW ARTIFACT TEXT ---\n" + body, encoding="utf-8")
    return path


def add_text_document(con, run_id: str, root: Path, path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8")
    doc_id = store.insert_document(con, run_id, root, path, text)
    chunk_id = store.insert_chunk(con, doc_id, 0, 0, len(text), text)
    return doc_id, chunk_id, text


def add_mention(con, run_id: str, doc_id: str, chunk_id: str, text: str, surface: str, entity_type: str) -> str:
    start = text.index(surface)
    span_id = store.insert_span(con, doc_id, chunk_id, start, start + len(surface), surface, entity_type)
    mention_id = store.stable_id("ment", run_id, doc_id, surface, start)
    con.execute(
        """
        INSERT OR REPLACE INTO mentions
        (mention_id, run_id, span_id, surface, surface_norm, mention_kind, entity_type, confidence, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (mention_id, run_id, span_id, surface, store.norm_text(surface), entity_type, entity_type, 1.0, "test"),
    )
    referent_id = store.stable_id("ref", run_id, doc_id, surface, entity_type)
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


def add_frame(con, run_id: str, doc_id: str, chunk_id: str, text: str, predicate: str, args: list[tuple[str, str]]) -> None:
    trigger_start = text.index(predicate)
    trigger_span = store.insert_span(con, doc_id, chunk_id, trigger_start, trigger_start + len(predicate), predicate, "event")
    context_id = store.stable_id("ctx", run_id, doc_id, "asserted")
    con.execute(
        "INSERT OR REPLACE INTO contexts(context_id, run_id, kind, parent_context_id, holder_mention_id, evidence_surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (context_id, run_id, "asserted", None, None, None, 1.0),
    )
    frame_id = store.stable_id("frame", run_id, doc_id, predicate, trigger_start)
    con.execute(
        """
        INSERT OR REPLACE INTO frames(frame_id, run_id, context_id, predicate, predicate_norm, trigger_surface, confidence, source, span_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (frame_id, run_id, context_id, predicate, store.norm_text(predicate), predicate, 1.0, "test", trigger_span),
    )
    for idx, (role, mention_id) in enumerate(args):
        con.execute(
            "INSERT OR REPLACE INTO frame_arguments(argument_id, frame_id, role, mention_id, referent_id, confidence) VALUES (?, ?, ?, ?, ?, ?)",
            (store.stable_id("arg", frame_id, idx, role, mention_id), frame_id, role, mention_id, None, 1.0),
        )


def build_test_db(base: Path) -> tuple[Path, Path, str]:
    root = base / "folder"
    db = base / "query_path.sqlite"
    con = store.connect(db)
    store.init_db(con)
    run_id = store.insert_run(con, str(root), "test-config", "model-query-path-test", run_id="run_query_path_test")
    write_doc(
        root,
        "products/novapilot/market.txt",
        {
            "artifact_id": "doc_novapilot_market",
            "artifact_type": "document",
            "product_id": "novapilot",
            "product_name": "NovaPilot",
            "source_title": "Market Survey Report",
            "author": "EMP-NOVA-A",
            "employee_ids": ["EMP-NOVA-A", "EMP-NOVA-R"],
        },
        "Document Type: Market Survey Report\nProduct: NovaPilot\nAuthors and reviewers are EMP-NOVA-A and EMP-NOVA-R.",
    )
    write_doc(
        root,
        "products/orbitdesk/market.txt",
        {
            "artifact_id": "doc_orbitdesk_market",
            "artifact_type": "document",
            "product_id": "orbitdesk",
            "product_name": "OrbitDesk",
            "source_title": "Market Survey Report",
            "author": "Document Type",
            "employee_ids": ["EMP-ORBIT-WRONG"],
        },
        "Document Type: Market Survey Report\nProduct: OrbitDesk\nThis is the wrong product distractor.",
    )
    graph_doc = write_doc(
        root,
        "products/novapilot/pr_notes.txt",
        {
            "artifact_id": "note_pr_771",
            "artifact_type": "engineering_note",
            "product_id": "novapilot",
            "product_name": "NovaPilot",
            "source_title": "NovaPilot PR Notes",
            "employee_ids": ["EMP-MIRA"],
        },
        "PR-771 was carried by Mira Sol for NovaPilot.",
    )
    for idx in range(35):
        write_doc(
            root,
            f"distractors/distractor_{idx}.txt",
            {
                "artifact_id": f"doc_distractor_{idx}",
                "artifact_type": "document",
                "product_id": "orbitdesk",
                "product_name": "OrbitDesk",
                "source_title": f"Distractor {idx}",
            },
            f"OrbitDesk distractor document {idx}.",
        )
    for path in sorted(root.rglob("*.txt")):
        doc_id, chunk_id, text = add_text_document(con, run_id, root, path)
        if path == graph_doc:
            pr = add_mention(con, run_id, doc_id, chunk_id, text, "PR-771", "pr")
            person = add_mention(con, run_id, doc_id, chunk_id, text, "Mira Sol", "person")
            add_frame(con, run_id, doc_id, chunk_id, text, "carried", [("theme", pr), ("author", person)])
    store.finish_run(con, run_id, "completed", {"test": True})
    con.commit()
    con.close()
    return db, root, run_id


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="dspg_query_model_path_") as temp:
        base = Path(temp)
        db, _root, run_id = build_test_db(base)
        config = query.load_config(Path(__file__).resolve().parents[1] / "config" / "dspg_system.yaml")
        con = store.connect(db)
        metadata = query.load_metadata_index(con, run_id, allow_prepared_metadata=True)
        model_calls = {"count": 0}

        def wrong_model_plan(_question, _config):
            model_calls["count"] += 1
            return {
                "intent": "who_author",
                "target_surface": "document",
                "answer_role": "author",
                "requires_asserted": True,
                "source": "model",
                "accepted": True,
                "elapsed": 0.0,
            }

        query.call_model_query_plan = wrong_model_plan
        item = {
            "id": "selftest_q_metadata",
            "product_id": "novapilot",
            "product_name": "NovaPilot",
            "question": "Find employee IDs of the authors and key reviewers of the Market Survey Report for the NovaPilot product?",
        }
        result = query.run_query(con, item["question"], config, True, run_id, metadata_index=metadata, item=item, allow_prepared_metadata=True, commit=False)
        answers = {answer["answer"] for answer in result["answers"]}
        assert result["status"] == "answered", result
        assert model_calls["count"] == 0, "metadata guard was incorrectly sent to the model"
        assert "EMP-NOVA-A" in answers and "EMP-NOVA-R" in answers, answers
        assert "EMP-ORBIT-WRONG" not in answers and "Document Type" not in answers, answers
        assert result["diagnostics"]["route"] == "deterministic_metadata_guard", result["diagnostics"]

        def graph_model_plan(_question, _config):
            model_calls["count"] += 1
            return {
                "intent": "who_author",
                "target_surface": "PR-771",
                "answer_role": "author",
                "requires_asserted": True,
                "source": "model",
                "accepted": True,
                "elapsed": 0.0,
            }

        query.call_model_query_plan = graph_model_plan
        graph_item = {
            "id": "selftest_q_graph",
            "product_id": "novapilot",
            "product_name": "NovaPilot",
            "question": "Which engineer carried PR-771 for NovaPilot?",
        }
        graph_result = query.run_query(con, graph_item["question"], config, True, run_id, metadata_index=metadata, item=graph_item, bounded_doc_limit=5, allow_prepared_metadata=True, commit=False)
        assert graph_result["status"] == "answered", graph_result
        assert any(answer["answer"] == "Mira Sol" for answer in graph_result["answers"]), graph_result["answers"]
        assert graph_result["diagnostics"]["records_scope"] == "bounded", graph_result["diagnostics"]
        assert graph_result["diagnostics"]["record_counts"]["documents"] < 10, graph_result["diagnostics"]["record_counts"]
        con.commit()
        con.close()

        questions = base / "questions.jsonl"
        item2 = dict(item)
        item2["id"] = "selftest_q_metadata_2"
        questions.write_text(json.dumps(item) + "\n" + json.dumps(item2) + "\n", encoding="utf-8")
        out = base / "cli_out.json"
        progress = base / "cli_progress.jsonl"
        checkpoint = base / "cli_checkpoint.jsonl"
        subprocess.run(
            [
                "python3",
                "-u",
                str(Path(__file__).resolve().parent / "dspg_query.py"),
                "--db",
                str(db),
                "--config",
                str(Path(__file__).resolve().parents[1] / "config" / "dspg_system.yaml"),
                "--questions-jsonl",
                str(questions),
                "--use-model-query",
                "--output",
                str(out),
                "--progress-log",
                str(progress),
                "--checkpoint-jsonl",
                str(checkpoint),
                "--allow-prepared-metadata",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        assert sum(1 for _ in checkpoint.open(encoding="utf-8")) == 2
        progress_events = [json.loads(line)["event"] for line in progress.read_text(encoding="utf-8").splitlines()]
        assert progress_events.count("db_commit") == 2, progress_events
        assert progress_events.count("checkpoint_write") == 2, progress_events
        print("model query path structural tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
