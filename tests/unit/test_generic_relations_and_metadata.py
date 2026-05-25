from __future__ import annotations

import json

from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.ingest import ingest_folder
from knowmoredirt.relations import extract_relations


def test_generic_relation_extractor_covers_common_discourse_shapes() -> None:
    text = "\n".join(
        [
            "Author: Mira Sol.",
            "record_id|surface|value",
            "row-7|NOTE-314|Mira Sol",
            '{"object":{"status":"ready","count":3},"url":"https://example.invalid/a"}',
            "2026-02-03 shelf state: stable.",
        ]
    )

    relations = []
    for line in text.splitlines():
        relations.extend(extract_relations(line))
    facts = {(item.relation_type, item.predicate, item.subject, item.object, item.value) for item in relations}

    assert ("label_value", "label", "Author", "", "Mira Sol") in facts
    assert any(item.relation_type == "table_cell" and item.subject == "row-7" and item.value == "NOTE-314" for item in relations)
    assert any(item.relation_type == "record_value" and item.subject == "object.status" and item.value == "ready" for item in relations)
    assert any(item.relation_type == "identifier" and item.predicate == "url" and item.value == "https://example.invalid/a" for item in relations)
    assert any(item.relation_type == "temporal" and item.predicate == "timestamp" and item.value == "2026-02-03" for item in relations)


def test_ingest_stores_relations_and_enriched_file_metadata(tmp_path) -> None:
    nested = tmp_path / "r4" / "odd.name"
    nested.parent.mkdir()
    nested.write_text(
        "Subject: garden pump note.\nOwner: Tessa Vale.\n2026-02-03 pump state: repaired.",
        encoding="utf-8",
    )

    store, run_id, documents, _ = ingest_folder(tmp_path)
    counts = store.counts()
    metadata = json.loads(
        store.execute("SELECT metadata_json FROM documents WHERE document_id=?", (documents[0].document_id,)).fetchone()[
            "metadata_json"
        ]
    )

    assert run_id
    assert counts["relations"] >= 3
    assert counts["metadata_records"] >= 10
    assert counts["context_carriers"] >= 3
    assert counts["context_assignments"] >= 1
    assert metadata["file_name"] == "odd.name"
    assert metadata["suffix"] == ".name"
    assert metadata["parent_rel_path"] == "r4"
    assert metadata["line_count"] == 3
    assert metadata["text_quality"]["semantic_quality"] == "meaningful_discourse"

    carrier_row = store.execute(
        "SELECT temporal_value_type FROM context_carriers WHERE document_id=? AND temporal_value_type='file_modified_time'",
        (documents[0].document_id,),
    ).fetchone()
    assert carrier_row is not None


def test_generic_query_answers_new_raw_text_without_fixture_literals(tmp_path) -> None:
    (tmp_path / "vortex").mkdir()
    (tmp_path / "vortex" / "note").write_text(
        "\n".join(
            [
                "Field log for lantern shelf.",
                "NOTE-314 reviewer: Mira Sol.",
                "Shelf state: stable.",
                "Glossary entry: kora / morning bell.",
            ]
        ),
        encoding="utf-8",
    )

    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine.answer("What is the reviewer for NOTE-314?").text == "Mira Sol"
    assert engine.answer("What is the shelf state?").text == "stable"
