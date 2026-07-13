from __future__ import annotations
import json
from knowmoredirt.catalog import SourceCatalog, values_at_path


def test_catalog_discovers_json_collections_maps_and_provenance(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "records.json").write_text(
        json.dumps(
            {
                "items": [
                    {"name": "A", "owner": {"id": "E-1"}},
                    {"name": "B", "owner": {"id": "E-2"}},
                ],
                "directory": {"E-1": {"name": "Mara"}, "E-2": {"name": "Omar"}},
            }
        )
    )
    catalog = SourceCatalog(tmp_path)
    item_collection = "nested/records.json::items[]"
    map_collection = "nested/records.json::directory{}"
    assert len(catalog.collection_records(item_collection)) == 2
    assert len(catalog.collection_records(map_collection)) == 2
    first = catalog.collection_records(item_collection)[0]
    assert values_at_path(first.data, "owner.id") == ["E-1"]
    assert values_at_path(first.data, "source.path") == ["nested/records.json"]


def test_catalog_discovers_raw_text_blocks_labels_and_tables(tmp_path):
    (tmp_path / "labels.txt").write_text("Asset: Cedar\nOwner: Mara\nState: ready\n")
    (tmp_path / "table.tsv").write_text("name\tcount\nalpha\t2\nbeta\t5\n")
    catalog = SourceCatalog(tmp_path)
    assert len(catalog.collection_records("logical_documents")) == 2
    assert catalog.collection_records("labels.txt::labeled_records[]")[0].data["Owner"] == "Mara"
    table = catalog.collection_records("table.tsv::table_rows[]")
    assert [row.data["count"] for row in table] == ["2", "5"]


def test_json_catalog_does_not_merge_unrelated_nested_records(tmp_path):
    (tmp_path / "data.json").write_text(json.dumps({
        "events": [
            {"topic": "billing export redesign", "state": "proposed"},
            {"topic": "other change", "approved_by": "Omar"},
        ]
    }))
    catalog = SourceCatalog(tmp_path)
    root_records = [record for record in catalog.records.values() if record.collection_path.endswith("::$root")]
    assert root_records == []
    event_records = catalog.collection_records("data.json::events[]")
    assert len(event_records) == 2


def test_url_colon_inside_key_value_row_is_not_a_label_separator(tmp_path):
    (tmp_path / "events.log").write_text(
        "owner=Mara | component=retry scheduler | canonical_pr=https://example.test/pull/7\n"
    )
    catalog = SourceCatalog(tmp_path)
    assert catalog.collection_records("events.log::labeled_records[]") == []
    line = catalog.collection_records("events.log::lines[]")[0]
    assert "https://example.test/pull/7" in line.text


def test_catalog_discovers_delimited_key_value_rows(tmp_path):
    (tmp_path / "events.log").write_text(
        "record=evt-1 | component=retry scheduler | state=blocked | canonical_pr=https://example.test/pull/7\n"
    )
    catalog = SourceCatalog(tmp_path)
    rows = catalog.collection_records("events.log::key_value_rows[]")
    assert len(rows) == 1
    assert rows[0].data["state"] == "blocked"
    assert rows[0].data["canonical_pr"] == "https://example.test/pull/7"


def test_all_records_prefers_one_coherent_text_document(tmp_path):
    (tmp_path / "note.txt").write_text("Topic: Cedar\nOwner: Mara\nState: ready\n")
    catalog = SourceCatalog(tmp_path)
    records = catalog.collection_records("all_records")
    assert len(records) == 1
    assert records[0].collection_path == "logical_documents"
    assert records[0].data["Owner"] == "Mara"
    assert "State: ready" in records[0].text


def test_table_header_after_preamble_is_indexed(tmp_path):
    (tmp_path / "owners.tsv").write_text(
        "Table: owner state table.\n"
        "actor\titem\tstate\treference\n"
        "Mira Sol\tAster One\topen\tAS-001\n"
        "Pax Neri\tBeryl One\topen\tBY-001\n"
    )
    catalog = SourceCatalog(tmp_path)
    rows = [
        record for record in catalog.collection_records("all_records")
        if record.collection_path.endswith("::table_rows[]")
    ]
    assert [row.data["actor"] for row in rows] == ["Mira Sol", "Pax Neri"]
    assert all(row.data["state"] == "open" for row in rows)


def test_loose_nested_objects_are_indexed_as_coherent_records(tmp_path):
    (tmp_path / "objects.raw").write_text(
        'records: [\n'
        '{ name: "Orchid Alpha", status: "ready", ids: { asset: "OA-1" } }\n'
        '{ name: "Orchid Beta", status: "paused", ids: { asset: "OB-2" } }\n'
        ']\n'
    )
    catalog = SourceCatalog(tmp_path)
    records = [
        record for record in catalog.collection_records("all_records")
        if record.collection_path.endswith("::loose_objects[]")
    ]
    assert [record.data["name"] for record in records] == ["Orchid Alpha", "Orchid Beta"]
    assert records[1].data["status"] == "paused"
    assert records[1].data["ids"]["asset"] == "OB-2"


def test_url_scheme_colon_is_not_parsed_as_label(tmp_path):
    (tmp_path / "note.txt").write_text(
        "Escrow import design.\n"
        "The canonical design URL is https://docs.example.test/escrow-r7.\n"
    )
    catalog = SourceCatalog(tmp_path)
    record = next(
        item for item in catalog.collection_records("logical_documents")
        if item.source_path == "note.txt"
    )
    assert "The canonical design URL is https" not in record.data
    assert "https://docs.example.test/escrow-r7" in record.text
