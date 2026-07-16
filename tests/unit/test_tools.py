from __future__ import annotations
import json
from knowmoredirt.catalog import SourceCatalog
from knowmoredirt.models import ToolResult
from knowmoredirt.tools import ToolExecutor, expand_step


def step(tool, **overrides):
    payload = {
        "tool": tool,
        "inputs": [],
        "collection": "",
        "terms": [],
        "mode": "none",
        "filters": [],
        "fields": [],
        "left_field": "",
        "right_field": "",
        "sort_field": "",
        "direction": "none",
        "aggregate": "none",
        "operation": "none",
        "numbers": [],
        "extractor": "none",
        "label": "",
        "start_phrase": "",
        "end_phrase": "",
        "pattern": "",
        "value_group": "",
        "time_group": "",
        "occurrence": "none",
        "value_kind": "text",
        "strip_chars": "",
        "distinct": False,
        "limit": 100,
    }
    payload.update(overrides)
    return payload


def test_generic_filter_project_and_aggregate(tmp_path):
    (tmp_path / "data.json").write_text(json.dumps({"rows": [{"state": "ready", "id": "A"}, {"state": "hold", "id": "B"}, {"state": "ready", "id": "C"}]}))
    catalog = SourceCatalog(tmp_path)
    executor = ToolExecutor(catalog)
    results = executor.execute(
        [
            step("filter_records", collection="data.json::rows[]", filters=[{"field_path": "state", "operator": "equals", "value": "ready", "values": []}]),
            step("project_values", inputs=[0], fields=["id"], distinct=True),
            step("aggregate_values", inputs=[1], aggregate="count", distinct=True),
        ]
    )
    assert results[1].values == ["A", "C"]
    assert results[2].scalar == 2


def test_search_uses_only_explicit_terms(tmp_path):
    (tmp_path / "a.txt").write_text("Cedar bridge owner is Mara")
    (tmp_path / "b.txt").write_text("Birch bridge owner is Omar")
    results = ToolExecutor(SourceCatalog(tmp_path)).execute(
        [step("search_records", collection="all_records", terms=["cedar"], mode="all", limit=10)]
    )
    assert [record.source_path for record in results[0].records] == ["a.txt"]


def test_search_prefers_localized_records_over_large_containers(tmp_path):
    (tmp_path / "data.json").write_text(json.dumps({"items": [{"text": "Omar should review the OAuth callback repair PR."}], "noise": "x" * 5000}))
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("search_records", collection="all_records", terms=["OAuth callback repair PR"], mode="any", limit=1)
    ])
    assert results[0].records[0].collection_path.endswith("items[]")


def test_calculate_executes_only_explicit_model_numbers(tmp_path):
    (tmp_path / "note.txt").write_text("Seven plus five appears here.")
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("calculate", operation="add", numbers=[7, 5])
    ])
    assert results[0].scalar == 12


def test_search_does_not_reward_repeated_terms_over_localized_evidence(tmp_path):
    (tmp_path / "data.json").write_text(json.dumps({
        "items": [
            {"text": "retry scheduler tracking PR https://example.test/pull/7"},
            {"text": ("retry scheduler " * 8) + ("unrelated detail " * 40)},
        ]
    }))
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("search_records", collection="all_records", terms=["retry scheduler"], mode="all", limit=1)
    ])
    assert "tracking PR" in results[0].records[0].text


def test_empty_upstream_does_not_fall_back_to_full_collection(tmp_path):
    (tmp_path / "data.json").write_text(json.dumps({"rows": [{"id": "A", "state": "ready"}]}))
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("filter_records", collection="data.json::rows[]", filters=[
            {"field_path": "state", "operator": "equals", "value": "missing", "values": []}
        ]),
        step("project_values", inputs=[0], collection="all_records", fields=["id"]),
    ])
    assert results[0].records == []
    assert results[1].values == []


def test_executor_clamps_excessive_model_limit(tmp_path):
    (tmp_path / "data.json").write_text(json.dumps({"rows": [{"id": str(i)} for i in range(8)]}))
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("sample_records", collection="data.json::rows[]", limit=1000000)
    ])
    assert len(results[0].records) == 8


def test_search_selected_fields_constrain_text_records(tmp_path):
    (tmp_path / "note.txt").write_text(
        "Biology notebook page.\nSpecimen code: BIO-22.\n"
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("search_records", collection="all_records", terms=["biology"], mode="any", fields=["Specimen code"], limit=1)
    ])
    assert "BIO-22" in results[0].records[0].text


def test_search_chain_uses_prior_records_instead_of_restarting_globally(tmp_path):
    (tmp_path / "a.txt").write_text("Cedar owner Mara")
    (tmp_path / "b.txt").write_text("Birch owner Omar")
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("search_records", collection="all_records", terms=["Cedar"], mode="all"),
        step("search_records", inputs=[0], terms=["owner"], mode="all"),
    ])
    assert [record.source_path for record in results[1].records] == ["a.txt"]


def test_extract_values_after_label_and_latest_event(tmp_path):
    (tmp_path / "note.txt").write_text(
        "Recipe: pear cakes\nOven temperature: 180C.\n"
        "2026-01-10 jar state: cloudy.\n2026-01-12 jar state: clear.\n"
    )
    executor = ToolExecutor(SourceCatalog(tmp_path))
    results = executor.execute([
        step("search_records", collection="all_records", terms=["pear cakes"], mode="all"),
        step(
            "extract_values", inputs=[0], extractor="after_label", label="Oven temperature",
            occurrence="first", value_kind="text", strip_chars="."
        ),
        step(
            "extract_values", inputs=[0], extractor="event_series",
            pattern=r"(?P<time>\d{4}-\d{2}-\d{2})\s+jar state:\s*(?P<value>[^.]+)",
            value_group="value", time_group="time", occurrence="latest_by_time",
            value_kind="text", strip_chars="."
        ),
    ])
    assert results[1].values == ["180C"]
    assert results[2].values == ["clear"]


def test_date_time_extractor_without_occurrence_preserves_all_candidates(tmp_path):
    (tmp_path / "schedule.txt").write_text(
        "2027-03-04 08:15 kiln check.\n2027-03-04 17:45 gallery opening.\n"
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("search_records", collection="all_records", terms=["gallery opening"], mode="all"),
        step("extract_values", inputs=[0], extractor="date_time"),
    ])
    assert results[1].values == ["2027-03-04 08:15", "2027-03-04 17:45"]
    assert results[1].diagnostics["candidate_count"] == 2


def test_search_combines_lexical_terms_and_structured_filters(tmp_path):
    (tmp_path / "events.log").write_text(
        "event=owner update | component=retry scheduler | state=current | owner=Mara\n"
        "event=owner update | component=other service | state=current | owner=Omar\n"
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step(
            "search_records",
            collection="all_records",
            terms=["retry scheduler"],
            mode="all",
            filters=[{"field_path": "state", "operator": "equals", "value": "current", "values": []}],
        )
    ])
    assert len(results[0].records) == 1
    assert results[0].records[0].data["owner"] == "Mara"


def test_latest_by_time_falls_back_to_source_order_without_timestamps(tmp_path):
    (tmp_path / "events.log").write_text(
        "component=retry scheduler | owner=Mara\n"
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("search_records", collection="all_records", terms=["retry scheduler"], mode="all"),
        step(
            "extract_values", inputs=[0], fields=["owner"], extractor="field",
            occurrence="latest_by_time"
        ),
    ])
    assert results[1].values == ["Mara"]


def test_model_extract_routes_bounded_prior_evidence_to_callback(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner is Mara.")
    catalog = SourceCatalog(tmp_path)
    executor = ToolExecutor(catalog)
    calls = []

    def callback(step_id, selected_step, prior_results):
        calls.append((step_id, selected_step, prior_results))
        records = prior_results[0].records
        return ToolResult(
            step_id,
            "values",
            values=["Mara"],
            records=records,
            diagnostics={"status": "extracted"},
        )

    results = executor.execute(
        [
            step("search_records", collection="all_records", terms=["Cedar", "owner"], mode="all"),
            step("model_extract", inputs=[0]),
        ],
        semantic_extractor=callback,
    )
    assert results[1].values == ["Mara"]
    assert calls[0][0] == "1"
    assert calls[0][2][0].records[0].source_path == "note.txt"


def test_compact_step_normalizes_missing_search_mode_and_extractor_arguments():
    search = expand_step({
        "tool": "search_records", "inputs": [], "collection": "all_records",
        "terms": ["Cedar", "owner"], "fields": [], "filters": [],
        "arguments": [], "limit": 10,
    })
    extraction = expand_step({
        "tool": "extract_values", "inputs": [0], "collection": "",
        "terms": [], "fields": [], "filters": [],
        "arguments": [{
            "name": "extractor", "value": "regex",
            "values": [r"owner is (?P<value>[A-Za-z]+)"], "numbers": [],
        }], "limit": 10,
    })
    assert search["mode"] == "any"
    assert extraction["pattern"] == r"owner is (?P<value>[A-Za-z]+)"
    assert extraction["value_group"] == "value"


def test_search_uses_generic_token_coverage_for_nonverbatim_phrases(tmp_path):
    (tmp_path / "note.txt").write_text(
        "The tracking PR is https://example.test/pull/7 for the retry scheduler in BeaconForce."
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step(
            "search_records",
            collection="all_records",
            terms=["tracking PR URL", "BeaconForce retry scheduler"],
            mode="all",
            limit=10,
        )
    ])
    assert len(results[0].records) == 1




def test_compact_regex_uses_numeric_capture_group_argument():
    expanded = expand_step({
        "tool": "extract_values", "inputs": [0], "collection": "",
        "terms": [], "fields": [], "filters": [],
        "arguments": [{
            "name": "extractor", "value": "regex",
            "values": [r"temperature:\s*(\d+C)"], "numbers": [1],
        }], "limit": 1,
    })
    assert expanded["pattern"] == r"temperature:\s*(\d+C)"
    assert expanded["value_group"] == "1"


def test_missing_single_term_mode_defaults_to_token_coverage_search():
    expanded = expand_step({
        "tool": "search_records", "inputs": [], "collection": "all_records",
        "terms": ["disagreed about library hours"], "fields": [], "filters": [],
        "arguments": [], "limit": 10,
    })
    assert expanded["mode"] == "all"


def test_phrase_search_falls_back_to_token_coverage_when_words_are_interleaved(tmp_path):
    (tmp_path / "homework.txt").write_text(
        "Math word problem: 7 apples plus 5 apples equals 12 apples."
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step(
            "search_records",
            collection="all_records",
            terms=["7 plus 5"],
            mode="phrase",
            limit=10,
        )
    ])
    assert len(results[0].records) == 1
    assert results[0].diagnostics["fallback_from_phrase"] == "7 plus 5"


def test_search_tolerates_stopwords_and_simple_morphology(tmp_path):
    (tmp_path / "debate.txt").write_text(
        "Ada: close at six.\nBen: I disagree; families need evening library hours."
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step(
            "search_records", collection="all_records",
            terms=["disagreed about library hours"], mode="all", limit=10,
        )
    ])
    assert len(results[0].records) == 1


def test_search_matches_raw_json_like_scope_without_exact_phrase(tmp_path):
    (tmp_path / "raw_json_like.blob").write_text(
        '{ project: "Not a schema", owner: "Zia Fern" }\nThis is ordinary raw text.'
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step(
            "search_records", collection="all_records",
            terms=["raw JSON-like text"], mode="all", limit=10,
        )
    ])
    assert len(results[0].records) == 1


def test_invalid_extractor_tool_alias_normalizes_to_generic_field_extraction():
    expanded = expand_step({
        "tool": "extract_values", "inputs": [0], "collection": "",
        "terms": [], "fields": ["text"], "filters": [],
        "arguments": [{
            "name": "extractor", "value": "extract_values", "values": [], "numbers": [],
        }], "limit": 1,
    })
    assert expanded["extractor"] == "field"


def test_after_phrase_stops_at_inline_object_delimiter_and_strips_quotes(tmp_path):
    (tmp_path / "raw.blob").write_text(
        '{ project: "Not a schema", owner: "Zia Fern", status: "observed" }'
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("search_records", collection="all_records", terms=["owner", "raw"], mode="all"),
        step(
            "extract_values", inputs=[0], extractor="after_phrase", start_phrase="owner",
            occurrence="first", value_kind="text", strip_chars=",.?!:;",
        ),
    ])
    assert results[1].values == ["Zia Fern"]




def test_model_extract_ignores_irrelevant_executor_arguments():
    expanded = expand_step({
        "tool": "model_extract", "inputs": [0], "collection": "",
        "terms": [], "fields": [], "filters": [],
        "arguments": [{
            "name": "extractor", "value": "relation", "values": [], "numbers": [],
        }], "limit": 1,
    })
    assert expanded["extractor"] == "none"




def test_search_tolerates_one_edit_irregular_tense(tmp_path):
    (tmp_path / "schedule.txt").write_text(
        "Final verified schedule: the parade began at 13:00."
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step(
            "search_records", collection="all_records",
            terms=["parade begin", "final verified schedule"], mode="all", limit=10,
        )
    ])
    assert len(results[0].records) == 1






def test_search_stems_shipping_to_ship(tmp_path):
    (tmp_path / "note.txt").write_text(
        "Correction: Mist Vale did not ship the red crate."
    )
    catalog = SourceCatalog(tmp_path)
    results = ToolExecutor(catalog).execute([
        step(
            "search_records",
            collection="all_records",
            terms=["correction", "Mist Vale", "shipping", "red crate"],
            mode="all",
            limit=10,
        )
    ])
    assert len(results[0].records) == 1




def test_after_phrase_does_not_slice_inside_inflected_word(tmp_path):
    (tmp_path / "music.txt").write_text(
        "Music lesson: Arlo practiced the D minor scale."
    )
    result = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("search_records", collection="all_records", terms=["Arlo", "scale"], mode="all"),
        step(
            "extract_values",
            inputs=[0],
            extractor="after_phrase",
            start_phrase="Arlo practice",
            limit=1,
        ),
    ])[1]
    assert result.values == []

def test_search_does_not_hide_unmatched_model_terms(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner is Mara.")
    catalog = SourceCatalog(tmp_path)
    executor = ToolExecutor(catalog)
    results = executor.execute([
        step(
            "search_records",
            collection="all_records",
            terms=["Cedar", "term absent from source"],
            mode="all",
        )
    ])
    assert results[0].records == []


def test_search_fields_are_ranking_hints_not_hard_filters(tmp_path):
    (tmp_path / "preferred.json").write_text(
        json.dumps({"text": "generic owl note", "Dataset URL": "https://example.test/generic"})
    )
    (tmp_path / "coherent.txt").write_text(
        "Owl calls study. Dataset URL: https://example.test/owl"
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step(
            "search_records",
            collection="all_records",
            terms=["owl calls study"],
            mode="all",
            fields=["Dataset URL"],
            limit=10,
        )
    ])
    assert results[0].records
    assert results[0].records[0].source_path == "coherent.txt"


def test_calculate_explicit_numbers_do_not_double_count_upstream_values(tmp_path):
    (tmp_path / "note.txt").write_text("7 plus 5 equals 12")
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("calculate", operation="add", numbers=[7, 5]),
        step("calculate", inputs=[0], operation="add", numbers=[7, 5]),
    ])
    assert results[0].scalar == 12
    assert results[1].scalar == 12


def test_narrow_collection_search_miss_retries_all_records(tmp_path):
    (tmp_path / "wrong.txt").write_text("Physics notes without arithmetic.")
    (tmp_path / "right.txt").write_text("Homework note: 7 + 5 = 12.")
    catalog = SourceCatalog(tmp_path)
    results = ToolExecutor(catalog).execute([
        step(
            "search_records",
            collection="wrong.txt::lines[]",
            terms=["7+5"],
            mode="any",
            fields=["text"],
            limit=10,
        )
    ])
    assert [record.source_path for record in results[0].records] == ["right.txt"]
    assert results[0].diagnostics["collection_expanded"] is True


def test_derived_values_and_scalars_carry_upstream_record_provenance(tmp_path):
    (tmp_path / "data.json").write_text(json.dumps({"rows": [{"value": 7}, {"value": 5}]}))
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("project_values", collection="data.json::rows[]", fields=["value"]),
        step("calculate", inputs=[0], operation="add"),
        step("aggregate_values", inputs=[0], aggregate="count"),
        step("union_values", inputs=[0]),
    ])
    assert results[1].scalar == 12
    assert len(results[1].records) == 2
    assert len(results[2].records) == 2
    assert len(results[3].records) == 2


def test_search_relaxes_nonmatching_field_hints_before_global_fallback(tmp_path):
    (tmp_path / "note.txt").write_text("Aurora safety note author: Mira Sol.")
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step(
            "search_records",
            collection="logical_documents",
            terms=["Aurora safety note"],
            mode="any",
            fields=["nonexistent.deep.path"],
            limit=10,
        )
    ])
    assert results[0].records
    assert results[0].diagnostics["fields"] == ["nonexistent.deep.path"]


def test_missing_mode_matches_noncontiguous_tokens_in_one_model_term(tmp_path):
    (tmp_path / "note.txt").write_text(
        "Project MarlinKind has artifacts. Reese requested the plan."
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step(
            "search_records",
            collection="all_records",
            terms=["MarlinKind plan"],
            mode="none",
            limit=10,
        )
    ])
    assert len(results[0].records) == 1


def test_parameterless_extract_values_preserves_upstream_material(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner Mara")
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("search_records", collection="all_records", terms=["Cedar"], mode="all"),
        step("project_values", inputs=[0], fields=["text"]),
        step("extract_values", inputs=[1], extractor="none", fields=[]),
    ])
    assert results[2].values == results[1].values
    assert [record.record_id for record in results[2].records] == [
        record.record_id for record in results[1].records
    ]
    assert results[2].diagnostics["passthrough"] is True


def test_source_path_token_never_excludes_stronger_content_match(tmp_path):
    (tmp_path / "essay-final.txt").write_text("Unrelated archival summary.")
    (tmp_path / "homework.txt").write_text(
        "Lina drafted the volcano homework essay for Meadow Class."
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step(
            "search_records",
            collection="logical_documents",
            terms=["volcano homework essay", "Meadow Class"],
            mode="any",
            fields=["text"],
            limit=10,
        )
    ])
    assert results[0].records
    assert results[0].records[0].source_path == "homework.txt"


def test_root_search_expands_narrow_collection_and_ranks_global_match(tmp_path):
    (tmp_path / "wrong.json").write_text(
        json.dumps([{"Teacher feedback": "cite one map source"}])
    )
    (tmp_path / "right.txt").write_text(
        "Lina drafted the volcano homework essay. Teacher feedback: Ms. Orin wrote the note."
    )
    catalog = SourceCatalog(tmp_path)
    narrow = next(
        name for name in catalog.collections
        if name.startswith("wrong.json::") and name != "wrong.json::${}"
    )
    results = ToolExecutor(catalog).execute([
        step(
            "search_records",
            collection=narrow,
            terms=["Teacher feedback", "wrote feedback on the volcano homework essay"],
            mode="any",
            fields=["Teacher feedback"],
            limit=10,
        )
    ])
    assert results[0].records[0].source_path == "right.txt"
    assert results[0].diagnostics["collection_expanded"] is True


def test_deterministic_extraction_miss_preserves_records_for_semantic_fallback(tmp_path):
    (tmp_path / "essay.txt").write_text(
        "Ravi Nolen authored the river trade history essay."
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step(
            "search_records",
            collection="all_records",
            terms=["river trade history essay"],
            mode="all",
            limit=10,
        ),
        step(
            "extract_values",
            inputs=[0],
            extractor="after_phrase",
            start_phrase="authored by",
            limit=10,
        ),
    ])
    assert results[1].values == []
    assert [record.record_id for record in results[1].records] == [
        record.record_id for record in results[0].records
    ]
    assert results[1].diagnostics["preserved_input_records_on_miss"] is True
