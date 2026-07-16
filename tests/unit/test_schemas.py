from knowmoredirt.schemas import (
    assert_portable_closed_schema,
    dataset_profile_schema,
    event_fact_verdict_schema,
    evidence_review_schema,
    grounded_answer_schema,
    query_program_schema,
    semantic_contract_schema,
    tool_extraction_schema,
)


def test_all_semantic_schemas_are_portable_and_closed():
    assert_portable_closed_schema(dataset_profile_schema("fingerprint"))
    assert_portable_closed_schema(event_fact_verdict_schema("contract"))
    assert_portable_closed_schema(evidence_review_schema("contract"))
    assert_portable_closed_schema(semantic_contract_schema("Who owns it?", "contract"))
    assert_portable_closed_schema(query_program_schema("contract"))
    assert_portable_closed_schema(tool_extraction_schema("contract"))
    assert_portable_closed_schema(grounded_answer_schema("contract"))


def test_query_schema_binds_contract_and_contains_generic_tools_only():
    schema = query_program_schema("contract")
    program = schema["properties"]["query_program"]
    assert program["properties"]["contract_id"]["enum"] == ["contract"]
    tools = program["properties"]["steps"]["items"]["properties"]["tool"]["enum"]
    assert tools == [
        "sample_records",
        "search_records",
        "expand_source_context",
        "filter_records",
        "project_values",
        "extract_values",
        "model_extract",
        "join_records",
        "union_values",
        "intersect_values",
        "sort_records",
        "aggregate_values",
        "calculate",
    ]


def test_numeric_value_repair_schema_is_closed():
    from knowmoredirt.schemas import numeric_value_repair_schema
    schema = numeric_value_repair_schema("contract")
    root = schema["properties"]["numeric_value_repair"]
    assert root["additionalProperties"] is False
    assert root["properties"]["value"]["type"] == "number"
    assert root["properties"]["contract_id"]["enum"] == ["contract"]
