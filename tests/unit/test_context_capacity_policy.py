from __future__ import annotations

import ast
from pathlib import Path

from knowmoredirt.context_budget import (
    context_relative_budget,
    context_safety_tokens,
    context_token_capacity,
    contextualize_json_schema,
    schema_array_capacity,
    schema_string_capacity,
)
from knowmoredirt.model_planner import (
    CHUNK_DRS_SOURCE_SPAN_POLICY,
    _context_limited_chunk_drs_text,
    chunk_drs_json_schema,
    chunk_drs_source_span_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FACING_FILES = (
    PROJECT_ROOT / "src/context_capacity.py",
    PROJECT_ROOT / "src/knowmoredirt/model.py",
    PROJECT_ROOT / "src/knowmoredirt/model_planner.py",
    PROJECT_ROOT / "src/knowmoredirt/ingest.py",
    PROJECT_ROOT / "src/knowmoredirt/engine.py",
    PROJECT_ROOT / "src/knowmoredirt/bounded_dspg.py",
    PROJECT_ROOT / "src/file_system_catalog/content_pipeline.py",
    PROJECT_ROOT / "src/file_system_catalog/content_cli.py",
    PROJECT_ROOT / "src/file_system_catalog/content_search_cli.py",
    PROJECT_ROOT / "src/file_system_catalog/folder_assistant.py",
    PROJECT_ROOT / "src/file_system_catalog/folder_assistant_cli.py",
)

OBSOLETE_ABSOLUTE_CAPACITY_TOKENS = (
    "max_candidates: int = 24",
    "\"KMD_SCAN_UNIT_MAX_CHARS\"",
    "\"KMD_SCAN_PACK_MAX_CHARS\"",
    "\"KMD_CHUNK_DRS_N_PREDICT\"",
    "\"KMD_CHUNK_DRS_MAX_ARRAY_ITEMS\"",
    "\"KMD_CHUNK_DRS_MAX_EVIDENCE_CHARS\"",
    "\"KMD_MENTIONS_MAX_PER_CHUNK\"",
    "\"KMD_TEMPORAL_SAME_SPAN_MAX_VALUES\"",
    "\"KMD_TEMPORAL_SAME_SPAN_MAX_EDGES\"",
    "\"KMD_DETERMINISTIC_FRAMES_MAX_PER_CHUNK\"",
    "\"KMD_VERIFIER_DISCOURSE_FRAME_LIMIT\"",
    "\"KMD_DISCOURSE_PAYLOAD_LIMIT\"",
    "\"KMD_EVIDENCE_WINDOW_RADIUS\"",
    "\"KMD_EVIDENCE_TEXT_CHARS\"",
    "\"KMD_LAZY_FRAME_SEARCH_LIMIT\"",
    "\"KMD_LAZY_FRAME_CHUNK_LIMIT\"",
    "\"KMD_FOCUSED_EVIDENCE_LIMIT\"",
    "\"KMD_FOCUSED_EVIDENCE_WINDOW_CHARS\"",
    "\"KMD_EVIDENCE_SEARCH_LIMIT\"",
    "\"KMD_EVIDENCE_PAYLOAD_LIMIT\"",
    "len(support) >= 6",
    "--embedding-batch-size",
    "--embedding-max-batch-characters",
    "--chunk-batch-size",
    "--chunk-batch-token-budget",
    "--max-evidence",
    "actions[:8]",
    "windows[:64]",
    "min(top_k, 12)",
    "source_span_candidate_count",
)


def test_obsolete_absolute_model_capacity_paths_are_absent() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in MODEL_FACING_FILES)
    for token in OBSOLETE_ABSOLUTE_CAPACITY_TOKENS:
        assert token not in source


def test_context_capacities_scale_monotonically() -> None:
    small = 4096
    large = 65536
    assert context_safety_tokens(large) > context_safety_tokens(small)
    assert context_relative_budget(large).output_tokens > context_relative_budget(small).output_tokens
    assert context_relative_budget(large).safe_input_tokens > context_relative_budget(small).safe_input_tokens
    assert context_token_capacity(large, ratio_default=1.0 / 64.0) > context_token_capacity(
        small,
        ratio_default=1.0 / 64.0,
    )
    assert schema_string_capacity(large, "evidence") > schema_string_capacity(small, "evidence")
    assert schema_array_capacity(large // 4, "dense") > schema_array_capacity(small // 4, "dense")


def test_schema_profiles_resolve_from_context_and_output_budget() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "x-kmd-array-profile": "dense",
                "items": {"type": "string", "x-kmd-string-profile": "evidence"},
            }
        },
    }
    small = contextualize_json_schema(schema, context_size=4096, output_tokens=1024)
    large = contextualize_json_schema(schema, context_size=65536, output_tokens=16384)
    small_items = small["properties"]["items"]
    large_items = large["properties"]["items"]
    assert "x-kmd-array-profile" not in small_items
    assert "x-kmd-string-profile" not in small_items["items"]
    assert large_items["maxItems"] > small_items["maxItems"]
    assert large_items["items"]["maxLength"] > small_items["items"]["maxLength"]


def test_chunk_drs_schema_never_enumerates_an_evidence_subset() -> None:
    schema = chunk_drs_json_schema(
        max_evidence_chars=4096,
        max_array_items=64,
        include_auxiliary_fields=False,
        source_id="records.jsonl",
        constrain_stable_ids=True,
    )

    evidence_schemas: list[dict[str, object]] = []

    def walk(node: object, property_name: str = "") -> None:
        if isinstance(node, dict):
            if property_name in {"evidence_text", "evidence_span"}:
                evidence_schemas.append(node)
            for key, value in node.items():
                walk(value, key if key not in {"properties", "items"} else property_name)
        elif isinstance(node, list):
            for value in node:
                walk(value, property_name)

    def walk_properties(node: object) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                for key, value in properties.items():
                    if key in {"evidence_text", "evidence_span"} and isinstance(value, dict):
                        evidence_schemas.append(value)
                    walk_properties(value)
            for key, value in node.items():
                if key != "properties":
                    walk_properties(value)
        elif isinstance(node, list):
            for value in node:
                walk_properties(value)

    walk_properties(schema)
    assert evidence_schemas
    assert all("enum" not in item for item in evidence_schemas)
    assert CHUNK_DRS_SOURCE_SPAN_POLICY == "exact-contiguous-source-span-record-consistent-v3"


def test_source_span_diagnostics_have_no_item_count_cap() -> None:
    chunk = "\n".join(f"field_{index}: value_{index}" for index in range(200))
    candidates = chunk_drs_source_span_candidates(chunk, max_evidence_chars=len(chunk))
    assert len(candidates) > 24
    assert "field_199: value_199" in candidates


def test_context_limited_chunk_path_never_silently_truncates_source() -> None:
    class Client:
        def context_size(self) -> int:
            return 65536

    source = "record " * 2000
    prompt_source, audit = _context_limited_chunk_drs_text(
        source,
        Client(),  # type: ignore[arg-type]
        rel_path="records.txt",
        n_predict=16384,
    )
    assert prompt_source == source
    assert audit["input_truncated"] is False


def test_model_facing_default_arguments_do_not_embed_absolute_capacity_numbers() -> None:
    capacity_names = {
        "limit",
        "max_items",
        "max_length",
        "max_chars",
        "max_candidates",
        "n_predict",
        "chunk_limit",
        "doc_limit",
        "window_chars",
        "radius",
        "target_chars",
        "overlap_chars",
        "max_tokens",
        "batch_size",
        "max_batch_characters",
        "excerpt_characters",
        "top_k",
    }
    violations: list[str] = []
    for path in MODEL_FACING_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            positional = [*node.args.posonlyargs, *node.args.args]
            defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
            pairs = list(zip(positional, defaults)) + list(zip(node.args.kwonlyargs, node.args.kw_defaults))
            for argument, default in pairs:
                if argument.arg not in capacity_names or default is None:
                    continue
                if isinstance(default, ast.Constant) and isinstance(default.value, (int, float)):
                    violations.append(f"{path}:{node.lineno}:{node.name}:{argument.arg}={default.value}")
    assert violations == []


def test_every_production_chunk_drs_schema_is_portable() -> None:
    from knowmoredirt.model import validate_portable_json_schema
    from knowmoredirt.model_planner import (
        chunk_drs_condition_json_schema,
        chunk_drs_json_schema,
        chunk_drs_skeleton_json_schema,
    )

    schemas = (
        chunk_drs_json_schema(
            max_evidence_chars=512,
            max_array_items=32,
            include_auxiliary_fields=False,
            source_id="source.json",
            constrain_stable_ids=True,
        ),
        chunk_drs_skeleton_json_schema("source.json", 32),
        chunk_drs_condition_json_schema(
            source_id="source.json",
            box_ids=["b0"],
            referent_ids=["r0"],
            temporal_ids=["t0"],
            max_conditions=32,
            max_arguments=4,
        ),
    )
    for schema in schemas:
        validate_portable_json_schema(schema)
