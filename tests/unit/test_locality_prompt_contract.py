from __future__ import annotations

from knowmoredirt.model_planner import (
    build_chunk_drs_condition_prompt,
    build_chunk_drs_prompt,
    build_chunk_drs_skeleton_prompt,
    build_compact_chunk_drs_prompt,
)


def test_all_chunk_extraction_prompts_define_source_locality_units() -> None:
    chunk = '[{"name":"Alice","status":"open"},{"name":"Bob","status":"closed"}]'
    prompts = [
        build_chunk_drs_prompt(chunk, rel_path="records.json"),
        build_chunk_drs_skeleton_prompt(chunk, rel_path="records.json"),
        build_chunk_drs_condition_prompt(
            chunk,
            rel_path="records.json",
            referents=[],
            boxes=[{"id": "b0"}],
        ),
        build_compact_chunk_drs_prompt(chunk, rel_path="records.json"),
    ]

    for prompt in prompts:
        lowered = prompt.lower()
        assert "json array element" in lowered
        assert "plain-text sentence" in lowered
        assert "whole" in lowered
        assert "chunk" in lowered
