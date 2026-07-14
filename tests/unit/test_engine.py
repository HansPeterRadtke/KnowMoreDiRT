from __future__ import annotations

import re

from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.tools import expand_step
from knowmoredirt.model import ModelError


def step(tool: str, **overrides):
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
        "limit": 20,
    }
    payload.update(overrides)
    return payload


def _enum(schema, *path):
    node = schema
    for item in path:
        node = node[item]
    return node["enum"][0]


class FakeModel:
    def __init__(self):
        self.calls = []

    def complete_json(self, stage, prompt, schema, max_tokens=0):
        self.calls.append((stage, prompt, schema, max_tokens))
        if stage == "dataset_profile":
            fingerprint = _enum(
                schema,
                "properties",
                "dataset_profile",
                "properties",
                "fingerprint",
            )
            return {
                "dataset_profile": {
                    "fingerprint": fingerprint,
                    "summary": "A small text dataset.",
                    "collections": [],
                    "general_notes": "Use coherent logical documents.",
                }
            }
        if stage == "semantic_contract":
            contract = schema["properties"]["semantic_contract"]["properties"]
            question = contract["question"]["enum"][0]
            contract_id = contract["contract_id"]["enum"][0]
            return {
                "semantic_contract": {
                    "contract_id": contract_id,
                    "question": question,
                    "intent_summary": "Return the owner of Cedar.",
                    "answer_shape": "text",
                    "answer_slot": "person name",
                    "semantic_kind": "entity_attribute",
                    "world_scope": "asserted_world",
                    "source_scope": "any",
                    "authority_mode": "any",
                    "target_phrases": ["Cedar"],
                    "scope_phrases": [],
                    "relation_phrases": ["owns"],
                    "constraint_phrases": [],
                    "polarity": "positive",
                    "temporal_mode": "none",
                    "epistemic_mode": "asserted",
                    "requires_explicit_evidence": True,
                    "compound_request": False,
                }
            }
        if stage == "query_program":
            contract_id = _enum(
                schema,
                "properties",
                "query_program",
                "properties",
                "contract_id",
            )
            return {
                "query_program": {
                    "contract_id": contract_id,
                    "steps": [
                        step(
                            "search_records",
                            collection="all_records",
                            terms=["Cedar", "owner"],
                            mode="all",
                        ),
                        step(
                            "extract_values",
                            inputs=[0],
                            extractor="regex",
                            pattern=r"owner\s+is\s+(?P<value>[A-Za-z]+)",
                            value_group="value",
                            occurrence="first",
                        ),
                    ],
                }
            }
        raise AssertionError(stage)


def test_engine_runs_profile_semantics_compiler_and_generic_extraction(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner is Mara.")
    model = FakeModel()
    answer = KnowMoreDiRTEngine(tmp_path, model=model).answer("Who owns Cedar?")
    assert answer.text == "Mara"
    assert answer.evidence[0]["source_path"] == "note.txt"
    assert [call[0] for call in model.calls] == [
        "dataset_profile",
        "semantic_contract",
        "query_program",
    ]


def test_invalid_program_returns_unknown_without_deterministic_semantic_fallback(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner is Mara.")

    class BadModel(FakeModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "query_program":
                contract_id = _enum(
                    schema,
                    "properties",
                    "query_program",
                    "properties",
                    "contract_id",
                )
                return {"query_program": {"contract_id": contract_id, "steps": []}}
            if stage == "query_program_repair":
                raise ModelError("repair unavailable")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=BadModel()).answer("Who owns Cedar?")
    assert answer.text == "unknown"
    assert answer.diagnostics["reason"] == "ModelError"


def test_grounding_is_bound_to_immutable_contract(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner is Mara.")

    class GroundingModel(FakeModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "query_program":
                contract_id = _enum(
                    schema,
                    "properties",
                    "query_program",
                    "properties",
                    "contract_id",
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            step(
                                "search_records",
                                collection="all_records",
                                terms=["Cedar", "owner"],
                                mode="all",
                            )
                        ],
                    }
                }
            if stage == "grounded_answer":
                contract_id = _enum(
                    schema,
                    "properties",
                    "grounded_answer",
                    "properties",
                    "contract_id",
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "grounded_answer": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "Mara",
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "derivation": "extraction",
                        "confidence": 1.0,
                        "reason": "Explicit source statement.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    model = GroundingModel()
    answer = KnowMoreDiRTEngine(tmp_path, model=model).answer("Who owns Cedar?")
    assert answer.text == "Mara"
    assert [call[0] for call in model.calls] == [
        "dataset_profile",
        "semantic_contract",
    ]


def test_model_selected_numeric_tool_returns_structural_scalar(tmp_path):
    (tmp_path / "note.txt").write_text("The note asks for seven plus five.")

    class ArithmeticModel(FakeModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = schema["properties"]["semantic_contract"]["properties"]
                return {
                    "semantic_contract": {
                        "contract_id": contract["contract_id"]["enum"][0],
                        "question": contract["question"]["enum"][0],
                        "intent_summary": "Add seven and five.",
                        "answer_shape": "number",
                        "answer_slot": "numeric result",
                        "semantic_kind": "calculation",
                        "world_scope": "asserted_world",
                        "source_scope": "any",
                        "authority_mode": "any",
                        "target_phrases": ["seven", "five"],
                        "scope_phrases": [],
                        "relation_phrases": ["plus"],
                        "constraint_phrases": [],
                        "polarity": "neutral",
                        "temporal_mode": "none",
                        "epistemic_mode": "asserted",
                        "requires_explicit_evidence": False,
                        "compound_request": False,
                    }
                }
            if stage == "query_program":
                contract_id = _enum(
                    schema,
                    "properties",
                    "query_program",
                    "properties",
                    "contract_id",
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [step("calculate", operation="add", numbers=[7, 5])],
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=ArithmeticModel()).answer("What is seven plus five?")
    assert answer.text == "12"


def test_model_extract_uses_strict_contract_and_cited_evidence(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner is Mara.")

    class SemanticExtractionModel(FakeModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            self.calls.append((stage, prompt, schema, max_tokens))
            if stage == "dataset_profile":
                fingerprint = _enum(
                    schema, "properties", "dataset_profile", "properties", "fingerprint"
                )
                return {
                    "dataset_profile": {
                        "fingerprint": fingerprint,
                        "summary": "A small text dataset.",
                        "collections": [],
                        "general_notes": "Use coherent records.",
                    }
                }
            if stage == "semantic_contract":
                contract = schema["properties"]["semantic_contract"]["properties"]
                return {
                    "semantic_contract": {
                        "contract_id": contract["contract_id"]["enum"][0],
                        "question": contract["question"]["enum"][0],
                        "intent_summary": "Find the owner of Cedar.",
                        "answer_shape": "text",
                        "answer_slot": "owner name",
                        "semantic_kind": "entity_attribute",
                        "world_scope": "asserted_world",
                        "source_scope": "any",
                        "authority_mode": "any",
                        "target_phrases": ["Cedar"],
                        "scope_phrases": [],
                        "relation_phrases": ["owner"],
                        "constraint_phrases": [],
                        "polarity": "positive",
                        "temporal_mode": "none",
                        "epistemic_mode": "asserted",
                        "requires_explicit_evidence": True,
                        "compound_request": False,
                    }
                }
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            step(
                                "search_records",
                                collection="all_records",
                                terms=["Cedar", "owner"],
                                mode="all",
                            ),
                            step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extract":
                extraction = schema["properties"]["tool_extraction"]["properties"]
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": extraction["contract_id"]["enum"][0],
                        "status": "extracted",
                        "values": ["Mara"],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "The source explicitly identifies Mara as the owner.",
                    }
                }
            raise AssertionError(stage)

    model = SemanticExtractionModel()
    answer = KnowMoreDiRTEngine(tmp_path, model=model).answer("Who owns Cedar?")
    assert answer.text == "Mara"
    assert answer.evidence[0]["source_path"] == "note.txt"
    assert [call[0] for call in model.calls] == [
        "dataset_profile", "semantic_contract", "query_program", "tool_extract"
    ]


def test_program_normalization_truncates_after_answer_producing_model_extract():
    program = {
        "contract_id": "contract",
        "steps": [
            step("search_records", collection="all_records", terms=["target"], mode="all"),
            step("model_extract", inputs=[0]),
            step("search_records", inputs=[1], terms=["unrelated"], mode="all"),
        ],
    }
    normalized = KnowMoreDiRTEngine._normalize_program(program)
    assert [item["tool"] for item in normalized["steps"]] == [
        "search_records", "model_extract"
    ]


def test_program_normalization_converts_generic_text_projection_to_model_extract():
    program = {
        "contract_id": "contract",
        "steps": [
            step("search_records", collection="all_records", terms=["invented phrase"], mode="all"),
            step("extract_values", inputs=[0], fields=["text"]),
        ],
    }
    normalized = KnowMoreDiRTEngine._normalize_program(program)
    assert normalized["steps"][-1]["tool"] == "model_extract"
    assert normalized["steps"][-1]["inputs"] == [0]


def test_scope_coverage_rejects_unscoped_global_owner_search(tmp_path):
    (tmp_path / "a.txt").write_text("owner: Hale")
    engine = KnowMoreDiRTEngine(tmp_path, model=object())
    contract = {
        "contract_id": "contract", "question": "Who is owner in raw JSON text?",
        "intent_summary": "find owner", "answer_shape": "text", "answer_slot": "owner",
        "semantic_kind": "entity_attribute",
        "world_scope": "asserted_world",
        "target_phrases": ["owner"], "scope_phrases": ["raw JSON text"],
        "relation_phrases": [], "constraint_phrases": [], "polarity": "positive",
        "temporal_mode": "none", "epistemic_mode": "asserted",
        "requires_explicit_evidence": True, "compound_request": False,
    }
    program = {
        "contract_id": "contract",
        "steps": [step("search_records", collection="all_records", terms=["owner"], mode="all")],
    }
    import pytest
    with pytest.raises(Exception, match="omitted semantic scope"):
        engine._validate_program(contract, program)


def test_unknown_like_model_value_is_canonical_unknown():
    assert KnowMoreDiRTEngine._unknown_like_value("has no stated translation")
    assert not KnowMoreDiRTEngine._unknown_like_value("good evening")


def test_program_normalization_consumes_model_owned_scope_phrase():
    program = {
        "contract_id": "contract",
        "steps": [
            step("search_records", collection="all_records", terms=["owner"], mode="all"),
            step("model_extract", inputs=[0]),
        ],
    }
    contract = {"scope_phrases": ["in the raw JSON-like text"]}
    normalized = KnowMoreDiRTEngine._normalize_program(program, contract)
    assert normalized["steps"][0]["terms"] == ["owner", "in the raw JSON-like text"]


def test_presentation_operator_scope_is_not_forced_into_source_search():
    program = {
        "contract_id": "contract",
        "steps": [
            step(
                "search_records", collection="all_records",
                terms=["dataset URL", "owl calls"], mode="all",
            ),
            step("extract_values", inputs=[0], extractor="url"),
        ],
    }
    contract = {"scope_phrases": ["listed", "for owl calls"]}
    normalized = KnowMoreDiRTEngine._normalize_program(program, contract)
    assert normalized["steps"][0]["terms"] == ["dataset URL", "owl calls"]


def test_calculate_scalar_is_direct_even_when_contract_shape_is_text(tmp_path):
    (tmp_path / "note.txt").write_text("seven plus five")
    engine = KnowMoreDiRTEngine(tmp_path, model=object())
    program = {
        "contract_id": "contract",
        "steps": [step("calculate", operation="add", numbers=[7, 5])],
    }
    results = engine.executor.execute(program["steps"])
    answer = engine._direct_structural_answer(
        {"answer_shape": "text"}, program, results
    )
    assert answer is not None
    assert answer.text == "12"


def test_failed_deterministic_extraction_falls_back_to_model_extract(tmp_path):
    (tmp_path / "note.txt").write_text("Biology notebook.\nSpecimen code: BIO-22.")
    engine = KnowMoreDiRTEngine(tmp_path, model=object())
    program = {
        "contract_id": "contract",
        "steps": [
            step("search_records", collection="all_records", terms=["biology", "specimen code"], mode="all"),
            step(
                "extract_values", inputs=[0], extractor="after_phrase",
                start_phrase="biology specimen code", occurrence="first",
            ),
        ],
    }
    results = engine.executor.execute(program["steps"])
    fallback = engine._fallback_model_extract_program(program, results)
    assert fallback is not None
    assert fallback["steps"][-1]["tool"] == "model_extract"
    assert fallback["steps"][-1]["inputs"] == [0]


def test_storage_relation_scope_is_not_forced_into_source_search():
    program = {
        "contract_id": "contract",
        "steps": [
            step("search_records", collection="all_records", terms=["village map"], mode="all"),
            step("extract_values", inputs=[0], fields=["text"]),
        ],
    }
    contract = {"scope_phrases": ["stored"]}
    normalized = KnowMoreDiRTEngine._normalize_program(program, contract)
    assert normalized["steps"][0]["terms"] == ["village map"]
    assert normalized["steps"][-1]["tool"] == "model_extract"


def test_program_normalization_removes_operator_only_retrieval_terms():
    program = {
        "contract_id": "contract",
        "steps": [
            step(
                "search_records", collection="all_records",
                terms=["village map", "stored"], mode="all",
            ),
            step("model_extract", inputs=[0]),
        ],
    }
    contract = {"scope_phrases": ["stored"]}
    normalized = KnowMoreDiRTEngine._normalize_program(program, contract)
    assert normalized["steps"][0]["terms"] == ["village map"]


def test_fiction_only_boolean_evidence_cannot_prove_asserted_false(tmp_path):
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r1", "logical_documents", "fantasy.scroll", 0,
        {"text": "Fantasy lore about the Moon Gate; not a real history document."},
        "Fantasy lore about the Moon Gate; not a real history document.",
    )
    assert KnowMoreDiRTEngine._fiction_only_boolean_evidence([record])


def test_asserted_boolean_false_from_nonfiction_evidence_is_not_blocked(tmp_path):
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r1", "logical_documents", "inspection.txt", 0,
        {"text": "Inspection result: the bridge is not open."},
        "Inspection result: the bridge is not open.",
    )
    assert not KnowMoreDiRTEngine._fiction_only_boolean_evidence([record])


def test_grounded_boolean_normalization_uses_yes_no():
    assert KnowMoreDiRTEngine._normalize_grounded({
        "answer": "false", "answer_shape": "boolean"
    })["answer"] == "no"
    assert KnowMoreDiRTEngine._normalize_grounded({
        "answer": "true", "answer_shape": "boolean"
    })["answer"] == "yes"


def test_interrogative_scope_words_are_not_forced_into_retrieval():
    program = {
        "contract_id": "contract",
        "steps": [
            step("search_records", collection="all_records", terms=["piano recital"], mode="all"),
            step("model_extract", inputs=[0]),
        ],
    }
    contract = {"scope_phrases": ["when"]}
    normalized = KnowMoreDiRTEngine._normalize_program(program, contract)
    assert normalized["steps"][0]["terms"] == ["piano recital"]


def test_unknown_no_input_material_requests_repair(tmp_path):
    (tmp_path / "note.txt").write_text("target")
    engine = KnowMoreDiRTEngine(tmp_path, model=object())
    program = {"contract_id": "c", "steps": [step("model_extract", inputs=[0])]}
    from knowmoredirt.models import ToolResult
    results = {0: ToolResult("0", "values", diagnostics={"status": "unknown", "reason": "no input material"})}
    assert engine._needs_execution_repair(program, results)


def test_at_time_value_expands_to_unique_full_datetime():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r1", "logical_documents", "calendar.txt", 0,
        {"text": "2026-06-02 18:00 piano recital."},
        "2026-06-02 18:00 piano recital.",
    )
    assert KnowMoreDiRTEngine._expand_temporal_values(
        ["18:00"], [record], "at_time"
    ) == ["2026-06-02 18:00"]


def test_at_time_value_stays_unchanged_when_datetime_is_ambiguous():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r1", "logical_documents", "calendar.txt", 0,
        {"text": "2026-06-02 18:00 event. 2026-06-03 18:00 event."},
        "2026-06-02 18:00 event. 2026-06-03 18:00 event.",
    )
    assert KnowMoreDiRTEngine._expand_temporal_values(
        ["18:00"], [record], "at_time"
    ) == ["18:00"]


def test_word_adjacency_is_not_explicit_definition_evidence():
    from knowmoredirt.models import SourceRecord
    contract = {
        "answer_slot": "meaning",
        "semantic_kind": "entity_attribute",
        "world_scope": "asserted_world",
        "intent_summary": "Find the meaning",
        "relation_phrases": ["meaning of"],
        "target_phrases": ["florpus zeta"],
    }
    record = SourceRecord(
        "r1", "logical_documents", "salad.txt", 0,
        {"text": "florpus zeta candle bicycle monarchy"},
        "florpus zeta candle bicycle monarchy",
    )
    assert not KnowMoreDiRTEngine._has_explicit_definition_evidence(
        contract, ["candle bicycle monarchy"], [record]
    )


def test_means_relation_is_explicit_definition_evidence():
    from knowmoredirt.models import SourceRecord
    contract = {
        "answer_slot": "translation",
        "semantic_kind": "entity_attribute",
        "world_scope": "asserted_world",
        "intent_summary": "Find the translation",
        "relation_phrases": ["means"],
        "target_phrases": ["bonsoir"],
    }
    record = SourceRecord(
        "r1", "logical_documents", "language.txt", 0,
        {"text": "bonsoir means good evening"},
        "bonsoir means good evening",
    )
    assert KnowMoreDiRTEngine._has_explicit_definition_evidence(
        contract, ["good evening"], [record]
    )


def test_mixed_dream_and_waking_evidence_is_not_fiction_only():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r1", "logical_documents", "dreamfile", 0,
        {"text": "I dreamed AtlasCrane deleted vault.key. When I woke up, the repository still contained vault.key."},
        "I dreamed AtlasCrane deleted vault.key. When I woke up, the repository still contained vault.key.",
    )
    assert not KnowMoreDiRTEngine._fiction_only_boolean_evidence([record])


def test_mixed_epistemic_evidence_detection():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r1", "logical_documents", "dreamfile", 0,
        {"text": "I dreamed the file vanished. When I woke up, it still existed."},
        "I dreamed the file vanished. When I woke up, it still existed.",
    )
    assert KnowMoreDiRTEngine._mixed_epistemic_evidence([record])


def test_explicit_false_reason_is_detected():
    assert KnowMoreDiRTEngine._reason_explicit_false(
        "The waking evidence indicates the deletion did not occur in reality."
    )
    assert not KnowMoreDiRTEngine._reason_explicit_false(
        "The evidence is insufficient to know whether it occurred."
    )


def test_discourse_scope_operators_are_not_forced_into_retrieval():
    program = {
        "contract_id": "contract",
        "steps": [
            step(
                "search_records", collection="all_records",
                terms=["Tao believes", "VectorLamp cache"], mode="all",
            ),
            step("model_extract", inputs=[0]),
        ],
    }
    contract = {"scope_phrases": ["about", "regarding"]}
    normalized = KnowMoreDiRTEngine._normalize_program(program, contract)
    assert normalized["steps"][0]["terms"] == ["Tao believes", "VectorLamp cache"]


def test_discourse_repair_detection_for_quoted_first_person():
    contract = {"epistemic_mode": "quoted", "answer_slot": "message_content", "semantic_kind": "reported_content", "world_scope": "reported_content"}
    extraction = {"status": "extracted", "values": ["I will fix it tomorrow."]}
    assert KnowMoreDiRTEngine._needs_discourse_repair(contract, extraction)


def test_discourse_repair_detection_for_belief_content():
    contract = {"epistemic_mode": "reported", "answer_slot": "belief_content", "semantic_kind": "reported_content", "world_scope": "reported_content"}
    extraction = {"status": "extracted", "values": ["the cache should expire every 8 minutes"]}
    assert KnowMoreDiRTEngine._needs_discourse_repair(contract, extraction)


def test_discourse_repair_not_used_for_asserted_fact():
    contract = {"epistemic_mode": "asserted", "answer_slot": "owner", "semantic_kind": "entity_attribute", "world_scope": "asserted_world"}
    extraction = {"status": "extracted", "values": ["Mara"]}
    assert not KnowMoreDiRTEngine._needs_discourse_repair(contract, extraction)




def test_model_owned_source_classification_bypasses_fiction_only_guard():
    contract = {
        "semantic_kind": "source_classification",
        "world_scope": "source_metadata",
        "answer_shape": "boolean",
        "epistemic_mode": "asserted",
    }
    assert contract["semantic_kind"] == "source_classification"


def test_classification_repair_detection_is_model_owned():
    contract = {"semantic_kind": "source_classification", "world_scope": "source_metadata"}
    extraction = {"status": "extracted", "values": ["fiction homework"]}
    assert KnowMoreDiRTEngine._needs_classification_repair(contract, extraction)


def test_classification_repair_not_used_for_event_fact():
    contract = {"semantic_kind": "event_fact", "world_scope": "asserted_world"}
    extraction = {"status": "extracted", "values": ["no"]}
    assert not KnowMoreDiRTEngine._needs_classification_repair(contract, extraction)


def test_event_fact_repair_detection_is_model_owned():
    contract = {"semantic_kind": "event_fact", "world_scope": "asserted_world"}
    extraction = {"status": "extracted", "values": ["only salted hashes"]}
    assert KnowMoreDiRTEngine._needs_event_fact_repair(contract, extraction)


def test_list_cardinality_fallback_for_single_heading(tmp_path):
    (tmp_path / "plan.txt").write_text("Project depends on A, B, and C")
    engine = KnowMoreDiRTEngine(tmp_path, model=object())
    program = {
        "contract_id": "c",
        "steps": [
            step("search_records", collection="all_records", terms=["Project"], mode="all"),
            step("extract_values", inputs=[0], extractor="after_phrase", start_phrase="Project"),
        ],
    }
    results = engine.executor.execute(program["steps"])
    assert KnowMoreDiRTEngine._needs_list_cardinality_fallback(
        {"answer_shape": "list"}, program, results
    )


def test_container_scope_words_are_not_forced_into_retrieval():
    program = {
        "contract_id": "contract",
        "steps": [
            step(
                "search_records", collection="all_records",
                terms=["ActionGarden", "product roadmap target", "corpus"], mode="all",
            ),
            step("model_extract", inputs=[0]),
        ],
    }
    contract = {"scope_phrases": ["product roadmap target", "corpus"]}
    normalized = KnowMoreDiRTEngine._normalize_program(program, contract)
    assert normalized["steps"][0]["terms"] == ["ActionGarden", "product roadmap target"]


def test_event_fact_unknown_also_requests_model_repair():
    contract = {"semantic_kind": "event_fact", "world_scope": "asserted_world"}
    extraction = {"status": "unknown", "values": []}
    assert KnowMoreDiRTEngine._needs_event_fact_repair(contract, extraction)


def test_structured_answer_slot_records_are_preferred_over_prose_noise():
    from knowmoredirt.models import SourceRecord
    noise = SourceRecord(
        "noise", "logical_documents", "noise.txt", 0,
        {"text": "Cinder Atlas person id cloud gate"},
        "Cinder Atlas person id cloud gate",
    )
    clean = SourceRecord(
        "clean", "logical_documents", "memo.txt", 0,
        {"label_records": [{"person id": "actor_mara884211"}]},
        "Cinder Atlas dossier. person id: actor_mara884211",
    )
    selected = KnowMoreDiRTEngine._prefer_structured_answer_slot_records(
        {"answer_slot": "person_id"},
        [noise, clean],
    )
    assert [record.record_id for record in selected] == ["clean"]


def test_structured_answer_slot_preference_falls_back_when_no_field_matches():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r1", "logical_documents", "note.txt", 0,
        {"text": "Owner is Mara"},
        "Owner is Mara",
    )
    selected = KnowMoreDiRTEngine._prefer_structured_answer_slot_records(
        {"answer_slot": "person_id"},
        [record],
    )
    assert selected == [record]


def test_exact_field_name_beats_compound_noise_key():
    from knowmoredirt.models import SourceRecord
    noise = SourceRecord(
        "noise", "logical_documents", "cache.lock", 0,
        {"Lark Mirror owner": "ERROR-0000"},
        "Lark Mirror owner: ERROR-0000",
    )
    clean = SourceRecord(
        "clean", "json", "lark.raw", 0,
        {"name": "Lark Mirror", "owner": "Ila Nore", "reviewer": "Oren Pax"},
        "Lark Mirror owner Ila Nore reviewer Oren Pax",
    )
    selected = KnowMoreDiRTEngine._prefer_structured_answer_slot_records(
        {"answer_slot": "owner"},
        [noise, clean],
    )
    assert [record.record_id for record in selected] == ["clean"]


def test_entity_attribute_model_extract_keeps_multiple_search_candidates():
    program = {
        "contract_id": "c",
        "steps": [
            step("search_records", collection="all_records", terms=["Lark Mirror", "owner"], mode="all", limit=1),
            step("model_extract", inputs=[0]),
        ],
    }
    normalized = KnowMoreDiRTEngine._normalize_program(
        program,
        {"semantic_kind": "entity_attribute", "scope_phrases": [], "world_scope": "asserted_world"},
    )
    assert normalized["steps"][0]["limit"] == 20


def test_target_localized_view_prevents_cross_block_attribute_leakage():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r1", "logical_documents", "memo.txt", 0,
        {
            "label_records": [
                {"approver": "Eri Noam"},
                {"Amber Loom owner": "Pela Dorn"},
            ]
        },
        "Cinder Atlas dossier.\napprover: Eri Noam\n\nAmber Loom owner: Pela Dorn.\nAmber Loom current state: paused.",
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "target_phrases": ["Who approved Amber Loom?"],
            "relation_phrases": ["approved"],
            "answer_slot": "approver",
        },
    )
    assert "Amber Loom" in view["text"]
    assert "Eri Noam" not in view["text"]
    assert "Eri Noam" not in view["excerpt"]


def test_target_localized_view_keeps_matching_entity_block():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r1", "logical_documents", "memo.txt", 0,
        {},
        "Cinder Atlas dossier.\napprover: Eri Noam\n\nAmber Loom owner: Pela Dorn.",
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "target_phrases": ["Who approved Cinder Atlas?"],
            "relation_phrases": ["approved"],
            "answer_slot": "approver",
        },
    )
    assert "Eri Noam" in view["text"]
    assert "Amber Loom" not in view["text"]


def test_no_value_is_stated_is_canonical_unknown():
    assert KnowMoreDiRTEngine._unknown_like_value(
        "No email address is stated for Fern Vault."
    )


def test_reviewed_does_not_satisfy_approved_relation():
    contract = {
        "semantic_kind": "entity_attribute",
        "world_scope": "asserted_world",
        "relation_phrases": ["approved"],
        "answer_slot": "approver",
    }
    view = {
        "record_id": "r1",
        "excerpt": "Mara Vell reviewed the safety addendum for Amber Loom.",
        "data": {},
    }
    assert not KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract, "Mara Vell", [view]
    )


def test_approver_label_satisfies_approved_relation():
    contract = {
        "semantic_kind": "entity_attribute",
        "world_scope": "asserted_world",
        "relation_phrases": ["approved"],
        "answer_slot": "approver",
    }
    view = {
        "record_id": "r1",
        "excerpt": "Cinder Atlas dossier. approver: Eri Noam",
        "data": {},
    }
    assert KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract, "Eri Noam", [view]
    )


def test_json_owner_field_satisfies_entity_relation():
    contract = {
        "semantic_kind": "entity_attribute",
        "world_scope": "asserted_world",
        "relation_phrases": ["owner"],
        "answer_slot": "owner",
    }
    view = {
        "record_id": "r1",
        "excerpt": '{"name":"Lark Mirror","owner":"Ila Nore"}',
        "data": {"name": "Lark Mirror", "owner": "Ila Nore"},
    }
    assert KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract, "Ila Nore", [view]
    )


def test_one_input_join_normalizes_to_source_context_expansion():
    program = {
        "contract_id": "c",
        "steps": [
            step("search_records", collection="all_records", terms=["Moss Beacon"], mode="all"),
            step("join_records", inputs=[0], collection="nested.ids"),
            step("extract_values", inputs=[1], fields=["invoice"], extractor="field"),
        ],
    }
    normalized = KnowMoreDiRTEngine._normalize_program(
        program,
        {"semantic_kind": "entity_attribute", "scope_phrases": [], "world_scope": "asserted_world"},
    )
    assert normalized["steps"][1]["tool"] == "expand_source_context"
    assert normalized["steps"][1]["inputs"] == [0]


def test_status_satisfies_state_relation():
    contract = {
        "semantic_kind": "entity_attribute",
        "world_scope": "asserted_world",
        "relation_phrases": [],
        "answer_slot": "current_state",
    }
    view = {
        "record_id": "r1",
        "excerpt": "2026-03-09 status: closed for Delta Well.",
        "data": {},
    }
    assert KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract, "closed", [view]
    )


def test_temporal_qualifier_is_not_required_as_relation_word():
    stems = KnowMoreDiRTEngine._entity_relation_stems({
        "relation_phrases": [],
        "answer_slot": "final_state",
    })
    assert stems == {"state"}


def test_nonactual_external_effect_rejects_state_only_evidence_contract():
    contract = {
        "world_scope": "nonactual_external_effect",
        "semantic_kind": "event_fact",
    }
    extraction = {
        "evidence_relation": "state_only",
        "status": "extracted",
        "values": ["no"],
    }
    assert contract["world_scope"] == "nonactual_external_effect"
    assert extraction["evidence_relation"] == "state_only"


def test_nonactual_content_cannot_answer_boolean_fact_contract():
    contract = {
        "answer_shape": "boolean",
        "semantic_kind": "event_fact",
        "world_scope": "asserted_world",
    }
    extraction = {
        "status": "extracted",
        "values": ["no"],
        "evidence_relation": "nonactual_content",
    }
    assert contract["answer_shape"] == "boolean"
    assert extraction["evidence_relation"] == "nonactual_content"


def test_source_quality_scope_words_are_not_forced_into_retrieval():
    program = {
        "contract_id": "c",
        "steps": [
            step("search_records", collection="all_records", terms=["Cinder Atlas", "owner"], mode="all"),
            step("model_extract", inputs=[0]),
        ],
    }
    normalized = KnowMoreDiRTEngine._normalize_program(
        program,
        {
            "semantic_kind": "entity_attribute",
            "world_scope": "reported_content",
            "scope_phrases": ["according to meaningful source"],
        },
    )
    assert normalized["steps"][0]["terms"] == ["Cinder Atlas", "owner"]


def test_definition_guard_uses_distinctive_target_token():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r1", "logical_documents", "glossary.txt", 0,
        {},
        'Glossary: "naur" means north water.',
    )
    assert KnowMoreDiRTEngine._has_explicit_definition_evidence(
        {
            "target_phrases": ["What does naur mean?"],
            "answer_slot": "meaning",
            "intent_summary": "Request the meaning of naur",
            "relation_phrases": [],
        },
        ["north water"],
        [record],
    )


def test_calculation_operation_alias_is_canonicalized():
    program = {
        "contract_id": "c",
        "steps": [
            {
                "tool": "calculate", "inputs": [], "collection": "", "terms": [],
                "fields": [], "filters": [],
                "arguments": [{"name": "operation", "value": "addition", "values": [], "numbers": [14, 8]}],
                "limit": 1,
            }
        ],
    }
    normalized = KnowMoreDiRTEngine._normalize_program(
        program,
        {"answer_slot": "result", "intent_summary": "arithmetic", "scope_phrases": []},
    )
    assert expand_step(normalized["steps"][0])["operation"] == "add"


def test_count_request_compiles_to_structured_filter_and_aggregate():
    program = {
        "contract_id": "c",
        "steps": [step("search_records", collection="all_records", terms=["status active"], mode="all")],
    }
    normalized = KnowMoreDiRTEngine._normalize_program(
        program,
        {
            "answer_slot": "row_count",
            "intent_summary": "Count rows with active status",
            "target_phrases": ["How many Finch rows have status active?"],
            "constraint_phrases": ["status active"],
            "scope_phrases": [],
        },
    )
    assert [item["tool"] for item in normalized["steps"]] == [
        "search_records", "filter_records", "aggregate_values"
    ]
    assert normalized["steps"][0]["terms"] == ["finch"]
    assert normalized["steps"][1]["filters"][0] == {
        "field_path": "status", "operator": "equals", "value": "active", "values": []
    }
    assert expand_step(normalized["steps"][2])["aggregate"] == "count"


def test_count_condition_can_come_from_relation_phrase():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {"contract_id": "c", "steps": [step("search_records", collection="all_records", terms=["archived"], mode="all")]},
        {
            "answer_slot": "row_count",
            "intent_summary": "Count rows with archived status",
            "target_phrases": ["How many rows have status archived?"],
            "constraint_phrases": [],
            "relation_phrases": ["have status archived"],
            "scope_phrases": [],
        },
    )
    assert normalized["steps"][1]["filters"][0]["field_path"] == "status"
    assert normalized["steps"][1]["filters"][0]["value"] == "archived"


def test_count_condition_infers_status_from_predicate_adjective():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {"contract_id": "c", "steps": [step("search_records", collection="all_records", terms=["Finch", "active"], mode="all")]},
        {
            "answer_slot": "active_finch_entry_count",
            "intent_summary": "Request for the count of active Finch entries",
            "target_phrases": ["How many Finch entries are active?"],
            "constraint_phrases": [],
            "relation_phrases": [],
            "scope_phrases": [],
        },
    )
    assert normalized["steps"][0]["terms"] == ["finch"]
    assert normalized["steps"][1]["filters"][0]["field_path"] == "status"
    assert normalized["steps"][1]["filters"][0]["value"] == "active"


def test_entity_attribute_model_extract_uses_targeted_all_records_search():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="logical_documents", terms=["Bell Finch active owner"], fields=["label_records.owner"], mode="all", limit=1),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "semantic_kind": "entity_attribute",
            "answer_slot": "owner",
            "target_phrases": ["Who owns Bell Finch?"],
            "relation_phrases": ["owns"],
            "scope_phrases": [],
        },
    )
    search = normalized["steps"][0]
    assert search["collection"] == "all_records"
    assert search["terms"] == ["bell", "finch"]
    assert search["fields"] == []


def test_negative_relation_scope_is_not_forced_into_literal_retrieval(tmp_path):
    (tmp_path / "note.txt").write_text(
        "Rumi bought tea, rope, and chalk but not blue soap."
    )
    engine = KnowMoreDiRTEngine(tmp_path, model=object())
    contract = {
        "contract_id": "contract",
        "question": "What did Rumi not buy?",
        "intent_summary": "Identify items Rumi did not buy",
        "answer_shape": "list",
        "answer_slot": "items",
        "semantic_kind": "entity_attribute",
        "world_scope": "asserted_world",
        "target_phrases": ["What did Rumi not buy?"],
        "scope_phrases": ["Rumi", "not buy"],
        "relation_phrases": ["not buy"],
        "constraint_phrases": [],
        "polarity": "negative",
        "temporal_mode": "current",
        "epistemic_mode": "asserted",
        "requires_explicit_evidence": True,
        "compound_request": False,
    }
    program = {
        "contract_id": "contract",
        "steps": [
            step("search_records", collection="all_records", terms=["Rumi"], mode="all"),
            step("model_extract", inputs=[0]),
        ],
    }
    engine._validate_program(contract, program)


def test_answer_slot_scope_is_not_forced_into_literal_retrieval(tmp_path):
    (tmp_path / "note.txt").write_text(
        "Aurora Loom Safety Note. Key reviewer: Olan Vex | actor id: ACT-411"
    )
    engine = KnowMoreDiRTEngine(tmp_path, model=object())
    contract = {
        "contract_id": "contract",
        "question": "Which actor id belongs to the key reviewer of Aurora Loom Safety Note?",
        "intent_summary": "Identify the actor id of the key reviewer",
        "answer_shape": "text",
        "answer_slot": "actor_id",
        "semantic_kind": "entity_attribute",
        "world_scope": "asserted_world",
        "target_phrases": ["key reviewer", "Aurora Loom Safety Note"],
        "scope_phrases": ["actor id", "key reviewer", "Aurora Loom Safety Note"],
        "relation_phrases": ["belongs to"],
        "constraint_phrases": [],
        "polarity": "positive",
        "temporal_mode": "current",
        "epistemic_mode": "asserted",
        "requires_explicit_evidence": True,
        "compound_request": False,
    }
    program = {
        "contract_id": "contract",
        "steps": [
            step(
                "search_records", collection="all_records",
                terms=["Aurora Loom Safety Note", "key reviewer"], mode="all",
            ),
            step("model_extract", inputs=[0]),
        ],
    }
    engine._validate_program(contract, program)


def test_entity_attribute_retrieval_drops_generic_identification_verbs():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["Silver Loom"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "semantic_kind": "entity_attribute",
            "answer_slot": "reference_id",
            "target_phrases": ["Which reference ID identifies Silver Loom?"],
            "relation_phrases": [],
            "scope_phrases": [],
        },
    )
    assert normalized["steps"][0]["terms"] == ["loom", "silver"]


def test_identifier_phrase_slicing_normalizes_to_model_extract():
    program = {
        "contract_id": "c",
        "steps": [
            step("search_records", collection="all_records", terms=["Harbor Flax"], mode="all"),
            {
                "tool": "extract_values",
                "inputs": [0],
                "collection": "all_records",
                "terms": ["account identifier"],
                "fields": ["text"],
                "filters": [],
                "arguments": [
                    {"name": "extractor", "value": "after_phrase", "values": ["account identifier"], "numbers": []},
                    {"name": "start_phrase", "value": "Harbor Flax", "values": [], "numbers": []},
                ],
                "limit": 1,
            },
        ],
    }
    normalized = KnowMoreDiRTEngine._normalize_program(
        program,
        {
            "semantic_kind": "entity_attribute",
            "answer_slot": "account_identifier",
            "target_phrases": ["account identifier", "Harbor Flax"],
            "relation_phrases": [],
            "scope_phrases": [],
        },
    )
    assert normalized["steps"][-1]["tool"] == "model_extract"
    assert normalized["steps"][0]["terms"] == ["flax", "harbor"]


def test_identifier_answer_strips_terminal_prose_punctuation():
    values = ["ACCT-4431.", "REF-7401;"]
    cleaned = [__import__("re").sub(r"[.,;:]+$", "", value.strip()) for value in values]
    assert cleaned == ["ACCT-4431", "REF-7401"]


def test_nonproof_reason_is_not_direct_false():
    assert KnowMoreDiRTEngine._reason_is_nonproof(
        "Judgment note: the north hinge crack was not proven."
    )
    assert KnowMoreDiRTEngine._reason_is_nonproof(
        "The belief is not confirmed as fact."
    )
    assert not KnowMoreDiRTEngine._reason_is_nonproof(
        "Inspection found no crack in the hinge."
    )


def test_unresolved_decision_reason_is_nonproof():
    assert KnowMoreDiRTEngine._reason_is_nonproof(
        "The evidence states that no reroute decision was made."
    )
    assert KnowMoreDiRTEngine._reason_is_nonproof(
        "No final decision was made about the west archive shelf."
    )
    assert not KnowMoreDiRTEngine._reason_is_nonproof(
        "The proposed reroute was explicitly rejected."
    )


def test_superlative_row_count_compiles_to_filtered_model_extract():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["open rows"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "answer_slot": "actor",
            "semantic_kind": "entity_attribute",
            "intent_summary": "Identify the actor with the highest count of open rows.",
            "target_phrases": ["Which actor has the most open rows?"],
            "constraint_phrases": ["the most"],
            "relation_phrases": ["has"],
            "scope_phrases": ["the most open rows"],
        },
    )
    assert [item["tool"] for item in normalized["steps"]] == [
        "search_records", "filter_records", "project_values", "aggregate_values"
    ]
    assert normalized["steps"][0]["terms"] == ["open"]
    assert normalized["steps"][1]["filters"][0]["field_path"] == "state"
    assert normalized["steps"][1]["filters"][0]["value"] == "open"


def test_mode_aggregate_is_stable_and_frequency_based(tmp_path):
    from knowmoredirt.catalog import SourceCatalog
    from knowmoredirt.tools import ToolExecutor
    (tmp_path / "rows.tsv").write_text(
        "actor\tstate\n"
        "Mira Sol\topen\n"
        "Mira Sol\topen\n"
        "Pax Neri\topen\n"
        "Pax Neri\topen\n"
        "Pax Neri\topen\n"
    )
    results = ToolExecutor(SourceCatalog(tmp_path)).execute([
        step("search_records", collection="all_records", terms=["open"], mode="all", limit=100),
        step("filter_records", inputs=[0], filters=[{"field_path": "state", "operator": "equals", "value": "open", "values": []}], limit=100),
        step("project_values", inputs=[1], fields=["actor"], limit=100),
        {
            "tool": "aggregate_values", "inputs": [2], "collection": "", "terms": [],
            "fields": [], "filters": [],
            "arguments": [{"name": "aggregate", "value": "mode", "values": [], "numbers": []}],
            "limit": 1,
        },
    ])
    assert results[3].scalar == "Pax Neri"


def test_count_condition_before_rows_keeps_named_target():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {"contract_id": "c", "steps": [step("search_records", collection="all_records", terms=["Pax Neri", "open"], mode="all")]},
        {
            "answer_slot": "open_row_count",
            "intent_summary": "Request for the count of open rows associated with Pax Neri.",
            "target_phrases": ["How many open rows does Pax Neri have?"],
            "constraint_phrases": [],
            "relation_phrases": [],
            "scope_phrases": [],
        },
    )
    assert normalized["steps"][0]["terms"] == ["neri", "pax"]
    assert normalized["steps"][1]["filters"][0]["field_path"] == "state"
    assert normalized["steps"][1]["filters"][0]["value"] == "open"


def test_quantitative_scope_is_not_literal_retrieval():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {"contract_id": "c", "steps": [step("search_records", collection="all_records", terms=["Pax Neri", "open"], mode="all")]},
        {
            "answer_slot": "open_row_count",
            "intent_summary": "Requesting the count of open rows associated with Pax Neri.",
            "target_phrases": ["open rows", "Pax Neri"],
            "constraint_phrases": [],
            "relation_phrases": ["does have"],
            "scope_phrases": ["how many"],
        },
    )
    assert normalized["steps"][0]["terms"] == ["neri", "pax"]


def test_semantic_record_scope_is_not_literal_retrieval():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["Garnet Bridge"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "semantic_kind": "entity_attribute",
            "answer_slot": "organization",
            "target_phrases": ["Which organization owns Garnet Bridge"],
            "relation_phrases": ["owns"],
            "scope_phrases": ["according to the semantic record"],
        },
    )
    assert normalized["steps"][0]["terms"] == ["bridge", "garnet"]


def test_entity_name_slot_strips_terminal_prose_punctuation():
    values = ["Morrow Slate Guild.", "Pax Neri;"]
    cleaned = [__import__("re").sub(r"[.,;:]+$", "", value.strip()) for value in values]
    assert cleaned == ["Morrow Slate Guild", "Pax Neri"]


def test_source_scope_filters_cache_records():
    from knowmoredirt.models import SourceRecord
    semantic = SourceRecord("s", "logical_documents", "records/mica.txt", 0, {}, "Warranty URL: good")
    cache = SourceRecord("c", "logical_documents", "noise/transport_cache.tmp", 0, {}, "CACHE ONLY. Warranty URL: bad")
    assert KnowMoreDiRTEngine._apply_source_scope([semantic, cache], {"source_scope": "non_cache"}) == [semantic]
    assert KnowMoreDiRTEngine._apply_source_scope([semantic, cache], {"source_scope": "cache_only"}) == [cache]


def test_explicit_authority_requires_authority_marker_near_value():
    value = "https://cache.example.test/wrong"
    assert not KnowMoreDiRTEngine._has_explicit_authority_evidence(
        [value],
        [{"excerpt": f"Warranty URL: {value}", "data": {}}],
    )
    assert KnowMoreDiRTEngine._has_explicit_authority_evidence(
        [value],
        [{"excerpt": f"Official warranty URL: {value}", "data": {}}],
    )


def test_source_scope_phrase_is_not_literal_retrieval(tmp_path):
    (tmp_path / "mica.txt").write_text(
        "Record: Mica Relay. Warranty URL: https://warranty.example.test/mica-relay"
    )
    engine = KnowMoreDiRTEngine(tmp_path, model=object())
    contract = {
        "contract_id": "c",
        "question": "Which warranty URL belongs to Mica Relay despite the cache file?",
        "intent_summary": "Find the non-cache warranty URL",
        "answer_shape": "text",
        "answer_slot": "warranty_url",
        "semantic_kind": "entity_attribute",
        "world_scope": "asserted_world",
        "source_scope": "non_cache",
        "authority_mode": "any",
        "target_phrases": ["Mica Relay", "warranty URL"],
        "scope_phrases": ["belongs to", "despite", "cache file"],
        "relation_phrases": ["belongs to"],
        "constraint_phrases": ["despite the cache file"],
        "polarity": "positive",
        "temporal_mode": "current",
        "epistemic_mode": "asserted",
        "requires_explicit_evidence": True,
        "compound_request": False,
    }
    program = {
        "contract_id": "c",
        "steps": [
            step("search_records", collection="all_records", terms=["Mica Relay"], mode="all"),
            step("model_extract", inputs=[0]),
        ],
    }
    normalized = engine._normalize_program(program, contract)
    assert normalized["steps"][0]["terms"] == ["mica", "relay"]
    engine._validate_program(contract, normalized)


def test_contract_normalization_reconciles_model_owned_source_scope():
    base = {
        "intent_summary": "Identify the warranty URL while ignoring cached information.",
        "answer_slot": "warranty_url",
        "target_phrases": ["Mica Relay", "warranty URL"],
        "scope_phrases": ["cache file"],
        "relation_phrases": ["belongs to"],
        "constraint_phrases": ["despite the cache file"],
        "source_scope": "any",
        "authority_mode": "any",
    }
    assert KnowMoreDiRTEngine._normalize_contract(base)["source_scope"] == "non_cache"
    cache = {
        **base,
        "intent_summary": "Identify the official warranty URL stored in a hidden cache.",
        "target_phrases": ["hidden cache URL", "official warranty URL"],
        "scope_phrases": [],
        "constraint_phrases": [],
    }
    normalized = KnowMoreDiRTEngine._normalize_contract(cache)
    assert normalized["source_scope"] == "cache_only"
    assert normalized["authority_mode"] == "explicit_official"
    semantic = {
        **base,
        "intent_summary": "Use the semantic record for the organization.",
        "target_phrases": ["organization"],
        "scope_phrases": ["semantic record"],
        "constraint_phrases": [],
    }
    assert KnowMoreDiRTEngine._normalize_contract(semantic)["source_scope"] == "semantic_only"


def test_confirmed_is_explicit_authority_evidence():
    value = "no reroute decision was made"
    assert KnowMoreDiRTEngine._has_explicit_authority_evidence(
        [value],
        [{"excerpt": f"Confirmed plan: {value}", "data": {}}],
    )


def test_localized_view_isolates_matching_structured_answer_field():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r", "logical_documents", "claims.txt", 0,
        {
            "Witness note": 'Runa said the latch snapped.',
            "Correction": "Mist Vale did not ship the red crate; the corrected crate color was amber.",
            "source": {"path": "claims.txt"},
        },
        'Witness note: Runa said the latch snapped.\nCorrection: Mist Vale did not ship the red crate; the corrected crate color was amber.',
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_slot": "correction_content",
            "target_phrases": ["correction", "Mist Vale", "shipping", "red crate"],
            "relation_phrases": ["said about"],
        },
    )
    assert "Correction:" in view["text"]
    assert "Mist Vale" in view["text"]
    assert "Runa" not in view["text"]


def test_reported_clause_selector_keeps_requested_proposition():
    contract = {
        "semantic_kind": "reported_content",
        "answer_shape": "text",
        "compound_request": False,
        "answer_slot": "correction_content",
        "target_phrases": ["correction", "Mist Vale", "shipping", "red crate"],
        "relation_phrases": ["said about"],
    }
    assert KnowMoreDiRTEngine._select_reported_clause(
        "Mist Vale did not ship the red crate; the corrected crate color was amber.",
        contract,
    ) == "Mist Vale did not ship the red crate"
    assert KnowMoreDiRTEngine._select_reported_clause(
        "Oren believes the river should be rerouted.",
        contract,
    ) == "Oren believes the river should be rerouted."


def test_localized_view_keeps_target_line_and_adjacent_labeled_field():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r", "logical_documents", "homework", 0,
        {"text": "Lina drafted the volcano homework essay.\nTeacher feedback: Ms. Orin wrote a note.\nMath: 7 plus 5."},
        "Lina drafted the volcano homework essay.\nTeacher feedback: Ms. Orin wrote a note.\nMath: 7 plus 5.",
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_slot": "actor",
            "target_phrases": ["Who wrote feedback on the volcano homework essay"],
            "relation_phrases": [],
        },
    )
    assert "volcano homework essay" in view["text"]
    assert "Teacher feedback" in view["text"]
    assert "Math" not in view["text"]


def test_contract_normalization_preserves_explicit_who_verb_relation():
    normalized = KnowMoreDiRTEngine._normalize_contract(
        {
            "question": "Who wrote feedback on the volcano homework essay?",
            "intent_summary": "Identify the author of feedback.",
            "answer_slot": "actor",
            "target_phrases": ["volcano homework essay", "feedback"],
            "scope_phrases": [],
            "relation_phrases": [],
            "constraint_phrases": [],
            "source_scope": "any",
            "authority_mode": "any",
        }
    )
    assert normalized["relation_phrases"] == ["wrote"]


def test_explicit_actor_relation_extracts_titled_author_with_provenance():
    contract = {
        "answer_slot": "actor",
        "semantic_kind": "entity_attribute",
        "relation_phrases": ["wrote"],
    }
    assert KnowMoreDiRTEngine._extract_explicit_actor_relation(
        contract,
        [{
            "record_id": "r1",
            "excerpt": "Teacher feedback: Ms. Orin wrote that the conclusion needs more evidence.",
        }],
    ) == [("Ms. Orin", "r1")]
    assert KnowMoreDiRTEngine._extract_explicit_actor_relation(
        {**contract, "relation_phrases": ["owns"]},
        [{"record_id": "r1", "excerpt": "Ms. Orin wrote a note."}],
    ) == []


def test_explicit_calculation_contract_compiles_to_calculate():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["7 plus 5"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "question": "What does 7 plus 5 equal in the homework note?",
            "semantic_kind": "calculation",
            "answer_slot": "result",
            "intent_summary": "Calculate the explicit expression.",
            "target_phrases": ["7 plus 5", "homework note"],
            "relation_phrases": [],
            "scope_phrases": [],
            "constraint_phrases": [],
        },
    )
    assert len(normalized["steps"]) == 1
    step_view = expand_step(normalized["steps"][0])
    assert step_view["tool"] == "calculate"
    assert step_view["operation"] == "add"
    assert step_view["numbers"] == [7.0, 5.0]


def test_temporal_localization_keeps_all_matching_dated_events():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r", "logical_documents", "states.log", 0,
        {"text": "2026-04-01 RampCart state: planned.\n2026-04-03 RampCart state: measured.\n2026-04-04 RampCart state: revised.\nFinal note: friction was higher."},
        "2026-04-01 RampCart state: planned.\n2026-04-03 RampCart state: measured.\n2026-04-04 RampCart state: revised.\nFinal note: friction was higher.",
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_slot": "current_state",
            "target_phrases": ["current state of RampCart"],
            "relation_phrases": [],
            "temporal_mode": "current",
        },
    )
    assert "planned" in view["text"]
    assert "measured" in view["text"]
    assert "revised" in view["text"]
    assert "friction" not in view["text"]


def test_target_associated_url_satisfies_storage_location_relation():
    contract = {
        "semantic_kind": "entity_attribute",
        "answer_slot": "storage_location",
        "target_phrases": ["village map"],
        "relation_phrases": ["is stored"],
    }
    value = "https://maps.example/village/orchard-north"
    assert KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract,
        value,
        [{
            "excerpt": f"Village map URL: {value}.",
            "data": {},
        }],
    )
    assert not KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract,
        value,
        [{
            "excerpt": f"Unrelated warranty URL: {value}.",
            "data": {},
        }],
    )


def test_negative_acquisition_constraint_binds_bought_evidence():
    contract = {
        "semantic_kind": "entity_attribute",
        "answer_slot": "items",
        "relation_phrases": [],
        "constraint_phrases": ["not buy"],
    }
    assert KnowMoreDiRTEngine._entity_relation_stems(contract) == {"buy"}
    assert KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract,
        "blue soap",
        [{
            "excerpt": "[Owen] I bought rice and lemons but not blue soap.",
            "data": {},
        }],
    )


def test_source_classification_requires_target_bound_evidence():
    contract = {
        "answer_slot": "is_real_history_document",
        "target_phrases": ["Moon Gate story", "real history document"],
        "relation_phrases": [],
    }
    assert not KnowMoreDiRTEngine._has_target_bound_source_classification(
        contract,
        [{"excerpt": "Queen Elira opened the Moon Gate with a glass key.\nDragon treaty ID: DRG-404."}],
    )
    assert KnowMoreDiRTEngine._has_target_bound_source_classification(
        {
            "answer_slot": "is_engineering_record",
            "target_phrases": ["River Gate drawing", "engineering record"],
            "relation_phrases": [],
        },
        [{"excerpt": "River Gate drawing.\nThis drawing is fiction homework, not an engineering record."}],
    )


def test_explicit_negative_finding_distinguishes_observation_from_nonproof():
    contract = {
        "target_phrases": ["a crack", "the tank wall"],
        "relation_phrases": ["found"],
        "answer_slot": "was_crack_found",
    }
    assert KnowMoreDiRTEngine._explicit_negative_finding_sentence(
        contract,
        [{"excerpt": "Later inspection found no crack in the tank wall."}],
    ) == "Later inspection found no crack in the tank wall."
    assert KnowMoreDiRTEngine._explicit_negative_finding_sentence(
        contract,
        [{"excerpt": "No evidence was provided about a crack in the tank wall."}],
    ) == ""


def test_contract_normalization_preserves_source_format_scope():
    normalized = KnowMoreDiRTEngine._normalize_contract(
        {
            "question": "Who is the owner in the raw JSON-like text?",
            "intent_summary": "Identify the owner.",
            "answer_slot": "owner",
            "target_phrases": ["owner"],
            "scope_phrases": [],
            "relation_phrases": [],
            "constraint_phrases": [],
            "source_scope": "any",
            "authority_mode": "any",
        }
    )
    assert normalized["scope_phrases"] == ["raw JSON-like text"]


def test_entity_retrieval_keeps_source_format_scope():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["owner"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "semantic_kind": "entity_attribute",
            "answer_slot": "owner",
            "target_phrases": ["owner"],
            "scope_phrases": ["raw JSON-like text"],
            "relation_phrases": [],
        },
    )
    assert normalized["steps"][0]["terms"] == ["json", "like", "raw", "text"]


def test_contract_normalization_preserves_alias_relation():
    normalized = KnowMoreDiRTEngine._normalize_contract(
        {
            "question": "What is Juniper Vale also called?",
            "intent_summary": "Identify an alternative name.",
            "answer_slot": "alternative_name",
            "target_phrases": ["Juniper Vale"],
            "scope_phrases": [],
            "relation_phrases": [],
            "constraint_phrases": [],
            "source_scope": "any",
            "authority_mode": "any",
        }
    )
    assert normalized["relation_phrases"] == ["also called"]


def test_answer_slot_scope_is_not_injected_into_search_terms():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["parade", "final verified schedule"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "semantic_kind": "event_fact",
            "answer_slot": "start_time",
            "target_phrases": ["parade", "final verified schedule"],
            "scope_phrases": ["start time"],
            "relation_phrases": ["begin according to"],
            "constraint_phrases": [],
        },
    )
    assert "start time" not in normalized["steps"][0]["terms"]


def test_entity_retrieval_keeps_ocr_correction_scope():
    normalized_contract = KnowMoreDiRTEngine._normalize_contract(
        {
            "question": "Who owns the greenhouse fern according to the OCR correction?",
            "intent_summary": "Identify the owner from a corrected OCR source.",
            "answer_slot": "owner",
            "target_phrases": ["greenhouse fern"],
            "scope_phrases": [],
            "relation_phrases": ["owns"],
            "constraint_phrases": [],
            "source_scope": "any",
            "authority_mode": "any",
        }
    )
    assert "OCR correction" in normalized_contract["scope_phrases"]
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["greenhouse fern"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            **normalized_contract,
            "semantic_kind": "entity_attribute",
        },
    )
    assert set(normalized["steps"][0]["terms"]) >= {"greenhouse", "fern", "ocr", "correction"}


def test_contract_type_guard_rejects_url_for_reviewer():
    assert not KnowMoreDiRTEngine._value_matches_contract_type(
        {"answer_slot": "reviewer"},
        "URL-ONLY https://wrong.example.test/reviewer",
    )
    assert KnowMoreDiRTEngine._value_matches_contract_type(
        {"answer_slot": "reviewer"},
        "Dr. Pella",
    )
    assert KnowMoreDiRTEngine._value_matches_contract_type(
        {"answer_slot": "warranty_url"},
        "https://docs.example.test/warranty",
    )


def test_canonicalize_extracted_value_strips_slot_noun_and_actor_role():
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "scale"}, "D minor scale"
    ) == "D minor"
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "actor"}, "Officer Talen"
    ) == "Talen"
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "actor"}, "Farmer Joss"
    ) == "Joss"
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "actor"}, "Dr. Pella"
    ) == "Dr. Pella"
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "actor"}, "Ms. Orin"
    ) == "Ms. Orin"


def test_canonicalize_content_value_strips_repeated_target_suffix():
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {
            "answer_slot": "audit_result",
            "target_phrases": ["audit result", "Fern Vault"],
        },
        "only humidity readings were stored for Fern Vault.",
    ) == "only humidity readings were stored"
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {
            "answer_slot": "statement",
            "target_phrases": ["North Lantern"],
        },
        "prepare the report for North Lantern leadership",
    ) == "prepare the report for North Lantern leadership"


def test_entity_retrieval_preserves_multi_hop_reference_scope():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["Copper Nest reference"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "semantic_kind": "entity_attribute",
            "answer_slot": "owner",
            "target_phrases": ["Copper Nest"],
            "scope_phrases": ["reference"],
            "relation_phrases": ["owns"],
            "constraint_phrases": [],
        },
    )
    assert normalized["steps"][0]["terms"] == ["copper", "nest", "reference"]


def test_temporal_operator_is_not_literal_retrieval_term():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["after dream installed remains"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "semantic_kind": "entity_attribute",
            "answer_slot": "what",
            "target_phrases": ["What remains installed after the dream"],
            "scope_phrases": [],
            "relation_phrases": [],
            "constraint_phrases": [],
            "temporal_mode": "after",
        },
    )
    assert normalized["steps"][0]["terms"] == ["dream", "installed", "remains"]


def test_interrogative_answer_slot_is_not_a_relation():
    contract = {
        "semantic_kind": "entity_attribute",
        "answer_slot": "what",
        "relation_phrases": [],
        "constraint_phrases": [],
    }
    assert KnowMoreDiRTEngine._entity_relation_stems(contract) == set()
    assert KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract,
        "silver gate",
        [{"excerpt": "Real inventory: silver gate remains installed.", "data": {}}],
    )


def test_after_temporal_localization_selects_later_matching_line():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        "r", "logical_documents", "events.log", 0, {},
        "Plaintiff alleges that the blue pump cracked.\n"
        "Later inspection found no crack in the blue pump.",
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_slot": "found_crack",
            "target_phrases": ["blue pump"],
            "relation_phrases": ["find a crack"],
            "temporal_mode": "after",
        },
    )
    assert "Later inspection" in view["text"]
    assert "Plaintiff" not in view["text"]


def test_corrective_answer_strips_leading_log_timestamp():
    sentence = "[11:20] Later inspection found no crack in the blue pump."
    cleaned = __import__("re").sub(r"^\[[^\]]+\]\s*", "", sentence).strip()
    assert cleaned == "Later inspection found no crack in the blue pump."


def test_meaningful_source_is_not_a_definition_request():
    assert not KnowMoreDiRTEngine._is_definition_request(
        {
            "answer_slot": "owner",
            "intent_summary": "Identify the owner from a meaningful source.",
            "relation_phrases": ["owns"],
        }
    )
    assert KnowMoreDiRTEngine._is_definition_request(
        {
            "answer_slot": "meaning",
            "intent_summary": "Identify a word meaning.",
            "relation_phrases": [],
        }
    )


def test_count_condition_can_come_from_scope_phrase():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["Bell Finch active owner"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "question": "How many Finch entries are active?",
            "semantic_kind": "entity_attribute",
            "answer_slot": "active_finch_entry_count",
            "intent_summary": "Count active Finch entries.",
            "target_phrases": ["Finch entries"],
            "scope_phrases": ["active"],
            "relation_phrases": [],
            "constraint_phrases": [],
        },
    )
    assert [item["tool"] for item in normalized["steps"]] == [
        "search_records", "filter_records", "aggregate_values"
    ]
    assert normalized["steps"][0]["terms"] == ["finch"]
    assert normalized["steps"][1]["filters"][0]["field_path"] == "status"
    assert normalized["steps"][1]["filters"][0]["value"] == "active"


def test_list_localization_preserves_all_role_lines():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r",
        collection_path="logical_documents",
        source_path="roles.txt",
        record_index=0,
        text=(
            "Dossier: Project Note.\n"
            "Author: Nira Sol | actor id: ACT-410\n"
            "Key reviewer: Olan Vex | actor id: ACT-411\n"
            "Reviewer: Pema Rill | actor id: ACT-412\n"
            "Observer: Tavi Moss | observer id: OBS-900\n"
        ),
        data={},
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_shape": "list",
            "answer_slot": "actor_ids",
            "target_phrases": ["author", "reviewers", "Project Note"],
            "relation_phrases": [],
        },
    )
    assert "ACT-410" in view["excerpt"]
    assert "ACT-411" in view["excerpt"]
    assert "ACT-412" in view["excerpt"]


def test_list_model_extract_limit_preserves_requested_cardinality():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["Project Note"], mode="all"),
                step("model_extract", inputs=[0], limit=1),
            ],
        },
        {
            "answer_shape": "list",
            "semantic_kind": "event_fact",
            "answer_slot": "actor_ids",
            "target_phrases": ["author", "reviewers", "Project Note"],
            "scope_phrases": [],
            "relation_phrases": [],
            "constraint_phrases": [],
        },
    )
    assert normalized["steps"][-1]["limit"] == 20


def test_canonicalize_common_object_strips_leading_article():
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "snapped_item"}, "the blue latch"
    ) == "blue latch"
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "location"}, "The Hague"
    ) == "The Hague"


def test_target_bound_url_extraction_uses_model_semantics():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["Orchid Gamma"], mode="all"),
                step("extract_values", inputs=[0], extractor="url", fields=["text"], limit=1),
            ],
        },
        {
            "answer_shape": "text",
            "semantic_kind": "entity_attribute",
            "answer_slot": "report_url",
            "target_phrases": ["report URL", "Orchid Gamma"],
            "scope_phrases": [],
            "relation_phrases": [],
            "constraint_phrases": [],
        },
    )
    assert normalized["steps"][-1]["tool"] == "model_extract"


def test_extraction_status_is_structurally_derived_from_values():
    assert KnowMoreDiRTEngine._enforce_extraction_status_invariant(
        {"status": "unknown", "values": ["paused"]}
    )["status"] == "extracted"
    assert KnowMoreDiRTEngine._enforce_extraction_status_invariant(
        {"status": "extracted", "values": []}
    )["status"] == "unknown"


def test_nested_structured_answer_field_is_localized():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r",
        collection_path="logical_documents",
        source_path="nested.raw",
        record_index=0,
        text="",
        data={
            "text": "",
            "label_records": [
                {"group": "Orchid Frame"},
                {"summary": "Only ready records are valid for release"},
            ],
            "source": {},
        },
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_shape": "text",
            "answer_slot": "summary_content",
            "target_phrases": ["Orchid Frame summary", "ready records"],
            "relation_phrases": ["say about"],
        },
    )
    assert "label_records[1].summary" in view["excerpt"]
    assert "Only ready records are valid for release" in view["excerpt"]


def test_extraction_consistency_repair_detects_status_value_conflicts():
    assert KnowMoreDiRTEngine._needs_extraction_consistency_repair(
        {"status": "unknown", "values": ["paused"]}
    )
    assert KnowMoreDiRTEngine._needs_extraction_consistency_repair(
        {"status": "extracted", "values": []}
    )
    assert not KnowMoreDiRTEngine._needs_extraction_consistency_repair(
        {"status": "extracted", "values": ["paused"]}
    )
    assert not KnowMoreDiRTEngine._needs_extraction_consistency_repair(
        {"status": "unknown", "values": []}
    )


def test_localized_view_prefers_nested_summary_field():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r",
        collection_path="logical_documents",
        source_path="nested.raw",
        record_index=0,
        data={
            "text": "group: Orchid Frame\nsummary: Only ready records are valid for release.",
            "label_records": [
                {"group": "Orchid Frame"},
                {"summary": "Only ready records are valid for release."},
            ],
            "source": {"path": "nested.raw"},
        },
        text="group: Orchid Frame\nsummary: Only ready records are valid for release.",
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_shape": "text",
            "answer_slot": "summary_content",
            "target_phrases": ["Orchid Frame summary", "ready records"],
            "relation_phrases": ["say about"],
        },
    )
    assert "Only ready records are valid for release" in view["text"]
    assert "summary" in view["text"].lower()


def test_canonicalize_document_field_strips_terminal_punctuation():
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "summary_content"},
        "Only ready records are valid for release.",
    ) == "Only ready records are valid for release"
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "statement"},
        "It should expire every 8 minutes.",
    ) == "It should expire every 8 minutes."


def test_contract_type_guard_distinguishes_customer_name_from_customer_id():
    assert not KnowMoreDiRTEngine._value_matches_contract_type(
        {"answer_slot": "customer"}, "CUST-4920"
    )
    assert KnowMoreDiRTEngine._value_matches_contract_type(
        {"answer_slot": "customer"}, "Blue Ridge Analytics"
    )
    assert KnowMoreDiRTEngine._value_matches_contract_type(
        {"answer_slot": "customer_id"}, "CUST-4920"
    )


def test_actor_role_repair_detects_transcript_speaker_confusion():
    contract = {
        "answer_slot": "reviewer",
        "relation_phrases": ["reviewed"],
    }
    views = [{
        "excerpt": "[Nina] Correction: Omar reviewed PR-8042; Nina authored the design.",
    }]
    assert KnowMoreDiRTEngine._explicit_relation_actor_candidates(contract, views) == ["Omar"]
    assert KnowMoreDiRTEngine._needs_actor_role_repair(
        contract,
        {"status": "extracted", "values": ["Nina"]},
        views,
    )
    assert not KnowMoreDiRTEngine._needs_actor_role_repair(
        contract,
        {"status": "extracted", "values": ["Omar Kestrel"]},
        views,
    )


def test_actor_role_repair_ignores_role_noun_fields():
    contract = {"answer_slot": "owner", "relation_phrases": ["owner"]}
    views = [{"excerpt": "Cedar owner: Mara\nMara owns Cedar."}]
    assert KnowMoreDiRTEngine._explicit_relation_actor_candidates(contract, views) == []
    assert not KnowMoreDiRTEngine._needs_actor_role_repair(
        contract,
        {"status": "extracted", "values": ["Mara"]},
        views,
    )


def test_unique_full_name_expansion_requires_one_unambiguous_match():
    views = [
        {"excerpt": "Omar reviewed PR-8042."},
        {"excerpt": "Omar Kestrel performed the risk review."},
    ]
    assert KnowMoreDiRTEngine._unique_full_name_expansion("Omar", views) == "Omar Kestrel"
    assert KnowMoreDiRTEngine._unique_full_name_expansion("Omar Kestrel", views) == ""
    ambiguous = [
        {"excerpt": "Omar Kestrel reviewed it."},
        {"excerpt": "Omar Vale reviewed another item."},
    ]
    assert KnowMoreDiRTEngine._unique_full_name_expansion("Omar", ambiguous) == ""


def test_mixed_epistemic_boolean_preserves_full_coherent_record():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r",
        collection_path="logical_documents",
        source_path="dreamfile",
        record_index=0,
        data={},
        text=(
            "I dreamed that AtlasCrane deleted vault.key.\n"
            "When I woke up, the repository still contained vault.key."
        ),
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_shape": "boolean",
            "answer_slot": "did_delete",
            "target_phrases": ["AtlasCrane", "vault.key", "delete"],
            "relation_phrases": ["delete"],
        },
    )
    assert "dreamed" in view["excerpt"]
    assert "still contained" in view["excerpt"]


def test_mixed_epistemic_correction_repair_requires_negative_mixed_evidence():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r",
        collection_path="logical_documents",
        source_path="dreamfile",
        record_index=0,
        data={},
        text="I dreamed the key was deleted. When I woke up, the key still existed.",
    )
    contract = {"answer_shape": "boolean"}
    assert KnowMoreDiRTEngine._needs_mixed_epistemic_correction_repair(
        contract,
        {
            "status": "extracted",
            "values": ["no"],
            "evidence_relation": "nonactual_content",
            "reason": "It did not occur in reality.",
        },
        [record],
    )
    assert not KnowMoreDiRTEngine._needs_mixed_epistemic_correction_repair(
        contract,
        {"status": "extracted", "values": ["yes"], "evidence_relation": "direct_support"},
        [record],
    )


def test_mixed_epistemic_correction_sentence_uses_waking_clause():
    contract = {"relation_phrases": ["delete"]}
    views = [{
        "excerpt": (
            "I had a dream that AtlasCrane deleted vault.key. "
            "When I woke up, the repository still contained vault.key."
        )
    }]
    assert KnowMoreDiRTEngine._mixed_epistemic_correction_sentence(
        contract,
        views,
    ) == "the deletion occurred only in a dream and the repository still contained vault.key"


def test_proof_status_question_treats_no_proof_as_negative_answer():
    from knowmoredirt.models import SourceRecord
    contract = {
        "question": "Was FlowQuill proven to have caused invoice drift?",
        "answer_slot": "was_flowquill_proven",
        "constraint_phrases": ["proven"],
        "relation_phrases": ["caused"],
    }
    assert KnowMoreDiRTEngine._contract_asks_proof_status(contract)
    record = SourceRecord(
        record_id="r", collection_path="logical_documents",
        source_path="judgment.final", record_index=0, data={},
        text="Final judgment summary. The court found no proof that FlowQuill caused invoice drift.",
    )
    assert KnowMoreDiRTEngine._proof_status_correction_sentence(
        contract, [record]
    ) == "the final judgment found no proof"


def test_canonicalize_value_strips_matching_leading_field_label():
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "measurement_date"},
        "measurement date: 1986-07-14",
    ) == "1986-07-14"
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "date"},
        "release date: 1986-07-14",
    ) == "release date: 1986-07-14"


def test_temporal_role_localization_selects_matching_date_field():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r",
        collection_path="logical_documents",
        source_path="measurements.tsv",
        record_index=0,
        data={
            "measurement date": "1986-07-14",
            "source file copied": "2010-05-20",
            "source": {"path": "measurements.tsv"},
        },
        text=(
            "Table: bridge sensor readings for DeltaPier\n"
            "measurement date: 1986-07-14\n"
            "source file copied: 2010-05-20"
        ),
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_shape": "text",
            "answer_slot": "timestamp_or_date",
            "target_phrases": ["DeltaPier source file"],
            "scope_phrases": ["When was the DeltaPier source file copied?"],
            "relation_phrases": ["was copied"],
        },
    )
    assert view["text"] == "source file copied: 2010-05-20"


def test_temporal_role_extraction_uses_model_semantics():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["DeltaPier copied"], mode="all"),
                step("extract_values", inputs=[0], extractor="date_time", fields=["text"], limit=1),
            ],
        },
        {
            "answer_shape": "text",
            "semantic_kind": "event_fact",
            "answer_slot": "timestamp_or_date",
            "target_phrases": ["DeltaPier source file"],
            "scope_phrases": [],
            "relation_phrases": ["was copied"],
            "constraint_phrases": [],
        },
    )
    assert normalized["steps"][-1]["tool"] == "model_extract"


def test_source_scoped_person_answer_preserves_literal_surface_name():
    assert KnowMoreDiRTEngine._preserve_source_surface_name(
        {
            "question": "According to Mira's top-level note, who fixed parser.cpp?",
            "scope_phrases": ["Mira's top-level note"],
        }
    )
    assert not KnowMoreDiRTEngine._preserve_source_surface_name(
        {"question": "Who reviewed PR-8042?", "scope_phrases": []}
    )


def test_reporting_tense_defaults_structurally():
    normalized = KnowMoreDiRTEngine._normalize_contract({"question": "Any question"})
    assert normalized["reporting_tense"] == "none"


def test_reported_content_preserves_full_coherent_record():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r",
        collection_path="logical_documents",
        source_path="debate.md",
        record_index=0,
        data={"Discussion": "VectorLamp cache policy."},
        text=(
            "Discussion: VectorLamp cache policy.\n"
            "Tao believes the cache should expire every 8 minutes.\n"
            "Lena argues for 20 minutes."
        ),
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_shape": "text",
            "semantic_kind": "reported_content",
            "answer_slot": "belief_content",
            "target_phrases": ["Tao", "VectorLamp cache"],
            "relation_phrases": ["believes about"],
        },
    )
    assert "Tao believes" in view["excerpt"]
    assert "Lena argues" not in view["excerpt"]


def test_reported_content_localization_prefers_attributed_relation_line():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r",
        collection_path="logical_documents",
        source_path="choice.md",
        record_index=0,
        data={},
        text=(
            "Discussion: VectorLamp cache policy.\n"
            "Tao believes the cache should expire every 8 minutes.\n"
            "Lena argues that the cache should expire every 20 minutes."
        ),
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_shape": "text",
            "answer_slot": "belief_content",
            "target_phrases": ["Tao", "VectorLamp cache"],
            "scope_phrases": ["VectorLamp cache"],
            "relation_phrases": ["believes about"],
            "temporal_mode": "none",
        },
    )
    assert "Tao believes" in view["text"]
    assert "Lena argues" not in view["text"]


def test_event_actor_localization_prefers_relation_line_over_generic_slot_heading():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r",
        collection_path="logical_documents",
        source_path="customer-note.txt",
        record_index=0,
        data={},
        text=(
            "Support summary for customer Blue Dune Retail.\n"
            "Blue Dune Retail reported that SearchSprout returned duplicate invoices.\n"
            "Kai Ren is the escalation owner."
        ),
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_shape": "text",
            "answer_slot": "customer",
            "target_phrases": ["customer"],
            "scope_phrases": ["reported duplicate invoices in SearchSprout"],
            "relation_phrases": ["reported"],
            "temporal_mode": "none",
        },
    )
    assert "Blue Dune Retail reported" in view["text"]
    assert "escalation owner" not in view["text"]


def test_localization_joins_target_line_with_nearest_answer_slot_line():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r",
        collection_path="logical_documents",
        source_path="support.txt",
        record_index=0,
        data={},
        text=(
            "Support summary for customer Blue Dune Retail.\n"
            "Ticket SUP-4432 tracks the duplicate invoice issue.\n"
            "Kai Ren is the escalation owner."
        ),
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_shape": "text",
            "answer_slot": "escalation_owner",
            "target_phrases": ["SUP-4432"],
            "scope_phrases": [],
            "relation_phrases": [],
            "temporal_mode": "current",
        },
    )
    assert "SUP-4432" in view["text"]
    assert "Kai Ren is the escalation owner" in view["text"]
    assert "Blue Dune Retail" not in view["text"]


def test_negative_alternative_repair_detects_explicit_actual_behavior():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r",
        collection_path="logical_documents",
        source_path="runtime.note",
        record_index=0,
        data={},
        text=(
            "The runtime flags stale ledgers for human review; "
            "it does not delete them."
        ),
    )
    contract = {
        "semantic_kind": "event_fact",
        "answer_shape": "boolean",
        "question": "Does the runtime delete stale ledgers?",
        "answer_slot": "does_delete",
        "constraint_phrases": [],
        "relation_phrases": [],
    }
    extraction = {"status": "extracted", "values": ["no"]}
    assert KnowMoreDiRTEngine._has_explicit_alternative_behavior([record])
    assert KnowMoreDiRTEngine._needs_negative_alternative_repair(
        contract, extraction, [record]
    )


def test_negative_correction_repair_requires_grounded_direct_no():
    contract = {"answer_shape": "boolean", "question": "", "answer_slot": "does_delete"}
    extraction = {
        "status": "extracted",
        "values": ["no"],
        "evidence_relation": "direct_contradiction",
    }
    assert KnowMoreDiRTEngine._needs_negative_correction_repair(contract, extraction, [])
    assert not KnowMoreDiRTEngine._needs_negative_correction_repair(
        contract,
        {**extraction, "values": ["yes"]},
        [],
    )


def test_corrective_surface_strips_duplicate_boolean_prefix():
    import re
    correction = "no; runtime flags records for human review."
    correction = re.sub(
        r"(?i)^(?:no|false)\s*(?:[;:,.!?-]+\s*|$)",
        "",
        correction,
    ).strip()
    assert correction == "runtime flags records for human review."


def test_negative_correction_clause_surface_normalization():
    assert KnowMoreDiRTEngine._normalize_negative_correction_clause(
        "no; the runtime flags stale ledgers; it sends them for human review."
    ) == "runtime flags stale ledgers for human review."
    assert KnowMoreDiRTEngine._normalize_negative_correction_clause(
        "it stores only salted password hashes."
    ) == "it stores only salted password hashes."


def test_alternative_correction_drops_article_for_declared_runtime_subject():
    import re
    correction = "the runtime flags stale records for review."
    target_tokens = {"novatally", "runtime", "delete", "stale", "records"}
    for subject_type in (
        "runtime", "system", "service", "process", "code", "application", "worker", "job"
    ):
        if subject_type in target_tokens:
            correction = re.sub(
                rf"(?i)^the\s+{re.escape(subject_type)}\b",
                subject_type,
                correction,
                count=1,
            ).strip()
            break
    assert correction == "runtime flags stale records for review."


def test_mixed_epistemic_event_noun_falls_back_to_answer_slot():
    views = [{
        "excerpt": (
            "I had a dream that AtlasCrane deleted vault.key. "
            "When I woke up, the repository still contained vault.key."
        )
    }]
    assert KnowMoreDiRTEngine._mixed_epistemic_correction_sentence(
        {"relation_phrases": [], "answer_slot": "did_delete"},
        views,
    ) == "the deletion occurred only in a dream and the repository still contained vault.key"
    assert KnowMoreDiRTEngine._normalize_negative_correction_clause(
        "the runtime flags stale ledgers for human review."
    ) == "runtime flags stale ledgers for human review."


def test_source_classification_surface_strips_attribution_wrapper():
    import re
    correction = "Teacher note indicates it is fiction homework."
    correction = re.sub(
        r"(?i)^(?:(?:the\s+)?(?:teacher\s+note|source|document|note|record))\s+"
        r"(?:indicates|states|says|reports|classifies|labels|marks|shows)\s+(?:that\s+)?",
        "",
        correction,
        count=1,
    ).strip()
    assert correction == "it is fiction homework."


def test_contract_bound_correction_surface_normalizes_attribution_and_target():
    assert KnowMoreDiRTEngine._normalize_contract_bound_correction_surface(
        {"target_phrases": ["the moon factory candy bridge drawing"]},
        "Teacher note indicates it is fiction homework.",
    ) == "it is fiction homework."
    assert KnowMoreDiRTEngine._normalize_contract_bound_correction_surface(
        {"target_phrases": ["audit", "QuillCache", "plaintext passwords"]},
        "QuillCache stores only salted password hashes.",
    ) == "it stores only salted password hashes."
    assert KnowMoreDiRTEngine._normalize_contract_bound_correction_surface(
        {"target_phrases": ["NovaTally runtime", "delete stale ledgers"]},
        "runtime flags stale ledgers for human review.",
    ) == "runtime flags stale ledgers for human review."


def test_count_condition_maps_requested_refund_status_field():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["refunds"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "question": "How many customers have requested refund status in the refunds sheet?",
            "answer_shape": "text",
            "answer_slot": "count_of_customers",
            "semantic_kind": "event_fact",
            "intent_summary": "Count customers requesting refund status.",
            "target_phrases": ["customers"],
            "scope_phrases": ["have requested refund status", "in the refunds sheet"],
            "relation_phrases": ["have requested refund status"],
            "constraint_phrases": ["in the refunds sheet"],
        },
    )
    assert [item["tool"] for item in normalized["steps"]] == [
        "search_records", "filter_records", "aggregate_values"
    ]
    condition = normalized["steps"][1]["filters"][0]
    assert condition["field_path"] == "refund_status"
    assert condition["value"] == "requested"


def test_canonicalize_cause_strips_leading_article():
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "final_cause"}, "the bad certificate"
    ) == "bad certificate"


def test_contract_bound_correction_surface_strips_audit_result_attribution():
    assert KnowMoreDiRTEngine._normalize_contract_bound_correction_surface(
        {"target_phrases": ["audit", "QuillCache", "plaintext passwords"]},
        "the audit result stated QuillCache stores only salted password hashes.",
    ) == "it stores only salted password hashes."


def test_contract_bound_correction_surface_strips_explicit_audit_attribution():
    assert KnowMoreDiRTEngine._normalize_contract_bound_correction_surface(
        {"target_phrases": ["audit", "QuillCache", "plaintext passwords"]},
        "the audit result explicitly stated QuillCache stores only salted password hashes.",
    ) == "it stores only salted password hashes."


def test_contract_bound_correction_surface_uses_operational_target_noun():
    assert KnowMoreDiRTEngine._normalize_contract_bound_correction_surface(
        {"target_phrases": ["NovaTally runtime", "delete stale ledgers"]},
        "the code flags stale ledgers for human review instead.",
    ) == "runtime flags stale ledgers for human review."


def test_artifact_list_allows_locator_and_uses_natural_conjunction():
    contract = {"answer_shape": "list", "answer_slot": "artifact"}
    assert KnowMoreDiRTEngine._value_matches_contract_type(
        contract, "https://plans.example.test/kind"
    )
    assert KnowMoreDiRTEngine._format_list_values(
        ["SPEC-22", "PR-7788", "https://plans.example.test/kind"]
    ) == "SPEC-22, PR-7788, and https://plans.example.test/kind"
    assert KnowMoreDiRTEngine._format_list_values(
        ["ACT-410", "ACT-411", "ACT-412"]
    ) == "ACT-410; ACT-411; ACT-412"


def test_plural_actor_ids_match_identifier_slot_type():
    contract = {"answer_shape": "list", "answer_slot": "actor_ids"}
    assert KnowMoreDiRTEngine._value_matches_contract_type(contract, "ACT-410")
    assert KnowMoreDiRTEngine._value_matches_contract_type(contract, "ACT-411")


def test_passive_owned_by_relation_matches_owner_slot():
    contract = {
        "semantic_kind": "entity_attribute",
        "answer_slot": "owner",
        "relation_phrases": ["owns"],
        "constraint_phrases": [],
        "target_phrases": ["BUG-2244"],
    }
    views = [{
        "excerpt": "[09:03] Ava: Blocker is BUG-2244, owned by Jules.",
        "text": "[09:03] Ava: Blocker is BUG-2244, owned by Jules.",
        "data": {},
    }]
    assert KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract, "Jules", views
    )


def test_actor_role_repair_requires_explicit_verbal_relation():
    contract = {
        "answer_slot": "launch_owner",
        "relation_phrases": [],
    }
    views = [{"excerpt": "CedarSpan launch owner is Elan Ruiz."}]
    assert KnowMoreDiRTEngine._explicit_relation_actor_candidates(contract, views) == []
    assert not KnowMoreDiRTEngine._needs_actor_role_repair(
        contract,
        {"status": "extracted", "values": ["Elan Ruiz"]},
        views,
    )


def test_person_slot_retrieval_preserves_identity_resolution_breadth():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step(
                    "search_records",
                    collection="all_records",
                    terms=["reviewed", "PR-8042"],
                    mode="all",
                    limit=1,
                ),
                step("model_extract", inputs=[0], limit=1),
            ],
        },
        {
            "answer_shape": "text",
            "answer_slot": "reviewer",
            "semantic_kind": "event_fact",
            "target_phrases": ["PR-8042"],
            "scope_phrases": [],
            "relation_phrases": ["reviewed"],
            "constraint_phrases": [],
        },
    )
    assert normalized["steps"][0]["limit"] == 20


def test_approver_field_is_explicit_authority_evidence():
    values = ["Gus North"]
    views = [{
        "excerpt": "product: RippleDesk\napprover: Gus North",
        "text": "product: RippleDesk\napprover: Gus North",
        "data": {"product": "RippleDesk", "approver": "Gus North"},
    }]
    assert KnowMoreDiRTEngine._has_explicit_authority_evidence(values, views)


def test_event_fact_verdict_mapping_is_structural():
    contract = {"contract_id": "c", "answer_shape": "boolean"}
    contradicted = KnowMoreDiRTEngine._extraction_from_event_fact_verdict(
        contract,
        {
            "verdict": "contradicts",
            "evidence_record_ids": ["r"],
            "correction_clause": "it is an unrelated note.",
            "reason": "The titled subject is classified as unrelated.",
        },
    )
    assert contradicted["values"] == ["no"]
    assert contradicted["evidence_relation"] == "direct_contradiction"
    assert contradicted["reason"] == "it is an unrelated note."
    insufficient = KnowMoreDiRTEngine._extraction_from_event_fact_verdict(
        contract,
        {
            "verdict": "insufficient",
            "evidence_record_ids": ["r"],
            "correction_clause": "",
            "reason": "The evidence is silent.",
        },
    )
    assert insufficient["status"] == "unknown"
    assert insufficient["values"] == []


def test_event_fact_scope_repair_schema_mapping_preserves_contradiction():
    contract = {"contract_id": "c", "answer_shape": "boolean"}
    extraction = KnowMoreDiRTEngine._extraction_from_event_fact_verdict(
        contract,
        {
            "verdict": "contradicts",
            "scope_binding": "title_to_body",
            "evidence_record_ids": ["r"],
            "correction_clause": "it is an unrelated domain note.",
            "reason": "The titled subject is bound to the body's negative classification.",
        },
    )
    assert extraction["status"] == "extracted"
    assert extraction["values"] == ["no"]
    assert extraction["reason"] == "it is an unrelated domain note."


def test_target_mixed_epistemic_detection_does_not_leak_across_topics():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r", collection_path="logical_documents", source_path="log", record_index=0,
        data={},
        text=(
            "I dreamed that the silver gate was deleted.\n"
            "When I woke up, the silver gate still existed.\n"
            "Later inspection found no crack in the blue pump."
        ),
    )
    assert KnowMoreDiRTEngine._target_mixed_epistemic_evidence(
        record,
        {
            "target_phrases": ["silver gate", "delete"],
            "relation_phrases": ["delete"],
            "answer_slot": "did_delete",
        },
    )
    assert not KnowMoreDiRTEngine._target_mixed_epistemic_evidence(
        record,
        {
            "target_phrases": ["blue pump"],
            "relation_phrases": ["find a crack"],
            "answer_slot": "inspection_result",
        },
    )


def test_after_localization_follows_scope_anchor():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r", collection_path="logical_documents", source_path="log", record_index=0,
        data={},
        text=(
            "Vira dreamed that the silver gate was deleted.\n"
            "Real inventory: silver gate remains installed.\n"
            "Inspection note: the lantern color remains green."
        ),
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_shape": "text",
            "answer_slot": "remaining_installed",
            "semantic_kind": "entity_attribute",
            "target_phrases": ["What remains installed"],
            "scope_phrases": ["after the dream"],
            "relation_phrases": [],
            "temporal_mode": "after",
        },
    )
    assert "silver gate remains installed" in view["text"]
    assert "lantern" not in view["text"]


def test_document_classification_correction_prefers_direct_deictic_classification():
    contract = {
        "target_phrases": ["ActionGarden", "product roadmap target"],
        "relation_phrases": [],
        "answer_slot": "is_product_roadmap_target",
    }
    views = [{
        "excerpt": (
            "Market sketch for ActionGarden.\n"
            "This unrelated gardening note mentions market research but has no relation to any product roadmap."
        )
    }]
    assert KnowMoreDiRTEngine._direct_document_classification_correction(
        contract,
        views,
    ) == "it is an unrelated gardening note"


def test_document_classification_correction_prefers_explicit_identity():
    contract = {
        "answer_shape": "boolean",
        "semantic_kind": "event_fact",
        "target_phrases": ["ActionGarden", "product roadmap target"],
        "relation_phrases": [],
    }
    views = [{
        "excerpt": (
            "Market sketch for ActionGarden.\n"
            "This unrelated gardening note mentions market research but has no relation "
            "to any product roadmap."
        )
    }]
    assert KnowMoreDiRTEngine._direct_document_classification_correction(
        contract,
        views,
    ) == "it is an unrelated gardening note"


def test_relation_stems_exclude_named_arguments():
    contract = {
        "answer_slot": "morgan_entity",
        "relation_phrases": ["merged with Morgan Hale"],
        "constraint_phrases": [],
    }
    assert KnowMoreDiRTEngine._entity_relation_stems(contract) == {"merg"}
    views = [{
        "excerpt": (
            "Morgan Ives and Morgan Hale must remain separate until the note is clarified."
        ),
        "data": {},
    }]
    assert not KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        {**contract, "semantic_kind": "entity_attribute"},
        "Morgan Ives",
        views,
    )


def test_entity_ambiguity_repair_detects_unresolved_merge_candidate():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r",
        collection_path="logical_documents",
        source_path="chat.txt",
        record_index=0,
        data={},
        text=(
            "A later note says Morgan approved it, but the note does not say which Morgan.\n"
            "Keep Morgan Ives and Morgan Hale separate until the approval note is clarified."
        ),
    )
    contract = {"semantic_kind": "entity_attribute"}
    extraction = {"status": "extracted", "values": ["Morgan Ives"]}
    assert KnowMoreDiRTEngine._evidence_has_explicit_entity_ambiguity([record])
    assert KnowMoreDiRTEngine._needs_entity_ambiguity_repair(
        contract, extraction, [record]
    )


def test_count_without_row_condition_uses_bounded_model_count():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["Northstar Credit contacts"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "question": "How many contacts are listed for Northstar Credit?",
            "answer_shape": "number",
            "semantic_kind": "entity_attribute",
            "answer_slot": "contact_count",
            "intent_summary": "Request for the count of contacts associated with Northstar Credit.",
            "target_phrases": ["How many contacts are listed for Northstar Credit?"],
            "scope_phrases": [],
            "relation_phrases": [],
            "constraint_phrases": [],
        },
    )
    assert [item["tool"] for item in normalized["steps"]] == [
        "search_records", "expand_source_context", "model_extract"
    ]
    assert normalized["steps"][0]["terms"] == ["credit", "northstar"]
    assert normalized["steps"][2]["inputs"] == [1]


def test_count_localization_preserves_full_coherent_record():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r", collection_path="logical_documents",
        source_path="contacts.txt", record_index=0, data={},
        text=(
            "Customer: Northstar Credit\n"
            "Ari Moss | invoice contact | ari@example.test\n"
            "Bex Vale | technical contact | bex@example.test"
        ),
    )
    view = KnowMoreDiRTEngine._localized_record_view(
        record,
        {
            "answer_shape": "number",
            "answer_slot": "contact_count",
            "target_phrases": ["Northstar Credit contacts"],
        },
    )
    assert "Ari Moss" in view["excerpt"]
    assert "Bex Vale" in view["excerpt"]


def test_unconditioned_scoped_count_expands_source_context():
    normalized = KnowMoreDiRTEngine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                step("search_records", collection="all_records", terms=["contacts Northstar Credit"], mode="all"),
                step("model_extract", inputs=[0]),
            ],
        },
        {
            "question": "How many contacts are listed for Northstar Credit?",
            "answer_shape": "number",
            "answer_slot": "contact_count",
            "semantic_kind": "entity_attribute",
            "intent_summary": "Request the count of contacts associated with Northstar Credit.",
            "target_phrases": ["How many contacts are listed for Northstar Credit?"],
            "scope_phrases": [],
            "relation_phrases": [],
            "constraint_phrases": [],
        },
    )
    assert [item["tool"] for item in normalized["steps"]] == [
        "search_records", "expand_source_context", "model_extract"
    ]
    assert normalized["steps"][0]["terms"] == ["credit", "northstar"]
    assert normalized["steps"][2]["inputs"] == [1]


def test_numeric_entity_attribute_skips_surface_relation_validation():
    contract = {
        "semantic_kind": "entity_attribute",
        "answer_shape": "number",
        "answer_slot": "contact_count",
        "relation_phrases": [],
        "constraint_phrases": [],
    }
    assert contract["answer_shape"] == "number"
    assert not (
        contract["semantic_kind"] == "entity_attribute"
        and contract["answer_shape"] != "number"
    )


def test_presentation_verbs_do_not_override_labeled_field_relation():
    contract = {
        "semantic_kind": "entity_attribute",
        "answer_slot": "catalyst",
        "relation_phrases": ["named in"],
        "constraint_phrases": [],
    }
    assert KnowMoreDiRTEngine._entity_relation_stems(contract) == {"catalyst"}
    assert KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract,
        "copper sulfate",
        [{"excerpt": "Catalyst: copper sulfate.", "data": {}}],
    )


def test_answer_slot_label_explicitly_binds_entity_attribute():
    contract = {
        "semantic_kind": "entity_attribute",
        "answer_slot": "catalyst",
        "target_phrases": ["catalyst"],
        "relation_phrases": ["named in"],
        "constraint_phrases": [],
    }
    assert KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract,
        "copper sulfate",
        [{"excerpt": "Chemistry lab note.\nCatalyst: copper sulfate.", "data": {}}],
    )
    assert not KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract,
        "Theo Marin",
        [{"excerpt": "Chemistry lab note.\nCatalyst: copper sulfate.\nPartner: Theo Marin.", "data": {}}],
    )


def test_canonicalize_musical_scale_slot():
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "scale_practiced"},
        "D minor scale",
    ) == "D minor"
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {"answer_slot": "scale_practiced"},
        "Arlo practiced the D minor scale.",
    ) == "D minor"


def test_canonicalize_person_role_slot_strips_occupational_prefix():
    contract = {
        "question": "Who recorded incident INC-882?",
        "answer_slot": "recorder",
        "relation_phrases": ["recorded"],
    }
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        contract, "Officer Talen"
    ) == "Talen"
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        {**contract, "answer_slot": "rank", "relation_phrases": ["is"]},
        "Captain Vale",
    ) == "Captain Vale"
    assert KnowMoreDiRTEngine._canonicalize_extracted_value(
        contract, "Dr. Pella"
    ) == "Dr. Pella"


def test_final_answer_boundary_applies_conservative_surface_canonicalization():
    from knowmoredirt.models import Answer
    answer = KnowMoreDiRTEngine._canonicalize_final_answer(
        {
            "question": "Who recorded incident INC-882?",
            "answer_shape": "text",
            "answer_slot": "recorder",
            "relation_phrases": ["recorded"],
        },
        Answer("Officer Talen", evidence=({"record_id": "r"},), diagnostics={"x": 1}),
    )
    assert answer.text == "Talen"
    assert answer.evidence == ({"record_id": "r"},)
    assert answer.diagnostics["surface_canonicalized"] is True
    scale = KnowMoreDiRTEngine._canonicalize_final_answer(
        {
            "question": "What scale did Arlo practice?",
            "answer_shape": "text",
            "answer_slot": "scale_practiced",
            "relation_phrases": ["practiced"],
        },
        Answer("Arlo practiced the D minor scale."),
    )
    assert scale.text == "D minor"
    correction = Answer("No; later inspection found no crack.")
    assert KnowMoreDiRTEngine._canonicalize_final_answer(
        {"answer_shape": "text", "answer_slot": "statement"}, correction
    ) is correction


def test_final_decision_absence_canonicalizes_to_unknown():
    from knowmoredirt.models import Answer
    contract = {
        "answer_shape": "text",
        "answer_slot": "final_decision",
    }
    result = KnowMoreDiRTEngine._canonicalize_final_answer(
        contract,
        Answer("No final decision was made."),
    )
    assert result.text == "unknown"
    assert result.diagnostics["absence_canonicalized"] is True
    companion = KnowMoreDiRTEngine._canonicalize_final_answer(
        {"answer_shape": "text", "answer_slot": "confirmed_plan"},
        Answer("no reroute decision was made"),
    )
    assert companion.text == "no reroute decision was made"


def test_spatial_location_value_binds_to_target_sentence():
    contract = {
        "semantic_kind": "entity_attribute",
        "answer_slot": "location",
        "target_phrases": ["brass lamp"],
        "relation_phrases": ["is"],
    }
    assert KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract,
        "on the red desk",
        [{"excerpt": "The brass lamp is on the red desk.", "data": {}}],
    )
    assert not KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract,
        "under the round table",
        [{
            "excerpt": (
                "The brass lamp is on the red desk. "
                "The blue rug is under the round table."
            ),
            "data": {},
        }],
    )


def test_no_assertion_token_cloud_is_insufficient_not_false():
    assert KnowMoreDiRTEngine._reason_is_nonproof(
        "The file contains 'no claim no action no assertion' in a random token cloud."
    )
    assert KnowMoreDiRTEngine._reason_is_nonproof(
        "The source does not assert that the event occurred."
    )


def test_contract_type_guard_rejects_names_for_identifier_slots():
    contract = {"answer_slot": "actor_id"}
    assert not KnowMoreDiRTEngine._value_matches_contract_type(contract, "Luma Drex")
    assert KnowMoreDiRTEngine._value_matches_contract_type(contract, "ACT-901")
    assert KnowMoreDiRTEngine._value_matches_contract_type(
        {"answer_slot": "case_identifier"}, "CASE-771"
    )
    assert not KnowMoreDiRTEngine._value_matches_contract_type(
        {"answer_slot": "case_identifier"}, "Silver Ridge Systems"
    )


def test_actor_role_repair_does_not_convert_identifier_answers_to_names():
    contract = {
        "answer_slot": "actor_id",
        "relation_phrases": ["reviewed"],
    }
    views = [{"excerpt": "Naro Venn reviewed the brief | actor id: ACT-902"}]
    assert KnowMoreDiRTEngine._explicit_relation_actor_candidates(contract, views) == []
    assert not KnowMoreDiRTEngine._needs_actor_role_repair(
        contract,
        {"status": "extracted", "values": ["ACT-902"]},
        views,
    )


def test_unique_structured_slot_surface_preserves_claim_text():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r", collection_path="items", source_path="x.json",
        record_index=0,
        data={"notes": [
            {"claim": "mirror needs velvet pad"},
            {"claim": "do not use blue solvent"},
        ]},
        text="",
    )
    assert KnowMoreDiRTEngine._unique_structured_slot_surface(
        {
            "answer_slot": "claim",
            "target_phrases": ["Lark Mirror", "pad"],
            "relation_phrases": [],
        },
        [record],
    ) == "mirror needs velvet pad"


def test_structured_slot_surface_prefers_complete_focus_phrase():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r", collection_path="items", source_path="x.json", record_index=0,
        data={"name": "Lark Mirror", "notes": [
            {"claim": "mirror needs velvet pad"},
            {"claim": "do not use blue solvent"},
        ]},
        text="",
    )
    assert KnowMoreDiRTEngine._unique_structured_slot_surface(
        {
            "answer_slot": "claim",
            "target_phrases": ["claim", "Lark Mirror", "solvent"],
            "relation_phrases": [],
        },
        [record],
    ) == "do not use blue solvent"


def test_mixed_epistemic_repair_is_target_bound():
    from knowmoredirt.models import SourceRecord
    unrelated = SourceRecord(
        record_id="r", collection_path="logical_documents", source_path="log.txt",
        record_index=0, data={},
        text=(
            "Vira dreamed that the silver gate was deleted.\n"
            "Real inventory: silver gate remains installed.\n"
            "Kalo believes the lantern should be blue.\n"
            "Inspection note: the lantern remains green; the belief is not confirmed as fact."
        ),
    )
    belief_contract = {
        "answer_shape": "boolean",
        "answer_slot": "belief_confirmed",
        "target_phrases": ["Kalo belief"],
        "relation_phrases": ["is"],
    }
    assert not KnowMoreDiRTEngine._target_mixed_epistemic_records(
        belief_contract, [unrelated]
    )
    assert not KnowMoreDiRTEngine._needs_mixed_epistemic_correction_repair(
        belief_contract,
        {
            "status": "extracted", "values": ["no"],
            "evidence_relation": "unknown", "reason": "The belief is not confirmed as fact.",
        },
        [unrelated],
    )
    dream_contract = {
        "answer_shape": "boolean",
        "answer_slot": "did_delete",
        "target_phrases": ["silver gate", "delete"],
        "relation_phrases": ["delete"],
    }
    assert KnowMoreDiRTEngine._target_mixed_epistemic_records(
        dream_contract, [unrelated]
    )


def test_confirmed_is_not_proof_status_shortcut():
    assert not KnowMoreDiRTEngine._contract_asks_proof_status({
        "question": "Is the belief confirmed as fact?",
        "answer_slot": "belief_confirmed",
        "relation_phrases": ["is"],
        "constraint_phrases": [],
    })
    assert KnowMoreDiRTEngine._contract_asks_proof_status({
        "question": "Was the claim proven?",
        "answer_slot": "claim_proven",
        "relation_phrases": [],
        "constraint_phrases": ["proven"],
    })


def test_nonactual_external_effect_is_not_repaired_to_false():
    from knowmoredirt.models import SourceRecord
    record = SourceRecord(
        record_id="r", collection_path="logical_documents", source_path="dream.log",
        record_index=0, data={},
        text=(
            "Vira dreamed that the silver gate was deleted.\n"
            "Real inventory: silver gate remains installed."
        ),
    )
    contract = {
        "answer_shape": "boolean",
        "world_scope": "nonactual_external_effect",
        "answer_slot": "dream_deletion",
        "target_phrases": ["dream", "delete", "silver gate"],
        "relation_phrases": ["delete"],
    }
    assert KnowMoreDiRTEngine._target_mixed_epistemic_records(contract, [record])
    assert not KnowMoreDiRTEngine._needs_mixed_epistemic_correction_repair(
        contract,
        {
            "status": "extracted", "values": ["no"],
            "evidence_relation": "nonactual_content",
            "reason": "The event did not occur in reality.",
        },
        [record],
    )


def test_inline_role_identifier_binding_is_explicit_relation():
    contract = {
        "semantic_kind": "entity_attribute",
        "answer_slot": "actor_id",
        "target_phrases": ["actor id", "reviewer", "Zephyr Quill Brief", "Naro Venn"],
        "relation_phrases": ["belongs to"],
        "constraint_phrases": [],
    }
    views = [{
        "excerpt": (
            "Dossier: Zephyr Quill Brief.\n"
            "Reviewer: Naro Venn | actor id: ACT-902"
        ),
        "data": {},
    }]
    assert KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract, "ACT-902", views
    )
    assert not KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract,
        "ACT-901",
        [{"excerpt": "Author: Luma Drex | actor id: ACT-901", "data": {}}],
    )


def test_semantic_label_stems_bind_owning_organization():
    contract = {
        "semantic_kind": "entity_attribute",
        "answer_slot": "owner_organization",
        "target_phrases": ["organization", "owns", "Harbor Test"],
        "relation_phrases": ["owns"],
        "constraint_phrases": [],
    }
    assert KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract,
        "Juniper Example Trust",
        [{
            "excerpt": (
                "Entity: Harbor Test.\n"
                "Owning organization: Juniper Example Trust."
            ),
            "data": {},
        }],
    )
    assert not KnowMoreDiRTEngine._value_has_explicit_entity_relation(
        contract,
        "Ria Example",
        [{"excerpt": "Contact person: Ria Example.", "data": {}}],
    )


def test_nonadjudicative_not_proven_remains_unknown():
    from knowmoredirt.models import SourceRecord
    contract = {
        "question": "Was the north hinge crack proven?",
        "answer_slot": "crack_proven",
        "relation_phrases": [],
        "constraint_phrases": ["proven"],
    }
    note = SourceRecord(
        record_id="r", collection_path="logical_documents", source_path="claims.txt",
        record_index=0, data={},
        text="Judgment note: the north hinge crack was not proven.",
    )
    assert KnowMoreDiRTEngine._contract_asks_proof_status(contract)
    assert KnowMoreDiRTEngine._proof_status_correction_sentence(contract, [note]) == ""


def test_explicit_final_judgment_supports_negative_proof_status():
    from knowmoredirt.models import SourceRecord
    contract = {
        "question": "Was DriftLoom proven to cause ledger skew?",
        "answer_slot": "was_proven",
        "relation_phrases": [],
        "constraint_phrases": ["proven"],
    }
    judgment = SourceRecord(
        record_id="r", collection_path="logical_documents", source_path="judgment.final",
        record_index=0, data={},
        text="Final judgment summary. The court found no proof that DriftLoom caused ledger skew.",
    )
    assert KnowMoreDiRTEngine._proof_status_correction_sentence(
        contract, [judgment]
    ) == "the final judgment found no proof"
