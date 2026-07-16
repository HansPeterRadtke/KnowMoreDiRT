from __future__ import annotations

import json
import re

from knowmoredirt.engine import KnowMoreDiRTEngine, ProgramValidationError
from knowmoredirt.models import ToolResult
from knowmoredirt.schemas import semantic_contract_schema


def _enum(schema, *path):
    node = schema
    for item in path:
        node = node[item]
    return node["enum"][0]


def _contract(schema, *, shape="text", slot="value"):
    properties = schema["properties"]["semantic_contract"]["properties"]
    return {
        "contract_id": properties["contract_id"]["enum"][0],
        "question": properties["question"]["enum"][0],
        "intent_summary": "Return the requested value.",
        "answer_shape": shape,
        "answer_slot": slot,
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
        "reporting_tense": "none",
        "requires_explicit_evidence": True,
        "compound_request": False,
    }


def _compact_step(tool, **overrides):
    step = {
        "tool": tool,
        "inputs": [],
        "collection": "",
        "terms": [],
        "fields": [],
        "filters": [],
        "arguments": [],
        "limit": 20,
    }
    step.update(overrides)
    return step


class PipelineModel:
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
                    "summary": "Generic records.",
                    "collections": [],
                    "general_notes": "Use observed fields.",
                }
            }
        if stage == "semantic_contract":
            return {"semantic_contract": _contract(schema)}
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
                        _compact_step(
                            "search_records",
                            collection="all_records",
                            terms=["Cedar", "owner"],
                            arguments=[
                                {
                                    "name": "mode",
                                    "value": "all",
                                    "values": [],
                                    "numbers": [],
                                }
                            ],
                        )
                    ],
                }
            }
        if stage == "terminal_record_answer":
            contract_id = _enum(
                schema, "properties", "grounded_answer", "properties", "contract_id"
            )
            answer_shape = schema["properties"]["grounded_answer"]["properties"]["answer_shape"]["enum"][0]
            return {
                "grounded_answer": {
                    "contract_id": contract_id,
                    "status": "unknown",
                    "answer": "",
                    "answer_shape": answer_shape,
                    "evidence_record_ids": [],
                    "derivation": "unknown",
                    "confidence": 0.0,
                    "reason": "A further review or source is required.",
                }
            }
        if stage == "evidence_review":
            contract_id = _enum(
                schema,
                "properties",
                "evidence_review",
                "properties",
                "contract_id",
            )
            record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
            return {
                "evidence_review": {
                    "contract_id": contract_id,
                    "status": "answered",
                    "answer": "Mara",
                    "answer_items": [],
                    "answer_shape": "text",
                    "evidence_record_ids": [record_id],
                    "searches": [],
                    "confidence": 1.0,
                    "reason": "Directly supported.",
                }
            }
        raise AssertionError(stage)


def test_engine_uses_model_owned_contract_program_and_review(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner is Mara.")
    model = PipelineModel()
    answer = KnowMoreDiRTEngine(tmp_path, model=model).answer("Who owns Cedar?")
    assert answer.text == "Mara"
    assert answer.evidence[0]["source_path"] == "note.txt"
    assert [call[0] for call in model.calls] == [
        "dataset_profile",
        "semantic_contract",
        "query_program",
        "evidence_review",
    ]


def test_semantic_prompt_distinguishes_labeled_values_from_definitions(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner is Mara.")

    class PromptGuardModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                assert "specific labeled-value request" in prompt
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=PromptGuardModel()).answer(
        "Who owns Cedar?"
    )
    assert answer.text == "Mara"


def test_followup_search_is_model_owned_and_supports_multi_hop(tmp_path):
    (tmp_path / "reports.json").write_text(
        json.dumps([{"title": "Quarterly Review", "authors": ["Rhea Vale"]}])
    )
    (tmp_path / "directory.json").write_text(
        json.dumps([{"name": "Rhea Vale", "employee_id": "E-9"}])
    )

    class MultiHopModel(PipelineModel):
        def __init__(self):
            super().__init__()
            self.review_count = 0

        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {
                    "semantic_contract": _contract(
                        schema, shape="identifier", slot="employee identifier"
                    )
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
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["Quarterly Review"],
                                arguments=[
                                    {
                                        "name": "mode",
                                        "value": "phrase",
                                        "values": [],
                                        "numbers": [],
                                    }
                                ],
                            )
                        ],
                    }
                }
            if stage == "evidence_review":
                self.review_count += 1
                contract_id = _enum(
                    schema,
                    "properties",
                    "evidence_review",
                    "properties",
                    "contract_id",
                )
                if self.review_count == 1:
                    assert "Rhea Vale" in prompt
                    return {
                        "evidence_review": {
                            "contract_id": contract_id,
                            "status": "search",
                            "answer": "",
                            "answer_items": [],
                            "answer_shape": "identifier",
                            "evidence_record_ids": [],
                            "searches": [
                                {
                                    "collection": "all_records",
                                    "terms": ["Rhea Vale"],
                                    "mode": "phrase",
                                    "fields": [],
                                    "limit": 20,
                                }
                            ],
                            "confidence": 0.5,
                            "reason": "Resolve the observed name through another source.",
                        }
                    }
                record_ids = re.findall(r'"record_id":\s*"([^"]+)"', prompt)
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "E-9",
                        "answer_items": [],
                        "answer_shape": "identifier",
                        "evidence_record_ids": [record_ids[-1]],
                        "searches": [],
                        "confidence": 1.0,
                        "reason": "The follow-up record binds the name to the identifier.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=MultiHopModel()).answer(
        "What is the employee identifier for the author of the Quarterly Review?"
    )
    assert answer.text == "E-9"
    assert any("directory.json" == item["source_path"] for item in answer.evidence)


def test_review_prompt_allows_specific_labeled_value_over_definition_contract(tmp_path):
    (tmp_path / "map.txt").write_text(
        "Route diagram notes.\nPriority route: Gate P -> Gate Q.\n"
    )

    class LabeledValueModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="text", slot="definition")
                contract["semantic_kind"] = "definition"
                contract["intent_summary"] = "Provide a definition."
                contract["target_phrases"] = ["priority route"]
                contract["relation_phrases"] = ["is"]
                return {"semantic_contract": contract}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["priority route"],
                            ),
                            _compact_step(
                                "project_values",
                                inputs=[0],
                                fields=["Priority route"],
                            ),
                            _compact_step("model_extract", inputs=[1]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "unknown",
                        "values": [],
                        "answer_shape": "text",
                        "evidence_record_ids": [],
                        "evidence_relation": "absence",
                        "reason": "No glossary definition is present.",
                    }
                }
            if stage == "evidence_review":
                assert "direct label-value pair" in prompt
                contract_id = _enum(
                    schema, "properties", "evidence_review", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "Gate P -> Gate Q",
                        "answer_items": [],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "searches": [],
                        "confidence": 1.0,
                        "reason": "The labeled value is directly supplied.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=LabeledValueModel()).answer(
        "What is the priority route?"
    )
    assert answer.text == "Gate P -> Gate Q"


def test_program_normalization_is_structural_and_preserves_model_terms(tmp_path):
    (tmp_path / "note.txt").write_text("alpha")
    engine = KnowMoreDiRTEngine(tmp_path, model=PipelineModel())
    program = engine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                _compact_step(
                    "search_records",
                    collection="all_records",
                    terms=["literal relation phrase"],
                    limit=999999,
                ),
                _compact_step(
                    "search_records",
                    inputs=[0],
                    terms=["second literal"],
                ),
            ],
        }
    )
    assert program["steps"][0]["terms"] == ["literal relation phrase"]
    assert program["steps"][1]["terms"] == ["second literal"]
    assert program["steps"][0]["limit"] == 5000


def test_single_input_join_normalizes_to_source_context_expansion(tmp_path):
    (tmp_path / "note.txt").write_text("alpha")
    engine = KnowMoreDiRTEngine(tmp_path, model=PipelineModel())
    program = engine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                _compact_step("search_records", collection="all_records", terms=["alpha"]),
                _compact_step("join_records", inputs=[0, 1], collection="records"),
                _compact_step("project_values", inputs=[1], fields=["value"]),
            ],
        }
    )
    assert program["steps"][1]["tool"] == "expand_source_context"
    assert program["steps"][1]["inputs"] == [0]


def test_root_searches_are_bound_to_model_owned_contract_surfaces(tmp_path):
    (tmp_path / "note.txt").write_text("alpha")
    engine = KnowMoreDiRTEngine(tmp_path, model=PipelineModel())
    program = {
        "contract_id": "c",
        "steps": [
            _compact_step(
                "search_records",
                collection="all_records",
                terms=["generic field label"],
            ),
            _compact_step("model_extract", inputs=[0]),
        ],
    }
    contract = {
        "target_phrases": ["specific target"],
        "constraint_phrases": ["specific scope"],
        "scope_phrases": [],
        "relation_phrases": ["requested relation"],
    }
    bound = engine._bind_root_searches_to_contract(program, contract)
    assert bound["steps"][0]["terms"] == ["generic field label"]
    assert bound["steps"][0]["_contract_terms"] == [
        "specific target",
        "specific scope",
        "requested relation",
    ]
    assert bound["steps"][1]["terms"] == []


def test_contract_terms_are_advisory_for_conjunctive_root_search(tmp_path):
    (tmp_path / "route_note.txt").write_text(
        "Route comparison memo.\nCounterpoint: Nia said the tram was safer.\n"
    )
    engine = KnowMoreDiRTEngine(tmp_path, model=PipelineModel())
    program = {
        "contract_id": "c",
        "steps": [
            _compact_step(
                "search_records",
                collection="all_records",
                terms=["tram", "safer"],
                arguments=[
                    {
                        "name": "mode",
                        "value": "all",
                        "values": [],
                        "numbers": [],
                    }
                ],
            )
        ],
    }
    contract = {
        "target_phrases": ["the tram"],
        "constraint_phrases": [],
        "scope_phrases": ["position"],
        "relation_phrases": ["said that ... was safer"],
    }
    bound = engine._bind_root_searches_to_contract(program, contract)
    result = engine.executor.execute(bound["steps"])[0]
    assert bound["steps"][0]["terms"] == ["tram", "safer"]
    assert "position" in bound["steps"][0]["_contract_terms"]
    assert [record.source_path for record in result.records] == ["route_note.txt"]
    assert "position" in result.diagnostics["contract_terms"]


def test_unknown_deterministic_extractor_routes_to_model_extract(tmp_path):
    (tmp_path / "note.txt").write_text("Ravi authored the essay.")
    engine = KnowMoreDiRTEngine(tmp_path, model=PipelineModel())
    program = engine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                _compact_step(
                    "search_records",
                    collection="all_records",
                    terms=["essay"],
                ),
                _compact_step(
                    "extract_values",
                    inputs=[0],
                    terms=["authored"],
                    fields=["text"],
                    arguments=[
                        {
                            "name": "extractor",
                            "value": "author_extractor",
                            "values": [],
                            "numbers": [],
                        }
                    ],
                ),
            ],
        }
    )
    assert program["steps"][1]["tool"] == "model_extract"
    assert program["steps"][1]["inputs"] == [0]
    assert program["steps"][1]["terms"] == ["authored"]
    assert program["steps"][1]["fields"] == ["text"]
    assert program["steps"][1]["arguments"] == []

    supported = engine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                _compact_step(
                    "extract_values",
                    arguments=[
                        {
                            "name": "extractor",
                            "value": "url",
                            "values": [],
                            "numbers": [],
                        }
                    ],
                )
            ],
        }
    )
    assert supported["steps"][0]["tool"] == "extract_values"


def test_program_validation_rejects_invalid_dependencies(tmp_path):
    (tmp_path / "note.txt").write_text("alpha")
    engine = KnowMoreDiRTEngine(tmp_path, model=PipelineModel())
    contract = {"contract_id": "c"}
    program = {
        "contract_id": "c",
        "steps": [_compact_step("join_records", inputs=[0])],
    }
    try:
        engine._validate_program(contract, program)
    except ProgramValidationError:
        pass
    else:
        raise AssertionError("invalid self-referential join was accepted")


def test_review_rejects_unavailable_evidence_ids(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner is Mara.")

    class BadReview(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "evidence_review":
                contract_id = _enum(
                    schema,
                    "properties",
                    "evidence_review",
                    "properties",
                    "contract_id",
                )
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "Mara",
                        "answer_items": [],
                        "answer_shape": "text",
                        "evidence_record_ids": ["invented-record"],
                        "searches": [],
                        "confidence": 1.0,
                        "reason": "Unsupported.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=BadReview()).answer("Who owns Cedar?")
    assert answer.text == "unknown"
    assert answer.diagnostics["reason"] == "ProgramValidationError"


def test_review_search_status_discards_placeholder_answer_text(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner is Mara.")
    engine = KnowMoreDiRTEngine(tmp_path, model=PipelineModel())
    contract = {"contract_id": "c", "answer_shape": "text"}
    review = {
        "contract_id": "c",
        "status": "search",
        "answer": "continue searching",
        "answer_items": ["not an answer"],
        "answer_shape": "text",
        "evidence_record_ids": [],
        "searches": [
            {
                "collection": "all_records",
                "terms": ["Cedar"],
                "mode": "all",
                "fields": [],
                "limit": 10,
            }
        ],
        "confidence": 0.2,
        "reason": "More evidence is needed.",
    }
    normalized = engine._normalize_review(contract, review, {})
    assert normalized["status"] == "search"
    assert normalized["answer"] == ""
    assert normalized["answer_items"] == []
    assert normalized["searches"] == review["searches"]


def test_review_normalization_accepts_grounded_items_with_inline_source_tags(tmp_path):
    (tmp_path / "glossary.txt").write_text(
        "Glossary entry: the requested form for zorb is zorbette."
    )
    engine = KnowMoreDiRTEngine(tmp_path, model=PipelineModel())
    record = engine.catalog.preferred_records()[0]
    contract = {"contract_id": "c", "answer_shape": "list"}
    review = {
        "contract_id": "c",
        "status": "search",
        "answer": "",
        "answer_items": [f"zorbette【{record.record_id}】"],
        "answer_shape": "list",
        "evidence_record_ids": [record.record_id],
        "searches": [
            {
                "collection": "all_records",
                "terms": ["zorb"],
                "mode": "all",
                "fields": [],
                "limit": 10,
            }
        ],
        "confidence": 0.9,
        "reason": "The cited record contains the requested form.",
    }
    normalized = engine._normalize_review(
        contract,
        review,
        {0: ToolResult("0", "values", records=[record], values=["zorbette"])},
    )
    assert normalized["status"] == "answered"
    assert normalized["answer"] == ""
    assert normalized["answer_items"] == ["zorbette"]
    assert normalized["searches"] == []


def test_bare_boolean_review_requires_derived_boolean_value(tmp_path):
    (tmp_path / "archive.txt").write_text(
        "Archive note: the lantern story is decorative, not certified."
    )

    class BareBooleanReviewModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {"semantic_contract": _contract(schema, shape="boolean", slot="is_certified")}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step("search_records", collection="all_records", terms=["lantern"]),
                            _compact_step(
                                "filter_records",
                                inputs=[0],
                                filters=[
                                    {
                                        "field_path": "source.path",
                                        "operator": "contains",
                                        "value": "missing-section",
                                        "values": ["missing-section"],
                                    }
                                ],
                            ),
                        ],
                    }
                }
            if stage == "evidence_review":
                contract_id = _enum(
                    schema,
                    "properties",
                    "evidence_review",
                    "properties",
                    "contract_id",
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "false",
                        "answer_items": [],
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "searches": [],
                        "confidence": 1.0,
                        "reason": "The evidence was not converted by a boolean tool.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=BareBooleanReviewModel()).answer(
        "Is the lantern story certified?"
    )
    assert answer.text == "unknown"
    assert answer.diagnostics["reason"] == "ProgramValidationError"


def test_person_review_answer_is_rendered_to_minimal_surface(tmp_path):
    (tmp_path / "incident.txt").write_text(
        "Incident note.\nRecorder Vale logged ticket T-100 at noon.\n"
    )

    class PersonReviewRenderModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="list", slot="person")
                contract["target_phrases"] = ["T-100"]
                contract["relation_phrases"] = ["logged"]
                return {"semantic_contract": contract}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step("search_records", collection="all_records", terms=["T-100"]),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "unknown",
                        "values": [],
                        "answer_shape": "list",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "The record contains the person surface.",
                    }
                }
            if stage == "evidence_review":
                contract_id = _enum(
                    schema,
                    "properties",
                    "evidence_review",
                    "properties",
                    "contract_id",
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "",
                        "answer_items": ["Recorder Vale"],
                        "answer_shape": "list",
                        "evidence_record_ids": [record_id],
                        "searches": [],
                        "confidence": 1.0,
                        "reason": "The record identifies who logged the ticket.",
                    }
                }
            if stage == "grounded_answer":
                contract_id = _enum(
                    schema, "properties", "grounded_answer", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "grounded_answer": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "Vale",
                        "answer_shape": "list",
                        "evidence_record_ids": [record_id],
                        "derivation": "summary",
                        "confidence": 1.0,
                        "reason": "Minimal person surface.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=PersonReviewRenderModel()).answer(
        "Who logged ticket T-100?"
    )
    assert answer.text == "Vale"
    assert answer.diagnostics["grounded_answer"]["answer"] == "Vale"


def test_terminal_model_extract_entity_answer_is_rendered_to_minimal_surface(tmp_path):
    (tmp_path / "notice.txt").write_text(
        "Notice: Claimant Solen Works filed the report."
    )

    class TerminalEntityRenderModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="text", slot="claiming_entity")
                contract["target_phrases"] = ["report"]
                contract["relation_phrases"] = ["filed"]
                return {"semantic_contract": contract}
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
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["report"],
                            ),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema,
                    "properties",
                    "tool_extraction",
                    "properties",
                    "contract_id",
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["Claimant Solen Works"],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "The cited record names the filing entity.",
                    }
                }
            if stage == "grounded_answer":
                assert "Claimant Solen Works" in prompt
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
                        "answer": "Solen Works",
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "derivation": "summary",
                        "confidence": 1.0,
                        "reason": "Minimal named entity surface.",
                    }
                }
            if stage == "evidence_review":
                raise AssertionError("terminal renderer should handle role prefix")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=TerminalEntityRenderModel()).answer(
        "Who filed the report?"
    )
    assert answer.text == "Solen Works"
    assert answer.diagnostics["grounded_answer"]["answer"] == "Solen Works"


def test_unchanged_terminal_entity_rendering_falls_back_to_review(tmp_path):
    (tmp_path / "notice.txt").write_text(
        "Notice: Claimant Orro Labs filed the report."
    )

    class UnchangedTerminalEntityModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="text", slot="claiming_entity")
                contract["target_phrases"] = ["report"]
                contract["relation_phrases"] = ["filed"]
                return {"semantic_contract": contract}
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
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["report"],
                            ),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema,
                    "properties",
                    "tool_extraction",
                    "properties",
                    "contract_id",
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["Claimant Orro Labs"],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "The cited record names the filing entity.",
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
                        "answer": "Claimant Orro Labs",
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "derivation": "summary",
                        "confidence": 1.0,
                        "reason": "Renderer preserved the source span.",
                    }
                }
            if stage == "evidence_review":
                contract_id = _enum(
                    schema,
                    "properties",
                    "evidence_review",
                    "properties",
                    "contract_id",
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "Orro Labs",
                        "answer_items": [],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "searches": [],
                        "confidence": 1.0,
                        "reason": "The role label is not part of the entity name.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=UnchangedTerminalEntityModel()).answer(
        "Who filed the report?"
    )
    assert answer.text == "Orro Labs"
    assert answer.diagnostics["review"]["status"] == "answered"


def test_model_extract_is_bounded_and_evidence_cited(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner is Mara.")

    class ExtractModel(PipelineModel):
        def __init__(self):
            super().__init__()
            self.extract_called = False

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
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["Cedar", "owner"],
                                arguments=[
                                    {
                                        "name": "mode",
                                        "value": "all",
                                        "values": [],
                                        "numbers": [],
                                    }
                                ],
                            ),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                self.extract_called = True
                contract_id = _enum(
                    schema,
                    "properties",
                    "tool_extraction",
                    "properties",
                    "contract_id",
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["Mara"],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "Explicit.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    model = ExtractModel()
    answer = KnowMoreDiRTEngine(tmp_path, model=model).answer("Who owns Cedar?")
    assert answer.text == "Mara"
    assert model.extract_called is True


def test_absence_relation_from_scalar_model_extraction_is_not_terminal(tmp_path):
    (tmp_path / "panel.txt").write_text("Panel note: no route was selected.")

    class AbsenceModel(PipelineModel):
        def __init__(self):
            super().__init__()
            self.stages = []

        def complete_json(self, stage, prompt, schema, max_tokens=0):
            self.stages.append(stage)
            if stage == "semantic_contract":
                contract = _contract(schema, shape="text", slot="selected_route")
                contract["target_phrases"] = ["route"]
                contract["relation_phrases"] = ["selected"]
                return {"semantic_contract": contract}
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
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["route", "selected"],
                            ),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema,
                    "properties",
                    "tool_extraction",
                    "properties",
                    "contract_id",
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["No route was selected."],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "absence",
                        "reason": "The evidence says the requested value is absent.",
                    }
                }
            if stage == "evidence_review":
                assert "Derived answer candidates: []" in prompt
                contract_id = _enum(
                    schema,
                    "properties",
                    "evidence_review",
                    "properties",
                    "contract_id",
                )
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "unknown",
                        "answer": "",
                        "answer_items": [],
                        "answer_shape": "text",
                        "evidence_record_ids": [],
                        "searches": [],
                        "confidence": 0.0,
                        "reason": "The cited record supplies absence rather than a value.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    model = AbsenceModel()
    answer = KnowMoreDiRTEngine(tmp_path, model=model).answer("Which route was selected?")
    assert answer.text == "unknown"
    assert model.stages == [
        "dataset_profile",
        "semantic_contract",
        "query_program",
        "tool_extraction",
        "evidence_review",
    ]


def test_temporal_direct_support_absence_uses_terminal_record_unknown(tmp_path):
    (tmp_path / "panel.txt").write_text("Panel note: no route was selected.")

    class TemporalAbsenceModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="text", slot="selected_route")
                contract["temporal_mode"] = "final"
                contract["target_phrases"] = ["route"]
                contract["relation_phrases"] = ["selection"]
                return {"semantic_contract": contract}
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
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["route", "selection"],
                            ),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema,
                    "properties",
                    "tool_extraction",
                    "properties",
                    "contract_id",
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["No route was selected."],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "The cited record explicitly addresses the requested slot.",
                    }
                }
            if stage == "terminal_record_answer":
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
                        "status": "unknown",
                        "answer": "",
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "derivation": "extraction",
                        "confidence": 1.0,
                        "reason": "The cited record establishes absence of the requested value.",
                    }
                }
            if stage == "evidence_review":
                raise AssertionError("cited terminal absence should stop as unknown")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=TemporalAbsenceModel()).answer(
        "Which route was finally selected?"
    )
    assert answer.text == "unknown"
    assert answer.diagnostics["reason"] == "validated_terminal_record_unknown"
    assert answer.evidence


def test_terminal_record_unknown_with_cited_answer_is_promoted(tmp_path):
    (tmp_path / "timeline.txt").write_text(
        "2026-02-01 Shuttle state: queued.\n"
        "2026-02-03 Shuttle state: ready.\n"
    )

    class TerminalContradictionModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="text", slot="state")
                contract["temporal_mode"] = "current"
                contract["target_phrases"] = ["Shuttle"]
                contract["relation_phrases"] = ["state"]
                return {"semantic_contract": contract}
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
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["Shuttle"],
                            ),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema,
                    "properties",
                    "tool_extraction",
                    "properties",
                    "contract_id",
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "unknown",
                        "values": [],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "state_only",
                        "reason": "The extraction stage deferred temporal adjudication.",
                    }
                }
            if stage == "terminal_record_answer":
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
                        "status": "unknown",
                        "answer": "ready",
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "derivation": "extraction",
                        "confidence": 1.0,
                        "reason": "The latest dated state is ready.",
                    }
                }
            if stage == "evidence_review":
                raise AssertionError("cited terminal answer should not require review")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=TerminalContradictionModel()).answer(
        "What is the current state of Shuttle?"
    )
    assert answer.text == "ready"
    assert answer.diagnostics["reason"] == "validated_terminal_record_answer"
    assert answer.diagnostics["grounded_answer"]["status"] == "answered"


def test_review_repairs_citation_from_exact_derived_value_provenance(tmp_path):
    (tmp_path / "log.txt").write_text("Workshop note: Lumo tagged the sample.")

    class CitationRepairModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {"semantic_contract": _contract(schema, shape="list", slot="tagger")}
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
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["sample"],
                            ),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema,
                    "properties",
                    "tool_extraction",
                    "properties",
                    "contract_id",
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["Lumo"],
                        "answer_shape": "list",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "The cited record explicitly names the tagger.",
                    }
                }
            if stage == "evidence_review":
                contract_id = _enum(
                    schema,
                    "properties",
                    "evidence_review",
                    "properties",
                    "contract_id",
                )
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "search",
                        "answer": "",
                        "answer_items": ["Lumo"],
                        "answer_shape": "list",
                        "evidence_record_ids": ["mistyped-record-id"],
                        "searches": [
                            {
                                "collection": "all_records",
                                "terms": ["sample"],
                                "mode": "any",
                                "fields": [],
                                "limit": 20,
                            }
                        ],
                        "confidence": 1.0,
                        "reason": "The current evidence already supports the value.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=CitationRepairModel()).answer(
        "Who tagged the sample?"
    )
    assert answer.text == "Lumo"
    assert answer.diagnostics["review"]["status"] == "answered"
    assert answer.evidence
    assert answer.diagnostics["review"]["evidence_record_ids"] == [
        answer.evidence[0]["record_id"]
    ]


def test_review_repairs_citation_from_surface_grounded_value(tmp_path):
    (tmp_path / "glossary.txt").write_text("Glossary note: zeno means quiet dawn.")

    class SurfaceCitationRepairModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {"semantic_contract": _contract(schema, shape="list", slot="meaning")}
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
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["zeno"],
                            ),
                            _compact_step(
                                "project_values",
                                inputs=[0],
                                fields=["text"],
                            ),
                        ],
                    }
                }
            if stage == "evidence_review":
                contract_id = _enum(
                    schema,
                    "properties",
                    "evidence_review",
                    "properties",
                    "contract_id",
                )
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "",
                        "answer_items": ["quiet dawn"],
                        "answer_shape": "list",
                        "evidence_record_ids": ["mistyped-record-id"],
                        "searches": [],
                        "confidence": 1.0,
                        "reason": "The current evidence states the meaning.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=SurfaceCitationRepairModel()).answer(
        "What does zeno mean?"
    )
    assert answer.text == "quiet dawn"
    assert answer.diagnostics["review"]["status"] == "answered"
    assert answer.evidence


def test_state_only_evidence_does_not_negate_an_event(tmp_path):
    (tmp_path / "diary.txt").write_text(
        "A dream described an event. Morning fact: the object remained in its room."
    )

    class StateOnlyModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="boolean", slot="did_happen")
                contract["semantic_kind"] = "event_fact"
                return {"semantic_contract": contract}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step("search_records", collection="all_records", terms=["dream", "object"]),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["false"],
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "The object was present later.",
                    }
                }
            if stage == "event_fact_verdict":
                contract_id = _enum(
                    schema, "properties", "event_fact_verdict", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "event_fact_verdict": {
                        "contract_id": contract_id,
                        "verdict": "contradicts",
                        "scope_binding": "direct",
                        "evidence_basis": "state_only",
                        "evidence_record_ids": [record_id],
                        "authority_label": "",
                        "decisive_predicate": "",
                        "correction_clause": "",
                        "reason": "A later state does not prove the event never occurred.",
                    }
                }
            if stage == "grounded_answer":
                raise AssertionError("insufficient verdict must terminate as unknown")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=StateOnlyModel()).answer(
        "Did the event really happen?"
    )
    assert answer.text == "unknown"
    assert answer.diagnostics["reason"] == "inconsistent_event_evidence_relation"
    assert answer.evidence


def test_state_only_event_evidence_remains_unknown(tmp_path):
    (tmp_path / "diary.txt").write_text(
        "Nola dreamed that a train carried the table away. Morning fact: the table remained in the room."
    )

    class StateOnlyModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {"semantic_contract": _contract(schema, shape="boolean", slot="did_carry")}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step("search_records", collection="all_records", terms=["train", "table"]),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "unknown",
                        "values": [],
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "state_only",
                        "reason": "Only a resulting state and dream content are supplied.",
                    }
                }
            if stage in {"event_fact_verdict", "grounded_answer", "evidence_review"}:
                raise AssertionError("insufficient event evidence must terminate as unknown")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=StateOnlyModel()).answer(
        "Did the train really carry the table away?"
    )
    assert answer.text == "unknown"
    assert answer.diagnostics["reason"] == "insufficient_event_evidence"
    assert answer.evidence


def test_inconsistent_event_relation_and_polarity_remains_unknown(tmp_path):
    (tmp_path / "state.txt").write_text("Morning fact: the table remained in the room.")

    class InconsistentEventModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {"semantic_contract": _contract(schema, shape="boolean", slot="did_carry")}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step("search_records", collection="all_records", terms=["table"]),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["false"],
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "The state consequence was mislabeled as direct support.",
                    }
                }
            if stage in {"event_fact_verdict", "grounded_answer", "evidence_review"}:
                raise AssertionError("inconsistent extraction must terminate as unknown")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=InconsistentEventModel()).answer(
        "Did the table move?"
    )
    assert answer.text == "unknown"
    assert answer.diagnostics["reason"] == "inconsistent_event_evidence_relation"
    assert answer.evidence


def test_unbound_event_contradiction_verdict_remains_unknown(tmp_path):
    (tmp_path / "memo.txt").write_text(
        "Scenario note: Ari imagined that a bronze drone moved the lantern.\n"
        "Later state: the lantern was on the workshop shelf.\n"
    )

    class UnboundContradictionModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="boolean", slot="did_move")
                contract["semantic_kind"] = "event_fact"
                contract["target_phrases"] = ["bronze drone", "lantern"]
                contract["relation_phrases"] = ["moved"]
                return {"semantic_contract": contract}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step("search_records", collection="all_records", terms=["lantern"]),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "unknown",
                        "values": [],
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_contradiction",
                        "reason": "A later state was mistaken for a direct denial.",
                    }
                }
            if stage == "event_fact_verdict":
                contract_id = _enum(
                    schema, "properties", "event_fact_verdict", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "event_fact_verdict": {
                        "contract_id": contract_id,
                        "verdict": "contradicts",
                        "scope_binding": "none",
                        "evidence_basis": "explicit_denial",
                        "evidence_record_ids": [record_id],
                        "authority_label": "",
                        "decisive_predicate": "",
                        "correction_clause": "",
                        "reason": "The state was not bound to the event proposition.",
                    }
                }
            if stage == "grounded_answer":
                raise AssertionError("unbound contradiction must not render a No answer")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=UnboundContradictionModel()).answer(
        "Did the bronze drone move the lantern?"
    )
    assert answer.text == "unknown"
    assert answer.diagnostics["reason"] == "validated_event_fact_insufficient"
    assert answer.diagnostics["event_fact_verdict"]["verdict"] == "insufficient"
    assert answer.evidence


def test_boolean_slot_shape_is_normalized_to_boolean_verdict_path(tmp_path):
    (tmp_path / "audit.txt").write_text("Later audit found no leak in the pipe.")

    class BooleanSlotListShapeModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="list", slot="boolean")
                contract["semantic_kind"] = "entity_attribute"
                contract["target_phrases"] = ["leak", "pipe"]
                contract["relation_phrases"] = ["found"]
                contract["authority_mode"] = "explicit_official"
                return {"semantic_contract": contract}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step("search_records", collection="all_records", terms=["leak", "pipe"]),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "unknown",
                        "values": [],
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_contradiction",
                        "reason": "The audit found no leak.",
                    }
                }
            if stage == "event_fact_verdict":
                contract_id = _enum(
                    schema, "properties", "event_fact_verdict", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "event_fact_verdict": {
                        "contract_id": contract_id,
                        "verdict": "insufficient",
                        "scope_binding": "none",
                        "evidence_basis": "state_only",
                        "evidence_record_ids": [record_id],
                        "authority_label": "",
                        "decisive_predicate": "",
                        "correction_clause": "",
                        "reason": "The verifier under-classified the direct official denial.",
                    }
                }
            if stage == "grounded_answer":
                contract_id = _enum(
                    schema, "properties", "grounded_answer", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "grounded_answer": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "No; Later audit found no leak in the pipe. (source: audit.txt, record cited).",
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "derivation": "extraction",
                        "confidence": 1.0,
                        "reason": "Grounded negative rendering.",
                    }
                }
            if stage == "evidence_review":
                raise AssertionError("boolean slot should use event verdict path")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=BooleanSlotListShapeModel()).answer(
        "Was a leak found in the pipe?"
    )
    assert answer.text == "No; later audit found no leak in the pipe."
    assert answer.diagnostics["semantic_contract"]["answer_shape"] == "boolean"
    assert answer.diagnostics["reason"] == "validated_event_fact_verdict"


def test_short_negative_event_rendering_uses_grounded_reason_clause(tmp_path):
    (tmp_path / "review.txt").write_text(
        "Post-incident audit found no missing seals in the shipping crate."
    )

    class ShortNegativeRenderModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="boolean", slot="boolean")
                contract["semantic_kind"] = "event_fact"
                contract["target_phrases"] = ["missing seals", "shipping crate"]
                contract["relation_phrases"] = ["found"]
                contract["authority_mode"] = "explicit_official"
                return {"semantic_contract": contract}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["missing seals", "shipping crate"],
                            ),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "unknown",
                        "values": [],
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_contradiction",
                        "reason": "The cited audit denies the event.",
                    }
                }
            if stage == "event_fact_verdict":
                contract_id = _enum(
                    schema, "properties", "event_fact_verdict", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "event_fact_verdict": {
                        "contract_id": contract_id,
                        "verdict": "contradicts",
                        "scope_binding": "direct",
                        "evidence_basis": "explicit_denial",
                        "evidence_record_ids": [record_id],
                        "authority_label": "",
                        "decisive_predicate": "",
                        "correction_clause": "",
                        "reason": "The audit directly denied the requested proposition.",
                    }
                }
            if stage == "grounded_answer":
                contract_id = _enum(
                    schema, "properties", "grounded_answer", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "grounded_answer": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "No; audit",
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "derivation": "summary",
                        "confidence": 1.0,
                        "reason": "Post-incident audit found no missing seals in the shipping crate.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=ShortNegativeRenderModel()).answer(
        "Were missing seals found in the shipping crate?"
    )
    assert (
        answer.text
        == "No; post-incident audit found no missing seals in the shipping crate."
    )
    assert answer.diagnostics["reason"] == "validated_event_fact_verdict"


def test_review_date_prefix_preserves_unique_cited_timestamp(tmp_path):
    (tmp_path / "schedule.txt").write_text(
        "2027-03-03 08:15 kiln check.\n2027-03-04 17:45 gallery opening.\n"
    )

    class TemporalPrefixModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="list", slot="date")
                contract["semantic_kind"] = "event_fact"
                contract["target_phrases"] = ["gallery opening"]
                contract["relation_phrases"] = ["when"]
                return {"semantic_contract": contract}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["gallery opening"],
                            )
                        ],
                    }
                }
            if stage == "evidence_review":
                contract_id = _enum(
                    schema, "properties", "evidence_review", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "",
                        "answer_items": ["2027-03-04"],
                        "answer_shape": "list",
                        "evidence_record_ids": [record_id],
                        "searches": [],
                        "confidence": 1.0,
                        "reason": "The cited entry is '2027-03-04 17:45 gallery opening'.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=TemporalPrefixModel()).answer(
        "When is the gallery opening?"
    )
    assert answer.text == "2027-03-04 17:45"


def test_boolean_claim_preserves_grounded_corrective_clause(tmp_path):
    (tmp_path / "judgment.txt").write_text(
        "Final judgment summary. The tribunal found no proof that the system caused the drift."
    )

    class BooleanVerdictModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {"semantic_contract": _contract(schema, shape="boolean", slot="was_proven")}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step("search_records", collection="all_records", terms=["no proof"]),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["false"],
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_contradiction",
                        "reason": "The final judgment found no proof.",
                    }
                }
            if stage == "event_fact_verdict":
                contract_id = _enum(
                    schema, "properties", "event_fact_verdict", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "event_fact_verdict": {
                        "contract_id": contract_id,
                        "verdict": "contradicts",
                        "scope_binding": "document_scope",
                        "evidence_basis": "authoritative_not_proven",
                        "evidence_record_ids": [record_id],
                        "authority_label": "the final judgment",
                        "decisive_predicate": "found no proof",
                        "correction_clause": "the final judgment found no proof.",
                        "reason": "Authoritative correction.",
                    }
                }
            if stage == "grounded_answer":
                contract_id = _enum(
                    schema, "properties", "grounded_answer", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "grounded_answer": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "No; the final judgment found no proof.",
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "derivation": "summary",
                        "confidence": 1.0,
                        "reason": "Canonical grounded rendering.",
                    }
                }
            if stage == "evidence_review":
                raise AssertionError("boolean event verdict should be terminal")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=BooleanVerdictModel()).answer(
        "Was the system proven to have caused the drift?"
    )
    assert answer.text == "No; the final judgment found no proof."
    assert answer.diagnostics["reason"] == "validated_event_fact_verdict"


def test_boolean_verdict_is_normalized_to_grounded_terminal_boolean(tmp_path):
    (tmp_path / "judgment.txt").write_text("Final judgment found no proof.")

    class InconsistentVerdictModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {"semantic_contract": _contract(schema, shape="boolean", slot="was_proven")}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step("search_records", collection="all_records", terms=["no proof"]),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["false"],
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_contradiction",
                        "reason": "the final judgment found no proof",
                    }
                }
            if stage == "event_fact_verdict":
                contract_id = _enum(
                    schema, "properties", "event_fact_verdict", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "event_fact_verdict": {
                        "contract_id": contract_id,
                        "verdict": "supports",
                        "scope_binding": "document_scope",
                        "evidence_basis": "authoritative_not_proven",
                        "evidence_record_ids": [record_id],
                        "authority_label": "the final judgment",
                        "decisive_predicate": "found no proof",
                        "correction_clause": "",
                        "reason": "The final judgment found no proof.",
                    }
                }
            if stage == "grounded_answer":
                contract_id = _enum(
                    schema, "properties", "grounded_answer", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "grounded_answer": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "No; the final judgment found no proof",
                        "answer_shape": "boolean",
                        "evidence_record_ids": [record_id],
                        "derivation": "summary",
                        "confidence": 1.0,
                        "reason": "Canonical grounded rendering.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=InconsistentVerdictModel()).answer(
        "Was the claim proven?"
    )
    assert answer.text == "No; the final judgment found no proof."
    assert answer.diagnostics["event_fact_verdict"]["verdict"] == "contradicts"


def test_grounded_corrective_scalar_is_surfaced(tmp_path):
    (tmp_path / "timeline.txt").write_text("BUG-9 final state: closed")

    class CorrectiveModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step("search_records", collection="all_records", terms=["BUG-9"]),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["closed"],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_contradiction",
                        "reason": "The final record corrects the prior state.",
                    }
                }
            if stage == "evidence_review":
                raise AssertionError("grounded corrective scalar should be direct")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=CorrectiveModel()).answer("What is the final state of BUG-9?")
    assert answer.text == "closed"
    assert answer.diagnostics["reason"] == "validated_terminal_scalar"


def test_duplicate_projected_scalar_surfaces_are_terminal(tmp_path):
    (tmp_path / "reservation.txt").write_text("Access key: ZX-88.\n")

    class DuplicateProjectionModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="text", slot="access_key")
                contract["target_phrases"] = ["Access key"]
                contract["relation_phrases"] = ["value"]
                contract["temporal_mode"] = "at_time"
                return {"semantic_contract": contract}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="reservation.txt::labeled_records[]",
                                terms=["Access key"],
                                arguments=[
                                    {
                                        "name": "mode",
                                        "value": "all",
                                        "values": [],
                                        "numbers": [],
                                    }
                                ],
                            ),
                            _compact_step(
                                "project_values",
                                inputs=[0],
                                fields=["Access key"],
                            ),
                        ],
                    }
                }
            if stage in {"terminal_record_answer", "evidence_review"}:
                raise AssertionError("duplicate identical scalar should not require model rendering")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=DuplicateProjectionModel()).answer(
        "What is the access key?"
    )
    assert answer.text == "ZX-88."
    assert answer.diagnostics["reason"] == "validated_terminal_scalar"
    assert len(answer.evidence) == 2


def test_deterministic_number_extractor_preserves_matching_projected_unit_surface(tmp_path):
    (tmp_path / "data.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"recipe": "pear oat cakes", "temperature": "180C."},
                    {"recipe": "pear oat cakes", "temperature": "180C."},
                ]
            }
        )
    )

    class DeterministicUnitModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {"semantic_contract": _contract(schema, shape="number", slot="temperature")}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="data.json::rows[]",
                                terms=["pear oat cakes"],
                            ),
                            _compact_step("project_values", inputs=[0], fields=["temperature"]),
                            _compact_step(
                                "extract_values",
                                inputs=[1],
                                arguments=[
                                    {
                                        "name": "extractor",
                                        "value": "number",
                                        "values": [],
                                        "numbers": [],
                                    }
                                ],
                            ),
                        ],
                    }
                }
            if stage == "evidence_review":
                raise AssertionError("matching deterministic unit surface should be direct")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=DeterministicUnitModel()).answer(
        "What is the temperature for pear oat cakes?"
    )
    assert answer.text == "180C"
    assert answer.diagnostics["reason"] == "validated_terminal_scalar"


def test_unit_bearing_numeric_surface_is_preserved_without_repair(tmp_path):
    (tmp_path / "recipe.txt").write_text("Oven setting: 180C.")

    class UnitBearingNumericModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {"semantic_contract": _contract(schema, shape="number", slot="oven_setting")}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["Oven setting"],
                            ),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["180C"],
                        "answer_shape": "number",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "The written quantity is explicit.",
                    }
                }
            if stage == "numeric_value_repair":
                raise AssertionError("unit-bearing quantities must not lose their unit")
            if stage == "evidence_review":
                raise AssertionError("grounded unit-bearing extraction should be direct")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=UnitBearingNumericModel()).answer(
        "What is the oven setting?"
    )
    assert answer.text == "180C"
    assert answer.diagnostics["reason"] == "validated_terminal_scalar"


def test_invalid_numeric_model_surface_is_repaired_and_surfaced(tmp_path):
    (tmp_path / "rows.txt").write_text("state: paused")

    class NumericRepairModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {"semantic_contract": _contract(schema, shape="number", slot="row_count")}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step("search_records", collection="all_records", terms=["paused"]),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["/rows[0]"],
                        "answer_shape": "number",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "One matching row.",
                    }
                }
            if stage == "numeric_value_repair":
                contract_id = _enum(
                    schema, "properties", "numeric_value_repair", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "numeric_value_repair": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "value": 1,
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "One matching record.",
                    }
                }
            if stage == "evidence_review":
                raise AssertionError("repaired terminal numeric extraction should be direct")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=NumericRepairModel()).answer("How many rows are paused?")
    assert answer.text == "1"
    assert answer.diagnostics["reason"] == "validated_terminal_scalar"


def test_single_deterministic_url_extraction_is_terminal(tmp_path):
    (tmp_path / "thread.txt").write_text(
        "Retry scheduler tracking PR: https://example.test/pull/42."
    )

    class UrlExtractionModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {"semantic_contract": _contract(schema, shape="url", slot="tracking_url")}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["retry scheduler"],
                            ),
                            _compact_step(
                                "extract_values",
                                inputs=[0],
                                arguments=[
                                    {
                                        "name": "extractor",
                                        "value": "url",
                                        "values": [],
                                        "numbers": [],
                                    }
                                ],
                                limit=1,
                            ),
                        ],
                    }
                }
            if stage == "evidence_review":
                raise AssertionError("single cited URL extraction should be terminal")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=UrlExtractionModel()).answer(
        "What is the tracking URL?"
    )
    assert answer.text == "https://example.test/pull/42"
    assert answer.diagnostics["reason"] == "validated_terminal_scalar"
    assert answer.evidence


def test_terminal_records_resolve_latest_target_state(tmp_path):
    (tmp_path / "other.txt").write_text("Current state: approved.")
    (tmp_path / "timeline.txt").write_text(
        "2026-04-01 Cart state: planned.\n"
        "2026-04-03 Cart state: measured.\n"
        "2026-04-04 Cart state: revised.\n"
    )

    class TemporalRecordModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="text", slot="current_state")
                contract["temporal_mode"] = "current"
                contract["target_phrases"] = ["current state of Cart"]
                return {"semantic_contract": contract}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["Cart", "current state"],
                                limit=10,
                            )
                        ],
                    }
                }
            if stage == "terminal_record_answer":
                contract_id = _enum(
                    schema, "properties", "grounded_answer", "properties", "contract_id"
                )
                ids = re.findall(r'"record_id":\s*"([^"]+)"', prompt)
                timeline_id = next(
                    record_id
                    for record_id in ids
                    if "timeline.txt" in prompt[prompt.find(record_id):]
                ) if False else ids[-1]
                return {
                    "grounded_answer": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "revised",
                        "answer_shape": "text",
                        "evidence_record_ids": [timeline_id],
                        "derivation": "extraction",
                        "confidence": 1.0,
                        "reason": "The latest dated target state is revised.",
                    }
                }
            if stage == "evidence_review":
                raise AssertionError("terminal records should resolve before review")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=TemporalRecordModel()).answer(
        "What is the current state of Cart?"
    )
    assert answer.text == "revised"
    assert answer.diagnostics["reason"] == "validated_terminal_record_answer"
    assert answer.evidence


def test_inconclusive_temporal_extraction_keeps_record_resolution(tmp_path):
    (tmp_path / "timeline.txt").write_text(
        "2026-04-01 Cart state: planned.\n"
        "2026-04-03 Cart state: measured.\n"
        "2026-04-04 Cart state: revised.\n"
    )

    class InconclusiveTemporalModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="text", slot="current_state")
                contract["temporal_mode"] = "current"
                contract["target_phrases"] = ["current state of Cart"]
                return {"semantic_contract": contract}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["Cart", "current state"],
                                limit=10,
                            ),
                            _compact_step("model_extract", inputs=[0]),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "unknown",
                        "values": [],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "unknown",
                        "reason": "The extraction stage declined the temporal inference.",
                    }
                }
            if stage == "terminal_record_answer":
                contract_id = _enum(
                    schema, "properties", "grounded_answer", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "grounded_answer": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "revised",
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "derivation": "extraction",
                        "confidence": 1.0,
                        "reason": "The latest dated state is revised.",
                    }
                }
            if stage == "evidence_review":
                raise AssertionError("temporal record resolution should precede review")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=InconclusiveTemporalModel()).answer(
        "What is the current state of Cart?"
    )
    assert answer.text == "revised"
    assert answer.diagnostics["reason"] == "validated_terminal_record_answer"
    assert answer.evidence


def test_directional_text_extraction_requires_semantic_review(tmp_path):
    (tmp_path / "essay.txt").write_text(
        "Lina Soto drafted the volcano homework essay."
    )

    class DirectionalModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["volcano homework essay"],
                            ),
                            _compact_step(
                                "extract_values",
                                inputs=[0],
                                arguments=[
                                    {
                                        "name": "extractor",
                                        "value": "after_phrase",
                                        "values": ["drafted"],
                                        "numbers": [],
                                    }
                                ],
                                limit=1,
                            ),
                        ],
                    }
                }
            if stage == "evidence_review":
                contract_id = _enum(
                    schema, "properties", "evidence_review", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "Lina Soto",
                        "answer_items": [],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "searches": [],
                        "confidence": 1.0,
                        "reason": "The evidence binds the author relation.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=DirectionalModel()).answer(
        "Who drafted the volcano homework essay?"
    )
    assert answer.text == "Lina Soto"
    assert answer.diagnostics["review"]["status"] == "answered"
    assert answer.diagnostics["review"]["reason"] == "The evidence binds the author relation."


def test_model_selected_terminal_calculation_is_surfaced_without_review(tmp_path):
    (tmp_path / "note.txt").write_text("Homework note: 7 plus 5 equals 12.")

    class CalculationModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="number", slot="sum")
                contract["temporal_mode"] = "at_time"
                return {"semantic_contract": contract}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["7 plus 5"],
                            ),
                            _compact_step(
                                "calculate",
                                inputs=[0],
                                arguments=[
                                    {
                                        "name": "operation",
                                        "value": "add",
                                        "values": [],
                                        "numbers": [7, 5],
                                    }
                                ],
                            ),
                        ],
                    }
                }
            if stage in {"terminal_record_answer", "evidence_review"}:
                raise AssertionError("deterministic terminal scalar should win before temporal review")
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=CalculationModel()).answer("What is seven plus five?")
    assert answer.text == "12"
    assert answer.diagnostics["reason"] == "validated_terminal_scalar"
    assert answer.evidence


def test_ungrounded_placeholder_calculation_operands_defer_to_review(tmp_path):
    (tmp_path / "note.txt").write_text(
        "Puzzle note: 4 shells plus 6 shells equals 10 shells."
    )

    class PlaceholderCalculationModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                contract = _contract(schema, shape="number", slot="unitless numeric scalar")
                contract["semantic_kind"] = "calculation"
                contract["target_phrases"] = ["4 plus 6"]
                contract["relation_phrases"] = ["addition"]
                return {"semantic_contract": contract}
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["4 plus 6"],
                            ),
                            _compact_step(
                                "extract_values",
                                inputs=[0],
                                arguments=[
                                    {
                                        "name": "extractor",
                                        "value": "regex",
                                        "values": [],
                                        "numbers": [],
                                    },
                                    {
                                        "name": "pattern",
                                        "value": "\\d+",
                                        "values": [],
                                        "numbers": [],
                                    },
                                ],
                            ),
                            _compact_step(
                                "calculate",
                                inputs=[1],
                                arguments=[
                                    {
                                        "name": "operation",
                                        "value": "add",
                                        "values": [],
                                        "numbers": [0, 0],
                                    }
                                ],
                            ),
                        ],
                    }
                }
            if stage == "evidence_review":
                assert "For number answers" in prompt
                contract_id = _enum(
                    schema, "properties", "evidence_review", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "10",
                        "answer_items": [],
                        "answer_shape": "number",
                        "evidence_record_ids": [record_id],
                        "searches": [],
                        "confidence": 1.0,
                        "reason": "The cited arithmetic statement gives the result.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=PlaceholderCalculationModel()).answer(
        "What does 4 plus 6 equal in the puzzle note?"
    )
    assert answer.text == "10"
    assert answer.diagnostics["review"]["answer"] == "10"


def test_program_normalization_repairs_generic_self_references(tmp_path):
    (tmp_path / "note.txt").write_text("alpha")
    engine = KnowMoreDiRTEngine(tmp_path, model=PipelineModel())
    program = engine._normalize_program(
        {
            "contract_id": "c",
            "steps": [
                _compact_step("search_records", inputs=[0], collection="all_records", terms=["alpha"]),
                _compact_step("model_extract", inputs=[1]),
                _compact_step("project_values", inputs=[1]),
            ],
        }
    )
    assert program["steps"][0]["inputs"] == []
    assert program["steps"][1]["inputs"] == [0]
    assert program["steps"][2]["inputs"] == [1]


def test_grounded_answer_with_search_status_is_structurally_normalized(tmp_path):
    (tmp_path / "note.txt").write_text("Dataset URL: https://example.test/data.")

    class ContradictoryReview(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "query_program":
                contract_id = _enum(
                    schema, "properties", "query_program", "properties", "contract_id"
                )
                return {
                    "query_program": {
                        "contract_id": contract_id,
                        "steps": [
                            _compact_step(
                                "search_records",
                                collection="all_records",
                                terms=["Dataset URL"],
                            ),
                            _compact_step(
                                "model_extract",
                                inputs=[0],
                            ),
                        ],
                    }
                }
            if stage == "tool_extraction":
                contract_id = _enum(
                    schema, "properties", "tool_extraction", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "tool_extraction": {
                        "contract_id": contract_id,
                        "status": "extracted",
                        "values": ["https://example.test/data."],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "evidence_relation": "direct_support",
                        "reason": "Explicit value.",
                    }
                }
            if stage == "evidence_review":
                contract_id = _enum(
                    schema, "properties", "evidence_review", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "search",
                        "answer": "https://example.test/data.",
                        "answer_items": [],
                        "answer_shape": "text",
                        "evidence_record_ids": [record_id],
                        "searches": [
                            {
                                "collection": "all_records",
                                "terms": ["Dataset URL"],
                                "mode": "any",
                                "fields": [],
                                "limit": 10,
                            }
                        ],
                        "confidence": 1.0,
                        "reason": "Grounded answer plus redundant search.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=ContradictoryReview()).answer("What URL is listed?")
    assert answer.text == "https://example.test/data"
    assert answer.diagnostics["reason"] == "validated_terminal_scalar"


def test_review_normalizer_accepts_only_exact_grounded_derived_answer(tmp_path):
    (tmp_path / "note.txt").write_text("Dataset URL: https://example.test/data.")
    engine = KnowMoreDiRTEngine(tmp_path, model=PipelineModel())
    record = engine.catalog.preferred_records()[0]
    results = {
        0: ToolResult(
            "0",
            "values",
            records=[record],
            values=["https://example.test/data."],
        )
    }
    contract = _contract(
        semantic_contract_schema("What URL is listed?", "contract-review"),
        shape="text",
        slot="url",
    )
    review = {
        "contract_id": contract["contract_id"],
        "status": "search",
        "answer": "https://example.test/data.",
        "answer_items": [],
        "answer_shape": "text",
        "evidence_record_ids": [record.record_id],
        "searches": [
            {
                "collection": "all_records",
                "terms": ["Dataset URL"],
                "mode": "any",
                "fields": [],
                "limit": 10,
            }
        ],
        "confidence": 1.0,
        "reason": "Redundant search.",
    }
    normalized = engine._normalize_review(contract, review, results)
    assert normalized["status"] == "answered"
    assert normalized["searches"] == []
    surface_only_results = {
        0: ToolResult(
            "0",
            "values",
            records=[record],
            values=["wrong directional candidate"],
        )
    }
    surface_normalized = engine._normalize_review(contract, review, surface_only_results)
    assert surface_normalized["status"] == "answered"
    assert surface_normalized["searches"] == []
    invented = {**review, "answer": "https://invented.test/data"}
    assert engine._normalize_review(contract, invented, surface_only_results)["status"] == "search"


def test_review_formats_model_owned_list_items(tmp_path):
    (tmp_path / "note.txt").write_text("Cedar owner and reviewers are Mara and Omar.")

    class ListModel(PipelineModel):
        def complete_json(self, stage, prompt, schema, max_tokens=0):
            if stage == "semantic_contract":
                return {"semantic_contract": _contract(schema, shape="list", slot="reviewers")}
            if stage == "evidence_review":
                contract_id = _enum(
                    schema, "properties", "evidence_review", "properties", "contract_id"
                )
                record_id = re.search(r'"record_id":\s*"([^"]+)"', prompt).group(1)
                return {
                    "evidence_review": {
                        "contract_id": contract_id,
                        "status": "answered",
                        "answer": "",
                        "answer_items": ["Mara", "Omar"],
                        "answer_shape": "list",
                        "evidence_record_ids": [record_id],
                        "searches": [],
                        "confidence": 1.0,
                        "reason": "Both members are explicit.",
                    }
                }
            return super().complete_json(stage, prompt, schema, max_tokens)

    answer = KnowMoreDiRTEngine(tmp_path, model=ListModel()).answer("Who reviews Cedar?")
    assert answer.text == "Mara; Omar"


def test_empty_question_is_unknown_without_model_call(tmp_path):
    (tmp_path / "note.txt").write_text("alpha")
    model = PipelineModel()
    answer = KnowMoreDiRTEngine(tmp_path, model=model).answer("  ")
    assert answer.text == "unknown"
    assert model.calls == []


def test_dspg_counts_and_integrity(tmp_path):
    (tmp_path / "note.txt").write_text("alpha")
    engine = KnowMoreDiRTEngine(tmp_path, model=PipelineModel())
    assert engine.dspg_integrity() == "ok"
    assert engine.dspg_counts()["preferred_records"] == 1
