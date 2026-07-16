from __future__ import annotations

import ast
import re
from pathlib import Path

from conftest import REPO_ROOT


FORBIDDEN_CORE_MARKERS = [
    "HERB",
    "HELP",
    "benchmark",
    "benchmark family",
    "question family",
    "benchmark routing",
    "benchmark intents",
    "parity",
    "scorer",
    "gold",
    "answerability",
    "question_id",
    "official question",
    "prepared",
    "HERB RAW ARTIFACT",
    "allow_prepared_metadata",
    "DRT_HERB_PREP_ROOT",
    "artifact_manifest_by_rel_path",
    "source_corpus",
    "product_id",
    "source_title",
    "product_name",
    "employee_ids",
    "customer_id",
    "which_pr",
    "which_customer",
    "which_ticket",
    "which_issue",
    "max-PR",
    "unresolved-bug",
    "employee-ID",
    "artifact search",
    "role_lookup",
    "reference_lookup",
    "url_lookup",
    "file_lookup",
    "state_lookup",
    "answer_role",
    "_answer_who_role",
    "_answer_identifier_or_url",
    "_answer_final_state",
]

FORBIDDEN_CORE_REGEXES = [
    r"\bPR\b",
    r"\bpr\b",
    r"\bticket\b",
    r"\bissue\b",
    r"\bcustomer\b",
    r"\bemployee\b",
    r"\bartifact\b",
    r"\bif\s+.*['\"](?:owner|reviewer|approver|reporter|author)['\"]",
    r"\belif\s+.*['\"](?:owner|reviewer|approver|reporter|author)['\"]",
]

FORBIDDEN_SEMANTIC_BRANCH_WORDS = {
    "owner",
    "reviewer",
    "manual",
    "runbook",
    "assignment",
    "claim",
    "warranty",
    "endpoint",
    "company",
    "customer",
    "author",
    "maintainer",
    "assignee",
    "decision",
    "scale",
    "snapped",
    "translation",
}

CORE_BRANCH_FILES = {
    "answer_types.py",
    "bounded_dspg.py",
    "engine.py",
    "model_planner.py",
    "query.py",
    "relations.py",
}


def test_core_package_has_no_benchmark_or_prepared_input_markers() -> None:
    source_files = list((REPO_ROOT / "src" / "knowmoredirt").glob("*.py"))
    assert source_files
    findings: list[str] = []
    for path in source_files:
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_CORE_MARKERS:
            if marker in text:
                findings.append(f"{path.relative_to(REPO_ROOT)}:{marker}")
        for pattern in FORBIDDEN_CORE_REGEXES:
            if re.search(pattern, text):
                findings.append(f"{path.relative_to(REPO_ROOT)}:{pattern}")
    assert findings == []


def test_public_api_exports_only_two_user_functions() -> None:
    init_file = REPO_ROOT / "src" / "knowmoredirt" / "__init__.py"
    text = init_file.read_text(encoding="utf-8")
    assert '__all__ = ["initialize", "question"]' in text


def test_core_package_has_no_fixture_or_domain_shaped_literals() -> None:
    forbidden = [
        "FlowQuill",
        "ActionGarden",
        "vault.key",
        "stale ledgers",
        "plaintext",
        "cache expiration",
        "parser.cpp",
        "MarlinKind",
        "RippleDesk",
        "Blue Dune",
        "Northstar Credit",
    ]
    findings: list[str] = []
    for path in (REPO_ROOT / "src" / "knowmoredirt").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker.lower() in text.lower():
                findings.append(f"{path.relative_to(REPO_ROOT)}:{marker}")
    assert findings == []


def test_core_has_no_string_triggered_semantic_relation_branches() -> None:
    findings: list[str] = []
    for path in (REPO_ROOT / "src" / "knowmoredirt").glob("*.py"):
        if path.name not in CORE_BRANCH_FILES:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.If, ast.IfExp, ast.While)):
                continue
            literals = [
                child.value.lower()
                for child in ast.walk(node.test)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            ]
            literal_tokens = set(re.findall(r"[a-z0-9_]+", " ".join(literals)))
            branch_words = [
                word for word in FORBIDDEN_SEMANTIC_BRANCH_WORDS
                if word in literal_tokens
            ]
            if branch_words:
                findings.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{','.join(sorted(branch_words))}:"
                    f"{ast.dump(node.test, include_attributes=False)}"
                )
    assert findings == []


def test_surface_extractor_has_no_deterministic_semantic_event_regexes() -> None:
    text = (REPO_ROOT / "src" / "knowmoredirt" / "relations.py").read_text(encoding="utf-8")
    forbidden = [
        "COPULAR_RE",
        "PASSIVE_EVENT_RE",
        "ACTIVE_EVENT_RE",
        "PERSON_PATTERN",
        "surface_verb",
        'relation_type", "event"',
        '"event",',
        '"assertion",',
        '"polarity_marker",',
    ]
    findings = [marker for marker in forbidden if marker in text]
    assert findings == []


def test_production_model_path_does_not_call_deterministic_semantic_answer_tools() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KnowMoreDiRTEngine"
    )
    method = next(
        node for node in target.body
        if isinstance(node, ast.FunctionDef) and node.name == "_answer_with_local_model"
    )
    calls = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_run_model_planned_answer_tools"
    ]
    assert calls == []


def test_production_model_path_has_no_unverified_bounded_answer_shortcuts() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    engine_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KnowMoreDiRTEngine"
    )
    method = next(
        node for node in engine_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_answer_with_local_model"
    )
    text = ast.get_source_segment(source, method) or ""
    forbidden = [
        "bounded DSPG deterministic arithmetic execution",
        "local model query-frame count aggregation",
        "local model query-frame temporal binding",
        "_trusted_exact_structural_bounded_answer",
        "KMD_VERIFY_MODEL_DRS_BOUND_ANSWERS",
    ]
    assert [marker for marker in forbidden if marker in text] == []
    assert "_verify_with_local_model" in text


def test_production_cleanup_is_presentation_only() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    engine_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KnowMoreDiRTEngine"
    )
    method = next(
        node for node in engine_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_cleanup_public_answer"
    )
    text = ast.get_source_segment(source, method) or ""
    forbidden = [
        "_cleanup_canonical_answer",
        "_central_answer_guard",
        "_restore_where_preposition",
        "_expand_single_name_from_evidence",
        "plan_question",
        "classify_value",
    ]
    assert [marker for marker in forbidden if marker in text] == []


def test_production_search_uses_model_query_plan() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    engine_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KnowMoreDiRTEngine"
    )
    method = next(
        node for node in engine_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_search"
    )
    text = ast.get_source_segment(source, method) or ""
    assert "model_query_trace.last_plan" in text
    assert "elif self._test_no_model_runtime" in text
    assert "Production evidence retrieval requires" in text


def test_model_owned_ingest_has_no_deterministic_speaker_coreference_injection() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "ingest.py").read_text(encoding="utf-8")
    forbidden = [
        "deterministic_speaker_turn",
        "deterministic_structural_speaker",
        "speaker_turn_identity_hypotheses",
        "structural_speaker_identity_hypotheses",
        "_link_labeled_turn_speaker_referents",
        "_link_first_person_referents_to_speaker_surface",
    ]
    assert [marker for marker in forbidden if marker in source] == []


def test_model_semantic_ingest_has_no_premodel_quality_skip() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "ingest.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in ("_grounded_model_frames", "_ingest_model_drs_for_sentence"):
        text = ast.get_source_segment(source, functions[name]) or ""
        assert "_model_semantic_skip_reason" not in text
        assert "KMD_ALLOW_PREMODEL_SEMANTIC_SKIP" not in text


def test_model_query_drs_bounded_execution_does_not_reinterpret_raw_question_visibility() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "bounded_dspg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    target_text = ast.get_source_segment(source, functions["_target_terms"]) or ""
    visible_text = ast.get_source_segment(source, functions["_visible_target_terms"]) or ""
    scope_text = ast.get_source_segment(source, functions["_discourse_scope_query_terms"]) or ""
    assert 'frame.source == "model_query_drs"' in target_text
    assert 'frame.source == "model_query_drs"' in visible_text
    assert 'frame.source == "model_query_drs"' in scope_text
    assert 'question_material = ""' in target_text
    assert 'semantic_question = "" if frame.source == "model_query_drs" else question' in scope_text


def test_production_metadata_retrieval_uses_model_query_fields() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    engine_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KnowMoreDiRTEngine"
    )
    method = next(
        node for node in engine_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_metadata_bounded_candidates"
    )
    text = ast.get_source_segment(source, method) or ""
    assert "model_query_trace.last_plan" in text
    assert 'source="model_query_drs"' in text
    assert "elif self._test_no_model_runtime" in text


def test_model_query_drs_metadata_binding_has_no_raw_question_fallback() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "bounded_dspg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_bind_metadata"
    )
    text = ast.get_source_segment(source, method) or ""
    assert 'frame.source == "model_query_drs"' in text
    assert 'fallback_terms = [] if frame.source == "model_query_drs" else _query_terms(question)' in text


def test_model_query_drs_answer_slot_binding_uses_model_roles() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "bounded_dspg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_answer_slot_terms"
    )
    text = ast.get_source_segment(source, method) or ""
    assert 'frame.source == "model_query_drs" and frame.binding_roles' in text
    assert "expand_terms(frame.binding_roles)" in text


def test_model_query_drs_does_not_use_hardcoded_current_state_preference() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "bounded_dspg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_structured_current_state_preference"
    )
    text = ast.get_source_segment(source, method) or ""
    assert 'if frame.source == "model_query_drs":' in text
    assert "return candidates" in text


def test_model_query_drs_disables_identifier_typography_scoring() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "bounded_dspg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_bounded_query"
    )
    text = ast.get_source_segment(source, method) or ""
    assert 'allow_identifier_shape_bonus=frame.source != "model_query_drs"' in text


def test_model_query_drs_list_binding_preserves_all_formal_values_for_model_verification() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "bounded_dspg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_bounded_query"
    )
    text = ast.get_source_segment(source, method) or ""
    assert 'allow_named_entity_surface_filter=frame.source != "model_query_drs"' in text


def test_model_query_drs_does_not_delete_formal_argument_values_by_surface_overlap() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "bounded_dspg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_prefer_drs_argument_values"
    )
    text = ast.get_source_segment(source, method) or ""
    assert 'if frame.source == "model_query_drs":' in text
    assert "return candidates" in text


def test_production_retrieval_does_not_downrank_text_by_deterministic_quality_class() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    engine_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KnowMoreDiRTEngine"
    )
    method = next(
        node for node in engine_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_search"
    )
    text = ast.get_source_segment(source, method) or ""
    assert "if self._test_no_model_runtime and sentence.rel_path in self._low_semantic_noise_paths" in text


def test_model_query_drs_bounded_ranking_uses_only_model_semantic_terms() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "bounded_dspg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_rank_scope"
    )
    text = ast.get_source_segment(source, method) or ""
    assert 'all_terms = [] if frame.source == "model_query_drs" else _query_terms(question)' in text
    assert 'frame.source != "model_query_drs" and document_low_priority_by_id' in text
    assert 'frame.source != "model_query_drs" and _source_is_low_priority' in text


def test_model_query_drs_binders_do_not_reject_sources_by_text_shape() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "bounded_dspg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    binder_names = {
        "_bind_frame_conditions",
        "_bind_relation_conditions",
        "_bind_document_scoped_label_values",
        "_document_scoped_drs_condition_candidates",
        "_document_scoped_relation_value_candidates",
        "_structural_chain_rows",
        "_bind_record_groups",
        "_count_matching_record_groups",
        "_temporal_relation_candidates",
        "_selected_temporal_span_ids",
    }
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in binder_names
    }
    assert set(functions) == binder_names
    for name, method in functions.items():
        text = ast.get_source_segment(source, method) or ""
        if "_source_is_low_priority" in text:
            assert 'frame.source != "model_query_drs"' in text, name


def test_relation_only_target_fallback_is_legacy_only() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "bounded_dspg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_rank_scope"
    )
    text = ast.get_source_segment(source, method) or ""
    assert 'legacy_relation_fallback = frame.source != "model_query_drs" and not selected_docs' in text


def test_model_query_drs_finalization_skips_deterministic_semantic_rewrites() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    engine_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KnowMoreDiRTEngine"
    )
    method = next(
        node for node in engine_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_finalize_answer"
    )
    text = ast.get_source_segment(source, method) or ""
    assert 'production_model_query = frame is not None and frame.source == "model_query_drs"' in text
    assert "if not production_model_query:" in text


def test_production_engine_contains_no_model_guided_legacy_semantic_dispatch() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "engine.py").read_text(encoding="utf-8")
    assert "def _model_planned_answer_tools" not in source
    assert "def _run_model_planned_answer_tools" not in source


def test_model_production_methods_do_not_reference_legacy_source_handlers() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    engine_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KnowMoreDiRTEngine"
    )
    production_methods = {
        "_answer_with_local_model",
        "_answer_with_model_query_evidence",
        "_answer_with_model_evidence_extraction",
        "_answer_with_bounded_dspg",
        "_verify_with_local_model",
        "_canonicalize_model_answer_with_local_model",
        "_source_resolve_model_answer_with_local_model",
    }
    methods = {
        node.name: node for node in engine_class.body
        if isinstance(node, ast.FunctionDef) and node.name in production_methods
    }
    assert set(methods) == production_methods
    violations: list[str] = []
    for method_name, method in methods.items():
        for node in ast.walk(method):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            name = node.func.attr
            if name.startswith("_answer_with_") and name.endswith("_source"):
                violations.append(f"{method_name}:{name}")
    assert violations == []


def test_public_answer_legacy_semantic_loop_is_pytest_only() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    engine_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KnowMoreDiRTEngine"
    )
    method = next(
        node for node in engine_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "answer"
    )
    text = ast.get_source_segment(source, method) or ""
    guard = 'if not self._test_no_model_runtime:'
    loop = 'for source_answer_fn in ('
    assert guard in text
    assert loop in text
    assert text.index(guard) < text.index(loop)
    assert "Legacy deterministic semantic handlers are restricted to the explicit pytest-only no-model runtime." in text


def test_architecture_docs_describe_required_model_semantics() -> None:
    architecture = (REPO_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    audit = (REPO_ROOT / "docs" / "raw_folder_architecture_audit.md").read_text(encoding="utf-8")
    assert "conservative deterministic answer extraction" not in architecture
    assert "optional localhost-only query planning" not in audit
    assert "disabled by default" not in audit
    assert "Required local-model bounded reasoning" in audit


def test_model_evidence_path_requires_authoritative_model_query_plan() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    engine_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KnowMoreDiRTEngine"
    )
    method = next(
        node for node in engine_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_answer_with_model_query_evidence"
    )
    text = ast.get_source_segment(source, method) or ""
    assert "plan_question(" not in text
    assert "if frame_data is None:" in text
    assert 'source="model_query_drs"' in text


def test_dead_capitalization_identity_canonicalizer_is_absent() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "engine.py").read_text(encoding="utf-8")
    assert "def _canonicalize_identity_with_local_model" not in source


def test_query_drs_repair_is_provenance_only() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "model_planner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_repair_query_drs_payload"
    )
    text = ast.get_source_segment(source, method) or ""
    forbidden = [
        "visible_anchors(",
        "uncovered_predicate_tokens",
        "relation_groups_for_repair",
        "slot_descriptor_tokens_for_repair",
        "generic_anchor_tokens_for_repair",
        'argument["target_kind"] =',
        'argument["target_id"] =',
        'query_drs["requested_conditions"] =',
        'target_items.append(',
        'conditions.append(',
    ]
    assert [marker for marker in forbidden if marker in text] == []
    assert "Repair exact provenance surfaces only" in text


def test_query_drs_validation_has_no_relation_specific_vocabulary() -> None:
    source = (REPO_ROOT / "src" / "knowmoredirt" / "model_planner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_query_drs_payload"
    )
    text = ast.get_source_segment(source, method) or ""
    forbidden = [
        "visible_anchors(",
        "required_relation_groups",
        "dropped_requested_relation",
        "dropped_visible_anchor",
        '"reviewer"',
        '"approver"',
        '"owner"',
        '"author"',
    ]
    assert [marker for marker in forbidden if marker in text] == []
    assert 'errors.append("missing_requested_conditions")' in text
