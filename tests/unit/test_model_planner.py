from knowmoredirt.model_planner import build_compact_query_drs_prompt, build_query_drs_prompt, build_query_evidence_answer_prompt, build_answer_verification_prompt, build_compact_chunk_drs_prompt, build_query_evidence_answer_repair_prompt, _query_evidence_payload_from_result



def test_production_query_drs_prompt_distinguishes_urls_paths_arithmetic_and_plurality() -> None:
    prompt = build_query_drs_prompt("Where is the village map stored?")
    assert "HTTP or HTTPS location is url" in prompt
    assert "file_path is only a filesystem-style path" in prompt
    assert "arithmetic result is count" in prompt
    assert "explicitly requests multiple answers" in prompt
    assert "subordinate non-asserted box" in prompt


def test_compact_query_drs_prompt_preserves_the_same_semantic_contract() -> None:
    prompt = build_compact_query_drs_prompt("Where is the runbook?")
    assert "HTTP or HTTPS values are url" in prompt
    assert "file_path is only" in prompt
    assert "arithmetic results are count" in prompt
    assert "explicitly requests multiple answers" in prompt
    assert "not asserted real-world fact" in prompt


def test_query_evidence_prompt_enforces_open_world_boolean_entailment() -> None:
    prompt = build_query_evidence_answer_prompt(
        "Did the silver train really carry the kitchen table away?",
        [{"rel_path": "diary.dream", "text": "I dreamed a silver train carried the kitchen table away. Morning fact: the kitchen table remained in the dining room."}],
    )
    assert "Apply open-world DRT entailment" in prompt
    assert "lack of an asserted positive fact does not entail its negation" in prompt
    assert "answer no only when accessible asserted evidence explicitly negates" in prompt
    assert "otherwise return unknown" in prompt


def test_answer_verification_prompt_enforces_open_world_negative_proof() -> None:
    prompt = build_answer_verification_prompt(
        "Did the silver train really carry the kitchen table away?",
        {"answer_type": "boolean", "requested_relation": "carry away"},
        "no",
        [{"rel_path": "diary.dream", "text": "Morning fact: the kitchen table remained in the dining room."}],
        [],
    )
    assert "Apply open-world DRT entailment" in prompt
    assert "lack of an asserted positive fact does not entail a negative answer" in prompt
    assert "same relevant temporal scope" in prompt
    assert "subordinate content is not asserted fact" in prompt


def test_compact_chunk_prompt_preserves_attribute_value_structure() -> None:
    prompt = build_compact_chunk_drs_prompt("Greenhouse pump state: repaired.", rel_path="garden/farm.log")
    assert "Preserve attribute-value structure" in prompt
    assert "emit predicate state" in prompt
    assert "a subject argument X" in prompt
    assert "a value argument Y" in prompt
    assert "do not use Y as the predicate" in prompt


def test_verifier_cache_key_uses_verifier_schema(monkeypatch, tmp_path) -> None:
    from knowmoredirt import model_planner as planner
    captured = {}
    monkeypatch.setenv("KMD_VERIFIER_CACHE_DIR", str(tmp_path))
    def fake_cache_hash(kind, prompt, client, settings):
        captured["settings"] = settings
        return "x"
    monkeypatch.setattr(planner, "_cache_hash", fake_cache_hash)
    class Dummy:
        def complete_json(self, *args, **kwargs):
            return {"verification": {"entailed": False, "answer_type": "unknown", "answer": "unknown", "evidence_span": "", "proof_kind": "unknown", "accessibility": "unknown", "temporal_alignment": "unspecified", "explicit_negation": False, "incompatible_condition_span": "", "reason": ""}}
        def cache_fingerprint(self): return {}
    planner.call_model_answer_verification("q", {"answer_type":"boolean"}, "no", [], [], Dummy(), n_predict=32)
    assert captured["settings"]["schema"] == planner.VERIFIER_SCHEMA_VERSION


def test_query_evidence_prompt_covers_role_field_table_and_source_scope() -> None:
    prompt = build_query_evidence_answer_prompt(
        "Which warranty URL belongs to Mica Relay?",
        [{"rel_path": "references/mica.txt", "text": "Warranty URL: https://warranty.example.test/mica-relay"}],
    )
    assert "in passive clauses bind the agent after by" in prompt
    assert "bind the entity that remains installed" in prompt
    assert "distinguishing measurement date, copy date, record time" in prompt
    assert "require the requested field label" in prompt
    assert "cache, noise, transport" in prompt
    assert "For table counts, count only rows" in prompt
    assert "author and reviewers" in prompt


def test_query_drs_prompt_binds_interrogative_semantic_role() -> None:
    prompt = build_query_drs_prompt("What remains installed after the dream?")
    assert "what in 'what remains installed' is the installed entity" in prompt
    assert "who stated/reported is the speaker" in prompt
    assert "Preserve requested field labels" in prompt


def test_query_evidence_prompt_allows_reported_content_for_non_actuality_binding() -> None:
    prompt = build_query_evidence_answer_prompt(
        "Who signed the orchard lease?",
        [{"rel_path": "letter.txt", "text": "She said the orchard lease was signed by Clara Reed."}],
    )
    assert "reported or quoted content may directly answer attribution" in prompt
    assert "really, actually, proven, confirmed" in prompt


def test_verification_prompt_distinguishes_report_binding_from_actuality() -> None:
    prompt = build_answer_verification_prompt(
        "Who signed the orchard lease?",
        {"answer_type": "person", "requested_relation": "signed"},
        "Clara Reed",
        [{"rel_path": "letter.txt", "text": "She said the orchard lease was signed by Clara Reed."}],
        [],
    )
    assert "Reported or quoted content can verify attribution" in prompt
    assert "do not reject such an answer merely because its source is a report" in prompt


def test_query_drs_prompt_does_not_target_answer_coreferential_pronoun() -> None:
    prompt = build_query_drs_prompt("Who stated she heard a bang?")
    assert "Do not create a target referent from a pronoun" in prompt
    assert "coreferential with the speaker answer variable" in prompt


def test_verification_prompt_rejects_self_declared_nonsemantic_cache() -> None:
    prompt = build_answer_verification_prompt(
        "Which warranty URL belongs to Mica Relay?",
        {"answer_type": "url", "requested_relation": "warranty URL"},
        "https://cache.example.test/wrong-mica",
        [{"rel_path": "noise/cache.tmp", "text": "CACHE ONLY -- not a semantic record."}],
        [],
    )
    assert "cache only, noise, transport cache" in prompt
    assert "cannot verify the answer" in prompt


def test_query_evidence_count_payload_rejects_line_count_mismatch() -> None:
    result = {
        "query_frame": {"target_anchors": ["rows"], "answer_variables": ["count"], "requested_relation": "state", "relation_terms": ["state"], "constraints": ["open"], "scope_requirements": [], "modality_requirements": [], "answer_type": "count", "temporal_scope": "", "negated": False, "aggregation": "count", "requires_evidence": True},
        "sufficient_evidence": True,
        "answer_type": "count",
        "answer": "2",
        "evidence_span": "row one\nrow two\nrow three",
        "reason": "counted rows",
    }
    payload = _query_evidence_payload_from_result("How many rows?", result, [{"rel_path": "t.tsv", "text": "row one\nrow two\nrow three"}], "", 0.0, "p", "g", fresh_or_cached="fresh")
    assert payload["accepted"] is False
    assert payload["reason"] == "grounding_validation_failed"


def test_query_evidence_repair_prompt_requires_exact_count_rows() -> None:
    prompt = build_query_evidence_answer_repair_prompt("How many rows?", [{"rel_path": "t.tsv", "text": "a\nb"}], "{}")
    assert "exactly the counted matching source rows" in prompt
    assert "number of nonempty lines must equal answer" in prompt


def test_query_evidence_count_payload_accepts_noncontiguous_grounded_rows() -> None:
    result = {
        "query_frame": {"target_anchors": ["rows"], "answer_variables": ["count"], "requested_relation": "state", "relation_terms": ["state"], "constraints": ["open"], "scope_requirements": [], "modality_requirements": [], "answer_type": "count", "temporal_scope": "", "negated": False, "aggregation": "count", "requires_evidence": True},
        "sufficient_evidence": True,
        "answer_type": "count",
        "answer": "2",
        "evidence_span": "row one open\nrow three open",
        "reason": "counted matching rows",
    }
    evidence = [{"rel_path": "t.tsv", "text": "row one open\nrow two closed\nrow three open"}]
    payload = _query_evidence_payload_from_result("How many rows are open?", result, evidence, "", 0.0, "p", "g", fresh_or_cached="fresh")
    assert payload["accepted"] is True
    assert payload["answer"] == "2"


def test_verification_prompt_supports_authoritative_explicit_exclusion() -> None:
    prompt = build_answer_verification_prompt(
        "Does the audit say QuillCache stores plaintext passwords?",
        {"answer_type": "boolean", "requested_relation": "stores plaintext passwords"},
        "no",
        [{"rel_path": "audit.txt", "text": "Audit result: QuillCache stores only salted password hashes."}],
        [],
    )
    assert "proof_kind=explicit_exclusion" in prompt
    assert "stores only salted hashes" in prompt
    assert "mere absence of evidence" in prompt


def test_verification_prompt_marks_absence_of_record_as_non_entailing() -> None:
    prompt = build_answer_verification_prompt(
        "Did Pearl Engine really open the hidden gate?",
        {"answer_type": "boolean", "requested_relation": "open"},
        "no",
        [{"rel_path": "dream.txt", "text": "Waking note: no real gate opening is recorded."}],
        [],
    )
    assert "absence_of_record_only=true" in prompt
    assert "no record, not recorded, no report" in prompt
