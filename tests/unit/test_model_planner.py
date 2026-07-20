from knowmoredirt.model_planner import build_compact_query_drs_prompt, build_query_drs_prompt, build_query_evidence_answer_prompt, build_answer_verification_prompt, build_compact_chunk_drs_prompt



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
