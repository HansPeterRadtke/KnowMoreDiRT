from __future__ import annotations

from pathlib import Path

from knowmoredirt.evaluation import answer_matches, semantic_answer_judgment


class FakeJudgeClient:
    calls = 0

    def cache_fingerprint(self):
        return {"model_id": "fake", "context_size": 1024, "request_settings": {}, "transport_settings": {}}

    def complete_json(self, prompt, *, n_predict=None, json_schema=None):
        self.calls += 1
        assert "EXPECTED ANSWER" in prompt and "PREDICTED ANSWER" in prompt
        assert n_predict is None
        assert json_schema["required"] == ["equivalent", "reason"]
        return {"equivalent": True, "reason": "same meaning in different wording"}


def test_unknown_qualified_with_subordinate_evidence_remains_unknown() -> None:
    assert answer_matches("unknown — Timmy's dream mentions a flying-car law", "unknown")


def test_semantic_judge_is_used_only_after_deterministic_mismatch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KMD_EVALUATION_JUDGE_CACHE_DIR", str(tmp_path))
    client = FakeJudgeClient()
    result = semantic_answer_judgment("Who approved it?", "It was approved by Mara.", "Mara", client=client)
    assert result["equivalent"] is True
    assert result["judge_used"] is True
    assert client.calls == 1
    second = semantic_answer_judgment("Who approved it?", "It was approved by Mara.", "Mara", client=client)
    assert second["equivalent"] is True
    assert second["cache_hit"] is True
    assert client.calls == 1


def test_deterministic_match_does_not_call_judge(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KMD_EVALUATION_JUDGE_CACHE_DIR", str(tmp_path))
    client = FakeJudgeClient()
    result = semantic_answer_judgment("Who?", "Mara", "Mara", client=client)
    assert result["equivalent"] is True
    assert result["judge_used"] is False
    assert client.calls == 0
