from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from knowmoredirt import public
from knowmoredirt.bounded_dspg import _identity_expansion
from knowmoredirt.store import DSPGStore


class _Answer:
    def __init__(self, text: str) -> None:
        self.text = text


class _SerializedFakeEngine:
    instances: list["_SerializedFakeEngine"] = []
    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.closed = False
        self.calls = 0
        self.store = DSPGStore(":memory:")
        self.store.execute("CREATE TABLE probe(value TEXT)")
        self.store.execute("INSERT INTO probe(value) VALUES (?)", (self.path,))
        self.store.commit()
        self.__class__.instances.append(self)

    def answer(self, text: str) -> _Answer:
        with self.__class__.counter_lock:
            self.__class__.active += 1
            self.__class__.max_active = max(self.__class__.max_active, self.__class__.active)
        try:
            time.sleep(0.01)
            self.calls += 1
            value = str(self.store.execute("SELECT value FROM probe").fetchone()[0])
            return _Answer(f"{value}:{text}")
        finally:
            with self.__class__.counter_lock:
                self.__class__.active -= 1

    def close(self) -> None:
        self.closed = True
        self.store.close()


@pytest.fixture(autouse=True)
def _reset_public_state(monkeypatch: pytest.MonkeyPatch):
    public._reset()
    _SerializedFakeEngine.instances.clear()
    _SerializedFakeEngine.active = 0
    _SerializedFakeEngine.max_active = 0
    monkeypatch.setattr(public, "KnowMoreDiRTEngine", _SerializedFakeEngine)
    yield
    public._reset()


def test_public_questions_are_serialized_and_work_across_threads() -> None:
    public.initialize("A")

    with ThreadPoolExecutor(max_workers=8) as executor:
        answers = list(executor.map(lambda index: public.question(f"q{index}"), range(24)))

    assert sorted(answers) == sorted(f"A:q{index}" for index in range(24))
    assert _SerializedFakeEngine.max_active == 1
    assert _SerializedFakeEngine.instances[0].calls == 24


def test_reinitialize_closes_previous_engine_after_atomic_replacement() -> None:
    public.initialize("A")
    first = _SerializedFakeEngine.instances[-1]

    public.initialize("B")
    second = _SerializedFakeEngine.instances[-1]

    assert first.closed is True
    assert second.closed is False
    assert public.question("q") == "B:q"


def test_failed_reinitialize_preserves_previous_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    public.initialize("A")
    first = _SerializedFakeEngine.instances[-1]

    class FailingEngine:
        def __init__(self, _path: str | Path) -> None:
            raise RuntimeError("failed initialization")

    monkeypatch.setattr(public, "KnowMoreDiRTEngine", FailingEngine)
    with pytest.raises(RuntimeError, match="failed initialization"):
        public.initialize("B")

    assert first.closed is False
    assert public.question("q") == "A:q"


def test_reset_closes_and_clears_engine() -> None:
    public.initialize("A")
    engine = _SerializedFakeEngine.instances[-1]

    public._reset()

    assert engine.closed is True
    with pytest.raises(RuntimeError, match="not initialized"):
        public.question("q")


def _identity_records(confidence: float) -> dict[str, Any]:
    return {
        "referents": [
            {"referent_id": "r1", "canonical_label": "Alex Smith"},
            {"referent_id": "r2", "canonical_label": "Alex Smith Consulting"},
            {"referent_id": "r3", "canonical_label": "Unrelated Owner"},
        ],
        "identity_hypotheses": [
            {"hypothesis_id": "h1", "left_referent_id": "r1", "right_referent_id": "r2", "relation": "same_surface", "confidence": confidence, "context_id": "", "evidence": "edge one"},
            {"hypothesis_id": "h2", "left_referent_id": "r2", "right_referent_id": "r3", "relation": "alias", "confidence": confidence, "context_id": "", "evidence": "edge two"},
        ],
        "contexts": [],
        "source_spans": [],
        "chunks": [],
        "documents": [],
    }


def test_low_confidence_identity_edges_do_not_expand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KMD_IDENTITY_EXPANSION_MIN_CONFIDENCE", "0.75")

    expanded, evidence = _identity_expansion(_identity_records(0.001), ["Alex Smith"])

    assert "alex smith" in expanded
    assert "alex smith consulting" not in expanded
    assert "unrelated owner" not in expanded
    assert evidence == []


def test_high_confidence_identity_edges_expand_transitively(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KMD_IDENTITY_EXPANSION_MIN_CONFIDENCE", "0.75")

    expanded, evidence = _identity_expansion(_identity_records(0.95), ["Alex Smith"])

    assert "alex smith consulting" in expanded
    assert "unrelated owner" in expanded
    assert isinstance(evidence, list)


def test_invalid_identity_threshold_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KMD_IDENTITY_EXPANSION_MIN_CONFIDENCE", "2")

    with pytest.raises(ValueError, match="between 0 and 1"):
        _identity_expansion(_identity_records(0.95), ["Alex Smith"])
