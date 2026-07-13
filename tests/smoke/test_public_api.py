from __future__ import annotations
import pytest
from knowmoredirt import public


def test_question_requires_initialization(monkeypatch):
    monkeypatch.setattr(public, "_ENGINE", None)
    with pytest.raises(RuntimeError):
        public.question("Anything?")


def test_public_api_delegates_to_engine(monkeypatch, tmp_path):
    class Answer:
        text = "ok"
    class Engine:
        def __init__(self, folder):
            self.folder = folder
        def answer(self, text):
            return Answer()
    monkeypatch.setattr(public, "KnowMoreDiRTEngine", Engine)
    public.initialize(tmp_path)
    assert public.question("question") == "ok"
