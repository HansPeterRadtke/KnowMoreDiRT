from pathlib import Path

from knowmoredirt import filesystem


class FakeConfig:
    def __init__(self):
        self.analysis = object()
        self.embedding = object()

    def clients(self):
        return self.analysis, self.embedding


def test_initialize_filesystem_database_delegates_without_drt(monkeypatch, tmp_path):
    captured = {}
    def fake_initialize(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}
    monkeypatch.setattr(filesystem, "initialize_text_folder", fake_initialize)
    config = FakeConfig()
    result = filesystem.initialize_filesystem_database(tmp_path / "raw", tmp_path / "catalog.sqlite3", config=config)
    assert result == {"status": "ok"}
    assert captured["analysis_client"] is config.analysis
    assert captured["embedding_client"] is config.embedding
    assert captured["root"] == tmp_path / "raw"


def test_question_filesystem_database_uses_isolated_assistant(monkeypatch, tmp_path):
    captured = {}
    class FakeAssistant:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def ask(self, question):
            return {"result": {"answer": question}}
    monkeypatch.setattr(filesystem, "FolderQuestionAssistant", FakeAssistant)
    config = FakeConfig()
    result = filesystem.question_filesystem_database(
        tmp_path / "raw", tmp_path / "catalog.sqlite3", "What happened?", config=config, max_evidence=7
    )
    assert result["result"]["answer"] == "What happened?"
    assert captured["max_evidence"] == 7
    assert captured["analysis_client"] is config.analysis


def test_environment_endpoint_is_normalized_to_server_root(monkeypatch):
    monkeypatch.setenv("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:14829/v1")
    assert filesystem.FilesystemModelConfig.from_environment().analysis_url == "http://127.0.0.1:14829"
