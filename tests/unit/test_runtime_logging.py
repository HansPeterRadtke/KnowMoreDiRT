from __future__ import annotations

import logging
from pathlib import Path

import kmd_runtime_config as config
from knowmoredirt import runtime_logging


def _reset() -> None:
    config._USER_CACHE = None
    runtime_logging._CONFIGURED = False


def test_persistent_rotating_file_logging(tmp_path: Path, monkeypatch) -> None:
    log = tmp_path / "kmd.log"
    monkeypatch.setenv("KMD_LOG_ENABLED", "1")
    monkeypatch.setenv("KMD_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("KMD_LOG_FILE", str(log))
    monkeypatch.setenv("KMD_LOG_MAX_BYTES", "1024")
    monkeypatch.setenv("KMD_LOG_BACKUP_COUNT", "2")
    monkeypatch.setenv("KMD_LOG_STDERR", "0")
    _reset()
    runtime_logging.configure_logging(force=True)
    logger = runtime_logging.get_logger("test")
    for index in range(80):
        logger.info("persistent-log-record index=%s payload=%s", index, "x" * 64)
    for handler in logging.getLogger("knowmoredirt").handlers:
        if hasattr(handler, "flush"):
            handler.flush()
    assert log.is_file()
    assert list(tmp_path.glob("kmd.log.*"))
    combined = "\n".join(path.read_text(errors="replace") for path in sorted(tmp_path.glob("kmd.log*")))
    assert "persistent-log-record" in combined


def test_logging_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    log = tmp_path / "disabled.log"
    monkeypatch.setenv("KMD_LOG_ENABLED", "0")
    monkeypatch.setenv("KMD_LOG_FILE", str(log))
    _reset()
    runtime_logging.configure_logging(force=True)
    runtime_logging.get_logger("disabled").error("must-not-create-file")
    assert not log.exists()
