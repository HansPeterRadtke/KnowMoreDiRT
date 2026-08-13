"""Persistent configurable logging for KMD runtime code."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path

from kmd_runtime_config import boolean, integer, text, validate_all


_LOCK = threading.Lock()
_CONFIGURED = False
_LOGGER_NAME = "knowmoredirt"


def _fallback_log_file() -> Path:
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_home / "knowmoredirt" / "knowmoredirt.log"


def configure_logging(*, force: bool = False) -> logging.Logger:
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED and not force:
            return logging.getLogger(_LOGGER_NAME)
        validate_all()
        logger = logging.getLogger(_LOGGER_NAME)
        logger.propagate = False
        for handler in list(logger.handlers):
            if getattr(handler, "_kmd_managed", False):
                logger.removeHandler(handler)
                handler.close()
        if not boolean("KMD_LOG_ENABLED"):
            null_handler = logging.NullHandler()
            null_handler._kmd_managed = True  # type: ignore[attr-defined]
            logger.addHandler(null_handler)
            logger.setLevel(logging.CRITICAL + 1)
            _CONFIGURED = True
            return logger
        level_name = text("KMD_LOG_LEVEL").upper()
        level = getattr(logging, level_name)
        logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(process)d %(threadName)s %(name)s %(message)s"
        )
        configured_path = Path(text("KMD_LOG_FILE")).expanduser()
        try:
            configured_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                configured_path,
                maxBytes=integer("KMD_LOG_MAX_BYTES"),
                backupCount=integer("KMD_LOG_BACKUP_COUNT"),
                encoding="utf-8",
            )
        except OSError:
            fallback = _fallback_log_file()
            fallback.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                fallback,
                maxBytes=integer("KMD_LOG_MAX_BYTES"),
                backupCount=integer("KMD_LOG_BACKUP_COUNT"),
                encoding="utf-8",
            )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler._kmd_managed = True  # type: ignore[attr-defined]
        logger.addHandler(file_handler)
        if boolean("KMD_LOG_STDERR"):
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setLevel(level)
            stderr_handler.setFormatter(formatter)
            stderr_handler._kmd_managed = True  # type: ignore[attr-defined]
            logger.addHandler(stderr_handler)
        _CONFIGURED = True
        logger.info("kmd-runtime logging_initialized level=%s file=%s", level_name, file_handler.baseFilename)
        return logger


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
