from pathlib import Path

import pytest

from knowmoredirt.storage import StorageBackendError, StoreConfig
from knowmoredirt.store import DSPGStore


def test_memory_store_preserves_existing_default() -> None:
    store = DSPGStore()
    assert store.path == ":memory:"
    assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_file_store_uses_durable_wal(tmp_path: Path) -> None:
    path = tmp_path / "drt.sqlite3"
    store = DSPGStore(config=StoreConfig.sqlite(path))
    assert store.path == str(path)
    assert store.connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert store.connection.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_unsupported_backend_fails_explicitly() -> None:
    with pytest.raises(StorageBackendError, match="not implemented"):
        DSPGStore(config=StoreConfig(backend="postgresql", location="postgresql://example"))
