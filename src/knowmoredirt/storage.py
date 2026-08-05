"""Physical storage configuration for the logical DRT/DSPG database."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3


class StorageBackendError(RuntimeError):
    """Raised when a configured physical backend is unavailable or unsupported."""


@dataclass(frozen=True, slots=True)
class StoreConfig:
    """Select the physical database without changing DRT semantics.

    SQLite is the supported local backend.  Larger deployments can add another
    transactional backend behind this boundary while preserving the normalized
    DRT schema, provenance records, accessibility rules, and query behavior.
    """

    backend: str = "sqlite"
    location: str = ":memory:"
    create_indexes: bool = True
    durable: bool = True
    busy_timeout_ms: int = 30_000

    @classmethod
    def sqlite(
        cls,
        path: str | Path = ":memory:",
        *,
        create_indexes: bool = True,
        durable: bool = True,
        busy_timeout_ms: int = 30_000,
    ) -> "StoreConfig":
        return cls(
            backend="sqlite",
            location=str(path),
            create_indexes=create_indexes,
            durable=durable,
            busy_timeout_ms=busy_timeout_ms,
        )


def open_sqlite(config: StoreConfig) -> sqlite3.Connection:
    if config.backend != "sqlite":
        raise StorageBackendError(
            f"storage backend {config.backend!r} is not implemented; "
            "use sqlite or provide a backend adapter"
        )
    connection = sqlite3.connect(
        config.location,
        timeout=max(config.busy_timeout_ms, 0) / 1000,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout={max(config.busy_timeout_ms, 0)}")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA temp_store=MEMORY")
    if config.location == ":memory:":
        connection.execute("PRAGMA journal_mode=MEMORY")
        connection.execute("PRAGMA synchronous=OFF")
    elif config.durable:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
    else:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=OFF")
    return connection
