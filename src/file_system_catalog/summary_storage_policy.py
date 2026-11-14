"""Bounded capacity admission for Raspi's canonical source-linked summary FileDB.

The database can grow incrementally without reserving the configured ceiling
up front. A fail-closed host free-disk guard remains separate from SQLite's
transactional page-count ceiling. Neither guard deletes or alters evidence.
"""
from __future__ import annotations

import os
from pathlib import Path

_GIB = 1024 ** 3
DEFAULT_DATABASE_BUDGET_BYTES = 32 * _GIB
MAX_DATABASE_BUDGET_BYTES = 32 * _GIB
MIN_FREE_DISK_RESERVE_BYTES = 32 * _GIB
SQLITE_WRITE_HEADROOM_BYTES = 256 * 1024 ** 2


def storage_budget_snapshot(
    database: Path,
    *,
    budget_bytes: int = DEFAULT_DATABASE_BUDGET_BYTES,
    reserve_bytes: int = MIN_FREE_DISK_RESERVE_BYTES,
    journal_headroom_bytes: int = SQLITE_WRITE_HEADROOM_BYTES,
) -> dict:
    """Inspect effective unprivileged filesystem headroom before a native write."""
    database = Path(database)
    if (type(budget_bytes) is not int or
            not _GIB <= budget_bytes <= MAX_DATABASE_BUDGET_BYTES):
        raise ValueError("invalid bounded FileDB budget")
    if (type(reserve_bytes) is not int or not _GIB <= reserve_bytes <= 128 * _GIB
            or type(journal_headroom_bytes) is not int
            or not 16 * 1024 ** 2 <= journal_headroom_bytes <= 2 * _GIB):
        raise ValueError("invalid disk headroom policy")
    if database.is_symlink() or not database.is_file():
        raise RuntimeError("canonical FileDB missing or unsafe")
    stats = os.statvfs(database.parent)
    free_bytes = stats.f_bavail * stats.f_frsize
    if free_bytes < 0 or stats.f_frsize <= 0:
        raise RuntimeError("invalid measured filesystem headroom")
    current_bytes = database.stat().st_size
    required_free_bytes = reserve_bytes + journal_headroom_bytes
    return {
        "database_bytes": current_bytes,
        "configured_database_budget_bytes": budget_bytes,
        "database_remaining_budget_bytes": max(0, budget_bytes - current_bytes),
        "unprivileged_filesystem_available_bytes": free_bytes,
        "minimum_free_disk_reserve_bytes": reserve_bytes,
        "sqlite_write_headroom_bytes": journal_headroom_bytes,
        "admission_safe": current_bytes < budget_bytes and free_bytes >= required_free_bytes,
    }


def require_storage_admission(database: Path, **kwargs) -> dict:
    """Reject insufficient space BEFORE starting the native KMD write."""
    snapshot = storage_budget_snapshot(database, **kwargs)
    if snapshot["database_bytes"] >= snapshot["configured_database_budget_bytes"]:
        raise RuntimeError("canonical FileDB budget is exhausted; no source rows changed")
    if (snapshot["unprivileged_filesystem_available_bytes"] <
            snapshot["minimum_free_disk_reserve_bytes"] +
            snapshot["sqlite_write_headroom_bytes"]):
        raise RuntimeError("disk reserve would be threatened by a FileDB write; no source rows changed")
    return snapshot
