from __future__ import annotations

import argparse
import json
import os
import sys

from .scanner import FilesystemScanner, PENDING_CONTENT


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Scan one filesystem tree into one wide SQLite table.")
    value.add_argument("root", help="filesystem tree to scan")
    value.add_argument("database", help="SQLite database to create")
    value.add_argument("--replace", action="store_true", help="atomically replace an existing database")
    value.add_argument("--max-hash-bytes", type=int, default=256 * 1024 * 1024)
    value.add_argument("--content-placeholder", default=PENDING_CONTENT)
    value.add_argument("--progress-every", type=int, default=1000)
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        scanner = FilesystemScanner(
            os.fsencode(arguments.root),
            max_hash_bytes=arguments.max_hash_bytes,
            content_placeholder=arguments.content_placeholder,
            progress_every=arguments.progress_every,
        )
        result = scanner.scan_to_database(arguments.database, replace=arguments.replace)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(f"scan failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
