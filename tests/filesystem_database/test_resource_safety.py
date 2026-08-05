from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest

from file_system_catalog import content_pipeline, scanner


def test_content_reader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable")

    with pytest.raises(OSError):
        content_pipeline._read_regular_file_bytes(link, max_bytes=100)


def test_content_reader_rejects_oversized_file_without_truncation(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * 101)

    with pytest.raises(RuntimeError, match="exceeds host-memory safety limit"):
        content_pipeline._read_regular_file_bytes(path, max_bytes=100)


def test_content_reader_reads_exact_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "small.txt"
    path.write_bytes(b"complete")

    assert content_pipeline._read_regular_file_bytes(path, max_bytes=100) == b"complete"


def test_office_metadata_rejects_expanded_member_over_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "bomb.docx"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("docProps/core.xml", "<root>" + ("x" * 10000) + "</root>")
    monkeypatch.setenv("KMD_OFFICE_METADATA_MEMORY_RATIO", "1")
    monkeypatch.setenv("KMD_OFFICE_METADATA_MAX_EXPANSION_RATIO", "1")

    metadata, error = scanner.office_metadata(os.fsencode(path))

    assert metadata == {}
    assert error is not None
    assert "exceeds safety limit" in error


def test_office_metadata_reads_small_core_xml(tmp_path: Path) -> None:
    path = tmp_path / "small.docx"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("docProps/core.xml", "<root><title>Safe</title></root>")

    metadata, error = scanner.office_metadata(os.fsencode(path))

    assert error is None
    assert metadata["core"]["title"] == "Safe"
