from __future__ import annotations

import os
from pathlib import Path

import pytest

from knowmoredirt.scanner import scan_folder
from knowmoredirt.text import split_units


def test_scanner_does_not_follow_file_symlink_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("external secret", encoding="utf-8")
    (root / "inside.txt").write_text("inside", encoding="utf-8")
    try:
        os.symlink(outside, root / "linked.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable")

    documents, units = scan_folder(root)

    assert [document.rel_path for document in documents] == ["inside.txt"]
    assert all("external secret" not in unit.text for unit in units)


def test_scanner_skips_file_that_disappears_during_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    disappearing = root / "gone.txt"
    disappearing.write_text("gone", encoding="utf-8")
    stable = root / "stable.txt"
    stable.write_text("stable", encoding="utf-8")

    original_resolve = Path.resolve

    def resolving(path: Path, *args: object, **kwargs: object) -> Path:
        result = original_resolve(path, *args, **kwargs)
        if path == disappearing and disappearing.exists():
            disappearing.unlink()
        return result

    monkeypatch.setattr(Path, "resolve", resolving)
    documents, _units = scan_folder(root)

    assert [document.rel_path for document in documents] == ["stable.txt"]


def test_long_document_preserves_sentence_boundaries_before_hard_split() -> None:
    text = "Alice is open. " + ("Middle filler sentence. " * 100) + "Bob is closed."

    units = split_units(text, max_unit_chars=80)

    assert units[0][2] == "Alice is open."
    assert units[-1][2] == "Bob is closed."
    assert all(len(value) <= 80 for _start, _end, value in units)
    assert sum(value == "Middle filler sentence." for _start, _end, value in units) == 100
    assert all(text[start:end] == value for start, end, value in units)


def test_scanner_uses_finite_default_unit_bound_without_dropping_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    text = "A" * 250
    (root / "long.txt").write_text(text, encoding="utf-8")
    monkeypatch.setenv("KMD_SCANNER_DEFAULT_UNIT_CHARS", "100")

    documents, units = scan_folder(root)

    assert documents[0].text == text
    assert len(units) == 3
    assert all(len(unit.text) <= 100 for unit in units)
    assert "".join(unit.text for unit in units) == text
