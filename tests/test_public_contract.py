from __future__ import annotations

from pathlib import Path

import knowmoredirt

from conftest import REPO_ROOT


def test_public_contract_is_documented() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    contract = (REPO_ROOT / "docs" / "public_contract.md").read_text(encoding="utf-8")
    combined = readme + "\n" + contract

    assert "initialize(folder_path)" in combined
    assert "question(text) -> string" in combined
    assert "raw text" in combined
    assert "prepared corpora" in combined


def test_initialize_accepts_only_a_raw_folder_and_question_returns_string(tmp_path: Path) -> None:
    (tmp_path / "x9" / "nested").mkdir(parents=True)
    (tmp_path / "x9" / "nested" / "no_extension").write_text(
        "Weird raw note: Orla tested WidgetMoss on Tuesday.", encoding="utf-8"
    )
    (tmp_path / "strange.qqq").write_text(
        "{this looks like jsonish text but is only raw text}", encoding="utf-8"
    )

    session = knowmoredirt.initialize(tmp_path)
    answer = session.question("Who tested WidgetMoss?")

    assert session.is_stub is True
    assert len(session.readable_files) == 2
    assert isinstance(answer, str)
    assert answer == "unknown"


def test_initialize_rejects_non_folder(tmp_path: Path) -> None:
    file_path = tmp_path / "plain-file"
    file_path.write_text("not a folder", encoding="utf-8")
    try:
        knowmoredirt.initialize(file_path)
    except NotADirectoryError:
        return
    raise AssertionError("initialize must reject non-folder paths")

