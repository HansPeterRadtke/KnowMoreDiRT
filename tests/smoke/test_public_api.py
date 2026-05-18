from __future__ import annotations

import inspect
from pathlib import Path

import knowmoredirt

from conftest import FIXTURE_ROOT, REPO_ROOT


def test_public_contract_exports_only_initialize_and_question() -> None:
    assert knowmoredirt.__all__ == ["initialize", "question"]
    assert callable(knowmoredirt.initialize)
    assert callable(knowmoredirt.question)
    assert "KnowMoreDiRTEngine" not in knowmoredirt.__all__

    public_callables = {
        name
        for name, value in inspect.getmembers(knowmoredirt)
        if callable(value) and not name.startswith("_")
    }
    assert public_callables == {"initialize", "question"}


def test_public_contract_is_documented() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    contract = (REPO_ROOT / "docs" / "public_contract.md").read_text(encoding="utf-8")
    combined = readme + "\n" + contract

    assert "initialize(folder_path)" in combined
    assert "question(text) -> string" in combined
    assert "raw text" in combined
    assert "prepared corpora" in combined


def test_initialize_and_question_work_on_random_raw_folder(tmp_path: Path) -> None:
    (tmp_path / "x9" / "nested").mkdir(parents=True)
    (tmp_path / "x9" / "nested" / "no_extension").write_text(
        "Weird raw note: Orla tested WidgetMoss on Tuesday.", encoding="utf-8"
    )
    (tmp_path / "strange.qqq").write_text(
        "{this looks like jsonish text but is only raw text}", encoding="utf-8"
    )

    knowmoredirt.initialize(tmp_path)
    answer = knowmoredirt.question("Who tested WidgetMoss?")

    assert isinstance(answer, str)


def test_smoke_answers_simple_fixture_question() -> None:
    knowmoredirt.initialize(FIXTURE_ROOT)

    assert knowmoredirt.question("Who drafted the escrow import design for LumaLedger?") == "Nina Vale"

