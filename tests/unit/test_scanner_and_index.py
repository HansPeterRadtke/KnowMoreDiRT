from __future__ import annotations

from pathlib import Path

from knowmoredirt.index import LexicalIndex
from knowmoredirt.scanner import scan_folder

from conftest import FIXTURE_ROOT


def test_scanner_collects_documents_sentences_and_metadata() -> None:
    documents, sentences = scan_folder(FIXTURE_ROOT)

    assert len(documents) == 30
    assert len(sentences) > 50
    assert all(document.sha256 for document in documents)
    assert all(document.rel_path for document in documents)
    assert any(document.rel_path.endswith("no-extension-note") for document in documents)


def test_lexical_index_retrieves_source_sentences(tmp_path: Path) -> None:
    (tmp_path / "plain").write_text("Omar reviewed REF-8042 before noon.", encoding="utf-8")
    _, sentences = scan_folder(tmp_path)
    index = LexicalIndex(sentences)

    results = index.search("Who reviewed REF-8042?", limit=3)

    assert results
    assert any("Omar reviewed REF-8042" in sentence.text for sentence, _ in results)


def test_scanner_bounds_very_long_line_units(tmp_path: Path) -> None:
    text = " ".join(f"token{i:03d}" for i in range(80))
    (tmp_path / "long.jsonish").write_text(text, encoding="utf-8")

    _, sentences = scan_folder(tmp_path, max_unit_chars=120)

    assert len(sentences) > 1
    assert all(len(sentence.text) <= 120 for sentence in sentences)
    assert "token000" in sentences[0].text
    assert "token079" in sentences[-1].text
