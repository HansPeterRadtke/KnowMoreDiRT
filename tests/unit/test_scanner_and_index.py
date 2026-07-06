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


def test_scanner_skips_configured_kmd_cache_inside_source_root(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "source.txt").write_text("Source fact Alpha state green.", encoding="utf-8")
    cache_root = tmp_path / ".kmd-generated-cache"
    cache_root.mkdir()
    (cache_root / "cached.json").write_text(
        '{"drs":{"conditions":[{"evidence_text":"Alpha state red"}]}}',
        encoding="utf-8",
    )
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "user-note.txt").write_text("User cache folder is raw source.", encoding="utf-8")
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_root))

    documents, sentences = scan_folder(tmp_path)

    rel_paths = {document.rel_path for document in documents}
    assert "notes/source.txt" in rel_paths
    assert "cache/user-note.txt" in rel_paths
    assert ".kmd-generated-cache/cached.json" not in rel_paths
    assert all("Alpha state red" not in sentence.text for sentence in sentences)


def test_scanner_packs_sentence_units_when_configured(tmp_path: Path) -> None:
    text = "Alpha state red.\nBeta owner Iris.\nGamma deadline Friday."
    (tmp_path / "notes.txt").write_text(text, encoding="utf-8")

    _, unpacked = scan_folder(tmp_path)
    _, packed = scan_folder(tmp_path, pack_unit_chars=1000)

    assert len(unpacked) == 3
    assert len(packed) == 1
    assert packed[0].text == text
    assert packed[0].char_start == 0
    assert packed[0].char_end == len(text)


def test_scanner_pack_respects_pack_limit(tmp_path: Path) -> None:
    text = "Alpha state red.\nBeta owner Iris.\nGamma deadline Friday."
    (tmp_path / "notes.txt").write_text(text, encoding="utf-8")

    _, packed = scan_folder(tmp_path, pack_unit_chars=34)

    assert len(packed) == 2
    assert packed[0].text == "Alpha state red.\nBeta owner Iris."
    assert packed[1].text == "Gamma deadline Friday."
