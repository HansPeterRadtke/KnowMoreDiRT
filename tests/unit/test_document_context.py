from __future__ import annotations

from pathlib import Path

from knowmoredirt.bounded_dspg import _context_accessible
from knowmoredirt.document_context import (
    DocumentContextEnvelope,
    DocumentDiscourseRelation,
    apply_document_context_envelope,
    apply_document_context_map,
    classify_document_context_map,
    _boundary_material,
    _uncovered_windows,
)
from knowmoredirt.models import Document, Sentence
from knowmoredirt.query import QueryFrame
from knowmoredirt.store import DSPGStore


class FakeContextClient:
    def __init__(self):
        self.calls = 0

    def context_size(self): return 65536
    def cache_fingerprint(self): return {"model_id": "fake", "context_size": 65536, "request_settings": {}, "transport_settings": {}}
    def complete_json(self, prompt, *, n_predict, json_schema):
        self.calls += 1
        assert "[CHUNK 0]" in prompt and "[CHUNK 2]" in prompt
        return {
            "context_segments": [{
                "kind": "dreamed", "start_chunk": 0, "end_chunk": 1, "evidence_chunk": 2,
                "evidence_text": "Then I woke up. It had all been a dream.", "holder_surface": "",
                "reason": "closing boundary scopes prior chunks", "confidence": 0.99,
            }],
            "temporal_scopes": [{
                "temporal_value": "2026-07-01", "start_chunk": 1, "end_chunk": 1, "evidence_chunk": 1,
                "evidence_text": "Dated 2026-07-01", "reason": "dated section header", "confidence": 0.95,
            }],
        }


def _document(tmp_path: Path):
    parts = [
        "The flying-car law requires blue lights.",
        "Dated 2026-07-01. More events happened inside the dream.",
        "Then I woke up. It had all been a dream.",
    ]
    text = "\n".join(parts)
    path = tmp_path / "story.txt"; path.write_text(text, encoding="utf-8")
    doc = Document("d1", path, "story.txt", text, len(text.encode()), 1.0, 1.0, "sha")
    sents=[]; pos=0
    for i,part in enumerate(parts):
        start=text.index(part,pos); end=start+len(part); pos=end
        sents.append(Sentence(f"s{i}","d1","story.txt",part,i,start,end))
    return doc,sents


def _store(tmp_path: Path, doc: Document, sentences: list[Sentence]) -> DSPGStore:
    store=DSPGStore(); run_id="run"
    store.execute("INSERT INTO extraction_runs VALUES (?, ?, ?, ?, ?)",(run_id,1.0,str(tmp_path),"running","{}"))
    store.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",(doc.document_id,run_id,str(doc.path),doc.rel_path,doc.sha256,doc.size_bytes,doc.mtime,doc.ctime,len(doc.text),"{}"))
    for i,sentence in enumerate(sentences):
        chunk_id=f"c{i}"; span_id=f"sp{i}"; ctx=f"ctx{i}"; box=f"box{i}"
        store.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",(chunk_id,doc.document_id,i,sentence.char_start,sentence.char_end,sentence.text,10))
        store.execute("INSERT INTO source_spans VALUES (?, ?, ?, ?, ?, ?, ?, ?)",(span_id,doc.document_id,chunk_id,sentence.char_start,sentence.char_end,sentence.text,sentence.text.lower(),"sentence"))
        store.execute("INSERT INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, NULL, NULL, ?, ?)",(ctx,run_id,"drs:asserted","asserted",1.0))
        store.execute("INSERT INTO drs_boxes VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, ?, ?, ?, ?)",(box,run_id,span_id,"b0",ctx,"asserted","asserted",1.0,"local_model_drs","{}"))
        store.execute("INSERT INTO context_assignments VALUES (?, ?, ?, 'source_span', ?, ?, ?)",(f"ca{i}",run_id,ctx,span_id,span_id,1.0))
    store.execute("INSERT INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, NULL, NULL, ?, ?)",("global_asserted",run_id,"asserted","asserted",1.0))
    store.execute("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",("f1",run_id,"global_asserted","require","require","requires",1.0,"deterministic","sp0"))
    store.commit(); return store


def test_document_context_map_is_cached_and_source_grounded(tmp_path: Path, monkeypatch) -> None:
    doc,sentences=_document(tmp_path); monkeypatch.setenv("KMD_DOCUMENT_CONTEXT_CACHE_DIR",str(tmp_path/"cache")); client=FakeContextClient()
    contexts,temporals=classify_document_context_map(doc,sentences,client)
    again=classify_document_context_map(doc,sentences,client)
    assert contexts[0].kind=="dreamed" and (contexts[0].start_chunk,contexts[0].end_chunk)==(0,1)
    assert temporals[0].temporal_value=="2026-07-01"
    assert again==(contexts,temporals) and client.calls==1


def test_context_range_can_cover_middle_chunks_and_temporal_scope(tmp_path: Path) -> None:
    doc,sentences=_document(tmp_path); store=_store(tmp_path,doc,sentences)
    env=DocumentContextEnvelope(True,"dreamed","chunk_range","Then I woke up. It had all been a dream.","","closure",0.99,0,1,2)
    from knowmoredirt.document_context import DocumentTemporalScope
    temporal=DocumentTemporalScope("2026-07-01","Dated 2026-07-01","header",0.95,1,1,1)
    result=apply_document_context_map(store,"run",doc,[env],[temporal])
    assert result["spans_rebound"]==2 and result["temporal_edges_added"]==1
    parent0=store.execute("SELECT parent_context_id FROM contexts WHERE context_id='ctx0'").fetchone()[0]
    parent1=store.execute("SELECT parent_context_id FROM contexts WHERE context_id='ctx1'").fetchone()[0]
    assert parent0==parent1 and store.execute("SELECT kind FROM contexts WHERE context_id=?",(parent0,)).fetchone()[0]=="drs:dreamed"
    assert store.execute("SELECT parent_context_id FROM contexts WHERE context_id='ctx2'").fetchone()[0] is None
    assert store.execute("SELECT context_id FROM frames WHERE frame_id='f1'").fetchone()[0]==parent0
    edge=store.execute("SELECT temporal_value,context_id FROM temporal_edges WHERE source_span_id='sp1'").fetchone()
    assert edge[0]=="2026-07-01" and edge[1]=="ctx1"
    assert store.execute("SELECT parent_context_id FROM contexts WHERE context_id='ctx1'").fetchone()[0] == parent0
    edge_types = {row[0] for row in store.execute("SELECT relation_type FROM discourse_edges").fetchall()}
    assert "retroactive_scope" in edge_types
    assert "continuation" in edge_types
    records={"contexts":[{"context_id":"ctx0","kind":"drs:asserted","parent_context_id":parent0},{"context_id":parent0,"kind":"drs:dreamed","parent_context_id":""}]}
    frame=QueryFrame("What is the real law?","content_phrase",(),(),"law",(),())
    assert _context_accessible("ctx0",records,frame) is False


def test_single_envelope_wrapper_remains_compatible(tmp_path: Path) -> None:
    doc,sentences=_document(tmp_path); store=_store(tmp_path,doc,sentences)
    env=DocumentContextEnvelope(True,"reported","chunk_range","Dated 2026-07-01","","report",0.9,1,1,1)
    assert apply_document_context_envelope(store,"run",doc,env)==1


def test_document_context_map_rejects_same_range_ambiguity() -> None:
    from knowmoredirt.document_context import _validate_nested_or_disjoint
    from knowmoredirt.model import LocalModelUnavailableError
    import pytest

    left = DocumentContextEnvelope(True, "dreamed", "chunk_range", "e", "", "", 0.9, 0, 2, 2)
    right = DocumentContextEnvelope(True, "reported", "chunk_range", "e", "", "", 0.9, 0, 2, 2)
    with pytest.raises(LocalModelUnavailableError, match="same chunk range"):
        _validate_nested_or_disjoint([left, right])


def test_empty_document_context_map_has_complete_stats_shape(tmp_path: Path) -> None:
    doc, sentences = _document(tmp_path)
    store = _store(tmp_path, doc, sentences)
    result = apply_document_context_map(store, "run", doc, [], [])
    assert result == {
        "contexts_applied": 0,
        "temporal_scopes_applied": 0,
        "spans_rebound": 0,
        "temporal_edges_added": 0,
        "discourse_edges_added": 0,
        "authority_contexts_updated": 0,
    }


def test_invalid_document_context_result_is_not_cached(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from knowmoredirt.model import LocalModelUnavailableError

    doc, sentences = _document(tmp_path)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("KMD_DOCUMENT_CONTEXT_CACHE_DIR", str(cache_root))

    class BadClient(FakeContextClient):
        def complete_json(self, prompt, *, n_predict, json_schema):
            self.calls += 1
            return {
                "context_segments": [{
                    "kind": "dreamed", "start_chunk": 0, "end_chunk": 1, "evidence_chunk": 2,
                    "evidence_text": "fabricated evidence", "holder_surface": "", "reason": "bad", "confidence": 0.99,
                }],
                "temporal_scopes": [],
            }

    client = BadClient()
    with pytest.raises(LocalModelUnavailableError, match="exact source substring"):
        classify_document_context_map(doc, sentences, client)
    assert list(cache_root.glob("*.json")) == []


def test_boundary_material_trims_header_overhead_instead_of_failing(monkeypatch) -> None:
    from context_capacity import context_char_capacity
    from knowmoredirt.document_context import _boundary_material

    monkeypatch.setenv("KMD_DOCUMENT_CONTEXT_BOUNDARY_RATIO", "0.20")
    sentences = [
        Sentence("a", "d", "large.json", "A" * 60000, 0, 0, 60000),
        Sentence("b", "d", "large.json", "B" * 60000, 1, 60001, 120001),
    ]
    material = _boundary_material(sentences, 65536)
    limit = context_char_capacity(
        65536,
        ratio_names=("KMD_DOCUMENT_CONTEXT_BOUNDARY_RATIO",),
        ratio_default=0.20,
    )
    assert len(material) <= limit
    assert material.startswith("[CHUNK 0]\n")
    assert "[CHUNK 1]\n" in material
    assert "A" in material and "B" in material


def test_document_context_schema_is_portable_and_keeps_index_bounds_local() -> None:
    from knowmoredirt.document_context import _map_schema
    from knowmoredirt.model import validate_portable_json_schema

    schema = _map_schema(3)
    validate_portable_json_schema(schema)
    index_schema = schema["properties"]["context_segments"]["items"]["properties"]["start_chunk"]
    assert index_schema == {"type": "integer"}


def test_document_context_rejects_out_of_range_index_after_decode(tmp_path, monkeypatch) -> None:
    from knowmoredirt.document_context import classify_document_context_map
    from knowmoredirt.model import LocalModelUnavailableError

    class BadIndexClient:
        def context_size(self) -> int:
            return 65536
        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "bad-index", "context_size": 65536}
        def complete_json(self, *_args, **_kwargs):
            return {
                "context_segments": [{
                    "kind": "dreamed",
                    "start_chunk": 0,
                    "end_chunk": 9,
                    "evidence_chunk": 0,
                    "evidence_text": "dream",
                    "holder_surface": "",
                    "reason": "test",
                    "confidence": 1.0,
                }],
                "temporal_scopes": [],
            }

    monkeypatch.setenv("KMD_DOCUMENT_CONTEXT_CACHE_DIR", str(tmp_path / "bad-index-cache"))
    doc, sentences = _document(tmp_path)
    import pytest
    with pytest.raises(LocalModelUnavailableError, match="invalid context chunk range"):
        classify_document_context_map(doc, sentences, BadIndexClient())


def test_document_context_prompt_allows_unmistakable_sleep_scope_without_dream_word(tmp_path: Path, monkeypatch) -> None:
    class PromptClient(FakeContextClient):
        def complete_json(self, prompt, *, n_predict, json_schema):
            self.calls += 1
            assert "does not require the literal words dream or dreamed" in prompt
            assert "occurred only during sleep" in prompt
            assert "fantastical content by itself is never sufficient" in prompt
            return {"context_segments": [], "temporal_scopes": []}

    doc, sentences = _document(tmp_path)
    monkeypatch.setenv("KMD_DOCUMENT_CONTEXT_CACHE_DIR", str(tmp_path / "implicit-prompt-cache"))
    classify_document_context_map(doc, sentences, PromptClient())


def test_source_grounded_typed_discourse_relation_is_persisted(tmp_path: Path) -> None:
    doc, sentences = _document(tmp_path)
    store = _store(tmp_path, doc, sentences)
    relation = DocumentDiscourseRelation(
        "correction",
        0,
        2,
        2,
        "Then I woke up. It had all been a dream.",
        "later text revises the interpretation of prior text",
        0.99,
    )
    result = apply_document_context_map(store, "run", doc, [], [], [relation])
    assert result["discourse_edges_added"] >= 3  # two deterministic continuation edges + correction
    row = store.execute(
        "SELECT relation_type, from_span_id, to_span_id, evidence_surface, confidence "
        "FROM discourse_edges WHERE relation_type='correction'"
    ).fetchone()
    assert row is not None
    assert row[0] == "correction"
    assert row[1] == "sp0" and row[2] == "sp2"
    assert row[3] == "Then I woke up. It had all been a dream."
    assert row[4] == 0.99


def test_retroactive_scope_carrier_fifty_chunks_away_rebinds_prior_region(tmp_path: Path) -> None:
    parts = [f"Chunk {index} ordinary narrative." for index in range(50)] + [
        "Then I woke up. It had all been a dream."
    ]
    text = "\n".join(parts)
    path = tmp_path / "long_story.txt"
    path.write_text(text, encoding="utf-8")
    doc = Document("long-doc", path, "long_story.txt", text, len(text.encode()), 1.0, 1.0, "long-sha")
    sentences: list[Sentence] = []
    cursor = 0
    for index, part in enumerate(parts):
        start = text.index(part, cursor)
        end = start + len(part)
        cursor = end
        sentences.append(Sentence(f"ls{index}", doc.document_id, doc.rel_path, part, index, start, end))
    store = _store(tmp_path, doc, sentences)
    envelope = DocumentContextEnvelope(
        True,
        "dreamed",
        "chunk_range",
        "Then I woke up. It had all been a dream.",
        "",
        "retroactive closing carrier",
        0.99,
        0,
        49,
        50,
    )
    result = apply_document_context_map(store, "run", doc, [envelope], [])
    assert result["spans_rebound"] == 50
    parent_first = store.execute("SELECT parent_context_id FROM contexts WHERE context_id='ctx0'").fetchone()[0]
    parent_last = store.execute("SELECT parent_context_id FROM contexts WHERE context_id='ctx49'").fetchone()[0]
    assert parent_first and parent_first == parent_last
    assert store.execute("SELECT kind FROM contexts WHERE context_id=?", (parent_first,)).fetchone()[0] == "drs:dreamed"
    assert store.execute("SELECT parent_context_id FROM contexts WHERE context_id='ctx50'").fetchone()[0] is None
    edge = store.execute(
        "SELECT relation_type, source_span_id FROM discourse_edges WHERE relation_type='retroactive_scope'"
    ).fetchone()
    assert edge is not None and edge[0] == "retroactive_scope" and edge[1] == "sp50"


def test_hundred_large_chunks_keep_middle_scope_carrier_in_full_coverage_scan() -> None:
    sentences: list[Sentence] = []
    marker = "MIDDLE_SCOPE_CARRIER then I woke up and everything before this was a dream"
    for index in range(100):
        words = [f"token{index}"] * 1000
        if index == 50:
            words[500] = marker
        text = " ".join(words)
        sentences.append(Sentence(f"s{index}", "doc", "large.txt", text, index, 0, len(text)))
    boundary = _boundary_material(sentences, 131072)
    assert "[CHUNK 0]" in boundary and "[CHUNK 99]" in boundary
    assert marker not in boundary  # deliberately buried away from both boundaries
    windows = _uncovered_windows(sentences[50], boundary, 131072)
    assert windows
    assert any(marker in window for window in windows)
    # The recording's concrete stress shape is represented directly: one hundred
    # chunks with roughly one thousand source tokens each.
    assert len(sentences) == 100
    assert all(len(sentence.text.split()) >= 1000 for sentence in sentences)
