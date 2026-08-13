from __future__ import annotations

from pathlib import Path

from knowmoredirt.bounded_dspg import _context_accessible
from knowmoredirt.document_context import (
    DocumentContextEnvelope,
    apply_document_context_envelope,
    apply_document_context_map,
    classify_document_context_map,
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
        store.execute("INSERT INTO contexts VALUES (?, ?, ?, NULL, NULL, ?, ?)",(ctx,run_id,"drs:asserted","asserted",1.0))
        store.execute("INSERT INTO drs_boxes VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, ?, ?, ?, ?)",(box,run_id,span_id,"b0",ctx,"asserted","asserted",1.0,"local_model_drs","{}"))
        store.execute("INSERT INTO context_assignments VALUES (?, ?, ?, 'source_span', ?, ?, ?)",(f"ca{i}",run_id,ctx,span_id,span_id,1.0))
    store.execute("INSERT INTO contexts VALUES (?, ?, ?, NULL, NULL, ?, ?)",("global_asserted",run_id,"asserted","asserted",1.0))
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
