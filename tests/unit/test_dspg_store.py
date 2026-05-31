from __future__ import annotations

from pathlib import Path

from knowmoredirt.answer_types import ExpectedAnswer
from knowmoredirt.bounded_dspg import _context_accessible, _identity_expanded_terms, execute_bounded_query
from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.ingest import ingest_folder
from knowmoredirt.models import Evidence
from knowmoredirt.query import QueryFrame
from knowmoredirt.store import DSPGStore, stable_id

from conftest import FIXTURE_ROOT


def test_ingest_builds_normalized_dspg_tables() -> None:
    store, run_id, documents, sentences = ingest_folder(FIXTURE_ROOT)
    counts = store.counts()

    assert run_id
    assert len(documents) == 30
    assert len(sentences) > 50
    assert store.integrity_check() == "ok"
    assert counts["documents"] == 30
    assert counts["chunks"] == len(sentences)
    assert counts["source_spans"] >= counts["chunks"]
    assert counts["mentions"] > 50
    assert counts["referents"] > 30
    assert "identity_hypotheses" in counts
    assert counts["contexts"] >= 3
    assert counts["context_carriers"] >= counts["documents"]
    assert counts["context_assignments"] >= counts["chunks"]
    assert counts["frames"] > 20
    assert counts["frame_arguments"] > 20
    assert "temporal_edges" in counts
    assert counts["relations"] > 20
    assert counts["metadata_records"] >= counts["documents"]


def test_engine_exposes_internal_dspg_counts_for_diagnostics_only() -> None:
    engine = KnowMoreDiRTEngine(FIXTURE_ROOT)
    counts = engine.dspg_counts()

    assert engine.dspg_integrity() == "ok"
    assert counts["documents"] == 30
    assert counts["mentions"] > 50
    assert counts["frames"] > 20


def test_evidence_window_uses_chunk_order_for_duplicate_text(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "0")
    (tmp_path / "duplicates.txt").write_text(
        "\n".join(
                [
                    "Repeated source line.",
                    "First neighbor only.",
                    "Neutral spacer.",
                    "Repeated source line.",
                    "Second neighbor only.",
                ]
        ),
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    evidence = Evidence(
        "duplicates.txt",
        "Repeated source line.",
        0.8,
        chunk_order=3,
    )

    window = engine._evidence_window_text(evidence, radius=1, max_chars=500)

    assert "Second neighbor only." in window
    assert "First neighbor only." not in window


def test_store_supports_referent_centric_candidate_retrieval(tmp_path: Path) -> None:
    (tmp_path / "unstructured.note").write_text(
        "A raw note says BlueTensor reviewed REF-4321 for the ledger cache.",
        encoding="utf-8",
    )
    store, run_id, _, _ = ingest_folder(tmp_path)

    rows = store.referent_candidate_chunks(run_id, ["REF-4321"], limit=3)

    assert rows
    assert "BlueTensor reviewed REF-4321" in rows[0]["text"]


def test_store_materializes_model_drs_without_same_surface_merging(tmp_path: Path) -> None:
    text = "Mira Chen said Aero Gate is ready. The release note names Mira Chen as reviewer."
    store = DSPGStore()
    run_id = store.start_run(tmp_path)
    document_id = stable_id("doc", run_id, "note.txt")
    chunk_id = stable_id("chunk", document_id, 0)
    span_id = stable_id("span", chunk_id, "sentence")
    store.execute(
        """
        INSERT INTO documents(
          document_id, run_id, path, rel_path, content_hash, size_bytes, mtime, ctime, char_count, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (document_id, run_id, str(tmp_path / "note.txt"), "note.txt", "sha", len(text), 0.0, 0.0, len(text), "{}"),
    )
    store.execute(
        "INSERT INTO chunks(chunk_id, document_id, chunk_order, char_start, char_end, text, token_estimate) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (chunk_id, document_id, 0, 0, len(text), text, 16),
    )
    store.execute(
        "INSERT INTO source_spans(span_id, document_id, chunk_id, char_start, char_end, surface, surface_norm, span_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (span_id, document_id, chunk_id, 0, len(text), text, "mira chen said aero gate is ready", "sentence"),
    )
    payload = {
        "drs": {
            "schema_version": "chunk-drs-v1",
            "source_id": "note.txt",
            "referents": [
                {"id": "r1", "label": "Mira Chen", "kind": "person", "evidence_text": "Mira Chen"},
                {"id": "r2", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"},
                {"id": "r3", "label": "The release note", "kind": "document", "evidence_text": "The release note"},
            ],
            "boxes": [
                {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text},
                {
                    "id": "b1",
                    "kind": "reported",
                    "parent_id": "b0",
                    "holder_referent_id": "r1",
                    "evidence_text": "Mira Chen said Aero Gate is ready.",
                },
            ],
            "conditions": [
                {
                    "id": "c1",
                    "predicate": "said",
                    "box_id": "b0",
                    "polarity": "positive",
                    "modality": "reported",
                    "temporal_id": "",
                    "arguments": [
                        {
                            "role": "speaker",
                            "target_kind": "referent",
                            "target_id": "r1",
                            "value": "Mira Chen",
                            "value_type": "person",
                            "evidence_text": "Mira Chen",
                        },
                        {
                            "role": "content",
                            "target_kind": "box",
                            "target_id": "b1",
                            "value": "Aero Gate is ready",
                            "value_type": "clause",
                            "evidence_text": "Aero Gate is ready",
                        },
                    ],
                    "evidence_text": "Mira Chen said Aero Gate is ready.",
                },
                {
                    "id": "c2",
                    "predicate": "ready",
                    "box_id": "b1",
                    "polarity": "positive",
                    "modality": "asserted",
                    "temporal_id": "",
                    "arguments": [
                        {
                            "role": "entity",
                            "target_kind": "referent",
                            "target_id": "r2",
                            "value": "",
                            "value_type": "entity",
                            "evidence_text": "Aero Gate",
                        }
                    ],
                    "evidence_text": "Aero Gate is ready",
                },
            ],
            "identity_hypotheses": [
                {
                    "left_referent_id": "r1",
                    "right_referent_id": "r1",
                    "status": "accepted",
                    "evidence_text": "Mira Chen",
                    "confidence": 1.0,
                }
            ],
            "temporal_records": [],
            "evidence_spans": ["Mira Chen said Aero Gate is ready.", "The release note names Mira Chen as reviewer."],
            "semantic_notes": [],
        }
    }

    result = store.materialize_drs_payload(run_id, span_id, text, payload)

    assert result["accepted"] is True
    assert result["inserted"]["drs_boxes"] == 2
    assert store.counts()["drs_conditions"] == 2
    assert store.counts()["drs_condition_arguments"] == 3
    assert store.counts()["drs_identity_hypotheses"] == 1
    assert store.counts()["identity_hypotheses"] == 1
    row = store.execute(
        "SELECT target_kind, target_box_id FROM drs_condition_arguments WHERE role='content'"
    ).fetchone()
    assert row["target_kind"] == "box"
    assert row["target_box_id"]

    reported_context_id = store.execute(
        "SELECT context_id FROM drs_boxes WHERE kind='reported'"
    ).fetchone()["context_id"]
    reported_holder = store.execute(
        "SELECT holder_surface FROM contexts WHERE context_id=?", (reported_context_id,)
    ).fetchone()["holder_surface"]
    assert reported_holder == "Mira Chen"
    ready_arg = store.execute(
        """
        SELECT fa.surface, fa.referent_id
        FROM frame_arguments fa
        JOIN frames f ON f.frame_id=fa.frame_id
        WHERE f.predicate='ready' AND fa.role='entity'
        """
    ).fetchone()
    assert ready_arg["surface"] == "Aero Gate"
    assert ready_arg["referent_id"]
    records = {"contexts": [dict(row) for row in store.execute("SELECT * FROM contexts").fetchall()]}
    unscoped_frame = QueryFrame(
        question_text="What is ready?",
        answer_type="content_phrase",
        answer_variables=("entity",),
        target_anchors=(),
        requested_relation="ready",
        relation_terms=("ready",),
        constraints=(),
    )
    assert _context_accessible(reported_context_id, records, unscoped_frame) is False

    scoped_frame = QueryFrame(
        question_text="What was reported as ready?",
        answer_type="content_phrase",
        answer_variables=("entity",),
        target_anchors=(),
        requested_relation="ready",
        relation_terms=("ready",),
        constraints=(),
        scope_requirements=("reported",),
    )
    assert _context_accessible(reported_context_id, records, scoped_frame) is True

    bad = store.materialize_drs_payload(
        run_id,
        span_id,
        text,
        {
            "drs": {
                **payload["drs"],
                "evidence_spans": ["not in source"],
            }
        },
    )
    assert bad["accepted"] is False
    assert bad["reason"] == "grounding_validation_failed"


def test_model_drs_referents_remain_source_local_without_identity(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "north.txt").write_text(
        "North intake names Jordan Vale as the witness for beacon A.",
        encoding="utf-8",
    )
    (tmp_path / "south.txt").write_text(
        "South intake names Jordan Vale as the witness for beacon B.",
        encoding="utf-8",
    )

    class TwoSurfaceModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-source-local-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "beacon A" in prompt:
                source_id = "north.txt"
                text = "North intake names Jordan Vale as the witness for beacon A."
                beacon = "beacon A"
            else:
                source_id = "south.txt"
                text = "South intake names Jordan Vale as the witness for beacon B."
                beacon = "beacon B"
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": source_id,
                    "referents": [
                        {"id": "r0", "label": "Jordan Vale", "kind": "person", "evidence_text": "Jordan Vale"},
                        {"id": "r1", "label": beacon, "kind": "entity", "evidence_text": beacon},
                    ],
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "witness",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "person",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "person",
                                    "evidence_text": "Jordan Vale",
                                },
                                {
                                    "role": "object",
                                    "target_kind": "referent",
                                    "target_id": "r1",
                                    "value": "",
                                    "value_type": "entity",
                                    "evidence_text": beacon,
                                },
                            ],
                            "evidence_text": text,
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    cache_dir = tmp_path.parent / f"{tmp_path.name}-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    store, _, _, _ = ingest_folder(
        tmp_path,
        semantic_client=TwoSurfaceModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )

    rows = store.execute(
        """
        SELECT referent_id, source_span_id
        FROM drs_referents
        WHERE surface='Jordan Vale'
        ORDER BY source_span_id
        """
    ).fetchall()

    assert len(rows) == 2
    assert len({row["source_span_id"] for row in rows}) == 2
    assert len({row["referent_id"] for row in rows}) == 2
    assert store.execute("SELECT COUNT(*) FROM identity_hypotheses WHERE source='local_model_drs'").fetchone()[0] == 0


def test_store_rejects_identity_hypothesis_without_bilateral_grounding() -> None:
    store = DSPGStore()
    text = "Ari Kade and Bo Noll appear in the roster."
    payload = {
        "drs": {
            "schema_version": "chunk-drs-v2",
            "source_id": "roster.txt",
            "referents": [
                {"id": "r0", "label": "Ari Kade", "kind": "person", "evidence_text": "Ari Kade"},
                {"id": "r1", "label": "Bo Noll", "kind": "person", "evidence_text": "Bo Noll"},
            ],
            "boxes": [
                {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
            ],
            "conditions": [],
            "identity_hypotheses": [
                {
                    "left_referent_id": "r0",
                    "right_referent_id": "r1",
                    "status": "accepted",
                    "evidence_text": "Ari Kade",
                    "confidence": 0.9,
                }
            ],
            "temporal_records": [],
        }
    }

    result = store.materialize_drs_payload("run", "span", text, payload)

    assert result["accepted"] is False
    assert result["reason"] == "schema_validation_failed"
    assert any(str(error).startswith("identity_evidence_missing_side:") for error in result["errors"])


def test_candidate_drs_identity_is_preserved_but_not_used_for_expansion() -> None:
    store = DSPGStore()
    text = "AX-9 may be the same artifact as Amber Kite."
    payload = {
        "drs": {
            "schema_version": "chunk-drs-v2",
            "source_id": "link.txt",
            "referents": [
                {"id": "r0", "label": "AX-9", "kind": "identifier", "evidence_text": "AX-9"},
                {"id": "r1", "label": "Amber Kite", "kind": "artifact", "evidence_text": "Amber Kite"},
            ],
            "boxes": [
                {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
            ],
            "conditions": [],
            "identity_hypotheses": [
                {
                    "left_referent_id": "r0",
                    "right_referent_id": "r1",
                    "status": "candidate",
                    "evidence_text": "AX-9 may be the same artifact as Amber Kite",
                    "confidence": 0.62,
                }
            ],
            "temporal_records": [],
        }
    }

    result = store.materialize_drs_payload("run", "span", text, payload)

    assert result["accepted"] is True
    assert store.counts()["drs_identity_hypotheses"] == 1
    assert store.counts()["identity_hypotheses"] == 0


def test_identity_expanded_retrieval_merges_scattered_drs_chunks(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "opening").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "ending").mkdir()
    (tmp_path / "opening" / "intro.txt").write_text(
        "Opening memo introduces Vesper Key as the expedition object.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "status.txt").write_text(
        "Field code VX-17 status is amber-ready under Delta review.",
        encoding="utf-8",
    )
    (tmp_path / "ending" / "resolution.txt").write_text(
        "The concordance states VX-17 is the same artifact as Vesper Key.",
        encoding="utf-8",
    )

    class ScatteredModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-scattered-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Opening memo" in prompt:
                text = "Opening memo introduces Vesper Key as the expedition object."
                return {
                    "drs": {
                        "schema_version": "chunk-drs-v2",
                        "source_id": "opening/intro.txt",
                        "referents": [
                            {"id": "r0", "label": "Vesper Key", "kind": "artifact", "evidence_text": "Vesper Key"}
                        ],
                        "boxes": [
                            {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                        ],
                        "conditions": [
                            {
                                "id": "c0",
                                "predicate": "introduce",
                                "box_id": "b0",
                                "polarity": "positive",
                                "modality": "asserted",
                                "temporal_id": "",
                                "arguments": [
                                    {
                                        "role": "object",
                                        "target_kind": "referent",
                                        "target_id": "r0",
                                        "value": "",
                                        "value_type": "artifact",
                                        "evidence_text": "Vesper Key",
                                    }
                                ],
                                "evidence_text": text,
                            }
                        ],
                        "identity_hypotheses": [],
                        "temporal_records": [],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            if "Field code VX-17" in prompt:
                text = "Field code VX-17 status is amber-ready under Delta review."
                return {
                    "drs": {
                        "schema_version": "chunk-drs-v2",
                        "source_id": "middle/status.txt",
                        "referents": [
                            {"id": "r0", "label": "VX-17", "kind": "identifier", "evidence_text": "VX-17"}
                        ],
                        "boxes": [
                            {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                        ],
                        "conditions": [
                            {
                                "id": "c0",
                                "predicate": "status",
                                "box_id": "b0",
                                "polarity": "positive",
                                "modality": "asserted",
                                "temporal_id": "",
                                "arguments": [
                                    {
                                        "role": "subject",
                                        "target_kind": "referent",
                                        "target_id": "r0",
                                        "value": "",
                                        "value_type": "identifier",
                                        "evidence_text": "VX-17",
                                    },
                                    {
                                        "role": "state",
                                        "target_kind": "literal",
                                        "target_id": "",
                                        "value": "amber-ready under Delta review",
                                        "value_type": "state",
                                        "evidence_text": "amber-ready under Delta review",
                                    },
                                ],
                                "evidence_text": text,
                            }
                        ],
                        "identity_hypotheses": [],
                        "temporal_records": [],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            text = "The concordance states VX-17 is the same artifact as Vesper Key."
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "ending/resolution.txt",
                    "referents": [
                        {"id": "r0", "label": "VX-17", "kind": "identifier", "evidence_text": "VX-17"},
                        {"id": "r1", "label": "Vesper Key", "kind": "artifact", "evidence_text": "Vesper Key"},
                    ],
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "same_artifact",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "left",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "identifier",
                                    "evidence_text": "VX-17",
                                },
                                {
                                    "role": "right",
                                    "target_kind": "referent",
                                    "target_id": "r1",
                                    "value": "",
                                    "value_type": "artifact",
                                    "evidence_text": "Vesper Key",
                                },
                            ],
                            "evidence_text": text,
                        }
                    ],
                    "identity_hypotheses": [
                        {
                            "left_referent_id": "r0",
                            "right_referent_id": "r1",
                            "status": "accepted",
                            "evidence_text": text,
                            "confidence": 0.92,
                        }
                    ],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    cache_dir = tmp_path.parent / f"{tmp_path.name}-scattered-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=ScatteredModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What status is recorded for Vesper Key?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Vesper Key",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
    )

    answer, diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        frame.question_text,
        frame,
    )

    assert answer is not None
    assert answer.text == "amber-ready under Delta review"
    assert answer.evidence[0].rel_path == "middle/status.txt"
    assert "vx-17" in diagnostics["ranking"]["identity_expanded_target_terms"]
    assert diagnostics["ranking"]["identity_reranked_selected_document_count"] >= 3


def test_identity_expansion_iterates_across_scattered_drs_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "beginning").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "ending").mkdir()
    (tmp_path / "deep").mkdir()
    (tmp_path / "beginning" / "intro.txt").write_text(
        "Opening register introduces Prism Relay as the archive target.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "crosswalk_a.txt").write_text(
        "Crosswalk A says PX-11 is the same artifact as Prism Relay.",
        encoding="utf-8",
    )
    (tmp_path / "ending" / "crosswalk_b.txt").write_text(
        "Crosswalk B says PX-11 is the same artifact as Relay-Prime.",
        encoding="utf-8",
    )
    (tmp_path / "deep" / "status.txt").write_text(
        "Deep status note says Relay-Prime status is cleared.",
        encoding="utf-8",
    )

    class MultiHopIdentityModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-multihop-scattered-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Opening register" in prompt:
                text = "Opening register introduces Prism Relay as the archive target."
                referents = [{"id": "r0", "label": "Prism Relay", "kind": "artifact", "evidence_text": "Prism Relay"}]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "introduce",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "object",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Prism Relay",
                            }
                        ],
                        "evidence_text": text,
                    }
                ]
                identities = []
            elif "Crosswalk A" in prompt:
                text = "Crosswalk A says PX-11 is the same artifact as Prism Relay."
                referents = [
                    {"id": "r0", "label": "PX-11", "kind": "identifier", "evidence_text": "PX-11"},
                    {"id": "r1", "label": "Prism Relay", "kind": "artifact", "evidence_text": "Prism Relay"},
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "same_artifact",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "left",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "PX-11",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Prism Relay",
                            },
                        ],
                        "evidence_text": text,
                    }
                ]
                identities = [
                    {
                        "left_referent_id": "r0",
                        "right_referent_id": "r1",
                        "status": "accepted",
                        "evidence_text": "PX-11 is the same artifact as Prism Relay",
                        "confidence": 0.92,
                    }
                ]
            elif "Crosswalk B" in prompt:
                text = "Crosswalk B says PX-11 is the same artifact as Relay-Prime."
                referents = [
                    {"id": "r0", "label": "PX-11", "kind": "identifier", "evidence_text": "PX-11"},
                    {"id": "r1", "label": "Relay-Prime", "kind": "artifact", "evidence_text": "Relay-Prime"},
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "same_artifact",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "left",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "PX-11",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Relay-Prime",
                            },
                        ],
                        "evidence_text": text,
                    }
                ]
                identities = [
                    {
                        "left_referent_id": "r0",
                        "right_referent_id": "r1",
                        "status": "accepted",
                        "evidence_text": "PX-11 is the same artifact as Relay-Prime",
                        "confidence": 0.92,
                    }
                ]
            else:
                text = "Deep status note says Relay-Prime status is cleared."
                referents = [{"id": "r0", "label": "Relay-Prime", "kind": "artifact", "evidence_text": "Relay-Prime"}]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "status",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "subject",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Relay-Prime",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "cleared",
                                "value_type": "state",
                                "evidence_text": "cleared",
                            },
                        ],
                        "evidence_text": "Relay-Prime status is cleared",
                    }
                ]
                identities = []
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "multi-hop.txt",
                    "referents": referents,
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                    ],
                    "conditions": conditions,
                    "identity_hypotheses": identities,
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=MultiHopIdentityModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What status is recorded for Prism Relay?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Prism Relay",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
    )

    answer, diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        frame.question_text,
        frame,
    )

    assert answer is not None
    assert answer.text == "cleared"
    assert answer.evidence[0].rel_path == "deep/status.txt"
    assert {"px-11", "relay-prime"}.issubset(set(diagnostics["ranking"]["identity_expanded_target_terms"]))
    assert diagnostics["ranking"]["identity_expansion_rounds"] >= 2


def test_identity_expanded_retrieval_respects_reported_scope_against_asserted_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "opening").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "ending").mkdir()
    (tmp_path / "opening" / "registry.txt").write_text(
        "Opening registry introduces Cobalt Lens as the safety artifact.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "report.txt").write_text(
        "Rhea Vale reports that CB-44 status is green.",
        encoding="utf-8",
    )
    (tmp_path / "ending" / "audit.txt").write_text(
        "Audit states CB-44 is Cobalt Lens and CB-44 status is red.",
        encoding="utf-8",
    )

    class ScopedScatteredModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-scoped-scattered-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Opening registry" in prompt:
                text = "Opening registry introduces Cobalt Lens as the safety artifact."
                return {
                    "drs": {
                        "schema_version": "chunk-drs-v2",
                        "source_id": "opening/registry.txt",
                        "referents": [
                            {"id": "r0", "label": "Cobalt Lens", "kind": "artifact", "evidence_text": "Cobalt Lens"}
                        ],
                        "boxes": [
                            {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                        ],
                        "conditions": [
                            {
                                "id": "c0",
                                "predicate": "introduce",
                                "box_id": "b0",
                                "polarity": "positive",
                                "modality": "asserted",
                                "temporal_id": "",
                                "arguments": [
                                    {
                                        "role": "object",
                                        "target_kind": "referent",
                                        "target_id": "r0",
                                        "value": "",
                                        "value_type": "artifact",
                                        "evidence_text": "Cobalt Lens",
                                    }
                                ],
                                "evidence_text": text,
                            }
                        ],
                        "identity_hypotheses": [],
                        "temporal_records": [],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            if "reports that CB-44" in prompt:
                text = "Rhea Vale reports that CB-44 status is green."
                reported = "CB-44 status is green"
                return {
                    "drs": {
                        "schema_version": "chunk-drs-v2",
                        "source_id": "middle/report.txt",
                        "referents": [
                            {"id": "r0", "label": "Rhea Vale", "kind": "person", "evidence_text": "Rhea Vale"},
                            {"id": "r1", "label": "CB-44", "kind": "identifier", "evidence_text": "CB-44"},
                        ],
                        "boxes": [
                            {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text},
                            {
                                "id": "b1",
                                "kind": "reported",
                                "parent_id": "b0",
                                "holder_referent_id": "r0",
                                "evidence_text": reported,
                            },
                        ],
                        "conditions": [
                            {
                                "id": "c0",
                                "predicate": "report",
                                "box_id": "b0",
                                "polarity": "positive",
                                "modality": "asserted",
                                "temporal_id": "",
                                "arguments": [
                                    {
                                        "role": "speaker",
                                        "target_kind": "referent",
                                        "target_id": "r0",
                                        "value": "",
                                        "value_type": "person",
                                        "evidence_text": "Rhea Vale",
                                    },
                                    {
                                        "role": "content",
                                        "target_kind": "box",
                                        "target_id": "b1",
                                        "value": "",
                                        "value_type": "box",
                                        "evidence_text": reported,
                                    },
                                ],
                                "evidence_text": text,
                            },
                            {
                                "id": "c1",
                                "predicate": "status",
                                "box_id": "b1",
                                "polarity": "positive",
                                "modality": "asserted",
                                "temporal_id": "",
                                "arguments": [
                                    {
                                        "role": "subject",
                                        "target_kind": "referent",
                                        "target_id": "r1",
                                        "value": "",
                                        "value_type": "identifier",
                                        "evidence_text": "CB-44",
                                    },
                                    {
                                        "role": "state",
                                        "target_kind": "literal",
                                        "target_id": "",
                                        "value": "green",
                                        "value_type": "state",
                                        "evidence_text": "green",
                                    },
                                ],
                                "evidence_text": reported,
                            },
                        ],
                        "identity_hypotheses": [],
                        "temporal_records": [],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            text = "Audit states CB-44 is Cobalt Lens and CB-44 status is red."
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "ending/audit.txt",
                    "referents": [
                        {"id": "r0", "label": "CB-44", "kind": "identifier", "evidence_text": "CB-44"},
                        {"id": "r1", "label": "Cobalt Lens", "kind": "artifact", "evidence_text": "Cobalt Lens"},
                    ],
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "status",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "subject",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "identifier",
                                    "evidence_text": "CB-44",
                                },
                                {
                                    "role": "state",
                                    "target_kind": "literal",
                                    "target_id": "",
                                    "value": "red",
                                    "value_type": "state",
                                    "evidence_text": "red",
                                },
                            ],
                            "evidence_text": "CB-44 status is red",
                        }
                    ],
                    "identity_hypotheses": [
                        {
                            "left_referent_id": "r0",
                            "right_referent_id": "r1",
                            "status": "accepted",
                            "evidence_text": "CB-44 is Cobalt Lens",
                            "confidence": 0.93,
                        }
                    ],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=ScopedScatteredModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    asserted_frame = QueryFrame(
        question_text="What status is recorded for Cobalt Lens?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Cobalt Lens",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
    )
    reported_frame = QueryFrame(
        question_text="What status did Rhea Vale report for Cobalt Lens?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Cobalt Lens",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
        scope_requirements=("reported",),
    )

    asserted_answer, asserted_diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        asserted_frame.question_text,
        asserted_frame,
    )
    reported_answer, reported_diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        reported_frame.question_text,
        reported_frame,
    )

    assert asserted_answer is not None
    assert asserted_answer.text == "red"
    assert asserted_answer.evidence[0].rel_path == "ending/audit.txt"
    assert reported_answer is not None
    assert reported_answer.text == "green"
    assert reported_answer.evidence[0].rel_path == "middle/report.txt"
    assert "cb-44" in asserted_diagnostics["ranking"]["identity_expanded_target_terms"]
    assert "cb-44" in reported_diagnostics["ranking"]["identity_expanded_target_terms"]


def test_scattered_identity_conflict_returns_unknown_with_source_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "opening").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "ending").mkdir()
    (tmp_path / "opening" / "intro.txt").write_text(
        "Opening registry introduces Marble Lens as the survey artifact.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "status.txt").write_text(
        "Field code ML-9 status is green.",
        encoding="utf-8",
    )
    (tmp_path / "ending" / "identity.txt").write_text(
        "Resolution states ML-9 is the same artifact as Marble Lens.",
        encoding="utf-8",
    )
    (tmp_path / "ending" / "correction.txt").write_text(
        "Audit correction records ML-9 status is red.",
        encoding="utf-8",
    )

    class ConflictingScatteredModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-conflicting-scattered-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Opening registry" in prompt:
                text = "Opening registry introduces Marble Lens as the survey artifact."
                referents = [{"id": "r0", "label": "Marble Lens", "kind": "artifact", "evidence_text": "Marble Lens"}]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "introduce",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "object",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Marble Lens",
                            }
                        ],
                        "evidence_text": text,
                    }
                ]
                identities = []
            elif "Field code ML-9" in prompt:
                text = "Field code ML-9 status is green."
                referents = [{"id": "r0", "label": "ML-9", "kind": "identifier", "evidence_text": "ML-9"}]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "status",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "subject",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "ML-9",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "green",
                                "value_type": "state",
                                "evidence_text": "green",
                            },
                        ],
                        "evidence_text": "ML-9 status is green",
                    }
                ]
                identities = []
            elif "same artifact" in prompt:
                text = "Resolution states ML-9 is the same artifact as Marble Lens."
                referents = [
                    {"id": "r0", "label": "ML-9", "kind": "identifier", "evidence_text": "ML-9"},
                    {"id": "r1", "label": "Marble Lens", "kind": "artifact", "evidence_text": "Marble Lens"},
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "same_artifact",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "left",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "ML-9",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Marble Lens",
                            },
                        ],
                        "evidence_text": "ML-9 is the same artifact as Marble Lens",
                    }
                ]
                identities = [
                    {
                        "left_referent_id": "r0",
                        "right_referent_id": "r1",
                        "status": "accepted",
                        "evidence_text": "ML-9 is the same artifact as Marble Lens",
                        "confidence": 0.93,
                    }
                ]
            else:
                text = "Audit correction records ML-9 status is red."
                referents = [{"id": "r0", "label": "ML-9", "kind": "identifier", "evidence_text": "ML-9"}]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "status",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "subject",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "ML-9",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "red",
                                "value_type": "state",
                                "evidence_text": "red",
                            },
                        ],
                        "evidence_text": "ML-9 status is red",
                    }
                ]
                identities = []
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "fixture.txt",
                    "referents": referents,
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                    ],
                    "conditions": conditions,
                    "identity_hypotheses": identities,
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    cache_dir = tmp_path.parent / f"{tmp_path.name}-conflict-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=ConflictingScatteredModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What status is recorded for Marble Lens?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Marble Lens",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
    )

    answer, diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        frame.question_text,
        frame,
    )

    conflict = diagnostics["execution"]["answer_conflict_without_query_scope"]
    values = {item["value"] for item in conflict["values"]}
    evidence_paths = {
        evidence["rel_path"]
        for item in conflict["values"]
        for evidence in item["evidence"]
    }
    assert answer is None
    assert any("green" in value for value in values)
    assert any("red" in value for value in values)
    assert "middle/status.txt" in evidence_paths
    assert "ending/correction.txt" in evidence_paths
    assert "ml-9" in diagnostics["ranking"]["identity_expanded_target_terms"]


def test_scattered_unlinked_referent_status_returns_unknown_with_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "begin").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "ending").mkdir()
    (tmp_path / "begin" / "registry.txt").write_text(
        "Registry introduces Lumen Core as the archive engine.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "status.txt").write_text(
        "Middle status log says LC-71 status is silver under sealed review.",
        encoding="utf-8",
    )
    (tmp_path / "ending" / "crosswalk.txt").write_text(
        "Ending crosswalk mentions Lumen Core while LC-71 status is silver and no confirmed identity link exists.",
        encoding="utf-8",
    )

    class UnlinkedStatusModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-unlinked-scattered-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Registry introduces" in prompt:
                text = "Registry introduces Lumen Core as the archive engine."
                referents = [{"id": "r0", "label": "Lumen Core", "kind": "artifact", "evidence_text": "Lumen Core"}]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "introduces",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "object",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Lumen Core",
                            }
                        ],
                        "evidence_text": text,
                    }
                ]
                source_id = "begin/registry.txt"
            elif "Middle status log" in prompt:
                text = "Middle status log says LC-71 status is silver under sealed review."
                referents = [{"id": "r0", "label": "LC-71", "kind": "identifier", "evidence_text": "LC-71"}]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "status",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "subject",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "LC-71",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "silver",
                                "value_type": "state",
                                "evidence_text": "silver",
                            },
                        ],
                        "evidence_text": "LC-71 status is silver",
                    }
                ]
                source_id = "middle/status.txt"
            else:
                text = "Ending crosswalk mentions Lumen Core while LC-71 status is silver and no confirmed identity link exists."
                referents = [
                    {"id": "r0", "label": "Lumen Core", "kind": "artifact", "evidence_text": "Lumen Core"},
                    {"id": "r1", "label": "LC-71", "kind": "identifier", "evidence_text": "LC-71"},
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "status",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "subject",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "LC-71",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "silver",
                                "value_type": "state",
                                "evidence_text": "silver",
                            },
                        ],
                        "evidence_text": "LC-71 status is silver",
                    },
                    {
                        "id": "c1",
                        "predicate": "identity_link",
                        "box_id": "b0",
                        "polarity": "negative",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "left",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Lumen Core",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "LC-71",
                            },
                        ],
                        "evidence_text": "no confirmed identity link exists",
                    },
                ]
                source_id = "ending/crosswalk.txt"
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": source_id,
                    "referents": referents,
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                    ],
                    "conditions": conditions,
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    cache_dir = tmp_path.parent / f"{tmp_path.name}-unlinked-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=UnlinkedStatusModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What status is recorded for Lumen Core?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Lumen Core",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
    )

    answer, diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        frame.question_text,
        frame,
    )

    assert answer is None
    assert diagnostics["execution"]["no_answer_reason"] == "no_candidate"
    provenance = diagnostics["execution"]["source_provenance_sample"]
    paths = {item["rel_path"] for item in provenance}
    assert "ending/crosswalk.txt" in paths
    crosswalk = next(item for item in provenance if item["rel_path"] == "ending/crosswalk.txt")
    assert crosswalk["span_id"]
    assert crosswalk["chunk_id"]
    assert crosswalk["char_start"] == 0
    assert crosswalk["document"]["file_name"] == "crosswalk.txt"
    assert "LC-71 status is silver" in crosswalk["text"]


def test_store_rejects_invalid_drs_condition_graphs() -> None:
    store = DSPGStore()
    payload = {
        "drs": {
            "schema_version": "chunk-drs-v2",
            "source_id": "note.txt",
            "referents": [
                {"id": "r0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"}
            ],
            "boxes": [
                {
                    "id": "b0",
                    "kind": "asserted",
                    "parent_id": "",
                    "holder_referent_id": "",
                    "evidence_text": "Aero Gate is ready.",
                }
            ],
            "conditions": [
                {
                    "id": "c0",
                    "predicate": "ready",
                    "box_id": "b0",
                    "polarity": "positive",
                    "modality": "asserted",
                    "temporal_id": "",
                    "arguments": [
                        {
                            "role": "content",
                            "target_kind": "condition",
                            "target_id": "c1",
                            "value": "",
                            "value_type": "condition",
                            "evidence_text": "Aero Gate is ready.",
                        }
                    ],
                    "evidence_text": "Aero Gate is ready.",
                },
                {
                    "id": "c1",
                    "predicate": "state",
                    "box_id": "b0",
                    "polarity": "positive",
                    "modality": "asserted",
                    "temporal_id": "",
                    "arguments": [
                        {
                            "role": "content",
                            "target_kind": "condition",
                            "target_id": "c0",
                            "value": "",
                            "value_type": "condition",
                            "evidence_text": "Aero Gate is ready.",
                        }
                    ],
                    "evidence_text": "Aero Gate is ready.",
                },
            ],
            "identity_hypotheses": [],
            "temporal_records": [],
        }
    }

    result = store.materialize_drs_payload("run", "span", "Aero Gate is ready.", payload)

    assert result["accepted"] is False
    assert result["reason"] == "schema_validation_failed"
    assert "cyclic_condition_argument:c0->c1->c0" in result["errors"]


def test_store_rejects_cyclic_drs_box_parent_graphs() -> None:
    store = DSPGStore()
    text = "Rhea Vale reports that CB-44 status is green."
    payload = {
        "drs": {
            "schema_version": "chunk-drs-v2",
            "source_id": "report.txt",
            "referents": [
                {"id": "r0", "label": "Rhea Vale", "kind": "person", "evidence_text": "Rhea Vale"},
                {"id": "r1", "label": "CB-44", "kind": "identifier", "evidence_text": "CB-44"},
            ],
            "boxes": [
                {"id": "b0", "kind": "asserted", "parent_id": "b1", "holder_referent_id": "", "evidence_text": text},
                {
                    "id": "b1",
                    "kind": "reported",
                    "parent_id": "b0",
                    "holder_referent_id": "r0",
                    "evidence_text": "CB-44 status is green",
                },
            ],
            "conditions": [
                {
                    "id": "c0",
                    "predicate": "status",
                    "box_id": "b1",
                    "polarity": "positive",
                    "modality": "asserted",
                    "temporal_id": "",
                    "arguments": [
                        {
                            "role": "subject",
                            "target_kind": "referent",
                            "target_id": "r1",
                            "value": "",
                            "value_type": "identifier",
                            "evidence_text": "CB-44",
                        },
                        {
                            "role": "state",
                            "target_kind": "literal",
                            "target_id": "",
                            "value": "green",
                            "value_type": "state",
                            "evidence_text": "green",
                        },
                    ],
                    "evidence_text": "CB-44 status is green",
                }
            ],
            "identity_hypotheses": [],
            "temporal_records": [],
        }
    }

    result = store.materialize_drs_payload("run", "span", text, payload)

    assert result["accepted"] is False
    assert result["reason"] == "schema_validation_failed"
    assert "cyclic_box_parent:b0->b1->b0" in result["errors"]


def test_store_rejects_multiple_drs_root_boxes() -> None:
    store = DSPGStore()
    text = "Opening memo introduces Vesper Key. Ending note says VX-17 is Vesper Key."
    payload = {
        "drs": {
            "schema_version": "chunk-drs-v2",
            "source_id": "combined.txt",
            "referents": [
                {"id": "r0", "label": "Vesper Key", "kind": "artifact", "evidence_text": "Vesper Key"},
                {"id": "r1", "label": "VX-17", "kind": "identifier", "evidence_text": "VX-17"},
            ],
            "boxes": [
                {
                    "id": "b0",
                    "kind": "asserted",
                    "parent_id": "",
                    "holder_referent_id": "",
                    "evidence_text": "Opening memo introduces Vesper Key.",
                },
                {
                    "id": "b1",
                    "kind": "asserted",
                    "parent_id": "",
                    "holder_referent_id": "",
                    "evidence_text": "Ending note says VX-17 is Vesper Key.",
                },
            ],
            "conditions": [],
            "identity_hypotheses": [],
            "temporal_records": [],
        }
    }

    result = store.materialize_drs_payload("run", "span", text, payload)

    assert result["accepted"] is False
    assert result["reason"] == "schema_validation_failed"
    assert "multiple_root_boxes:b0,b1" in result["errors"]


def test_bounded_query_uses_relation_level_drs_scope(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "belief.txt").write_text(
        "Kalo Reed believes that Mira Stone archived the Slate Quill.",
        encoding="utf-8",
    )

    class ReportedConditionModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-reported-condition", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "belief.txt",
                    "referents": [
                        {"id": "r0", "label": "Kalo Reed", "kind": "person", "evidence_text": "Kalo Reed"},
                        {"id": "r1", "label": "Mira Stone", "kind": "person", "evidence_text": "Mira Stone"},
                        {"id": "r2", "label": "the Slate Quill", "kind": "artifact", "evidence_text": "the Slate Quill"},
                    ],
                    "boxes": [
                        {
                            "id": "b0",
                            "kind": "asserted",
                            "parent_id": "",
                            "holder_referent_id": "r0",
                            "evidence_text": "believes that Mira Stone archived the Slate Quill",
                        }
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "archive",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "reported",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "agent",
                                    "target_kind": "referent",
                                    "target_id": "r1",
                                    "value": "Mira Stone",
                                    "value_type": "literal",
                                    "evidence_text": "Mira Stone",
                                },
                                {
                                    "role": "theme",
                                    "target_kind": "referent",
                                    "target_id": "r2",
                                    "value": "the Slate Quill",
                                    "value_type": "literal",
                                    "evidence_text": "the Slate Quill",
                                },
                            ],
                            "evidence_text": "Mira Stone archived the Slate Quill",
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / "chunk-drs-cache"))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=ReportedConditionModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    unscoped_frame = QueryFrame(
        question_text="What did Kalo Reed archive?",
        answer_type="content_phrase",
        answer_variables=("what",),
        target_anchors=("Kalo Reed",),
        requested_relation="archive",
        relation_terms=("archive",),
        constraints=(),
    )
    frame = QueryFrame(
        question_text="What does Kalo Reed believe?",
        answer_type="content_phrase",
        answer_variables=("what",),
        target_anchors=("Kalo Reed",),
        requested_relation="believe",
        relation_terms=("believe", "content"),
        constraints=(),
        scope_requirements=("reported",),
    )

    unscoped_answer, _unscoped_diagnostics = execute_bounded_query(
        store, run_id, documents, sentences_by_document, unscoped_frame.question_text, unscoped_frame
    )
    answer, _diagnostics = execute_bounded_query(store, run_id, documents, sentences_by_document, frame.question_text, frame)

    assert unscoped_answer is None
    assert answer is not None
    assert answer.text == "Mira Stone archived the Slate Quill"
    assert answer.reason == "relation_condition_binding"


def test_ingest_can_materialize_schema_constrained_model_drs(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "note.txt").write_text("Aero Gate is ready.\n", encoding="utf-8")

    class FakeDrsModel:
        def __init__(self) -> None:
            self.json_schema_seen = False

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.json_schema_seen = json_schema is not None
            assert "source-grounded DRS" in prompt
            return {
                "drs": {
                    "schema_version": "chunk-drs-v1",
                    "source_id": "note.txt",
                    "referents": [
                        {"id": "r0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"},
                    ],
                    "boxes": [
                        {
                            "id": "b0",
                            "kind": "asserted",
                            "parent_id": "",
                            "holder_referent_id": "",
                            "evidence_text": "Aero Gate is ready.",
                        },
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "ready",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "entity",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "Aero Gate",
                                    "value_type": "entity",
                                    "evidence_text": "Aero Gate",
                                }
                            ],
                            "evidence_text": "Aero Gate is ready.",
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    model = FakeDrsModel()

    store, _, _, _ = ingest_folder(
        tmp_path,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )

    assert model.json_schema_seen is True
    assert store.counts()["drs_boxes"] == 1
    assert store.counts()["drs_conditions"] == 1
    assert store.counts()["drs_condition_arguments"] == 1


def test_drs_ingest_skips_low_semantic_noise_chunks(tmp_path: Path, monkeypatch) -> None:
    noise = "\\x00\\x01@@@###%%%^^^^~~~~" + ("A7f!?" * 80)
    (tmp_path / "noise.blob").write_text(noise, encoding="utf-8")
    (tmp_path / "note.txt").write_text("Aero Gate is ready.\n", encoding="utf-8")

    class CountingDrsModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-counting-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls.append(prompt)
            assert "Aero Gate is ready" in prompt
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "note.txt",
                    "referents": [
                        {"id": "r0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"},
                    ],
                    "boxes": [
                        {
                            "id": "b0",
                            "kind": "asserted",
                            "parent_id": "",
                            "holder_referent_id": "",
                            "evidence_text": "Aero Gate is ready.",
                        },
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "ready",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "entity",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "entity",
                                    "evidence_text": "Aero Gate",
                                }
                            ],
                            "evidence_text": "Aero Gate is ready.",
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    cache_dir = tmp_path.parent / f"{tmp_path.name}-noise-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    model = CountingDrsModel()

    store, _, _, _ = ingest_folder(
        tmp_path,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )

    assert len(model.calls) == 1
    assert store.counts()["drs_conditions"] == 1


def test_ingest_can_incrementally_merge_new_files_into_existing_store(tmp_path: Path) -> None:
    store = DSPGStore()
    (tmp_path / "alpha.txt").write_text("Alpha note one.", encoding="utf-8")

    store, first_run_id, first_documents, first_sentences = ingest_folder(tmp_path, store=store)
    (tmp_path / "beta.txt").write_text("Beta note two.", encoding="utf-8")
    store, second_run_id, second_documents, second_sentences = ingest_folder(tmp_path, store=store)

    assert first_run_id == second_run_id
    assert len(first_documents) == 1
    assert len(first_sentences) == 1
    assert len(second_documents) == 2
    assert len(second_sentences) == 2
    assert first_documents[0].document_id in {document.document_id for document in second_documents}
    assert store.integrity_check() == "ok"
    assert store.counts()["documents"] == 2
    assert store.counts()["chunks"] == 2
    assert store.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0] == 1


def test_incremental_removed_identity_source_does_not_expand_current_query(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "link.txt").write_text(
        "First link says AX-9 is the same artifact as Amber Kite.",
        encoding="utf-8",
    )

    class IncrementalIdentityModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-incremental-identity-provenance", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "First link" in prompt:
                text = "First link says AX-9 is the same artifact as Amber Kite."
                referents = [
                    {"id": "r0", "label": "AX-9", "kind": "identifier", "evidence_text": "AX-9"},
                    {"id": "r1", "label": "Amber Kite", "kind": "artifact", "evidence_text": "Amber Kite"},
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "same_artifact",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "left",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "AX-9",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Amber Kite",
                            },
                        ],
                        "evidence_text": "AX-9 is the same artifact as Amber Kite",
                    }
                ]
                identities = [
                    {
                        "left_referent_id": "r0",
                        "right_referent_id": "r1",
                        "status": "accepted",
                        "evidence_text": "AX-9 is the same artifact as Amber Kite",
                        "confidence": 0.92,
                    }
                ]
                source_id = "link.txt"
            elif "Catalog mentions" in prompt:
                text = "Catalog mentions Amber Kite as an archived item."
                referents = [{"id": "r0", "label": "Amber Kite", "kind": "artifact", "evidence_text": "Amber Kite"}]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "mentions",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "object",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Amber Kite",
                            }
                        ],
                        "evidence_text": text,
                    }
                ]
                identities = []
                source_id = "catalog.txt"
            else:
                text = "Current status note says AX-9 status is blue."
                referents = [{"id": "r0", "label": "AX-9", "kind": "identifier", "evidence_text": "AX-9"}]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "status",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "subject",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "AX-9",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "blue",
                                "value_type": "state",
                                "evidence_text": "blue",
                            },
                        ],
                        "evidence_text": "AX-9 status is blue",
                    }
                ]
                identities = []
                source_id = "status.txt"
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": source_id,
                    "referents": referents,
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                    ],
                    "conditions": conditions,
                    "identity_hypotheses": identities,
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    cache_dir = tmp_path.parent / f"{tmp_path.name}-incremental-identity-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    store = DSPGStore()
    model = IncrementalIdentityModel()
    store, first_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    (tmp_path / "link.txt").unlink()
    (tmp_path / "catalog.txt").write_text(
        "Catalog mentions Amber Kite as an archived item.",
        encoding="utf-8",
    )
    (tmp_path / "status.txt").write_text(
        "Current status note says AX-9 status is blue.",
        encoding="utf-8",
    )
    store, second_run_id, documents, sentences = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )

    assert first_run_id == second_run_id
    stale_identity = store.execute("SELECT source_span_id FROM identity_hypotheses").fetchone()
    assert stale_identity["source_span_id"]
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What status is recorded for Amber Kite?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Amber Kite",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
    )

    answer, diagnostics = execute_bounded_query(
        store,
        second_run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        frame.question_text,
        frame,
    )

    assert answer is None
    assert "identity_expanded_target_terms" not in diagnostics["ranking"]
    provenance = diagnostics["execution"]["source_provenance_sample"]
    assert {item["rel_path"] for item in provenance} == {"catalog.txt"}


def test_incremental_drs_ingest_reuses_existing_materialized_chunks(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "note.txt").write_text("Aero Gate is ready.\n", encoding="utf-8")

    class CountingDrsModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-incremental-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            assert "Aero Gate is ready" in prompt
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "note.txt",
                    "referents": [
                        {"id": "r0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"},
                    ],
                    "boxes": [
                        {
                            "id": "b0",
                            "kind": "asserted",
                            "parent_id": "",
                            "holder_referent_id": "",
                            "evidence_text": "Aero Gate is ready.",
                        },
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "ready",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "entity",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "entity",
                                    "evidence_text": "Aero Gate",
                                }
                            ],
                            "evidence_text": "Aero Gate is ready.",
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    incremental_cache_dir = tmp_path.parent / f"{tmp_path.name}-incremental-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(incremental_cache_dir))
    model = CountingDrsModel()
    store = DSPGStore()

    store, first_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    calls_after_first_ingest = model.calls
    store, second_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )

    assert first_run_id == second_run_id
    assert calls_after_first_ingest >= 1
    assert model.calls == calls_after_first_ingest
    assert store.counts()["drs_conditions"] == 1


def test_incremental_drs_ingest_skips_previous_failed_attempts(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path.parent / f"{tmp_path.name}-failed-attempt-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("KMD_CHUNK_DRS_STAGED_FALLBACK", "0")
    (tmp_path / "note.txt").write_text("Aero Gate is ready.\n", encoding="utf-8")

    class FailingDrsModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-failing-incremental-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "note.txt",
                    "referents": [
                        {"id": "r0", "label": "Aero Gate", "kind": "entity", "evidence_text": "Aero Gate"},
                    ],
                    "boxes": [
                        {
                            "id": "b0",
                            "kind": "asserted",
                            "parent_id": "",
                            "holder_referent_id": "",
                            "evidence_text": "not in source",
                        },
                    ],
                    "conditions": [],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    model = FailingDrsModel()
    store = DSPGStore()

    store, first_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    calls_after_first_ingest = model.calls
    for cache_file in cache_dir.glob("*.json"):
        cache_file.unlink()
    store, second_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )

    assert first_run_id == second_run_id
    assert calls_after_first_ingest == 1
    assert model.calls == calls_after_first_ingest
    assert store.counts()["drs_boxes"] == 0
    assert store.counts()["model_attempts"] == 1
    attempt = store.execute("SELECT accepted, materialized, reason FROM model_attempts").fetchone()
    assert attempt["accepted"] == 0
    assert attempt["materialized"] == 0
    assert attempt["reason"] == "grounding_validation_failed"


def test_incremental_frame_ingest_reuses_existing_materialized_chunks(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("Aero Gate is ready.\n", encoding="utf-8")

    class CountingFrameModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-incremental-frames", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            assert "Aero Gate is ready" in prompt
            return {
                "frames": [
                    {
                        "frame_type": "state",
                        "predicate": "ready",
                        "arguments": [
                            {"role": "entity", "text": "Aero Gate", "value_type": "entity"},
                        ],
                        "identity_hypotheses": [],
                        "polarity": "positive",
                        "modality": "asserted",
                        "context_holder": "",
                        "temporal_text": "",
                        "evidence_text": "Aero Gate is ready.",
                        "confidence": 0.9,
                    }
                ],
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    model = CountingFrameModel()
    store = DSPGStore()

    store, first_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=True,
        use_drs_semantics=False,
    )
    calls_after_first_ingest = model.calls
    store, second_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=True,
        use_drs_semantics=False,
    )

    assert first_run_id == second_run_id
    assert calls_after_first_ingest == 1
    assert model.calls == calls_after_first_ingest
    assert store.execute("SELECT COUNT(*) FROM frames WHERE source='local_model'").fetchone()[0] == 1


def test_incremental_frame_ingest_skips_previous_failed_attempts(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("Aero Gate is ready.\n", encoding="utf-8")

    class FailingFrameModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-failing-incremental-frames", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            assert "Aero Gate is ready" in prompt
            return {
                "frames": [
                    {
                        "frame_type": "state",
                        "predicate": "ready",
                        "arguments": [
                            {"role": "entity", "text": "Aero Gate", "value_type": "entity"},
                        ],
                        "identity_hypotheses": [],
                        "polarity": "positive",
                        "modality": "asserted",
                        "context_holder": "",
                        "temporal_text": "",
                        "evidence_text": "not in source",
                        "confidence": 0.9,
                    }
                ],
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    model = FailingFrameModel()
    store = DSPGStore()

    store, first_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=True,
        use_drs_semantics=False,
    )
    calls_after_first_ingest = model.calls
    store, second_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=True,
        use_drs_semantics=False,
    )

    assert first_run_id == second_run_id
    assert calls_after_first_ingest == 1
    assert model.calls == calls_after_first_ingest
    assert store.execute("SELECT COUNT(*) FROM frames WHERE source='local_model'").fetchone()[0] == 0
    assert store.counts()["model_attempts"] == 1
    attempt = store.execute(
        "SELECT accepted, materialized, reason FROM model_attempts WHERE task='chunk_frames'"
    ).fetchone()
    assert attempt["accepted"] == 0
    assert attempt["materialized"] == 0
    assert attempt["reason"] == "grounding_validation_failed"


def test_temporal_query_drs_uses_latest_temporal_edge(tmp_path: Path) -> None:
    (tmp_path / "random_blob").write_text(
        "\n".join(
            [
                "2026-01-01 AuroraGate state: open.",
                "Noise terms should not decide the answer.",
                "2026-01-03 AuroraGate state: paused.",
                "2026-01-05 AuroraGate state: closed.",
            ]
        ),
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)

    assert engine.dspg_counts()["temporal_edges"] == 3
    frame = QueryFrame(
        question_text="model-produced temporal query DRS",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("AuroraGate",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
    )
    answer = engine._answer_with_bounded_dspg(
        "model-produced temporal query DRS",
        frame,
        ExpectedAnswer("state"),
    )

    assert answer is not None
    assert answer.text == "closed"
    assert answer.reason == "bounded DSPG query-frame execution"


def test_model_drs_temporal_records_project_latest_literal_values(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "early.txt").write_text("Phase note: Lumen Core marker T001 state amber.", encoding="utf-8")
    (tmp_path / "late.txt").write_text("Phase note: Lumen Core marker T003 state green.", encoding="utf-8")

    class TemporalDrsModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-temporal-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "T001" in prompt:
                marker = "T001"
                state = "amber"
                text = "Phase note: Lumen Core marker T001 state amber."
            else:
                marker = "T003"
                state = "green"
                text = "Phase note: Lumen Core marker T003 state green."
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "temporal.txt",
                    "referents": [
                        {"id": "r0", "label": "Lumen Core", "kind": "entity", "evidence_text": "Lumen Core"},
                    ],
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "state",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "t0",
                            "arguments": [
                                {
                                    "role": "subject",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "entity",
                                    "evidence_text": "Lumen Core",
                                },
                                {
                                    "role": "state",
                                    "target_kind": "literal",
                                    "target_id": "",
                                    "value": state,
                                    "value_type": "state",
                                    "evidence_text": state,
                                },
                            ],
                            "evidence_text": f"Lumen Core marker {marker} state {state}",
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [
                        {"id": "t0", "value": marker, "value_type": "sequence_marker", "evidence_text": marker}
                    ],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    cache_dir = tmp_path.parent / f"{tmp_path.name}-temporal-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=TemporalDrsModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What is the latest state for Lumen Core?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Lumen Core",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
    )

    answer, _diagnostics = execute_bounded_query(store, run_id, documents, sentences_by_document, frame.question_text, frame)

    assert store.counts()["temporal_edges"] == 2
    assert answer is not None
    assert answer.text == "green"
    assert answer.evidence[0].rel_path == "late.txt"


def test_unscoped_model_temporal_values_return_unknown_even_with_same_source_span(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "timeline.txt").write_text(
        "Timeline note: Lumen Core marker T001 state draft and marker T003 state final.",
        encoding="utf-8",
    )

    class SameSpanTemporalDrsModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-same-span-temporal-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            text = "Timeline note: Lumen Core marker T001 state draft and marker T003 state final."
            conditions = []
            for index, (marker, state) in enumerate([("T001", "draft"), ("T003", "final")]):
                conditions.append(
                    {
                        "id": f"c{index}",
                        "predicate": "state",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": f"t{index}",
                        "arguments": [
                            {
                                "role": "subject",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "entity",
                                "evidence_text": "Lumen Core",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": state,
                                "value_type": "state",
                                "evidence_text": state,
                            },
                        ],
                        "evidence_text": f"marker {marker} state {state}",
                    }
                )
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "timeline.txt",
                    "referents": [
                        {"id": "r0", "label": "Lumen Core", "kind": "entity", "evidence_text": "Lumen Core"},
                    ],
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                    ],
                    "conditions": conditions,
                    "identity_hypotheses": [],
                    "temporal_records": [
                        {"id": "t0", "value": "T001", "value_type": "sequence_marker", "evidence_text": "T001"},
                        {"id": "t1", "value": "T003", "value_type": "sequence_marker", "evidence_text": "T003"},
                    ],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    cache_dir = tmp_path.parent / f"{tmp_path.name}-same-span-temporal-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=SameSpanTemporalDrsModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What state is recorded for Lumen Core?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Lumen Core",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
    )

    answer, diagnostics = execute_bounded_query(store, run_id, documents, sentences_by_document, frame.question_text, frame)

    assert store.counts()["temporal_edges"] == 2
    assert answer is None
    assert diagnostics["execution"]["temporal_ambiguity_without_query_scope"] is True
    values = {item["value"] for item in diagnostics["execution"]["candidate_evidence_sample"]}
    assert {"draft", "final"}.issubset(values)


def test_ingest_skips_cartesian_temporal_edges_for_dense_time_chunks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KMD_TEMPORAL_SAME_SPAN_MAX_VALUES", "2")
    (tmp_path / "dense.log").write_text(
        " ".join(
            f"2026-01-{index:02d} item_{index}: ready"
            for index in range(1, 8)
        ),
        encoding="utf-8",
    )

    store, _, _, _ = ingest_folder(tmp_path)

    assert store.counts()["relations"] >= 7
    assert store.counts()["temporal_edges"] == 0


def test_ingest_caps_compatibility_frames_without_dropping_relations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KMD_DETERMINISTIC_FRAMES_MAX_PER_CHUNK", "1")
    (tmp_path / "records.txt").write_text(
        "Alpha owner: Mira; Beta owner: Jonas; Gamma owner: Lina",
        encoding="utf-8",
    )

    store, _, _, _ = ingest_folder(tmp_path)

    assert store.counts()["relations"] >= 3
    assert store.counts()["frames"] == 1


def test_count_aggregation_requires_each_query_drs_term_group(tmp_path: Path) -> None:
    (tmp_path / "states.txt").write_text(
        "\n".join(
            [
                "Alpha unit status: ready.",
                "Beta unit status: ready.",
                "Gamma unit status: blocked.",
            ]
        ),
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="model-produced count query DRS",
        answer_type="count",
        answer_variables=("units",),
        target_anchors=(),
        requested_relation="status",
        relation_terms=("units", "ready"),
        constraints=(),
        aggregation="count",
    )
    answer = engine._answer_with_bounded_dspg(
        "model-produced count query DRS",
        frame,
        ExpectedAnswer("count"),
    )

    assert answer is not None
    assert answer.text == "2"
    assert answer.reason == "bounded DSPG query-frame execution"


def test_model_query_drs_compound_slot_matches_structural_record_field(tmp_path: Path) -> None:
    (tmp_path / "object.raw").write_text(
        '{ name: "Orchid Gamma", owner: "Tessa Noll", '
        'links: { report: "https://reports.example.test/orchid-gamma" } }',
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="What report link is listed for Orchid Gamma?",
        answer_type="url",
        answer_variables=("report_link",),
        target_anchors=("Orchid Gamma",),
        requested_relation="listed",
        relation_terms=("listed", "report_link"),
        constraints=(),
    )

    answer = engine._answer_with_bounded_dspg(
        "What report link is listed for Orchid Gamma?",
        frame,
        ExpectedAnswer("url"),
    )

    assert answer is not None
    assert answer.text == "https://reports.example.test/orchid-gamma"
    assert answer.reason == "bounded DSPG query-frame execution"


def test_identity_expansion_seeds_only_exact_referent_surfaces() -> None:
    records = {
        "referents": [
            {"referent_id": "r0", "canonical_label": "Orchid Gamma"},
            {
                "referent_id": "r1",
                "canonical_label": "https://reports.example.test/orchid-gamma",
            },
        ],
        "identity_hypotheses": [],
    }

    expanded = _identity_expanded_terms(records, ["orchid gamma"])

    assert "orchid gamma" in expanded
    assert "https reports example test orchid gamma" not in expanded


def test_model_query_drs_answer_slot_terms_participate_in_structural_binding(tmp_path: Path) -> None:
    (tmp_path / "object.raw").write_text(
        '{ name: "Olan Vex", ids: { actor: "OV-8801" } }',
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="What actor id is listed for Olan Vex?",
        answer_type="identifier",
        answer_variables=("actor id",),
        target_anchors=("Olan Vex",),
        requested_relation="listed",
        relation_terms=("listed",),
        constraints=(),
    )

    answer = engine._answer_with_bounded_dspg(
        "What actor id is listed for Olan Vex?",
        frame,
        ExpectedAnswer("identifier"),
    )

    assert answer is not None
    assert answer.text == "OV-8801"
    assert answer.reason == "bounded DSPG query-frame execution"
