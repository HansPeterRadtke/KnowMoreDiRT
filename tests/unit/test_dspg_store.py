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
