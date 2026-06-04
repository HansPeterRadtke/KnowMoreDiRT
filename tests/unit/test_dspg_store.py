from __future__ import annotations

import json
import os
import time
from pathlib import Path

from knowmoredirt.answer_types import ExpectedAnswer
from knowmoredirt.bounded_dspg import (
    _answer_conflict_diagnostics,
    _context_accessible,
    _fetch_identity_hypotheses,
    _identity_expanded_terms,
    _load_records,
    _rank_scope,
    _terms_match_material,
    _target_terms,
    _locative_answer_value,
    execute_bounded_query,
)
from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.ingest import ingest_folder
from knowmoredirt.models import Evidence
from knowmoredirt.query import QueryFrame, term_variants
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


def test_context_requirement_matching_uses_morphology_variants() -> None:
    assert _terms_match_material(["report"], "drs:reported observer")
    assert _terms_match_material(["believe"], "drs:believed Kalo Reed")
    assert term_variants("state") == {"state"}


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
    assert "ending/resolution.txt" in {item.rel_path for item in answer.evidence}
    assert "ending/resolution.txt" in {
        item["rel_path"] for item in diagnostics["execution"]["identity_expansion_evidence"]
    }
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
    evidence_paths = {item.rel_path for item in answer.evidence}
    assert {"middle/crosswalk_a.txt", "ending/crosswalk_b.txt"}.issubset(evidence_paths)
    assert {"px-11", "relay-prime"}.issubset(set(diagnostics["ranking"]["identity_expanded_target_terms"]))
    assert diagnostics["ranking"]["identity_expansion_rounds"] >= 2


def test_identity_expansion_reaches_deep_scattered_crosswalk_fixed_point(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "beginning").mkdir()
    (tmp_path / "links").mkdir()
    (tmp_path / "ending").mkdir()
    (tmp_path / "beginning" / "intro.txt").write_text(
        "Opening register introduces Aurora Relay as the target artifact.",
        encoding="utf-8",
    )
    (tmp_path / "links" / "a.txt").write_text(
        "Crosswalk A says AR-1 is the same artifact as Aurora Relay.",
        encoding="utf-8",
    )
    (tmp_path / "links" / "b.txt").write_text(
        "Crosswalk B says Beacon-2 is the same artifact as AR-1.",
        encoding="utf-8",
    )
    (tmp_path / "links" / "c.txt").write_text(
        "Crosswalk C says Cedar-3 is the same artifact as Beacon-2.",
        encoding="utf-8",
    )
    (tmp_path / "links" / "d.txt").write_text(
        "Crosswalk D says Delta-4 is the same artifact as Cedar-3.",
        encoding="utf-8",
    )
    (tmp_path / "ending" / "status.txt").write_text(
        "Final status note says Delta-4 status is stable.",
        encoding="utf-8",
    )

    def identity_drs(left: str, right: str, text: str) -> dict[str, object]:
        evidence = f"{left} is the same artifact as {right}"
        return {
            "schema_version": "chunk-drs-v2",
            "source_id": "deep-crosswalk",
            "referents": [
                {"id": "r0", "label": left, "kind": "identifier", "evidence_text": left},
                {"id": "r1", "label": right, "kind": "artifact", "evidence_text": right},
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
                            "evidence_text": left,
                        },
                        {
                            "role": "right",
                            "target_kind": "referent",
                            "target_id": "r1",
                            "value": "",
                            "value_type": "artifact",
                            "evidence_text": right,
                        },
                    ],
                    "evidence_text": evidence,
                }
            ],
            "identity_hypotheses": [
                {
                    "left_referent_id": "r0",
                    "right_referent_id": "r1",
                    "status": "accepted",
                    "evidence_text": evidence,
                    "confidence": 0.93,
                }
            ],
            "temporal_records": [],
        }

    class DeepCrosswalkModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-deep-crosswalk-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Opening register" in prompt:
                text = "Opening register introduces Aurora Relay as the target artifact."
                drs = {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "deep-crosswalk",
                    "referents": [
                        {"id": "r0", "label": "Aurora Relay", "kind": "artifact", "evidence_text": "Aurora Relay"},
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
                                    "evidence_text": "Aurora Relay",
                                }
                            ],
                            "evidence_text": text,
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                }
            elif "Crosswalk A" in prompt:
                text = "Crosswalk A says AR-1 is the same artifact as Aurora Relay."
                drs = identity_drs("AR-1", "Aurora Relay", text)
            elif "Crosswalk B" in prompt:
                text = "Crosswalk B says Beacon-2 is the same artifact as AR-1."
                drs = identity_drs("Beacon-2", "AR-1", text)
            elif "Crosswalk C" in prompt:
                text = "Crosswalk C says Cedar-3 is the same artifact as Beacon-2."
                drs = identity_drs("Cedar-3", "Beacon-2", text)
            elif "Crosswalk D" in prompt:
                text = "Crosswalk D says Delta-4 is the same artifact as Cedar-3."
                drs = identity_drs("Delta-4", "Cedar-3", text)
            else:
                text = "Final status note says Delta-4 status is stable."
                drs = {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "deep-crosswalk",
                    "referents": [
                        {"id": "r0", "label": "Delta-4", "kind": "identifier", "evidence_text": "Delta-4"},
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
                                    "evidence_text": "Delta-4",
                                },
                                {
                                    "role": "state",
                                    "target_kind": "literal",
                                    "target_id": "",
                                    "value": "stable",
                                    "value_type": "state",
                                    "evidence_text": "stable",
                                },
                            ],
                            "evidence_text": "Delta-4 status is stable",
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                }
            return {"drs": drs, "_model_raw": "{}", "_model_elapsed_seconds": 0.01}

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=DeepCrosswalkModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What status is recorded for Aurora Relay?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Aurora Relay",),
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
    assert answer.text == "stable"
    assert answer.evidence[0].rel_path == "ending/status.txt"
    assert "delta-4" in diagnostics["ranking"]["identity_expanded_target_terms"]
    assert diagnostics["ranking"]["identity_expansion_rounds"] >= 4
    assert {
        "links/a.txt",
        "links/b.txt",
        "links/c.txt",
        "links/d.txt",
    }.issubset({item["rel_path"] for item in diagnostics["execution"]["identity_expansion_evidence"]})


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
    assert "ending/audit.txt" in {item.rel_path for item in asserted_answer.evidence}
    assert "ending/audit.txt" in {item.rel_path for item in reported_answer.evidence}
    assert "cb-44" in asserted_diagnostics["ranking"]["identity_expanded_target_terms"]
    assert "cb-44" in reported_diagnostics["ranking"]["identity_expanded_target_terms"]


def test_scattered_reported_identity_does_not_leak_into_unscoped_merge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "begin").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "ending").mkdir()
    (tmp_path / "begin" / "registry.txt").write_text(
        "Registry introduces North Lantern as the sealed device.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "status.txt").write_text(
        "Middle status board says NL-7 status is armed.",
        encoding="utf-8",
    )
    (tmp_path / "ending" / "reported_crosswalk.txt").write_text(
        "Iris Vale reports that NL-7 is North Lantern and NL-7 status is blue.",
        encoding="utf-8",
    )

    class ReportedIdentityModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-reported-identity-scope", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Registry introduces" in prompt:
                text = "Registry introduces North Lantern as the sealed device."
                referents = [
                    {"id": "r0", "label": "North Lantern", "kind": "artifact", "evidence_text": "North Lantern"}
                ]
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
                                "evidence_text": "North Lantern",
                            }
                        ],
                        "evidence_text": text,
                    }
                ]
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                ]
                identities = []
                temporals = []
                source_id = "begin/registry.txt"
            elif "Middle status board" in prompt:
                text = "Middle status board says NL-7 status is armed."
                referents = [{"id": "r0", "label": "NL-7", "kind": "identifier", "evidence_text": "NL-7"}]
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
                                "evidence_text": "NL-7",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "armed",
                                "value_type": "state",
                                "evidence_text": "armed",
                            },
                        ],
                        "evidence_text": "NL-7 status is armed",
                    }
                ]
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                ]
                identities = []
                temporals = []
                source_id = "middle/status.txt"
            else:
                text = "Iris Vale reports that NL-7 is North Lantern and NL-7 status is blue."
                reported = "NL-7 is North Lantern and NL-7 status is blue"
                referents = [
                    {"id": "r0", "label": "Iris Vale", "kind": "person", "evidence_text": "Iris Vale"},
                    {"id": "r1", "label": "NL-7", "kind": "identifier", "evidence_text": "NL-7"},
                    {"id": "r2", "label": "North Lantern", "kind": "artifact", "evidence_text": "North Lantern"},
                ]
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text},
                    {
                        "id": "b1",
                        "kind": "reported",
                        "parent_id": "b0",
                        "holder_referent_id": "r0",
                        "evidence_text": reported,
                    },
                ]
                conditions = [
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
                                "evidence_text": "Iris Vale",
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
                        "predicate": "same_artifact",
                        "box_id": "b1",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "left",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "NL-7",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r2",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "North Lantern",
                            },
                        ],
                        "evidence_text": "NL-7 is North Lantern",
                    },
                    {
                        "id": "c2",
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
                                "evidence_text": "NL-7",
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
                        "evidence_text": "NL-7 status is blue",
                    },
                ]
                identities = [
                    {
                        "left_referent_id": "r1",
                        "right_referent_id": "r2",
                        "box_id": "b1",
                        "status": "accepted",
                        "evidence_text": "NL-7 is North Lantern",
                        "confidence": 0.93,
                    }
                ]
                temporals = []
                source_id = "ending/reported_crosswalk.txt"
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": source_id,
                    "referents": referents,
                    "boxes": boxes,
                    "conditions": conditions,
                    "identity_hypotheses": identities,
                    "temporal_records": temporals,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=ReportedIdentityModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    unscoped_frame = QueryFrame(
        question_text="What status is recorded for North Lantern?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("North Lantern",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
    )
    reported_frame = QueryFrame(
        question_text="What status did Iris Vale report for North Lantern?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("North Lantern",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
        scope_requirements=("reported",),
    )

    unscoped_answer, unscoped_diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        unscoped_frame.question_text,
        unscoped_frame,
    )
    reported_answer, reported_diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        reported_frame.question_text,
        reported_frame,
    )

    assert unscoped_answer is None
    assert "nl-7" not in unscoped_diagnostics["ranking"].get("identity_expanded_target_terms", [])
    assert reported_answer is not None
    assert reported_answer.text == "blue"
    assert reported_answer.evidence[0].rel_path == "ending/reported_crosswalk.txt"
    assert "ending/reported_crosswalk.txt" in {
        item["rel_path"] for item in reported_diagnostics["execution"]["identity_expansion_evidence"]
    }
    assert "nl-7" in reported_diagnostics["ranking"]["identity_expanded_target_terms"]


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
    assert diagnostics["execution"]["no_answer_reason"] == "answer_conflict_without_query_scope"
    assert {
        item["evidence"]["rel_path"]
        for item in diagnostics["execution"]["candidate_evidence_sample"]
    }.issuperset({"middle/status.txt", "ending/correction.txt"})
    for item in diagnostics["execution"]["candidate_evidence_sample"]:
        evidence = item["evidence"]
        assert evidence.get("chunk_id")
        assert evidence.get("span_id")
        assert evidence.get("document", {}).get("document_id") == evidence.get("document_id")
    assert {
        item["rel_path"]
        for item in diagnostics["execution"]["source_provenance_sample"]
    }.issuperset({"middle/status.txt", "ending/correction.txt"})


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
    assert crosswalk["document"]["document_id"] == crosswalk["document_id"]
    assert crosswalk["document"]["file_name"] == "crosswalk.txt"
    assert crosswalk["document"]["semantic_quality"]
    assert "LC-71 status is silver" in crosswalk["text"]


def test_incremental_new_identity_bridge_reaches_existing_drs_chunk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "middle").mkdir()
    (tmp_path / "middle" / "state.txt").write_text(
        "Middle state says RG-4 status is amber.",
        encoding="utf-8",
    )

    class IncrementalBridgeModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-incremental-bridge-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Middle state" in prompt:
                text = "Middle state says RG-4 status is amber."
                source_id = "middle/state.txt"
                referents = [{"id": "r0", "label": "RG-4", "kind": "identifier", "evidence_text": "RG-4"}]
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
                                "role": "entity",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "RG-4",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "amber",
                                "value_type": "state",
                                "evidence_text": "amber",
                            },
                        ],
                        "evidence_text": "RG-4 status is amber",
                    }
                ]
                identities = []
            elif "Opening catalog" in prompt:
                text = "Opening catalog introduces Raven Gate."
                source_id = "begin/catalog.txt"
                referents = [{"id": "r0", "label": "Raven Gate", "kind": "artifact", "evidence_text": "Raven Gate"}]
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
                                "role": "entity",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Raven Gate",
                            }
                        ],
                        "evidence_text": text,
                    }
                ]
                identities = []
            else:
                text = "Ending crosswalk says RG-4 is the same artifact as Raven Gate."
                source_id = "end/crosswalk.txt"
                referents = [
                    {"id": "r0", "label": "RG-4", "kind": "identifier", "evidence_text": "RG-4"},
                    {"id": "r1", "label": "Raven Gate", "kind": "artifact", "evidence_text": "Raven Gate"},
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
                                "evidence_text": "RG-4",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Raven Gate",
                            },
                        ],
                        "evidence_text": "RG-4 is the same artifact as Raven Gate",
                    }
                ]
                identities = [
                    {
                        "left_referent_id": "r0",
                        "right_referent_id": "r1",
                        "status": "accepted",
                        "evidence_text": "RG-4 is the same artifact as Raven Gate",
                        "confidence": 0.93,
                    }
                ]
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

    cache_dir = tmp_path.parent / f"{tmp_path.name}-incremental-bridge-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    store = DSPGStore()
    model = IncrementalBridgeModel()
    store, first_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    (tmp_path / "begin").mkdir()
    (tmp_path / "begin" / "catalog.txt").write_text(
        "Opening catalog introduces Raven Gate.",
        encoding="utf-8",
    )
    (tmp_path / "end").mkdir()
    (tmp_path / "end" / "crosswalk.txt").write_text(
        "Ending crosswalk says RG-4 is the same artifact as Raven Gate.",
        encoding="utf-8",
    )
    store, second_run_id, documents, sentences = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What status is recorded for Raven Gate?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Raven Gate",),
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

    assert first_run_id == second_run_id
    assert answer is not None
    assert answer.text == "amber"
    assert "rg-4" in diagnostics["ranking"]["identity_expanded_target_terms"]
    assert {"end/crosswalk.txt", "middle/state.txt"}.issubset(
        {item.rel_path for item in answer.evidence}
    )


def test_incremental_reported_then_asserted_identity_preserves_scope_and_reuses_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "begin").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "begin" / "catalog.txt").write_text(
        "Opening registry names Cerulean Anchor.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "state.txt").write_text(
        "Asserted record CA-9 marker T002 state amber.",
        encoding="utf-8",
    )

    class IncrementalScopedIdentityModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-incremental-scoped-identity-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            def referent(ref_id: str, label: str, kind: str) -> dict[str, object]:
                return {"id": ref_id, "label": label, "kind": kind, "evidence_text": label}

            def ref_arg(role: str, target_id: str, value_type: str, evidence_text: str) -> dict[str, object]:
                return {
                    "role": role,
                    "target_kind": "referent",
                    "target_id": target_id,
                    "value": "",
                    "value_type": value_type,
                    "evidence_text": evidence_text,
                }

            def literal_arg(role: str, value: str, value_type: str) -> dict[str, object]:
                return {
                    "role": role,
                    "target_kind": "literal",
                    "target_id": "",
                    "value": value,
                    "value_type": value_type,
                    "evidence_text": value,
                }

            if "Opening registry" in prompt:
                self.calls.append("begin/catalog.txt")
                text = "Opening registry names Cerulean Anchor."
                referents = [referent("r0", "Cerulean Anchor", "artifact")]
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "names",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [ref_arg("entity", "r0", "artifact", "Cerulean Anchor")],
                        "evidence_text": text,
                    }
                ]
                identities = []
                temporals = []
            elif "Asserted record" in prompt:
                self.calls.append("middle/state.txt")
                text = "Asserted record CA-9 marker T002 state amber."
                referents = [referent("r0", "CA-9", "identifier")]
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "state",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "t0",
                        "arguments": [
                            ref_arg("entity", "r0", "identifier", "CA-9"),
                            literal_arg("state", "amber", "state"),
                        ],
                        "evidence_text": "CA-9 marker T002 state amber",
                    }
                ]
                identities = []
                temporals = [{"id": "t0", "value": "T002", "value_type": "sequence_marker", "evidence_text": "T002"}]
            elif "Reporter says" in prompt:
                self.calls.append("reports/reported.txt")
                text = "Reporter says CA-9 is Cerulean Anchor and CA-9 marker T003 state green."
                reported = "CA-9 is Cerulean Anchor and CA-9 marker T003 state green"
                referents = [
                    referent("r0", "Reporter", "person"),
                    referent("r1", "CA-9", "identifier"),
                    referent("r2", "Cerulean Anchor", "artifact"),
                ]
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text},
                    {
                        "id": "b1",
                        "kind": "reported",
                        "parent_id": "b0",
                        "holder_referent_id": "r0",
                        "evidence_text": reported,
                    },
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "report",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            ref_arg("source", "r0", "person", "Reporter"),
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
                        "predicate": "same_artifact",
                        "box_id": "b1",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            ref_arg("left", "r1", "identifier", "CA-9"),
                            ref_arg("right", "r2", "artifact", "Cerulean Anchor"),
                        ],
                        "evidence_text": "CA-9 is Cerulean Anchor",
                    },
                    {
                        "id": "c2",
                        "predicate": "state",
                        "box_id": "b1",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "t0",
                        "arguments": [
                            ref_arg("entity", "r1", "identifier", "CA-9"),
                            literal_arg("state", "green", "state"),
                        ],
                        "evidence_text": "CA-9 marker T003 state green",
                    },
                ]
                identities = [
                    {
                        "left_referent_id": "r1",
                        "right_referent_id": "r2",
                        "status": "accepted",
                        "evidence_text": "CA-9 is Cerulean Anchor",
                        "confidence": 0.91,
                    }
                ]
                temporals = [{"id": "t0", "value": "T003", "value_type": "sequence_marker", "evidence_text": "T003"}]
            else:
                self.calls.append("end/asserted_identity.txt")
                text = "Final registry confirms CA-9 is Cerulean Anchor."
                referents = [
                    referent("r0", "CA-9", "identifier"),
                    referent("r1", "Cerulean Anchor", "artifact"),
                ]
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
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
                            ref_arg("left", "r0", "identifier", "CA-9"),
                            ref_arg("right", "r1", "artifact", "Cerulean Anchor"),
                        ],
                        "evidence_text": "CA-9 is Cerulean Anchor",
                    }
                ]
                identities = [
                    {
                        "left_referent_id": "r0",
                        "right_referent_id": "r1",
                        "status": "accepted",
                        "evidence_text": "CA-9 is Cerulean Anchor",
                        "confidence": 0.94,
                    }
                ]
                temporals = []
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "incremental-scoped-identity",
                    "referents": referents,
                    "boxes": boxes,
                    "conditions": conditions,
                    "identity_hypotheses": identities,
                    "temporal_records": temporals,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    def sentences_by_document(sentences) -> dict[str, dict[int, object]]:
        grouped: dict[str, dict[int, object]] = {}
        for sentence in sentences:
            grouped.setdefault(sentence.rel_path, {})[sentence.order] = sentence
        return grouped

    unscoped_frame = QueryFrame(
        question_text="What is the latest state for Cerulean Anchor?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Cerulean Anchor",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
    )
    reported_frame = QueryFrame(
        question_text="What is the latest reported state for Cerulean Anchor?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Cerulean Anchor",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
        scope_requirements=("reported",),
    )

    cache_dir = tmp_path.parent / f"{tmp_path.name}-incremental-scoped-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    store = DSPGStore()
    model = IncrementalScopedIdentityModel()
    store, first_run_id, documents, sentences = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    initial_answer, initial_diagnostics = execute_bounded_query(
        store,
        first_run_id,
        documents,
        sentences_by_document(sentences),  # type: ignore[arg-type]
        unscoped_frame.question_text,
        unscoped_frame,
    )
    assert initial_answer is None
    assert initial_diagnostics["execution"]["no_answer_reason"] == "no_candidate"
    assert model.calls == ["begin/catalog.txt", "middle/state.txt"]

    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "reported.txt").write_text(
        "Reporter says CA-9 is Cerulean Anchor and CA-9 marker T003 state green.",
        encoding="utf-8",
    )
    store, second_run_id, documents, sentences = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    grouped = sentences_by_document(sentences)
    unscoped_after_report, unscoped_reported_diagnostics = execute_bounded_query(
        store,
        second_run_id,
        documents,
        grouped,  # type: ignore[arg-type]
        unscoped_frame.question_text,
        unscoped_frame,
    )
    reported_answer, reported_diagnostics = execute_bounded_query(
        store,
        second_run_id,
        documents,
        grouped,  # type: ignore[arg-type]
        reported_frame.question_text,
        reported_frame,
    )
    assert first_run_id == second_run_id
    assert model.calls == ["begin/catalog.txt", "middle/state.txt", "reports/reported.txt"]
    assert unscoped_after_report is None
    assert "ca-9" not in unscoped_reported_diagnostics["ranking"].get("identity_expanded_target_terms", [])
    assert reported_answer is not None
    assert reported_answer.text == "green"
    assert "ca-9" in reported_diagnostics["ranking"]["identity_expanded_target_terms"]
    reported_identity_source = next(
        item
        for item in reported_diagnostics["execution"]["identity_expansion_evidence"]
        if item["rel_path"] == "reports/reported.txt"
    )
    assert reported_identity_source["document"]["document_id"] == reported_identity_source["document_id"]
    assert reported_identity_source["chunk_id"]
    assert reported_answer.evidence[0].rel_path == "reports/reported.txt"

    (tmp_path / "end").mkdir()
    (tmp_path / "end" / "asserted_identity.txt").write_text(
        "Final registry confirms CA-9 is Cerulean Anchor.",
        encoding="utf-8",
    )
    store, third_run_id, documents, sentences = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    final_answer, final_diagnostics = execute_bounded_query(
        store,
        third_run_id,
        documents,
        sentences_by_document(sentences),  # type: ignore[arg-type]
        unscoped_frame.question_text,
        unscoped_frame,
    )

    assert third_run_id == first_run_id
    assert model.calls == [
        "begin/catalog.txt",
        "middle/state.txt",
        "reports/reported.txt",
        "end/asserted_identity.txt",
    ]
    assert final_answer is not None
    assert final_answer.text == "amber"
    assert "ca-9" in final_diagnostics["ranking"]["identity_expanded_target_terms"]
    final_identity_source = next(
        item
        for item in final_diagnostics["execution"]["identity_expansion_evidence"]
        if item["rel_path"] == "end/asserted_identity.txt"
    )
    assert final_identity_source["document"]["document_id"] == final_identity_source["document_id"]
    assert final_identity_source["chunk_id"]
    assert {"end/asserted_identity.txt", "middle/state.txt"}.issubset(
        {item.rel_path for item in final_answer.evidence}
    )
    assert "reports/reported.txt" not in {item.rel_path for item in final_answer.evidence}


def test_reported_identity_without_box_id_does_not_expand_as_asserted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "begin").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "end").mkdir()
    (tmp_path / "begin" / "intro.txt").write_text(
        "Registry introduces Nova Case as the inspected artifact.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "state.txt").write_text(
        "Maintenance code NC-1 status is silver.",
        encoding="utf-8",
    )
    (tmp_path / "end" / "reported_crosswalk.txt").write_text(
        "Mira report says NC-1 is the same artifact as Nova Case.",
        encoding="utf-8",
    )

    class UngroundedReportedIdentityModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-ungrounded-reported-identity-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Registry introduces" in prompt:
                text = "Registry introduces Nova Case as the inspected artifact."
                referents = [{"id": "r0", "label": "Nova Case", "kind": "artifact", "evidence_text": "Nova Case"}]
                boxes = [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}]
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
                                "evidence_text": "Nova Case",
                            }
                        ],
                        "evidence_text": text,
                    }
                ]
                identities = []
            elif "Maintenance code" in prompt:
                text = "Maintenance code NC-1 status is silver."
                referents = [{"id": "r0", "label": "NC-1", "kind": "identifier", "evidence_text": "NC-1"}]
                boxes = [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}]
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
                                "evidence_text": "NC-1",
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
                        "evidence_text": "NC-1 status is silver",
                    }
                ]
                identities = []
            else:
                text = "Mira report says NC-1 is the same artifact as Nova Case."
                reported = "NC-1 is the same artifact as Nova Case"
                referents = [
                    {"id": "r0", "label": "Mira report", "kind": "document", "evidence_text": "Mira report"},
                    {"id": "r1", "label": "NC-1", "kind": "identifier", "evidence_text": "NC-1"},
                    {"id": "r2", "label": "Nova Case", "kind": "artifact", "evidence_text": "Nova Case"},
                ]
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text},
                    {"id": "b1", "kind": "reported", "parent_id": "b0", "holder_referent_id": "r0", "evidence_text": reported},
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "report",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "source",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "document",
                                "evidence_text": "Mira report",
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
                        "predicate": "same_artifact",
                        "box_id": "b1",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "left",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "NC-1",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r2",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Nova Case",
                            },
                        ],
                        "evidence_text": reported,
                    },
                ]
                identities = [
                    {
                        "left_referent_id": "r1",
                        "right_referent_id": "r2",
                        "status": "accepted",
                        "evidence_text": text,
                        "confidence": 0.91,
                    }
                ]
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "ungrounded-reported-identity",
                    "referents": referents,
                    "boxes": boxes,
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
        semantic_client=UngroundedReportedIdentityModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    grouped: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        grouped.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What status is recorded for Nova Case?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Nova Case",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
    )

    answer, diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        grouped,  # type: ignore[arg-type]
        frame.question_text,
        frame,
    )

    assert answer is None
    assert "nc-1" not in diagnostics["ranking"].get("identity_expanded_target_terms", [])
    assert store.execute(
        "SELECT COUNT(*) FROM drs_identity_hypotheses WHERE source_span_id IS NOT NULL"
    ).fetchone()[0] == 1
    assert store.execute(
        "SELECT COUNT(*) FROM identity_hypotheses WHERE source='local_model_drs'"
    ).fetchone()[0] == 0
    blocked_identity = diagnostics["execution"]["blocked_identity_source_provenance"]
    assert blocked_identity[0]["rel_path"] == "end/reported_crosswalk.txt"
    assert blocked_identity[0]["expansion_blocked_reason"] == "missing_grounded_box"
    assert (
        blocked_identity[0]["identity_evidence"]
        == "Mira report says NC-1 is the same artifact as Nova Case."
    )
    provenance_paths = {
        item["rel_path"] for item in diagnostics["execution"].get("source_provenance_sample", [])
    }
    assert {"begin/intro.txt", "middle/state.txt", "end/reported_crosswalk.txt"}.issubset(provenance_paths)


def test_extra_identity_spans_carry_blocked_drs_identity_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "registry.txt").write_text(
        "\n".join(
            [
                "Boreal Node registry entry exists.",
                "Filler alpha marker.",
                "Filler beta marker.",
                "Filler gamma marker.",
                "Filler delta marker.",
                "Filler epsilon marker.",
                (
                    "Bridge note says BN-2 is the same artifact as Boreal Node while "
                    "the observer reported BN-2 matches Boreal Node only in a drill."
                ),
            ]
        ),
        encoding="utf-8",
    )

    class ExtraSpanIdentityModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-extra-span-blocked-identity", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Bridge note says" in prompt:
                text = (
                    "Bridge note says BN-2 is the same artifact as Boreal Node while "
                    "the observer reported BN-2 matches Boreal Node only in a drill."
                )
                root_evidence = "BN-2 is the same artifact as Boreal Node"
                reported_evidence = "BN-2 matches Boreal Node"
                return {
                    "drs": {
                        "schema_version": "chunk-drs-v2",
                        "source_id": "registry.txt",
                        "referents": [
                            {"id": "r0", "label": "BN-2", "kind": "identifier", "evidence_text": "BN-2"},
                            {
                                "id": "r1",
                                "label": "Boreal Node",
                                "kind": "artifact",
                                "evidence_text": "Boreal Node",
                            },
                            {"id": "r2", "label": "observer", "kind": "actor", "evidence_text": "observer"},
                        ],
                        "boxes": [
                            {
                                "id": "b0",
                                "kind": "asserted",
                                "parent_id": "",
                                "holder_referent_id": "",
                                "evidence_text": text,
                            },
                            {
                                "id": "b1",
                                "kind": "reported",
                                "parent_id": "b0",
                                "holder_referent_id": "r2",
                                "evidence_text": "observer reported BN-2 matches Boreal Node only in a drill",
                            },
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
                                        "evidence_text": "BN-2",
                                    },
                                    {
                                        "role": "right",
                                        "target_kind": "referent",
                                        "target_id": "r1",
                                        "value": "",
                                        "value_type": "artifact",
                                        "evidence_text": "Boreal Node",
                                    },
                                ],
                                "evidence_text": root_evidence,
                            },
                        ],
                        "identity_hypotheses": [
                            {
                                "left_referent_id": "r0",
                                "right_referent_id": "r1",
                                "status": "accepted",
                                "box_id": "b0",
                                "evidence_text": root_evidence,
                                "confidence": 0.91,
                            },
                            {
                                "left_referent_id": "r0",
                                "right_referent_id": "r1",
                                "status": "accepted",
                                "evidence_text": reported_evidence,
                                "confidence": 0.88,
                            },
                        ],
                        "temporal_records": [],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }

            text = "Boreal Node registry entry exists." if "Boreal Node" in prompt else "Filler alpha marker."
            label = "Boreal Node" if "Boreal Node" in text else text.rstrip(".")
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "registry.txt",
                    "referents": [
                        {"id": "r0", "label": label, "kind": "source_text", "evidence_text": label},
                    ],
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                    ],
                    "conditions": [
                        {
                            "id": "c0",
                            "predicate": "mentions",
                            "box_id": "b0",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "",
                            "arguments": [
                                {
                                    "role": "content",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "source_text",
                                    "evidence_text": label,
                                },
                            ],
                            "evidence_text": text,
                        },
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    cache_dir = tmp_path.parent / f"{tmp_path.name}-extra-span-blocked-identity-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=ExtraSpanIdentityModel(),
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What status is recorded for Boreal Node?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Boreal Node",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
    )
    selected_docs, selected_chunks, _ranking = _rank_scope(
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        frame.question_text,
        frame,
        40,
        1,
    )
    records = _load_records(store, run_id, selected_docs, selected_chunks)

    bridge_spans = [
        span for span in records["source_spans"] if "Bridge note says" in str(span.get("surface") or "")
    ]
    assert bridge_spans
    assert all(span["chunk_id"] not in set(selected_chunks) for span in bridge_spans)
    assert any("same artifact" in str(row.get("evidence") or "") for row in records["identity_hypotheses"])

    blocked_rows = [
        row
        for row in records["drs_identity_hypotheses"]
        if "BN-2 matches Boreal Node" in str(row.get("evidence_surface") or "")
    ]
    assert blocked_rows
    blocked_metadata = json.loads(blocked_rows[0]["metadata_json"])
    assert blocked_metadata["expansion_blocked_reason"] == "missing_grounded_box"


def test_scattered_identity_reported_contradiction_respects_drs_scope(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "begin").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "late").mkdir()
    (tmp_path / "end").mkdir()
    (tmp_path / "begin" / "intro.txt").write_text(
        "Opening ledger introduces Thistle Node as the field object.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "reported.txt").write_text(
        "Mira Sol reports that TN-8 status is orange.",
        encoding="utf-8",
    )
    (tmp_path / "late" / "audit.txt").write_text(
        "Final audit asserts TN-8 status is blue.",
        encoding="utf-8",
    )
    (tmp_path / "end" / "identity.txt").write_text(
        "Ending bridge states TN-8 is the same artifact as Thistle Node.",
        encoding="utf-8",
    )

    class ScatteredReportedContradictionModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-scattered-reported-contradiction-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Opening ledger" in prompt:
                text = "Opening ledger introduces Thistle Node as the field object."
                source_id = "begin/intro.txt"
                referents = [
                    {"id": "r0", "label": "Thistle Node", "kind": "artifact", "evidence_text": "Thistle Node"}
                ]
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                ]
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
                                "role": "entity",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Thistle Node",
                            }
                        ],
                        "evidence_text": text,
                    }
                ]
                identities = []
            elif "Mira Sol reports" in prompt:
                text = "Mira Sol reports that TN-8 status is orange."
                source_id = "middle/reported.txt"
                referents = [
                    {"id": "r0", "label": "Mira Sol", "kind": "person", "evidence_text": "Mira Sol"},
                    {"id": "r1", "label": "TN-8", "kind": "identifier", "evidence_text": "TN-8"},
                ]
                boxes = [
                    {
                        "id": "b1",
                        "kind": "reported",
                        "parent_id": "b0",
                        "holder_referent_id": "r0",
                        "evidence_text": "TN-8 status is orange",
                    },
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text},
                ]
                conditions = [
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
                                "evidence_text": "TN-8",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "orange",
                                "value_type": "state",
                                "evidence_text": "orange",
                            },
                        ],
                        "evidence_text": "TN-8 status is orange",
                    }
                ]
                identities = []
            elif "Final audit" in prompt:
                text = "Final audit asserts TN-8 status is blue."
                source_id = "late/audit.txt"
                referents = [{"id": "r0", "label": "TN-8", "kind": "identifier", "evidence_text": "TN-8"}]
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
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
                                "target_id": "r0",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "TN-8",
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
                        "evidence_text": "TN-8 status is blue",
                    }
                ]
                identities = []
            else:
                text = "Ending bridge states TN-8 is the same artifact as Thistle Node."
                source_id = "end/identity.txt"
                referents = [
                    {"id": "r0", "label": "TN-8", "kind": "identifier", "evidence_text": "TN-8"},
                    {"id": "r1", "label": "Thistle Node", "kind": "artifact", "evidence_text": "Thistle Node"},
                ]
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
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
                                "evidence_text": "TN-8",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Thistle Node",
                            },
                        ],
                        "evidence_text": "TN-8 is the same artifact as Thistle Node",
                    }
                ]
                identities = [
                    {
                        "left_referent_id": "r0",
                        "right_referent_id": "r1",
                        "status": "accepted",
                        "evidence_text": "TN-8 is the same artifact as Thistle Node",
                        "confidence": 0.92,
                    }
                ]
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": source_id,
                    "referents": referents,
                    "boxes": boxes,
                    "conditions": conditions,
                    "identity_hypotheses": identities,
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    cache_dir = tmp_path.parent / f"{tmp_path.name}-reported-contradiction-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=ScatteredReportedContradictionModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    unscoped_frame = QueryFrame(
        question_text="What status is recorded for Thistle Node?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Thistle Node",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
    )
    reported_frame = QueryFrame(
        question_text="What status did Mira Sol report for Thistle Node?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Thistle Node",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
        scope_requirements=("reported",),
    )

    unscoped_answer, unscoped_diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        unscoped_frame.question_text,
        unscoped_frame,
    )
    reported_answer, reported_diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        reported_frame.question_text,
        reported_frame,
    )

    assert unscoped_answer is not None
    assert unscoped_answer.text == "blue"
    assert "answer_conflict_without_query_scope" not in unscoped_diagnostics["execution"]
    assert "late/audit.txt" in {item.rel_path for item in unscoped_answer.evidence}
    assert reported_answer is not None
    assert reported_answer.text == "orange"
    assert {"end/identity.txt", "middle/reported.txt"}.issubset(
        {item.rel_path for item in reported_answer.evidence}
    )
    assert "tn-8" in reported_diagnostics["ranking"]["identity_expanded_target_terms"]
    reported_provenance = reported_diagnostics["execution"]["answer_source_provenance"]
    assert {"end/identity.txt", "middle/reported.txt"}.issubset(
        {item["rel_path"] for item in reported_provenance}
    )
    reported_source = next(item for item in reported_provenance if item["rel_path"] == "middle/reported.txt")
    assert reported_source["document"]["document_id"] == reported_source["document_id"]
    assert reported_source["document"]["file_name"] == "reported.txt"
    assert reported_source["document"]["semantic_quality"]
    assert reported_source["span_id"]
    assert reported_source["chunk_id"]
    assert "TN-8 status is orange" in reported_source["text"]


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


def test_store_preserves_out_of_order_drs_box_parent_links() -> None:
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
                {
                    "id": "b1",
                    "kind": "reported",
                    "parent_id": "b0",
                    "holder_referent_id": "r0",
                    "evidence_text": "CB-44 status is green",
                },
                {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text},
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

    assert result["accepted"] is True
    row = store.execute(
        """
        SELECT child.parent_external_box_id, child.parent_drs_box_id, parent.external_box_id AS stored_parent,
               child_ctx.parent_context_id AS child_parent_context_id,
               parent.context_id AS stored_parent_context_id
        FROM drs_boxes child
        JOIN drs_boxes parent ON parent.drs_box_id = child.parent_drs_box_id
        JOIN contexts child_ctx ON child_ctx.context_id = child.context_id
        WHERE child.external_box_id='b1'
        """
    ).fetchone()
    assert row is not None
    assert row["parent_external_box_id"] == "b0"
    assert row["stored_parent"] == "b0"
    assert row["child_parent_context_id"] is not None
    assert row["child_parent_context_id"] == row["stored_parent_context_id"]


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


def test_incremental_reingest_updates_current_document_metadata(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("Stable source text for metadata refresh.", encoding="utf-8")
    store = DSPGStore()

    store, first_run_id, first_documents, _ = ingest_folder(tmp_path, store=store)
    document_id = first_documents[0].document_id
    refreshed_mtime = float(first_documents[0].mtime) + 20.0
    os.utime(path, (refreshed_mtime, refreshed_mtime))
    store, second_run_id, second_documents, _ = ingest_folder(tmp_path, store=store)

    assert first_run_id == second_run_id
    assert second_documents[0].document_id == document_id
    stored_document = store.execute(
        "SELECT mtime, metadata_json FROM documents WHERE document_id=?",
        (document_id,),
    ).fetchone()
    assert stored_document is not None
    assert float(stored_document["mtime"]) == float(second_documents[0].mtime)
    metadata = store.execute(
        "SELECT value FROM metadata_records WHERE document_id=? AND key='mtime'",
        (document_id,),
    ).fetchall()
    assert [row["value"] for row in metadata] == [str(second_documents[0].metadata["mtime"])]
    modified_time = store.execute(
        """
        SELECT temporal_value
        FROM context_carriers
        WHERE document_id=? AND temporal_value_type='file_modified_time'
        """,
        (document_id,),
    ).fetchone()
    assert modified_time is not None
    assert modified_time["temporal_value"] == time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(float(second_documents[0].mtime)),
    )


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
    assert {item["rel_path"] for item in provenance} == {"catalog.txt", "status.txt"}
    scattered = diagnostics["execution"]["scattered_source_provenance_without_binding"]
    assert scattered["target_rel_paths"] == ["catalog.txt"]
    assert scattered["relation_rel_paths"] == ["status.txt"]


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


def test_incremental_drs_ingest_reprocesses_when_model_fingerprint_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "note.txt").write_text("Aero Gate is ready.\n", encoding="utf-8")

    class VersionedDrsModel:
        def __init__(self) -> None:
            self.calls = 0
            self.version = "v1"

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": f"fake-versioned-drs-{self.version}", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            predicate = f"ready_{self.version}"
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
                            "predicate": predicate,
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

    cache_dir = tmp_path.parent / f"{tmp_path.name}-versioned-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    model = VersionedDrsModel()
    store = DSPGStore()

    store, first_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    store, second_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    calls_after_same_fingerprint = model.calls
    model.version = "v2"
    store, third_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )

    assert first_run_id == second_run_id == third_run_id
    assert calls_after_same_fingerprint == 1
    assert model.calls == 2
    predicates = {
        row["predicate"]
        for row in store.execute("SELECT predicate FROM drs_conditions WHERE source='local_model_drs'").fetchall()
    }
    assert predicates == {"ready_v2"}
    assert store.counts()["model_attempts"] == 2


def test_incremental_drs_reprocess_replaces_stale_scattered_semantics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "begin").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "end").mkdir()
    (tmp_path / "begin" / "registry.txt").write_text(
        "Registry introduces Helio Marker as the monitored artifact.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "state.txt").write_text(
        "State bulletin for HM-7 lists state green and archive state red.",
        encoding="utf-8",
    )
    (tmp_path / "end" / "crosswalk.txt").write_text(
        "Crosswalk states HM-7 is the same artifact as Helio Marker.",
        encoding="utf-8",
    )

    class VersionedScatteredDrsModel:
        def __init__(self) -> None:
            self.calls = 0
            self.version = "v1"

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": f"fake-scattered-reprocess-{self.version}", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            if "Registry introduces Helio Marker" in prompt:
                text = "Registry introduces Helio Marker as the monitored artifact."
                referents = [
                    {"id": "r0", "label": "Helio Marker", "kind": "artifact", "evidence_text": "Helio Marker"},
                ]
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
                                "evidence_text": "Helio Marker",
                            }
                        ],
                        "evidence_text": text,
                    }
                ]
                identities = []
            elif "State bulletin" in prompt:
                text = "State bulletin for HM-7 lists state green and archive state red."
                state = "red" if self.version == "v1" else "green"
                referents = [
                    {"id": "r0", "label": "HM-7", "kind": "identifier", "evidence_text": "HM-7"},
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "state",
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
                                "evidence_text": "HM-7",
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
                        "evidence_text": text,
                    }
                ]
                identities = []
            else:
                text = "Crosswalk states HM-7 is the same artifact as Helio Marker."
                referents = [
                    {"id": "r0", "label": "HM-7", "kind": "identifier", "evidence_text": "HM-7"},
                    {"id": "r1", "label": "Helio Marker", "kind": "artifact", "evidence_text": "Helio Marker"},
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
                                "evidence_text": "HM-7",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Helio Marker",
                            },
                        ],
                        "evidence_text": "HM-7 is the same artifact as Helio Marker",
                    }
                ]
                identities = [
                    {
                        "left_referent_id": "r0",
                        "right_referent_id": "r1",
                        "status": "accepted",
                        "evidence_text": "HM-7 is the same artifact as Helio Marker",
                        "confidence": 0.94,
                    }
                ]
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "versioned-scattered",
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

    cache_dir = tmp_path.parent / f"{tmp_path.name}-versioned-scattered-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    store = DSPGStore()
    model = VersionedScatteredDrsModel()

    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What state is recorded for Helio Marker?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Helio Marker",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
    )
    first_answer, _first_diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        frame.question_text,
        frame,
    )

    model.version = "v2"
    store, second_run_id, second_documents, second_sentences = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    second_sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in second_sentences:
        second_sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    second_answer, second_diagnostics = execute_bounded_query(
        store,
        second_run_id,
        second_documents,
        second_sentences_by_document,  # type: ignore[arg-type]
        frame.question_text,
        frame,
    )

    assert first_answer is not None
    assert first_answer.text == "red"
    assert second_answer is not None
    assert second_answer.text == "green"
    assert store.integrity_check() == "ok"
    assert "answer_conflict_without_query_scope" not in second_diagnostics["execution"]
    state_values = {
        row["value"]
        for row in store.execute(
            """
            SELECT a.value
            FROM drs_condition_arguments a
            JOIN drs_conditions c ON c.drs_condition_id=a.drs_condition_id
            JOIN source_spans s ON s.span_id=c.source_span_id
            JOIN chunks ch ON ch.chunk_id=s.chunk_id
            JOIN documents d ON d.document_id=ch.document_id
            WHERE c.source='local_model_drs'
              AND c.predicate='state'
              AND d.rel_path='middle/state.txt'
              AND a.role='state'
            """
        ).fetchall()
    }
    assert state_values == {"green"}
    assert "end/crosswalk.txt" in {
        item["rel_path"] for item in second_diagnostics["execution"]["identity_expansion_evidence"]
    }
    assert store.execute(
        "SELECT COUNT(*) FROM model_attempts WHERE task='chunk_drs' AND materialized=1"
    ).fetchone()[0] == 3

    model.version = "v1"
    store, third_run_id, third_documents, third_sentences = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    third_sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in third_sentences:
        third_sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    third_answer, third_diagnostics = execute_bounded_query(
        store,
        third_run_id,
        third_documents,
        third_sentences_by_document,  # type: ignore[arg-type]
        frame.question_text,
        frame,
    )

    assert third_answer is not None
    assert third_answer.text == "red"
    assert "answer_conflict_without_query_scope" not in third_diagnostics["execution"]
    assert store.integrity_check() == "ok"
    assert store.execute(
        "SELECT COUNT(*) FROM model_attempts WHERE task='chunk_drs' AND materialized=1"
    ).fetchone()[0] == 3


def test_incremental_ingest_uses_chunk_boundary_ids_when_scan_policy_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    text = (
        "Aero Gate carries a very long operational note with Alpha status and "
        "Beta status details that should be re-chunked when the scan policy changes."
    )
    (tmp_path / "note.txt").write_text(text, encoding="utf-8")
    store = DSPGStore()

    monkeypatch.setenv("KMD_SCAN_UNIT_MAX_CHARS", "0")
    store, first_run_id, _, first_sentences = ingest_folder(tmp_path, store=store)
    assert len(first_sentences) == 1
    first_span_id = stable_id("span", first_sentences[0].sentence_id, "sentence")
    first_chunk_id = stable_id("chunk", first_sentences[0].sentence_id)

    monkeypatch.setenv("KMD_SCAN_UNIT_MAX_CHARS", "48")
    store, second_run_id, second_documents, second_sentences = ingest_folder(tmp_path, store=store)

    assert first_run_id == second_run_id
    assert len(second_sentences) > 1
    assert first_span_id not in {
        stable_id("span", sentence.sentence_id, "sentence") for sentence in second_sentences
    }
    for sentence in second_sentences:
        chunk_id = stable_id("chunk", sentence.sentence_id)
        stored = store.execute("SELECT text, char_start, char_end FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        assert stored is not None
        assert stored["text"] == sentence.text
        assert stored["char_start"] == sentence.char_start
        assert stored["char_end"] == sentence.char_end
    second_sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in second_sentences:
        second_sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What status details are recorded for Aero Gate?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Aero Gate",),
        requested_relation="status",
        relation_terms=("status",),
        constraints=(),
    )
    selected_docs, selected_chunk_ids, _ranking = _rank_scope(
        second_documents,
        second_sentences_by_document,  # type: ignore[arg-type]
        frame.question_text,
        frame,
        40,
        160,
    )
    records = _load_records(store, second_run_id, selected_docs, selected_chunk_ids)
    current_chunk_ids = {stable_id("chunk", sentence.sentence_id) for sentence in second_sentences}
    loaded_chunk_ids = {row["chunk_id"] for row in records["chunks"]}
    assert first_chunk_id not in loaded_chunk_ids
    assert loaded_chunk_ids <= current_chunk_ids


def test_incremental_rechunk_excludes_stale_document_identity_hypotheses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    text = (
        "Aero Gate begins the registry with neutral filler segment alpha bravo "
        "charlie delta echo foxtrot golf hotel india juliet kilo lima before "
        "the identifier AG-1 marker T001 state blue."
    )
    (tmp_path / "note.txt").write_text(text, encoding="utf-8")

    class RechunkIdentityModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-rechunk-stale-identity-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Aero Gate begins" in prompt and "AG-1 marker T001 state blue" in prompt:
                referents = [
                    {"id": "r0", "label": "Aero Gate", "kind": "artifact", "evidence_text": "Aero Gate"},
                    {"id": "r1", "label": "AG-1", "kind": "identifier", "evidence_text": "AG-1"},
                ]
                conditions = [
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
                                "target_id": "r1",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "AG-1",
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
                        "evidence_text": "AG-1 marker T001 state blue",
                    }
                ]
                identities = [
                    {
                        "left_referent_id": "r0",
                        "right_referent_id": "r1",
                        "status": "accepted",
                        "evidence_text": text,
                        "confidence": 0.94,
                    }
                ]
                temporals = [{"id": "t0", "value": "T001", "value_type": "sequence_marker", "evidence_text": "T001"}]
                box_evidence = text
            elif "AG-1 marker T001 state blue" in prompt:
                referents = [{"id": "r0", "label": "AG-1", "kind": "identifier", "evidence_text": "AG-1"}]
                conditions = [
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
                                "value_type": "identifier",
                                "evidence_text": "AG-1",
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
                        "evidence_text": "AG-1 marker T001 state blue",
                    }
                ]
                identities = []
                temporals = [{"id": "t0", "value": "T001", "value_type": "sequence_marker", "evidence_text": "T001"}]
                box_evidence = "AG-1 marker T001 state blue"
            elif "Aero Gate" in prompt:
                referents = [{"id": "r0", "label": "Aero Gate", "kind": "artifact", "evidence_text": "Aero Gate"}]
                conditions = []
                identities = []
                temporals = []
                box_evidence = "Aero Gate"
            else:
                referents = []
                conditions = []
                identities = []
                temporals = []
                box_evidence = ""
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "note.txt",
                    "referents": referents,
                    "boxes": [
                        {
                            "id": "b0",
                            "kind": "asserted",
                            "parent_id": "",
                            "holder_referent_id": "",
                            "evidence_text": box_evidence,
                        }
                    ],
                    "conditions": conditions,
                    "identity_hypotheses": identities,
                    "temporal_records": temporals,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    store = DSPGStore()
    monkeypatch.setenv("KMD_SCAN_UNIT_MAX_CHARS", "0")
    store, first_run_id, _, first_sentences = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=RechunkIdentityModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    assert len(first_sentences) == 1
    first_span_id = stable_id("span", first_sentences[0].sentence_id, "sentence")
    first_chunk_id = stable_id("chunk", first_sentences[0].sentence_id)
    assert store.execute(
        "SELECT COUNT(*) FROM identity_hypotheses WHERE source_span_id=?",
        (first_span_id,),
    ).fetchone()[0] >= 1

    monkeypatch.setenv("KMD_SCAN_UNIT_MAX_CHARS", "64")
    store, second_run_id, second_documents, second_sentences = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=RechunkIdentityModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    assert first_run_id == second_run_id
    assert len(second_sentences) > 1
    current_chunk_ids = {stable_id("chunk", sentence.sentence_id) for sentence in second_sentences}
    assert first_chunk_id not in current_chunk_ids

    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in second_sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What is the latest state for Aero Gate?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Aero Gate",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
    )
    selected_docs, selected_chunk_ids, _ranking = _rank_scope(
        second_documents,
        sentences_by_document,  # type: ignore[arg-type]
        frame.question_text,
        frame,
        40,
        160,
    )
    records = _load_records(
        store,
        second_run_id,
        selected_docs,
        selected_chunk_ids,
        current_document_chunk_ids=list(current_chunk_ids),
    )
    assert first_chunk_id not in {row["chunk_id"] for row in records["chunks"]}
    assert first_span_id not in {row["source_span_id"] for row in records["identity_hypotheses"]}

    answer, diagnostics = execute_bounded_query(
        store,
        second_run_id,
        second_documents,
        sentences_by_document,  # type: ignore[arg-type]
        frame.question_text,
        frame,
    )

    assert answer is None
    assert "ag-1" not in diagnostics["ranking"].get("identity_expanded_target_terms", [])


def test_identity_hypothesis_loading_batches_current_chunk_scope(tmp_path: Path) -> None:
    store = DSPGStore()
    run_id = store.start_run(tmp_path)
    document_id = stable_id("doc", "large-current-scope")
    store.execute(
        """
        INSERT INTO documents(
          document_id, run_id, path, rel_path, content_hash, size_bytes, mtime, ctime, char_count, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (document_id, run_id, str(tmp_path / "large.txt"), "large.txt", "hash", 0, 0.0, 0.0, 0, "{}"),
    )
    left_ref = store.upsert_referent(run_id, "Node Alpha", "artifact")
    right_ref = store.upsert_referent(run_id, "Node Beta", "identifier")
    current_chunk_ids: list[str] = []
    for index in range(450):
        chunk_id = stable_id("chunk", "current", index)
        span_id = stable_id("span", "current", index)
        current_chunk_ids.append(chunk_id)
        store.execute(
            "INSERT INTO chunks(chunk_id, document_id, chunk_order, char_start, char_end, text, token_estimate) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chunk_id, document_id, index, index * 10, index * 10 + 5, f"chunk {index}", 1),
        )
        store.execute(
            "INSERT INTO source_spans(span_id, document_id, chunk_id, char_start, char_end, surface, surface_norm, span_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (span_id, document_id, chunk_id, index * 10, index * 10 + 5, f"chunk {index}", f"chunk {index}", "sentence"),
        )
    current_span_id = stable_id("span", "current", 449)
    stale_chunk_id = stable_id("chunk", "stale")
    stale_span_id = stable_id("span", "stale")
    store.execute(
        "INSERT INTO chunks(chunk_id, document_id, chunk_order, char_start, char_end, text, token_estimate) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (stale_chunk_id, document_id, 999, 9990, 9999, "stale chunk", 1),
    )
    store.execute(
        "INSERT INTO source_spans(span_id, document_id, chunk_id, char_start, char_end, surface, surface_norm, span_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (stale_span_id, document_id, stale_chunk_id, 9990, 9999, "stale chunk", "stale chunk", "sentence"),
    )
    for span_id, label in [(current_span_id, "current"), (stale_span_id, "stale")]:
        store.execute(
            """
            INSERT INTO identity_hypotheses(
              hypothesis_id, run_id, source_span_id, context_id, drs_box_id, box_external_id,
              left_referent_id, right_referent_id, relation, evidence, confidence, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_id("idh", label),
                run_id,
                span_id,
                None,
                None,
                None,
                left_ref,
                right_ref,
                "same_referent",
                label,
                0.9,
                "test",
            ),
        )

    rows = _fetch_identity_hypotheses(
        store.connection,
        run_id,
        [],
        [document_id],
        current_document_chunk_ids=current_chunk_ids,
    )

    assert {row["source_span_id"] for row in rows} == {current_span_id}


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


def test_incremental_drs_ingest_retries_failed_attempt_when_output_budget_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path.parent / f"{tmp_path.name}-failed-budget-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("KMD_CHUNK_DRS_STAGED_FALLBACK", "0")
    (tmp_path / "note.txt").write_text("Aero Gate is ready.\n", encoding="utf-8")

    class BudgetSensitiveFailingDrsModel:
        def __init__(self) -> None:
            self.calls = 0
            self.n_predicts: list[int] = []

        def context_size(self) -> int:
            return 32768

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-failing-budget-drs", "context_size": 32768}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            self.n_predicts.append(int(n_predict))
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

    model = BudgetSensitiveFailingDrsModel()
    store = DSPGStore()

    monkeypatch.setenv("KMD_CHUNK_DRS_N_PREDICT", "544")
    store, first_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    monkeypatch.setenv("KMD_CHUNK_DRS_N_PREDICT", "768")
    store, second_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )

    assert first_run_id == second_run_id
    assert model.calls == 2
    assert model.n_predicts == [544, 768]
    rows = store.execute(
        "SELECT cache_key, metadata_json FROM model_attempts WHERE task='chunk_drs' ORDER BY cache_key"
    ).fetchall()
    assert len(rows) == 2
    contexts = [json.loads(row["metadata_json"])["cache_context"] for row in rows]
    assert {context["n_predict"] for context in contexts} == {544, 768}


def test_incremental_drs_ingest_retries_previous_request_failures(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path.parent / f"{tmp_path.name}-request-failed-drs-cache"
    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(cache_dir))
    (tmp_path / "note.txt").write_text("Aero Gate is ready.\n", encoding="utf-8")

    class TransientDrsModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-transient-incremental-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary DRS request failure")
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "note.txt",
                    "referents": [
                        {"id": "r0", "label": "Aero Gate", "kind": "artifact", "evidence_text": "Aero Gate"},
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
                                    "role": "subject",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "artifact",
                                    "evidence_text": "Aero Gate",
                                }
                            ],
                            "evidence_text": "Aero Gate is ready",
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    model = TransientDrsModel()
    store = DSPGStore()

    store, first_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    store, second_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=False,
        use_drs_semantics=True,
    )

    assert first_run_id == second_run_id
    assert model.calls == 2
    assert store.execute("SELECT COUNT(*) FROM drs_boxes WHERE source='local_model_drs'").fetchone()[0] == 1
    attempt = store.execute(
        "SELECT accepted, materialized, reason FROM model_attempts WHERE task='chunk_drs'"
    ).fetchone()
    assert attempt["accepted"] == 1
    assert attempt["materialized"] == 1


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


def test_drs_ingest_runs_after_frames_only_materialization_is_reused(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "note.txt").write_text("Aero Gate is ready.\n", encoding="utf-8")

    class FrameAndDrsModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-frame-drs-incremental", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            text = "Aero Gate is ready."
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
                        "evidence_text": text,
                        "confidence": 0.9,
                    }
                ],
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
                            "evidence_text": text,
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
                            "evidence_text": text,
                        }
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    model = FrameAndDrsModel()
    store = DSPGStore()

    store, first_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=True,
        use_drs_semantics=False,
    )
    calls_after_frame_only = model.calls
    store, second_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=True,
        use_drs_semantics=True,
    )

    assert first_run_id == second_run_id
    assert calls_after_frame_only == 1
    assert model.calls == calls_after_frame_only + 1
    assert store.execute("SELECT COUNT(*) FROM frames WHERE source='local_model'").fetchone()[0] == 1
    assert store.counts()["drs_conditions"] == 1
    assert store.counts()["model_attempts"] == 2


def test_incremental_frame_ingest_reprocesses_when_model_fingerprint_changes(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("Aero Gate is ready.\n", encoding="utf-8")

    class VersionedFrameModel:
        def __init__(self) -> None:
            self.calls = 0
            self.version = "v1"

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": f"fake-versioned-frames-{self.version}", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            assert "Aero Gate is ready" in prompt
            return {
                "frames": [
                    {
                        "frame_type": "state",
                        "predicate": f"ready_{self.version}",
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

    model = VersionedFrameModel()
    store = DSPGStore()

    store, first_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=True,
        use_drs_semantics=False,
    )
    store, second_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=True,
        use_drs_semantics=False,
    )
    calls_after_same_fingerprint = model.calls
    model.version = "v2"
    store, third_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=True,
        use_drs_semantics=False,
    )

    assert first_run_id == second_run_id == third_run_id
    assert calls_after_same_fingerprint == 1
    assert model.calls == 2
    predicates = {
        row["predicate"]
        for row in store.execute("SELECT predicate FROM frames WHERE source='local_model'").fetchall()
    }
    assert predicates == {"ready_v2"}
    relation_predicates = {
        row["predicate"]
        for row in store.execute(
            """
            SELECT predicate
            FROM relations
            WHERE relation_type IN ('semantic_frame', 'semantic_argument')
            """
        ).fetchall()
    }
    assert relation_predicates == {"ready_v2"}
    assert store.integrity_check() == "ok"
    assert store.counts()["model_attempts"] == 2
    assert store.execute(
        "SELECT COUNT(*) FROM model_attempts WHERE task='chunk_frames' AND materialized=1"
    ).fetchone()[0] == 1

    model.version = "v1"
    store, fourth_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=True,
        use_drs_semantics=False,
    )

    assert fourth_run_id == first_run_id
    assert model.calls == 3
    predicates = {
        row["predicate"]
        for row in store.execute("SELECT predicate FROM frames WHERE source='local_model'").fetchall()
    }
    assert predicates == {"ready_v1"}
    assert store.counts()["model_attempts"] == 2
    assert store.execute(
        "SELECT COUNT(*) FROM model_attempts WHERE task='chunk_frames' AND materialized=1"
    ).fetchone()[0] == 1


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


def test_incremental_frame_ingest_retries_previous_request_failures(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("Aero Gate is ready.\n", encoding="utf-8")

    class TransientFrameModel:
        def __init__(self) -> None:
            self.calls = 0

        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-transient-incremental-frames", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary frame request failure")
            assert "Aero Gate is ready" in prompt
            return {
                "frames": [
                    {
                        "frame_type": "state",
                        "predicate": "ready",
                        "arguments": [
                            {"role": "entity", "text": "Aero Gate", "value_type": "artifact"},
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

    model = TransientFrameModel()
    store = DSPGStore()

    store, first_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=True,
        use_drs_semantics=False,
    )
    store, second_run_id, _, _ = ingest_folder(
        tmp_path,
        store=store,
        semantic_client=model,
        use_semantic_frames=True,
        use_drs_semantics=False,
    )

    assert first_run_id == second_run_id
    assert model.calls == 2
    assert store.execute("SELECT COUNT(*) FROM frames WHERE source='local_model'").fetchone()[0] == 1
    attempt = store.execute(
        "SELECT accepted, materialized, reason FROM model_attempts WHERE task='chunk_frames'"
    ).fetchone()
    assert attempt["accepted"] == 1
    assert attempt["materialized"] == 1


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
    temporal_refs = store.execute(
        """
        SELECT te.referent_id, r.canonical_label
        FROM temporal_edges te
        JOIN referents r ON r.referent_id=te.referent_id
        ORDER BY te.temporal_value
        """
    ).fetchall()
    assert [row["canonical_label"] for row in temporal_refs] == ["Lumen Core", "Lumen Core"]
    assert answer is not None
    assert answer.text == "green"
    assert answer.evidence[0].rel_path == "late.txt"


def test_scattered_identity_temporal_latest_answer_keeps_crosswalk_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "begin").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "timeline").mkdir()
    (tmp_path / "begin" / "registry.txt").write_text(
        "Registry introduces Cloud Dial as the weather artifact.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "crosswalk.txt").write_text(
        "Crosswalk states CD-2 is the same artifact as Cloud Dial.",
        encoding="utf-8",
    )
    (tmp_path / "timeline" / "early.txt").write_text(
        "Timeline entry: CD-2 marker T001 state draft.",
        encoding="utf-8",
    )
    (tmp_path / "timeline" / "late.txt").write_text(
        "Timeline entry: CD-2 marker T004 state final.",
        encoding="utf-8",
    )

    class ScatteredTemporalIdentityModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-scattered-temporal-identity-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Registry introduces" in prompt:
                text = "Registry introduces Cloud Dial as the weather artifact."
                referents = [{"id": "r0", "label": "Cloud Dial", "kind": "artifact", "evidence_text": "Cloud Dial"}]
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
                                "evidence_text": "Cloud Dial",
                            }
                        ],
                        "evidence_text": text,
                    }
                ]
                identities = []
                temporals = []
            elif "Crosswalk states" in prompt:
                text = "Crosswalk states CD-2 is the same artifact as Cloud Dial."
                referents = [
                    {"id": "r0", "label": "CD-2", "kind": "identifier", "evidence_text": "CD-2"},
                    {"id": "r1", "label": "Cloud Dial", "kind": "artifact", "evidence_text": "Cloud Dial"},
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
                                "evidence_text": "CD-2",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Cloud Dial",
                            },
                        ],
                        "evidence_text": "CD-2 is the same artifact as Cloud Dial",
                    }
                ]
                identities = [
                    {
                        "left_referent_id": "r0",
                        "right_referent_id": "r1",
                        "status": "accepted",
                        "evidence_text": "CD-2 is the same artifact as Cloud Dial",
                        "confidence": 0.93,
                    }
                ]
                temporals = []
            else:
                marker = "T001" if "T001" in prompt else "T004"
                state = "draft" if marker == "T001" else "final"
                text = f"Timeline entry: CD-2 marker {marker} state {state}."
                referents = [{"id": "r0", "label": "CD-2", "kind": "identifier", "evidence_text": "CD-2"}]
                conditions = [
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
                                "value_type": "identifier",
                                "evidence_text": "CD-2",
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
                        "evidence_text": f"CD-2 marker {marker} state {state}",
                    }
                ]
                identities = []
                temporals = [{"id": "t0", "value": marker, "value_type": "sequence_marker", "evidence_text": marker}]
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "scattered-temporal.txt",
                    "referents": referents,
                    "boxes": [
                        {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}
                    ],
                    "conditions": conditions,
                    "identity_hypotheses": identities,
                    "temporal_records": temporals,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=ScatteredTemporalIdentityModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What is the latest state for Cloud Dial?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Cloud Dial",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
    )

    answer, diagnostics = execute_bounded_query(store, run_id, documents, sentences_by_document, frame.question_text, frame)

    temporal_refs = store.execute("SELECT referent_id FROM temporal_edges ORDER BY temporal_value").fetchall()
    assert all(row["referent_id"] for row in temporal_refs)
    assert answer is not None
    assert answer.text == "final"
    assert answer.evidence[0].rel_path == "timeline/late.txt"
    assert "middle/crosswalk.txt" in {item.rel_path for item in answer.evidence}
    assert "middle/crosswalk.txt" in {
        item["rel_path"] for item in diagnostics["execution"]["identity_expansion_evidence"]
    }


def test_document_local_identity_bridge_outside_chunk_budget_reaches_scattered_temporal_states(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "begin").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "end").mkdir()
    (tmp_path / "reports").mkdir()
    registry_lines = ["Registry page opens for Cloud Dial."]
    registry_lines.extend(
        f"Registry filler {index:02d} repeats Cloud Dial without an operational state."
        for index in range(30)
    )
    registry_lines.append("Crosswalk near the end states CD-2 is the same artifact as Cloud Dial.")
    (tmp_path / "begin" / "registry.txt").write_text("\n".join(registry_lines), encoding="utf-8")
    (tmp_path / "middle" / "state_early.log").write_text(
        "Update row CD-2 marker T001 state draft.",
        encoding="utf-8",
    )
    (tmp_path / "end" / "state_final.log").write_text(
        "Update row CD-2 marker T009 state sealed.",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "reported_late.log").write_text(
        "Observer report says CD-2 marker T010 state broken.",
        encoding="utf-8",
    )

    class ScatteredBridgeModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-scattered-bridge-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Crosswalk near the end" in prompt:
                text = "Crosswalk near the end states CD-2 is the same artifact as Cloud Dial."
                referents = [
                    {"id": "r0", "label": "CD-2", "kind": "identifier", "evidence_text": "CD-2"},
                    {"id": "r1", "label": "Cloud Dial", "kind": "artifact", "evidence_text": "Cloud Dial"},
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
                                "evidence_text": "CD-2",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Cloud Dial",
                            },
                        ],
                        "evidence_text": "CD-2 is the same artifact as Cloud Dial",
                    }
                ]
                identities = [
                    {
                        "left_referent_id": "r0",
                        "right_referent_id": "r1",
                        "status": "accepted",
                        "evidence_text": "CD-2 is the same artifact as Cloud Dial",
                        "confidence": 0.94,
                    }
                ]
                temporals = []
                boxes = [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}]
            elif "T001" in prompt or "T009" in prompt:
                marker = "T001" if "T001" in prompt else "T009"
                state = "draft" if marker == "T001" else "sealed"
                text = f"Update row CD-2 marker {marker} state {state}."
                referents = [{"id": "r0", "label": "CD-2", "kind": "identifier", "evidence_text": "CD-2"}]
                conditions = [
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
                                "value_type": "identifier",
                                "evidence_text": "CD-2",
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
                        "evidence_text": f"CD-2 marker {marker} state {state}",
                    }
                ]
                identities = []
                temporals = [{"id": "t0", "value": marker, "value_type": "sequence_marker", "evidence_text": marker}]
                boxes = [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}]
            elif "T010" in prompt:
                text = "Observer report says CD-2 marker T010 state broken."
                reported = "CD-2 marker T010 state broken"
                referents = [
                    {"id": "r0", "label": "Observer report", "kind": "document", "evidence_text": "Observer report"},
                    {"id": "r1", "label": "CD-2", "kind": "identifier", "evidence_text": "CD-2"},
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "report",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "source",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "document",
                                "evidence_text": "Observer report",
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
                        "predicate": "state",
                        "box_id": "b1",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "t0",
                        "arguments": [
                            {
                                "role": "subject",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "CD-2",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "broken",
                                "value_type": "state",
                                "evidence_text": "broken",
                            },
                        ],
                        "evidence_text": reported,
                    },
                ]
                identities = []
                temporals = [{"id": "t0", "value": "T010", "value_type": "sequence_marker", "evidence_text": "T010"}]
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text},
                    {
                        "id": "b1",
                        "kind": "reported",
                        "parent_id": "b0",
                        "holder_referent_id": "r0",
                        "evidence_text": reported,
                    },
                ]
            else:
                text = "Cloud Dial"
                referents = [{"id": "r0", "label": "Cloud Dial", "kind": "artifact", "evidence_text": "Cloud Dial"}]
                conditions = []
                identities = []
                temporals = []
                boxes = [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}]
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "scattered-bridge",
                    "referents": referents,
                    "boxes": boxes,
                    "conditions": conditions,
                    "identity_hypotheses": identities,
                    "temporal_records": temporals,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=ScatteredBridgeModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    unscoped_frame = QueryFrame(
        question_text="What is the latest state for Cloud Dial?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Cloud Dial",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
    )
    reported_frame = QueryFrame(
        question_text="What is the latest reported state for Cloud Dial?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Cloud Dial",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
        scope_requirements=("reported",),
    )

    unscoped_answer, unscoped_diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        unscoped_frame.question_text,
        unscoped_frame,
        chunk_limit=12,
    )
    reported_answer, _reported_diagnostics = execute_bounded_query(
        store,
        run_id,
        documents,
        sentences_by_document,  # type: ignore[arg-type]
        reported_frame.question_text,
        reported_frame,
        chunk_limit=12,
    )

    assert unscoped_answer is not None
    assert unscoped_answer.text == "sealed"
    assert {item.rel_path for item in unscoped_answer.evidence} >= {
        "begin/registry.txt",
        "end/state_final.log",
    }
    assert "begin/registry.txt" in {
        item["rel_path"] for item in unscoped_diagnostics["execution"]["identity_expansion_evidence"]
    }
    assert reported_answer is not None
    assert reported_answer.text == "broken"
    assert reported_answer.evidence[0].rel_path == "reports/reported_late.log"


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


def test_latest_temporal_query_respects_reported_drs_scope(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "asserted.txt").write_text(
        "Timeline HN-7 marker T001 state open.",
        encoding="utf-8",
    )
    (tmp_path / "reported.txt").write_text(
        "Analyst report says HN-7 marker T003 state closed.",
        encoding="utf-8",
    )

    class ScopedTemporalDrsModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-scoped-temporal-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "T001" in prompt:
                text = "Timeline HN-7 marker T001 state open."
                return {
                    "drs": {
                        "schema_version": "chunk-drs-v2",
                        "source_id": "asserted.txt",
                        "referents": [
                            {"id": "r0", "label": "HN-7", "kind": "identifier", "evidence_text": "HN-7"},
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
                                        "value_type": "identifier",
                                        "evidence_text": "HN-7",
                                    },
                                    {
                                        "role": "state",
                                        "target_kind": "literal",
                                        "target_id": "",
                                        "value": "open",
                                        "value_type": "state",
                                        "evidence_text": "open",
                                    },
                                ],
                                "evidence_text": "HN-7 marker T001 state open",
                            }
                        ],
                        "identity_hypotheses": [],
                        "temporal_records": [
                            {"id": "t0", "value": "T001", "value_type": "sequence_marker", "evidence_text": "T001"}
                        ],
                    },
                    "_model_raw": "{}",
                    "_model_elapsed_seconds": 0.01,
                }
            text = "Analyst report says HN-7 marker T003 state closed."
            reported = "HN-7 marker T003 state closed"
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "reported.txt",
                    "referents": [
                        {"id": "r0", "label": "Analyst report", "kind": "document", "evidence_text": "Analyst report"},
                        {"id": "r1", "label": "HN-7", "kind": "identifier", "evidence_text": "HN-7"},
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
                                    "role": "source",
                                    "target_kind": "referent",
                                    "target_id": "r0",
                                    "value": "",
                                    "value_type": "document",
                                    "evidence_text": "Analyst report",
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
                            "predicate": "state",
                            "box_id": "b1",
                            "polarity": "positive",
                            "modality": "asserted",
                            "temporal_id": "t0",
                            "arguments": [
                                {
                                    "role": "subject",
                                    "target_kind": "referent",
                                    "target_id": "r1",
                                    "value": "",
                                    "value_type": "identifier",
                                    "evidence_text": "HN-7",
                                },
                                {
                                    "role": "state",
                                    "target_kind": "literal",
                                    "target_id": "",
                                    "value": "closed",
                                    "value_type": "state",
                                    "evidence_text": "closed",
                                },
                            ],
                            "evidence_text": reported,
                        },
                    ],
                    "identity_hypotheses": [],
                    "temporal_records": [
                        {"id": "t0", "value": "T003", "value_type": "sequence_marker", "evidence_text": "T003"}
                    ],
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=ScopedTemporalDrsModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    for index in range(25):
        store.execute(
            """
            INSERT INTO contexts(
              context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_id("ctx", run_id, "unrelated", index),
                run_id,
                f"drs:unrelated-{index}",
                None,
                None,
                f"unrelated scoped material {index}",
                0.1,
            ),
        )
    store.commit()
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    unscoped_frame = QueryFrame(
        question_text="What is the latest state for HN-7?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("HN-7",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
    )
    reported_frame = QueryFrame(
        question_text="What is the latest reported state for HN-7?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("HN-7",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
        scope_requirements=("reported",),
    )

    unscoped_answer, unscoped_diagnostics = execute_bounded_query(
        store, run_id, documents, sentences_by_document, unscoped_frame.question_text, unscoped_frame
    )
    reported_answer, reported_diagnostics = execute_bounded_query(
        store, run_id, documents, sentences_by_document, reported_frame.question_text, reported_frame
    )

    assert unscoped_answer is not None
    assert unscoped_answer.text == "open"
    assert unscoped_answer.evidence[0].rel_path == "asserted.txt"
    assert reported_answer is not None
    assert reported_answer.text == "closed"
    assert reported_answer.evidence[0].rel_path == "reported.txt"
    assert unscoped_diagnostics["execution"]["record_counts"]["contexts"] < store.counts()["contexts"]
    assert reported_diagnostics["execution"]["record_counts"]["contexts"] < store.counts()["contexts"]


def test_latest_temporal_query_with_boundary_conflict_returns_unknown_with_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "begin").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "ending").mkdir()
    (tmp_path / "begin" / "registry.txt").write_text(
        "Registry introduces Hinge Node as the monitored artifact.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "state_a.txt").write_text(
        "State channel A says HN-9 marker T005 state open.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "state_b.txt").write_text(
        "State channel B says HN-9 marker T005 state closed.",
        encoding="utf-8",
    )
    (tmp_path / "ending" / "crosswalk.txt").write_text(
        "Crosswalk states HN-9 is the same artifact as Hinge Node.",
        encoding="utf-8",
    )

    class BoundaryConflictTemporalModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-boundary-conflict-temporal-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Registry introduces" in prompt:
                text = "Registry introduces Hinge Node as the monitored artifact."
                referents = [{"id": "r0", "label": "Hinge Node", "kind": "artifact", "evidence_text": "Hinge Node"}]
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
                                "evidence_text": "Hinge Node",
                            }
                        ],
                        "evidence_text": text,
                    }
                ]
                identities = []
                temporals = []
                source_id = "begin/registry.txt"
            elif "channel A" in prompt:
                text = "State channel A says HN-9 marker T005 state open."
                referents = [{"id": "r0", "label": "HN-9", "kind": "identifier", "evidence_text": "HN-9"}]
                conditions = [
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
                                "value_type": "identifier",
                                "evidence_text": "HN-9",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "open",
                                "value_type": "state",
                                "evidence_text": "open",
                            },
                        ],
                        "evidence_text": "HN-9 marker T005 state open",
                    }
                ]
                identities = []
                temporals = [{"id": "t0", "value": "T005", "value_type": "sequence_marker", "evidence_text": "T005"}]
                source_id = "middle/state_a.txt"
            elif "channel B" in prompt:
                text = "State channel B says HN-9 marker T005 state closed."
                referents = [{"id": "r0", "label": "HN-9", "kind": "identifier", "evidence_text": "HN-9"}]
                conditions = [
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
                                "value_type": "identifier",
                                "evidence_text": "HN-9",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "closed",
                                "value_type": "state",
                                "evidence_text": "closed",
                            },
                        ],
                        "evidence_text": "HN-9 marker T005 state closed",
                    }
                ]
                identities = []
                temporals = [{"id": "t0", "value": "T005", "value_type": "sequence_marker", "evidence_text": "T005"}]
                source_id = "middle/state_b.txt"
            else:
                text = "Crosswalk states HN-9 is the same artifact as Hinge Node."
                referents = [
                    {"id": "r0", "label": "HN-9", "kind": "identifier", "evidence_text": "HN-9"},
                    {"id": "r1", "label": "Hinge Node", "kind": "artifact", "evidence_text": "Hinge Node"},
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
                                "evidence_text": "HN-9",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Hinge Node",
                            },
                        ],
                        "evidence_text": "HN-9 is the same artifact as Hinge Node",
                    }
                ]
                identities = [
                    {
                        "left_referent_id": "r0",
                        "right_referent_id": "r1",
                        "status": "accepted",
                        "evidence_text": "HN-9 is the same artifact as Hinge Node",
                        "confidence": 0.93,
                    }
                ]
                temporals = []
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
                    "identity_hypotheses": identities,
                    "temporal_records": temporals,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=BoundaryConflictTemporalModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What is the latest state for Hinge Node?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Hinge Node",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
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
    assert diagnostics["execution"]["no_answer_reason"] == "temporal_answer_conflict_at_boundary"
    conflict = diagnostics["execution"]["temporal_answer_conflict_at_boundary"]
    assert {item["value"] for item in conflict["values"]} == {"closed", "open"}
    evidence_paths = {
        evidence["rel_path"]
        for item in conflict["values"]
        for evidence in item["evidence"]
    }
    assert {"middle/state_a.txt", "middle/state_b.txt"}.issubset(evidence_paths)
    conflict_evidence = [
        evidence
        for item in conflict["values"]
        for evidence in item["evidence"]
    ]
    assert all(evidence.get("chunk_id") and evidence.get("span_id") for evidence in conflict_evidence)
    assert all(evidence.get("document", {}).get("document_id") == evidence.get("document_id") for evidence in conflict_evidence)
    assert "ending/crosswalk.txt" in {
        item["rel_path"] for item in diagnostics["execution"]["identity_expansion_evidence"]
    }


def test_unlinked_scattered_sources_return_unknown_with_source_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "begin").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "ending").mkdir()
    (tmp_path / "begin" / "registry.txt").write_text(
        "Registry introduces Ember Array as the monitored artifact.",
        encoding="utf-8",
    )
    (tmp_path / "middle" / "state.txt").write_text(
        "Maintenance row EA-7 marker T002 state amber.",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "reported.txt").write_text(
        "Observer report says EA-7 marker T003 state green.",
        encoding="utf-8",
    )
    (tmp_path / "ending" / "similarity.txt").write_text(
        "Closing note says EA-7 resembles Ember Array.",
        encoding="utf-8",
    )

    class UnlinkedScatteredModel:
        def context_size(self) -> int:
            return 4096

        def cache_fingerprint(self) -> dict[str, object]:
            return {"model_id": "fake-unlinked-scattered-drs", "context_size": 4096}

        def complete_json(self, prompt: str, *, n_predict: int = 128, grammar=None, json_schema=None):
            if "Registry introduces" in prompt:
                text = "Registry introduces Ember Array as the monitored artifact."
                referents = [{"id": "r0", "label": "Ember Array", "kind": "artifact", "evidence_text": "Ember Array"}]
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
                                "evidence_text": "Ember Array",
                            }
                        ],
                        "evidence_text": text,
                    }
                ]
                temporals = []
                identities = []
                boxes = [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}]
            elif "T002" in prompt:
                text = "Maintenance row EA-7 marker T002 state amber."
                referents = [{"id": "r0", "label": "EA-7", "kind": "identifier", "evidence_text": "EA-7"}]
                conditions = [
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
                                "value_type": "identifier",
                                "evidence_text": "EA-7",
                            },
                            {
                                "role": "state",
                                "target_kind": "literal",
                                "target_id": "",
                                "value": "amber",
                                "value_type": "state",
                                "evidence_text": "amber",
                            },
                        ],
                        "evidence_text": "EA-7 marker T002 state amber",
                    }
                ]
                temporals = [{"id": "t0", "value": "T002", "value_type": "sequence_marker", "evidence_text": "T002"}]
                identities = []
                boxes = [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}]
            elif "T003" in prompt:
                text = "Observer report says EA-7 marker T003 state green."
                reported = "EA-7 marker T003 state green"
                referents = [
                    {"id": "r0", "label": "Observer report", "kind": "document", "evidence_text": "Observer report"},
                    {"id": "r1", "label": "EA-7", "kind": "identifier", "evidence_text": "EA-7"},
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "report",
                        "box_id": "b0",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "",
                        "arguments": [
                            {
                                "role": "source",
                                "target_kind": "referent",
                                "target_id": "r0",
                                "value": "",
                                "value_type": "document",
                                "evidence_text": "Observer report",
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
                        "predicate": "state",
                        "box_id": "b1",
                        "polarity": "positive",
                        "modality": "asserted",
                        "temporal_id": "t0",
                        "arguments": [
                            {
                                "role": "subject",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "identifier",
                                "evidence_text": "EA-7",
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
                ]
                temporals = [{"id": "t0", "value": "T003", "value_type": "sequence_marker", "evidence_text": "T003"}]
                identities = []
                boxes = [
                    {"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text},
                    {"id": "b1", "kind": "reported", "parent_id": "b0", "holder_referent_id": "r0", "evidence_text": reported},
                ]
            else:
                text = "Closing note says EA-7 resembles Ember Array."
                referents = [
                    {"id": "r0", "label": "EA-7", "kind": "identifier", "evidence_text": "EA-7"},
                    {"id": "r1", "label": "Ember Array", "kind": "artifact", "evidence_text": "Ember Array"},
                ]
                conditions = [
                    {
                        "id": "c0",
                        "predicate": "resembles",
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
                                "evidence_text": "EA-7",
                            },
                            {
                                "role": "right",
                                "target_kind": "referent",
                                "target_id": "r1",
                                "value": "",
                                "value_type": "artifact",
                                "evidence_text": "Ember Array",
                            },
                        ],
                        "evidence_text": "EA-7 resembles Ember Array",
                    }
                ]
                temporals = []
                identities = []
                boxes = [{"id": "b0", "kind": "asserted", "parent_id": "", "holder_referent_id": "", "evidence_text": text}]
            return {
                "drs": {
                    "schema_version": "chunk-drs-v2",
                    "source_id": "unlinked-scattered",
                    "referents": referents,
                    "boxes": boxes,
                    "conditions": conditions,
                    "identity_hypotheses": identities,
                    "temporal_records": temporals,
                },
                "_model_raw": "{}",
                "_model_elapsed_seconds": 0.01,
            }

    monkeypatch.setenv("KMD_CHUNK_DRS_CACHE_DIR", str(tmp_path / ".drs-cache"))
    store, run_id, documents, sentences = ingest_folder(
        tmp_path,
        semantic_client=UnlinkedScatteredModel(),  # type: ignore[arg-type]
        use_semantic_frames=False,
        use_drs_semantics=True,
    )
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What is the latest state for Ember Array?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Ember Array",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
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
    assert "ea-7" not in diagnostics["ranking"].get("identity_expanded_target_terms", [])
    scattered = diagnostics["execution"]["scattered_source_provenance_without_binding"]
    assert {"begin/registry.txt", "ending/similarity.txt"}.issubset(set(scattered["target_rel_paths"]))
    assert {"middle/state.txt", "reports/reported.txt"}.issubset(set(scattered["relation_rel_paths"]))
    assert {"begin/registry.txt", "ending/similarity.txt"}.issubset(
        {item["rel_path"] for item in scattered["target_sources"]}
    )
    assert {"middle/state.txt", "reports/reported.txt"}.issubset(
        {item["rel_path"] for item in scattered["relation_sources"]}
    )
    assert all(item.get("chunk_id") and item.get("span_id") for item in scattered["target_sources"])
    assert all(item.get("chunk_id") and item.get("span_id") for item in scattered["relation_sources"])
    provenance = diagnostics["execution"]["source_provenance_sample"]
    provenance_paths = {item["rel_path"] for item in provenance}
    assert {"begin/registry.txt", "middle/state.txt", "reports/reported.txt", "ending/similarity.txt"}.issubset(
        provenance_paths
    )
    assert all(item.get("chunk_id") and item.get("span_id") for item in provenance)
    assert all(item.get("document", {}).get("document_id") == item.get("document_id") for item in provenance)


def test_no_answer_provenance_balances_scattered_target_and_relation_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KMD_USE_LOCAL_MODEL", "0")
    (tmp_path / "begin").mkdir()
    (tmp_path / "middle").mkdir()
    (tmp_path / "ending").mkdir()
    (tmp_path / "begin" / "registry.txt").write_text(
        "\n".join(
            f"Registry target line {index:02d}: Iris Vault remains the monitored artifact."
            for index in range(14)
        ),
        encoding="utf-8",
    )
    (tmp_path / "middle" / "state.txt").write_text(
        "Maintenance lane IV-4 marker T002 state blue.",
        encoding="utf-8",
    )
    (tmp_path / "ending" / "similarity.txt").write_text(
        "Closing note says IV-4 resembles Iris Vault.",
        encoding="utf-8",
    )

    store, run_id, documents, sentences = ingest_folder(tmp_path)
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="What is the latest state for Iris Vault?",
        answer_type="state",
        answer_variables=("state",),
        target_anchors=("Iris Vault",),
        requested_relation="state",
        relation_terms=("state",),
        constraints=(),
        temporal_scope="latest",
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
    provenance_paths = {item["rel_path"] for item in provenance}
    assert "begin/registry.txt" in provenance_paths
    assert "middle/state.txt" in provenance_paths
    scattered = diagnostics["execution"]["scattered_source_provenance_without_binding"]
    assert scattered["target_rel_paths"] == ["begin/registry.txt", "ending/similarity.txt"]
    assert scattered["relation_rel_paths"] == ["middle/state.txt"]
    assert {"begin/registry.txt", "ending/similarity.txt"}.issubset(
        {item["rel_path"] for item in scattered["target_sources"]}
    )
    assert {item["rel_path"] for item in scattered["relation_sources"]} == {"middle/state.txt"}
    assert all(item.get("document", {}).get("document_id") == item.get("document_id") for item in scattered["target_sources"])
    assert all(item.get("document", {}).get("document_id") == item.get("document_id") for item in scattered["relation_sources"])
    assert all(item.get("chunk_id") and item.get("span_id") for item in provenance)
    assert all(item.get("document", {}).get("document_id") == item.get("document_id") for item in provenance)


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


def test_count_aggregation_is_not_blocked_by_unscoped_temporal_candidates(tmp_path: Path) -> None:
    (tmp_path / "rows.tsv").write_text(
        "\n".join(
            [
                "unit\tstate",
                "Alpha unit\topen",
                "Beta unit\topen",
                "Gamma unit\tpaused",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "timeline.log").write_text(
        "\n".join(
            [
                "2026-04-02 08:00 Delta unit state: open.",
                "2026-04-02 10:00 Delta unit state: closed.",
            ]
        ),
        encoding="utf-8",
    )
    store, run_id, documents, sentences = ingest_folder(tmp_path)
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="How many rows have state open?",
        answer_type="count",
        answer_variables=("rows",),
        target_anchors=(),
        requested_relation="state",
        relation_terms=("state", "open"),
        constraints=(),
        aggregation="count",
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
    assert answer.text == "2"
    assert answer.reason == "record-group aggregation DRS binding"
    assert "temporal_ambiguity_without_query_scope" not in diagnostics["execution"]


def test_count_aggregation_ignores_query_unit_terms_for_record_groups(tmp_path: Path) -> None:
    (tmp_path / "rows.tsv").write_text(
        "\n".join(
            [
                "actor\titem\tstate\tid",
                "Mira Sol\tAster One\topen\tAS-001",
                "Mira Sol\tAster Two\topen\tAS-002",
                "Pax Neri\tBeryl One\topen\tBY-001",
                "Mira Sol\tCedar One\tclosed\tCD-001",
            ]
        ),
        encoding="utf-8",
    )
    store, run_id, documents, sentences = ingest_folder(tmp_path)
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="How many rows for Mira Sol have state open?",
        answer_type="count",
        answer_variables=(),
        target_anchors=("Mira Sol",),
        requested_relation="count",
        relation_terms=("rows", "state", "open"),
        constraints=(),
        aggregation="count",
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
    assert answer.text == "2"
    assert answer.reason == "record-group aggregation DRS binding"
    assert "no_answer_reason" not in diagnostics["execution"]


def test_row_count_aggregation_excludes_non_table_state_mentions(tmp_path: Path) -> None:
    (tmp_path / "rows.tsv").write_text(
        "\n".join(
            [
                "actor\titem\tstate",
                "Mira Sol\tAster One\topen",
                "Mira Sol\tAster Two\topen",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text(
        "Delta marker state open in prose but not as a table row.",
        encoding="utf-8",
    )
    store, run_id, documents, sentences = ingest_folder(tmp_path)
    sentences_by_document: dict[str, dict[int, object]] = {}
    for sentence in sentences:
        sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
    frame = QueryFrame(
        question_text="How many rows have state open?",
        answer_type="count",
        answer_variables=(),
        target_anchors=(),
        requested_relation="count",
        relation_terms=("state", "open"),
        constraints=(),
        aggregation="count",
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
    assert answer.text == "2"
    assert answer.reason == "record-group aggregation DRS binding"
    assert {item.rel_path for item in answer.evidence} == {"rows.tsv"}
    assert "no_answer_reason" not in diagnostics["execution"]


def test_target_entity_kept_when_answer_variable_is_entity_plus_field(tmp_path: Path) -> None:
    frame = QueryFrame(
        question_text="What is the greenhouse pump state?",
        answer_type="state",
        answer_variables=("greenhouse pump state",),
        target_anchors=("greenhouse pump",),
        requested_relation="state",
        relation_terms=("state", "greenhouse", "pump"),
        constraints=("greenhouse", "pump", "state"),
    )

    assert "greenhouse pump" in _target_terms(frame, frame.question_text)


def test_table_selector_returns_subject_identifier_when_value_matches_requested_state(tmp_path: Path) -> None:
    (tmp_path / "invoice.csvish").write_text(
        "invoice_id|customer|amount|status\n"
        "INV-101|River Clinic|125|unpaid\n"
        "INV-102|River Clinic|125|paid\n",
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="Which invoice is unpaid?",
        answer_type="identifier",
        answer_variables=("invoice",),
        target_anchors=("invoice",),
        requested_relation="is unpaid",
        relation_terms=("is unpaid", "invoice", "unpaid"),
        constraints=("invoice", "unpaid"),
    )

    answer = engine._answer_with_bounded_dspg(frame.question_text, frame, ExpectedAnswer("identifier"))

    assert answer is not None
    assert answer.text == "INV-101"


def test_where_frame_preserves_locative_preposition() -> None:
    assert (
        _locative_answer_value(
            {"predicate": "on", "trigger_surface": "on"},
            "the red desk",
            ["where", "brass", "lamp"],
        )
        == "on the red desk"
    )


def test_object_like_source_format_terms_bind_raw_json_owner(tmp_path: Path) -> None:
    (tmp_path / "raw_json_like.blob").write_text(
        '{ project: "Not a schema", owner: "Zia Fern", status: "observed", ticket: "TXT-991" }',
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="Who is the owner in the raw JSON-like text?",
        answer_type="person",
        answer_variables=("owner",),
        target_anchors=("owner", "JSON-like"),
        requested_relation="",
        relation_terms=("owner", "raw"),
        constraints=("raw", "json-like", "text", "owner"),
    )

    answer = engine._answer_with_bounded_dspg(frame.question_text, frame, ExpectedAnswer("person"))

    assert answer is not None
    assert answer.text == "Zia Fern"


def test_clear_structural_candidate_not_blocked_by_unrelated_dated_evidence(tmp_path: Path) -> None:
    (tmp_path / "hotel.txt").write_text(
        "Hotel confirmation code: HTL-7712. 2026-07-12 picnic state: planned.",
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="What is the hotel confirmation code?",
        answer_type="identifier",
        answer_variables=("hotel confirmation code",),
        target_anchors=("hotel confirmation code",),
        requested_relation="is",
        relation_terms=("is", "answer", "argument", "hotel", "confirmation", "code"),
        constraints=("hotel", "confirmation", "code"),
    )

    answer = engine._answer_with_bounded_dspg(frame.question_text, frame, ExpectedAnswer("identifier"))

    assert answer is not None
    assert answer.text == "HTL-7712"


def test_target_terms_keep_real_anchors_that_also_appear_in_relation_terms() -> None:
    frame = QueryFrame(
        question_text="Who drafted the volcano homework essay for Meadow Class?",
        answer_type="person",
        answer_variables=("Who",),
        target_anchors=("volcano homework essay", "Meadow Class"),
        requested_relation="drafted",
        relation_terms=("drafted", "who", "answer", "argument", "volcano", "homework", "essay"),
        constraints=("drafted", "volcano", "homework", "essay"),
    )

    terms = _target_terms(frame, frame.question_text)

    assert "volcano homework essay" in terms
    assert "meadow class" in terms


def test_count_aggregation_ignores_how_many_relation_term_from_model_query(tmp_path: Path) -> None:
    (tmp_path / "noise.log").write_text(
        "Bell Finch active owner: BAD-1234 0000 ==== //// ++++ !!!!\n",
        encoding="utf-8",
    )
    (tmp_path / "rows.tsv").write_text(
        "item\tstatus\towner\treference\n"
        "Bell Finch\tactive\tOla Nym\tBF-1201\n"
        "Bell Finch\tarchived\tLio Fern\tBF-1200\n"
        "Cedar Finch\tactive\tPax Neri\tCF-2201\n"
        "Dune Finch\tblocked\tRae Sol\tDF-3301\n"
        "Ember Finch\tactive\tUma Korr\tEF-4401\n",
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="How many Finch rows have status active?",
        answer_type="count",
        answer_variables=("How many",),
        target_anchors=("Finch rows", "Finch"),
        requested_relation="have status",
        relation_terms=("have status", "how many", "answer", "argument", "status", "active"),
        constraints=("active", "status"),
        aggregation="count",
    )

    answer = engine._answer_with_bounded_dspg(frame.question_text, frame, ExpectedAnswer("count"))

    assert answer is not None
    assert answer.text == "3"


def test_count_aggregation_treats_entries_as_row_units_and_skips_noise(tmp_path: Path) -> None:
    (tmp_path / "noise.log").write_text(
        "Bell Finch active owner: BAD-1234 0000 ==== //// ++++ !!!!\n",
        encoding="utf-8",
    )
    (tmp_path / "rows.tsv").write_text(
        "item\tstatus\towner\n"
        "Bell Finch\tactive\tOla Nym\n"
        "Bell Finch\tarchived\tLio Fern\n"
        "Cedar Finch\tactive\tPax Neri\n"
        "Ember Finch\tactive\tUma Korr\n",
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="How many Finch entries are active?",
        answer_type="count",
        answer_variables=("How many Finch entries",),
        target_anchors=("Finch",),
        requested_relation="are active",
        relation_terms=("are active", "active", "entries"),
        constraints=(),
        aggregation="count",
    )

    answer = engine._answer_with_bounded_dspg(frame.question_text, frame, ExpectedAnswer("count"))

    assert answer is not None
    assert answer.text == "3"


def test_count_aggregation_matches_field_value_tokens_not_whole_verb_phrase(tmp_path: Path) -> None:
    (tmp_path / "rows.tsv").write_text(
        "name\tstatus\n"
        "Bell Finch\tactive\n"
        "Dune Finch\tactive\n"
        "Lake Finch\tactive\n"
        "Oak Finch\tarchived\n"
        "Mira Sol\topen\n",
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="How many Finch rows have status active?",
        answer_type="count",
        answer_variables=("How many Finch rows",),
        target_anchors=("Finch",),
        requested_relation="have status active",
        relation_terms=("have status active", "status active", "rows"),
        constraints=(),
        aggregation="count",
    )

    answer = engine._answer_with_bounded_dspg(frame.question_text, frame, ExpectedAnswer("count"))

    assert answer is not None
    assert answer.text == "3"


def test_document_scoped_label_values_bind_single_target_field(tmp_path: Path) -> None:
    (tmp_path / "recipe.txt").write_text(
        "Recipe: pear oat cakes. Oven temperature: 180C. Bake time: 22 minutes. Author: Aunt Mira.",
        encoding="utf-8",
    )
    engine = KnowMoreDiRTEngine(tmp_path)
    frame = QueryFrame(
        question_text="What is the oven temperature for pear oat cakes?",
        answer_type="identifier",
        answer_variables=("oven temperature",),
        target_anchors=("pear oat cakes",),
        requested_relation="is",
        relation_terms=("is", "oven temperature", "answer", "argument"),
        constraints=(),
    )

    answer = engine._answer_with_bounded_dspg(
        frame.question_text,
        frame,
        ExpectedAnswer("identifier"),
    )

    assert answer is not None
    assert answer.text == "180C"
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


def test_conflict_detection_allows_candidate_with_more_query_anchor_coverage() -> None:
    expected = ExpectedAnswer("identifier")
    candidates = [
        (
            10.0,
            "NS-200",
            Evidence("folder/source.txt", "Reviewer: Nia Sol | marker: NS-200", 0.8),
            "relation_condition_binding",
        ),
        (
            9.5,
            "MV-100",
            Evidence("folder/source.txt", "Reviewer: Mira Vale | marker: MV-100", 0.8),
            "relation_condition_binding",
        ),
    ]

    assert _answer_conflict_diagnostics(candidates, expected, ["nia sol"]) is None


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
