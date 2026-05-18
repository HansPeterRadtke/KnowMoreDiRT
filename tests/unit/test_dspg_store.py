from __future__ import annotations

from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.ingest import ingest_folder

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
    assert counts["contexts"] >= 3
    assert counts["frames"] > 20
    assert counts["frame_arguments"] > 20


def test_engine_exposes_internal_dspg_counts_for_diagnostics_only() -> None:
    engine = KnowMoreDiRTEngine(FIXTURE_ROOT)
    counts = engine.dspg_counts()

    assert engine.dspg_integrity() == "ok"
    assert counts["documents"] == 30
    assert counts["mentions"] > 50
    assert counts["frames"] > 20

