from __future__ import annotations

from pathlib import Path

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


def test_store_supports_referent_centric_candidate_retrieval(tmp_path: Path) -> None:
    (tmp_path / "unstructured.note").write_text(
        "A raw note says BlueTensor reviewed REF-4321 for the ledger cache.",
        encoding="utf-8",
    )
    store, run_id, _, _ = ingest_folder(tmp_path)

    rows = store.referent_candidate_chunks(run_id, ["REF-4321"], limit=3)

    assert rows
    assert "BlueTensor reviewed REF-4321" in rows[0]["text"]


def test_temporal_state_query_uses_dspg_latest_event(tmp_path: Path) -> None:
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
    answer = engine.answer("What is the current state of AuroraGate?")

    assert answer.text == "closed"
    assert answer.reason == "latest temporal state from DSPG"
