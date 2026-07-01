from __future__ import annotations

from knowmoredirt.ingest import _skip_model_semantics_for_quality
from knowmoredirt import bounded_dspg
from knowmoredirt.text import is_low_semantic_noise, text_quality_metrics


def test_text_quality_flags_random_character_noise() -> None:
    noise = "\\x00\\x01@@@###%%%^^^^~~~~" + ("A7f!?" * 80)
    metrics = text_quality_metrics(noise)

    assert metrics["char_count"] > 100
    assert metrics["low_semantic_noise"] is True
    assert metrics["semantic_quality"] == "random_character_noise"
    assert is_low_semantic_noise(noise)


def test_text_quality_labels_machine_token_blobs() -> None:
    blob = (
        "QWxwaGE9Pz8/Pz8= 4f2a9d00beefcafe 00ff11ee22dd33cc "
        + "deadbeef" * 40
    )
    metrics = text_quality_metrics(blob)

    assert metrics["machine_blob_char_ratio"] >= 0.5
    assert metrics["low_semantic_noise"] is False
    assert metrics["semantic_quality"] == "base64_or_hex_blob"
    assert not is_low_semantic_noise(blob)




def test_text_quality_does_not_treat_long_urls_as_chunk_blobs() -> None:
    text = " ".join(
        f"Record {index}: Mara owns retry item {index}. See "
        f"https://docs.example.com/internal/platform/retry-scheduler/2026/08/{index:02d}/"
        f"how-the-retry-scheduler-handles-callback-replay-and-customer-impact/ "
        f"and https://github.com/example/project/pull/{2780 + index}."
        for index in range(40)
    )

    metrics = text_quality_metrics(text)

    assert metrics["low_semantic_noise"] is False
    assert metrics["semantic_quality"] != "base64_or_hex_blob"
    assert not is_low_semantic_noise(text)


def test_text_quality_still_labels_dominant_machine_blobs() -> None:
    text = " ".join(["deadbeef" * 8 for _ in range(12)])

    metrics = text_quality_metrics(text)

    assert metrics["machine_blob_token_ratio"] >= 0.5
    assert metrics["machine_blob_char_ratio"] >= 0.5
    assert metrics["semantic_quality"] == "base64_or_hex_blob"

def test_text_quality_keeps_plain_discourse_as_semantic_text() -> None:
    text = "Mira wrote a garden note. The greenhouse fern state is healthy."
    metrics = text_quality_metrics(text)

    assert metrics["token_count"] >= 8
    assert metrics["low_semantic_noise"] is False
    assert metrics["semantic_quality"] == "meaningful_discourse"
    assert not is_low_semantic_noise(text)


def test_identifier_bearing_word_salad_is_not_skipped_for_model_semantics() -> None:
    text = "Alpha case names support ticket SUP-1207 and says client Bright Harbor requested review."
    metrics = text_quality_metrics(text)

    assert metrics["semantic_quality"] == "word_salad"
    assert _skip_model_semantics_for_quality(metrics, text) is False


def test_word_salad_without_structured_identifier_still_skips_model_semantics() -> None:
    text = "alpha bravo cedar delta ember frost garden harbor island juniper kelp lagoon"
    metrics = text_quality_metrics(text)

    assert metrics["semantic_quality"] == "word_salad"
    assert _skip_model_semantics_for_quality(metrics, text) is True

def test_bounded_source_low_priority_reuses_identical_text_quality(monkeypatch):
    calls = {"n": 0}
    original = bounded_dspg.text_quality_metrics

    def wrapped(text: str):
        calls["n"] += 1
        return original(text)

    bounded_dspg._SOURCE_LOW_PRIORITY_CACHE.clear()
    monkeypatch.setattr(bounded_dspg, "text_quality_metrics", wrapped)
    text = "Name: Alice. Status: approved."

    first = bounded_dspg._source_is_low_priority("a.json", text)
    second = bounded_dspg._source_is_low_priority("a.json", text)

    assert first == second
    assert calls["n"] == 1
    cache_key = next(iter(bounded_dspg._SOURCE_LOW_PRIORITY_CACHE))
    assert isinstance(cache_key, tuple)
    assert isinstance(cache_key[2], str)
    assert len(cache_key[2]) == 64

def test_bounded_contains_any_for_records_reuses_material_match_cache(monkeypatch) -> None:
    records: dict[str, object] = {}
    material = "ActionGenie database management logging system"
    calls = {"n": 0}
    original = bounded_dspg._contains_any

    def wrapped(value: str, terms: list[str]) -> bool:
        calls["n"] += 1
        return original(value, terms)

    monkeypatch.setattr(bounded_dspg, "_contains_any", wrapped)

    assert bounded_dspg._contains_any_for_records(records, material, ["ActionGenie"]) is True
    assert bounded_dspg._contains_any_for_records(records, material, ["ActionGenie"]) is True
    assert calls["n"] == 1
    cache_key = next(iter(records["_material_match_cache"]))
    assert isinstance(cache_key[1], str)
    assert len(cache_key[1]) == 64
