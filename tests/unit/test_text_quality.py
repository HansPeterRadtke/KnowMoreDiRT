from __future__ import annotations

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

    assert metrics["machine_blob_token_ratio"] >= 0.5
    assert metrics["low_semantic_noise"] is False
    assert metrics["semantic_quality"] == "base64_or_hex_blob"
    assert not is_low_semantic_noise(blob)


def test_text_quality_keeps_plain_discourse_as_semantic_text() -> None:
    text = "Mira wrote a garden note. The greenhouse fern state is healthy."
    metrics = text_quality_metrics(text)

    assert metrics["token_count"] >= 8
    assert metrics["low_semantic_noise"] is False
    assert metrics["semantic_quality"] == "meaningful_discourse"
    assert not is_low_semantic_noise(text)
