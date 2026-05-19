from __future__ import annotations

from knowmoredirt.text import is_low_semantic_noise, text_quality_metrics


def test_text_quality_flags_random_character_noise() -> None:
    noise = "\\x00\\x01@@@###%%%^^^^~~~~" + ("A7f!?" * 80)
    metrics = text_quality_metrics(noise)

    assert metrics["char_count"] > 100
    assert metrics["low_semantic_noise"] is True
    assert is_low_semantic_noise(noise)


def test_text_quality_keeps_plain_discourse_as_semantic_text() -> None:
    text = "Mira wrote a garden note. The greenhouse fern state is healthy."
    metrics = text_quality_metrics(text)

    assert metrics["token_count"] >= 8
    assert metrics["low_semantic_noise"] is False
    assert not is_low_semantic_noise(text)
