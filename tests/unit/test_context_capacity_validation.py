from __future__ import annotations

import pytest

from context_capacity import context_ratio, positive_float


def test_malformed_configured_ratio_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KMD_BAD_RATIO", "not-a-number")
    with pytest.raises(ValueError, match="KMD_BAD_RATIO must be a finite number"):
        context_ratio(("KMD_BAD_RATIO",), 0.5)


def test_nonfinite_configured_ratio_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KMD_BAD_RATIO", "nan")
    with pytest.raises(ValueError, match="finite number"):
        context_ratio(("KMD_BAD_RATIO",), 0.5)


def test_out_of_range_ratio_is_rejected_instead_of_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KMD_BAD_RATIO", "1.25")
    with pytest.raises(ValueError, match="between 0 and 1"):
        context_ratio(("KMD_BAD_RATIO",), 0.5)


def test_nonpositive_configured_scale_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KMD_BAD_SCALE", "0")
    with pytest.raises(ValueError, match="must be positive"):
        positive_float(("KMD_BAD_SCALE",), 3.0)


def test_deleted_output_ratio_environment_variable_has_no_capacity_api_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    # The old KMD_CONTEXT_OUTPUT_RATIO setting is deliberately not consumed by
    # production capacity code anymore. Generic ratio parsing still works for
    # legitimate input/retrieval settings.
    monkeypatch.setenv("KMD_CONTEXT_OUTPUT_RATIO", "0.99")
    assert context_ratio(("KMD_OTHER_INPUT_RATIO",), 0.5) == 0.5
