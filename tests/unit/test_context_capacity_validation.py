from __future__ import annotations

import pytest

from context_capacity import context_ratio, context_relative_budget, positive_float


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


def test_reserved_ratios_must_leave_model_input_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KMD_CONTEXT_OUTPUT_RATIO", "0.7")
    monkeypatch.setenv("KMD_CONTEXT_SAFETY_RATIO", "0.2")
    monkeypatch.setenv("KMD_CONTEXT_OVERHEAD_RATIO", "0.1")

    with pytest.raises(ValueError, match="sum to less than 1"):
        context_relative_budget(65536)


def test_valid_context_budget_remains_context_relative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KMD_CONTEXT_OUTPUT_RATIO", "0.25")
    monkeypatch.setenv("KMD_CONTEXT_SAFETY_RATIO", "0.02")
    monkeypatch.setenv("KMD_CONTEXT_OVERHEAD_RATIO", "0.03")

    budget = context_relative_budget(65536)

    assert budget.safe_input_tokens == 45876
    assert budget.output_tokens == 16384
    assert budget.safety_margin_tokens == 1310
    assert budget.fixed_overhead_tokens == 1966
