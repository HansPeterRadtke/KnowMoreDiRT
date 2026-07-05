"""Context-relative budgeting helpers for local-model calls."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ContextBudget:
    context_size: int
    output_tokens: int
    safety_margin_tokens: int
    fixed_overhead_tokens: int
    safe_input_tokens: int
    chars_per_token: float
    safe_input_chars: int


def _env_float(names: Iterable[str], default: float) -> float:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return float(default)


def context_ratio(names: Iterable[str], default: float) -> float:
    value = _env_float(names, default)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def positive_float(names: Iterable[str], default: float) -> float:
    value = _env_float(names, default)
    return value if value > 0.0 else float(default)


def context_relative_budget(
    context_size: int,
    *,
    output_ratio_names: tuple[str, ...] = (),
    safety_ratio_names: tuple[str, ...] = (),
    overhead_ratio_names: tuple[str, ...] = (),
    chars_per_token_names: tuple[str, ...] = (),
    output_ratio_default: float = 0.25,
    safety_ratio_default: float = 0.05,
    overhead_ratio_default: float = 0.03,
    chars_per_token_default: float = 3.0,
) -> ContextBudget:
    context_size = max(0, int(context_size))
    if context_size <= 0:
        return ContextBudget(
            context_size=0,
            output_tokens=0,
            safety_margin_tokens=0,
            fixed_overhead_tokens=0,
            safe_input_tokens=0,
            chars_per_token=positive_float((*chars_per_token_names, "KMD_CONTEXT_CHARS_PER_TOKEN"), chars_per_token_default),
            safe_input_chars=0,
        )
    output_ratio = context_ratio((*output_ratio_names, "KMD_CONTEXT_OUTPUT_RATIO"), output_ratio_default)
    safety_ratio = context_ratio((*safety_ratio_names, "KMD_CONTEXT_SAFETY_RATIO"), safety_ratio_default)
    overhead_ratio = context_ratio((*overhead_ratio_names, "KMD_CONTEXT_OVERHEAD_RATIO"), overhead_ratio_default)
    chars_per_token = positive_float((*chars_per_token_names, "KMD_CONTEXT_CHARS_PER_TOKEN"), chars_per_token_default)
    output_tokens = int(round(context_size * output_ratio))
    safety_tokens = int(round(context_size * safety_ratio))
    overhead_tokens = int(round(context_size * overhead_ratio))
    reserved = output_tokens + safety_tokens + overhead_tokens
    safe_input_tokens = max(0, context_size - reserved)
    safe_input_chars = int(round(safe_input_tokens * chars_per_token))
    return ContextBudget(
        context_size=context_size,
        output_tokens=max(1, output_tokens) if output_ratio > 0.0 else 0,
        safety_margin_tokens=safety_tokens,
        fixed_overhead_tokens=overhead_tokens,
        safe_input_tokens=safe_input_tokens,
        chars_per_token=chars_per_token,
        safe_input_chars=safe_input_chars,
    )
