"""Context-relative budgeting helpers for local-model calls.

Every model-facing capacity is derived from the active model context.  The only
configuration values are dimensionless ratios.  No source or semantic item is
silently dropped to satisfy a fixed absolute cap.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable


CONTEXT_CAPACITY_POLICY = "context-relative-capacities-v1"


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
            value = float(raw)
        except ValueError as error:
            raise ValueError(f"{name} must be a finite number") from error
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        return value
    value = float(default)
    if not math.isfinite(value):
        raise ValueError("default capacity value must be finite")
    return value


def context_ratio(names: Iterable[str], default: float) -> float:
    names_tuple = tuple(names)
    value = _env_float(names_tuple, default)
    if not 0.0 <= value <= 1.0:
        label = names_tuple[0] if names_tuple else "context ratio"
        raise ValueError(f"{label} must be between 0 and 1")
    return value


def positive_float(names: Iterable[str], default: float) -> float:
    names_tuple = tuple(names)
    value = _env_float(names_tuple, default)
    if value <= 0.0:
        label = names_tuple[0] if names_tuple else "capacity value"
        raise ValueError(f"{label} must be positive")
    return value


def require_context_size(context_size: int) -> int:
    value = int(context_size)
    if value <= 0:
        raise ValueError("a positive model context size is required for every model-facing capacity")
    return value


def context_token_capacity(
    context_size: int,
    *,
    ratio_names: tuple[str, ...] = (),
    ratio_default: float,
    allow_zero: bool = False,
) -> int:
    context_size = require_context_size(context_size)
    ratio = context_ratio(ratio_names, ratio_default)
    value = int(math.floor(context_size * ratio))
    if allow_zero:
        return max(0, value)
    return max(1, value)


def context_char_capacity(
    context_size: int,
    *,
    ratio_names: tuple[str, ...] = (),
    ratio_default: float,
    chars_per_token_names: tuple[str, ...] = (),
    chars_per_token_default: float = 3.0,
) -> int:
    tokens = context_token_capacity(
        context_size,
        ratio_names=ratio_names,
        ratio_default=ratio_default,
    )
    chars_per_token = positive_float(
        (*chars_per_token_names, "KMD_CONTEXT_CHARS_PER_TOKEN"),
        chars_per_token_default,
    )
    return max(1, int(math.floor(tokens * chars_per_token)))


def context_safety_tokens(context_size: int) -> int:
    return context_token_capacity(
        context_size,
        ratio_names=("KMD_LOCAL_MODEL_CONTEXT_SAFETY_RATIO", "KMD_CONTEXT_SAFETY_RATIO"),
        ratio_default=0.02,
    )


def context_relative_budget(
    context_size: int,
    *,
    output_ratio_names: tuple[str, ...] = (),
    safety_ratio_names: tuple[str, ...] = (),
    overhead_ratio_names: tuple[str, ...] = (),
    chars_per_token_names: tuple[str, ...] = (),
    output_ratio_default: float = 0.25,
    safety_ratio_default: float = 0.02,
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
            chars_per_token=positive_float(
                (*chars_per_token_names, "KMD_CONTEXT_CHARS_PER_TOKEN"),
                chars_per_token_default,
            ),
            safe_input_chars=0,
        )
    output_ratio = context_ratio((*output_ratio_names, "KMD_CONTEXT_OUTPUT_RATIO"), output_ratio_default)
    safety_ratio = context_ratio((*safety_ratio_names, "KMD_CONTEXT_SAFETY_RATIO"), safety_ratio_default)
    overhead_ratio = context_ratio((*overhead_ratio_names, "KMD_CONTEXT_OVERHEAD_RATIO"), overhead_ratio_default)
    reserved_ratio = output_ratio + safety_ratio + overhead_ratio
    if reserved_ratio >= 1.0 - 1e-12:
        raise ValueError(
            "context output, safety, and overhead ratios must sum to less than 1"
        )
    chars_per_token = positive_float((*chars_per_token_names, "KMD_CONTEXT_CHARS_PER_TOKEN"), chars_per_token_default)
    output_tokens = int(math.floor(context_size * output_ratio))
    safety_tokens = int(math.floor(context_size * safety_ratio))
    overhead_tokens = int(math.floor(context_size * overhead_ratio))
    reserved = output_tokens + safety_tokens + overhead_tokens
    safe_input_tokens = max(0, context_size - reserved)
    safe_input_chars = int(math.floor(safe_input_tokens * chars_per_token))
    return ContextBudget(
        context_size=context_size,
        output_tokens=max(1, output_tokens) if output_ratio > 0.0 else 0,
        safety_margin_tokens=max(1, safety_tokens) if safety_ratio > 0.0 else 0,
        fixed_overhead_tokens=max(1, overhead_tokens) if overhead_ratio > 0.0 else 0,
        safe_input_tokens=safe_input_tokens,
        chars_per_token=chars_per_token,
        safe_input_chars=safe_input_chars,
    )


# Profiles are dimensionless multipliers. String profiles scale with sqrt(context)
# and array profiles with sqrt(stage output), so both dimensions grow while their
# product remains proportional to the available output budget.
_STRING_PROFILE_RATIO_DEFAULTS: dict[str, float] = {
    "id": 0.25,
    "short": 0.375,
    "label": 0.625,
    "value": 1.0,
    "evidence": 2.0,
    "reason": 2.0,
}
_ARRAY_PROFILE_RATIO_DEFAULTS: dict[str, float] = {
    "arguments": 0.0625,
    "compact": 0.5,
    "standard": 0.25,
    "dense": 0.5,
}


def schema_string_capacity(context_size: int, profile: str) -> int:
    context_size = require_context_size(context_size)
    default = _STRING_PROFILE_RATIO_DEFAULTS.get(profile)
    if default is None:
        raise ValueError(f"unknown schema string profile: {profile}")
    ratio = positive_float((f"KMD_SCHEMA_{profile.upper()}_SQRT_CONTEXT_RATIO",), default)
    return max(1, int(math.floor(math.sqrt(context_size) * ratio)))


def schema_array_capacity(output_tokens: int, profile: str) -> int:
    output_tokens = max(1, int(output_tokens))
    default = _ARRAY_PROFILE_RATIO_DEFAULTS.get(profile)
    if default is None:
        raise ValueError(f"unknown schema array profile: {profile}")
    ratio = positive_float((f"KMD_SCHEMA_{profile.upper()}_SQRT_OUTPUT_RATIO",), default)
    return max(1, int(math.floor(math.sqrt(output_tokens) * ratio)))


def contextualize_json_schema(
    schema: dict[str, Any],
    *,
    context_size: int,
    output_tokens: int,
) -> dict[str, Any]:
    """Resolve KMD relational schema profiles into native JSON Schema bounds."""

    context_size = require_context_size(context_size)
    output_tokens = max(1, int(output_tokens))
    result = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            string_profile = node.pop("x-kmd-string-profile", None)
            array_profile = node.pop("x-kmd-array-profile", None)
            if string_profile is not None:
                derived = schema_string_capacity(context_size, str(string_profile))
                existing = node.get("maxLength")
                node["maxLength"] = min(int(existing), derived) if isinstance(existing, int) else derived
            if array_profile is not None:
                derived = schema_array_capacity(output_tokens, str(array_profile))
                existing = node.get("maxItems")
                node["maxItems"] = min(int(existing), derived) if isinstance(existing, int) else derived
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result
