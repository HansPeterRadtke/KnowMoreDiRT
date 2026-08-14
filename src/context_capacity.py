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

from kmd_runtime_config import default_specs as _config_specs, explicit_raw as _config_explicit_raw


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
    names_tuple = tuple(names)
    specs = _config_specs()
    # Explicit environment/user-config overrides retain ordered precedence among
    # aliases. Packaged XML defaults are considered only after no explicit alias
    # is present, so a blank model-specific override can fall through to the
    # shared context setting exactly as before.
    for name in names_tuple:
        if name in specs:
            raw_value = _config_explicit_raw(name)
        else:
            raw_value = os.environ.get(name)
        raw = str(raw_value or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError(f"{name} must be a finite number") from error
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        return value
    for name in names_tuple:
        spec = specs.get(name)
        raw = str(spec.value if spec is not None else "").strip()
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





def contextualize_json_schema(
    schema: dict[str, Any],
    *,
    context_size: int,
    output_tokens: int | None = None,
) -> dict[str, Any]:
    """Return portable JSON Schema without KMD heuristic capacity annotations.

    Output size is bounded only by the model's actual remaining context capacity.
    Explicit source/contract bounds already present in the schema are preserved;
    KMD-specific profile annotations are removed rather than converted into
    guessed ``maxItems``/``maxLength`` ceilings.
    """

    require_context_size(context_size)
    del output_tokens
    result = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("x-kmd-string-profile", None)
            node.pop("x-kmd-array-profile", None)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result
