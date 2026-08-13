"""Local-model integration hooks.

KMD never uses a cloud model. Normal runtime requires a reachable localhost
llama.cpp-compatible endpoint and returns raw, source-grounded JSON objects to
the engine. The public API remains ``initialize`` and ``question``.
"""

from __future__ import annotations

import hashlib
import math
import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

from kmd_model_call_cache import model_content_fingerprint, read_model_call, semantic_request_hash, write_model_call

from kmd_runtime_config import (
    boolean as _config_boolean,
    csv_integers as _config_csv_integers,
    default_specs as _config_specs,
    explicit_raw as _config_explicit_raw,
    floating as _config_float,
    integer as _config_int,
    optional_float as _config_optional_float,
    text as _config_text,
)

from .runtime_logging import get_logger
from .context_budget import (
    CONTEXT_CAPACITY_POLICY,
    context_relative_budget,
    context_safety_tokens,
    contextualize_json_schema,
)


LOGGER = get_logger("model")


def _server_root(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    for suffix in [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/models",
        "/models",
        "/completion",
        "/v1",
    ]:
        if value.endswith(suffix):
            root = value[: -len(suffix)]
            return root or value
    return value


def _completion_endpoint(endpoint: str) -> str:
    root = _server_root(endpoint)
    if endpoint.rstrip("/").endswith("/completion"):
        return endpoint.rstrip("/")
    return root + "/completion"


def _models_endpoint(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    if value.endswith("/completion"):
        return _server_root(value) + "/v1/models"
    if value.endswith("/models"):
        return value
    if value.endswith("/v1"):
        return value + "/models"
    return value + "/v1/models"


def _chat_endpoint(endpoint: str) -> str | None:
    value = endpoint.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return _server_root(value) + "/v1/chat/completions"


def _local_endpoint_required(endpoint: str) -> None:
    if not (
        endpoint.startswith("http://127.0.0.1:")
        or endpoint.startswith("http://localhost:")
        or endpoint.startswith("http://[::1]:")
    ):
        raise ValueError("KMD local model endpoint must be localhost-only")



def _default_per_token_timeout_seconds() -> float:
    value = _env_float("KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS", 180.0)
    return value if value > 0 else 180.0


def _default_context_safety_tokens(context_size: int) -> int:
    return context_safety_tokens(context_size)


def _configured_or_fallback(name: str, default: Any) -> str:
    specs = _config_specs()
    if name in specs:
        explicit = _config_explicit_raw(name)
        if explicit is not None:
            return explicit
        packaged = str(specs[name].value or "")
        if packaged.strip():
            return packaged
    return str(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(_configured_or_fallback(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_configured_or_fallback(name, default))
    except ValueError:
        return default


def _env_true(name: str, default: str | bool = "0") -> bool:
    raw_default = "1" if default is True else "0" if default is False else str(default)
    return _configured_or_fallback(name, raw_default).strip().lower() in {"1", "true", "yes", "on"}


_MODEL_THROUGHPUT_OBSERVATIONS: deque[dict[str, float]] = deque(maxlen=128)


def _metric_int(obj: Any, keys: set[str]) -> int:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in keys:
                value_int = _first_int(value)
                if value_int:
                    return value_int
            nested = _metric_int(value, keys)
            if nested:
                return nested
    elif isinstance(obj, list):
        for item in obj:
            nested = _metric_int(item, keys)
            if nested:
                return nested
    return 0


def _metric_float(obj: Any, keys: set[str]) -> float:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in keys:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = 0.0
                if number > 0.0:
                    return number
            nested = _metric_float(value, keys)
            if nested > 0.0:
                return nested
    elif isinstance(obj, list):
        for item in obj:
            nested = _metric_float(item, keys)
            if nested > 0.0:
                return nested
    return 0.0


def _estimated_output_tokens(text: str) -> int:
    if not text:
        return 0
    chars_per_token = _env_float("KMD_MODEL_THROUGHPUT_CHARS_PER_TOKEN", 3.0)
    if chars_per_token <= 0:
        chars_per_token = 3.0
    return max(1, int(math.ceil(len(text) / chars_per_token)))


def _model_throughput_observation(response_obj: dict[str, Any], raw: str, elapsed_seconds: float) -> dict[str, Any]:
    completion_tokens = _metric_int(
        response_obj,
        {"tokens_predicted", "completion_tokens", "predicted_n", "eval_count", "n_decoded", "n_predict"},
    )
    token_source = "server"
    if not completion_tokens:
        completion_tokens = _estimated_output_tokens(raw)
        token_source = "estimated_from_output_chars" if completion_tokens else "unavailable"
    prompt_tokens = _metric_int(response_obj, {"tokens_evaluated", "prompt_tokens", "prompt_n", "prompt_eval_count"})
    server_tps = _metric_float(
        response_obj,
        {"predicted_per_second", "eval_per_second", "tokens_per_second", "tok_per_second", "tps"},
    )
    completion_tps = server_tps if server_tps > 0.0 else (completion_tokens / elapsed_seconds if completion_tokens and elapsed_seconds > 0.0 else 0.0)
    observation = {
        "elapsed_seconds": round(elapsed_seconds, 3),
        "completion_tokens": int(completion_tokens),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens_per_second": round(completion_tps, 3),
        "token_source": token_source,
        "server_reported_tokens_per_second": round(server_tps, 3) if server_tps > 0.0 else 0.0,
    }
    window = max(1, _env_int("KMD_MODEL_THROUGHPUT_WINDOW", 20))
    if completion_tps > 0.0 and completion_tokens > 0:
        _MODEL_THROUGHPUT_OBSERVATIONS.append({
            "completion_tokens": float(completion_tokens),
            "prompt_tokens": float(prompt_tokens),
            "elapsed_seconds": float(elapsed_seconds),
            "completion_tokens_per_second": float(completion_tps),
        })
    observed = list(_MODEL_THROUGHPUT_OBSERVATIONS)[-window:]
    if observed:
        total_tokens = sum(item["completion_tokens"] for item in observed)
        total_elapsed = sum(item["elapsed_seconds"] for item in observed)
        avg_tps = total_tokens / total_elapsed if total_elapsed > 0.0 else 0.0
        observation["rolling_window"] = len(observed)
        observation["rolling_completion_tokens"] = int(total_tokens)
        observation["rolling_elapsed_seconds"] = round(total_elapsed, 3)
        observation["rolling_completion_tokens_per_second"] = round(avg_tps, 3)
    return observation


def _log_model_throughput(observation: dict[str, Any], *, endpoint: str, context_size: int, effective_n_predict: int) -> None:
    if not observation.get("completion_tokens"):
        return
    default_enabled = "1" if _env_true("KMD_PROGRESS") or _env_true("KMD_EVAL_PROGRESS") else "0"
    if not _env_true("KMD_MODEL_THROUGHPUT_LOG", default_enabled):
        return
    every = max(1, _env_int("KMD_MODEL_THROUGHPUT_LOG_EVERY", 1))
    window = int(observation.get("rolling_window") or 0)
    if window and window % every:
        return
    print(
        "kmd-model throughput "
        f"tokens={observation.get('completion_tokens')} "
        f"tps={float(observation.get('completion_tokens_per_second') or 0.0):.2f} "
        f"avg_window={observation.get('rolling_window', 0)} "
        f"avg_tps={float(observation.get('rolling_completion_tokens_per_second') or 0.0):.2f} "
        f"avg_tokens={observation.get('rolling_completion_tokens', 0)} "
        f"elapsed={float(observation.get('elapsed_seconds') or 0.0):.2f}s "
        f"prompt_tokens={observation.get('prompt_tokens', 0)} "
        f"token_source={observation.get('token_source')} "
        f"ctx={context_size} "
        f"n_predict={effective_n_predict} "
        f"endpoint={endpoint}",
        flush=True,
    )


def _first_int(*values: Any) -> int:
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return 0


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""




def _default_control_timeout_seconds() -> float:
    value = _config_float("KMD_LOCAL_MODEL_CONTROL_TIMEOUT_SECONDS")
    if value <= 0:
        raise ValueError("KMD_LOCAL_MODEL_CONTROL_TIMEOUT_SECONDS must be a positive number")
    return value


def _stream_total_timeout_seconds(*, per_token_timeout_seconds: float, max_tokens: int) -> float:
    configured = _config_optional_float("KMD_LOCAL_MODEL_STREAM_TOTAL_TIMEOUT_SECONDS")
    if configured is not None:
        if configured <= 0:
            raise ValueError("KMD_LOCAL_MODEL_STREAM_TOTAL_TIMEOUT_SECONDS must be a positive number")
        return configured
    return max(60.0, min(21600.0, float(per_token_timeout_seconds) * max(1, int(max_tokens))))


def _stream_event_limit(max_tokens: int) -> int:
    multiplier = _config_int("KMD_LOCAL_MODEL_STREAM_EVENT_MULTIPLIER")
    if multiplier < 1:
        raise ValueError("KMD_LOCAL_MODEL_STREAM_EVENT_MULTIPLIER must be positive")
    return max(64, int(max_tokens) * multiplier + 64)


def _stream_byte_limit(max_tokens: int) -> int:
    multiplier = _config_int("KMD_LOCAL_MODEL_STREAM_BYTES_PER_TOKEN")
    if multiplier < 1:
        raise ValueError("KMD_LOCAL_MODEL_STREAM_BYTES_PER_TOKEN must be positive")
    return max(65536, int(max_tokens) * multiplier + 65536)

def _retry_attempts(env_name: str, default: int) -> int:
    try:
        value = int(_configured_or_fallback(env_name, default))
    except ValueError as error:
        raise ValueError(f"{env_name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{env_name} must be a positive integer")
    return value


def _retry_backoff_seconds(env_name: str, default: float) -> float:
    raw = _configured_or_fallback(env_name, default).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{env_name} must be a non-negative number") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{env_name} must be a non-negative number")
    return value


def _retry_backoff_multiplier() -> float:
    raw = _config_text("KMD_LOCAL_MODEL_RETRY_BACKOFF_MULTIPLIER").strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError("KMD_LOCAL_MODEL_RETRY_BACKOFF_MULTIPLIER must be at least 1") from error
    if not math.isfinite(value) or value < 1:
        raise ValueError("KMD_LOCAL_MODEL_RETRY_BACKOFF_MULTIPLIER must be at least 1")
    return value


def _retry_http_statuses() -> set[int]:
    values = set(_config_csv_integers("KMD_LOCAL_MODEL_RETRY_HTTP_STATUSES"))
    if any(code < 100 or code > 599 for code in values):
        raise ValueError("KMD_LOCAL_MODEL_RETRY_HTTP_STATUSES contains an invalid HTTP status")
    return values


def _retryable_transport_exception(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return int(exc.code) in _retry_http_statuses()
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError))


def _retry_delay_seconds(attempt_index: int, *, env_name: str, default: float) -> float:
    base = _retry_backoff_seconds(env_name, default)
    if base <= 0:
        return 0.0
    return base * (_retry_backoff_multiplier() ** max(0, int(attempt_index)))


def _control_json_request(request_or_url: Any, *, timeout: float) -> Any:
    attempts = _retry_attempts("KMD_LOCAL_MODEL_CONTROL_RETRY_ATTEMPTS", 3)
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request_or_url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            if not _retryable_transport_exception(exc) or attempt + 1 >= attempts:
                raise
            last_error = exc
            delay = _retry_delay_seconds(
                attempt,
                env_name="KMD_LOCAL_MODEL_CONTROL_RETRY_BACKOFF_SECONDS",
                default=0.25,
            )
            LOGGER.warning(
                "model_control_retry attempt=%s/%s error=%s delay_seconds=%g",
                attempt + 1,
                attempts,
                f"{type(exc).__name__}: {exc}",
                delay,
            )
            if delay > 0:
                time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("control request retry loop exhausted without an attempt")


def _fetch_json(url: str, *, timeout: float | None = None) -> Any:
    _local_endpoint_required(url)
    effective_timeout = _default_control_timeout_seconds() if timeout is None else float(timeout)
    if effective_timeout <= 0:
        raise ValueError("control timeout must be positive")
    return _control_json_request(url, timeout=effective_timeout)


def _post_json(url: str, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
    _local_endpoint_required(url)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    effective_timeout = _default_control_timeout_seconds() if timeout is None else float(timeout)
    if effective_timeout <= 0:
        raise ValueError("control timeout must be positive")
    return _control_json_request(request, timeout=effective_timeout)


def _response_content(response_obj: dict[str, Any]) -> str:
    raw = str(response_obj.get("content") or "")
    if raw:
        return raw
    choices = response_obj.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                return str(message.get("content") or "")
            return str(choice.get("text") or "")
    return ""


def _event_content(event: dict[str, Any]) -> str | None:
    if "content" in event:
        return str(event.get("content") or "")
    choices = event.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            delta = choice.get("delta")
            if isinstance(delta, dict):
                return str(delta.get("content") or "")
            message = choice.get("message")
            if isinstance(message, dict):
                return str(message.get("content") or "")
            return str(choice.get("text") or "")
    return None



def _event_reasoning_content(event: dict[str, Any]) -> str:
    choices = event.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            delta = choice.get("delta")
            if isinstance(delta, dict):
                return str(delta.get("reasoning_content") or "")
            message = choice.get("message")
            if isinstance(message, dict):
                return str(message.get("reasoning_content") or "")
    return ""


def _model_id_looks_like_reasoning_control_token_model(model_id: str) -> bool:
    normalized = model_id.lower()
    compact = normalized.replace("-", "").replace("_", "")
    return (
        ("gpt" in compact and "oss" in compact)
        or "harmony" in compact
        or "qwen35" in compact
        or "qwen3.5" in normalized
    )


def _schema_hint(json_schema: dict[str, Any] | None) -> str:
    if not json_schema:
        return ""
    try:
        rendered = json.dumps(json_schema, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return ""
    if len(rendered) > 6000:
        rendered = rendered[:6000] + " ...TRUNCATED_SCHEMA"
    return (
        "\nReturn JSON that validates against this schema. "
        "All required fields must be present. Do not add prose, markdown, comments, "
        "reasoning, or fields outside the schema.\nSCHEMA:\n"
        + rendered
    )


def _json_only_user_prompt(
    prompt: str,
    json_schema: dict[str, Any] | None,
    *,
    include_schema_hint: bool,
) -> str:
    return (
        prompt.rstrip()
        + (_schema_hint(json_schema) if include_schema_hint else "")
        + "\nOutput exactly one complete JSON object or JSON array as the final answer. "
        "Do not include analysis, chain of thought, markdown fences, or explanatory text."
    )


def _extract_balanced_json(raw: str) -> str | None:
    object_start = raw.find("{")
    array_start = raw.find("[")
    candidates = [index for index in [object_start, array_start] if index >= 0]
    if not candidates:
        return None
    start = min(candidates)
    opener = raw[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(raw[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    return None


def validate_portable_json_schema(schema: dict[str, Any]) -> None:
    """Validate the strict llama.cpp Structured Outputs subset used by KMD."""

    forbidden = {
        "const",
        "maximum",
        "minimum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "patternProperties",
    }

    def visit(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            raise ValueError(f"JSON schema node at {path} must be an object")
        bad = forbidden.intersection(node)
        if bad:
            raise ValueError(f"nonportable JSON schema keywords at {path}: {sorted(bad)}")
        for minimum_key, maximum_key in (("minLength", "maxLength"), ("minItems", "maxItems")):
            minimum = node.get(minimum_key)
            maximum = node.get(maximum_key)
            if minimum is not None and (
                not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0
            ):
                raise ValueError(f"{minimum_key} at {path} must be a non-negative integer")
            if maximum is not None and (
                not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0
            ):
                raise ValueError(f"{maximum_key} at {path} must be a non-negative integer")
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(f"{minimum_key} exceeds {maximum_key} at {path}")
        node_type = node.get("type")
        if node_type == "object":
            properties = node.get("properties")
            required = node.get("required")
            if not isinstance(properties, dict):
                raise ValueError(f"object schema at {path} requires properties")
            if node.get("additionalProperties") is not False:
                raise ValueError(f"object schema at {path} requires additionalProperties=false")
            if not isinstance(required, list) or set(required) != set(properties):
                raise ValueError(f"object schema at {path} must require every property")
            for key, child in properties.items():
                visit(child, f"{path}.properties.{key}")
        elif node_type == "array":
            if "items" not in node:
                raise ValueError(f"array schema at {path} requires items")
            visit(node["items"], f"{path}.items")
        elif "anyOf" in node:
            variants = node.get("anyOf")
            if not isinstance(variants, list) or not variants:
                raise ValueError(f"anyOf at {path} must be a non-empty list")
            for index, child in enumerate(variants):
                visit(child, f"{path}.anyOf[{index}]")
        elif node_type not in {"string", "number", "integer", "boolean", "null"}:
            raise ValueError(f"unsupported portable JSON schema type at {path}: {node_type!r}")

    if schema.get("type") != "object":
        raise ValueError("portable constrained decoding requires an object root schema")
    visit(schema, "root")
    Draft202012Validator.check_schema(schema)


class LocalModelJSONError(ValueError):
    """Raised when a structured generation is incomplete or invalid."""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str,
        snippet: str,
        reason: str = "invalid_json",
        response_metadata: dict[str, Any] | None = None,
        model_input_audit: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.snippet = snippet
        self.reason = reason
        self.response_metadata = response_metadata or {}
        self.model_input_audit = model_input_audit or {}


class LocalModelUnavailableError(RuntimeError):
    """Raised when required localhost llama.cpp access is unavailable."""

    def __init__(self, message: str, *, cache_context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.cache_context = cache_context or {}


class LocalModelContextError(LocalModelUnavailableError):
    """Raised before generation when prompt plus output reserve cannot fit."""


def complete_json_with_transport_retry(
    client: Any,
    prompt: str,
    *,
    n_predict: int,
    json_schema: dict[str, Any],
) -> dict[str, Any]:
    """Retry only transient transport failures for direct semantic callers.

    Planner-owned semantic calls already have higher-level retry policies and do
    not use this helper.  This wrapper exists for direct callers such as the
    long-range document-context classifier and semantic evaluator so a temporary
    localhost disconnect does not make the whole operation terminal.
    """

    attempts = _retry_attempts("KMD_LOCAL_MODEL_DIRECT_RETRY_ATTEMPTS", 3)
    transient_json_reasons = {
        "incomplete_stream",
        "stream_total_timeout_exhausted",
    }
    for attempt in range(attempts):
        retry_error: BaseException | None = None
        try:
            return client.complete_json(prompt, n_predict=n_predict, json_schema=json_schema)
        except LocalModelJSONError as exc:
            retryable = str(exc.reason or "") in transient_json_reasons
            if not retryable or attempt + 1 >= attempts:
                raise
            retry_error = exc
        except Exception as exc:
            if not _retryable_transport_exception(exc) or attempt + 1 >= attempts:
                raise
            retry_error = exc
        delay = _retry_delay_seconds(
            attempt,
            env_name="KMD_LOCAL_MODEL_DIRECT_RETRY_BACKOFF_SECONDS",
            default=0.25,
        )
        LOGGER.warning(
            "model_direct_retry attempt=%s/%s error=%s delay_seconds=%g",
            attempt + 1,
            attempts,
            f"{type(retry_error).__name__}: {retry_error}",
            delay,
        )
        if delay > 0:
            time.sleep(delay)
    raise RuntimeError("direct model retry loop exhausted without an attempt")


@dataclass
class LocalModelClient:
    endpoint: str = field(default_factory=lambda: _config_text("KMD_LOCAL_MODEL_ENDPOINT"))
    # Socket/read timeout between streamed token chunks. Not a whole-answer wall timeout.
    per_token_timeout_seconds: float = field(default_factory=_default_per_token_timeout_seconds)
    _metadata: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def models(self) -> dict:
        return _fetch_json(_models_endpoint(self.endpoint))

    def server_metadata(self, *, refresh: bool = False) -> dict[str, Any]:
        """Best-effort llama.cpp runtime metadata used for budgeting and cache keys."""

        if self._metadata is not None and not refresh:
            return self._metadata
        root = _server_root(self.endpoint)
        metadata: dict[str, Any] = {"endpoint": self.endpoint, "root": root, "errors": {}}
        for name, path in {
            "models": "/v1/models",
            "slots": "/slots",
            "props": "/props",
        }.items():
            try:
                metadata[name] = _fetch_json(root + path)
            except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
                metadata["errors"][name] = f"{type(exc).__name__}: {exc}"
        metadata["derived"] = {
            "model_id": self.model_id(metadata),
            "context_size": self.context_size(metadata),
            "context_source": self.context_source(metadata),
        }
        self._metadata = metadata
        return metadata

    def context_source(self, metadata: dict[str, Any] | None = None) -> str:
        data = metadata or self._metadata or self.server_metadata()
        slots = data.get("slots")
        if isinstance(slots, list) and slots and _first_int(slots[0].get("n_ctx")):
            return "/slots[0].n_ctx"
        props = data.get("props")
        if isinstance(props, dict):
            settings = props.get("default_generation_settings")
            if isinstance(settings, dict) and _first_int(settings.get("n_ctx")):
                return "/props.default_generation_settings.n_ctx"
        models = data.get("models")
        if isinstance(models, dict):
            first = (models.get("data") or [{}])[0] if isinstance(models.get("data"), list) and models.get("data") else {}
            meta = first.get("meta") if isinstance(first, dict) else {}
            if isinstance(meta, dict) and _first_int(meta.get("n_ctx"), meta.get("n_ctx_train")):
                return "/v1/models.data[0].meta"
        if _first_int(_config_text("KMD_LOCAL_MODEL_CONTEXT_SIZE")):
            return "KMD_LOCAL_MODEL_CONTEXT_SIZE"
        return "unavailable"

    def context_size(self, metadata: dict[str, Any] | None = None) -> int:
        data = metadata or self._metadata or self.server_metadata()
        slots = data.get("slots")
        if isinstance(slots, list) and slots:
            slot_value = _first_int(slots[0].get("n_ctx"))
            if slot_value:
                return slot_value
        props = data.get("props")
        if isinstance(props, dict):
            settings = props.get("default_generation_settings")
            if isinstance(settings, dict):
                prop_value = _first_int(settings.get("n_ctx"))
                if prop_value:
                    return prop_value
        models = data.get("models")
        if isinstance(models, dict):
            first = (models.get("data") or [{}])[0] if isinstance(models.get("data"), list) and models.get("data") else {}
            meta = first.get("meta") if isinstance(first, dict) else {}
            if isinstance(meta, dict):
                model_value = _first_int(meta.get("n_ctx"), meta.get("n_ctx_train"))
                if model_value:
                    return model_value
        return _first_int(_config_text("KMD_LOCAL_MODEL_CONTEXT_SIZE"))

    def model_id(self, metadata: dict[str, Any] | None = None) -> str:
        data = metadata or self._metadata or self.server_metadata()
        models = data.get("models")
        if isinstance(models, dict):
            first = (models.get("data") or [{}])[0] if isinstance(models.get("data"), list) and models.get("data") else {}
            if isinstance(first, dict):
                found = _first_text(first.get("id"), first.get("model"), first.get("name"))
                if found:
                    return found
            first_model = (models.get("models") or [{}])[0] if isinstance(models.get("models"), list) and models.get("models") else {}
            if isinstance(first_model, dict):
                found = _first_text(first_model.get("model"), first_model.get("name"))
                if found:
                    return found
        props = data.get("props")
        if isinstance(props, dict):
            found = _first_text(props.get("model_alias"), props.get("model_path"))
            if found:
                return found
        return _first_text(_config_text("KMD_LOCAL_MODEL_ID"), self.endpoint, "local-llama")

    def default_generation_params(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        data = metadata or self._metadata or self.server_metadata()
        params: dict[str, Any] = {}
        props = data.get("props")
        if isinstance(props, dict):
            settings = props.get("default_generation_settings")
            if isinstance(settings, dict) and isinstance(settings.get("params"), dict):
                params.update(settings["params"])
        slots = data.get("slots")
        if isinstance(slots, list) and slots and isinstance(slots[0], dict) and isinstance(slots[0].get("params"), dict):
            params.update(slots[0]["params"])
        return params

    def request_settings(self) -> dict[str, Any]:
        defaults = self.default_generation_params()
        top_k_override = _config_explicit_raw("KMD_LOCAL_MODEL_TOP_K")
        min_p_override = _config_explicit_raw("KMD_LOCAL_MODEL_MIN_P")
        repeat_override = _config_explicit_raw("KMD_LOCAL_MODEL_REPEAT_PENALTY")
        return {
            "seed": _config_int("KMD_LOCAL_MODEL_SEED"),
            "temperature": _config_float("KMD_LOCAL_MODEL_TEMPERATURE"),
            "top_p": _config_float("KMD_LOCAL_MODEL_TOP_P"),
            "top_k": int(top_k_override) if str(top_k_override or "").strip() else (_first_int(defaults.get("top_k")) or 40),
            "min_p": float(min_p_override) if str(min_p_override or "").strip() else float(defaults.get("min_p") or 0.05),
            "repeat_penalty": float(repeat_override) if str(repeat_override or "").strip() else float(defaults.get("repeat_penalty") or 1.0),
        }

    def transport_settings(self) -> dict[str, Any]:
        model_id = self.model_id()
        constrained_mode = _config_text("KMD_LOCAL_MODEL_CONSTRAINT_MODE").strip().lower() or "auto"
        if constrained_mode not in {"auto", "native", "prompt"}:
            constrained_mode = "auto"
        reasoning_control_model = _model_id_looks_like_reasoning_control_token_model(model_id)
        native_constraints = constrained_mode != "prompt"
        thinking_control_override = str(_config_explicit_raw("KMD_LOCAL_MODEL_SEND_THINKING_CONTROLS") or "").strip().lower()
        return {
            "api": _config_text("KMD_LOCAL_MODEL_API").strip().lower() or "chat",
            "cache_prompt": _config_boolean("KMD_LOCAL_MODEL_CACHE_PROMPT"),
            "context_contract_policy": "exact-rendered-prompt-explicit-output-terminal-stream-v4",
            "capacity_policy": CONTEXT_CAPACITY_POLICY,
            "context_safety_ratio": _env_float(
                "KMD_LOCAL_MODEL_CONTEXT_SAFETY_RATIO",
                _config_float("KMD_CONTEXT_SAFETY_RATIO"),
            ),
            "context_safety_tokens": _default_context_safety_tokens(self.context_size()),
            "schema_bounds_native": True,
            "terminal_stream_required": True,
            "constraint_mode": constrained_mode,
            "native_constraints": native_constraints,
            "reasoning_control_token_model": reasoning_control_model,
            # The automatic behavior is already determined by model_id above.  Only an
            # explicit override is output-influencing state and therefore belongs in
            # the stable cache fingerprint.  Stream byte/event ceilings are safety
            # guards, not output semantics: raising them must preserve accepted cache
            # entries while allowing truncated failures to retry.
            "thinking_control_override": thinking_control_override or "auto",
        }

    def token_count(self, text: str) -> int:
        payload = _post_json(
            _server_root(self.endpoint) + "/tokenize",
            {"content": text, "add_special": False},
        )
        tokens = payload.get("tokens") if isinstance(payload, dict) else None
        if not isinstance(tokens, list):
            raise LocalModelUnavailableError("local tokenizer did not return a token list")
        return len(tokens)

    def rendered_prompt(self, endpoint: str, body: dict[str, Any]) -> str:
        if endpoint.endswith("/chat/completions"):
            payload: dict[str, Any] = {
                "messages": body.get("messages") or [],
                "add_generation_prompt": True,
            }
            if isinstance(body.get("chat_template_kwargs"), dict):
                payload["chat_template_kwargs"] = body["chat_template_kwargs"]
            rendered = _post_json(_server_root(self.endpoint) + "/apply-template", payload)
            prompt = rendered.get("prompt") if isinstance(rendered, dict) else None
            if not isinstance(prompt, str):
                raise LocalModelUnavailableError("local chat template endpoint did not return a prompt")
            return prompt
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise LocalModelUnavailableError("completion request did not contain a prompt")
        return prompt

    def exact_context_budget(
        self,
        endpoint: str,
        body: dict[str, Any],
        *,
        output_tokens: int,
    ) -> dict[str, int]:
        context_size = self.context_size()
        if context_size <= 0:
            raise LocalModelUnavailableError("local model context size is unavailable")
        rendered_prompt = self.rendered_prompt(endpoint, body)
        prompt_tokens = self.token_count(rendered_prompt)
        safety_tokens = _default_context_safety_tokens(context_size)
        available_output_tokens = context_size - prompt_tokens - safety_tokens
        return {
            "context_size": context_size,
            "prompt_tokens": prompt_tokens,
            "output_tokens": int(output_tokens),
            "safety_tokens": safety_tokens,
            "available_output_tokens": available_output_tokens,
            "total_reserved_tokens": prompt_tokens + int(output_tokens) + safety_tokens,
        }

    def semantic_transport_settings(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        data = metadata or self._metadata or self.server_metadata()
        props = data.get("props") if isinstance(data, dict) else {}
        chat_template = str(props.get("chat_template") or "") if isinstance(props, dict) else ""
        constrained_mode = _config_text("KMD_LOCAL_MODEL_CONSTRAINT_MODE").strip().lower() or "auto"
        if constrained_mode not in {"auto", "native", "prompt"}:
            constrained_mode = "auto"
        thinking_override = str(_config_explicit_raw("KMD_LOCAL_MODEL_SEND_THINKING_CONTROLS") or "").strip().lower() or "auto"
        return {
            "api": _config_text("KMD_LOCAL_MODEL_API").strip().lower() or "chat",
            "constraint_mode": constrained_mode,
            "thinking_control_override": thinking_override,
            "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8", errors="replace")).hexdigest() if chat_template else "",
        }

    def cache_fingerprint(self) -> dict[str, Any]:
        metadata = self.server_metadata()
        model_id = self.model_id(metadata)
        return {
            "fingerprint_schema": "local-model-semantic-v5",
            "model": model_content_fingerprint(model_id),
            "model_id": model_id,
            "context_size": self.context_size(metadata),
            "request_settings": self.request_settings(),
            "transport_settings": self.semantic_transport_settings(metadata),
        }

    def complete_json(
        self,
        prompt: str,
        *,
        n_predict: int | None = None,
        grammar: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from the local completion endpoint."""

        api = _config_text("KMD_LOCAL_MODEL_API").strip().lower()
        endpoint = _chat_endpoint(self.endpoint) if api == "chat" else _completion_endpoint(self.endpoint)
        if endpoint is None:
            endpoint = _completion_endpoint(self.endpoint)
        _local_endpoint_required(endpoint)
        settings = self.request_settings()
        if grammar is not None and json_schema is None:
            raise LocalModelUnavailableError(
                "Semantic model calls must use strict JSON Schema constrained decoding; grammar-only contracts are forbidden.",
                cache_context={"structured_call": True},
            )
        if json_schema is None:
            raise LocalModelUnavailableError(
                "Semantic model calls require a portable strict JSON Schema.",
                cache_context={"structured_call": True},
            )
        validate_portable_json_schema(json_schema)
        transport = self.transport_settings()
        native_constraints = bool(transport.get("native_constraints"))
        context_size = self.context_size()
        if context_size <= 0:
            raise LocalModelUnavailableError("local model context size is unavailable")
        if n_predict is None:
            requested_n_predict = context_relative_budget(context_size).output_tokens
        else:
            requested_n_predict = int(n_predict)
        if requested_n_predict <= 0:
            raise ValueError("n_predict must be positive")
        effective_n_predict = requested_n_predict
        json_schema = contextualize_json_schema(
            json_schema,
            context_size=context_size,
            output_tokens=effective_n_predict,
        )
        validate_portable_json_schema(json_schema)
        allow_prompt_constraints = _config_boolean("KMD_LOCAL_MODEL_ALLOW_PROMPT_CONSTRAINTS")
        if not native_constraints and not allow_prompt_constraints:
            raise LocalModelUnavailableError(
                "Structured local model calls require native constrained decoding. "
                "KMD_LOCAL_MODEL_CONSTRAINT_MODE=prompt is diagnostic-only; set "
                "KMD_LOCAL_MODEL_ALLOW_PROMPT_CONSTRAINTS=1 only for an explicit soft-JSON measurement run.",
                cache_context={"transport_settings": transport, "structured_call": True},
            )
        effective_prompt = _json_only_user_prompt(
            prompt,
            json_schema,
            include_schema_hint=not native_constraints,
        )
        thinking_control_env = _config_text("KMD_LOCAL_MODEL_SEND_THINKING_CONTROLS").strip().lower() or "auto"
        if thinking_control_env in {"0", "false", "no", "off"}:
            send_thinking_controls = False
        elif thinking_control_env in {"1", "true", "yes", "on"}:
            send_thinking_controls = True
        else:
            send_thinking_controls = bool(transport.get("reasoning_control_token_model"))
        # Local model calls must stream. The timeout below is only the socket/read
        # timeout while waiting for the next streamed token chunk. There is no
        # whole-answer, whole-question, or whole-chunk wall timeout here.
        use_cache_prompt = _config_boolean("KMD_LOCAL_MODEL_CACHE_PROMPT")
        if endpoint.endswith("/chat/completions"):
            body = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return exactly one valid JSON object or array in the final answer. "
                            "Do not reveal reasoning, do not use markdown fences, and do not add prose."
                        ),
                    },
                    {"role": "user", "content": effective_prompt},
                ],
                "max_tokens": effective_n_predict,
                "temperature": settings["temperature"],
                "top_p": settings["top_p"],
                "seed": settings["seed"],
                "stream": True,
                "cache_prompt": bool(use_cache_prompt),
            }
        else:
            body = {
                "prompt": effective_prompt,
                "n_predict": effective_n_predict,
                "temperature": settings["temperature"],
                "top_p": settings["top_p"],
                "top_k": settings["top_k"],
                "min_p": settings["min_p"],
                "repeat_penalty": settings["repeat_penalty"],
                "seed": settings["seed"],
                "stream": True,
                "cache_prompt": bool(use_cache_prompt),
            }
        constraint_settings: dict[str, Any] = {"mode": "none"}
        if native_constraints and json_schema:
            if endpoint.endswith("/chat/completions"):
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "kmd_response",
                        "strict": True,
                        "schema": json_schema,
                    },
                }
                constraint_settings = {"mode": "chat_response_format_json_schema"}
            else:
                body["json_schema"] = json_schema
                constraint_settings = {"mode": "completion_json_schema"}
        elif json_schema:
            constraint_settings = {"mode": "prompt_json_schema"}
        if "openrouter.ai" in endpoint:
            body["provider"] = {"require_parameters": True}
        if send_thinking_controls:
            body["enable_thinking"] = False
            qwen_thinking_model = "qwen3.5" in self.model_id().lower() or "qwen35" in self.model_id().lower().replace("-", "").replace("_", "")
            if qwen_thinking_model:
                body["reasoning_format"] = "deepseek"
                body["reasoning_budget"] = 0
            else:
                body["reasoning_format"] = "hidden"
            if endpoint.endswith("/chat/completions"):
                body["chat_template_kwargs"] = {"enable_thinking": False}
        context_budget = self.exact_context_budget(
            endpoint,
            body,
            output_tokens=effective_n_predict,
        )
        request_body_json = json.dumps(body)
        model_input_audit: dict[str, Any] = {
            "audit_schema": "kmd-model-input-v1",
            "endpoint": endpoint,
            "api": "chat" if endpoint.endswith("/chat/completions") else "completion",
            "request_body_json": request_body_json,
            "request_body_sha256": hashlib.sha256(request_body_json.encode("utf-8")).hexdigest(),
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest(),
            "effective_prompt": effective_prompt,
            "effective_prompt_sha256": hashlib.sha256(
                effective_prompt.encode("utf-8", errors="replace")
            ).hexdigest(),
            "request_settings": {
                **settings,
                "n_predict": requested_n_predict,
                "effective_n_predict": effective_n_predict,
            },
            "transport_settings": {
                **transport,
                "cache_prompt": bool(use_cache_prompt),
                "thinking_controls_sent": send_thinking_controls,
            },
            "constraint_settings": {
                **constraint_settings,
                "requested_n_predict": requested_n_predict,
                "effective_n_predict": effective_n_predict,
                "native_constraints_applied": native_constraints,
                "schema_prompt_hint": bool(json_schema and not native_constraints),
                "grammar_prompt_only": bool(grammar and not native_constraints),
            },
            "context_budget": {
                **context_budget,
                "policy": "exact-rendered-prompt-plus-explicit-output-plus-safety-v1",
            },
        }
        if context_budget["available_output_tokens"] < effective_n_predict:
            raise LocalModelContextError(
                "structured generation does not fit model context: "
                f"prompt={context_budget['prompt_tokens']} output={effective_n_predict} "
                f"safety={context_budget['safety_tokens']} context={context_budget['context_size']}",
                cache_context={"model_input_audit": model_input_audit},
            )
        semantic_body = {key: value for key, value in body.items() if key not in {"stream", "cache_prompt"}}
        model_call_request = {
            "cache_schema": "kmd-exact-model-request-v1",
            "model_fingerprint": self.cache_fingerprint(),
            "request": semantic_body,
        }
        model_call_hash = semantic_request_hash(model_call_request)
        cached_call = read_model_call(model_call_hash)
        if cached_call is not None:
            cached = dict(cached_call)
            cached["_model_elapsed_seconds"] = 0.0
            cached["_model_call_cache_hit"] = True
            cached["_model_call_cache_hash"] = model_call_hash
            cached["_model_input_audit"] = model_input_audit
            return cached
        request = urllib.request.Request(
            endpoint,
            data=request_body_json.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw = ""
        response_obj: dict[str, Any] = {}
        saw_done = False
        saw_terminal_event = False
        finish_reason = ""
        stop_reason = ""
        started = time.time()
        total_timeout_seconds = _stream_total_timeout_seconds(
            per_token_timeout_seconds=self.per_token_timeout_seconds,
            max_tokens=effective_n_predict,
        )
        event_limit = _stream_event_limit(effective_n_predict)
        byte_limit = _stream_byte_limit(effective_n_predict)
        stream_events = 0
        stream_bytes = 0
        try:
            with urllib.request.urlopen(request, timeout=self.per_token_timeout_seconds) as response:
                for raw_line in response:
                    stream_bytes += len(raw_line)
                    if stream_bytes > byte_limit:
                        raise LocalModelJSONError(
                            "structured generation exceeded the client stream byte limit",
                            raw_text=raw,
                            snippet=raw,
                            reason="stream_byte_limit_exhausted",
                            response_metadata={"stream_bytes": stream_bytes, "stream_byte_limit": byte_limit},
                            model_input_audit=model_input_audit,
                        )
                    if time.time() - started > total_timeout_seconds:
                        raise LocalModelJSONError(
                            "structured generation exceeded the client total stream timeout",
                            raw_text=raw,
                            snippet=raw,
                            reason="stream_total_timeout_exhausted",
                            response_metadata={"total_timeout_seconds": total_timeout_seconds},
                            model_input_audit=model_input_audit,
                        )
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        saw_done = True
                        break
                    stream_events += 1
                    if stream_events > event_limit:
                        raise LocalModelJSONError(
                            "structured generation exceeded the client stream event limit",
                            raw_text=raw,
                            snippet=raw,
                            reason="stream_event_limit_exhausted",
                            response_metadata={"stream_events": stream_events, "stream_event_limit": event_limit},
                            model_input_audit=model_input_audit,
                        )
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    response_obj = event
                    raw += _event_content(event) or ""
                    choices = event.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        found_finish = str(choices[0].get("finish_reason") or "").strip()
                        if found_finish:
                            finish_reason = found_finish
                            saw_terminal_event = True
                    found_stop = str(event.get("stop_reason") or "").strip()
                    if found_stop:
                        stop_reason = found_stop
                    if event.get("stop") is True or any(
                        event.get(key) is True
                        for key in ("stopped_eos", "stopped_word", "stopped_limit")
                    ):
                        saw_terminal_event = True
        except Exception as exc:
            try:
                setattr(exc, "model_input_audit", model_input_audit)
            except Exception:
                pass
            raise
        elapsed_seconds = round(time.time() - started, 3)
        throughput = _model_throughput_observation(response_obj, raw, elapsed_seconds)
        completion_tokens = int(throughput.get("completion_tokens") or 0)
        response_metadata = {
            "saw_done": saw_done,
            "saw_terminal_event": saw_terminal_event,
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "completion_tokens": completion_tokens,
            "prompt_tokens": int(throughput.get("prompt_tokens") or context_budget["prompt_tokens"]),
            "requested_output_tokens": effective_n_predict,
            "context_size": context_budget["context_size"],
            "context_safety_tokens": context_budget["safety_tokens"],
            "stream_events": stream_events,
            "stream_event_limit": event_limit,
            "stream_bytes": stream_bytes,
            "stream_byte_limit": byte_limit,
            "stream_total_timeout_seconds": total_timeout_seconds,
        }
        snippet = _extract_balanced_json(raw)
        if snippet is None:
            reason = "incomplete_stream"
            if finish_reason == "length" or response_obj.get("stopped_limit") is True:
                if completion_tokens >= effective_n_predict:
                    reason = "output_limit_exhausted"
                elif (
                    context_budget["prompt_tokens"]
                    + completion_tokens
                    + context_budget["safety_tokens"]
                    >= context_budget["context_size"]
                ):
                    reason = "context_limit_exhausted"
                else:
                    reason = "generation_limit_exhausted"
            elif (
                context_budget["prompt_tokens"]
                + completion_tokens
                + context_budget["safety_tokens"]
                >= context_budget["context_size"]
            ):
                reason = "context_limit_exhausted"
            raise LocalModelJSONError(
                f"structured generation ended without a complete JSON value ({reason})",
                raw_text=raw,
                snippet=raw,
                reason=reason,
                response_metadata=response_metadata,
                model_input_audit=model_input_audit,
            )
        if not (saw_done or saw_terminal_event):
            raise LocalModelJSONError(
                "structured generation stream ended without terminal completion metadata",
                raw_text=raw,
                snippet=snippet,
                reason="incomplete_stream",
                response_metadata=response_metadata,
                model_input_audit=model_input_audit,
            )
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise LocalModelJSONError(
                str(exc),
                raw_text=raw,
                snippet=snippet,
                reason="invalid_json",
                response_metadata=response_metadata,
                model_input_audit=model_input_audit,
            ) from exc
        if isinstance(parsed, list):
            parsed = {"items": parsed}
        if not isinstance(parsed, dict):
            raise ValueError("local model did not return a JSON object or array")
        try:
            Draft202012Validator(json_schema).validate(parsed)
        except JSONSchemaValidationError as exc:
            raise LocalModelJSONError(
                f"completed JSON failed response schema validation: {exc.message}",
                raw_text=raw,
                snippet=snippet,
                reason="schema_validation_failed",
                response_metadata=response_metadata,
                model_input_audit=model_input_audit,
            ) from exc
        context_size = context_budget["context_size"]
        parsed["_model_raw"] = raw
        parsed["_model_elapsed_seconds"] = elapsed_seconds
        parsed["_model_endpoint"] = endpoint
        parsed["_model_stream"] = True
        parsed["_model_per_token_timeout_seconds"] = self.per_token_timeout_seconds
        parsed["_model_stream_closed_after_json"] = False
        parsed["_model_response_metadata"] = response_metadata
        parsed["_model_context_size"] = context_size
        parsed["_model_id"] = self.model_id()
        parsed["_model_throughput"] = throughput
        parsed["_model_request_settings"] = {
            **settings,
            "n_predict": requested_n_predict,
            "effective_n_predict": effective_n_predict,
        }
        parsed["_model_constraint_settings"] = {
            **constraint_settings,
            "requested_n_predict": requested_n_predict,
            "effective_n_predict": effective_n_predict,
            "native_constraints_applied": native_constraints,
            "schema_prompt_hint": bool(json_schema and not native_constraints),
            "grammar_prompt_only": bool(grammar and not native_constraints),
        }
        parsed["_model_transport_settings"] = {
            **transport,
            "cache_prompt": bool(use_cache_prompt),
            "thinking_controls_sent": send_thinking_controls,
        }
        parsed["_model_input_audit"] = model_input_audit
        parsed["_model_call_cache_hit"] = False
        parsed["_model_call_cache_hash"] = model_call_hash
        cache_response = dict(parsed)
        cache_response.pop("_model_elapsed_seconds", None)
        cache_response.pop("_model_per_token_timeout_seconds", None)
        cache_response.pop("_model_throughput", None)
        cache_response.pop("_model_input_audit", None)
        write_model_call(model_call_hash, cache_response)
        _log_model_throughput(throughput, endpoint=endpoint, context_size=context_size, effective_n_predict=effective_n_predict)
        return parsed
