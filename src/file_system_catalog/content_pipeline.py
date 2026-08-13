from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

from kmd_runtime_config import (
    csv_integers as _config_csv_integers,
    floating as _config_float,
    integer as _config_int,
    optional_float as _config_optional_float,
)

from context_capacity import (
    CONTEXT_CAPACITY_POLICY,
    context_char_capacity,
    context_ratio,
    context_safety_tokens,
    context_token_capacity,
    contextualize_json_schema,
    positive_float,
)

from .content_schema import (
    CHUNK_COLUMN_NAMES,
    CHUNK_TABLE_NAME,
    CONTENT_CREATE_SQL,
    CONTENT_INDEX_NAMES,
    CONTENT_INDEX_SQL,
    LEGACY_CONTENT_TABLE_NAME,
    REPRESENTATION_COLUMN_NAMES,
    REPRESENTATION_KIND_VALUES,
    REPRESENTATION_TABLE_NAME,
    STRENGTH_VALUES,
)
from .schema import SCHEMA_VERSION, TABLE_NAME

PIPELINE_VERSION = "0.8.0"
PROMPT_VERSION = "facet-representations-v2"
DEFAULT_SEED = 42
FILE_ID_NAMESPACE = uuid.UUID("40d1a28c-b8a2-53cb-9d63-4c560f846035")
CHUNK_ID_NAMESPACE = uuid.UUID("1d4965c2-d389-5a0a-a2da-4d5d9c0444c8")
REPRESENTATION_ID_NAMESPACE = uuid.UUID("a93d9714-8d3b-5363-b355-99a3041fcb48")
WORD_PATTERN = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
STRENGTH_SET = set(STRENGTH_VALUES)
KIND_SET = set(REPRESENTATION_KIND_VALUES)


@dataclass(frozen=True)
class Chunk:
    index: int
    start_char: int
    end_char: int
    text: str
    token_count: int


@dataclass(frozen=True)
class GeneratedAnalysis:
    value: dict[str, Any]
    response_metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelContext:
    configured_tokens: int
    trained_tokens: int


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _default_control_timeout_seconds() -> float:
    value = _config_float("KMD_LOCAL_MODEL_CONTROL_TIMEOUT_SECONDS")
    if value <= 0:
        raise ValueError("KMD_LOCAL_MODEL_CONTROL_TIMEOUT_SECONDS must be a positive number")
    return value


def _stream_total_timeout_seconds(*, per_token_timeout_seconds: float, max_tokens: int) -> float:
    configured = _config_optional_float("KMD_LOCAL_MODEL_STREAM_TOTAL_TIMEOUT_SECONDS")
    if configured is not None:
        if configured <= 0:
            raise ValueError("KMD_LOCAL_MODEL_STREAM_TOTAL_TIMEOUT_SECONDS must be positive")
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


def discover_model_context(base_url: str, model: str = "", *, timeout: float | None = None) -> ModelContext:
    base = base_url.rstrip("/")
    effective_timeout = _default_control_timeout_seconds() if timeout is None else float(timeout)
    payload = request_json(f"{base}/v1/models", timeout=effective_timeout)
    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"model endpoint did not advertise context metadata: {base}/v1/models")
    selected: dict[str, Any] | None = None
    for record in records:
        if not isinstance(record, dict):
            continue
        names = {str(record.get(key) or "") for key in ("id", "name", "model")}
        aliases = record.get("aliases")
        if isinstance(aliases, list):
            names.update(str(value) for value in aliases)
        if model and model in names:
            selected = record
            break
    if selected is None and model:
        advertised: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            advertised.update(str(record.get(key) or "") for key in ("id", "name", "model"))
            aliases = record.get("aliases")
            if isinstance(aliases, list):
                advertised.update(str(value) for value in aliases)
        raise RuntimeError(
            f"configured model is not advertised by endpoint: model={model!r} "
            f"endpoint={base}/v1/models advertised={sorted(name for name in advertised if name)!r}"
        )
    if selected is None:
        selected = next((record for record in records if isinstance(record, dict)), None)
    if selected is None:
        raise RuntimeError(f"model endpoint returned no usable model metadata: {base}/v1/models")
    meta = selected.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    configured = _positive_int(meta.get("n_ctx"))
    trained = _positive_int(meta.get("n_ctx_train"))
    if configured is None:
        props = request_json(f"{base}/props", timeout=effective_timeout)
        configured = _positive_int(
            ((props.get("default_generation_settings") or {}).get("params") or {}).get("n_ctx")
        ) or _positive_int((props.get("default_generation_settings") or {}).get("n_ctx"))
    if configured is None:
        raise RuntimeError(f"model endpoint did not advertise a configured context size: {base}")
    if trained is None:
        trained = configured
    if configured > trained:
        raise RuntimeError(
            f"configured context exceeds trained context for {model or selected.get('id')}: "
            f"{configured} > {trained}"
        )
    return ModelContext(configured_tokens=configured, trained_tokens=trained)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def count_words(value: str) -> int:
    return len(WORD_PATTERN.findall(value))


def stable_file_id(collection_id: str, relative_path_b64: str) -> str:
    return str(uuid.uuid5(FILE_ID_NAMESPACE, f"{collection_id}\0{relative_path_b64}"))


def stable_chunk_id(
    file_id: str,
    content_sha256: str,
    chunk_kind: str,
    chunk_index: int,
    start_char: int,
    end_char: int,
    text_sha256: str,
) -> str:
    value = "\0".join(
        [file_id, content_sha256, chunk_kind, str(chunk_index), str(start_char), str(end_char), text_sha256]
    )
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, value))


def stable_representation_id(
    chunk_id: str,
    analysis_model: str,
    prompt_version: str,
    embedding_model: str,
    global_rank: int,
) -> str:
    value = "\0".join(
        [chunk_id, analysis_model, prompt_version, embedding_model, str(global_rank)]
    )
    return str(uuid.uuid5(REPRESENTATION_ID_NAMESPACE, value))


def request_json(url: str, payload: dict[str, Any] | None = None, *, timeout: float | None = None) -> dict[str, Any]:
    effective_timeout = _default_control_timeout_seconds() if timeout is None else float(timeout)
    if effective_timeout <= 0:
        raise ValueError("request timeout must be positive")
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    attempts = _config_int("KMD_LOCAL_MODEL_CONTROL_RETRY_ATTEMPTS")
    retry_statuses = set(_config_csv_integers("KMD_LOCAL_MODEL_RETRY_HTTP_STATUSES"))
    backoff = _config_float("KMD_LOCAL_MODEL_CONTROL_RETRY_BACKOFF_SECONDS")
    backoff_multiplier = _config_float("KMD_LOCAL_MODEL_RETRY_BACKOFF_MULTIPLIER")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=effective_timeout) as response:
                raw = response.read()
            break
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "backslashreplace")
            retryable = int(error.code) in retry_statuses
            if not retryable or attempt + 1 >= attempts:
                raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
        except urllib.error.URLError as error:
            if attempt + 1 >= attempts:
                raise RuntimeError(f"request failed for {url}: {error}") from error
        delay = backoff * (backoff_multiplier ** attempt)
        if delay > 0:
            time.sleep(delay)
    else:
        raise RuntimeError(f"request retries exhausted for {url}")
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as error:
        raise RuntimeError(f"invalid JSON from {url}: {raw!r}") from error


def _default_per_token_timeout_seconds() -> float:
    value = _config_float("KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS")
    if value <= 0:
        raise ValueError("KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS must be a positive number")
    return value


def _default_embedding_request_retries() -> int:
    value = _config_int("KMD_EMBEDDING_REQUEST_RETRIES")
    if value <= 0:
        raise ValueError("KMD_EMBEDDING_REQUEST_RETRIES must be a positive integer")
    return value


def _embedding_request_timeout_seconds() -> float:
    value = _config_optional_float("KMD_EMBEDDING_REQUEST_TIMEOUT_SECONDS")
    if value is None:
        return _default_per_token_timeout_seconds()
    if value <= 0:
        raise ValueError("KMD_EMBEDDING_REQUEST_TIMEOUT_SECONDS must be a positive number")
    return value


def stream_chat_completion_json(
    url: str,
    payload: dict[str, Any],
    *,
    per_token_timeout_seconds: float,
) -> dict[str, Any]:
    """Run one streamed chat completion with a timeout between token events.

    The same timeout applies while waiting for the first streamed event and for
    every subsequent event. There is no whole-response wall-clock timeout.
    """
    if per_token_timeout_seconds <= 0:
        raise ValueError("per_token_timeout_seconds must be positive")
    body = dict(payload)
    body["stream"] = True
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    metadata: dict[str, Any] = {}
    saw_event = False
    saw_done = False
    saw_terminal_event = False
    max_tokens = int(body.get("max_tokens") or body.get("n_predict") or 1)
    total_timeout_seconds = _stream_total_timeout_seconds(
        per_token_timeout_seconds=per_token_timeout_seconds,
        max_tokens=max_tokens,
    )
    event_limit = _stream_event_limit(max_tokens)
    byte_limit = _stream_byte_limit(max_tokens)
    started = time.monotonic()
    stream_events = 0
    stream_bytes = 0
    try:
        with urllib.request.urlopen(request, timeout=per_token_timeout_seconds) as response:
            for raw_line in response:
                stream_bytes += len(raw_line)
                if stream_bytes > byte_limit:
                    raise RuntimeError(f"model stream exceeded byte limit {byte_limit}: {url}")
                if time.monotonic() - started > total_timeout_seconds:
                    raise RuntimeError(f"model stream exceeded total timeout {total_timeout_seconds}: {url}")
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    saw_done = True
                    break
                stream_events += 1
                if stream_events > event_limit:
                    raise RuntimeError(f"model stream exceeded event limit {event_limit}: {url}")
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"invalid streamed JSON event from {url}: {line!r}") from error
                if not isinstance(event, dict):
                    raise RuntimeError(f"non-object streamed event from {url}: {event!r}")
                if event.get("error"):
                    raise RuntimeError(f"model stream error from {url}: {event['error']!r}")
                saw_event = True
                for key in ("model", "system_fingerprint", "usage", "timings"):
                    if event.get(key) is not None:
                        metadata[key] = event[key]
                choices = event.get("choices")
                if isinstance(choices, list) and choices:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            if delta.get("content") is not None:
                                content_parts.append(str(delta.get("content") or ""))
                            if delta.get("reasoning_content") is not None:
                                reasoning_parts.append(str(delta.get("reasoning_content") or ""))
                        message = choice.get("message")
                        if isinstance(message, dict):
                            if message.get("content") is not None:
                                content_parts.append(str(message.get("content") or ""))
                            if message.get("reasoning_content") is not None:
                                reasoning_parts.append(str(message.get("reasoning_content") or ""))
                        if choice.get("text") is not None:
                            content_parts.append(str(choice.get("text") or ""))
                        if choice.get("finish_reason") is not None:
                            finish_reason = str(choice.get("finish_reason"))
                            saw_terminal_event = True
                elif event.get("content") is not None:
                    content_parts.append(str(event.get("content") or ""))
                    if event.get("stop") is True:
                        finish_reason = "stop"
                        saw_terminal_event = True
    except urllib.error.HTTPError as error:
        body_text = error.read().decode("utf-8", "backslashreplace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body_text[:2000]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"stream request failed for {url}: {error}") from error
    if not saw_event:
        raise RuntimeError(f"model stream returned no events: {url}")
    if not (saw_done or saw_terminal_event):
        raise RuntimeError(f"model stream ended without terminal completion metadata: {url}")
    return {
        **metadata,
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "finish_reason": finish_reason,
        "saw_done": saw_done,
        "saw_terminal_event": saw_terminal_event,
        "stream": True,
        "per_token_timeout_seconds": per_token_timeout_seconds,
        "stream_events": stream_events,
        "stream_event_limit": event_limit,
        "stream_bytes": stream_bytes,
        "stream_byte_limit": byte_limit,
        "stream_total_timeout_seconds": total_timeout_seconds,
    }


class AnalysisClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        seed: int = DEFAULT_SEED,
        temperature: float = 0.0,
        per_token_timeout_seconds: float | None = None,
        control_timeout_seconds: float | None = None,
        retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.per_token_timeout_seconds = (
            _default_per_token_timeout_seconds()
            if per_token_timeout_seconds is None
            else float(per_token_timeout_seconds)
        )
        if self.per_token_timeout_seconds <= 0:
            raise ValueError("per_token_timeout_seconds must be positive")
        self.control_timeout_seconds = (
            self.per_token_timeout_seconds
            if control_timeout_seconds is None
            else float(control_timeout_seconds)
        )
        if self.control_timeout_seconds <= 0:
            raise ValueError("control_timeout_seconds must be positive")
        self.retries = retries
        self._token_cache: dict[str, int] = {}
        self._prompt_token_cache: dict[str, int] = {}
        self._model_context: ModelContext | None = None

    def health(self) -> dict[str, Any]:
        return request_json(f"{self.base_url}/health", timeout=self.control_timeout_seconds)

    def model_context(self) -> ModelContext:
        if self._model_context is None:
            self._model_context = discover_model_context(self.base_url, self.model, timeout=self.control_timeout_seconds)
        return self._model_context

    def token_count(self, text: str) -> int:
        digest = sha256_text(text)
        cached = self._token_cache.get(digest)
        if cached is not None:
            return cached
        response = request_json(
            f"{self.base_url}/tokenize",
            {"content": text, "add_special": False},
            timeout=self.control_timeout_seconds,
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"tokenizer response has no token list: {response}")
        count = len(tokens)
        self._token_cache[digest] = count
        return count

    def _render_prompt(self, *, system: str, user: str) -> str:
        response = request_json(
            f"{self.base_url}/apply-template",
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=self.control_timeout_seconds,
        )
        prompt = response.get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeError(f"chat-template response has no prompt: {response}")
        return prompt

    def prompt_token_count(self, *, system: str, user: str) -> int:
        digest = sha256_text(system + "\0" + user)
        cached = self._prompt_token_cache.get(digest)
        if cached is not None:
            return cached
        count = self.token_count(self._render_prompt(system=system, user=user))
        self._prompt_token_cache[digest] = count
        return count

    def output_token_budget(
        self,
        *,
        ratio_names: tuple[str, ...] = (),
        ratio_default: float = 1.0 / 32.0,
    ) -> int:
        return context_token_capacity(
            self.model_context().configured_tokens,
            ratio_names=ratio_names,
            ratio_default=ratio_default,
        )

    def maximum_attempt_tokens(self, max_tokens: int) -> int:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        context = self.model_context().configured_tokens
        policy_maximum = context_token_capacity(
            context,
            ratio_names=("KMD_FILESYSTEM_MAX_OUTPUT_RATIO",),
            ratio_default=1.0 / 2.0,
        )
        return max(int(max_tokens), policy_maximum)

    def _output_token_budgets(
        self,
        *,
        system: str,
        user: str,
        base_tokens: int,
    ) -> tuple[int, list[int]]:
        if base_tokens < 1:
            raise ValueError("base_tokens must be positive")
        prompt_tokens = self.prompt_token_count(system=system, user=user)
        context = self.model_context().configured_tokens
        safety = context_safety_tokens(context)
        exact_maximum = context - prompt_tokens - safety
        if exact_maximum < base_tokens:
            raise RuntimeError(
                "analysis request would exceed configured model context before transmission: "
                f"prompt={prompt_tokens} output_limit={base_tokens} safety={safety} context={context}"
            )
        maximum = min(self.maximum_attempt_tokens(base_tokens), exact_maximum)
        multiplier = positive_float(("KMD_FILESYSTEM_RETRY_OUTPUT_MULTIPLIER",), 2.0)
        budgets: list[int] = []
        current = min(int(base_tokens), maximum)
        while True:
            budgets.append(current)
            if current >= maximum:
                break
            candidate = max(current + 1, int(current * multiplier))
            current = min(candidate, maximum)
        return prompt_tokens, budgets

    def request_fits(
        self, *, system: str, user: str, max_tokens: int, worst_retry: bool = True
    ) -> bool:
        try:
            prompt_tokens, budgets = self._output_token_budgets(
                system=system,
                user=user,
                base_tokens=max_tokens,
            )
        except RuntimeError:
            return False
        output_tokens = budgets[-1] if worst_retry else budgets[0]
        context = self.model_context().configured_tokens
        safety = context_safety_tokens(context)
        return prompt_tokens + output_tokens + safety <= context


    def available_content_tokens(
        self, *, system: str, user_without_content: str, max_tokens: int
    ) -> int:
        output_tokens = self.maximum_attempt_tokens(max_tokens)
        overhead = self.prompt_token_count(system=system, user=user_without_content)
        context = self.model_context().configured_tokens
        safety = context_safety_tokens(context)
        return max(1, context - output_tokens - overhead - safety)

    def _ensure_request_fits(
        self, *, system: str, user: str, output_tokens: int
    ) -> int:
        prompt_tokens = self.prompt_token_count(system=system, user=user)
        context = self.model_context().configured_tokens
        safety = context_safety_tokens(context)
        if prompt_tokens + output_tokens + safety > context:
            raise RuntimeError(
                "analysis request would exceed configured model context before transmission: "
                f"prompt={prompt_tokens} output_limit={output_tokens} safety={safety} context={context}"
            )
        return prompt_tokens


    def complete(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        system: str,
        user: str,
        max_tokens: int | None = None,
    ) -> GeneratedAnalysis:
        context = self.model_context().configured_tokens
        base_tokens = max_tokens or self.output_token_budget(
            ratio_names=("KMD_FILESYSTEM_ANALYSIS_OUTPUT_RATIO",),
            ratio_default=1.0 / 32.0,
        )
        prompt_tokens, output_budgets = self._output_token_budgets(
            system=system,
            user=user,
            base_tokens=base_tokens,
        )
        retry_delay_multiplier = positive_float(("KMD_FILESYSTEM_RETRY_DELAY_MULTIPLIER",), 2.0)
        last_error: Exception | None = None
        total_attempt = 0
        for budget_index, attempt_max_tokens in enumerate(output_budgets, start=1):
            transient_attempt = 0
            while True:
                total_attempt += 1
                try:
                    self._ensure_request_fits(
                        system=system,
                        user=user,
                        output_tokens=attempt_max_tokens,
                    )
                    resolved_schema = contextualize_json_schema(
                        schema,
                        context_size=context,
                        output_tokens=attempt_max_tokens,
                    )
                    Draft202012Validator.check_schema(resolved_schema)
                    payload = {
                        "model": self.model,
                        "temperature": self.temperature,
                        "seed": self.seed,
                        "max_tokens": attempt_max_tokens,
                        "provider": {"require_parameters": True},
                        "enable_thinking": False,
                        "reasoning_format": "deepseek",
                        "reasoning_budget": 0,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {"name": schema_name, "strict": True, "schema": resolved_schema},
                        },
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "stream": True,
                    }
                    response = stream_chat_completion_json(
                        f"{self.base_url}/v1/chat/completions",
                        payload,
                        per_token_timeout_seconds=self.per_token_timeout_seconds,
                    )
                    finish_reason = response.get("finish_reason")
                    if finish_reason == "length":
                        last_error = RuntimeError(
                            f"generation reached output budget {attempt_max_tokens} before terminal completion"
                        )
                        break
                    if finish_reason != "stop":
                        raise RuntimeError(f"generation did not finish cleanly: {finish_reason}")
                    if response.get("reasoning_content"):
                        raise RuntimeError("reasoning mode was not disabled")
                    content = response.get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise RuntimeError("generation returned no content")
                    value = json.loads(content)
                    try:
                        Draft202012Validator(resolved_schema).validate(value)
                    except JSONSchemaValidationError as error:
                        raise RuntimeError(
                            f"generation failed response schema validation: {error.message}"
                        ) from error
                    model_context = self.model_context()
                    metadata = {
                        "attempt": total_attempt,
                        "output_budget_index": budget_index,
                        "output_budget_count": len(output_budgets),
                        "max_tokens": attempt_max_tokens,
                        "prompt_tokens": prompt_tokens,
                        "safety_tokens": context_safety_tokens(context),
                        "capacity_policy": CONTEXT_CAPACITY_POLICY,
                        "configured_context_tokens": model_context.configured_tokens,
                        "trained_context_tokens": model_context.trained_tokens,
                        "model": response.get("model"),
                        "system_fingerprint": response.get("system_fingerprint"),
                        "usage": response.get("usage"),
                        "timings": response.get("timings"),
                        "finish_reason": finish_reason,
                        "saw_done": response.get("saw_done"),
                        "saw_terminal_event": response.get("saw_terminal_event"),
                        "stream": True,
                        "per_token_timeout_seconds": self.per_token_timeout_seconds,
                        "parsed": value,
                    }
                    return GeneratedAnalysis(value=value, response_metadata=metadata)
                except Exception as error:
                    last_error = error
                    transient_attempt += 1
                    if transient_attempt >= self.retries:
                        raise RuntimeError(
                            f"analysis failed after {transient_attempt} transient attempts "
                            f"at output budget {attempt_max_tokens}: {last_error}"
                        ) from last_error
                    time.sleep(min(retry_delay_multiplier ** transient_attempt, 8.0))
        assert last_error is not None
        raise RuntimeError(
            "analysis exhausted every context-derived output budget without terminal completion: "
            f"budgets={output_budgets} last_error={last_error}"
        ) from last_error


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        revision: str,
        expected_dimension: int = 1024,
        batch_size: int | None = None,
        max_batch_characters: int | None = None,
        request_timeout_seconds: float | None = None,
        retries: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.revision = revision
        self.expected_dimension = expected_dimension
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_batch_characters is not None and max_batch_characters < 1:
            raise ValueError("max_batch_characters must be positive")
        self.batch_size = batch_size
        self.max_batch_characters = max_batch_characters
        self.request_timeout_seconds = (
            float(request_timeout_seconds)
            if request_timeout_seconds is not None
            else _embedding_request_timeout_seconds()
        )
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self.retries = int(retries) if retries is not None else _default_embedding_request_retries()
        if self.retries <= 0:
            raise ValueError("retries must be positive")
        self._model_context: ModelContext | None = None
        self._token_cache: dict[tuple[str, bool], int] = {}

    def health(self) -> dict[str, Any]:
        return request_json(f"{self.base_url}/health", timeout=self.request_timeout_seconds)

    def model_context(self) -> ModelContext:
        if self._model_context is None:
            self._model_context = discover_model_context(self.base_url, self.model)
        return self._model_context

    def token_count(self, text: str, *, add_special: bool = True) -> int:
        key = (sha256_text(text), add_special)
        cached = self._token_cache.get(key)
        if cached is not None:
            return cached
        response = request_json(
            f"{self.base_url}/tokenize",
            {"content": text, "add_special": add_special},
            timeout=self.request_timeout_seconds,
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"tokenizer response has no token list: {response}")
        count = len(tokens)
        self._token_cache[key] = count
        return count

    def _validate_input(self, text: str) -> int:
        count = self.token_count(text, add_special=True)
        context = self.model_context().configured_tokens
        if count > context:
            raise RuntimeError(
                "embedding input would exceed configured model context before transmission: "
                f"tokens={count} context={context}"
            )
        return count

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        context = self.model_context().configured_tokens
        batch_size = self.batch_size or context_token_capacity(
            context,
            ratio_names=("KMD_EMBEDDING_BATCH_COUNT_RATIO",),
            ratio_default=1.0 / 1024.0,
        )
        max_batch_characters = self.max_batch_characters or context_char_capacity(
            context,
            ratio_names=("KMD_EMBEDDING_BATCH_CHARACTER_RATIO",),
            ratio_default=1.0,
        )
        batch_token_budget = context_token_capacity(
            context,
            ratio_names=("KMD_EMBEDDING_BATCH_TOKEN_RATIO",),
            ratio_default=1.0 / 8.0,
        )
        batches: list[list[str]] = []
        current: list[str] = []
        current_characters = 0
        current_tokens = 0
        for text in texts:
            token_count = self._validate_input(text)
            if current and (
                len(current) >= batch_size
                or current_characters + len(text) > max_batch_characters
                or current_tokens + token_count > batch_token_budget
            ):
                batches.append(current)
                current = []
                current_characters = 0
                current_tokens = 0
            current.append(text)
            current_characters += len(text)
            current_tokens += token_count
            if token_count > batch_token_budget:
                batches.append(current)
                current = []
                current_characters = 0
                current_tokens = 0
        if current:
            batches.append(current)
        for batch in batches:
            last_error: Exception | None = None
            response: dict[str, Any] | None = None
            for attempt in range(1, self.retries + 1):
                try:
                    response = request_json(
                        f"{self.base_url}/v1/embeddings",
                        {"model": self.model, "input": batch},
                        timeout=self.request_timeout_seconds,
                    )
                    break
                except Exception as error:
                    last_error = error
                    if attempt < self.retries:
                        time.sleep(min(2.0 ** (attempt - 1), 8.0))
            if response is None:
                assert last_error is not None
                raise RuntimeError(
                    f"embedding request failed after {self.retries} attempts: {last_error}"
                ) from last_error
            items = sorted(response.get("data", []), key=lambda item: int(item["index"]))
            if len(items) != len(batch):
                raise RuntimeError(f"embedding count mismatch: expected {len(batch)}, got {len(items)}")
            for item in items:
                vector = np.asarray(item["embedding"], dtype="<f4")
                if vector.ndim != 1 or vector.shape[0] != self.expected_dimension:
                    raise RuntimeError(f"unexpected embedding shape: {vector.shape}")
                norm = float(np.linalg.norm(vector))
                if not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError(f"invalid embedding norm: {norm}")
                vectors.append(np.asarray(vector / norm, dtype="<f4"))
        return vectors


def _closed_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _representation_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "kind": {
                "type": "string",
                "enum": ["description", "sentence", "keyphrase", "keyword", "entity", "topic"],
                "description": "The linguistic form of this retrieval string.",
            },
            "item_strength": {
                "type": "string",
                "enum": list(STRENGTH_VALUES),
                "description": "Retrieval importance of this specific string, not its topic name.",
            },
            "text": {
                "type": "string",
                "x-kmd-string-profile": "value",
                "description": "Faithful retrieval text preserving the source meaning and grammar.",
            },
        }
    )


def _facet_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "facet_name": {
                "type": "string",
                "x-kmd-string-profile": "label",
                "description": "A concise semantic topic name such as power resilience or tax compliance. Never use a strength word such as essential, strong, moderate, weak, or very_weak here.",
            },
            "facet_strength": {
                "type": "string",
                "enum": list(STRENGTH_VALUES),
                "description": "Importance of this semantic topic within the source.",
            },
            "representations": {"type": "array", "x-kmd-array-profile": "dense", "items": _representation_schema()},
        }
    )


def chunk_analysis_schema_for_keys(keys: Sequence[str]) -> dict[str, Any]:
    if not keys:
        raise ValueError("keys must not be empty")
    analysis = _closed_object(
        {
            "chunk_key": {"type": "string", "enum": list(keys)},
            "document_summary": {"type": "string", "x-kmd-string-profile": "reason"},
            "facets": {"type": "array", "x-kmd-array-profile": "dense", "items": _facet_schema()},
        }
    )
    return _closed_object({"analyses": {"type": "array", "minItems": len(keys), "maxItems": len(keys), "items": analysis}})


FILE_ANALYSIS_SCHEMA = _closed_object(
    {
        "document_summary": {"type": "string", "x-kmd-string-profile": "reason"},
        "facets": {"type": "array", "x-kmd-array-profile": "dense", "items": _facet_schema()},
    }
)

STRENGTH_GUIDANCE = """Use verbal strength labels consistently: essential means indispensable to recognizing the source as a whole; very_strong means a major independent theme; strong means a substantial secondary theme; moderate means a useful supporting subject; weak means a specific detail; very_weak means a narrow detail with limited retrieval value."""

CHUNK_SYSTEM_PROMPT = f"""You create retrieval representations for source-document chunks. Return only schema-valid JSON. First identify every distinct meaningful semantic facet, including minority facets, and order facets from most to least important. The facet_name field must contain the actual subject, such as power resilience, bicycle courier backup, tax compliance, or privacy controls; it must never contain an importance label such as essential, very_strong, strong, moderate, weak, or very_weak. Put importance only in facet_strength and item_strength. Within each facet, return every distinct useful representation supported by the text, ordered from most to least useful. Use complete sentences when precise grammar, actors, objects, negation, chronology or qualifications matter; use concise descriptions and keyphrases for concepts; use single words only when the isolated word is independently useful. Preserve original meaning, punctuation, capitalization and factual qualifications. Do not pad with near-duplicates. For meaningless gibberish, return a truthful summary and an empty facets array. {STRENGTH_GUIDANCE} Never merge or omit requested chunk keys, even when chunks are duplicated."""

FILE_SYSTEM_PROMPT = f"""You combine ordered chunk analyses into a file-level retrieval description. Return only schema-valid JSON. Identify every distinct meaningful file facet, including brief facets that appear in only one chunk, and order facets from most to least important. The facet_name field must be the actual semantic subject and must never be an importance label. Put importance only in facet_strength and item_strength. Within each facet, return every distinct useful representation, ordered from most to least useful. Preserve long-distance references, corrections, negation, actors and objects. Repetition must not multiply importance. Keep ambiguous senses separate. Ignore meaningless identifiers. {STRENGTH_GUIDANCE}"""


def _split_oversized_span(text: str, start: int, end: int, max_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    overlap_chars = min(max(0, overlap_chars), max(0, max_chars // 4))
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        desired = min(end, cursor + max_chars)
        cut = desired
        if desired < end:
            search_start = max(cursor + max_chars // 2, desired - overlap_chars)
            candidates = [
                text.rfind("\n\n", search_start, desired),
                text.rfind(". ", search_start, desired),
                text.rfind(" ", search_start, desired),
            ]
            valid = [value for value in candidates if value > cursor]
            if valid:
                cut = max(valid) + (2 if text[max(valid) : max(valid) + 2] == ". " else 0)
        if cut <= cursor:
            cut = desired
        spans.append((cursor, cut))
        if cut >= end:
            break
        cursor = max(cursor + 1, cut - overlap_chars)
    return spans


def chunk_text(
    text: str,
    analysis_client: AnalysisClient,
    *,
    embedding_client: EmbeddingClient | None = None,
    target_chars: int | None = None,
    max_chars: int | None = None,
    overlap_chars: int | None = None,
    max_tokens: int | None = None,
) -> list[Chunk]:
    if not text:
        return [Chunk(index=0, start_char=0, end_char=0, text="", token_count=0)]
    context_reader = getattr(analysis_client, "model_context", None)
    if context_reader is None:
        raise RuntimeError("analysis client must expose model_context for context-derived chunking")
    context = int(context_reader().configured_tokens)
    if max_tokens is None:
        max_tokens = context_token_capacity(
            context,
            ratio_names=("KMD_FILESYSTEM_CHUNK_INPUT_RATIO",),
            ratio_default=0.16,
        )
    chars_per_token = positive_float(("KMD_FILESYSTEM_CHUNK_CHARS_PER_TOKEN",), 4.0)
    if max_chars is None:
        max_chars = max(1, int(max_tokens * chars_per_token))
    if target_chars is None:
        target_chars = max(1, int(max_chars * context_ratio(("KMD_FILESYSTEM_CHUNK_TARGET_RATIO",), 6.0 / 7.0)))
    if overlap_chars is None:
        overlap_chars = max(0, int(max_chars * context_ratio(("KMD_FILESYSTEM_CHUNK_OVERLAP_RATIO",), 5.0 / 84.0)))
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    paragraph_spans: list[tuple[int, int]] = []
    cursor = 0
    for match in re.finditer(r"\n\s*\n", text):
        end = match.end()
        if end > cursor:
            paragraph_spans.append((cursor, end))
        cursor = end
    if cursor < len(text):
        paragraph_spans.append((cursor, len(text)))
    expanded: list[tuple[int, int]] = []
    for start, end in paragraph_spans:
        if end - start > max_chars:
            expanded.extend(_split_oversized_span(text, start, end, max_chars, overlap_chars))
        else:
            expanded.append((start, end))
    chunk_spans: list[tuple[int, int]] = []
    unit_index = 0
    while unit_index < len(expanded):
        start = expanded[unit_index][0]
        end = expanded[unit_index][1]
        next_index = unit_index + 1
        while next_index < len(expanded):
            candidate_end = expanded[next_index][1]
            if candidate_end - start > target_chars and end > start:
                break
            if candidate_end - start > max_chars:
                break
            end = candidate_end
            next_index += 1
        chunk_spans.append((start, end))
        if next_index >= len(expanded):
            break
        overlap_start = next_index
        while overlap_start > unit_index + 1 and end - expanded[overlap_start - 1][0] < overlap_chars:
            overlap_start -= 1
        unit_index = overlap_start
    validated: list[tuple[int, int, int]] = []
    pending = list(chunk_spans)
    embedding_context_reader = (
        getattr(embedding_client, "model_context", None) if embedding_client is not None else None
    )
    embedding_token_counter = (
        getattr(embedding_client, "token_count", None) if embedding_client is not None else None
    )
    embedding_limit = (
        context_token_capacity(
            int(embedding_context_reader().configured_tokens),
            ratio_names=("KMD_EMBEDDING_INPUT_RATIO",),
            ratio_default=1.0 / 4.0,
        )
        if embedding_context_reader is not None and embedding_token_counter is not None
        else None
    )
    while pending:
        start, end = pending.pop(0)
        value = text[start:end]
        analysis_count = analysis_client.token_count(value)
        embedding_count = (
            embedding_token_counter(value, add_special=True)
            if embedding_limit is not None and embedding_token_counter is not None
            else 0
        )
        analysis_fits = analysis_count <= max_tokens
        embedding_fits = embedding_limit is None or embedding_count <= embedding_limit
        if analysis_fits and embedding_fits:
            validated.append((start, end, analysis_count))
            continue
        ratios = [max_tokens / max(1, analysis_count)]
        if embedding_limit is not None:
            ratios.append(embedding_limit / max(1, embedding_count))
        split_chars = max(1, int((end - start) * min(ratios) * 0.88))
        split_overlap = min(overlap_chars, max(0, split_chars // 4))
        pieces = _split_oversized_span(text, start, end, split_chars, split_overlap)
        if len(pieces) == 1 and pieces[0] == (start, end):
            raise RuntimeError(
                "unable to split chunk below endpoint token limits: "
                f"analysis={analysis_count}/{max_tokens} "
                f"embedding={embedding_count}/{embedding_limit}"
            )
        pending[0:0] = pieces
    validated.sort(key=lambda item: (item[0], item[1]))
    return [
        Chunk(index=index, start_char=start, end_char=end, text=text[start:end], token_count=count)
        for index, (start, end, count) in enumerate(validated)
    ]


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def normalize_analysis(value: dict[str, Any]) -> dict[str, Any]:
    summary = _normalize_text(value.get("document_summary", ""))
    if not summary:
        raise RuntimeError("analysis has no document summary")
    facets: list[dict[str, Any]] = []
    seen_facets: set[str] = set()
    seen_texts: set[str] = set()
    raw_facets = value.get("facets")
    if not isinstance(raw_facets, list):
        raise RuntimeError("analysis facets is not an array")
    for raw_facet in raw_facets:
        if not isinstance(raw_facet, dict):
            continue
        label = _normalize_text(raw_facet.get("facet_name", raw_facet.get("label", "")))
        strength = str(raw_facet.get("facet_strength", raw_facet.get("strength", "")))
        if strength not in STRENGTH_SET:
            continue
        representations: list[dict[str, str]] = []
        raw_representations = raw_facet.get("representations")
        if not isinstance(raw_representations, list):
            continue
        for raw_representation in raw_representations:
            if not isinstance(raw_representation, dict):
                continue
            kind = str(raw_representation.get("kind", ""))
            item_strength = str(raw_representation.get("item_strength", raw_representation.get("strength", "")))
            text = _normalize_text(raw_representation.get("text", ""))
            key = text.casefold()
            if kind not in KIND_SET or item_strength not in STRENGTH_SET or not text or key in seen_texts:
                continue
            seen_texts.add(key)
            representations.append({"kind": kind, "strength": item_strength, "text": text})
        if not representations:
            continue
        if not label or label.casefold() in STRENGTH_SET:
            label = representations[0]["text"]
        label_key = label.casefold()
        if label_key in seen_facets:
            continue
        seen_facets.add(label_key)
        facets.append(
            {"label": label, "strength": strength, "representations": representations}
        )
    return {"document_summary": summary, "facets": facets}


def flatten_representations(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "kind": "summary",
            "facet_label": "document summary",
            "facet_strength": "essential",
            "item_strength": "essential",
            "facet_rank": 0,
            "item_rank": 0,
            "global_rank": 0,
            "text": analysis["document_summary"],
        }
    ]
    global_rank = 1
    for facet_rank, facet in enumerate(analysis["facets"], start=1):
        rows.append(
            {
                "kind": "topic",
                "facet_label": facet["label"],
                "facet_strength": facet["strength"],
                "item_strength": facet["strength"],
                "facet_rank": facet_rank,
                "item_rank": 0,
                "global_rank": global_rank,
                "text": facet["label"],
            }
        )
        global_rank += 1
        for item_rank, item in enumerate(facet["representations"], start=1):
            if item["text"].casefold() == facet["label"].casefold():
                continue
            rows.append(
                {
                    "kind": item["kind"],
                    "facet_label": facet["label"],
                    "facet_strength": facet["strength"],
                    "item_strength": item["strength"],
                    "facet_rank": facet_rank,
                    "item_rank": item_rank,
                    "global_rank": global_rank,
                    "text": item["text"],
                }
            )
            global_rank += 1
    return rows


def _vector_blob(vector: np.ndarray) -> tuple[bytes, float, str]:
    normalized = np.asarray(vector, dtype="<f4")
    blob = normalized.tobytes(order="C")
    return blob, float(np.linalg.norm(normalized)), hashlib.sha256(blob).hexdigest()


def _insert_many(
    connection: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    if not rows:
        return
    column_sql = ",".join(f'"{name}"' for name in columns)
    placeholders = ",".join("?" for _ in columns)
    statement = f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholders})'
    connection.executemany(statement, ([row[name] for name in columns] for row in rows))


def _legacy_strength(kind: str) -> str:
    return {
        "summary_short": "essential",
        "summary_long": "essential",
        "topic": "strong",
        "search_phrase": "strong",
        "keyword": "moderate",
    }.get(kind, "moderate")


def migrate_legacy_content_schema(connection: sqlite3.Connection, root: Path) -> bool:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if CHUNK_TABLE_NAME in tables and REPRESENTATION_TABLE_NAME in tables:
        if LEGACY_CONTENT_TABLE_NAME in tables:
            raise RuntimeError("both legacy and normalized content tables exist")
        if connection.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        return False
    if LEGACY_CONTENT_TABLE_NAME not in tables:
        raise RuntimeError(f"missing content tables: {sorted(tables)}")
    legacy_rows = list(connection.execute(f'SELECT * FROM "{LEGACY_CONTENT_TABLE_NAME}"'))
    legacy_columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{LEGACY_CONTENT_TABLE_NAME}")')]
    legacy = [dict(zip(legacy_columns, row, strict=True)) for row in legacy_rows]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in legacy:
        grouped.setdefault(str(row["source_unit_id"]), []).append(row)
    chunk_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    for source_id, rows in grouped.items():
        first = rows[0]
        chunk_kind = "chunk" if first["source_level"] == "chunk" else "file"
        start = int(first["source_start_char"] or 0)
        end = int(first["source_end_char"] or start)
        raw = next((row for row in rows if row["analysis_kind"] == "raw_text"), None)
        text: str | None = str(raw["analysis_text"]) if raw is not None else None
        if text is None:
            relative = base64.b64decode(first["relative_path_b64"])
            path = os.fsencode(root) + (b"/" + relative if relative else b"")
            file_text = _read_regular_file_bytes(
                path,
                max_bytes=_content_file_memory_limit_bytes(),
            ).decode("utf-8", "replace")
            text = file_text[start:end]
        created = min(int(row["created_at_ns"]) for row in rows)
        updated = max(int(row["updated_at_ns"]) for row in rows)
        if raw is not None:
            embedding_model = raw["embedding_model"]
            embedding_revision = raw["embedding_model_revision"]
            dimension = raw["embedding_dimension"]
            dtype = raw["embedding_dtype"]
            norm = raw["embedding_norm"]
            blob = raw["embedding_blob"]
            blob_hash = raw["embedding_sha256"]
        else:
            embedding_model = embedding_revision = dimension = dtype = norm = blob = blob_hash = None
        chunk_rows.append(
            {
                "chunk_id": source_id,
                "file_id": first["file_id"],
                "collection_id": first["collection_id"],
                "filesystem_entry_id": int(first["filesystem_entry_id"]),
                "content_object_id": first["content_object_id"],
                "content_sha256": first["content_sha256"],
                "chunk_kind": chunk_kind,
                "chunk_index": int(first["source_index"]),
                "start_char": start,
                "end_char": end,
                "character_count": end - start,
                "word_count": count_words(text),
                "token_count": int(first["source_token_count"]),
                "text_sha256": first["source_text_sha256"],
                "embedding_model": embedding_model,
                "embedding_model_revision": embedding_revision,
                "embedding_dimension": dimension,
                "embedding_dtype": dtype,
                "embedding_norm": norm,
                "embedding_blob": blob,
                "embedding_sha256": blob_hash,
                "created_at_ns": created,
                "updated_at_ns": updated,
            }
        )
        generated = [row for row in rows if row["analysis_kind"] != "raw_text"]
        priority = {"summary_short": 0, "summary_long": 1, "topic": 2, "search_phrase": 3, "keyword": 4}
        generated.sort(key=lambda row: (priority.get(str(row["analysis_kind"]), 9), int(row["ordinal"])))
        for global_rank, row in enumerate(generated):
            kind = str(row["analysis_kind"])
            strength = _legacy_strength(kind)
            representation_rows.append(
                {
                    "representation_id": row["semantic_entry_id"],
                    "chunk_id": source_id,
                    "representation_kind": kind,
                    "facet_label": str(row["analysis_text"]) if kind == "topic" else kind.replace("_", " "),
                    "facet_strength": strength,
                    "item_strength": strength,
                    "facet_rank": global_rank,
                    "item_rank": int(row["ordinal"]),
                    "global_rank": global_rank,
                    "representation_text": row["analysis_text"],
                    "representation_text_sha256": row["analysis_text_sha256"],
                    "analysis_model": row["analysis_model"],
                    "analysis_model_fingerprint": row["analysis_model_fingerprint"],
                    "prompt_version": row["prompt_version"],
                    "generation_seed": int(row["generation_seed"]),
                    "pipeline_version": row["pipeline_version"],
                    "generation_json": row["generation_json"],
                    "attributes_json": row["attributes_json"],
                    "embedding_model": row["embedding_model"],
                    "embedding_model_revision": row["embedding_model_revision"],
                    "embedding_dimension": int(row["embedding_dimension"]),
                    "embedding_dtype": row["embedding_dtype"],
                    "embedding_norm": float(row["embedding_norm"]),
                    "embedding_blob": row["embedding_blob"],
                    "embedding_sha256": row["embedding_sha256"],
                    "analysis_status": row["analysis_status"],
                    "analysis_error": row["analysis_error"],
                    "created_at_ns": int(row["created_at_ns"]),
                    "updated_at_ns": int(row["updated_at_ns"]),
                }
            )
    connection.execute("BEGIN IMMEDIATE")
    try:
        for statement in CONTENT_CREATE_SQL:
            connection.execute(statement)
        _insert_many(connection, CHUNK_TABLE_NAME, CHUNK_COLUMN_NAMES, chunk_rows)
        _insert_many(connection, REPRESENTATION_TABLE_NAME, REPRESENTATION_COLUMN_NAMES, representation_rows)
        connection.execute(f'DROP TABLE "{LEGACY_CONTENT_TABLE_NAME}"')
        for statement in CONTENT_INDEX_SQL:
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    connection.execute("VACUUM")
    return True


def _host_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return 1024 * 1024 * 1024


def _content_file_memory_limit_bytes() -> int:
    ratio = _config_float("KMD_FILESYSTEM_CONTENT_MEMORY_RATIO")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("KMD_FILESYSTEM_CONTENT_MEMORY_RATIO must be between 0 and 1")
    return max(1, int(_host_memory_bytes() * ratio))


def _read_regular_file_bytes(path: bytes | os.PathLike[str] | str, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"content path is not a regular file: {os.fsdecode(path)}")
        if metadata.st_size > max_bytes:
            raise RuntimeError(
                f"content file exceeds host-memory safety limit: path={os.fsdecode(path)} "
                f"size={metadata.st_size} limit={max_bytes}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError(
                    f"content file grew beyond host-memory safety limit: path={os.fsdecode(path)} limit={max_bytes}"
                )
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


class ContentSemanticPipeline:
    def __init__(
        self,
        *,
        database: os.PathLike[str] | str,
        root: os.PathLike[str] | str,
        collection_id: str,
        analysis_client: AnalysisClient,
        embedding_client: EmbeddingClient,
        prompt_version: str = PROMPT_VERSION,
        pipeline_version: str = PIPELINE_VERSION,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if not collection_id.strip():
            raise ValueError("collection_id must not be empty")
        self.database = Path(database).resolve()
        self.root = Path(root).resolve()
        self.collection_id = collection_id
        self.analysis_client = analysis_client
        self.embedding_client = embedding_client
        self.prompt_version = prompt_version
        self.pipeline_version = pipeline_version
        self.seed = seed

    def _extract_text(self, entry: sqlite3.Row) -> str:
        relative = base64.b64decode(entry["relative_path_b64"])
        path = os.fsencode(self.root) + (b"/" + relative if relative else b"")
        data = _read_regular_file_bytes(
            path,
            max_bytes=_content_file_memory_limit_bytes(),
        )
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["content_sha256"]:
            raise RuntimeError(
                f"content changed after filesystem scan for {entry['relative_path_display']}: database={entry['content_sha256']} current={digest}"
            )
        encoding = (entry["magic_mime_encoding"] or "").lower()
        candidates = ["utf-8"]
        if encoding and encoding not in {"binary", "unknown-8bit", "us-ascii", "utf-8"}:
            candidates.append(encoding)
        candidates.extend(["utf-8-sig", "latin-1"])
        for candidate in candidates:
            try:
                return data.decode(candidate)
            except Exception:
                pass
        return data.decode("utf-8", "replace")

    def _analysis_request_fits(
        self, *, system: str, user: str, max_tokens: int
    ) -> bool:
        checker = getattr(self.analysis_client, "request_fits", None)
        if checker is None:
            return True
        return bool(checker(system=system, user=user, max_tokens=max_tokens))

    def _analysis_content_limit(
        self, *, system: str, user_without_content: str, max_tokens: int
    ) -> int:
        calculator = getattr(self.analysis_client, "available_content_tokens", None)
        if calculator is None:
            context_reader = getattr(self.analysis_client, "model_context", None)
            if context_reader is None:
                raise RuntimeError("analysis client must expose context sizing")
            return context_token_capacity(
                int(context_reader().configured_tokens),
                ratio_names=("KMD_FILESYSTEM_CHUNK_INPUT_RATIO",),
                ratio_default=0.16,
            )
        return int(
            calculator(
                system=system,
                user_without_content=user_without_content,
                max_tokens=max_tokens,
            )
        )

    def _analysis_output_tokens(self, profile: str) -> int:
        reader = getattr(self.analysis_client, "output_token_budget", None)
        ratio_names = (f"KMD_FILESYSTEM_{profile.upper()}_OUTPUT_RATIO",)
        ratio_default = 1.0 / 16.0 if profile in {"chunk_batch", "file"} else 1.0 / 32.0
        if reader is not None:
            return int(reader(ratio_names=ratio_names, ratio_default=ratio_default))
        context_reader = getattr(self.analysis_client, "model_context", None)
        if context_reader is None:
            raise RuntimeError("analysis client must expose context sizing")
        return context_token_capacity(
            int(context_reader().configured_tokens),
            ratio_names=ratio_names,
            ratio_default=ratio_default,
        )

    def _chunk_batches(self, chunks: Sequence[Chunk], relative_path: str) -> list[list[Chunk]]:
        batches: list[list[Chunk]] = []
        current: list[Chunk] = []
        output_tokens = self._analysis_output_tokens("chunk_batch")
        for chunk in chunks:
            candidate = [*current, chunk]
            if current and not self._analysis_request_fits(
                system=CHUNK_SYSTEM_PROMPT,
                user=self._render_chunks(candidate, relative_path),
                max_tokens=output_tokens,
            ):
                batches.append(current)
                current = [chunk]
            else:
                current = candidate
            if not self._analysis_request_fits(
                system=CHUNK_SYSTEM_PROMPT,
                user=self._render_chunks(current, relative_path),
                max_tokens=output_tokens,
            ):
                raise RuntimeError(
                    f"single chunk exceeds configured analysis context for {relative_path}: "
                    f"chunk={chunk.index} tokens={chunk.token_count}"
                )
        if current:
            batches.append(current)
        return batches


    def _render_chunks(self, chunks: Sequence[Chunk], relative_path: str) -> str:
        sections = []
        for chunk in chunks:
            key = str(chunk.index)
            sections.append(
                f'<chunk key="{key}" start_char="{chunk.start_char}" end_char="{chunk.end_char}">\n{chunk.text}\n</chunk>'
            )
        keys = ", ".join(str(chunk.index) for chunk in chunks)
        return (
            f"File: {relative_path}\nReturn exactly one analysis for each chunk key in this set: [{keys}].\n\n"
            + "\n\n".join(sections)
        )

    def _analyze_chunks(
        self, chunks: Sequence[Chunk], relative_path: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        analyses: list[dict[str, Any] | None] = [None] * len(chunks)
        metadata: list[dict[str, Any] | None] = [None] * len(chunks)
        for batch in self._chunk_batches(chunks, relative_path):
            expected = {str(chunk.index) for chunk in batch}
            returned: dict[str, dict[str, Any]] = {}
            batch_metadata: dict[str, Any] = {}
            try:
                generated = self.analysis_client.complete(
                    schema_name="chunk_facet_analyses",
                    schema=chunk_analysis_schema_for_keys(sorted(expected)),
                    system=CHUNK_SYSTEM_PROMPT,
                    user=self._render_chunks(batch, relative_path),
                    max_tokens=self._analysis_output_tokens("chunk_batch"),
                )
                values = generated.value.get("analyses")
                if isinstance(values, list):
                    for value in values:
                        if not isinstance(value, dict):
                            continue
                        key = str(value.get("chunk_key", ""))
                        if key in expected and key not in returned:
                            returned[key] = value
                batch_metadata = generated.response_metadata
            except Exception as error:
                batch_metadata = {"batch_error": f"{type(error).__name__}: {error}"}
            for chunk in batch:
                key = str(chunk.index)
                value = returned.get(key)
                recovered = False
                if value is None:
                    recovered = True
                    generated = self.analysis_client.complete(
                        schema_name="single_chunk_facet_analysis",
                        schema=chunk_analysis_schema_for_keys([key]),
                        system=CHUNK_SYSTEM_PROMPT,
                        user=self._render_chunks([chunk], relative_path),
                        max_tokens=self._analysis_output_tokens("chunk_single"),
                    )
                    values = generated.value.get("analyses")
                    if not isinstance(values, list) or len(values) != 1 or str(values[0].get("chunk_key")) != key:
                        raise RuntimeError(f"unable to recover chunk {key} for {relative_path}")
                    value = values[0]
                    chunk_metadata = dict(generated.response_metadata)
                else:
                    chunk_metadata = dict(batch_metadata)
                normalized = normalize_analysis(value)
                analyses[chunk.index] = normalized
                chunk_metadata["batch_expected_keys"] = sorted(expected)
                chunk_metadata["batch_returned_keys"] = sorted(returned)
                chunk_metadata["batch_recovery"] = recovered
                metadata[chunk.index] = chunk_metadata
        if any(value is None for value in analyses) or any(value is None for value in metadata):
            raise RuntimeError(f"missing final chunk analyses for {relative_path}")
        return [value for value in analyses if value is not None], [value for value in metadata if value is not None]

    def _render_file_analyses(
        self, analyses: Sequence[dict[str, Any]], relative_path: str
    ) -> str:
        reduced = [
            {"chunk_key": str(index), "analysis": analysis}
            for index, analysis in enumerate(analyses)
        ]
        return f"File: {relative_path}\nOrdered chunk analyses:\n" + json.dumps(
            reduced, ensure_ascii=False
        )

    def _reduce_file_group(
        self, analyses: Sequence[dict[str, Any]], relative_path: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        user = self._render_file_analyses(analyses, relative_path)
        generated = self.analysis_client.complete(
            schema_name="file_facet_analysis",
            schema=FILE_ANALYSIS_SCHEMA,
            system=FILE_SYSTEM_PROMPT,
            user=user,
            max_tokens=self._analysis_output_tokens("file"),
        )
        return normalize_analysis(generated.value), generated.response_metadata

    def _analyze_file(
        self, analyses: Sequence[dict[str, Any]], relative_path: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if len(analyses) <= 1:
            return None, None
        working = list(analyses)
        levels: list[list[dict[str, Any]]] = []
        while len(working) > 1:
            full_user = self._render_file_analyses(working, relative_path)
            if self._analysis_request_fits(
                system=FILE_SYSTEM_PROMPT, user=full_user, max_tokens=self._analysis_output_tokens("file")
            ):
                final, metadata = self._reduce_file_group(working, relative_path)
                metadata = dict(metadata)
                metadata["hierarchical_reduction_levels"] = levels
                return final, metadata
            groups: list[list[dict[str, Any]]] = []
            current: list[dict[str, Any]] = []
            for analysis in working:
                candidate = [*current, analysis]
                candidate_user = self._render_file_analyses(candidate, relative_path)
                if current and not self.analysis_client.request_fits(
                    system=FILE_SYSTEM_PROMPT, user=candidate_user, max_tokens=self._analysis_output_tokens("file")
                ):
                    groups.append(current)
                    current = [analysis]
                else:
                    current = candidate
                if not self._analysis_request_fits(
                    system=FILE_SYSTEM_PROMPT,
                    user=self._render_file_analyses(current, relative_path),
                    max_tokens=self._analysis_output_tokens("file"),
                ):
                    raise RuntimeError(
                        f"one chunk analysis exceeds configured file-reduction context for {relative_path}"
                    )
            if current:
                groups.append(current)
            if len(groups) >= len(working):
                raise RuntimeError(
                    f"file analysis cannot be reduced within configured context for {relative_path}"
                )
            next_level: list[dict[str, Any]] = []
            level_metadata: list[dict[str, Any]] = []
            for group in groups:
                reduced, metadata = self._reduce_file_group(group, relative_path)
                next_level.append(reduced)
                level_metadata.append(metadata)
            levels.append(level_metadata)
            working = next_level
        return working[0], {"hierarchical_reduction_levels": levels}

    def _chunk_row(
        self,
        *,
        entry: sqlite3.Row,
        file_id: str,
        chunk_kind: str,
        chunk_index: int,
        start: int,
        end: int,
        text: str,
        token_count: int,
        vector: np.ndarray | None,
        now_ns: int,
        created_at_ns: int | None = None,
    ) -> dict[str, Any]:
        text_hash = sha256_text(text)
        chunk_id = stable_chunk_id(
            file_id, entry["content_sha256"], chunk_kind, chunk_index, start, end, text_hash
        )
        if vector is None:
            blob = norm = blob_hash = dimension = dtype = embedding_model = revision = None
        else:
            blob, norm, blob_hash = _vector_blob(vector)
            dimension = int(vector.shape[0])
            dtype = "float32"
            embedding_model = self.embedding_client.model
            revision = self.embedding_client.revision
        return {
            "chunk_id": chunk_id,
            "file_id": file_id,
            "collection_id": self.collection_id,
            "filesystem_entry_id": int(entry["id"]),
            "content_object_id": f"sha256:{entry['content_sha256']}",
            "content_sha256": entry["content_sha256"],
            "chunk_kind": chunk_kind,
            "chunk_index": chunk_index,
            "start_char": start,
            "end_char": end,
            "character_count": end - start,
            "word_count": count_words(text),
            "token_count": token_count,
            "text_sha256": text_hash,
            "embedding_model": embedding_model,
            "embedding_model_revision": revision,
            "embedding_dimension": dimension,
            "embedding_dtype": dtype,
            "embedding_norm": norm,
            "embedding_blob": sqlite3.Binary(blob) if blob is not None else None,
            "embedding_sha256": blob_hash,
            "created_at_ns": created_at_ns if created_at_ns is not None else now_ns,
            "updated_at_ns": now_ns,
        }

    def _representation_rows(
        self,
        *,
        chunk_id: str,
        analysis: dict[str, Any],
        metadata: dict[str, Any],
        vectors: Sequence[np.ndarray],
        now_ns: int,
        existing_created: dict[str, int],
    ) -> list[dict[str, Any]]:
        flattened = flatten_representations(analysis)
        if len(flattened) != len(vectors):
            raise RuntimeError("representation vector count mismatch")
        rows: list[dict[str, Any]] = []
        generation_json = canonical_json(metadata)
        for item, vector in zip(flattened, vectors, strict=True):
            representation_id = stable_representation_id(
                chunk_id,
                self.analysis_client.model,
                self.prompt_version,
                self.embedding_client.model,
                int(item["global_rank"]),
            )
            blob, norm, blob_hash = _vector_blob(vector)
            rows.append(
                {
                    "representation_id": representation_id,
                    "chunk_id": chunk_id,
                    "representation_kind": item["kind"],
                    "facet_label": item["facet_label"],
                    "facet_strength": item["facet_strength"],
                    "item_strength": item["item_strength"],
                    "facet_rank": int(item["facet_rank"]),
                    "item_rank": int(item["item_rank"]),
                    "global_rank": int(item["global_rank"]),
                    "representation_text": item["text"],
                    "representation_text_sha256": sha256_text(item["text"]),
                    "analysis_model": self.analysis_client.model,
                    "analysis_model_fingerprint": metadata.get("system_fingerprint"),
                    "prompt_version": self.prompt_version,
                    "generation_seed": self.seed,
                    "pipeline_version": self.pipeline_version,
                    "generation_json": generation_json,
                    "attributes_json": canonical_json({}),
                    "embedding_model": self.embedding_client.model,
                    "embedding_model_revision": self.embedding_client.revision,
                    "embedding_dimension": int(vector.shape[0]),
                    "embedding_dtype": "float32",
                    "embedding_norm": norm,
                    "embedding_blob": sqlite3.Binary(blob),
                    "embedding_sha256": blob_hash,
                    "analysis_status": "complete",
                    "analysis_error": None,
                    "created_at_ns": existing_created.get(representation_id, now_ns),
                    "updated_at_ns": now_ns,
                }
            )
        return rows

    def _process_entry(self, connection: sqlite3.Connection, entry: sqlite3.Row) -> dict[str, Any]:
        text = self._extract_text(entry)
        file_id = stable_file_id(self.collection_id, entry["relative_path_b64"])
        relative_path = entry["relative_path_display"]
        empty_user = self._render_chunks([Chunk(0, 0, 0, "", 0)], relative_path)
        analysis_source_limit = self._analysis_content_limit(
            system=CHUNK_SYSTEM_PROMPT,
            user_without_content=empty_user,
            max_tokens=self._analysis_output_tokens("chunk_single"),
        )
        chunks = chunk_text(
            text,
            self.analysis_client,
            embedding_client=self.embedding_client,
            max_tokens=analysis_source_limit,
        )
        analyses, metadata = self._analyze_chunks(chunks, relative_path)
        file_analysis, file_metadata = self._analyze_file(analyses, entry["relative_path_display"])
        chunk_vectors = self.embedding_client.embed([chunk.text for chunk in chunks])
        representation_texts: list[str] = []
        flattened_per_source: list[list[dict[str, Any]]] = []
        for analysis in analyses:
            flattened = flatten_representations(analysis)
            flattened_per_source.append(flattened)
            representation_texts.extend(item["text"] for item in flattened)
        if file_analysis is not None:
            file_flattened = flatten_representations(file_analysis)
            representation_texts.extend(item["text"] for item in file_flattened)
        else:
            file_flattened = []
        representation_vectors = self.embedding_client.embed(representation_texts)
        now_ns = time.time_ns()
        existing_chunk_created = {
            row[0]: int(row[1])
            for row in connection.execute(
                f"SELECT chunk_id,created_at_ns FROM {CHUNK_TABLE_NAME} WHERE file_id=?", (file_id,)
            )
        }
        existing_representation_created = {
            row[0]: int(row[1])
            for row in connection.execute(
                f"""SELECT r.representation_id,r.created_at_ns FROM {REPRESENTATION_TABLE_NAME} r
                JOIN {CHUNK_TABLE_NAME} c ON c.chunk_id=r.chunk_id WHERE c.file_id=?""",
                (file_id,),
            )
        }
        chunk_rows: list[dict[str, Any]] = []
        representation_rows: list[dict[str, Any]] = []
        vector_offset = 0
        for chunk, analysis, source_metadata, chunk_vector, flattened in zip(
            chunks, analyses, metadata, chunk_vectors, flattened_per_source, strict=True
        ):
            provisional = self._chunk_row(
                entry=entry,
                file_id=file_id,
                chunk_kind="chunk",
                chunk_index=chunk.index,
                start=chunk.start_char,
                end=chunk.end_char,
                text=chunk.text,
                token_count=chunk.token_count,
                vector=chunk_vector,
                now_ns=now_ns,
            )
            provisional["created_at_ns"] = existing_chunk_created.get(provisional["chunk_id"], now_ns)
            chunk_rows.append(provisional)
            count = len(flattened)
            vectors = representation_vectors[vector_offset : vector_offset + count]
            vector_offset += count
            representation_rows.extend(
                self._representation_rows(
                    chunk_id=provisional["chunk_id"],
                    analysis=analysis,
                    metadata=source_metadata,
                    vectors=vectors,
                    now_ns=now_ns,
                    existing_created=existing_representation_created,
                )
            )
        if file_analysis is not None and file_metadata is not None:
            full_tokens = self.analysis_client.token_count(text)
            file_row = self._chunk_row(
                entry=entry,
                file_id=file_id,
                chunk_kind="file",
                chunk_index=0,
                start=0,
                end=len(text),
                text=text,
                token_count=full_tokens,
                vector=None,
                now_ns=now_ns,
            )
            file_row["created_at_ns"] = existing_chunk_created.get(file_row["chunk_id"], now_ns)
            chunk_rows.append(file_row)
            count = len(file_flattened)
            vectors = representation_vectors[vector_offset : vector_offset + count]
            vector_offset += count
            representation_rows.extend(
                self._representation_rows(
                    chunk_id=file_row["chunk_id"],
                    analysis=file_analysis,
                    metadata=file_metadata,
                    vectors=vectors,
                    now_ns=now_ns,
                    existing_created=existing_representation_created,
                )
            )
        if vector_offset != len(representation_vectors):
            raise RuntimeError("unused representation vectors")
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(f"DELETE FROM {CHUNK_TABLE_NAME} WHERE file_id=?", (file_id,))
            _insert_many(connection, CHUNK_TABLE_NAME, CHUNK_COLUMN_NAMES, chunk_rows)
            _insert_many(
                connection,
                REPRESENTATION_TABLE_NAME,
                REPRESENTATION_COLUMN_NAMES,
                representation_rows,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        return {
            "path": entry["relative_path_display"],
            "file_id": file_id,
            "chunks": len(chunks),
            "chunk_rows": len(chunk_rows),
            "representation_rows": len(representation_rows),
            "vectors": len(chunk_vectors) + len(representation_vectors),
        }

    def backfill_chunks(
        self, *, only_paths: set[str] | None = None, max_files: int | None = None
    ) -> dict[str, Any]:
        connection = sqlite3.connect(self.database, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            migrated = migrate_legacy_content_schema(connection, self.root)
            entries = list(
                connection.execute(
                    f"""SELECT id,relative_path_display,relative_path_b64,content_sha256,magic_mime_type,magic_mime_encoding
                    FROM {TABLE_NAME} WHERE entry_type='file' AND content_sha256 IS NOT NULL AND hash_status='complete'
                    ORDER BY relative_path_b64"""
                )
            )
            if only_paths is not None:
                entries = [entry for entry in entries if entry["relative_path_display"] in only_paths]
            if max_files is not None:
                entries = entries[:max_files]
            processed = 0
            for entry in entries:
                text = self._extract_text(entry)
                file_id = stable_file_id(self.collection_id, entry["relative_path_b64"])
                relative_path = entry["relative_path_display"]
                empty_user = self._render_chunks([Chunk(0, 0, 0, "", 0)], relative_path)
                analysis_source_limit = self._analysis_content_limit(
                    system=CHUNK_SYSTEM_PROMPT,
                    user_without_content=empty_user,
                    max_tokens=self._analysis_output_tokens("chunk_single"),
                )
                chunks = chunk_text(
                    text,
                    self.analysis_client,
                    embedding_client=self.embedding_client,
                    max_tokens=analysis_source_limit,
                )
                vectors = self.embedding_client.embed([chunk.text for chunk in chunks])
                now_ns = time.time_ns()
                existing = {
                    row[0]: int(row[1])
                    for row in connection.execute(
                        f"SELECT chunk_id,created_at_ns FROM {CHUNK_TABLE_NAME} WHERE file_id=? AND chunk_kind='chunk'",
                        (file_id,),
                    )
                }
                rows = []
                for chunk, vector in zip(chunks, vectors, strict=True):
                    row = self._chunk_row(
                        entry=entry,
                        file_id=file_id,
                        chunk_kind="chunk",
                        chunk_index=chunk.index,
                        start=chunk.start_char,
                        end=chunk.end_char,
                        text=chunk.text,
                        token_count=chunk.token_count,
                        vector=vector,
                        now_ns=now_ns,
                    )
                    row["created_at_ns"] = existing.get(row["chunk_id"], now_ns)
                    rows.append(row)
                connection.execute("BEGIN IMMEDIATE")
                try:
                    new_chunk_ids = {row["chunk_id"] for row in rows}
                    existing_ids = {
                        row[0]
                        for row in connection.execute(
                            f"SELECT chunk_id FROM {CHUNK_TABLE_NAME} WHERE file_id=? AND chunk_kind='chunk'",
                            (file_id,),
                        )
                    }
                    stale_ids = existing_ids - new_chunk_ids
                    if stale_ids:
                        placeholders = ",".join("?" for _ in stale_ids)
                        connection.execute(
                            f"DELETE FROM {CHUNK_TABLE_NAME} WHERE chunk_id IN ({placeholders})",
                            sorted(stale_ids),
                        )
                    for row in rows:
                        columns = ",".join(f'"{name}"' for name in CHUNK_COLUMN_NAMES)
                        placeholders = ",".join("?" for _ in CHUNK_COLUMN_NAMES)
                        updates = ",".join(
                            f'"{name}"=excluded."{name}"'
                            for name in CHUNK_COLUMN_NAMES
                            if name not in {"chunk_id", "created_at_ns"}
                        )
                        connection.execute(
                            f"INSERT INTO {CHUNK_TABLE_NAME} ({columns}) VALUES ({placeholders}) ON CONFLICT(chunk_id) DO UPDATE SET {updates}",
                            [row[name] for name in CHUNK_COLUMN_NAMES],
                        )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
                processed += 1
            return {"status": "ok", "migrated_schema": migrated, "processed_files": processed}
        finally:
            connection.close()

    def backfill_raw_chunks(
        self, *, only_paths: set[str] | None = None, max_files: int | None = None
    ) -> dict[str, Any]:
        return self.backfill_chunks(only_paths=only_paths, max_files=max_files)

    def run(self, *, only_paths: set[str] | None = None, max_files: int | None = None) -> dict[str, Any]:
        if not self.database.exists():
            raise FileNotFoundError(self.database)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        analysis_health = self.analysis_client.health()
        embedding_health = self.embedding_client.health()
        started = time.monotonic()
        connection = sqlite3.connect(self.database, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            migrated = migrate_legacy_content_schema(connection, self.root)
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            expected = {TABLE_NAME, CHUNK_TABLE_NAME, REPRESENTATION_TABLE_NAME}
            if tables != expected:
                raise RuntimeError(f"database tables differ: expected={sorted(expected)}, actual={sorted(tables)}")
            entries = list(
                connection.execute(
                    f"""SELECT id,relative_path_display,relative_path_b64,content_sha256,magic_mime_type,magic_mime_encoding
                    FROM {TABLE_NAME} WHERE entry_type='file' AND content_sha256 IS NOT NULL AND hash_status='complete'
                    ORDER BY relative_path_b64"""
                )
            )
            if only_paths is not None:
                missing = only_paths - {entry["relative_path_display"] for entry in entries}
                if missing:
                    raise RuntimeError(f"requested paths are missing from filesystem catalog: {sorted(missing)}")
                entries = [entry for entry in entries if entry["relative_path_display"] in only_paths]
            if max_files is not None:
                entries = entries[:max_files]
            results = []
            for index, entry in enumerate(entries, start=1):
                result = self._process_entry(connection, entry)
                results.append(result)
                print(canonical_json({"content_progress": index, "total": len(entries), **result}), flush=True)
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
            if integrity != "ok" or foreign_keys:
                raise RuntimeError(f"database validation failed: integrity={integrity}, foreign_keys={foreign_keys[:10]}")
            return {
                "status": "ok",
                "migrated_schema": migrated,
                "processed_files": len(results),
                "chunk_rows": connection.execute(f"SELECT count(*) FROM {CHUNK_TABLE_NAME}").fetchone()[0],
                "representation_rows": connection.execute(f"SELECT count(*) FROM {REPRESENTATION_TABLE_NAME}").fetchone()[0],
                "duration_seconds": round(time.monotonic() - started, 6),
                "analysis_health": analysis_health,
                "embedding_health": embedding_health,
                "files": results,
            }
        finally:
            connection.close()


def vector_from_blob(blob: bytes, dimension: int, dtype: str) -> np.ndarray:
    if dtype == "float32":
        value = np.frombuffer(blob, dtype="<f4")
    elif dtype == "float16":
        value = np.frombuffer(blob, dtype="<f2").astype(np.float32)
    elif dtype == "int8":
        value = np.frombuffer(blob, dtype=np.int8).astype(np.float32)
    else:
        raise ValueError(dtype)
    if value.shape != (dimension,):
        raise ValueError(f"vector blob shape differs: {value.shape} != {(dimension,)}")
    return value


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def search_semantic_entries(connection: sqlite3.Connection, query_vector: np.ndarray) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in connection.execute(
        f"""SELECT c.chunk_id,c.file_id,c.filesystem_entry_id,f.relative_path_display,f.relative_path_b64,
        c.chunk_kind,c.chunk_index,c.start_char,c.end_char,c.embedding_dimension,c.embedding_dtype,c.embedding_blob
        FROM {CHUNK_TABLE_NAME} c JOIN {TABLE_NAME} f ON f.id=c.filesystem_entry_id
        WHERE c.embedding_blob IS NOT NULL"""
    ):
        vector = vector_from_blob(row[11], int(row[9]), row[10])
        candidates.append(
            {
                "file_id": row[1],
                "filesystem_entry_id": int(row[2]),
                "relative_path_display": row[3],
                "relative_path_b64": row[4],
                "chunk_id": row[0],
                "source_level": row[5],
                "source_index": int(row[6]),
                "source_start_char": int(row[7]),
                "source_end_char": int(row[8]),
                "analysis_kind": "chunk",
                "analysis_text": "",
                "score": cosine_similarity(query_vector, vector),
            }
        )
    for row in connection.execute(
        f"""SELECT r.representation_id,r.representation_kind,r.representation_text,r.embedding_dimension,
        r.embedding_dtype,r.embedding_blob,c.chunk_id,c.file_id,c.filesystem_entry_id,f.relative_path_display,
        f.relative_path_b64,c.chunk_kind,c.chunk_index,c.start_char,c.end_char
        FROM {REPRESENTATION_TABLE_NAME} r JOIN {CHUNK_TABLE_NAME} c ON c.chunk_id=r.chunk_id
        JOIN {TABLE_NAME} f ON f.id=c.filesystem_entry_id WHERE r.analysis_status='complete'"""
    ):
        vector = vector_from_blob(row[5], int(row[3]), row[4])
        candidates.append(
            {
                "file_id": row[7],
                "filesystem_entry_id": int(row[8]),
                "relative_path_display": row[9],
                "relative_path_b64": row[10],
                "chunk_id": row[6],
                "representation_id": row[0],
                "source_level": row[11],
                "source_index": int(row[12]),
                "source_start_char": int(row[13]),
                "source_end_char": int(row[14]),
                "analysis_kind": row[1],
                "analysis_text": row[2],
                "score": cosine_similarity(query_vector, vector),
            }
        )
    best: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        previous = best.get(str(candidate["file_id"]))
        if previous is None or float(candidate["score"]) > float(previous["score"]):
            best[str(candidate["file_id"])] = candidate
    return sorted(best.values(), key=lambda item: (-float(item["score"]), str(item["relative_path_display"])))


def _read_file_text(root: Path, relative_path_b64: str) -> str:
    relative = base64.b64decode(relative_path_b64)
    path = os.fsencode(root) + (b"/" + relative if relative else b"")
    return _read_regular_file_bytes(
        path,
        max_bytes=_content_file_memory_limit_bytes(),
    ).decode("utf-8", "replace")


def search_literal_chunks(
    connection: sqlite3.Connection,
    root: os.PathLike[str] | str,
    query: str,
    *,
    case_sensitive: bool = False,
    whole_word: bool = False,
    max_matches: int | None = None,
    excerpt_characters: int | None = None,
) -> list[dict[str, Any]]:
    if not query:
        raise ValueError("query must not be empty")
    flags = 0 if case_sensitive else re.IGNORECASE
    escaped = re.escape(query)
    pattern = re.compile(rf"(?<!\w){escaped}(?!\w)" if whole_word else escaped, flags)
    root_path = Path(root).resolve()
    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    cache: dict[str, str] = {}
    rows = connection.execute(
        f"""SELECT c.chunk_id,c.file_id,c.filesystem_entry_id,f.relative_path_display,f.relative_path_b64,
        c.chunk_index,c.start_char,c.end_char FROM {CHUNK_TABLE_NAME} c
        JOIN {TABLE_NAME} f ON f.id=c.filesystem_entry_id WHERE c.chunk_kind='chunk'
        ORDER BY f.relative_path_display,c.chunk_index"""
    )
    for row in rows:
        path_key = row[4]
        text = cache.get(path_key)
        if text is None:
            text = _read_file_text(root_path, path_key)
            cache[path_key] = text
        start, end = int(row[6]), int(row[7])
        chunk_text_value = text[start:end]
        for found in pattern.finditer(chunk_text_value):
            absolute_start = start + found.start()
            absolute_end = start + found.end()
            key = (str(row[1]), absolute_start, absolute_end)
            if key in seen:
                continue
            seen.add(key)
            if excerpt_characters is None:
                excerpt_start = 0
                excerpt_end = len(chunk_text_value)
            else:
                excerpt_start = max(0, found.start() - excerpt_characters)
                excerpt_end = min(len(chunk_text_value), found.end() + excerpt_characters)
            matches.append(
                {
                    "file_id": row[1],
                    "filesystem_entry_id": int(row[2]),
                    "relative_path_display": row[3],
                    "relative_path_b64": row[4],
                    "chunk_id": row[0],
                    "source_index": int(row[5]),
                    "source_start_char": start,
                    "source_end_char": end,
                    "match_start_char": absolute_start,
                    "match_end_char": absolute_end,
                    "matched_text": found.group(0),
                    "excerpt": chunk_text_value[excerpt_start:excerpt_end],
                }
            )
            if max_matches is not None and len(matches) >= max_matches:
                return matches
    return matches
