"""Local-model integration hooks.

KMD never uses a cloud model. Normal runtime requires a reachable localhost
llama.cpp-compatible endpoint and returns raw, source-grounded JSON objects to
the engine. The public API remains ``initialize`` and ``question``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


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
    raw = os.environ.get("KMD_LOCAL_MODEL_PER_TOKEN_TIMEOUT_SECONDS", "180").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 180.0
    return value if value > 0 else 180.0


def _default_min_constrained_json_tokens() -> int:
    raw = os.environ.get("KMD_LOCAL_MODEL_MIN_CONSTRAINED_JSON_TOKENS", "4096").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 4096
    return value if value > 0 else 4096

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


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


def _fetch_json(url: str, timeout: float) -> Any:
    _local_endpoint_required(url)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


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
    return ("gpt" in compact and "oss" in compact) or "harmony" in compact


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


def _json_only_user_prompt(prompt: str, json_schema: dict[str, Any] | None) -> str:
    return (
        prompt.rstrip()
        + _schema_hint(json_schema)
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


def _append_missing_json_closers(snippet: str) -> str | None:
    text = snippet.strip()
    if not text or text[0] not in "{[":
        return None
    expected: list[str] = []
    in_string = False
    escape = False
    for char in text:
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
        elif char == "{":
            expected.append("}")
        elif char == "[":
            expected.append("]")
        elif char in "}]":
            if not expected or expected[-1] != char:
                return None
            expected.pop()
    if in_string or escape or not expected or len(expected) > 8:
        return None
    candidate = text + "".join(reversed(expected))
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return candidate


class LocalModelJSONError(ValueError):
    """Raised when the local model response cannot be parsed as JSON."""

    def __init__(self, message: str, *, raw_text: str, snippet: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.snippet = snippet


class LocalModelUnavailableError(RuntimeError):
    """Raised when required localhost llama.cpp access is unavailable."""

    def __init__(self, message: str, *, cache_context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.cache_context = cache_context or {}


@dataclass
class LocalModelClient:
    endpoint: str = os.environ.get("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:14829/v1")
    # Socket/read timeout between streamed token chunks. Not a whole-answer wall timeout.
    timeout_seconds: float = field(default_factory=_default_per_token_timeout_seconds)
    _metadata: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def models(self) -> dict:
        return _fetch_json(_models_endpoint(self.endpoint), self.timeout_seconds)

    def server_metadata(self, *, refresh: bool = False) -> dict[str, Any]:
        """Best-effort llama.cpp runtime metadata used for budgeting and cache keys."""

        if self._metadata is not None and not refresh:
            return self._metadata
        root = _server_root(self.endpoint)
        timeout = max(1.0, min(self.timeout_seconds, float(os.environ.get("KMD_LOCAL_MODEL_METADATA_TIMEOUT", "8"))))
        metadata: dict[str, Any] = {"endpoint": self.endpoint, "root": root, "errors": {}}
        for name, path in {
            "models": "/v1/models",
            "slots": "/slots",
            "props": "/props",
        }.items():
            try:
                metadata[name] = _fetch_json(root + path, timeout)
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
        if _first_int(os.environ.get("KMD_LOCAL_MODEL_CONTEXT_SIZE")):
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
        return _first_int(os.environ.get("KMD_LOCAL_MODEL_CONTEXT_SIZE"))

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
        return _first_text(os.environ.get("KMD_LOCAL_MODEL_ID"), self.endpoint, "local-llama")

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
        return {
            "seed": _env_int("KMD_LOCAL_MODEL_SEED", 1778779265),
            "temperature": _env_float("KMD_LOCAL_MODEL_TEMPERATURE", 0.0),
            "top_p": _env_float("KMD_LOCAL_MODEL_TOP_P", 1.0),
            "top_k": _env_int("KMD_LOCAL_MODEL_TOP_K", _first_int(defaults.get("top_k")) or 40),
            "min_p": _env_float("KMD_LOCAL_MODEL_MIN_P", float(defaults.get("min_p") or 0.05)),
            "repeat_penalty": _env_float("KMD_LOCAL_MODEL_REPEAT_PENALTY", float(defaults.get("repeat_penalty") or 1.0)),
        }

    def transport_settings(self) -> dict[str, Any]:
        model_id = self.model_id()
        constrained_mode = os.environ.get("KMD_LOCAL_MODEL_CONSTRAINT_MODE", "auto").strip().lower() or "auto"
        if constrained_mode not in {"auto", "native", "prompt"}:
            constrained_mode = "auto"
        reasoning_control_model = _model_id_looks_like_reasoning_control_token_model(model_id)
        native_constraints = constrained_mode == "native" or (constrained_mode == "auto" and not reasoning_control_model)
        return {
            "api": os.environ.get("KMD_LOCAL_MODEL_API", "chat").strip().lower() or "chat",
            "cache_prompt": os.environ.get("KMD_LOCAL_MODEL_CACHE_PROMPT", "1").strip().lower()
            not in {"0", "false", "no", "off"},
            "min_constrained_json_tokens": _default_min_constrained_json_tokens(),
            "constraint_mode": constrained_mode,
            "native_constraints": native_constraints,
            "reasoning_control_token_model": reasoning_control_model,
        }

    def cache_fingerprint(self) -> dict[str, Any]:
        metadata = self.server_metadata()
        return {
            "fingerprint_schema": "local-model-stable-v2",
            "model_id": self.model_id(metadata),
            "context_size": self.context_size(metadata),
            "context_source": self.context_source(metadata),
            "request_settings": self.request_settings(),
            "transport_settings": self.transport_settings(),
        }

    def complete_json(
        self,
        prompt: str,
        *,
        n_predict: int = 128,
        grammar: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from the local completion endpoint."""

        api = os.environ.get("KMD_LOCAL_MODEL_API", "chat").strip().lower()
        endpoint = _chat_endpoint(self.endpoint) if api == "chat" else _completion_endpoint(self.endpoint)
        if endpoint is None:
            endpoint = _completion_endpoint(self.endpoint)
        _local_endpoint_required(endpoint)
        settings = self.request_settings()
        transport = self.transport_settings()
        native_constraints = bool(transport.get("native_constraints"))
        effective_prompt = _json_only_user_prompt(prompt, json_schema)
        send_thinking_controls = os.environ.get("KMD_LOCAL_MODEL_SEND_THINKING_CONTROLS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        # Local model calls must stream. The timeout below is only the socket/read
        # timeout while waiting for the next streamed token chunk. There is no
        # whole-answer, whole-question, or whole-chunk wall timeout here.
        use_cache_prompt = os.environ.get("KMD_LOCAL_MODEL_CACHE_PROMPT", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        requested_n_predict = int(n_predict)
        effective_n_predict = requested_n_predict
        if json_schema or grammar:
            effective_n_predict = max(effective_n_predict, _default_min_constrained_json_tokens())
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
        elif native_constraints and grammar:
            body["grammar"] = grammar
            constraint_settings = {"mode": "grammar"}
        elif json_schema:
            constraint_settings = {"mode": "prompt_json_schema"}
        elif grammar:
            constraint_settings = {"mode": "prompt_grammar"}
        if send_thinking_controls:
            body["reasoning_format"] = "hidden"
            body["enable_thinking"] = False
        started = time.time()
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw = ""
        response_obj: dict[str, Any] = {}
        stream_closed_after_json = False
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    response_obj = event
                    raw += _event_content(event) or ""
                    if not raw and not native_constraints:
                        _event_reasoning_content(event)
                    if _extract_balanced_json(raw):
                        stream_closed_after_json = True
                        break
        snippet = _extract_balanced_json(raw) or raw
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError as exc:
            repaired_snippet = _append_missing_json_closers(snippet)
            if repaired_snippet is None:
                raise LocalModelJSONError(str(exc), raw_text=raw, snippet=snippet) from exc
            snippet = repaired_snippet
            parsed = json.loads(snippet)
        if isinstance(parsed, list):
            parsed = {"items": parsed}
        if not isinstance(parsed, dict):
            raise ValueError("local model did not return a JSON object or array")
        parsed["_model_raw"] = raw
        parsed["_model_elapsed_seconds"] = round(time.time() - started, 3)
        parsed["_model_endpoint"] = endpoint
        parsed["_model_stream"] = True
        parsed["_model_per_token_timeout_seconds"] = self.timeout_seconds
        parsed["_model_stream_closed_after_json"] = stream_closed_after_json
        parsed["_model_context_size"] = self.context_size()
        parsed["_model_id"] = self.model_id()
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
        return parsed
