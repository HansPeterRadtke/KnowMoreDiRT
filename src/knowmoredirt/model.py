"""Optional local-model integration hooks.

KMD never requires a cloud model. When enabled, this module talks only to a
local llama.cpp-compatible endpoint and returns raw, source-grounded JSON
objects to the engine. The public API remains ``initialize`` and ``question``.
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


class LocalModelJSONError(ValueError):
    """Raised when the local model response cannot be parsed as JSON."""

    def __init__(self, message: str, *, raw_text: str, snippet: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.snippet = snippet


@dataclass
class LocalModelClient:
    endpoint: str = os.environ.get("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:14829/v1")
    timeout_seconds: float = float(os.environ.get("KMD_LOCAL_MODEL_TIMEOUT", "180"))
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

    def cache_fingerprint(self) -> dict[str, Any]:
        metadata = self.server_metadata()
        return {
            "endpoint": self.endpoint,
            "model_id": self.model_id(metadata),
            "context_size": self.context_size(metadata),
            "context_source": self.context_source(metadata),
            "timeout_seconds": self.timeout_seconds,
            "request_settings": self.request_settings(),
        }

    def complete_json(
        self,
        prompt: str,
        *,
        n_predict: int = 128,
        grammar: str | None = None,
        json_schema: dict[str, Any] | None = None,
        stream: bool | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from the local completion endpoint."""

        api = os.environ.get("KMD_LOCAL_MODEL_API", "completion").strip().lower()
        endpoint = _chat_endpoint(self.endpoint) if api == "chat" else _completion_endpoint(self.endpoint)
        if endpoint is None:
            endpoint = _completion_endpoint(self.endpoint)
        _local_endpoint_required(endpoint)
        settings = self.request_settings()
        use_stream = (
            stream
            if stream is not None
            else os.environ.get("KMD_LOCAL_MODEL_STREAM", "1").strip().lower() not in {"0", "false", "no", "off"}
        )
        if endpoint.endswith("/chat/completions"):
            body = {
                "messages": [
                    {"role": "system", "content": "Return one valid JSON object or array and no prose."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": int(n_predict),
                "temperature": settings["temperature"],
                "top_p": settings["top_p"],
                "seed": settings["seed"],
                "stream": bool(use_stream),
            }
        else:
            body = {
                "prompt": prompt,
                "n_predict": int(n_predict),
                "temperature": settings["temperature"],
                "top_p": settings["top_p"],
                "top_k": settings["top_k"],
                "min_p": settings["min_p"],
                "repeat_penalty": settings["repeat_penalty"],
                "seed": settings["seed"],
                "stream": bool(use_stream),
            }
        if grammar:
            body["grammar"] = grammar
        if json_schema:
            body["json_schema"] = json_schema
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
            if use_stream:
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
                        if _extract_balanced_json(raw):
                            stream_closed_after_json = True
                            break
            else:
                response_obj = json.loads(response.read().decode("utf-8", errors="replace"))
                raw = _response_content(response_obj)
        snippet = _extract_balanced_json(raw) or raw
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise LocalModelJSONError(str(exc), raw_text=raw, snippet=snippet) from exc
        if isinstance(parsed, list):
            parsed = {"items": parsed}
        if not isinstance(parsed, dict):
            raise ValueError("local model did not return a JSON object or array")
        parsed["_model_raw"] = raw
        parsed["_model_elapsed_seconds"] = round(time.time() - started, 3)
        parsed["_model_endpoint"] = endpoint
        parsed["_model_stream"] = bool(use_stream)
        parsed["_model_stream_closed_after_json"] = stream_closed_after_json
        parsed["_model_context_size"] = self.context_size()
        parsed["_model_id"] = self.model_id()
        parsed["_model_request_settings"] = {**settings, "n_predict": int(n_predict)}
        return parsed
