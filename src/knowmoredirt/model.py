"""Strict Structured Outputs client. There is no prompt-only semantic fallback."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .schemas import assert_portable_closed_schema


class ModelError(RuntimeError):
    pass


def _chat_endpoint(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value + "/v1/chat/completions"


def _validate(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    if "enum" in schema and value not in schema["enum"]:
        raise ModelError(f"schema enum violation at {path}")
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            raise ModelError(f"expected object at {path}")
        properties = schema["properties"]
        missing = set(schema["required"]) - set(value)
        extra = set(value) - set(properties)
        if missing or extra:
            raise ModelError(f"object keys invalid at {path}: missing={sorted(missing)} extra={sorted(extra)}")
        for key, child in value.items():
            _validate(child, properties[key], f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ModelError(f"expected array at {path}")
        for index, child in enumerate(value):
            _validate(child, schema["items"], f"{path}[{index}]")
    elif kind == "string" and not isinstance(value, str):
        raise ModelError(f"expected string at {path}")
    elif kind == "boolean" and not isinstance(value, bool):
        raise ModelError(f"expected boolean at {path}")
    elif kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ModelError(f"expected integer at {path}")
    elif kind == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ModelError(f"expected number at {path}")


class StrictModelClient:
    def __init__(self, endpoint: str | None = None, cache_dir: str | Path | None = None):
        self.endpoint = endpoint or os.environ.get("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:14829/v1")
        self.cache_dir = Path(cache_dir or os.environ.get("KMD_MODEL_CACHE_DIR", "/data/var/knowmoredirt/model_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(os.environ.get("KMD_LOCAL_MODEL_TIMEOUT_SECONDS", "600"))

    def complete_json(self, stage: str, prompt: str, schema: dict[str, Any], max_tokens: int = 4096) -> dict[str, Any]:
        assert_portable_closed_schema(schema)
        material = json.dumps(
            {"stage": stage, "prompt": prompt, "schema": schema, "endpoint": self.endpoint, "version": "strict-model-owned-v2", "max_tokens": max_tokens},
            sort_keys=True,
            ensure_ascii=False,
        )
        key = hashlib.sha256(material.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            _validate(cached, schema)
            return cached
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "system",
                    "content": "Return exactly one object matching the strict JSON Schema. Do not reveal hidden reasoning and do not add prose.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": float(os.environ.get("KMD_LOCAL_MODEL_TEMPERATURE", "0")),
            "top_p": float(os.environ.get("KMD_LOCAL_MODEL_TOP_P", "1")),
            "seed": int(os.environ.get("KMD_LOCAL_MODEL_SEED", "1778779265")),
            "stream": False,
            "cache_prompt": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": f"kmd_{stage}", "strict": True, "schema": schema},
            },
        }
        if "openrouter.ai" in self.endpoint:
            body["provider"] = {"require_parameters": True}
        request = urllib.request.Request(
            _chat_endpoint(self.endpoint),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8", errors="replace"))
                content = payload["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                _validate(parsed, schema)
                temp = cache_path.with_suffix(".tmp")
                temp.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temp.replace(cache_path)
                return parsed
            except (OSError, urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, ModelError) as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(0.5)
        raise ModelError(f"strict model call failed during {stage}: {last_error}")
