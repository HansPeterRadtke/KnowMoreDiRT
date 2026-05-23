"""Optional local-model integration hooks.

KMD never requires a cloud model. When enabled, this module talks only to a
local llama.cpp-compatible endpoint and returns raw, source-grounded JSON
objects to the engine. The public API remains ``initialize`` and ``question``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any


def _completion_endpoint(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    if value.endswith("/v1"):
        return value[:-3] + "/completion"
    if value.endswith("/completion"):
        return value
    return value + "/completion"


def _chat_endpoint(endpoint: str) -> str | None:
    value = endpoint.rstrip("/")
    if value.endswith("/v1"):
        return value + "/chat/completions"
    if value.endswith("/chat/completions"):
        return value
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


@dataclass(frozen=True)
class LocalModelClient:
    endpoint: str = os.environ.get("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:14829/v1")
    timeout_seconds: float = float(os.environ.get("KMD_LOCAL_MODEL_TIMEOUT", "180"))

    def models(self) -> dict:
        url = self.endpoint.rstrip("/") + "/models"
        with urllib.request.urlopen(url, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def complete_json(self, prompt: str, *, n_predict: int = 128, grammar: str | None = None) -> dict[str, Any]:
        """Return a parsed JSON object from the local completion endpoint."""

        endpoint = _chat_endpoint(self.endpoint) or _completion_endpoint(self.endpoint)
        if not (
            endpoint.startswith("http://127.0.0.1:")
            or endpoint.startswith("http://localhost:")
            or endpoint.startswith("http://[::1]:")
        ):
            raise ValueError("KMD local model endpoint must be localhost-only")
        if endpoint.endswith("/chat/completions"):
            body = {
                "messages": [
                    {"role": "system", "content": "Return one valid JSON object or array and no prose."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": int(n_predict),
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": int(os.environ.get("KMD_LOCAL_MODEL_SEED", "1778779265")),
                "stream": False,
            }
        else:
            body = {
                "prompt": prompt,
                "n_predict": int(n_predict),
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": int(os.environ.get("KMD_LOCAL_MODEL_SEED", "1778779265")),
                "stream": False,
            }
        if grammar:
            body["grammar"] = grammar
        started = time.time()
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            response_obj = json.loads(response.read().decode("utf-8", errors="replace"))
        raw = str(response_obj.get("content") or "")
        if not raw and isinstance(response_obj.get("choices"), list) and response_obj["choices"]:
            choice = response_obj["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    raw = str(message.get("content") or "")
                else:
                    raw = str(choice.get("text") or "")
        snippet = _extract_balanced_json(raw) or raw
        parsed = json.loads(snippet)
        if isinstance(parsed, list):
            parsed = {"items": parsed}
        if not isinstance(parsed, dict):
            raise ValueError("local model did not return a JSON object or array")
        parsed["_model_raw"] = raw
        parsed["_model_elapsed_seconds"] = round(time.time() - started, 3)
        parsed["_model_endpoint"] = endpoint
        return parsed
