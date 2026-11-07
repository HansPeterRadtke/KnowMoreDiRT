"""Local llama.cpp-compatible summary model using exact native prompt token counts.

Does not use cloud APIs. Implements a finite max_output reservation (unlike the
general KMD facet pipeline, whose output allocation uses all remaining context).
Only accepted source-grounded summary strings are cached in KMD's local cache.
"""
from __future__ import annotations

import hashlib
import json

from kmd_model_call_cache import (
    model_content_fingerprint,
    read_model_call,
    semantic_request_hash,
    write_model_call,
)
from .content_pipeline import (
    AnalysisClient,
    _guard_system_prompt,
    stream_chat_completion_json,
)


class LocalSummaryModel:
    def __init__(
        self,
        base_url: str,
        *,
        model: str = "",
        timeout_seconds: float = 150.0,
        seed: int = 42,
    ):
        if not base_url.startswith(("http://127.0.0.1:", "http://localhost:", "http://10.")):
            raise ValueError("only an explicitly configured local trusted model endpoint is allowed")
        self.client = AnalysisClient(
            base_url,
            model=model,
            seed=seed,
            temperature=0.0,
            per_token_timeout_seconds=timeout_seconds,
            control_timeout_seconds=min(timeout_seconds, 30),
            retries=1,
        )

    @property
    def context_tokens(self) -> int:
        return self.client.model_context().configured_tokens

    def prompt_tokens(self, *, system: str, user: str) -> int:
        return self.client.prompt_token_count(system=system, user=user)

    def text_tokens(self, text: str) -> int:
        return self.client.token_count(text)

    def generate_summary(self, *, system: str, user: str, output_tokens: int) -> str:
        prompt_cost = self.prompt_tokens(system=system, user=user)
        if prompt_cost + output_tokens >= self.context_tokens:
            raise RuntimeError("summary request exceeds configured context/output reservation")
        effective_model = self.client.effective_model()
        schema = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        }
        payload = {
            "model": effective_model,
            "messages": [
                {"role": "system", "content": _guard_system_prompt(system)},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "seed": self.client.seed,
            "max_tokens": output_tokens,
            "stream": True,
            "enable_thinking": False,
            "reasoning_format": "deepseek",
            "reasoning_budget": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "kmd_source_summary_v1",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        cache_key = semantic_request_hash({
            "contract": "kmd-summary-context-reserved-v1",
            "model": model_content_fingerprint(effective_model),
            "context": self.context_tokens,
            "prompt_sha": hashlib.sha256(
                self.client._render_prompt(system=system, user=user).encode("utf-8")
            ).hexdigest(),
            "request": {key: value for key, value in payload.items() if key != "stream"},
        })
        cached = read_model_call(cache_key)
        if cached and isinstance(cached.get("summary"), str) and cached["summary"].strip():
            return cached["summary"]
        response = stream_chat_completion_json(
            f"{self.client.base_url}/v1/chat/completions",
            payload,
            per_token_timeout_seconds=self.client.per_token_timeout_seconds,
        )
        if response.get("finish_reason") != "stop":
            raise RuntimeError("local summary did not finish within reserved generation tokens")
        if response.get("reasoning_content"):
            raise RuntimeError("local summary model unexpectedly emitted non-disabled reasoning")
        data = json.loads(response.get("content") or "{}")
        if not isinstance(data, dict) or not isinstance(data.get("summary"), str) or not data["summary"].strip():
            raise RuntimeError("invalid empty or unstructured summary from local model")
        if set(data) != {"summary"}:
            raise RuntimeError("unexpected fields in local model summary")
        summary = data["summary"]
        if self.text_tokens(summary) > output_tokens:
            raise RuntimeError("summary content exceeded output token budget")
        write_model_call(cache_key, {"summary": summary})
        return summary
