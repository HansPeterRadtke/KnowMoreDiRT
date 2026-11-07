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
    FILE_ANALYSIS_SCHEMA,
    normalize_analysis,
    _guard_system_prompt,
    stream_chat_completion_json,
)
from context_capacity import contextualize_json_schema
from jsonschema import Draft202012Validator


class LocalSummaryModel:
    def __init__(
        self,
        base_url: str,
        *,
        model: str = "",
        timeout_seconds: float = 150.0,
        seed: int = 42,
        context_cap_tokens: int | None = None,
    ):
        if not base_url.startswith(("http://127.0.0.1:", "http://localhost:", "http://10.")):
            raise ValueError("only an explicitly configured local trusted model endpoint is allowed")
        if context_cap_tokens is not None and not 1024 <= context_cap_tokens <= 262144:
            raise ValueError("context cap outside safe bounds")
        self._context_cap_tokens = context_cap_tokens
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
        actual=self.client.model_context().configured_tokens
        return min(actual, self._context_cap_tokens) if self._context_cap_tokens else actual

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
    def generate_retrieval_analysis(
        self, *, system: str, user: str, output_tokens: int
    ) -> dict:
        """One constrained LLM call for native KMD summary AND facets.

        Reuses FILE_ANALYSIS_SCHEMA and normalize_analysis from the original
        KMD pipeline. This adapter only enforces an actual bounded generation
        reservation, unlike the older unrestricted AnalysisClient.complete.
        """
        prompt_cost = self.prompt_tokens(system=system, user=user)
        if prompt_cost + output_tokens >= self.context_tokens:
            raise RuntimeError("native summary/facets exceed reserved context")
        if output_tokens < 256:
            raise ValueError("native facet output budget is too small")
        schema=contextualize_json_schema(
            FILE_ANALYSIS_SCHEMA,
            context_size=self.context_tokens,
            output_tokens=output_tokens,
        )
        Draft202012Validator.check_schema(schema)
        payload={
            "model":self.client.effective_model(),
            "messages":[
                {"role":"system","content":_guard_system_prompt(system)},
                {"role":"user","content":user},
            ],
            "temperature":0.0,
            "seed":self.client.seed,
            "max_tokens":output_tokens,
            "stream":True,
            "enable_thinking":False,
            "reasoning_format":"deepseek",
            "reasoning_budget":0,
            "chat_template_kwargs":{"enable_thinking":False},
            "response_format":{
                "type":"json_schema",
                "json_schema":{
                    "name":"file_facet_analysis",
                    "strict":True,
                    "schema":schema,
                },
            },
        }
        cache_key=semantic_request_hash({
            "contract":"kmd-native-facet-recursive-summary-v1",
            "model":model_content_fingerprint(self.client.effective_model()),
            "context":self.context_tokens,
            "prompt_sha":hashlib.sha256(
                self.client._render_prompt(system=system,user=user).encode("utf-8")
            ).hexdigest(),
            "request":{k:v for k,v in payload.items() if k!="stream"},
        })
        cached=read_model_call(cache_key)
        if cached is not None and isinstance(cached.get("native_analysis"),dict):
            return normalize_analysis(cached["native_analysis"])
        response=stream_chat_completion_json(
            f"{self.client.base_url}/v1/chat/completions",
            payload,
            per_token_timeout_seconds=self.client.per_token_timeout_seconds,
        )
        if response.get("finish_reason")!="stop":
            raise RuntimeError("local summary did not finish within reserved generation tokens")
        if response.get("reasoning_content"):
            raise RuntimeError("local structured summary unexpectedly emitted reasoning")
        raw=json.loads(response.get("content") or "{}")
        Draft202012Validator(schema).validate(raw)
        normalized=normalize_analysis(raw)
        # The original E5 index has a 512-token maximum. Leave room for its
        # passage prefix and avoid truncating the root summary silently.
        if self.text_tokens(normalized["document_summary"])>400:
            raise RuntimeError("native summary exceeds safe embedding context")
        if self.text_tokens(normalized["document_summary"])>output_tokens:
            raise RuntimeError("native summary exceeds generation reservation")
        write_model_call(cache_key,{"native_analysis":normalized})
        return normalized
