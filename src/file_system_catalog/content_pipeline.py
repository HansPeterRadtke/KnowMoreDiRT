from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

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

PIPELINE_VERSION = "0.6.0"
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


def request_json(url: str, payload: dict[str, Any] | None = None, *, timeout: int = 600) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "backslashreplace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body[:2000]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"request failed for {url}: {error}") from error
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as error:
        raise RuntimeError(f"invalid JSON from {url}: {raw[:2000]!r}") from error


class AnalysisClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        seed: int = DEFAULT_SEED,
        temperature: float = 0.0,
        timeout: int = 900,
        retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self._token_cache: dict[str, int] = {}

    def health(self) -> dict[str, Any]:
        return request_json(f"{self.base_url}/health", timeout=30)

    def token_count(self, text: str) -> int:
        digest = sha256_text(text)
        cached = self._token_cache.get(digest)
        if cached is not None:
            return cached
        response = request_json(
            f"{self.base_url}/tokenize",
            {"content": text, "add_special": False},
            timeout=self.timeout,
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"tokenizer response has no token list: {response}")
        count = len(tokens)
        self._token_cache[digest] = count
        return count

    def complete(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        system: str,
        user: str,
        max_tokens: int,
    ) -> GeneratedAnalysis:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            attempt_max_tokens = int(math.ceil(max_tokens * (1.5 ** (attempt - 1))))
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": attempt_max_tokens,
                "provider": {"require_parameters": True},
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            try:
                response = request_json(
                    f"{self.base_url}/v1/chat/completions",
                    payload,
                    timeout=self.timeout,
                )
                choice = response["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise RuntimeError(f"generation did not finish cleanly: {choice.get('finish_reason')}")
                message = choice.get("message", {})
                if message.get("reasoning_content"):
                    raise RuntimeError("reasoning mode was not disabled")
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("generation returned no content")
                value = json.loads(content)
                metadata = {
                    "attempt": attempt,
                    "max_tokens": attempt_max_tokens,
                    "model": response.get("model"),
                    "system_fingerprint": response.get("system_fingerprint"),
                    "usage": response.get("usage"),
                    "timings": response.get("timings"),
                    "finish_reason": choice.get("finish_reason"),
                    "parsed": value,
                }
                return GeneratedAnalysis(value=value, response_metadata=metadata)
            except Exception as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise RuntimeError(f"analysis failed after {self.retries} attempts: {last_error}") from last_error


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        revision: str,
        expected_dimension: int = 1024,
        timeout: int = 300,
        batch_size: int = 32,
        max_batch_characters: int = 60000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.revision = revision
        self.expected_dimension = expected_dimension
        self.timeout = timeout
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_batch_characters < 1:
            raise ValueError("max_batch_characters must be positive")
        self.batch_size = batch_size
        self.max_batch_characters = max_batch_characters

    def health(self) -> dict[str, Any]:
        return request_json(f"{self.base_url}/health", timeout=30)

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        batches: list[list[str]] = []
        current: list[str] = []
        current_characters = 0
        for text in texts:
            if current and (
                len(current) >= self.batch_size
                or current_characters + len(text) > self.max_batch_characters
            ):
                batches.append(current)
                current = []
                current_characters = 0
            current.append(text)
            current_characters += len(text)
        if current:
            batches.append(current)
        for batch in batches:
            response = request_json(
                f"{self.base_url}/v1/embeddings",
                {"model": self.model, "input": batch},
                timeout=self.timeout,
            )
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
                "description": "Faithful retrieval text preserving the source meaning and grammar.",
            },
        }
    )


def _facet_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "facet_name": {
                "type": "string",
                "description": "A concise semantic topic name such as power resilience or tax compliance. Never use a strength word such as essential, strong, moderate, weak, or very_weak here.",
            },
            "facet_strength": {
                "type": "string",
                "enum": list(STRENGTH_VALUES),
                "description": "Importance of this semantic topic within the source.",
            },
            "representations": {"type": "array", "items": _representation_schema()},
        }
    )


def chunk_analysis_schema_for_keys(keys: Sequence[str]) -> dict[str, Any]:
    if not keys:
        raise ValueError("keys must not be empty")
    analysis = _closed_object(
        {
            "chunk_key": {"type": "string", "enum": list(keys)},
            "document_summary": {"type": "string"},
            "facets": {"type": "array", "items": _facet_schema()},
        }
    )
    return _closed_object({"analyses": {"type": "array", "items": analysis}})


FILE_ANALYSIS_SCHEMA = _closed_object(
    {
        "document_summary": {"type": "string"},
        "facets": {"type": "array", "items": _facet_schema()},
    }
)

STRENGTH_GUIDANCE = """Use verbal strength labels consistently: essential means indispensable to recognizing the source as a whole; very_strong means a major independent theme; strong means a substantial secondary theme; moderate means a useful supporting subject; weak means a specific detail; very_weak means a narrow detail with limited retrieval value."""

CHUNK_SYSTEM_PROMPT = f"""You create retrieval representations for source-document chunks. Return only schema-valid JSON. First identify every distinct meaningful semantic facet, including minority facets, and order facets from most to least important. The facet_name field must contain the actual subject, such as power resilience, bicycle courier backup, tax compliance, or privacy controls; it must never contain an importance label such as essential, very_strong, strong, moderate, weak, or very_weak. Put importance only in facet_strength and item_strength. Within each facet, return every distinct useful representation supported by the text, ordered from most to least useful. Use complete sentences when precise grammar, actors, objects, negation, chronology or qualifications matter; use concise descriptions and keyphrases for concepts; use single words only when the isolated word is independently useful. Preserve original meaning, punctuation, capitalization and factual qualifications. Do not pad with near-duplicates. For meaningless gibberish, return a truthful summary and an empty facets array. {STRENGTH_GUIDANCE} Never merge or omit requested chunk keys, even when chunks are duplicated."""

FILE_SYSTEM_PROMPT = f"""You combine ordered chunk analyses into a file-level retrieval description. Return only schema-valid JSON. Identify every distinct meaningful file facet, including brief facets that appear in only one chunk, and order facets from most to least important. The facet_name field must be the actual semantic subject and must never be an importance label. Put importance only in facet_strength and item_strength. Within each facet, return every distinct useful representation, ordered from most to least useful. Preserve long-distance references, corrections, negation, actors and objects. Repetition must not multiply importance. Keep ambiguous senses separate. Ignore meaningless identifiers. {STRENGTH_GUIDANCE}"""


def _split_oversized_span(text: str, start: int, end: int, max_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        desired = min(end, cursor + max_chars)
        cut = desired
        if desired < end:
            search_start = max(cursor + max_chars // 2, desired - 2500)
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
    target_chars: int = 36000,
    max_chars: int = 42000,
    overlap_chars: int = 2500,
    max_tokens: int = 10500,
) -> list[Chunk]:
    if not text:
        return [Chunk(index=0, start_char=0, end_char=0, text="", token_count=0)]
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
    for start, end in chunk_spans:
        value = text[start:end]
        count = analysis_client.token_count(value)
        if count <= max_tokens:
            validated.append((start, end, count))
            continue
        ratio = max_tokens / count
        split_chars = max(4000, int((end - start) * ratio * 0.88))
        for split_start, split_end in _split_oversized_span(text, start, end, split_chars, overlap_chars):
            split_text = text[split_start:split_end]
            split_count = analysis_client.token_count(split_text)
            if split_count > max_tokens:
                raise RuntimeError(f"unable to split chunk below token limit: {split_count} > {max_tokens}")
            validated.append((split_start, split_end, split_count))
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
            if len(label) > 180:
                label = label[:177].rstrip() + "..."
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
            with open(path, "rb") as handle:
                file_text = handle.read().decode("utf-8", "replace")
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
        chunk_batch_size: int = 3,
        chunk_batch_token_budget: int = 25000,
    ) -> None:
        if not collection_id.strip():
            raise ValueError("collection_id must not be empty")
        if not 1 <= chunk_batch_size <= 3:
            raise ValueError("chunk_batch_size must be between one and three")
        if chunk_batch_token_budget < 1000:
            raise ValueError("chunk_batch_token_budget must be at least one thousand")
        self.database = Path(database).resolve()
        self.root = Path(root).resolve()
        self.collection_id = collection_id
        self.analysis_client = analysis_client
        self.embedding_client = embedding_client
        self.prompt_version = prompt_version
        self.pipeline_version = pipeline_version
        self.seed = seed
        self.chunk_batch_size = chunk_batch_size
        self.chunk_batch_token_budget = chunk_batch_token_budget

    def _extract_text(self, entry: sqlite3.Row) -> str:
        relative = base64.b64decode(entry["relative_path_b64"])
        path = os.fsencode(self.root) + (b"/" + relative if relative else b"")
        with open(path, "rb") as handle:
            data = handle.read()
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

    def _chunk_batches(self, chunks: Sequence[Chunk]) -> list[list[Chunk]]:
        batches: list[list[Chunk]] = []
        current: list[Chunk] = []
        tokens = 0
        for chunk in chunks:
            if current and (
                len(current) >= self.chunk_batch_size
                or tokens + chunk.token_count > self.chunk_batch_token_budget
            ):
                batches.append(current)
                current = []
                tokens = 0
            if chunk.token_count > self.chunk_batch_token_budget:
                raise RuntimeError(
                    f"single chunk exceeds analysis batch token budget: {chunk.token_count}"
                )
            current.append(chunk)
            tokens += chunk.token_count
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
        for batch in self._chunk_batches(chunks):
            expected = {str(chunk.index) for chunk in batch}
            returned: dict[str, dict[str, Any]] = {}
            batch_metadata: dict[str, Any] = {}
            try:
                generated = self.analysis_client.complete(
                    schema_name="chunk_facet_analyses",
                    schema=chunk_analysis_schema_for_keys(sorted(expected)),
                    system=CHUNK_SYSTEM_PROMPT,
                    user=self._render_chunks(batch, relative_path),
                    max_tokens=2200,
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
                        max_tokens=1800,
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

    def _analyze_file(
        self, analyses: Sequence[dict[str, Any]], relative_path: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if len(analyses) <= 1:
            return None, None
        reduced = [
            {"chunk_key": str(index), "analysis": analysis}
            for index, analysis in enumerate(analyses)
        ]
        generated = self.analysis_client.complete(
            schema_name="file_facet_analysis",
            schema=FILE_ANALYSIS_SCHEMA,
            system=FILE_SYSTEM_PROMPT,
            user=f"File: {relative_path}\nOrdered chunk analyses:\n" + json.dumps(reduced, ensure_ascii=False),
            max_tokens=2600,
        )
        return normalize_analysis(generated.value), generated.response_metadata

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
        chunks = chunk_text(text, self.analysis_client)
        analyses, metadata = self._analyze_chunks(chunks, entry["relative_path_display"])
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
                chunks = chunk_text(text, self.analysis_client)
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
    with open(path, "rb") as handle:
        return handle.read().decode("utf-8", "replace")


def search_literal_chunks(
    connection: sqlite3.Connection,
    root: os.PathLike[str] | str,
    query: str,
    *,
    case_sensitive: bool = False,
    whole_word: bool = False,
    max_matches: int | None = None,
    excerpt_characters: int = 120,
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
