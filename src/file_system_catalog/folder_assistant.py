from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from context_capacity import context_char_capacity, context_token_capacity, schema_array_capacity

from .content_pipeline import (
    AnalysisClient,
    ContentSemanticPipeline,
    EmbeddingClient,
    canonical_json,
    cosine_similarity,
    migrate_legacy_content_schema,
    search_literal_chunks,
    search_semantic_entries,
    vector_from_blob,
)
from .content_schema import CHUNK_TABLE_NAME
from .scanner import FilesystemScanner
from .schema import TABLE_NAME

PLAN_ACTION_TYPES = ("literal", "semantic", "metadata")
ANSWER_MODES = ("files", "answer", "count", "summary", "comparison", "metadata")
COMBINE_MODES = ("union", "intersection", "independent")
SORT_MODES = ("path", "size_desc", "size_asc", "mtime_desc", "mtime_asc")
ANSWER_STATUSES = ("answered", "partial", "not_found")


def _closed_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def query_plan_schema() -> dict[str, Any]:
    action = _closed_object(
        {
            "action_type": {"type": "string", "enum": list(PLAN_ACTION_TYPES)},
            "purpose": {"type": "string", "x-kmd-string-profile": "reason"},
            "query": {"type": "string", "x-kmd-string-profile": "value"},
            "case_sensitive": {"type": "boolean"},
            "whole_word": {"type": "boolean"},
            "top_k": {"type": "integer"},
            "path_contains": {"type": "string", "x-kmd-string-profile": "value"},
            "name_contains": {"type": "string", "x-kmd-string-profile": "value"},
            "extension": {"type": "string", "x-kmd-string-profile": "short"},
            "mime_prefix": {"type": "string", "x-kmd-string-profile": "short"},
            "min_size_bytes": {"type": "integer"},
            "max_size_bytes": {"type": "integer"},
            "modified_after": {"type": "string", "x-kmd-string-profile": "short"},
            "modified_before": {"type": "string", "x-kmd-string-profile": "short"},
            "sort_by": {"type": "string", "enum": list(SORT_MODES)},
            "limit": {"type": "integer"},
        }
    )
    return _closed_object(
        {
            "answer_mode": {"type": "string", "enum": list(ANSWER_MODES)},
            "combine_mode": {"type": "string", "enum": list(COMBINE_MODES)},
            "actions": {"type": "array", "x-kmd-array-profile": "compact", "items": action},
            "rationale": {"type": "string", "x-kmd-string-profile": "reason"},
        }
    )


def answer_schema(evidence_ids: Sequence[str]) -> dict[str, Any]:
    evidence_enum = list(evidence_ids) or ["NONE"]
    citation = _closed_object(
        {
            "evidence_id": {"type": "string", "enum": evidence_enum},
            "claim": {"type": "string", "x-kmd-string-profile": "reason"},
        }
    )
    file_item = _closed_object(
        {
            "path": {"type": "string", "x-kmd-string-profile": "value"},
            "reason": {"type": "string", "x-kmd-string-profile": "reason"},
            "evidence_ids": {
                "type": "array",
                "x-kmd-array-profile": "dense",
                "items": {"type": "string", "enum": evidence_enum},
            },
        }
    )
    return _closed_object(
        {
            "status": {"type": "string", "enum": list(ANSWER_STATUSES)},
            "answer": {"type": "string", "x-kmd-string-profile": "reason"},
            "files": {"type": "array", "x-kmd-array-profile": "dense", "items": file_item},
            "citations": {"type": "array", "x-kmd-array-profile": "dense", "items": citation},
        }
    )


PLAN_SYSTEM_PROMPT = """You plan searches over a text-folder catalog. Return only schema-valid JSON and never answer the user's question. Use literal search for exact words, exact phrases, quotations, identifiers or wording such as contains, says, occurs or literally mentions. Use semantic search for concepts, paraphrases, related subjects, synonyms and ordinary content questions. If wording such as mentions could mean either exact occurrence or conceptual subject, use both literal and semantic actions. Use metadata search only for path, filename, extension, MIME type, size, modification time, file counts or ordering. Split genuinely independent concepts into separate semantic actions. Use intersection only when every concept must occur in the same file, union when any concept is acceptable, and independent for comparison or separate reporting. Do not search each ordinary word separately. Keep actions distinct and necessary. For irrelevant fields use empty strings, false or zero. Use positive top_k and limit values; the runtime will derive their maximums from the active model context. ISO dates may be placed in modified_after or modified_before; otherwise use empty strings."""

ANSWER_SYSTEM_PROMPT = """Answer the user's folder question using only the supplied tool evidence. Do not use outside knowledge. Distinguish literal matches from semantic matches. A semantic match means conceptual similarity, not that the exact words occurred. If the evidence is insufficient, say so and use partial or not_found. Every factual claim must be supported by one or more citation entries. Cite evidence IDs in the answer using square brackets such as [E1]. Do not invent files, paths, passages, counts or relationships. For file-list questions, include every supported returned file up to the evidence supplied. Keep the answer direct and useful."""


def default_collection_id(root: os.PathLike[str] | str) -> str:
    resolved = str(Path(root).resolve())
    digest = hashlib.sha256(resolved.encode("utf-8", "surrogatepass")).hexdigest()[:20]
    return f"text-folder:{digest}"


def normalize_plan(value: dict[str, Any], question: str, *, context_size: int) -> dict[str, Any]:
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    plan_output = context_token_capacity(
        context_size,
        ratio_names=("KMD_FOLDER_PLAN_OUTPUT_RATIO",),
        ratio_default=1.0 / 32.0,
    )
    action_capacity = schema_array_capacity(plan_output, "compact")
    result_capacity = context_token_capacity(
        context_size,
        ratio_names=("KMD_FOLDER_RESULT_COUNT_RATIO",),
        ratio_default=1.0 / 1024.0,
    )
    default_top_k = max(1, result_capacity // 2)
    answer_mode = str(value.get("answer_mode", "answer"))
    if answer_mode not in ANSWER_MODES:
        answer_mode = "answer"
    combine_mode = str(value.get("combine_mode", "union"))
    if combine_mode not in COMBINE_MODES:
        combine_mode = "union"
    raw_actions = value.get("actions")
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(raw_actions, list):
        for raw in raw_actions[:action_capacity]:
            if not isinstance(raw, dict):
                continue
            action_type = str(raw.get("action_type", ""))
            if action_type not in PLAN_ACTION_TYPES:
                continue
            query = " ".join(str(raw.get("query", "")).split()).strip()
            path_contains = " ".join(str(raw.get("path_contains", "")).split()).strip()
            name_contains = " ".join(str(raw.get("name_contains", "")).split()).strip()
            extension = str(raw.get("extension", "")).strip().lstrip(".")
            mime_prefix = str(raw.get("mime_prefix", "")).strip()
            if action_type in {"literal", "semantic"} and not query:
                continue
            top_k = max(1, min(result_capacity, int(raw.get("top_k", default_top_k) or default_top_k)))
            limit = max(1, min(result_capacity, int(raw.get("limit", result_capacity) or result_capacity)))
            min_size = max(0, int(raw.get("min_size_bytes", 0) or 0))
            max_size = max(0, int(raw.get("max_size_bytes", 0) or 0))
            sort_by = str(raw.get("sort_by", "path"))
            if sort_by not in SORT_MODES:
                sort_by = "path"
            normalized = {
                "action_type": action_type,
                "purpose": " ".join(str(raw.get("purpose", "")).split()).strip(),
                "query": query,
                "case_sensitive": bool(raw.get("case_sensitive", False)),
                "whole_word": bool(raw.get("whole_word", False)),
                "top_k": top_k,
                "path_contains": path_contains,
                "name_contains": name_contains,
                "extension": extension,
                "mime_prefix": mime_prefix,
                "min_size_bytes": min_size,
                "max_size_bytes": max_size,
                "modified_after": str(raw.get("modified_after", "")).strip(),
                "modified_before": str(raw.get("modified_before", "")).strip(),
                "sort_by": sort_by,
                "limit": limit,
            }
            key = canonical_json(normalized)
            if key in seen:
                continue
            seen.add(key)
            actions.append(normalized)
    action_types = {action["action_type"] for action in actions}
    if answer_mode == "metadata" and "metadata" not in action_types:
        answer_mode = "files" if re.search(r"\b(files?|documents?)\b", question, re.IGNORECASE) else "answer"
    if answer_mode == "count" and "metadata" not in action_types:
        answer_mode = "answer"
    if len(actions) == 1 and combine_mode == "independent":
        combine_mode = "union"
    if not actions:
        actions.append({
            "action_type": "semantic",
            "purpose": "Find passages relevant to the question.",
            "query": question,
            "case_sensitive": False,
            "whole_word": False,
            "top_k": default_top_k,
            "path_contains": "",
            "name_contains": "",
            "extension": "",
            "mime_prefix": "",
            "min_size_bytes": 0,
            "max_size_bytes": 0,
            "modified_after": "",
            "modified_before": "",
            "sort_by": "path",
            "limit": result_capacity,
        })
    return {
        "answer_mode": answer_mode,
        "combine_mode": combine_mode,
        "actions": actions,
        "rationale": " ".join(str(value.get("rationale", "")).split()).strip(),
    }


def _parse_iso_ns(value: str) -> int | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_metadata(connection: sqlite3.Connection, action: dict[str, Any]) -> dict[str, Any]:
    where = ["entry_type='file'"]
    parameters: list[Any] = []
    if action["path_contains"]:
        where.append("relative_path_display LIKE ? ESCAPE '\\'")
        parameters.append(f"%{_escape_like(action['path_contains'])}%")
    if action["name_contains"]:
        where.append("name_display LIKE ? ESCAPE '\\'")
        parameters.append(f"%{_escape_like(action['name_contains'])}%")
    if action["extension"]:
        where.append("lower(ltrim(extension,'.'))=lower(?)")
        parameters.append(action["extension"])
    if action["mime_prefix"]:
        where.append("coalesce(magic_mime_type,extension_mime_type,'') LIKE ?")
        parameters.append(f"{_escape_like(action['mime_prefix'])}%")
    if action["min_size_bytes"]:
        where.append("coalesce(size_bytes,0)>=?")
        parameters.append(action["min_size_bytes"])
    if action["max_size_bytes"]:
        where.append("coalesce(size_bytes,0)<=?")
        parameters.append(action["max_size_bytes"])
    after = _parse_iso_ns(action["modified_after"])
    before = _parse_iso_ns(action["modified_before"])
    if after is not None:
        where.append("mtime_ns>=?")
        parameters.append(after)
    if before is not None:
        where.append("mtime_ns<=?")
        parameters.append(before)
    order_sql = {
        "path": "relative_path_display COLLATE NOCASE,id",
        "size_desc": "coalesce(size_bytes,0) DESC,relative_path_display COLLATE NOCASE",
        "size_asc": "coalesce(size_bytes,0),relative_path_display COLLATE NOCASE",
        "mtime_desc": "coalesce(mtime_ns,0) DESC,relative_path_display COLLATE NOCASE",
        "mtime_asc": "coalesce(mtime_ns,0),relative_path_display COLLATE NOCASE",
    }[action["sort_by"]]
    predicate = " AND ".join(where)
    total = int(
        connection.execute(
            f"SELECT count(*) FROM {TABLE_NAME} WHERE {predicate}", parameters
        ).fetchone()[0]
    )
    rows = list(
        connection.execute(
            f"""SELECT id,relative_path_display,relative_path_b64,name_display,extension,
            coalesce(magic_mime_type,extension_mime_type) AS mime_type,size_bytes,mtime_ns,mtime_iso,content_sha256
            FROM {TABLE_NAME} WHERE {predicate} ORDER BY {order_sql} LIMIT ?""",
            [*parameters, action["limit"]],
        )
    )
    return {
        "total_count": total,
        "rows": [dict(row) if isinstance(row, sqlite3.Row) else {
            "id": row[0], "relative_path_display": row[1], "relative_path_b64": row[2],
            "name_display": row[3], "extension": row[4], "mime_type": row[5],
            "size_bytes": row[6], "mtime_ns": row[7], "mtime_iso": row[8],
            "content_sha256": row[9],
        } for row in rows],
    }


def _read_source(root: Path, relative_path_b64: str) -> str:
    relative = base64.b64decode(relative_path_b64)
    path = os.fsencode(root) + (b"/" + relative if relative else b"")
    with open(path, "rb") as handle:
        return handle.read().decode("utf-8", "replace")


def _window_text(text: str, absolute_start: int, *, target: int, overlap: int) -> list[dict[str, Any]]:
    if len(text) <= target:
        return [{"text": text, "start": absolute_start, "end": absolute_start + len(text)}]
    windows: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        desired = min(len(text), cursor + target)
        end = desired
        if desired < len(text):
            search_start = max(cursor + target // 2, desired - overlap)
            candidates = [
                text.rfind("\n\n", search_start, desired),
                text.rfind(". ", search_start, desired),
                text.rfind(" ", search_start, desired),
            ]
            valid = [value for value in candidates if value > cursor]
            if valid:
                end = max(valid)
                if text[end : end + 2] == ". ":
                    end += 1
        if end <= cursor:
            end = desired
        windows.append(
            {
                "text": text[cursor:end],
                "start": absolute_start + cursor,
                "end": absolute_start + end,
            }
        )
        if end >= len(text):
            break
        cursor = max(cursor + 1, end - overlap)
    return windows


def _best_chunk_row(
    connection: sqlite3.Connection, file_id: str, query_vector: np.ndarray
) -> sqlite3.Row | None:
    best: tuple[float, sqlite3.Row] | None = None
    rows = connection.execute(
        f"""SELECT c.*,f.relative_path_display,f.relative_path_b64 FROM {CHUNK_TABLE_NAME} c
        JOIN {TABLE_NAME} f ON f.id=c.filesystem_entry_id
        WHERE c.file_id=? AND c.chunk_kind='chunk' AND c.embedding_blob IS NOT NULL""",
        (file_id,),
    )
    for row in rows:
        vector = vector_from_blob(
            row["embedding_blob"], int(row["embedding_dimension"]), row["embedding_dtype"]
        )
        score = cosine_similarity(query_vector, vector)
        if best is None or score > best[0]:
            best = (score, row)
    return best[1] if best else None


def semantic_evidence(
    connection: sqlite3.Connection,
    root: Path,
    embedding_client: EmbeddingClient,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    query_vector = embedding_client.embed([query])[0]
    ranked = search_semantic_entries(connection, query_vector)[:top_k]
    embedding_context = int(embedding_client.model_context().configured_tokens)
    window_target = context_char_capacity(
        embedding_context,
        ratio_names=("KMD_FOLDER_SEMANTIC_WINDOW_RATIO",),
        ratio_default=1.0 / 64.0,
    )
    window_overlap = context_char_capacity(
        embedding_context,
        ratio_names=("KMD_FOLDER_SEMANTIC_OVERLAP_RATIO",),
        ratio_default=1.0 / 512.0,
    )
    evidence: list[dict[str, Any]] = []
    for result in ranked:
        chunk = _best_chunk_row(connection, str(result["file_id"]), query_vector)
        if chunk is None:
            continue
        source = _read_source(root, chunk["relative_path_b64"])
        start, end = int(chunk["start_char"]), int(chunk["end_char"])
        chunk_text = source[start:end]
        windows = _window_text(
            chunk_text,
            start,
            target=window_target,
            overlap=window_overlap,
        )
        window_vectors = embedding_client.embed([window["text"] for window in windows])
        scored = sorted(
            (
                cosine_similarity(query_vector, vector),
                index,
            )
            for index, vector in enumerate(window_vectors)
        )
        best_score, best_index = scored[-1]
        selected = windows[best_index]
        evidence.append(
            {
                "retrieval_type": "semantic",
                "search_query": query,
                "path": result["relative_path_display"],
                "relative_path_b64": result["relative_path_b64"],
                "filesystem_entry_id": result["filesystem_entry_id"],
                "file_id": result["file_id"],
                "chunk_id": chunk["chunk_id"],
                "start_char": selected["start"],
                "end_char": selected["end"],
                "score": float(result["score"]),
                "window_score": float(best_score),
                "matched_kind": result["analysis_kind"],
                "matched_representation": result.get("analysis_text", ""),
                "excerpt": selected["text"],
            }
        )
    return evidence


def initialize_text_folder(
    *,
    root: os.PathLike[str] | str,
    database: os.PathLike[str] | str,
    analysis_client: AnalysisClient,
    embedding_client: EmbeddingClient,
    collection_id: str | None = None,
    replace: bool = False,
    chunks_only: bool = False,
    progress_every: int = 0,
    max_hash_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    destination = Path(database).resolve()
    if destination == root_path or root_path in destination.parents:
        raise ValueError("database must be outside the indexed text root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise FileExistsError(destination)
    partial = destination.with_name(f".{destination.name}.initialize.{os.getpid()}")
    partial.unlink(missing_ok=True)
    started = time.monotonic()
    try:
        scan = FilesystemScanner(
            root_path,
            max_hash_bytes=max_hash_bytes,
            progress_every=progress_every,
        ).scan_to_database(partial, replace=True)
        pipeline = ContentSemanticPipeline(
            database=partial,
            root=root_path,
            collection_id=collection_id or default_collection_id(root_path),
            analysis_client=analysis_client,
            embedding_client=embedding_client,
            seed=analysis_client.seed,
        )
        semantic = pipeline.backfill_chunks() if chunks_only else pipeline.run()
        connection = sqlite3.connect(partial)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
            if integrity != "ok" or foreign_keys:
                raise RuntimeError(
                    f"initialized database validation failed: integrity={integrity}, foreign_keys={foreign_keys}"
                )
        finally:
            connection.close()
        os.replace(partial, destination)
        scan["database"] = str(destination)
        return {
            "status": "ok",
            "root": str(root_path),
            "database": str(destination),
            "collection_id": collection_id or default_collection_id(root_path),
            "scan": scan,
            "semantic": semantic,
            "duration_seconds": round(time.monotonic() - started, 6),
        }
    except Exception:
        partial.unlink(missing_ok=True)
        raise


class FolderQuestionAssistant:
    def __init__(
        self,
        *,
        root: os.PathLike[str] | str,
        database: os.PathLike[str] | str,
        analysis_client: AnalysisClient,
        embedding_client: EmbeddingClient,
    ) -> None:
        self.root = Path(root).resolve()
        self.database = Path(database).resolve()
        self.analysis_client = analysis_client
        self.embedding_client = embedding_client
        self.context_size = int(self.analysis_client.model_context().configured_tokens)
        self.plan_output_tokens = self.analysis_client.output_token_budget(
            ratio_names=("KMD_FOLDER_PLAN_OUTPUT_RATIO",), ratio_default=1.0 / 32.0
        )
        self.answer_output_tokens = self.analysis_client.output_token_budget(
            ratio_names=("KMD_FOLDER_ANSWER_OUTPUT_RATIO",), ratio_default=1.0 / 16.0
        )
        self.max_evidence = schema_array_capacity(self.answer_output_tokens, "dense")
        self.literal_excerpt_characters = context_char_capacity(
            self.context_size,
            ratio_names=("KMD_FOLDER_LITERAL_EXCERPT_RATIO",),
            ratio_default=1.0 / 64.0,
        )

    def plan(self, question: str) -> dict[str, Any]:
        generated = self.analysis_client.complete(
            schema_name="folder_query_plan",
            schema=query_plan_schema(),
            system=PLAN_SYSTEM_PROMPT,
            user=f"User question:\n{question}",
            max_tokens=self.plan_output_tokens,
        )
        return normalize_plan(generated.value, question, context_size=self.context_size)

    def _execute(self, plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        connection = sqlite3.connect(self.database, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        action_results: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        try:
            migrate_legacy_content_schema(connection, self.root)
            for action_index, action in enumerate(plan["actions"], start=1):
                action_type = action["action_type"]
                if action_type == "literal":
                    matches = search_literal_chunks(
                        connection,
                        self.root,
                        action["query"],
                        case_sensitive=action["case_sensitive"],
                        whole_word=action["whole_word"],
                        max_matches=action["limit"],
                        excerpt_characters=self.literal_excerpt_characters,
                    )
                    rows = []
                    for item in matches:
                        row = {
                            "retrieval_type": "literal",
                            "search_query": action["query"],
                            "path": item["relative_path_display"],
                            "relative_path_b64": item["relative_path_b64"],
                            "filesystem_entry_id": item["filesystem_entry_id"],
                            "file_id": item["file_id"],
                            "chunk_id": item["chunk_id"],
                            "start_char": item["match_start_char"],
                            "end_char": item["match_end_char"],
                            "score": None,
                            "matched_kind": "literal",
                            "matched_representation": item["matched_text"],
                            "excerpt": item["excerpt"],
                        }
                        rows.append(row)
                        evidence.append(row)
                    action_results.append(
                        {
                            "action_index": action_index,
                            "action": action,
                            "result_count": len(matches),
                            "file_paths": sorted({item["relative_path_display"] for item in matches}),
                        }
                    )
                elif action_type == "semantic":
                    rows = semantic_evidence(
                        connection,
                        self.root,
                        self.embedding_client,
                        action["query"],
                        action["top_k"],
                    )
                    evidence.extend(rows)
                    action_results.append(
                        {
                            "action_index": action_index,
                            "action": action,
                            "result_count": len(rows),
                            "file_paths": [row["path"] for row in rows],
                        }
                    )
                else:
                    result = search_metadata(connection, action)
                    action_results.append(
                        {
                            "action_index": action_index,
                            "action": action,
                            "result_count": result["total_count"],
                            "file_paths": [row["relative_path_display"] for row in result["rows"]],
                        }
                    )
                    evidence.append(
                        {
                            "retrieval_type": "metadata_summary",
                            "search_query": action["purpose"],
                            "path": "",
                            "relative_path_b64": "",
                            "filesystem_entry_id": 0,
                            "file_id": "",
                            "chunk_id": "",
                            "start_char": 0,
                            "end_char": 0,
                            "score": None,
                            "matched_kind": "metadata_count",
                            "matched_representation": str(result["total_count"]),
                            "excerpt": f"Metadata query matched {result['total_count']} files.",
                        }
                    )
                    for item in result["rows"]:
                        evidence.append(
                            {
                                "retrieval_type": "metadata",
                                "search_query": action["purpose"],
                                "path": item["relative_path_display"],
                                "relative_path_b64": item["relative_path_b64"],
                                "filesystem_entry_id": item["id"],
                                "file_id": "",
                                "chunk_id": "",
                                "start_char": 0,
                                "end_char": 0,
                                "score": None,
                                "matched_kind": "metadata",
                                "matched_representation": "",
                                "excerpt": canonical_json(
                                    {
                                        "name": item["name_display"],
                                        "extension": item["extension"],
                                        "mime_type": item["mime_type"],
                                        "size_bytes": item["size_bytes"],
                                        "mtime_iso": item["mtime_iso"],
                                        "content_sha256": item["content_sha256"],
                                    }
                                ),
                            }
                        )
        finally:
            connection.close()
        if plan["combine_mode"] == "intersection" and len(action_results) > 1:
            path_sets = [set(result["file_paths"]) for result in action_results]
            common = set.intersection(*path_sets) if path_sets else set()
            evidence = [item for item in evidence if not item["path"] or item["path"] in common]
        deduplicated: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for item in evidence:
            key = (
                item["retrieval_type"],
                item["path"],
                item["start_char"],
                item["end_char"],
                item["excerpt"],
            )
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(item)
        for index, item in enumerate(deduplicated[: self.max_evidence], start=1):
            item["evidence_id"] = f"E{index}"
        return action_results, deduplicated[: self.max_evidence]

    def answer(
        self,
        question: str,
        plan: dict[str, Any],
        action_results: Sequence[dict[str, Any]],
        evidence: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        evidence_payload = [
            {
                "evidence_id": item["evidence_id"],
                "retrieval_type": item["retrieval_type"],
                "path": item["path"],
                "start_char": item["start_char"],
                "end_char": item["end_char"],
                "score": item["score"],
                "matched_kind": item["matched_kind"],
                "matched_representation": item["matched_representation"],
                "excerpt": item["excerpt"],
            }
            for item in evidence
        ]
        ids = [item["evidence_id"] for item in evidence]
        generated = self.analysis_client.complete(
            schema_name="grounded_folder_answer",
            schema=answer_schema(ids),
            system=ANSWER_SYSTEM_PROMPT,
            user=(
                f"Question:\n{question}\n\nSearch plan:\n{json.dumps(plan, ensure_ascii=False)}"
                f"\n\nTool result summaries:\n{json.dumps(list(action_results), ensure_ascii=False)}"
                f"\n\nEvidence:\n{json.dumps(evidence_payload, ensure_ascii=False)}"
            ),
            max_tokens=self.answer_output_tokens,
        )
        result = generated.value
        valid_ids = set(ids)
        citations = [
            citation
            for citation in result.get("citations", [])
            if citation.get("evidence_id") in valid_ids
        ]
        files = []
        for item in result.get("files", []):
            if not isinstance(item, dict):
                continue
            evidence_ids = [value for value in item.get("evidence_ids", []) if value in valid_ids]
            files.append(
                {
                    "path": str(item.get("path", "")),
                    "reason": str(item.get("reason", "")),
                    "evidence_ids": evidence_ids,
                }
            )
        return {
            "status": result.get("status", "partial"),
            "answer": str(result.get("answer", "")),
            "files": files,
            "citations": citations,
        }

    def ask(self, question: str) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question must not be empty")
        plan = self.plan(question)
        action_results, evidence = self._execute(plan)
        answer = self.answer(question, plan, action_results, evidence)
        return {
            "question": question,
            "plan": plan,
            "action_results": action_results,
            "evidence": evidence,
            "evidence_capacity": self.max_evidence,
            "result": answer,
        }
