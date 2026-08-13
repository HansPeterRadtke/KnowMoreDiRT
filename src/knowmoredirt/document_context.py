"""Long-range document context layered over source-local cached chunk DRS.

Chunk DRS remains independently cacheable.  This module builds a compact map of
cross-chunk discourse and temporal scopes from chunk boundary excerpts, then
links that map into the in-memory DSPG store.  It exists for cases where a
header, footer, or middle transition changes how material in other chunks must
be interpreted (dream, report, quote, hypothetical, document-wide date, etc.).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from context_capacity import context_char_capacity
from kmd_runtime_config import boolean as _config_boolean, floating as _config_float, text as _config_text, model_cache_dir as _model_cache_dir

from .model import LocalModelClient, LocalModelUnavailableError, complete_json_with_transport_retry
from .models import Document, Sentence
from .store import DSPGStore, stable_id

DOCUMENT_CONTEXT_POLICY = "document-context-map-v4-implicit-sleep-scope"
CONTEXT_KINDS = (
    "dreamed",
    "reported",
    "quoted",
    "believed",
    "hypothetical",
    "fictional",
    "possible",
    "uncertain",
)


@dataclass(frozen=True)
class DocumentContextEnvelope:
    applies: bool
    kind: str
    direction: str
    evidence_text: str
    holder_surface: str
    reason: str
    confidence: float
    start_chunk: int = 0
    end_chunk: int = -1
    evidence_chunk: int = 0


@dataclass(frozen=True)
class DocumentTemporalScope:
    temporal_value: str
    evidence_text: str
    reason: str
    confidence: float
    start_chunk: int
    end_chunk: int
    evidence_chunk: int


def _enabled() -> bool:
    return _config_boolean("KMD_DOCUMENT_CONTEXT_ENVELOPES")


def _minimum_confidence() -> float:
    value = _config_float("KMD_DOCUMENT_CONTEXT_MIN_CONFIDENCE")
    if not 0.0 <= value <= 1.0:
        raise ValueError("KMD_DOCUMENT_CONTEXT_MIN_CONFIDENCE must be between 0 and 1")
    return value


def _cache_root() -> Path:
    return _model_cache_dir("KMD_DOCUMENT_CONTEXT_CACHE_DIR")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)


def _map_schema(chunk_count: int) -> dict[str, Any]:
    # llama.cpp's portable constrained-decoding subset rejects numeric
    # minimum/maximum keywords. Exact chunk-index bounds remain enforced by
    # _validate_range/evidence_chunk checks after decoding.
    index = {"type": "integer"}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["context_segments", "temporal_scopes"],
        "properties": {
            "context_segments": {
                "type": "array",
                "maxItems": max(1, chunk_count * 2),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "kind",
                        "start_chunk",
                        "end_chunk",
                        "evidence_chunk",
                        "evidence_text",
                        "holder_surface",
                        "reason",
                        "confidence",
                    ],
                    "properties": {
                        "kind": {"type": "string", "enum": list(CONTEXT_KINDS)},
                        "start_chunk": index,
                        "end_chunk": index,
                        "evidence_chunk": index,
                        "evidence_text": {"type": "string", "x-kmd-string-profile": "evidence"},
                        "holder_surface": {"type": "string", "x-kmd-string-profile": "label"},
                        "reason": {"type": "string", "x-kmd-string-profile": "reason"},
                        "confidence": {"type": "number"},
                    },
                },
            },
            "temporal_scopes": {
                "type": "array",
                "maxItems": max(1, chunk_count * 2),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "temporal_value",
                        "start_chunk",
                        "end_chunk",
                        "evidence_chunk",
                        "evidence_text",
                        "reason",
                        "confidence",
                    ],
                    "properties": {
                        "temporal_value": {"type": "string", "x-kmd-string-profile": "value"},
                        "start_chunk": index,
                        "end_chunk": index,
                        "evidence_chunk": index,
                        "evidence_text": {"type": "string", "x-kmd-string-profile": "evidence"},
                        "reason": {"type": "string", "x-kmd-string-profile": "reason"},
                        "confidence": {"type": "number"},
                    },
                },
            },
        },
    }


def _boundary_excerpt(sentence: Sentence, per_side: int) -> str:
    text = sentence.text
    if len(text) <= per_side * 2:
        return text
    return f"{text[:per_side]}\n…\n{text[-per_side:]}"


def _boundary_material(sentences: list[Sentence], context_size: int) -> str:
    max_chars = context_char_capacity(
        context_size,
        ratio_names=("KMD_DOCUMENT_CONTEXT_BOUNDARY_RATIO",),
        ratio_default=0.20,
    )
    per_chunk = max(256, max_chars // max(1, len(sentences)))
    per_side = max(128, per_chunk // 2)

    def build(side_chars: int) -> str:
        return "\n\n".join(
            f"[CHUNK {sentence.order}]\n{_boundary_excerpt(sentence, side_chars)}"
            for sentence in sentences
        )

    material = build(per_side)
    if len(material) <= max_chars:
        return material

    # Historical allocation omitted header/separator overhead. Preserve the old
    # prompt bytes whenever they fit, and only shrink symmetric boundary excerpts
    # when the assembled prompt actually overflows.
    low, high = 1, per_side
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = build(middle)
        if len(candidate) <= max_chars:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if best:
        return best
    raise LocalModelUnavailableError(
        f"document context boundary headers exceed capacity: {len(build(1))} > {max_chars}"
    )


def _validate_range(start: int, end: int, chunk_count: int, label: str) -> None:
    if start < 0 or end < start or end >= chunk_count:
        raise LocalModelUnavailableError(
            f"invalid {label} chunk range: start={start} end={end} chunks={chunk_count}"
        )


def _validate_nested_or_disjoint(segments: list[DocumentContextEnvelope]) -> None:
    for index, left in enumerate(segments):
        for right in segments[index + 1 :]:
            if (left.start_chunk, left.end_chunk) == (right.start_chunk, right.end_chunk):
                raise LocalModelUnavailableError(
                    "document context map contains multiple contexts with the same chunk range; "
                    "the model must return one directly governing context for that range"
                )
            overlap = max(left.start_chunk, right.start_chunk) <= min(left.end_chunk, right.end_chunk)
            if not overlap:
                continue
            left_contains = left.start_chunk <= right.start_chunk and left.end_chunk >= right.end_chunk
            right_contains = right.start_chunk <= left.start_chunk and right.end_chunk >= left.end_chunk
            if not (left_contains or right_contains):
                raise LocalModelUnavailableError(
                    "document context map contains crossing, non-nested scope ranges"
                )


def classify_document_context_map(
    document: Document,
    sentences: list[Sentence],
    client: LocalModelClient,
) -> tuple[list[DocumentContextEnvelope], list[DocumentTemporalScope]]:
    if len(sentences) <= 1 or not _enabled():
        return [], []
    material = _boundary_material(sentences, client.context_size())
    cache_context = {
        "policy": DOCUMENT_CONTEXT_POLICY,
        "document_sha256": document.sha256,
        "rel_path": document.rel_path,
        "chunk_count": len(sentences),
        "boundary_sha256": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "model_fingerprint": client.cache_fingerprint(),
    }
    digest = hashlib.sha256(_canonical(cache_context).encode("utf-8")).hexdigest()
    path = _cache_root() / f"{digest}.json"
    raw: dict[str, Any] | None = None
    fresh_result = False
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("cache_context") == cache_context:
            result = payload.get("result")
            if isinstance(result, dict):
                raw = result
    if raw is None:
        fresh_result = True
        prompt = (
            "Build a long-range context map for this document from its ordered chunk boundary excerpts. Do not "
            "classify genre merely because content is unusual. Return only scopes explicitly supported by the "
            "source boundary text. context_segments represent non-asserted discourse/epistemic scopes that govern "
            "one or more complete chunks: dreamed, reported, quoted, believed, hypothetical, fictional, possible, "
            "or uncertain. A dreamed scope does not require the literal words dream or dreamed when boundary text "
            "unmistakably establishes that the governed events occurred only during sleep; unusual, impossible, or "
            "fantastical content by itself is never sufficient. start_chunk and end_chunk are inclusive source chunk "
            "indexes. A closing statement such as "
            "'then I woke up; it had all been a dream' can establish a dreamed segment covering preceding chunks; "
            "a header can establish a following segment; middle transitions can start or end a segment. Return all "
            "distinct nested or disjoint scopes needed for the document. List outer scopes before inner scopes. Never create crossing partial overlaps or two different contexts with the same chunk range; choose the single directly governing context for that exact range. "
            "evidence_chunk identifies the chunk containing evidence_text, which must be an exact contiguous source "
            "substring. temporal_scopes represent a date/time statement that explicitly governs chunks beyond its "
            "local sentence (for example a dated section header). temporal_value must be copied from or be a concise "
            "normalization of that explicit source date/time; evidence_text remains exact source text. Do not infer "
            "dates or contexts from world knowledge. Empty arrays are correct when no long-range scope is explicit.\n\n"
            f"FILE: {document.rel_path}\nCHUNK COUNT: {len(sentences)}\n\n{material}"
        )
        raw = complete_json_with_transport_retry(
            client,
            prompt,
            n_predict=max(512, min(4096, 256 + 192 * len(sentences))),
            json_schema=_map_schema(len(sentences)),
        )

    contexts: list[DocumentContextEnvelope] = []
    for item in raw.get("context_segments", []) if isinstance(raw, dict) else []:
        start = int(item.get("start_chunk", -1))
        end = int(item.get("end_chunk", -1))
        evidence_chunk = int(item.get("evidence_chunk", -1))
        _validate_range(start, end, len(sentences), "context")
        if evidence_chunk < 0 or evidence_chunk >= len(sentences):
            raise LocalModelUnavailableError("document context evidence_chunk is out of range")
        evidence = str(item.get("evidence_text") or "")
        if not evidence or evidence not in sentences[evidence_chunk].text:
            raise LocalModelUnavailableError("document context evidence is not an exact source substring")
        kind = str(item.get("kind") or "")
        if kind not in CONTEXT_KINDS:
            raise LocalModelUnavailableError(f"unsupported document context kind: {kind!r}")
        contexts.append(
            DocumentContextEnvelope(
                True,
                kind,
                "chunk_range",
                evidence,
                str(item.get("holder_surface") or ""),
                str(item.get("reason") or ""),
                max(0.0, min(1.0, float(item.get("confidence") or 0.0))),
                start,
                end,
                evidence_chunk,
            )
        )
    _validate_nested_or_disjoint(contexts)

    temporals: list[DocumentTemporalScope] = []
    for item in raw.get("temporal_scopes", []) if isinstance(raw, dict) else []:
        start = int(item.get("start_chunk", -1))
        end = int(item.get("end_chunk", -1))
        evidence_chunk = int(item.get("evidence_chunk", -1))
        _validate_range(start, end, len(sentences), "temporal")
        if evidence_chunk < 0 or evidence_chunk >= len(sentences):
            raise LocalModelUnavailableError("document temporal evidence_chunk is out of range")
        evidence = str(item.get("evidence_text") or "")
        if not evidence or evidence not in sentences[evidence_chunk].text:
            raise LocalModelUnavailableError("document temporal evidence is not an exact source substring")
        value = str(item.get("temporal_value") or "").strip()
        if not value:
            raise LocalModelUnavailableError("document temporal scope is missing temporal_value")
        temporals.append(
            DocumentTemporalScope(
                value,
                evidence,
                str(item.get("reason") or ""),
                max(0.0, min(1.0, float(item.get("confidence") or 0.0))),
                start,
                end,
                evidence_chunk,
            )
        )
    if fresh_result:
        _atomic_write(path, {"cache_context": cache_context, "result": raw})
    return contexts, temporals


def classify_document_context(
    document: Document,
    sentences: list[Sentence],
    client: LocalModelClient,
) -> DocumentContextEnvelope:
    """Compatibility wrapper returning the first long-range epistemic scope."""
    contexts, _ = classify_document_context_map(document, sentences, client)
    if contexts:
        return contexts[0]
    return DocumentContextEnvelope(False, "asserted", "none", "", "", "no long-range context", 1.0)


def _span_rows(store: DSPGStore, document: Document) -> list[Any]:
    return store.execute(
        """
        SELECT ss.span_id, ss.char_start, ss.char_end, c.chunk_order
        FROM source_spans ss
        JOIN chunks c ON c.chunk_id=ss.chunk_id
        WHERE ss.document_id=?
        ORDER BY c.chunk_order, ss.char_start, ss.char_end, ss.span_id
        """,
        (document.document_id,),
    ).fetchall()


def _selected_span_rows(
    store: DSPGStore,
    document: Document,
    envelope: DocumentContextEnvelope,
) -> list[Any]:
    rows = _span_rows(store, document)
    if envelope.direction == "chunk_range" and envelope.end_chunk >= envelope.start_chunk:
        return [
            row
            for row in rows
            if envelope.start_chunk <= int(row["chunk_order"]) <= envelope.end_chunk
        ]
    if envelope.direction == "whole_document":
        return rows
    if not envelope.evidence_text:
        return []
    if envelope.direction == "preceding_document":
        boundary = document.text.rfind(envelope.evidence_text)
        return [row for row in rows if int(row["char_start"]) < boundary]
    if envelope.direction == "following_document":
        boundary = document.text.find(envelope.evidence_text) + len(envelope.evidence_text)
        return [row for row in rows if int(row["char_end"]) > boundary]
    return []


def _context_parent_map(segments: list[DocumentContextEnvelope], context_ids: dict[int, str]) -> dict[int, str | None]:
    parents: dict[int, str | None] = {}
    for index, segment in enumerate(segments):
        containers = [
            (other.end_chunk - other.start_chunk, other_index)
            for other_index, other in enumerate(segments)
            if other_index != index
            and other.start_chunk <= segment.start_chunk
            and other.end_chunk >= segment.end_chunk
            and (other.start_chunk, other.end_chunk) != (segment.start_chunk, segment.end_chunk)
        ]
        parents[index] = context_ids[min(containers)[1]] if containers else None
    return parents


def _innermost_segment_index(segments: list[DocumentContextEnvelope], chunk_order: int) -> int | None:
    matches = [
        (segment.end_chunk - segment.start_chunk, index)
        for index, segment in enumerate(segments)
        if segment.start_chunk <= chunk_order <= segment.end_chunk
    ]
    return min(matches)[1] if matches else None


def _effective_span_context(store: DSPGStore, run_id: str, span_id: str) -> str | None:
    row = store.execute(
        """
        SELECT ca.context_id
        FROM context_assignments ca
        WHERE ca.run_id=? AND ca.applies_to_type='source_span' AND ca.applies_to_id=?
        ORDER BY ca.confidence DESC, ca.assignment_id
        LIMIT 1
        """,
        (run_id, span_id),
    ).fetchone()
    if row is not None:
        return str(row["context_id"])
    row = store.execute(
        "SELECT context_id FROM contexts WHERE run_id=? AND kind='asserted' ORDER BY context_id LIMIT 1",
        (run_id,),
    ).fetchone()
    return str(row["context_id"]) if row is not None else None


def apply_document_context_map(
    store: DSPGStore,
    run_id: str,
    document: Document,
    contexts: list[DocumentContextEnvelope],
    temporals: list[DocumentTemporalScope],
) -> dict[str, int]:
    minimum = _minimum_confidence()
    contexts = [item for item in contexts if item.applies and item.confidence >= minimum]
    temporals = [item for item in temporals if item.confidence >= minimum]
    if not contexts and not temporals:
        return {
            "contexts_applied": 0,
            "temporal_scopes_applied": 0,
            "spans_rebound": 0,
            "temporal_edges_added": 0,
        }
    rows = _span_rows(store, document)
    by_chunk: dict[int, list[Any]] = {}
    for row in rows:
        by_chunk.setdefault(int(row["chunk_order"]), []).append(row)

    context_ids: dict[int, str] = {}
    for index, segment in enumerate(contexts):
        context_ids[index] = stable_id(
            "ctx",
            run_id,
            "document_context_map",
            document.document_id,
            DOCUMENT_CONTEXT_POLICY,
            index,
            segment.kind,
            segment.start_chunk,
            segment.end_chunk,
            segment.evidence_text,
        )
    parents = _context_parent_map(contexts, context_ids)
    for index, segment in enumerate(contexts):
        store.execute(
            """
            INSERT OR REPLACE INTO contexts(
              context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                context_ids[index],
                run_id,
                f"drs:{segment.kind}",
                parents[index],
                segment.holder_surface or None,
                segment.evidence_text,
                segment.confidence,
            ),
        )

    rebound_span_ids: set[str] = set()
    for chunk_order, chunk_rows in by_chunk.items():
        segment_index = _innermost_segment_index(contexts, chunk_order)
        if segment_index is None:
            continue
        context_id = context_ids[segment_index]
        span_ids = [str(row["span_id"]) for row in chunk_rows]
        rebound_span_ids.update(span_ids)
        placeholders = ",".join("?" for _ in span_ids)
        root_rows = store.execute(
            f"""
            SELECT DISTINCT db.context_id
            FROM drs_boxes db
            WHERE db.run_id=? AND db.source_span_id IN ({placeholders})
              AND db.parent_drs_box_id IS NULL AND db.kind='asserted'
            """,
            (run_id, *span_ids),
        ).fetchall()
        for row in root_rows:
            store.execute(
                """
                UPDATE contexts SET parent_context_id=?
                WHERE context_id=? AND kind='drs:asserted' AND parent_context_id IS NULL
                """,
                (context_id, str(row["context_id"])),
            )
        for table, span_column in (
            ("frames", "span_id"),
            ("relations", "source_span_id"),
            ("temporal_edges", "source_span_id"),
            ("drs_identity_hypotheses", "source_span_id"),
        ):
            store.execute(
                f"""
                UPDATE {table} SET context_id=?
                WHERE {span_column} IN ({placeholders})
                  AND context_id IN (SELECT context_id FROM contexts WHERE kind='asserted')
                """,
                (context_id, *span_ids),
            )
        for span_id in span_ids:
            updated = store.execute(
                """
                UPDATE context_assignments SET context_id=?, confidence=?
                WHERE run_id=? AND applies_to_type='source_span' AND applies_to_id=?
                  AND context_id IN (SELECT context_id FROM contexts WHERE kind='asserted')
                """,
                (context_id, contexts[segment_index].confidence, run_id, span_id),
            ).rowcount
            if not updated:
                assignment_id = stable_id("ctxassign", run_id, "document_context_map", span_id, context_id)
                store.execute(
                    """
                    INSERT OR IGNORE INTO context_assignments(
                      assignment_id, run_id, context_id, applies_to_type, applies_to_id, source_span_id, confidence
                    ) VALUES (?, ?, ?, 'source_span', ?, ?, ?)
                    """,
                    (
                        assignment_id,
                        run_id,
                        context_id,
                        span_id,
                        span_id,
                        contexts[segment_index].confidence,
                    ),
                )

    temporal_count = 0
    for index, scope in enumerate(temporals):
        selected_rows = [
            row
            for chunk_order, chunk_rows in by_chunk.items()
            if scope.start_chunk <= chunk_order <= scope.end_chunk
            for row in chunk_rows
        ]
        for row in selected_rows:
            span_id = str(row["span_id"])
            context_id = _effective_span_context(store, run_id, span_id)
            store.execute(
                """
                INSERT OR REPLACE INTO temporal_edges(
                  edge_id, run_id, source_span_id, referent_id, context_id,
                  relation, temporal_value, state_value, confidence
                ) VALUES (?, ?, ?, NULL, ?, 'document_temporal_scope', ?, NULL, ?)
                """,
                (
                    stable_id(
                        "tmp",
                        run_id,
                        "document_temporal_scope",
                        document.document_id,
                        index,
                        span_id,
                        scope.temporal_value,
                    ),
                    run_id,
                    span_id,
                    context_id,
                    scope.temporal_value,
                    scope.confidence,
                ),
            )
            temporal_count += 1
    store.commit()
    return {
        "contexts_applied": len(contexts),
        "temporal_scopes_applied": len(temporals),
        "spans_rebound": len(rebound_span_ids),
        "temporal_edges_added": temporal_count,
    }


def apply_document_context_envelope(
    store: DSPGStore,
    run_id: str,
    document: Document,
    envelope: DocumentContextEnvelope,
) -> int:
    """Compatibility wrapper for a single envelope."""
    result = apply_document_context_map(store, run_id, document, [envelope], [])
    return int(result["spans_rebound"])


def apply_document_context_envelopes(
    store: DSPGStore,
    run_id: str,
    documents: list[Document],
    sentences: list[Sentence],
    client: LocalModelClient | None,
) -> dict[str, int]:
    stats = {
        "documents_considered": 0,
        "context_segments_applied": 0,
        "temporal_scopes_applied": 0,
        "spans_rebound": 0,
        "temporal_edges_added": 0,
    }
    if client is None or not _enabled():
        return stats
    by_document: dict[str, list[Sentence]] = {}
    for sentence in sentences:
        by_document.setdefault(sentence.document_id, []).append(sentence)
    for document in documents:
        chunks = sorted(by_document.get(document.document_id, []), key=lambda item: item.order)
        if len(chunks) <= 1:
            continue
        stats["documents_considered"] += 1
        contexts, temporals = classify_document_context_map(document, chunks, client)
        result = apply_document_context_map(store, run_id, document, contexts, temporals)
        stats["context_segments_applied"] += result["contexts_applied"]
        stats["temporal_scopes_applied"] += result["temporal_scopes_applied"]
        stats["spans_rebound"] += result["spans_rebound"]
        stats["temporal_edges_added"] += result["temporal_edges_added"]
    return stats
