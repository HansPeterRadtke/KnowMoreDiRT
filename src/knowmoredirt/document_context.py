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
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from context_capacity import context_char_capacity
from kmd_runtime_config import boolean as _config_boolean, floating as _config_float, text as _config_text, model_cache_dir as _model_cache_dir

from .model import LocalModelClient, LocalModelUnavailableError, complete_json_with_transport_retry
from .models import Document, Sentence
from .store import DSPGStore, stable_id

DOCUMENT_CONTEXT_POLICY = "document-context-map-v6-sanitized-model-items"
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
DISCOURSE_RELATION_TYPES = (
    "sequence", "elaboration", "explanation", "contrast", "condition",
    "consequence", "result", "attribution", "reporting", "correction",
    "revision", "scope_close", "temporal_before", "temporal_after",
    "temporal_overlap", "same_topic", "section_membership",
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentDiscourseRelation:
    relation_type: str
    from_chunk: int
    to_chunk: int
    evidence_chunk: int
    evidence_text: str
    reason: str
    confidence: float


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
    if not _config_boolean("KMD_DOCUMENT_CONTEXT_ENVELOPES"):
        if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("KMD_TEST_ALLOW_SEMANTIC_INVARIANT_BYPASS", "").strip().lower() in {"1", "true", "yes", "on"}:
            return False
        raise LocalModelUnavailableError(
            "KnowMoreDiRT production runtime requires document context mapping; "
            "KMD_DOCUMENT_CONTEXT_ENVELOPES=0 is not supported."
        )
    return True


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
        "required": ["context_segments", "temporal_scopes", "discourse_relations"],
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
            "discourse_relations": {
                "type": "array",
                "maxItems": max(1, chunk_count * 4),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["relation_type", "from_chunk", "to_chunk", "evidence_chunk", "evidence_text", "reason", "confidence"],
                    "properties": {
                        "relation_type": {"type": "string", "enum": list(DISCOURSE_RELATION_TYPES)},
                        "from_chunk": index,
                        "to_chunk": index,
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


def _coverage_clue_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["clues"],
        "properties": {
            "clues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["kind", "evidence_text"],
                    "properties": {
                        "kind": {"type": "string", "enum": [*CONTEXT_KINDS, "temporal"]},
                        "evidence_text": {"type": "string", "x-kmd-string-profile": "evidence"},
                    },
                },
            }
        },
    }


def _uncovered_windows(sentence: Sentence, boundary_material: str, context_size: int) -> list[str]:
    text = sentence.text
    if not text or text in boundary_material:
        return []
    max_chars = max(
        1024,
        context_char_capacity(
            context_size,
            ratio_names=("KMD_DOCUMENT_CONTEXT_COVERAGE_RATIO",),
            ratio_default=0.25,
        ),
    )
    overlap = min(512, max_chars // 8)
    step = max(1, max_chars - overlap)
    return [text[start : start + max_chars] for start in range(0, len(text), step)]


def _full_coverage_clues(
    document: Document,
    sentences: list[Sentence],
    client: LocalModelClient,
    boundary_material: str,
) -> list[dict[str, Any]]:
    clues: list[dict[str, Any]] = []
    for sentence in sentences:
        windows = _uncovered_windows(sentence, boundary_material, client.context_size())
        for window_index, window in enumerate(windows):
            prompt = (
                "Inspect this exact source window only for text that can open, close, or retroactively establish "
                "a discourse/epistemic scope (dreamed, reported, quoted, believed, hypothetical, fictional, "
                "possible, uncertain) or an explicit date/time scope. Do not infer scope from unusual content. "
                "Return only exact contiguous evidence substrings from SOURCE_WINDOW. Empty clues is correct. "
                "This is a retrieval clue pass, not a final range decision.\n\n"
                f"FILE: {document.rel_path}\nCHUNK: {sentence.order}\nWINDOW: {window_index}\n"
                f"SOURCE_WINDOW:\n{window}"
            )
            raw = complete_json_with_transport_retry(
                client,
                prompt,
                json_schema=_coverage_clue_schema(),
            )
            for item in raw.get("clues", []) if isinstance(raw, dict) else []:
                kind = str(item.get("kind") or "")
                evidence = str(item.get("evidence_text") or "")
                if kind not in {*CONTEXT_KINDS, "temporal"}:
                    continue
                if not evidence or evidence not in window or evidence not in sentence.text:
                    LOGGER.warning(
                        "document_context_discard_invalid kind=coverage_clue reason=not_exact_source_substring"
                    )
                    continue
                clue = {"chunk": sentence.order, "kind": kind, "evidence_text": evidence}
                if clue not in clues:
                    clues.append(clue)
    return clues


def _coverage_clue_material(clues: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[FULL-COVERAGE CLUE chunk={item['chunk']} kind={item['kind']}]\n{item['evidence_text']}"
        for item in clues
    )


def _classify_document_context_map_full(
    document: Document,
    sentences: list[Sentence],
    client: LocalModelClient,
) -> tuple[list[DocumentContextEnvelope], list[DocumentTemporalScope], list[DocumentDiscourseRelation]]:
    if len(sentences) <= 1 or not _enabled():
        return [], [], []
    material = _boundary_material(sentences, client.context_size())
    coverage_clues = _full_coverage_clues(document, sentences, client, material)
    coverage_material = _coverage_clue_material(coverage_clues)
    cache_context = {
        "policy": DOCUMENT_CONTEXT_POLICY,
        "document_sha256": document.sha256,
        "rel_path": document.rel_path,
        "chunk_count": len(sentences),
        "boundary_sha256": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "coverage_clues_sha256": hashlib.sha256(coverage_material.encode("utf-8")).hexdigest(),
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
            "dates or contexts from world knowledge. discourse_relations represent explicit rhetorical/discourse links between source chunks: sequence, elaboration, explanation, contrast, condition, consequence/result, attribution/reporting, correction/revision, explicit scope-close, temporal before/after/overlap, same-topic, or section membership. Emit a discourse relation only when exact source text explicitly licenses it; from_chunk and to_chunk identify the related source chunks and evidence_text must be an exact contiguous substring of evidence_chunk. Do not infer rhetorical relations merely from topical similarity. Empty arrays are correct when no long-range scope or discourse relation is explicit.\n\n"
            f"FILE: {document.rel_path}\nCHUNK COUNT: {len(sentences)}\n\n{material}"
            + (
                "\n\nFULL-COVERAGE CLUES FROM SOURCE TEXT OMITTED BY BOUNDARY EXCERPTS:\n"
                + coverage_material
                if coverage_material
                else ""
            )
        )
        raw = complete_json_with_transport_retry(
            client,
            prompt,
            json_schema=_map_schema(len(sentences)),
        )

    contexts: list[DocumentContextEnvelope] = []
    valid_context_items: list[dict[str, Any]] = []
    for item in raw.get("context_segments", []) if isinstance(raw, dict) else []:
        if not isinstance(item, dict):
            LOGGER.warning("document_context_discard_invalid kind=context reason=non_object")
            continue
        try:
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
            candidate = DocumentContextEnvelope(
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
            _validate_nested_or_disjoint([*contexts, candidate])
        except (LocalModelUnavailableError, TypeError, ValueError) as exc:
            LOGGER.warning("document_context_discard_invalid kind=context reason=%s", exc)
            continue
        contexts.append(candidate)
        valid_context_items.append(item)

    temporals: list[DocumentTemporalScope] = []
    valid_temporal_items: list[dict[str, Any]] = []
    for item in raw.get("temporal_scopes", []) if isinstance(raw, dict) else []:
        if not isinstance(item, dict):
            LOGGER.warning("document_context_discard_invalid kind=temporal reason=non_object")
            continue
        try:
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
            candidate = DocumentTemporalScope(
                value,
                evidence,
                str(item.get("reason") or ""),
                max(0.0, min(1.0, float(item.get("confidence") or 0.0))),
                start,
                end,
                evidence_chunk,
            )
        except (LocalModelUnavailableError, TypeError, ValueError) as exc:
            LOGGER.warning("document_context_discard_invalid kind=temporal reason=%s", exc)
            continue
        temporals.append(candidate)
        valid_temporal_items.append(item)

    discourse_relations: list[DocumentDiscourseRelation] = []
    valid_discourse_items: list[dict[str, Any]] = []
    for item in raw.get("discourse_relations", []) if isinstance(raw, dict) else []:
        if not isinstance(item, dict):
            LOGGER.warning("document_context_discard_invalid kind=discourse reason=non_object")
            continue
        try:
            relation_type = str(item.get("relation_type") or "")
            if relation_type not in DISCOURSE_RELATION_TYPES:
                raise LocalModelUnavailableError(f"unsupported document discourse relation: {relation_type!r}")
            from_chunk = int(item.get("from_chunk", -1))
            to_chunk = int(item.get("to_chunk", -1))
            evidence_chunk = int(item.get("evidence_chunk", -1))
            for value, label in ((from_chunk, "from_chunk"), (to_chunk, "to_chunk"), (evidence_chunk, "evidence_chunk")):
                if value < 0 or value >= len(sentences):
                    raise LocalModelUnavailableError(f"document discourse {label} is out of range")
            evidence = str(item.get("evidence_text") or "")
            if not evidence or evidence not in sentences[evidence_chunk].text:
                raise LocalModelUnavailableError("document discourse evidence is not an exact source substring")
            candidate = DocumentDiscourseRelation(
                relation_type,
                from_chunk,
                to_chunk,
                evidence_chunk,
                evidence,
                str(item.get("reason") or ""),
                max(0.0, min(1.0, float(item.get("confidence") or 0.0))),
            )
        except (LocalModelUnavailableError, TypeError, ValueError) as exc:
            LOGGER.warning("document_context_discard_invalid kind=discourse reason=%s", exc)
            continue
        discourse_relations.append(candidate)
        valid_discourse_items.append(item)

    if fresh_result:
        _atomic_write(path, {
            "cache_context": cache_context,
            "result": {
                "context_segments": valid_context_items,
                "temporal_scopes": valid_temporal_items,
                "discourse_relations": valid_discourse_items,
            },
        })
    return contexts, temporals, discourse_relations


def classify_document_context_map(
    document: Document,
    sentences: list[Sentence],
    client: LocalModelClient,
) -> tuple[list[DocumentContextEnvelope], list[DocumentTemporalScope]]:
    contexts, temporals, _relations = _classify_document_context_map_full(document, sentences, client)
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


def _evidence_span_for_chunk_text(
    store: DSPGStore, document: Document, chunk_order: int, evidence_text: str
) -> str | None:
    rows = store.execute(
        """
        SELECT ss.span_id, ss.surface
        FROM source_spans ss
        JOIN chunks c ON c.chunk_id=ss.chunk_id
        WHERE ss.document_id=? AND c.chunk_order=?
        ORDER BY ss.char_start, ss.char_end, ss.span_id
        """,
        (document.document_id, chunk_order),
    ).fetchall()
    for row in rows:
        surface = str(row["surface"] or "")
        if evidence_text and evidence_text in surface:
            return str(row["span_id"])
    return str(rows[0]["span_id"]) if rows else None


def _materialize_document_authority(store: DSPGStore, run_id: str, document: Document) -> int:
    # Authority is a source claim.  A self-declaration is stored separately from
    # verified authority, which requires explicit source metadata.
    declaration = ""
    for line in document.text.splitlines()[:64]:
        candidate = line.strip()
        low = candidate.lower()
        if candidate and (
            low.startswith("official ")
            or low.startswith("official:")
            or low.startswith("authoritative ")
            or low.startswith("authoritative:")
            or re.match(r"^(?:title|document|section)\s*:\s*(?:official|authoritative)\b", low)
        ):
            declaration = candidate
            break
    metadata = document.metadata if isinstance(document.metadata, dict) else {}
    verified_value = ""
    if metadata.get("verified_authority"):
        verified_value = str(metadata.get("verified_authority"))
    elif metadata.get("authority_verified") is True:
        verified_value = declaration or "verified_by_source_metadata"
    if not declaration and not verified_value:
        return 0
    span_row = store.execute(
        """
        SELECT ss.span_id
        FROM source_spans ss
        WHERE ss.document_id=? AND instr(ss.surface, ?) > 0
        ORDER BY ss.char_start LIMIT 1
        """,
        (document.document_id, declaration),
    ).fetchone()
    span_id = str(span_row["span_id"]) if span_row is not None else None
    context_rows = store.execute(
        """
        SELECT DISTINCT context_id FROM (
          SELECT f.context_id AS context_id FROM frames f JOIN source_spans ss ON ss.span_id=f.span_id WHERE ss.document_id=?
          UNION ALL
          SELECT dc.context_id AS context_id FROM drs_conditions dc JOIN source_spans ss ON ss.span_id=dc.source_span_id WHERE ss.document_id=?
          UNION ALL
          SELECT ca.context_id AS context_id FROM context_assignments ca JOIN source_spans ss ON ss.span_id=ca.source_span_id WHERE ss.document_id=?
        ) WHERE context_id IS NOT NULL
        """,
        (document.document_id, document.document_id, document.document_id),
    ).fetchall()
    updated = 0
    for row in context_rows:
        updated += max(0, store.execute(
            """
            UPDATE contexts
            SET declared_authority=?, verified_authority=?, authority_source_span_id=?
            WHERE context_id=?
            """,
            (declaration, verified_value, span_id, str(row["context_id"])),
        ).rowcount or 0)
    return updated


def _insert_discourse_edge(
    store: DSPGStore,
    *,
    edge_id: str,
    run_id: str,
    relation_type: str,
    document_id: str,
    source_span_id: str | None,
    from_context_id: str | None = None,
    to_context_id: str | None = None,
    from_span_id: str | None = None,
    to_span_id: str | None = None,
    evidence_surface: str = "",
    confidence: float = 1.0,
    source: str = "document_context_map",
    metadata: dict[str, Any] | None = None,
) -> None:
    store.execute(
        """
        INSERT OR REPLACE INTO discourse_edges(
          edge_id, run_id, relation_type, document_id, source_span_id, from_context_id, to_context_id,
          from_span_id, to_span_id, evidence_surface, confidence, source, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            edge_id, run_id, relation_type, document_id, source_span_id, from_context_id, to_context_id,
            from_span_id, to_span_id, evidence_surface, confidence, source,
            json.dumps(metadata or {}, sort_keys=True),
        ),
    )


def apply_document_context_map(
    store: DSPGStore,
    run_id: str,
    document: Document,
    contexts: list[DocumentContextEnvelope],
    temporals: list[DocumentTemporalScope],
    discourse_relations: list[DocumentDiscourseRelation] | None = None,
) -> dict[str, int]:
    minimum = _minimum_confidence()
    contexts = [item for item in contexts if item.applies and item.confidence >= minimum]
    temporals = [item for item in temporals if item.confidence >= minimum]
    discourse_relations = [item for item in (discourse_relations or []) if item.confidence >= minimum]
    authority_contexts_updated = _materialize_document_authority(store, run_id, document)
    if not contexts and not temporals and not discourse_relations:
        store.commit()
        return {
            "contexts_applied": 0,
            "temporal_scopes_applied": 0,
            "spans_rebound": 0,
            "temporal_edges_added": 0,
            "discourse_edges_added": 0,
            "authority_contexts_updated": authority_contexts_updated,
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

    discourse_edge_count = 0
    for index, segment in enumerate(contexts):
        evidence_span_id = _evidence_span_for_chunk_text(
            store, document, segment.evidence_chunk, segment.evidence_text
        )
        if evidence_span_id:
            carrier_id = stable_id("ctxcarrier", run_id, "document_context_map", document.document_id, index, evidence_span_id)
            store.execute(
                """
                INSERT OR REPLACE INTO context_carriers(
                  carrier_id, run_id, context_id, document_id, source_span_id, carrier_kind,
                  carrier_surface, temporal_value, temporal_value_type, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
                """,
                (
                    carrier_id, run_id, context_ids[index], document.document_id, evidence_span_id,
                    "retroactive_scope" if segment.evidence_chunk > segment.start_chunk else "scope_open",
                    segment.evidence_text, segment.confidence,
                ),
            )
            relation_type = "retroactive_scope" if segment.evidence_chunk > segment.start_chunk else "scope_open"
            _insert_discourse_edge(
                store,
                edge_id=stable_id("disc", run_id, relation_type, document.document_id, index, evidence_span_id),
                run_id=run_id, relation_type=relation_type, document_id=document.document_id,
                source_span_id=evidence_span_id, to_context_id=context_ids[index], from_span_id=evidence_span_id,
                evidence_surface=segment.evidence_text, confidence=segment.confidence,
                metadata={
                    "start_chunk": segment.start_chunk, "end_chunk": segment.end_chunk,
                    "evidence_chunk": segment.evidence_chunk, "reason": segment.reason,
                },
            )
            discourse_edge_count += 1

    # Deterministic document sequence edges preserve adjacency without asking a
    # model to invent rhetorical labels. They are useful for bounded one-hop
    # context expansion and provide the baseline discourse graph.
    ordered_chunks = sorted(by_chunk)
    for left_order, right_order in zip(ordered_chunks, ordered_chunks[1:]):
        left_rows = by_chunk.get(left_order) or []
        right_rows = by_chunk.get(right_order) or []
        if not left_rows or not right_rows:
            continue
        left_span = str(left_rows[-1]["span_id"])
        right_span = str(right_rows[0]["span_id"])
        _insert_discourse_edge(
            store, edge_id=stable_id("disc", run_id, "continuation", document.document_id, left_span, right_span),
            run_id=run_id, relation_type="continuation", document_id=document.document_id,
            source_span_id=left_span, from_span_id=left_span, to_span_id=right_span,
            evidence_surface="", confidence=1.0, source="deterministic_document_sequence",
            metadata={"from_chunk": left_order, "to_chunk": right_order},
        )
        discourse_edge_count += 1

    for index, relation in enumerate(discourse_relations):
        from_rows = by_chunk.get(relation.from_chunk) or []
        to_rows = by_chunk.get(relation.to_chunk) or []
        evidence_span_id = _evidence_span_for_chunk_text(
            store, document, relation.evidence_chunk, relation.evidence_text
        )
        if not from_rows or not to_rows or not evidence_span_id:
            continue
        from_span_id = str(from_rows[0]["span_id"])
        to_span_id = str(to_rows[0]["span_id"])
        _insert_discourse_edge(
            store,
            edge_id=stable_id(
                "disc", run_id, relation.relation_type, document.document_id,
                index, from_span_id, to_span_id, evidence_span_id,
            ),
            run_id=run_id,
            relation_type=relation.relation_type,
            document_id=document.document_id,
            source_span_id=evidence_span_id,
            from_span_id=from_span_id,
            to_span_id=to_span_id,
            evidence_surface=relation.evidence_text,
            confidence=relation.confidence,
            source="document_context_map",
            metadata={"reason": relation.reason, "from_chunk": relation.from_chunk, "to_chunk": relation.to_chunk},
        )
        discourse_edge_count += 1

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
        "discourse_edges_added": discourse_edge_count,
        "authority_contexts_updated": authority_contexts_updated,
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
        "discourse_edges_added": 0,
        "authority_contexts_updated": 0,
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
        contexts, temporals, discourse_relations = _classify_document_context_map_full(document, chunks, client)
        result = apply_document_context_map(store, run_id, document, contexts, temporals, discourse_relations)
        stats["context_segments_applied"] += result["contexts_applied"]
        stats["temporal_scopes_applied"] += result["temporal_scopes_applied"]
        stats["spans_rebound"] += result["spans_rebound"]
        stats["temporal_edges_added"] += result["temporal_edges_added"]
        stats["discourse_edges_added"] += result["discourse_edges_added"]
        stats["authority_contexts_updated"] += result["authority_contexts_updated"]
    return stats
