"""DRS-centered bounded retrieval and variable binding.

This module is deliberately relation-agnostic.  It does not dispatch on source
relation names or question-family labels.  It selects a bounded grounded
subgraph, treats stored frames/relations as DRS conditions, and binds a query
variable by unification-like matching over anchors, predicate text, context,
temporal scope, and broad structural answer type.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
from typing import Any

from .answer_types import ExpectedAnswer, canonicalize_answer, is_value_compatible
from .extractors import identifiers, urls
from .models import Answer, Document, Evidence, Sentence
from .query import QueryFrame, expand_terms, frame_from_mapping, normalize_temporal_scope, plan_question, term_variants
from .store import identity_relation_allows_expansion, stable_id
from .text import clean_extracted_value, content_tokens, normalize, text_quality_metrics

DATE_TIME_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2})?|\d{1,2}:\d{2})\b")
PATH_RE = re.compile(r"\b[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+\b|\b[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12}\b")
INACCESSIBLE_CONTEXT_PREFIXES = ("modality:",)
IDENTITY_GRAPH_MAX_DEPTH = 3
IDENTITY_RERANK_MAX_ROUNDS = 6
ANSWER_SLOT_SKIP_TERMS = {
    "answer",
    "content",
    "entity",
    "how",
    "item",
    "text",
    "thing",
    "value",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}
COUNT_AGGREGATION_SKIP_TERMS = {
    "count",
    "counts",
    "entry",
    "entries",
    "number",
    "numbers",
    "record",
    "records",
    "row",
    "rows",
}


@lru_cache(maxsize=8192)
def _normalized_token_set(value: str) -> frozenset[str]:
    return frozenset(token for token in re.split(r"[^a-z0-9]+", normalize(value)) if token)


@lru_cache(maxsize=16384)
def _material_parts(material: str) -> tuple[str, ...]:
    return tuple(part for part in re.split(r"[^a-z0-9]+", material) if part)


@lru_cache(maxsize=2048)
def _normalized_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(term_norm for term in terms if (term_norm := normalize(term))))


@lru_cache(maxsize=2048)
def _normalized_term_set(terms: tuple[str, ...]) -> frozenset[str]:
    return frozenset(_normalized_terms(terms))


@lru_cache(maxsize=2048)
def _normalized_term_token_sets(terms: tuple[str, ...]) -> tuple[frozenset[str], ...]:
    return tuple(token_set for term in _normalized_terms(terms) if (token_set := _normalized_token_set(term)))


def _compound_term_variants(term: str) -> list[str]:
    norm = normalize(term)
    if not norm:
        return []
    values = [norm]
    parts = [part for part in re.split(r"[_-]+", norm) if part]
    if len(parts) > 1 and all(part.isalpha() for part in parts):
        for part in parts:
            values.extend(expand_terms([part]))
    return list(dict.fromkeys(value for value in values if value))


def _frame(plan: dict[str, Any] | QueryFrame | None, question: str) -> QueryFrame:
    if isinstance(plan, QueryFrame):
        return plan
    return frame_from_mapping(question, plan if isinstance(plan, dict) else None)


def _query_terms(text: str) -> list[str]:
    values: list[str] = []
    for token in content_tokens(text):
        if len(token) <= 1:
            continue
        for part in [token, *re.split(r"[-_]", token)]:
            if len(part) > 1 and part not in values:
                values.append(part)
    return expand_terms(values)


def _target_terms(frame: QueryFrame, question: str) -> list[str]:
    values: list[str] = []
    for anchor in frame.target_anchors:
        norm = normalize(anchor)
        if not norm:
            continue
        values.append(norm)
        if " " in norm:
            values.append(norm.replace(" ", "_"))
            values.append(norm.replace(" ", "-"))
    return list(dict.fromkeys(values))


def _relation_terms(frame: QueryFrame, question: str) -> list[str]:
    target = set(_target_terms(frame, question))
    raw_terms = list(frame.relation_terms) + _query_terms(frame.requested_relation) + list(frame.constraints)
    terms = [variant for term in raw_terms for variant in _compound_term_variants(term)]
    filtered = [
        term
        for term in terms
        if term and term not in target and normalize_temporal_scope(term) not in {"latest", "earliest"}
    ]
    return list(dict.fromkeys(term for term in expand_terms(filtered) if term and term not in target))


def _answer_slot_terms(frame: QueryFrame) -> list[str]:
    terms: list[str] = []
    for variable in frame.answer_variables:
        for term in _compound_term_variants(variable):
            if term not in ANSWER_SLOT_SKIP_TERMS:
                terms.append(term)
        for token in content_tokens(variable):
            if token not in ANSWER_SLOT_SKIP_TERMS:
                terms.append(token)
    return list(dict.fromkeys(term for term in expand_terms(terms) if term))


def _has_term(material: str, term: str) -> bool:
    if not term:
        return False
    if term in material:
        return True
    if re.search(r"[\s_./:-]", term):
        return False
    parts = _material_parts(material)
    if term in parts:
        return True
    if len(term) >= 3 and any(part.startswith(term) for part in parts if len(part) >= 3):
        return True
    return False


def _contains_any(material: str, terms: list[str]) -> bool:
    return any(_has_term(material, term) for term in terms)


def _document_material(document: Document, sentences: list[Sentence]) -> str:
    metadata = document.metadata or {}
    pieces = [
        str(metadata.get("file_name", "")),
        str(metadata.get("stem", "")),
        str(metadata.get("suffix", "")),
        str(metadata.get("parent_rel_path", "")),
        " ".join(sentence.text for sentence in sentences[:80]),
    ]
    return normalize(" ".join(pieces))


def _source_is_low_priority(rel_path: str, text: str) -> bool:
    quality = text_quality_metrics(text)
    quality_label = str(quality.get("semantic_quality") or "")
    token_count = int(quality.get("token_count") or 0)
    return bool(quality.get("low_semantic_noise")) or quality_label in {
        "random_character_noise",
        "base64_or_hex_blob",
    } or (
        token_count >= 20
        and quality_label in {"ocr_corruption", "multilingual_word_salad", "word_salad", "plausible_babble"}
    )


def _rank_scope(
    documents: list[Document],
    sentences_by_document: dict[str, dict[int, Sentence]],
    question: str,
    frame: QueryFrame,
    doc_limit: int,
    chunk_limit: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    target_terms = _target_terms(frame, question)
    relation_terms = list(dict.fromkeys([*_relation_terms(frame, question), *_answer_slot_terms(frame)]))
    all_terms = _query_terms(question)
    doc_scores: list[tuple[float, str, str]] = []
    relation_doc_scores: list[tuple[float, str, str]] = []
    document_material_by_id: dict[str, str] = {}
    document_low_priority_by_id: dict[str, bool] = {}
    for document in documents:
        sentences = list(sentences_by_document.get(document.rel_path, {}).values())
        material = _document_material(document, sentences)
        document_material_by_id[document.document_id] = material
        target_hits = sum(1 for term in target_terms if _has_term(material, term))
        relation_hits = sum(1 for term in relation_terms if _has_term(material, term))
        lexical_hits = sum(1 for term in all_terms if _has_term(material, term))
        score = target_hits * 16 + relation_hits * 8 + lexical_hits
        document_low_priority_by_id[document.document_id] = _source_is_low_priority(
            document.rel_path,
            " ".join(sentence.text for sentence in sentences),
        )
        if document_low_priority_by_id[document.document_id]:
            score *= 0.2
        if target_terms and not target_hits:
            if relation_hits and score:
                relation_doc_scores.append((score, document.document_id, document.rel_path))
            continue
        if score:
            doc_scores.append((score, document.document_id, document.rel_path))
    doc_scores.sort(key=lambda item: (-item[0], item[2]))
    relation_doc_scores.sort(key=lambda item: (-item[0], item[2]))
    selected_docs = [doc_id for _score, doc_id, _rel_path in doc_scores[:doc_limit]]
    relation_only_selected = 0
    if relation_doc_scores and len(selected_docs) < doc_limit:
        relation_budget = min(doc_limit - len(selected_docs), max(1, min(12, doc_limit // 4)))
        selected_doc_set = set(selected_docs)
        for _score, doc_id, _rel_path in relation_doc_scores:
            if doc_id in selected_doc_set:
                continue
            selected_docs.append(doc_id)
            selected_doc_set.add(doc_id)
            relation_only_selected += 1
            if relation_only_selected >= relation_budget:
                break
    selected_set = set(selected_docs)
    chunk_scores: list[tuple[float, str, int, str]] = []
    for document in documents:
        if document.document_id not in selected_set:
            continue
        ordered = sentences_by_document.get(document.rel_path, {})
        document_has_target = any(_has_term(document_material_by_id.get(document.document_id, ""), term) for term in target_terms)
        for order, sentence in ordered.items():
            material = normalize(sentence.text)
            score = sum(22 for term in target_terms if _has_term(material, term))
            score += sum(11 for term in relation_terms if _has_term(material, term))
            score += sum(2 for term in all_terms if _has_term(material, term))
            if document_has_target and relation_terms and _contains_any(material, relation_terms):
                score += 12
            if _source_is_low_priority(sentence.rel_path, sentence.text):
                score *= 0.15
            if score:
                chunk_scores.append((score, document.document_id, order, document.rel_path))
    chunk_scores.sort(key=lambda item: (-item[0], item[3], item[2]))
    selected_chunks: list[str] = []
    seen: set[str] = set()
    for _score, document_id, order, _rel_path in chunk_scores:
        if len(selected_chunks) >= chunk_limit:
            break
        ordered = sentences_by_document.get(_rel_path, {})
        for nearby in range(order - 4, order + 5):
            if nearby < 0:
                continue
            sentence = ordered.get(nearby)
            if sentence is None or sentence.document_id != document_id:
                continue
            key = stable_id("chunk", sentence.sentence_id)
            if key not in seen:
                seen.add(key)
                selected_chunks.append(key)
                if len(selected_chunks) >= chunk_limit:
                    break
    return selected_docs, selected_chunks, {
        "candidate_document_rows": len(doc_scores),
        "selected_document_count": len(selected_docs),
        "relation_only_candidate_document_rows": len(relation_doc_scores),
        "relation_only_selected_document_count": relation_only_selected,
        "candidate_chunk_rows": len(chunk_scores),
        "selected_chunk_count": len(selected_chunks),
        "target_terms": target_terms[:32],
        "relation_terms": relation_terms[:32],
    }


def _current_chunk_ids_for_documents(
    documents: list[Document],
    sentences_by_document: dict[str, dict[int, Sentence]],
    document_ids: list[str],
) -> list[str]:
    selected = set(document_ids)
    chunk_ids: list[str] = []
    for document in documents:
        if document.document_id not in selected:
            continue
        for sentence in sentences_by_document.get(document.rel_path, {}).values():
            if sentence.document_id == document.document_id:
                chunk_ids.append(stable_id("chunk", sentence.sentence_id))
    return list(dict.fromkeys(chunk_ids))


def _fetch_by_ids(connection: Any, table: str, key: str, ids: list[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    rows: list[dict[str, Any]] = []
    unique = list(dict.fromkeys(ids))
    for index in range(0, len(unique), 400):
        group = unique[index:index + 400]
        placeholders = ",".join("?" for _ in group)
        rows.extend(dict(row) for row in connection.execute(f"SELECT * FROM {table} WHERE {key} IN ({placeholders})", group))
    return rows


def _batched_values(values: list[str], *, size: int = 400) -> list[list[str]]:
    unique = list(dict.fromkeys(item for item in values if item))
    return [unique[index:index + size] for index in range(0, len(unique), size)]


def _fetch_chunks(connection: Any, chunk_ids: list[str]) -> list[dict[str, Any]]:
    return _fetch_by_ids(connection, "chunks", "chunk_id", chunk_ids)


def _merge_rows_by_id(rows: list[dict[str, Any]], extra_rows: list[dict[str, Any]], id_key: str) -> list[dict[str, Any]]:
    merged = {str(row.get(id_key)): row for row in rows}
    for row in extra_rows:
        row_id = str(row.get(id_key) or "")
        if row_id and row_id not in merged:
            merged[row_id] = row
    return list(merged.values())


def _fetch_identity_hypotheses(
    connection: Any,
    run_id: str,
    span_ids: list[str],
    document_ids: list[str],
    current_document_chunk_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}

    def add_rows(sql: str, params: tuple[Any, ...]) -> None:
        for row in connection.execute(sql, params):
            payload = dict(row)
            key = str(payload.get("hypothesis_id") or json.dumps(payload, sort_keys=True, default=str))
            rows_by_id.setdefault(key, payload)

    add_rows(
        """
        SELECT DISTINCT ih.*
        FROM identity_hypotheses ih
        WHERE ih.run_id=? AND ih.source_span_id IS NULL
        """,
        (run_id,),
    )
    for group in _batched_values(span_ids):
        placeholders = ",".join("?" for _ in group)
        add_rows(
            f"""
            SELECT DISTINCT ih.*
            FROM identity_hypotheses ih
            WHERE ih.run_id=? AND ih.source_span_id IN ({placeholders})
            """,
            (run_id, *group),
        )
    if current_document_chunk_ids is not None:
        for group in _batched_values(current_document_chunk_ids):
            placeholders = ",".join("?" for _ in group)
            add_rows(
                f"""
                SELECT DISTINCT ih.*
                FROM identity_hypotheses ih
                JOIN source_spans s ON s.span_id=ih.source_span_id
                WHERE ih.run_id=? AND s.chunk_id IN ({placeholders})
                """,
                (run_id, *group),
            )
    else:
        for group in _batched_values(document_ids):
            placeholders = ",".join("?" for _ in group)
            add_rows(
                f"""
                SELECT DISTINCT ih.*
                FROM identity_hypotheses ih
                JOIN source_spans s ON s.span_id=ih.source_span_id
                WHERE ih.run_id=? AND s.document_id IN ({placeholders})
                """,
                (run_id, *group),
            )
    return list(rows_by_id.values())


def _context_ids_from_rows(rows: list[dict[str, Any]], key: str = "context_id") -> list[str]:
    return list(dict.fromkeys(str(row.get(key) or "") for row in rows if str(row.get(key) or "")))


def _fetch_context_closure(connection: Any, run_id: str, seed_context_ids: list[str]) -> list[dict[str, Any]]:
    pending = list(dict.fromkeys(context_id for context_id in seed_context_ids if context_id))
    contexts_by_id: dict[str, dict[str, Any]] = {}
    while pending:
        next_pending: list[str] = []
        for group in _batched_values([context_id for context_id in pending if context_id not in contexts_by_id]):
            placeholders = ",".join("?" for _ in group)
            rows = connection.execute(
                f"""
                SELECT *
                FROM contexts
                WHERE run_id=? AND context_id IN ({placeholders})
                """,
                (run_id, *group),
            ).fetchall()
            for row in rows:
                context = dict(row)
                context_id = str(context.get("context_id") or "")
                if not context_id or context_id in contexts_by_id:
                    continue
                contexts_by_id[context_id] = context
                parent_id = str(context.get("parent_context_id") or "")
                if parent_id and parent_id not in contexts_by_id:
                    next_pending.append(parent_id)
        pending = list(dict.fromkeys(next_pending))
    return list(contexts_by_id.values())


def _load_records(
    store: Any,
    run_id: str,
    document_ids: list[str],
    chunk_ids: list[str],
    *,
    current_document_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    connection = store.connection
    documents = _fetch_by_ids(connection, "documents", "document_id", document_ids)
    chunks = _fetch_chunks(connection, chunk_ids)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    spans = _fetch_by_ids(connection, "source_spans", "chunk_id", chunk_ids)
    span_ids = [span["span_id"] for span in spans]
    identity_hypotheses = _fetch_identity_hypotheses(
        connection,
        run_id,
        span_ids,
        document_ids,
        current_document_chunk_ids,
    )
    identity_span_ids = list(
        dict.fromkeys(
            str(row.get("source_span_id") or "")
            for row in identity_hypotheses
            if str(row.get("source_span_id") or "")
        )
    )
    extra_span_ids = [span_id for span_id in identity_span_ids if span_id not in set(span_ids)]
    if extra_span_ids:
        extra_spans = _fetch_by_ids(connection, "source_spans", "span_id", extra_span_ids)
        spans = _merge_rows_by_id(spans, extra_spans, "span_id")
        extra_chunk_ids = [
            str(span.get("chunk_id") or "")
            for span in extra_spans
            if str(span.get("chunk_id") or "") and str(span.get("chunk_id") or "") not in set(chunk_ids)
        ]
        if extra_chunk_ids:
            chunks = _merge_rows_by_id(chunks, _fetch_by_ids(connection, "chunks", "chunk_id", extra_chunk_ids), "chunk_id")
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        span_ids = [span["span_id"] for span in spans]
    drs_identity_hypotheses = _fetch_by_ids(
        connection,
        "drs_identity_hypotheses",
        "source_span_id",
        span_ids,
    )
    frames = _fetch_by_ids(connection, "frames", "span_id", span_ids)
    arguments = _fetch_by_ids(connection, "frame_arguments", "frame_id", [frame["frame_id"] for frame in frames])
    relations = _fetch_by_ids(connection, "relations", "source_span_id", span_ids)
    temporal = _fetch_by_ids(connection, "temporal_edges", "source_span_id", span_ids)
    metadata_records = _fetch_by_ids(connection, "metadata_records", "document_id", document_ids)
    material_referent_ids = list(
        dict.fromkeys(
            str(row.get(key) or "")
            for row in identity_hypotheses
            for key in ["left_referent_id", "right_referent_id"]
            if str(row.get(key) or "")
        )
    )
    material_referent_ids = list(
        dict.fromkeys(
            [
                *material_referent_ids,
                *[
                    str(row.get("referent_id") or "")
                    for row in temporal
                    if str(row.get("referent_id") or "")
                ],
            ]
        )
    )
    referents = _fetch_by_ids(connection, "referents", "referent_id", material_referent_ids)
    context_carriers = _fetch_by_ids(connection, "context_carriers", "document_id", document_ids)
    seed_context_ids = list(
        dict.fromkeys(
            [
                *_context_ids_from_rows(frames),
                *_context_ids_from_rows(relations),
                *_context_ids_from_rows(temporal),
                *_context_ids_from_rows(identity_hypotheses),
                *_context_ids_from_rows(context_carriers),
            ]
        )
    )
    contexts = _fetch_context_closure(connection, run_id, seed_context_ids)
    docs_by_document_id = {str(doc.get("document_id")): doc for doc in documents}
    document_context_norm_by_rel_path: dict[str, str] = defaultdict(str)
    for chunk in chunks:
        doc = docs_by_document_id.get(str(chunk.get("document_id")), {})
        rel_path = str(doc.get("rel_path") or "")
        document_context_norm_by_rel_path[rel_path] += " " + normalize(str(chunk.get("text") or ""))
    return {
        "documents": documents,
        "chunks": chunks,
        "source_spans": spans,
        "frames": frames,
        "frame_arguments": arguments,
        "relations": relations,
        "temporal_edges": temporal,
        "metadata_records": metadata_records,
        "identity_hypotheses": identity_hypotheses,
        "drs_identity_hypotheses": drs_identity_hypotheses,
        "referents": referents,
        "contexts": contexts,
        "context_carriers": context_carriers,
        "document_context_norm_by_rel_path": dict(document_context_norm_by_rel_path),
        "record_counts": {
            "documents": len(documents),
            "chunks": len(chunks),
            "source_spans": len(spans),
            "frames": len(frames),
            "frame_arguments": len(arguments),
            "temporal_edges": len(temporal),
            "relations": len(relations),
            "metadata_records": len(metadata_records),
            "identity_hypotheses": len(identity_hypotheses),
            "drs_identity_hypotheses": len(drs_identity_hypotheses),
            "referents": len(referents),
            "contexts": len(contexts),
            "context_carriers": len(context_carriers),
        },
    }


def _indexed_rows(records: dict[str, Any], cache_key: str, table_key: str, id_key: str) -> dict[str, dict[str, Any]]:
    indexes = records.setdefault("_indexes", {})
    if cache_key not in indexes:
        indexes[cache_key] = {str(row.get(id_key)): row for row in records.get(table_key, [])}
    return indexes[cache_key]


def _docs_by_id(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _indexed_rows(records, "documents_by_id", "documents", "document_id")


def _docs_by_rel_path(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexes = records.setdefault("_indexes", {})
    if "documents_by_rel_path" not in indexes:
        indexes["documents_by_rel_path"] = {
            str(row.get("rel_path") or ""): row
            for row in records.get("documents", [])
        }
    return indexes["documents_by_rel_path"]


def _chunks_by_id(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _indexed_rows(records, "chunks_by_id", "chunks", "chunk_id")


def _spans_by_id(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _indexed_rows(records, "spans_by_id", "source_spans", "span_id")


def _contexts_by_id(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _indexed_rows(records, "contexts_by_id", "contexts", "context_id")


def _context_chain(context_id: str, records: dict[str, Any]) -> list[dict[str, Any]]:
    contexts = _contexts_by_id(records)
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_id = str(context_id or "")
    while current_id and current_id not in seen:
        seen.add(current_id)
        context = contexts.get(current_id)
        if not context:
            break
        chain.append(context)
        current_id = str(context.get("parent_context_id") or "")
    return chain


def _context_chain_material(context_id: str, records: dict[str, Any]) -> str:
    fields: list[str] = []
    for context in _context_chain(context_id, records):
        fields.extend(
            [
                str(context.get("kind") or ""),
                str(context.get("holder_surface") or ""),
                str(context.get("evidence_surface") or ""),
            ]
        )
    return normalize(" ".join(fields))


def _context_requirements(frame: QueryFrame) -> list[str]:
    values = [*frame.modality_requirements, *frame.scope_requirements]
    return list(dict.fromkeys(normalize(value) for value in values if normalize(value)))


def _terms_match_material(terms: list[str], material: str, *, use_morphology: bool = True) -> bool:
    if not terms or not material:
        return False
    material_tokens = set(content_tokens(material))
    expanded_material_tokens = set(material_tokens)
    if use_morphology:
        for token in material_tokens:
            expanded_material_tokens.update(term_variants(token))
    for term in terms:
        if term in material:
            return True
        term_tokens = [token for token in content_tokens(term) if token]
        if term_tokens and all(
            token in expanded_material_tokens
            or (
                use_morphology
                and any(variant in expanded_material_tokens for variant in term_variants(token))
            )
            for token in term_tokens
        ):
            return True
    return False


def _context_satisfies_terms(context_id: str, records: dict[str, Any], terms: list[str], *, require_all: bool) -> bool:
    material = _context_chain_material(context_id, records)
    if not terms:
        return True
    if require_all:
        return all(_terms_match_material([term], material) for term in terms)
    return _terms_match_material(terms, material)


def _context_satisfies_requirements(context_id: str, records: dict[str, Any], frame: QueryFrame) -> bool:
    requirements = _context_requirements(frame)
    return _context_satisfies_terms(context_id, records, requirements, require_all=True)


def _context_requested_by_relation(context_id: str, records: dict[str, Any], frame: QueryFrame) -> bool:
    requested = normalize(frame.requested_relation)
    if not requested:
        return False
    material = _context_chain_material(context_id, records)
    return _terms_match_material([requested], material, use_morphology=False)


def _identity_context_accessible(context_id: str, records: dict[str, Any], frame: QueryFrame | None) -> bool:
    if frame is None or not context_id:
        return True
    chain = _context_chain(context_id, records)
    if not chain:
        return True
    for context in chain:
        kind = normalize(str(context.get("kind") or "asserted"))
        if kind and kind not in {"asserted", "drs:asserted"}:
            return _context_accessible(context_id, records, frame)
    return True


def _referents_by_id(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _indexed_rows(records, "referents_by_id", "referents", "referent_id")


def _identity_expanded_terms(
    records: dict[str, Any],
    terms: list[str],
    frame: QueryFrame | None = None,
) -> list[str]:
    expanded_terms, _evidence = _identity_expansion(records, terms, frame)
    return expanded_terms


def _dedupe_evidence(items: list[Evidence], *, limit: int = 12) -> list[Evidence]:
    values: list[Evidence] = []
    seen: set[tuple[str, str, int | None, str]] = set()
    for item in items:
        key = (item.rel_path, item.span_id, item.chunk_order, item.text)
        if not item.rel_path or not item.text or key in seen:
            continue
        seen.add(key)
        values.append(item)
        if len(values) >= limit:
            break
    return values


def _identity_expansion(
    records: dict[str, Any],
    terms: list[str],
    frame: QueryFrame | None = None,
) -> tuple[list[str], list[Evidence]]:
    if not terms:
        return [], []
    referents = _referents_by_id(records)
    seed_ids: set[str] = set()
    normalized_terms = [normalize(term) for term in terms if normalize(term)]
    seed_terms = [
        term for term in normalized_terms
        if " " in term or "_" in term or "-" in term or "/" in term or "." in term
    ]
    if not seed_terms:
        return [], []
    seed_token_sets = [_normalized_token_set(term) for term in seed_terms]
    for referent_id, row in referents.items():
        label_norm = normalize(str(row.get("canonical_label") or row.get("canonical_label_norm") or ""))
        label_tokens = _normalized_token_set(label_norm)
        if label_norm and any(
            label_norm == term or (label_tokens and label_tokens == term_tokens)
            for term, term_tokens in zip(seed_terms, seed_token_sets)
        ):
            seed_ids.add(referent_id)
    if not seed_ids:
        return [], []
    expanded: list[str] = []
    expansion_evidence: list[Evidence] = []
    seen_edges: set[str] = set()
    frontier = set(seed_ids)
    visited = set(seed_ids)
    for _depth in range(IDENTITY_GRAPH_MAX_DEPTH):
        next_frontier: set[str] = set()
        for hypothesis in records.get("identity_hypotheses", []):
            if not identity_relation_allows_expansion(str(hypothesis.get("relation") or "")):
                continue
            context_id = str(hypothesis.get("context_id") or "")
            if not _identity_context_accessible(context_id, records, frame):
                continue
            left = str(hypothesis.get("left_referent_id") or "")
            right = str(hypothesis.get("right_referent_id") or "")
            if left in frontier and right and right not in visited:
                next_frontier.add(right)
                edge_id = str(hypothesis.get("hypothesis_id") or f"{left}->{right}")
                if edge_id not in seen_edges:
                    seen_edges.add(edge_id)
                    expansion_evidence.append(_evidence_for_span(str(hypothesis.get("source_span_id") or ""), records))
            if right in frontier and left and left not in visited:
                next_frontier.add(left)
                edge_id = str(hypothesis.get("hypothesis_id") or f"{right}->{left}")
                if edge_id not in seen_edges:
                    seen_edges.add(edge_id)
                    expansion_evidence.append(_evidence_for_span(str(hypothesis.get("source_span_id") or ""), records))
        if not next_frontier:
            break
        visited.update(next_frontier)
        frontier = next_frontier
    for referent_id in visited:
        row = referents.get(referent_id, {})
        label = str(row.get("canonical_label") or "")
        if label:
            label_norm = normalize(label)
            expanded.append(label_norm)
            if " " in label_norm:
                expanded.append(label_norm.replace(" ", "_"))
                expanded.append(label_norm.replace(" ", "-"))
    return list(dict.fromkeys(term for term in expanded if term)), _dedupe_evidence(expansion_evidence)


def _evidence_for_span(span_id: str, records: dict[str, Any]) -> Evidence:
    span = _spans_by_id(records).get(span_id, {})
    chunk = _chunks_by_id(records).get(str(span.get("chunk_id")), {})
    doc = _docs_by_id(records).get(str(span.get("document_id")), {})
    chunk_order: int | None
    try:
        chunk_order = int(chunk["chunk_order"]) if chunk.get("chunk_order") is not None else None
    except (TypeError, ValueError):
        chunk_order = None
    try:
        char_start = int(span["char_start"]) if span.get("char_start") is not None else None
    except (TypeError, ValueError):
        char_start = None
    try:
        char_end = int(span["char_end"]) if span.get("char_end") is not None else None
    except (TypeError, ValueError):
        char_end = None
    return Evidence(
        str(doc.get("rel_path") or ""),
        str(chunk.get("text") or span.get("surface") or ""),
        0.78,
        span_id=str(span.get("span_id") or span_id),
        chunk_order=chunk_order,
        char_start=char_start,
        char_end=char_end,
        source_kind=str(span.get("span_kind") or "source_span"),
    )


def _evidence_payload(item: Evidence, *, text_limit: int = 500) -> dict[str, Any]:
    return {
        "rel_path": item.rel_path,
        "span_id": item.span_id,
        "chunk_order": item.chunk_order,
        "char_start": item.char_start,
        "char_end": item.char_end,
        "source_kind": item.source_kind,
        "text": item.text[:text_limit],
    }


def _document_metadata_summary(row: dict[str, Any]) -> dict[str, Any]:
    try:
        metadata = json.loads(str(row.get("metadata_json") or "{}"))
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    text_quality = metadata.get("text_quality")
    semantic_quality = (
        text_quality.get("semantic_quality")
        if isinstance(text_quality, dict)
        else metadata.get("semantic_quality")
    )
    summary: dict[str, Any] = {
        "document_id": row.get("document_id"),
        "rel_path": row.get("rel_path"),
        "size_bytes": row.get("size_bytes"),
        "char_count": row.get("char_count"),
    }
    for key in ["file_name", "suffix", "parent_rel_path", "mime_type"]:
        if key in metadata:
            summary[key] = metadata[key]
    if semantic_quality is not None and semantic_quality != "":
        summary["semantic_quality"] = semantic_quality
    return {key: value for key, value in summary.items() if value is not None and value != ""}


def _span_provenance_payload(span: dict[str, Any], records: dict[str, Any]) -> dict[str, Any]:
    span_id = str(span.get("span_id") or "")
    evidence = _evidence_for_span(span_id, records)
    chunk = _chunks_by_id(records).get(str(span.get("chunk_id") or ""), {})
    doc = _docs_by_id(records).get(str(span.get("document_id") or ""), {})
    payload = _evidence_payload(evidence)
    payload.update(
        {
            "document_id": span.get("document_id"),
            "chunk_id": span.get("chunk_id"),
            "span_kind": span.get("span_kind"),
            "document": _document_metadata_summary(doc),
        }
    )
    if chunk.get("token_estimate") is not None:
        payload["token_estimate"] = chunk.get("token_estimate")
    return payload


def _source_provenance_sample(
    records: dict[str, Any],
    target_terms: list[str],
    relation_terms: list[str],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    rows: list[tuple[float, int, str, int, str, dict[str, Any]]] = []
    for span in records.get("source_spans", []):
        span_id = str(span.get("span_id") or "")
        if not span_id:
            continue
        evidence = _evidence_for_span(span_id, records)
        material = normalize(" ".join([evidence.rel_path, evidence.text]))
        target_hits = sum(1 for term in target_terms if _has_term(material, term))
        relation_hits = sum(1 for term in relation_terms if _has_term(material, term))
        if target_terms or relation_terms:
            if target_terms and relation_terms and not (target_hits or relation_hits):
                continue
            if target_terms and not relation_terms and not target_hits:
                continue
            if relation_terms and not target_terms and not relation_hits:
                continue
        order = evidence.chunk_order if evidence.chunk_order is not None else -1
        score = target_hits * 4.0 + relation_hits * 3.0 + 0.1
        span_kind = str(span.get("span_kind") or "")
        kind_priority = 0 if span_kind in {"chunk", "sentence"} else 1
        rows.append((score, kind_priority, evidence.rel_path, order, span_id, _span_provenance_payload(span, records)))
    rows.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    seen: set[tuple[str, str, int | None, str]] = set()
    seen_chunks: set[tuple[str, int | None, str]] = set()
    ranked_payloads: list[dict[str, Any]] = []
    for _score, _kind_priority, _rel_path, _order, _span_id, payload in rows:
        key = (
            str(payload.get("rel_path") or ""),
            str(payload.get("span_id") or ""),
            payload.get("chunk_order") if isinstance(payload.get("chunk_order"), int) else None,
            str(payload.get("text") or ""),
        )
        chunk_key = (
            str(payload.get("rel_path") or ""),
            payload.get("chunk_order") if isinstance(payload.get("chunk_order"), int) else None,
            normalize(str(payload.get("text") or "")),
        )
        if key in seen or chunk_key in seen_chunks:
            continue
        seen.add(key)
        seen_chunks.add(chunk_key)
        ranked_payloads.append(payload)

    if target_terms and relation_terms and limit >= 2:
        selected: list[dict[str, Any]] = []
        selected_keys: set[tuple[str, str, int | None, str]] = set()

        def add(payload: dict[str, Any]) -> None:
            key = _provenance_payload_key(payload)
            if key in selected_keys or len(selected) >= limit:
                return
            selected_keys.add(key)
            selected.append(payload)

        quota = max(1, min(3, limit // 3))
        target_payloads = [
            payload
            for payload in ranked_payloads
            if _contains_any(_provenance_payload_material(payload), target_terms)
        ]
        relation_payloads = [
            payload
            for payload in ranked_payloads
            if _contains_any(_provenance_payload_material(payload), relation_terms)
        ]

        def diverse(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
            values: list[dict[str, Any]] = []
            value_keys: set[tuple[str, str, int | None, str]] = set()
            seen_paths: set[str] = set()
            for payload in payloads:
                rel_path = str(payload.get("rel_path") or "")
                if rel_path and rel_path in seen_paths:
                    continue
                key = _provenance_payload_key(payload)
                value_keys.add(key)
                values.append(payload)
                if rel_path:
                    seen_paths.add(rel_path)
                if len(values) >= quota:
                    return values
            for payload in payloads:
                key = _provenance_payload_key(payload)
                if key in value_keys:
                    continue
                value_keys.add(key)
                values.append(payload)
                if len(values) >= quota:
                    break
            return values

        for payload in diverse(target_payloads):
            add(payload)
        for payload in diverse(relation_payloads):
            add(payload)
        for payload in ranked_payloads:
            add(payload)
        return selected[:limit]

    return ranked_payloads[:limit]


def _candidate_evidence_sample(
    candidates: list[tuple[float, str, Evidence, str]],
    expected: ExpectedAnswer,
    records: dict[str, Any] | None = None,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    rows: list[tuple[float, str, dict[str, Any]]] = []
    for score, value, evidence, reason in candidates:
        canonical = canonicalize_answer(expected, value) or clean_extracted_value(value)
        if not canonical:
            continue
        rows.append(
            (
                float(score),
                canonical,
                {
                    "value": canonical,
                    "score": round(float(score), 3),
                    "reason": reason,
                    "evidence": (
                        _evidence_provenance_payload(evidence, records)
                        if records is not None
                        else _evidence_payload(evidence)
                    ),
                },
            )
        )
    rows.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    return [payload for _score, _value, payload in rows[:limit]]


def _identity_row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("metadata_json") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _blocked_identity_provenance_sample(
    records: dict[str, Any],
    target_terms: list[str],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    rows: list[tuple[str, int, str, dict[str, Any]]] = []
    for row in records.get("drs_identity_hypotheses", []):
        metadata = _identity_row_metadata(row)
        reason = str(metadata.get("expansion_blocked_reason") or "")
        if not reason:
            continue
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        material = normalize(
            " ".join(
                [
                    evidence.rel_path,
                    evidence.text,
                    str(row.get("evidence_surface") or ""),
                ]
            )
        )
        if target_terms and not _contains_any(material, target_terms):
            continue
        payload = _evidence_provenance_payload(evidence, records)
        payload.update(
            {
                "expansion_blocked_reason": reason,
                "identity_evidence": str(row.get("evidence_surface") or ""),
                "box_external_id": row.get("box_external_id"),
                "resolved_box_external_id": metadata.get("resolved_box_external_id"),
            }
        )
        rows.append(
            (
                evidence.rel_path,
                evidence.chunk_order if evidence.chunk_order is not None else -1,
                str(row.get("drs_hypothesis_id") or ""),
                payload,
            )
        )
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    return [payload for _rel_path, _chunk_order, _row_id, payload in rows[:limit]]


def _provenance_payload_material(payload: dict[str, Any]) -> str:
    return normalize(" ".join([str(payload.get("rel_path") or ""), str(payload.get("text") or "")]))


def _provenance_payload_key(payload: dict[str, Any]) -> tuple[str, str, int | None, str]:
    return (
        str(payload.get("rel_path") or ""),
        str(payload.get("span_id") or ""),
        payload.get("chunk_order") if isinstance(payload.get("chunk_order"), int) else None,
        str(payload.get("text") or ""),
    )


def _scattered_source_provenance_without_binding(
    provenance_sample: list[dict[str, Any]],
    target_terms: list[str],
    relation_terms: list[str],
) -> dict[str, Any] | None:
    if not target_terms or not relation_terms:
        return None
    target_payloads = [
        payload
        for payload in provenance_sample
        if _contains_any(_provenance_payload_material(payload), target_terms)
    ]
    relation_payloads = [
        payload
        for payload in provenance_sample
        if _contains_any(_provenance_payload_material(payload), relation_terms)
    ]
    if not target_payloads or not relation_payloads:
        return None
    if not any(
        _provenance_payload_key(target_payload) != _provenance_payload_key(relation_payload)
        for target_payload in target_payloads
        for relation_payload in relation_payloads
    ):
        return None
    target_sources = _dedupe_provenance_payloads(target_payloads, limit=6)
    relation_sources = _dedupe_provenance_payloads(relation_payloads, limit=6)
    return {
        "target_rel_paths": sorted(
            {str(payload.get("rel_path") or "") for payload in target_payloads if str(payload.get("rel_path") or "")}
        )[:6],
        "relation_rel_paths": sorted(
            {str(payload.get("rel_path") or "") for payload in relation_payloads if str(payload.get("rel_path") or "")}
        )[:6],
        "target_sources": target_sources,
        "relation_sources": relation_sources,
    }


def _attach_no_answer_provenance(
    diagnostics: dict[str, Any],
    records: dict[str, Any],
    target_terms: list[str],
    relation_terms: list[str],
    candidates: list[tuple[float, str, Evidence, str]],
    expected: ExpectedAnswer,
    reason: str,
) -> None:
    execution = diagnostics.setdefault("execution", {})
    execution["no_answer_reason"] = reason
    candidate_sample = _candidate_evidence_sample(candidates, expected, records)
    if candidate_sample:
        execution["candidate_evidence_sample"] = candidate_sample
    provenance_sample = _source_provenance_sample(records, target_terms, relation_terms)
    if provenance_sample:
        execution["source_provenance_sample"] = provenance_sample
        scattered = _scattered_source_provenance_without_binding(
            provenance_sample,
            target_terms,
            relation_terms,
        )
        if scattered and not candidates:
            execution["scattered_source_provenance_without_binding"] = scattered
    blocked_identity_sample = _blocked_identity_provenance_sample(records, target_terms)
    if blocked_identity_sample:
        execution["blocked_identity_source_provenance"] = blocked_identity_sample


def _answer_source_provenance_sample(
    answer: Answer,
    records: dict[str, Any],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None, str]] = set()
    for evidence in answer.evidence:
        payload = _evidence_provenance_payload(evidence, records)
        key = (
            str(payload.get("rel_path") or ""),
            str(payload.get("span_id") or ""),
            payload.get("chunk_order") if isinstance(payload.get("chunk_order"), int) else None,
            str(payload.get("text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(payload)
        if len(rows) >= limit:
            break
    return rows


def _evidence_provenance_payload(evidence: Evidence, records: dict[str, Any]) -> dict[str, Any]:
    span = _spans_by_id(records).get(evidence.span_id)
    if span:
        return _span_provenance_payload(span, records)
    payload = _evidence_payload(evidence)
    doc = _docs_by_rel_path(records).get(evidence.rel_path)
    if doc:
        payload["document"] = _document_metadata_summary(doc)
        if doc.get("document_id") is not None:
            payload["document_id"] = doc.get("document_id")
    return payload


def _dedupe_provenance_payloads(payloads: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None, str]] = set()
    for payload in payloads:
        key = (
            str(payload.get("rel_path") or ""),
            str(payload.get("span_id") or ""),
            payload.get("chunk_order") if isinstance(payload.get("chunk_order"), int) else None,
            str(payload.get("text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(payload)
        if len(rows) >= limit:
            break
    return rows


def _attach_answer_provenance(
    diagnostics: dict[str, Any],
    records: dict[str, Any],
    answer: Answer | None,
) -> None:
    if answer is None:
        return
    provenance = _answer_source_provenance_sample(answer, records)
    if provenance:
        diagnostics.setdefault("execution", {})["answer_source_provenance"] = provenance


def _metadata_evidence(record: dict[str, Any], records: dict[str, Any]) -> Evidence:
    doc = _docs_by_id(records).get(str(record.get("document_id")), {})
    return Evidence(
        str(doc.get("rel_path") or ""),
        f"metadata {record.get('key')}: {record.get('value')}",
        0.72,
        source_kind="metadata",
    )


def _context_accessible(context_id: str, records: dict[str, Any], frame: QueryFrame) -> bool:
    chain = _context_chain(context_id, records)
    if not chain:
        return True
    if not _context_satisfies_requirements(context_id, records, frame):
        return False
    requirements = _context_requirements(frame)
    relation_requests_context = _context_requested_by_relation(context_id, records, frame)
    for context in chain:
        kind = normalize(str(context.get("kind") or "asserted"))
        if not kind or kind == "asserted":
            continue
        if kind.startswith("polarity:") and frame.answer_type != "boolean" and not frame.negated:
            context_surface = normalize(
                " ".join([kind, str(context.get("holder_surface") or "")])
            )
            if not _terms_match_material(requirements, context_surface):
                return False
        if kind.startswith(INACCESSIBLE_CONTEXT_PREFIXES):
            context_surface = normalize(
                " ".join([kind, str(context.get("holder_surface") or "")])
            )
            if kind.startswith("modality:") and (
                relation_requests_context or _terms_match_material(requirements, context_surface)
            ):
                continue
            return False
        if kind.startswith("drs:") and kind != "drs:asserted":
            context_surface = normalize(
                " ".join([kind, str(context.get("holder_surface") or ""), str(context.get("evidence_surface") or "")])
            )
            if _terms_match_material(requirements, context_surface):
                continue
            if kind == "drs:negated" and (frame.answer_type == "boolean" or frame.negated):
                continue
            return False
    return True


def _relation_scope_accessible(row: dict[str, Any], records: dict[str, Any], frame: QueryFrame) -> bool:
    context_id = str(row.get("context_id") or "")
    requirements = _context_requirements(frame)
    relation_type = str(row.get("relation_type") or "")
    declared_scope = normalize(str(row.get("subject") or "")) if relation_type == "drs_condition" else ""
    scope_material = normalize(
        " ".join(
            [
                declared_scope,
                str(row.get("object") or ""),
                _context_chain_material(context_id, records),
            ]
        )
    )
    if declared_scope and declared_scope != "asserted":
        if not (requirements and all(_terms_match_material([requirement], scope_material) for requirement in requirements)):
            return False
    if _context_accessible(context_id, records, frame):
        return True
    if not requirements:
        return False
    chain = _context_chain(context_id, records)
    for context in chain:
        kind = normalize(str(context.get("kind") or "asserted"))
        if kind.startswith(INACCESSIBLE_CONTEXT_PREFIXES) or kind.startswith("polarity:"):
            return False
        if kind.startswith("drs:") and kind != "drs:asserted":
            return False
    return all(_terms_match_material([requirement], scope_material) for requirement in requirements)


def _relation_metadata(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("metadata_json") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _structured_source_row(row: dict[str, Any]) -> bool:
    metadata = _relation_metadata(row)
    return str(row.get("relation_type") or "") in {"record_value", "table_cell"} or str(
        metadata.get("surface_format") or ""
    ) in {"json", "json_like", "object_like", "delimited_table", "label_url"}


def _expected_from_frame(frame: QueryFrame) -> ExpectedAnswer:
    allowed = {
        "person",
        "actor",
        "organization",
        "identifier",
        "url",
        "file_path",
        "count",
        "state",
        "date_time",
        "boolean",
        "content_phrase",
        "metadata_value",
        "unknown",
    }
    answer_type = frame.answer_type if frame.answer_type in allowed else "unknown"
    return ExpectedAnswer(answer_type, allow_metadata_evidence=answer_type == "metadata_value")  # type: ignore[arg-type]


def _condition_material(row: dict[str, Any], evidence: Evidence, records: dict[str, Any]) -> str:
    metadata = _relation_metadata(row)
    fields = [
        row.get("relation_type"), row.get("subject"), row.get("predicate"), row.get("object"), row.get("value"),
        metadata.get("record_path"), metadata.get("record_group"), metadata.get("row_key"), metadata.get("column_header"), metadata.get("section_anchor"),
        evidence.rel_path, evidence.text,
        (records.get("document_context_norm_by_rel_path") or {}).get(evidence.rel_path, ""),
    ]
    return normalize(" ".join(str(item or "") for item in fields))


def _relation_local_material(
    row: dict[str, Any],
    evidence: Evidence | None = None,
    *,
    include_evidence: bool = False,
    include_context: bool = False,
    records: dict[str, Any] | None = None,
) -> str:
    metadata = _relation_metadata(row)
    fields = [
        row.get("relation_type"),
        row.get("subject"),
        row.get("predicate"),
        row.get("object"),
        row.get("value"),
        metadata.get("record_path"),
        metadata.get("row_key"),
        metadata.get("column_header"),
        metadata.get("section_anchor"),
        metadata.get("argument_role"),
        metadata.get("argument_value_type"),
    ]
    if include_evidence and evidence is not None:
        fields.append(evidence.rel_path)
        fields.append(evidence.text)
    if include_context and records is not None:
        fields.append(_context_chain_material(str(row.get("context_id") or ""), records))
    return normalize(" ".join(str(item or "") for item in fields))


def _compatible_values(expected: ExpectedAnswer, values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        text = clean_extracted_value(str(value or ""))
        if not text:
            continue
        if expected.answer_type == "url":
            cleaned.extend(url.rstrip(".,;)") for url in urls(text))
        elif expected.answer_type == "identifier":
            cleaned.extend(identifier.rstrip(".,;)") for identifier in identifiers(text))
        elif expected.answer_type == "file_path":
            without_urls = text
            for url in urls(text):
                without_urls = without_urls.replace(url, " ")
            cleaned.extend(match.group(0).rstrip(".,;)") for match in PATH_RE.finditer(without_urls))
        elif expected.answer_type == "date_time":
            cleaned.extend(match.group(0) for match in DATE_TIME_RE.finditer(text))
        elif expected.answer_type == "count":
            cleaned.extend(match.group(0) for match in re.finditer(r"\b\d+\b", text))
        elif is_value_compatible(expected, text):
            cleaned.append(text)
    return list(dict.fromkeys(value for value in cleaned if canonicalize_answer(expected, value)))


def _unknown_query_requests_url(
    expected: ExpectedAnswer,
    relation_terms: list[str],
    answer_slot_terms: list[str] | None = None,
) -> bool:
    if expected.answer_type != "unknown":
        return False
    terms = [normalize(term) for term in [*relation_terms, *(answer_slot_terms or [])]]
    return any(term in {"url", "uri", "link"} for term in terms)


def _matching_structural_label_values(
    value: str,
    relation_terms: list[str],
    answer_slot_terms: list[str] | None = None,
) -> list[str]:
    text = clean_extracted_value(value)
    if not text:
        return []
    separator = ":" if ":" in text else "=" if "=" in text else ""
    if not separator:
        return []
    label, rest = text.split(separator, 1)
    label_material = normalize(label)
    answer_text = clean_extracted_value(rest)
    if not label_material or not answer_text:
        return []
    terms = list(dict.fromkeys([*relation_terms, *(answer_slot_terms or [])]))
    if terms and not _contains_any(label_material, terms):
        return []
    return [answer_text]


def _value_is_target(value: str, target_terms: list[str]) -> bool:
    material = normalize(value)
    if not material or not target_terms:
        return False
    terms_key = tuple(target_terms)
    if material in _normalized_term_set(terms_key):
        return True
    material_tokens = _normalized_token_set(material)
    for term_tokens in _normalized_term_token_sets(terms_key):
        if term_tokens and material_tokens == term_tokens:
            return True
    return False


def _value_contains_target(value: str, target_terms: list[str]) -> bool:
    material = normalize(value)
    if not material or not target_terms:
        return False
    material_tokens = _normalized_token_set(material)
    terms_key = tuple(target_terms)
    for term_norm, term_tokens in zip(_normalized_terms(terms_key), _normalized_term_token_sets(terms_key)):
        if not term_norm or material == term_norm:
            continue
        if term_tokens and term_tokens.issubset(material_tokens):
            return True
        if " " in term_norm and term_norm in material:
            return True
    return False


def _rejects_bound_target_value(expected: ExpectedAnswer, value: str, target_terms: list[str]) -> bool:
    if expected.answer_type in {"content_phrase", "metadata_value", "unknown"}:
        return False
    return _value_contains_target(value, target_terms)


def _answer_values_from_relation(
    row: dict[str, Any],
    evidence: Evidence,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
    answer_slot_terms: list[str] | None = None,
) -> list[str]:
    relation_type = str(row.get("relation_type") or "")
    if relation_type == "semantic_argument" and answer_slot_terms:
        metadata = _relation_metadata(row)
        slot_material = normalize(
            " ".join(
                [
                    str(row.get("subject") or ""),
                    str(metadata.get("argument_role") or ""),
                    str(metadata.get("argument_value_type") or ""),
                ]
            )
        )
        if slot_material and not _contains_any(slot_material, answer_slot_terms):
            return []
    if relation_type == "semantic_frame":
        primary_values = []
    elif relation_type == "drs_condition":
        primary_values = [str(row.get("value") or "")] if expected.answer_type in {"content_phrase", "unknown"} else []
    else:
        primary_values = [str(row.get(key) or "") for key in ["value", "object"]]
    fallback_values = (
        []
        if relation_type in {"semantic_argument", "semantic_frame", "drs_condition"}
        else [str(row.get("subject") or "")]
    )
    label_values = [
        split_value
        for value in [*primary_values, *fallback_values]
        for split_value in _matching_structural_label_values(value, relation_terms, answer_slot_terms)
    ]
    if label_values:
        primary_values = label_values
        fallback_values = []
    url_requested = _unknown_query_requests_url(expected, relation_terms, answer_slot_terms)
    structural = expected.answer_type in {"url", "identifier", "file_path", "date_time", "count"} or url_requested
    primary_values = [
        value for value in primary_values
        if value
        and (structural or not _value_is_target(value, target_terms))
        and (structural or not _rejects_bound_target_value(expected, value, target_terms))
        and (structural or not _value_is_target(value, relation_terms))
    ]
    fallback_values = [
        value for value in fallback_values
        if value
        and (structural or not _value_is_target(value, target_terms))
        and (structural or not _rejects_bound_target_value(expected, value, target_terms))
        and (structural or not _value_is_target(value, relation_terms))
    ]
    if url_requested:
        return list(dict.fromkeys(url.rstrip(".,;)") for value in [*primary_values, *fallback_values] for url in urls(value)))
    compatible = _compatible_values(expected, primary_values)
    if not compatible:
        compatible = _compatible_values(expected, fallback_values)
    if compatible or not structural:
        return compatible
    return _compatible_values(expected, [evidence.text])


def _answer_values_from_frame(
    frame_row: dict[str, Any],
    args: list[dict[str, Any]],
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
    answer_slot_terms: list[str] | None = None,
    frame_type_material: str = "",
) -> list[str]:
    candidate_args = args
    if answer_slot_terms:
        slot_args = [
            arg for arg in args
            if _contains_any(
                normalize(" ".join([str(arg.get("role") or ""), str(arg.get("value_type") or "")])),
                answer_slot_terms,
            )
        ]
        if slot_args:
            candidate_args = slot_args
        elif expected.answer_type not in {"content_phrase", "unknown"}:
            predicate_slot_material = normalize(
                " ".join(
                    [
                        frame_type_material,
                        str(frame_row.get("predicate") or ""),
                        str(frame_row.get("trigger_surface") or ""),
                    ]
                )
            )
            if str(frame_row.get("source") or "") != "local_model" or not _contains_any(
                predicate_slot_material,
                answer_slot_terms,
            ):
                return []
            candidate_args = []
    values = [str(arg.get("surface") or "") for arg in candidate_args]
    url_requested = _unknown_query_requests_url(expected, relation_terms, answer_slot_terms)
    structural = expected.answer_type in {"url", "identifier", "file_path", "date_time", "count"} or url_requested
    values = [
        value for value in values
        if value
        and (structural or not _value_is_target(value, target_terms))
        and (structural or not _rejects_bound_target_value(expected, value, target_terms))
        and (structural or not _value_is_target(value, relation_terms))
    ]
    if url_requested:
        return list(dict.fromkeys(url.rstrip(".,;)") for value in values for url in urls(value)))
    compatible = _compatible_values(expected, values)
    if compatible or structural:
        return compatible
    if str(frame_row.get("source") or "") != "local_model":
        return []
    predicate_values = [
        str(frame_row.get(key) or "")
        for key in ["predicate", "trigger_surface"]
        if str(frame_row.get(key) or "")
    ]
    predicate_values = [
        value for value in predicate_values
        if not _value_is_target(value, target_terms)
        and not _rejects_bound_target_value(expected, value, target_terms)
        and not _value_is_target(value, relation_terms)
    ]
    return _compatible_values(expected, predicate_values)


def _match_score(material: str, target_terms: list[str], relation_terms: list[str]) -> float:
    target_hits = sum(1 for term in target_terms if _has_term(material, term))
    relation_matches = {term[:5] for term in relation_terms if _has_term(material, term)}
    relation_hits = len(relation_matches)
    if target_terms and target_hits == 0:
        return 0.0
    if relation_terms and relation_hits == 0:
        return 0.0
    if not target_terms and len({term[:5] for term in relation_terms}) >= 2 and relation_hits < 2:
        return 0.0
    return target_hits * 4.0 + relation_hits * 3.0 + 1.0


def _split_match_score(full_material: str, local_material: str, target_terms: list[str], relation_terms: list[str]) -> float:
    target_hits = sum(1 for term in target_terms if _has_term(full_material, term))
    relation_matches = {term[:5] for term in relation_terms if _has_term(local_material, term)}
    relation_hits = len(relation_matches)
    if target_terms and target_hits == 0:
        return 0.0
    if relation_terms and relation_hits == 0:
        return 0.0
    if not target_terms and len({term[:5] for term in relation_terms}) >= 2 and relation_hits < 2:
        return 0.0
    return target_hits * 4.0 + relation_hits * 3.0 + 1.0


def _bind_frame_conditions(records: dict[str, Any], frame: QueryFrame, expected: ExpectedAnswer, target_terms: list[str], relation_terms: list[str]) -> list[tuple[float, str, Evidence, str]]:
    answer_slot_terms = _answer_slot_terms(frame)
    args_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arg in records.get("frame_arguments", []):
        args_by_frame[str(arg.get("frame_id"))].append(arg)
    drs_relations_by_frame_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for relation in records.get("relations", []):
        if str(relation.get("relation_type") or "") != "drs_condition":
            continue
        key = (
            str(relation.get("source_span_id") or ""),
            normalize(str(relation.get("predicate") or "")),
            str(relation.get("context_id") or ""),
        )
        drs_relations_by_frame_key[key].append(relation)
    frame_types_by_span_predicate: dict[tuple[str, str], list[str]] = defaultdict(list)
    for relation in records.get("relations", []):
        if str(relation.get("relation_type") or "") != "semantic_frame":
            continue
        key = (str(relation.get("source_span_id") or ""), normalize(str(relation.get("predicate") or "")))
        frame_type = str(relation.get("subject") or "")
        if frame_type:
            frame_types_by_span_predicate[key].append(frame_type)
    candidates: list[tuple[float, str, Evidence, str]] = []
    for row in records.get("frames", []):
        frame_scope_relations = drs_relations_by_frame_key.get(
            (
                str(row.get("span_id") or ""),
                normalize(str(row.get("predicate") or "")),
                str(row.get("context_id") or ""),
            ),
            [],
        )
        if frame_scope_relations:
            if not any(_relation_scope_accessible(relation, records, frame) for relation in frame_scope_relations):
                continue
        elif not _context_accessible(str(row.get("context_id") or ""), records, frame):
            continue
        evidence = _evidence_for_span(str(row.get("span_id") or ""), records)
        if _source_is_low_priority(evidence.rel_path, evidence.text) and not _structured_source_row(row):
            continue
        arg_text = " ".join(str(arg.get("surface") or "") for arg in args_by_frame.get(str(row.get("frame_id")), []))
        frame_type_material = " ".join(
            frame_types_by_span_predicate.get(
                (str(row.get("span_id") or ""), normalize(str(row.get("predicate") or ""))),
                [],
            )
        )
        local_material = normalize(
            " ".join(
                [
                    frame_type_material,
                    str(row.get("predicate") or ""),
                    str(row.get("trigger_surface") or ""),
                    arg_text,
                ]
            )
        )
        score = _match_score(local_material, target_terms, relation_terms)
        if score <= 0:
            continue
        for value in _answer_values_from_frame(
            row,
            args_by_frame.get(str(row.get("frame_id")), []),
            expected,
            target_terms,
            relation_terms,
            answer_slot_terms,
            frame_type_material,
        ):
            candidates.append((score, value, evidence, "frame_argument_binding"))
    return candidates


def _bind_relation_conditions(records: dict[str, Any], frame: QueryFrame, expected: ExpectedAnswer, target_terms: list[str], relation_terms: list[str]) -> list[tuple[float, str, Evidence, str]]:
    answer_slot_terms = _answer_slot_terms(frame)
    candidates: list[tuple[float, str, Evidence, str]] = []
    for row in records.get("relations", []):
        if not _relation_scope_accessible(row, records, frame):
            continue
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        if _source_is_low_priority(evidence.rel_path, evidence.text) and not _structured_source_row(row):
            continue
        row_material = _relation_local_material(row, evidence, include_evidence=False, include_context=True, records=records)
        evidence_material = normalize(" ".join([row_material, evidence.rel_path, evidence.text]))
        score = _split_match_score(evidence_material, row_material, target_terms, relation_terms)
        if score <= 0:
            continue
        for value in _answer_values_from_relation(row, evidence, expected, target_terms, relation_terms, answer_slot_terms):
            candidates.append((score * float(row.get("confidence") or 0.7), value, evidence, "relation_condition_binding"))
    return candidates


def _record_groups(records: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records.get("relations", []):
        metadata = _relation_metadata(row)
        group = str(metadata.get("record_group") or "")
        if not group:
            continue
        if group.startswith("section_group_"):
            source_scope = str(row.get("document_id") or metadata.get("document_id") or "")
        else:
            source_scope = str(metadata.get("sentence_group") or row.get("source_span_id") or "")
        groups["|".join([source_scope, group]) if source_scope else group].append(row)
    return groups


def _group_material(rows: list[dict[str, Any]], records: dict[str, Any]) -> str:
    parts: list[str] = []
    for row in rows:
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        parts.append(_relation_local_material(row, evidence, include_evidence=True))
    return normalize(" ".join(parts))


def _bind_record_groups(
    records: dict[str, Any],
    frame: QueryFrame,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
) -> list[tuple[float, str, Evidence, str]]:
    """Bind answer variables inside one source-grounded record group.

    This is a generic DRS operation: a group is a bounded source context created
    from an object, table row, sentence group, section, or model frame.  Target
    anchors and requested relation terms must both be satisfied inside that
    group before a value can be returned.  No relation label is privileged; keys
    and predicates are treated as data.
    """

    candidates: list[tuple[float, str, Evidence, str]] = []
    answer_slot_terms = _answer_slot_terms(frame)
    for _group_id, rows in _record_groups(records).items():
        if not rows:
            continue
        group_material = _group_material(rows, records)
        if target_terms and not _contains_any(group_material, target_terms):
            continue
        group_relation_hits = sum(1 for term in relation_terms if _has_term(group_material, term))
        if relation_terms and group_relation_hits == 0:
            continue
        target_hits = sum(1 for term in target_terms if _has_term(group_material, term))
        for row in rows:
            if not _context_accessible(str(row.get("context_id") or ""), records, frame):
                continue
            evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
            if _source_is_low_priority(evidence.rel_path, evidence.text) and not _structured_source_row(row):
                continue
            local_material = _relation_local_material(row, evidence, include_evidence=False)
            relation_hits = sum(1 for term in relation_terms if _has_term(local_material, term))
            if relation_terms and relation_hits == 0:
                continue
            values = _answer_values_from_relation(row, evidence, expected, target_terms, relation_terms, answer_slot_terms)
            for value in values:
                value_hits = sum(1 for term in relation_terms if _has_term(normalize(value), term))
                score = 5.0 + target_hits * 5.0 + relation_hits * 6.0 + group_relation_hits * 1.5
                score += value_hits * 4.0
                score *= float(row.get("confidence") or 0.7)
                candidates.append((score, value, evidence, "record_group_drs_binding"))
    return candidates


def _relation_term_groups_for_frame(frame: QueryFrame) -> list[list[str]]:
    groups: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    raw_items = [*frame.relation_terms, *list(frame.constraints), *_query_terms(frame.requested_relation)]
    for item in raw_items:
        if frame.aggregation == "count" and normalize(item) in COUNT_AGGREGATION_SKIP_TERMS:
            continue
        variants = _compound_term_variants(item)
        if not variants:
            variants = [normalize(item)]
        variants = list(dict.fromkeys(variant for variant in expand_terms(variants) if variant))
        key = tuple(sorted(variants))
        if variants and key not in seen:
            groups.append(variants)
            seen.add(key)
    return groups


def _material_matches_all_term_groups(material: str, groups: list[list[str]]) -> bool:
    return all(any(_has_term(material, term) for term in group) for group in groups)


def _frame_requests_row_units(frame: QueryFrame) -> bool:
    material = normalize(
        " ".join(
            [
                frame.question_text,
                frame.requested_relation,
                *frame.answer_variables,
                *frame.relation_terms,
            ]
        )
    )
    return bool(re.search(r"\brows?\b", material))


def _rows_are_table_like(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if str(row.get("relation_type") or "") == "table_cell":
            return True
        metadata = _relation_metadata(row)
        if str(metadata.get("surface_format") or "") == "delimited_table":
            return True
        if metadata.get("column_header") or metadata.get("cell_index") is not None:
            return True
    return False


def _count_matching_record_groups(
    records: dict[str, Any],
    frame: QueryFrame,
    target_terms: list[str],
    relation_terms: list[str],
) -> tuple[int, list[Evidence]]:
    groups = _record_groups(records)
    required_relation_groups = _relation_term_groups_for_frame(frame)
    require_table_row = _frame_requests_row_units(frame)
    matched: list[tuple[str, Evidence]] = []
    for group_id, rows in groups.items():
        accessible_rows = [
            row for row in rows if _context_accessible(str(row.get("context_id") or ""), records, frame)
        ]
        if not accessible_rows:
            continue
        rows_by_span: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in accessible_rows:
            rows_by_span[str(row.get("source_span_id") or "")].append(row)
        for span_id, span_rows in rows_by_span.items():
            if not span_rows:
                continue
            if require_table_row and not _rows_are_table_like(span_rows):
                continue
            evidence = _evidence_for_span(span_id, records)
            if _source_is_low_priority(evidence.rel_path, evidence.text) and not any(_structured_source_row(row) for row in span_rows):
                continue
            span_material = _group_material(span_rows, records)
            if target_terms and not _contains_any(span_material, target_terms):
                continue
            if required_relation_groups and not _material_matches_all_term_groups(span_material, required_relation_groups):
                continue
            provenance_key = span_id or group_id
            matched.append((provenance_key, evidence))
    unique: dict[str, Evidence] = {}
    for group_id, evidence in matched:
        unique.setdefault(group_id, evidence)
    return len(unique), list(unique.values())[:4]


def _bind_metadata(records: dict[str, Any], question: str, expected: ExpectedAnswer, target_terms: list[str], relation_terms: list[str]) -> list[tuple[float, str, Evidence, str]]:
    if not expected.allow_metadata_evidence and expected.answer_type != "unknown":
        return []
    docs = _docs_by_id(records)
    candidates: list[tuple[float, str, Evidence, str]] = []
    for row in records.get("metadata_records", []):
        doc = docs.get(str(row.get("document_id")), {})
        key_material = normalize(str(row.get("key") or ""))
        if expected.answer_type == "unknown" and relation_terms and not _contains_any(key_material, relation_terms):
            continue
        material = normalize(" ".join([str(doc.get("rel_path") or ""), str(row.get("key") or ""), str(row.get("value") or "")]))
        score = _match_score(material, target_terms, relation_terms or _query_terms(question))
        if score <= 0:
            continue
        value = canonicalize_answer(expected, str(row.get("value") or ""))
        if value:
            candidates.append((score, value, _metadata_evidence(row, records), "metadata_binding"))
    return candidates


def _bind_contexts(records: dict[str, Any], frame: QueryFrame, expected: ExpectedAnswer, target_terms: list[str], relation_terms: list[str]) -> list[tuple[float, str, Evidence, str]]:
    if expected.answer_type not in {"content_phrase", "state", "metadata_value"}:
        return []
    if not any(term.startswith("context") for term in relation_terms):
        return []
    if "context" not in normalize(frame.requested_relation) and not any(term.startswith("context") for term in relation_terms):
        return []
    contexts = _contexts_by_id(records)
    candidates: list[tuple[float, str, Evidence, str]] = []
    for carrier in records.get("context_carriers", []):
        context = contexts.get(str(carrier.get("context_id")), {})
        kind = str(context.get("kind") or "")
        span_id = str(carrier.get("source_span_id") or "")
        evidence = _evidence_for_span(span_id, records) if span_id else Evidence(str(carrier.get("document_id") or ""), str(carrier.get("carrier_surface") or ""), 0.6)
        material = normalize(" ".join([kind, str(carrier.get("carrier_kind") or ""), str(carrier.get("carrier_surface") or ""), evidence.text]))
        score = _match_score(material, target_terms, relation_terms)
        if score <= 0:
            continue
        value = kind.split(":", 1)[-1] if ":" in kind else kind
        value = canonicalize_answer(expected, value) or clean_extracted_value(value)
        if value:
            candidates.append((score, value, evidence, "context_accessibility_binding"))
    return candidates


def _temporal_candidates(records: dict[str, Any], frame: QueryFrame, expected: ExpectedAnswer, target_terms: list[str], relation_terms: list[str]) -> list[tuple[float, str, Evidence, str]]:
    if frame.temporal_scope not in {"latest", "earliest"} and expected.answer_type not in {"state", "date_time"}:
        return []
    referents = _referents_by_id(records)
    rows: list[tuple[str, dict[str, Any], Evidence]] = []
    for row in records.get("temporal_edges", []):
        if not _context_accessible(str(row.get("context_id") or ""), records, frame):
            continue
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        referent = referents.get(str(row.get("referent_id") or ""), {})
        material = normalize(
            " ".join(
                [
                    str(referent.get("canonical_label") or referent.get("canonical_label_norm") or ""),
                    str(row.get("relation") or ""),
                    str(row.get("temporal_value") or ""),
                    str(row.get("state_value") or ""),
                    evidence.text,
                ]
            )
        )
        if target_terms and not _contains_any(material, target_terms):
            continue
        if relation_terms and not _contains_any(material, relation_terms):
            continue
        rows.append((str(row.get("temporal_value") or ""), row, evidence))
    rows.sort(key=lambda item: item[0], reverse=frame.temporal_scope != "earliest")
    candidates: list[tuple[float, str, Evidence, str]] = []
    selected_rows = rows[:3]
    if frame.temporal_scope in {"latest", "earliest"} and rows:
        boundary_value = rows[0][0]
        selected_rows = [item for item in rows if item[0] == boundary_value]
    for _time_value, row, evidence in selected_rows:
        state_value = str(row.get("state_value") or "")
        temporal_value = str(row.get("temporal_value") or "")
        raw_values = [state_value] if state_value else []
        if expected.answer_type == "date_time" or not raw_values:
            raw_values.append(temporal_value)
        for value in _compatible_values(expected, raw_values):
            candidates.append((8.0, value, evidence, "temporal_binding"))
    return candidates


def _temporal_relation_candidates(
    records: dict[str, Any],
    frame: QueryFrame,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
) -> list[tuple[float, str, Evidence, str]]:
    if frame.temporal_scope not in {"latest", "earliest"}:
        return []
    if expected.answer_type not in {"state", "date_time", "content_phrase", "unknown"}:
        return []
    answer_slot_terms = _answer_slot_terms(frame)
    rows_by_span: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records.get("relations", []):
        rows_by_span[str(row.get("source_span_id") or "")].append(row)
    ordered: list[tuple[str, str, list[dict[str, Any]], Evidence]] = []
    for span_id, rows in rows_by_span.items():
        if not span_id:
            continue
        accessible_rows = [row for row in rows if _relation_scope_accessible(row, records, frame)]
        if not accessible_rows:
            continue
        temporal_values = [
            str(row.get("value") or "")
            for row in accessible_rows
            if str(row.get("relation_type") or "") == "temporal" or normalize(str(row.get("predicate") or "")) == "timestamp"
        ]
        temporal_values = [value for value in temporal_values if DATE_TIME_RE.search(value)]
        if not temporal_values:
            continue
        evidence = _evidence_for_span(span_id, records)
        if _source_is_low_priority(evidence.rel_path, evidence.text):
            continue
        material = _group_material(accessible_rows, records)
        if target_terms and not _contains_any(material, target_terms):
            continue
        if relation_terms and not _contains_any(material, relation_terms):
            continue
        ordered.append((max(temporal_values), span_id, accessible_rows, evidence))
    ordered.sort(key=lambda item: item[0], reverse=frame.temporal_scope != "earliest")
    candidates: list[tuple[float, str, Evidence, str]] = []
    selected_rows = ordered[:1]
    if ordered:
        boundary_value = ordered[0][0]
        selected_rows = [item for item in ordered if item[0] == boundary_value]
    for _time_value, _span_id, rows, evidence in selected_rows:
        for row in rows:
            if str(row.get("relation_type") or "") == "temporal":
                continue
            if not _context_accessible(str(row.get("context_id") or ""), records, frame):
                continue
            for value in _answer_values_from_relation(row, evidence, expected, target_terms, relation_terms, answer_slot_terms):
                candidates.append((9.0 * float(row.get("confidence") or 0.7), value, evidence, "temporal_relation_binding"))
    return candidates


def _choice_score(score: float, reason: str, expected: ExpectedAnswer) -> float:
    if reason == "frame_argument_binding":
        score += 3.0
    if reason == "relation_condition_binding" and expected.answer_type in {"content_phrase", "unknown"}:
        score += 7.0
    return score


def _choose_answer(candidates: list[tuple[float, str, Evidence, str]], expected: ExpectedAnswer) -> Answer | None:
    scored: dict[str, tuple[float, list[Evidence], str]] = {}
    for score, value, evidence, reason in candidates:
        canonical = canonicalize_answer(expected, value)
        if not canonical:
            continue
        score = _choice_score(score, reason, expected)
        previous = scored.get(canonical)
        if previous is None:
            scored[canonical] = (score, [evidence], reason)
        else:
            scored[canonical] = (previous[0] + score, [*previous[1], evidence][:4], previous[2])
    if not scored:
        return None
    ordered = sorted(scored.items(), key=lambda item: (-item[1][0], len(item[0]), item[0]))
    value, (score, evidence, reason) = ordered[0]
    return Answer(value, min(0.95, max(0.0, score / 10.0)), evidence, reason, expected.answer_type)


def _with_supporting_evidence(answer: Answer | None, supporting_evidence: list[Evidence]) -> Answer | None:
    if answer is None or not supporting_evidence:
        return answer
    answer.evidence = _dedupe_evidence([*answer.evidence, *supporting_evidence])
    return answer


def _answer_conflict_diagnostics(
    candidates: list[tuple[float, str, Evidence, str]],
    expected: ExpectedAnswer,
    target_terms: list[str],
    records: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    buckets: dict[str, dict[str, Any]] = {}
    for score, value, evidence, reason in candidates:
        canonical = canonicalize_answer(expected, value)
        if not canonical:
            continue
        bucket = buckets.setdefault(canonical, {"score": 0.0, "evidence": [], "reasons": set()})
        bucket["score"] = float(bucket["score"]) + _choice_score(score, reason, expected)
        bucket["reasons"].add(reason)
        if len(bucket["evidence"]) < 4:
            bucket["evidence"].append(evidence)
    if len(buckets) < 2:
        return None
    ordered = sorted(buckets.items(), key=lambda item: (-float(item[1]["score"]), len(item[0]), item[0]))
    top_value, top_bucket = ordered[0]
    next_value, next_bucket = ordered[1]
    top_score = float(top_bucket["score"])
    next_score = float(next_bucket["score"])
    if top_score <= 0.0 or next_score < top_score * 0.85:
        return None

    def target_coverage(bucket: dict[str, Any]) -> int:
        if not target_terms:
            return 0
        return max(
            (
                sum(
                    1
                    for term in target_terms
                    if _has_term(normalize(" ".join([item.rel_path, item.text])), term)
                )
                for item in bucket["evidence"]
                if isinstance(item, Evidence)
            ),
            default=0,
        )

    if target_coverage(top_bucket) > target_coverage(next_bucket):
        return None

    def evidence_keys(bucket: dict[str, Any]) -> set[tuple[str, str, int | None, str]]:
        return {
            (item.rel_path, item.span_id, item.chunk_order, item.text)
            for item in bucket["evidence"]
            if isinstance(item, Evidence)
        }

    if evidence_keys(top_bucket) & evidence_keys(next_bucket):
        return None

    values = []
    for value, bucket in ordered[:4]:
        values.append(
            {
                "value": value,
                "score": round(float(bucket["score"]), 3),
                "reasons": sorted(bucket["reasons"]),
                "evidence": [
                    _evidence_provenance_payload(item, records) if records is not None else _evidence_payload(item)
                    for item in bucket["evidence"]
                ],
            }
        )
    return {
        "top_value": top_value,
        "next_value": next_value,
        "top_score": round(top_score, 3),
        "next_score": round(next_score, 3),
        "values": values,
    }


def _choose_list_answer(candidates: list[tuple[float, str, Evidence, str]], expected: ExpectedAnswer) -> Answer | None:
    values: list[str] = []
    evidence: list[Evidence] = []
    for _score, value, item_evidence, _reason in candidates:
        canonical = canonicalize_answer(expected, value)
        if not canonical:
            continue
        parts = [part.strip() for part in canonical.split(";") if part.strip()]
        for part in parts or [canonical]:
            if part not in values:
                values.append(part)
                evidence.append(item_evidence)
    if not values:
        return None
    return Answer("; ".join(values), 0.86, evidence[:6], "list aggregation DRS binding", expected.answer_type)


def _has_unscoped_temporal_ambiguity(candidates: list[tuple[float, str, Evidence, str]]) -> bool:
    temporal_candidate_values = {
        normalize(value)
        for _score, value, _evidence, reason in candidates
        if reason in {"temporal_binding", "temporal_relation_binding"} and normalize(value)
    }
    if len(temporal_candidate_values) > 1:
        return True
    values_by_time: dict[str, set[str]] = defaultdict(set)
    for _score, value, evidence, _reason in candidates:
        match = DATE_TIME_RE.search(evidence.text)
        if match and value:
            values_by_time[match.group(0)].add(normalize(value))
    if len(values_by_time) < 2:
        return False
    distinct_values = {value for values in values_by_time.values() for value in values if value}
    return len(distinct_values) > 1


def execute_bounded_query(
    store: Any,
    run_id: str,
    documents: list[Document],
    sentences_by_document: dict[str, dict[int, Sentence]],
    question: str,
    plan: dict[str, Any] | QueryFrame | None = None,
    *,
    doc_limit: int = 40,
    chunk_limit: int = 160,
) -> tuple[Answer | None, dict[str, Any]]:
    frame = _frame(plan, question)
    expected = _expected_from_frame(frame)
    target_terms = _target_terms(frame, question)
    relation_terms = _relation_terms(frame, question)
    answer_slot_terms = _answer_slot_terms(frame)
    if answer_slot_terms:
        relation_terms = list(dict.fromkeys([*relation_terms, *answer_slot_terms]))
    selected_docs, selected_chunks, ranking = _rank_scope(documents, sentences_by_document, question, frame, doc_limit, chunk_limit)
    current_document_chunk_ids = _current_chunk_ids_for_documents(documents, sentences_by_document, selected_docs)
    records = _load_records(
        store,
        run_id,
        selected_docs,
        selected_chunks,
        current_document_chunk_ids=current_document_chunk_ids,
    )
    identity_expanded_terms: list[str] = []
    identity_expansion_evidence: list[Evidence] = []
    identity_expansion_provenance: list[dict[str, Any]] = []
    identity_expansion_rounds = 0
    for _round in range(IDENTITY_RERANK_MAX_ROUNDS):
        identity_terms, identity_evidence = _identity_expansion(records, target_terms, frame)
        identity_expansion_evidence = _dedupe_evidence([*identity_expansion_evidence, *identity_evidence])
        identity_expansion_provenance = _dedupe_provenance_payloads(
            [
                *identity_expansion_provenance,
                *[_evidence_provenance_payload(item, records) for item in identity_evidence],
            ]
        )
        new_identity_terms = [term for term in identity_terms if term and term not in target_terms]
        if not new_identity_terms:
            break
        identity_expansion_rounds += 1
        identity_expanded_terms = list(dict.fromkeys([*identity_expanded_terms, *new_identity_terms]))
        target_terms = list(dict.fromkeys([*target_terms, *new_identity_terms]))
        ranking["identity_expanded_target_terms"] = identity_expanded_terms[:32]
        expanded_frame = replace(
            frame,
            target_anchors=tuple(dict.fromkeys([*frame.target_anchors, *target_terms])),
        )
        expanded_docs, expanded_chunks, expanded_ranking = _rank_scope(
            documents,
            sentences_by_document,
            question,
            expanded_frame,
            doc_limit,
            chunk_limit,
        )
        merged_docs = list(dict.fromkeys([*expanded_docs, *selected_docs]))
        merged_chunks = list(dict.fromkeys([*expanded_chunks, *selected_chunks]))
        next_docs = merged_docs[:doc_limit]
        next_chunks = merged_chunks[:chunk_limit]
        ranking.update(
            {
                "identity_expansion_rounds": identity_expansion_rounds,
                "identity_reranked_candidate_document_rows": expanded_ranking.get("candidate_document_rows", 0),
                "identity_reranked_selected_document_count": len(next_docs),
                "identity_reranked_candidate_chunk_rows": expanded_ranking.get("candidate_chunk_rows", 0),
                "identity_reranked_selected_chunk_count": len(next_chunks),
            }
        )
        if next_docs == selected_docs and next_chunks == selected_chunks:
            break
        selected_docs = next_docs
        selected_chunks = next_chunks
        current_document_chunk_ids = _current_chunk_ids_for_documents(documents, sentences_by_document, selected_docs)
        records = _load_records(
            store,
            run_id,
            selected_docs,
            selected_chunks,
            current_document_chunk_ids=current_document_chunk_ids,
        )
    diagnostics = {"ranking": ranking, "execution": {"record_counts": records["record_counts"], "query_frame": frame.as_dict()}}
    if identity_expansion_provenance:
        diagnostics["execution"]["identity_expansion_evidence"] = identity_expansion_provenance

    if expected.answer_type == "boolean":
        _attach_no_answer_provenance(
            diagnostics,
            records,
            target_terms,
            relation_terms,
            [],
            expected,
            "boolean_not_bound_by_bounded_executor",
        )
        return None, diagnostics

    candidates: list[tuple[float, str, Evidence, str]] = []
    candidates.extend(_bind_record_groups(records, frame, expected, target_terms, relation_terms))
    candidates.extend(_bind_frame_conditions(records, frame, expected, target_terms, relation_terms))
    candidates.extend(_bind_relation_conditions(records, frame, expected, target_terms, relation_terms))
    temporal_candidates = _temporal_candidates(records, frame, expected, target_terms, relation_terms)
    temporal_candidates.extend(_temporal_relation_candidates(records, frame, expected, target_terms, relation_terms))
    if temporal_candidates and frame.temporal_scope in {"latest", "earliest"}:
        conflict = _answer_conflict_diagnostics(temporal_candidates, expected, target_terms, records)
        if conflict:
            diagnostics["execution"]["temporal_answer_conflict_at_boundary"] = conflict
            _attach_no_answer_provenance(
                diagnostics,
                records,
                target_terms,
                relation_terms,
                temporal_candidates,
                expected,
                "temporal_answer_conflict_at_boundary",
            )
            return None, diagnostics
        answer = _with_supporting_evidence(_choose_answer(temporal_candidates, expected), identity_expansion_evidence)
        if answer is None:
            _attach_no_answer_provenance(
                diagnostics,
                records,
                target_terms,
                relation_terms,
                temporal_candidates,
                expected,
                "no_compatible_temporal_candidate",
            )
        else:
            _attach_answer_provenance(diagnostics, records, answer)
        return answer, diagnostics
    candidates.extend(temporal_candidates)
    candidates.extend(_bind_metadata(records, question, expected, target_terms, relation_terms))
    candidates.extend(_bind_contexts(records, frame, expected, target_terms, relation_terms))

    if expected.answer_type == "count" and frame.aggregation == "count":
        group_count, group_evidence = _count_matching_record_groups(records, frame, target_terms, relation_terms)
        if group_count:
            answer = Answer(str(group_count), 0.86, group_evidence, "record-group aggregation DRS binding", "count")
            answer = _with_supporting_evidence(answer, identity_expansion_evidence)
            _attach_answer_provenance(diagnostics, records, answer)
            return answer, diagnostics
    if expected.answer_type == "count" and frame.aggregation == "count" and candidates:
        values = sorted({canonicalize_answer(expected, value) or value for _score, value, _evidence, _reason in candidates})
        evidence = [item[2] for item in candidates[:4]]
        answer = Answer(str(len(values)), 0.85, evidence, "aggregation DRS binding", "count")
        answer = _with_supporting_evidence(answer, identity_expansion_evidence)
        _attach_answer_provenance(diagnostics, records, answer)
        return answer, diagnostics

    if not frame.temporal_scope and _has_unscoped_temporal_ambiguity(candidates):
        diagnostics["execution"]["temporal_ambiguity_without_query_scope"] = True
        _attach_no_answer_provenance(
            diagnostics,
            records,
            target_terms,
            relation_terms,
            candidates,
            expected,
            "temporal_ambiguity_without_query_scope",
        )
        return None, diagnostics

    if frame.aggregation in {"list", "set"}:
        answer = _with_supporting_evidence(_choose_list_answer(candidates, expected), identity_expansion_evidence)
        if answer is None:
            _attach_no_answer_provenance(
                diagnostics,
                records,
                target_terms,
                relation_terms,
                candidates,
                expected,
                "no_compatible_list_candidate",
            )
        else:
            _attach_answer_provenance(diagnostics, records, answer)
        return answer, diagnostics
    if not frame.temporal_scope and expected.answer_type != "count":
        conflict = _answer_conflict_diagnostics(candidates, expected, target_terms, records)
        if conflict:
            diagnostics["execution"]["answer_conflict_without_query_scope"] = conflict
            _attach_no_answer_provenance(
                diagnostics,
                records,
                target_terms,
                relation_terms,
                candidates,
                expected,
                "answer_conflict_without_query_scope",
            )
            return None, diagnostics

    answer = _with_supporting_evidence(_choose_answer(candidates, expected), identity_expansion_evidence)
    if answer is None:
        _attach_no_answer_provenance(
            diagnostics,
            records,
            target_terms,
            relation_terms,
            candidates,
            expected,
            "no_compatible_candidate" if candidates else "no_candidate",
        )
    else:
        _attach_answer_provenance(diagnostics, records, answer)
    return answer, diagnostics
