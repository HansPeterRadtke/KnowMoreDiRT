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
from .extractors import capitalized_phrases, identifiers, urls
from .models import Answer, Document, Evidence, Sentence
from .query import QueryFrame, expand_terms, frame_from_mapping, normalize_temporal_scope, plan_question, term_variants, visible_anchors
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
STRUCTURAL_CHAIN_GENERIC_TERMS = {
    *ANSWER_SLOT_SKIP_TERMS,
    "a",
    "an",
    "and",
    "argument",
    "are",
    "be",
    "belongs",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "listed",
    "of",
    "the",
    "to",
    "was",
    "were",
    "with",
}
STRUCTURAL_CHAIN_SOURCE_ARG_ROLES = {
    "agent",
    "entity",
    "holder",
    "item",
    "key",
    "name",
    "record",
    "source",
    "subject",
    "topic",
}
STRUCTURAL_CHAIN_TARGET_ARG_ROLES = {
    "content",
    "destination",
    "identifier",
    "location",
    "object",
    "patient",
    "result",
    "state",
    "target",
    "theme",
    "value",
}
MODEL_NEGATION_TOKENS = {
    "cannot",
    "cant",
    "neither",
    "never",
    "no",
    "none",
    "not",
    "without",
}
BOOLEAN_GENERIC_TERMS = {
    "be",
    "been",
    "being",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "really",
    "should",
    "was",
    "were",
    "would",
}
RELATION_TERM_SKIP_TERMS = {"answer", "argument", "what", "which", "who", "why"}


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
    visible = {normalize(anchor) for anchor in visible_anchors(question)}
    question_material = normalize(question)
    answer_material = normalize(" ".join(frame.answer_variables))
    answer_tokens = _normalized_token_set(answer_material)
    for anchor in frame.target_anchors:
        norm = normalize(anchor)
        if not norm:
            continue
        anchor_visible = norm in visible or norm in question_material
        anchor_tokens = _normalized_token_set(norm)
        relation_material = normalize(" ".join([frame.requested_relation, *frame.relation_terms, *frame.constraints]))
        if anchor_tokens and anchor_tokens.issubset(answer_tokens) and not anchor_visible:
            field_words = {
                "state", "status", "code", "id", "identifier", "url", "link",
                "date", "time", "model", "confirmation",
                "temperature", "location", "where",
            }
            remainder = answer_tokens - anchor_tokens
            # Drop true answer-slot anchors, but keep real target entities when
            # the answer variable is target+field, e.g. target="greenhouse pump"
            # and answer="greenhouse pump state".
            if not (remainder & field_words):
                continue
        if anchor_tokens and len(anchor_tokens) == 1 and _has_term(relation_material, norm) and not anchor_visible:
            # Model query DRS can put a requested relation/slot such as
            # "feedback" into target_anchors.  That is not an entity target and
            # should stay available through relation terms instead.
            continue
        if frame.aggregation == "count" and anchor_tokens and not anchor_visible:
            relation_group_tokens: set[str] = set()
            for group in _relation_term_groups_for_frame(frame, target_terms=[]):
                for term in group:
                    relation_group_tokens.update(_normalized_token_set(term))
            if anchor_tokens.issubset(relation_group_tokens):
                continue
        values.append(norm)
        if " " in norm:
            values.append(norm.replace(" ", "_"))
            values.append(norm.replace(" ", "-"))
    return list(dict.fromkeys(values))


def _target_token_variants(target_terms: list[str] | None) -> set[str]:
    variants: set[str] = set()
    for term in target_terms or []:
        for token in content_tokens(term):
            variants.update(expand_terms([token]))
    return variants


def _term_covered_by_target_tokens(term: str, target_terms: list[str] | None) -> bool:
    tokens = [token for token in content_tokens(term) if token]
    if not tokens:
        return False
    target_tokens = _target_token_variants(target_terms)
    if not target_tokens:
        return False
    return all(any(variant in target_tokens for variant in expand_terms([token])) for token in tokens)


def _term_has_answer_wrapper(term: str) -> bool:
    return any(part in ANSWER_SLOT_SKIP_TERMS for part in _material_parts(normalize(term)))


def _skip_relation_term(term: str) -> bool:
    parts = _material_parts(normalize(term))
    if not parts:
        return True
    if len(parts) == 1 and parts[0] in RELATION_TERM_SKIP_TERMS:
        return True
    return any(part in RELATION_TERM_SKIP_TERMS for part in parts) and len(parts) > 1


def _relation_terms(frame: QueryFrame, question: str) -> list[str]:
    target_terms = _target_terms(frame, question)
    target = set(target_terms)
    raw_terms = list(frame.relation_terms) + _query_terms(frame.requested_relation) + list(frame.constraints)
    terms = [variant for term in raw_terms for variant in _compound_term_variants(term)]
    filtered = [
        term
        for term in terms
        if term
        and term not in target
        and not _skip_relation_term(term)
        and not _term_covered_by_target_tokens(term, target_terms)
        and normalize_temporal_scope(term) not in {"latest", "earliest"}
    ]
    return list(
        dict.fromkeys(
            term
            for term in expand_terms(filtered)
            if term
            and term not in target
            and not _skip_relation_term(term)
            and not _term_covered_by_target_tokens(term, target_terms)
        )
    )


def _answer_slot_terms(frame: QueryFrame, target_terms: list[str] | None = None) -> list[str]:
    terms: list[str] = []
    target_tokens = _target_token_variants(target_terms)
    requested_tokens = set(content_tokens(frame.requested_relation))
    for variable in frame.answer_variables:
        reduced_tokens: list[str] = []
        for token in content_tokens(variable):
            if token in ANSWER_SLOT_SKIP_TERMS:
                continue
            if token in requested_tokens:
                continue
            if target_tokens and any(variant in target_tokens for variant in expand_terms([token])):
                continue
            if token not in reduced_tokens:
                reduced_tokens.append(token)
        if len(reduced_tokens) > 1:
            terms.append(" ".join(reduced_tokens))
        elif len(reduced_tokens) == 1:
            terms.append(reduced_tokens[0])
        for term in _compound_term_variants(variable):
            if term in ANSWER_SLOT_SKIP_TERMS:
                continue
            if _term_has_answer_wrapper(term):
                continue
            if _term_covered_by_target_tokens(term, target_terms):
                continue
            term_tokens = [token for token in content_tokens(term) if token]
            if any(token in ANSWER_SLOT_SKIP_TERMS for token in term_tokens):
                continue
            if requested_tokens and all(token in requested_tokens for token in term_tokens):
                continue
            if term_tokens and all(token in reduced_tokens for token in term_tokens):
                continue
            terms.append(term)
    return list(dict.fromkeys(term for term in expand_terms(terms) if term))


def _answer_slot_constraints(
    answer_slot_terms: list[str],
    target_terms: list[str] | None = None,
) -> list[tuple[str, tuple[str, ...]]]:
    target_tokens = _target_token_variants(target_terms)
    constraints: list[tuple[str, tuple[str, ...]]] = []
    for term in answer_slot_terms:
        tokens: list[str] = []
        for token in content_tokens(term):
            if token in ANSWER_SLOT_SKIP_TERMS:
                continue
            token_variants = expand_terms([token])
            if target_tokens and any(variant in target_tokens for variant in token_variants):
                continue
            if token not in tokens:
                tokens.append(token)
        if 1 < len(tokens) <= 4:
            constraints.append((normalize(term), tuple(tokens)))
    return constraints


def _slot_token_matches(label_material: str, token: str) -> bool:
    return any(_has_term(label_material, variant) for variant in expand_terms([token]))


def _answer_slot_label_matches(
    label_material: str,
    answer_slot_terms: list[str],
    target_terms: list[str] | None = None,
) -> bool:
    material = normalize(label_material)
    if not material or not answer_slot_terms:
        return False
    constraints = _answer_slot_constraints(answer_slot_terms, target_terms)
    if constraints:
        material_tokens = _normalized_token_set(material)
        for term, tokens in constraints:
            if term and _has_term(material, term):
                return True
            if all(token in material_tokens or _slot_token_matches(material, token) for token in tokens):
                return True
        return False
    return _contains_any(material, answer_slot_terms)


def _count_answer_unit_tokens(frame: QueryFrame) -> set[str]:
    requested_tokens = set(content_tokens(frame.requested_relation))
    unit_tokens: list[str] = []
    for variable in frame.answer_variables:
        for token in content_tokens(variable):
            if token in requested_tokens:
                continue
            if token in ANSWER_SLOT_SKIP_TERMS or token in COUNT_AGGREGATION_SKIP_TERMS:
                continue
            unit_tokens.append(token)
    return set(expand_terms(unit_tokens))


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
    relation_terms = list(dict.fromkeys([*_relation_terms(frame, question), *_answer_slot_terms(frame, target_terms)]))
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
        relation_budget = doc_limit - len(selected_docs)
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
    drs_conditions = _fetch_by_ids(connection, "drs_conditions", "source_span_id", span_ids)
    drs_arguments = _fetch_by_ids(
        connection,
        "drs_condition_arguments",
        "drs_condition_id",
        [condition["drs_condition_id"] for condition in drs_conditions],
    )
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
                *[
                    str(row.get("referent_id") or "")
                    for row in drs_arguments
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
        "drs_conditions": drs_conditions,
        "drs_condition_arguments": drs_arguments,
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
            "drs_conditions": len(drs_conditions),
            "drs_condition_arguments": len(drs_arguments),
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


def _identity_labels_for_referent(
    records: dict[str, Any],
    referent_id: str,
    frame: QueryFrame | None = None,
) -> list[str]:
    if not referent_id:
        return []
    referents = _referents_by_id(records)
    if referent_id not in referents:
        return []
    visited = {referent_id}
    frontier = {referent_id}
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
            if right in frontier and left and left not in visited:
                next_frontier.add(left)
        if not next_frontier:
            break
        visited.update(next_frontier)
        frontier = next_frontier
    labels: list[str] = []
    for identity_ref in visited:
        if identity_ref == referent_id:
            continue
        label = str(referents.get(identity_ref, {}).get("canonical_label") or "")
        if label and label not in labels:
            labels.append(label)
    return labels


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
    execution = diagnostics.setdefault("execution", {})
    execution["answer_binding_reason"] = answer.reason
    execution["answer_binding_type"] = answer.answer_type
    provenance = _answer_source_provenance_sample(answer, records)
    if provenance:
        execution["answer_source_provenance"] = provenance


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



def _surface_format_alias_material(metadata: dict[str, Any]) -> str:
    surface_format = normalize(str(metadata.get("surface_format") or ""))
    if surface_format in {"object_like", "json_like", "json"}:
        return "raw json json-like object-like text"
    if surface_format == "delimited_table":
        return "table row rows entries records"
    return surface_format

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
        metadata.get("surface_format"),
        _surface_format_alias_material(metadata),
    ]
    if include_evidence and evidence is not None:
        fields.append(evidence.rel_path)
        fields.append(evidence.text)
    if include_context and records is not None:
        fields.append(_context_chain_material(str(row.get("context_id") or ""), records))
    return normalize(" ".join(str(item or "") for item in fields))


def _relation_selector_material(
    row: dict[str, Any],
    evidence: Evidence | None = None,
    *,
    include_evidence: bool = False,
) -> str:
    """Row-local material used for matching requested slots/relations.

    Section anchors and document context can establish source scope, but they
    should not make every inherited label/value row look like it satisfies the
    requested predicate.
    """

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
        metadata.get("argument_role"),
        metadata.get("argument_value_type"),
        metadata.get("surface_format"),
        _surface_format_alias_material(metadata),
    ]
    if include_evidence and evidence is not None:
        fields.append(evidence.text)
    return normalize(" ".join(str(item or "") for item in fields))


RELATION_BINDING_GENERIC_TERMS = {
    *ANSWER_SLOT_SKIP_TERMS,
    "answer",
    "argument",
    "did",
    "do",
    "does",
    "is",
    "are",
    "was",
    "were",
    "not",
    "no",
    "never",
    "negative",
    "positive",
}

UNRESOLVED_PRONOUN_ANSWER_VALUES = {
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "it",
    "its",
    "me",
    "mine",
    "my",
    "our",
    "ours",
    "she",
    "their",
    "theirs",
    "them",
    "they",
    "us",
    "we",
    "you",
    "your",
    "yours",
}


def _specific_relation_terms(relation_terms: list[str], target_terms: list[str]) -> list[str]:
    target_tokens = _target_token_variants(target_terms)
    values: list[str] = []
    for term in relation_terms:
        tokens = content_tokens(term)
        if len(tokens) != 1:
            continue
        token = tokens[0]
        if token in RELATION_BINDING_GENERIC_TERMS or token in target_tokens:
            continue
        for variant in expand_terms([token]):
            if variant and variant not in values:
                values.append(variant)
    return values


def _has_specific_relation_hit(
    material: str,
    relation_terms: list[str],
    target_terms: list[str],
) -> bool:
    specific_terms = _specific_relation_terms(relation_terms, target_terms)
    return not specific_terms or _contains_any(material, specific_terms)


def _drs_arguments_by_condition_id(records: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cached = records.get("_drs_arguments_by_condition_id")
    if isinstance(cached, dict):
        return cached
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arg in records.get("drs_condition_arguments", []):
        grouped[str(arg.get("drs_condition_id") or "")].append(arg)
    records["_drs_arguments_by_condition_id"] = grouped
    return grouped


def _drs_conditions_for_relation(row: dict[str, Any], records: dict[str, Any] | None) -> list[dict[str, Any]]:
    if records is None or str(row.get("relation_type") or "") != "drs_condition":
        return []
    metadata = _relation_metadata(row)
    external_condition_id = str(metadata.get("external_condition_id") or "")
    source_span_id = str(row.get("source_span_id") or "")
    context_id = str(row.get("context_id") or "")
    predicate_norm = normalize(str(row.get("predicate") or ""))
    polarity_norm = normalize(str(row.get("object") or ""))
    matches: list[dict[str, Any]] = []
    for condition in records.get("drs_conditions", []):
        if source_span_id and str(condition.get("source_span_id") or "") != source_span_id:
            continue
        if external_condition_id and str(condition.get("external_condition_id") or "") != external_condition_id:
            continue
        if context_id and str(condition.get("context_id") or "") != context_id:
            continue
        if predicate_norm and normalize(str(condition.get("predicate") or "")) != predicate_norm:
            continue
        if polarity_norm and normalize(str(condition.get("polarity") or "")) != polarity_norm:
            continue
        matches.append(condition)
    return matches


def _referent_surface_values(
    records: dict[str, Any],
    referent_id: str,
    frame: QueryFrame | None,
) -> list[str]:
    if not referent_id:
        return []
    referent = _referents_by_id(records).get(referent_id, {})
    values = [
        str(referent.get("canonical_label") or ""),
        *_identity_labels_for_referent(records, referent_id, frame),
    ]
    return list(dict.fromkeys(clean_extracted_value(value) for value in values if clean_extracted_value(value)))


def _drs_argument_surface_values(
    arg: dict[str, Any],
    records: dict[str, Any],
    frame: QueryFrame | None,
) -> list[str]:
    values = [
        str(arg.get("value") or ""),
        str(arg.get("evidence_surface") or ""),
        *_referent_surface_values(records, str(arg.get("referent_id") or ""), frame),
    ]
    return list(dict.fromkeys(clean_extracted_value(value) for value in values if clean_extracted_value(value)))


def _argument_values_match_target(values: list[str], target_terms: list[str]) -> bool:
    return any(
        _value_is_target(value, target_terms) or _value_contains_target(value, target_terms)
        for value in values
    )


def _drs_condition_is_negated(row: dict[str, Any], condition: dict[str, Any]) -> bool:
    polarity = normalize(str(condition.get("polarity") or row.get("object") or ""))
    if polarity == "negative":
        return True
    predicate_material = normalize(" ".join([str(row.get("predicate") or ""), str(condition.get("predicate") or "")]))
    return _has_term(predicate_material, "not")


def _frame_requests_negated_condition(frame: QueryFrame) -> bool:
    if frame.negated:
        return True
    material = normalize(" ".join([frame.requested_relation, *frame.constraints]))
    return _has_term(material, "not") or _has_term(material, "never") or _has_term(material, "no")


def _drs_condition_polarity_matches_query(
    row: dict[str, Any],
    records: dict[str, Any],
    frame: QueryFrame,
) -> bool:
    if str(row.get("relation_type") or "") != "drs_condition":
        return True
    conditions = _drs_conditions_for_relation(row, records)
    if not conditions:
        return True
    row_is_negated = any(_drs_condition_is_negated(row, condition) for condition in conditions)
    query_is_negated = _frame_requests_negated_condition(frame)
    return row_is_negated if query_is_negated else not row_is_negated


def _strip_condition_polarity_prefix(value: str, *, negated: bool) -> str:
    text = clean_extracted_value(value)
    if not negated:
        return text
    match = re.match(r"(?i)^\s*not[\s:_-]+(.+)$", text)
    return clean_extracted_value(match.group(1)) if match else text


def _drs_argument_role_material(arg: dict[str, Any]) -> str:
    return normalize(
        " ".join(
            [
                str(arg.get("role") or ""),
                str(arg.get("value_type") or ""),
                str(arg.get("target_kind") or ""),
            ]
        )
    )


def _drs_argument_matches_answer_slot(
    arg: dict[str, Any],
    answer_slot_terms: list[str] | None,
    target_terms: list[str],
) -> bool:
    if not answer_slot_terms:
        return False
    material = _drs_argument_role_material(arg)
    return _answer_slot_label_matches(material, answer_slot_terms, target_terms)


def _drs_argument_matches_expected_type(arg: dict[str, Any], expected: ExpectedAnswer) -> bool:
    expected_type = normalize(expected.answer_type)
    if expected_type in {"unknown", "content_phrase", "metadata_value"}:
        return False
    material = _drs_argument_role_material(arg)
    if _has_term(material, expected_type):
        return True
    if expected_type == "state" and (_has_term(material, "status") or _has_term(material, "condition")):
        return True
    if expected_type in {"person", "actor", "organization"}:
        return any(_has_term(material, term) for term in ["person", "actor", "agent", "speaker", "holder", "source"])
    return False


def _select_drs_answer_arguments(
    args: list[dict[str, Any]],
    expected: ExpectedAnswer,
    answer_slot_terms: list[str] | None,
    target_terms: list[str],
) -> list[dict[str, Any]]:
    slot_args = [
        arg for arg in args
        if _drs_argument_matches_answer_slot(arg, answer_slot_terms, target_terms)
    ]
    if slot_args:
        return slot_args
    if expected.answer_type in {"content_phrase", "unknown"}:
        return []
    expected_args = [arg for arg in args if _drs_argument_matches_expected_type(arg, expected)]
    if expected_args:
        return expected_args
    if expected.answer_type in {"person", "actor", "organization", "state"}:
        return []
    return args


def _drs_condition_argument_values(
    row: dict[str, Any],
    records: dict[str, Any] | None,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
    answer_slot_terms: list[str] | None,
    frame: QueryFrame | None,
) -> list[str]:
    conditions = _drs_conditions_for_relation(row, records)
    if not conditions or records is None:
        return []
    args_by_condition = _drs_arguments_by_condition_id(records)
    values: list[str] = []

    def append_surface_values(surface_values: list[str], *, negated: bool) -> int:
        before = len(values)
        for value in surface_values:
            if _value_is_target(value, relation_terms):
                continue
            stripped = _strip_condition_polarity_prefix(value, negated=negated)
            for candidate in [stripped, value]:
                for answer_value in _drs_argument_answer_values(
                    candidate,
                    expected,
                    target_terms,
                    relation_terms,
                ):
                    if answer_value and answer_value not in values:
                        values.append(answer_value)
        return len(values) - before

    for condition in conditions:
        negated = _drs_condition_is_negated(row, condition)
        raw_args = args_by_condition.get(str(condition.get("drs_condition_id") or ""), [])
        candidate_args = _select_drs_answer_arguments(raw_args, expected, answer_slot_terms, target_terms)
        condition_added = 0
        for arg in candidate_args:
            arg_values = _drs_argument_surface_values(arg, records, frame)
            if target_terms and _argument_values_match_target(arg_values, target_terms):
                continue
            condition_added += append_surface_values(arg_values, negated=negated)
        if condition_added:
            continue
        candidate_arg_ids = {id(arg) for arg in candidate_args}
        for arg in raw_args:
            if id(arg) in candidate_arg_ids:
                continue
            arg_values = _drs_argument_surface_values(arg, records, frame)
            if not _drs_argument_matches_requested_clause(arg_values, target_terms, relation_terms):
                continue
            append_surface_values(arg_values, negated=negated)
    return values


def _drs_argument_answer_values(
    value: str,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
) -> list[str]:
    text = clean_extracted_value(value)
    if not text:
        return []
    answer_type = expected.answer_type
    if answer_type in {"person", "actor", "organization"}:
        values: list[str] = []
        for phrase in capitalized_phrases(text):
            canonical = canonicalize_answer(expected, phrase)
            if not canonical:
                continue
            if normalize(canonical) in UNRESOLVED_PRONOUN_ANSWER_VALUES:
                continue
            if _value_is_target(canonical, target_terms) or _rejects_bound_target_value(expected, canonical, target_terms):
                continue
            if _value_is_target(canonical, relation_terms):
                continue
            if canonical not in values:
                values.append(canonical)
        return values
    canonical = canonicalize_answer(expected, text)
    if canonical:
        return [canonical]
    return [text] if answer_type in {"content_phrase", "metadata_value", "unknown"} else []


def _drs_argument_matches_requested_clause(
    values: list[str],
    target_terms: list[str],
    relation_terms: list[str],
) -> bool:
    if not values:
        return False
    if target_terms and not _argument_values_match_target(values, target_terms):
        return False
    material = normalize(" ".join(values))
    specific_terms = _specific_relation_terms(relation_terms, target_terms)
    return bool(specific_terms and _contains_any(material, specific_terms))


def _drs_condition_has_target_argument(
    row: dict[str, Any],
    records: dict[str, Any],
    target_terms: list[str],
    frame: QueryFrame,
) -> bool:
    if not target_terms or str(row.get("relation_type") or "") != "drs_condition":
        return True
    context_material = _context_chain_material(str(row.get("context_id") or ""), records)
    if context_material and _contains_any(context_material, target_terms):
        return True
    conditions = _drs_conditions_for_relation(row, records)
    if not conditions:
        return True
    args_by_condition = _drs_arguments_by_condition_id(records)
    saw_argument = False
    for condition in conditions:
        for arg in args_by_condition.get(str(condition.get("drs_condition_id") or ""), []):
            saw_argument = True
            if _argument_values_match_target(_drs_argument_surface_values(arg, records, frame), target_terms):
                return True
    return not saw_argument


def _compatible_values(expected: ExpectedAnswer, values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        text = clean_extracted_value(str(value or ""))
        if not text:
            continue
        if expected.answer_type == "url":
            cleaned.extend(url.rstrip(".,;)") for url in urls(text))
        elif expected.answer_type == "identifier":
            found_urls = [url.rstrip(".,;)") for url in urls(text)]
            if found_urls and text.strip().startswith(found_urls[0]):
                cleaned.extend(found_urls)
            else:
                found_ids = [identifier.rstrip(".,;)") for identifier in identifiers(text)]
                cleaned.extend(found_ids)
                if not found_ids:
                    cleaned.extend(found_urls or [text])
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


def _slot_adjacent_identifier_values(
    value: str,
    answer_slot_terms: list[str] | None,
    target_terms: list[str],
    relation_terms: list[str],
) -> list[str]:
    if not answer_slot_terms:
        return []
    text = clean_extracted_value(value)
    if not text:
        return []
    slot_terms = [
        term
        for term in answer_slot_terms
        if term and term not in ANSWER_SLOT_SKIP_TERMS and not _value_is_target(term, target_terms)
    ]
    if not slot_terms:
        return []
    lowered = text.lower()
    ranked: list[tuple[int, int, str]] = []
    for identifier in identifiers(text):
        cleaned = identifier.rstrip(".,;)")
        if not cleaned:
            continue
        if _value_is_target(cleaned, target_terms) or _rejects_bound_target_value(
            ExpectedAnswer("identifier"),
            cleaned,
            target_terms,
        ):
            continue
        if _value_is_target(cleaned, relation_terms):
            continue
        index = lowered.find(cleaned.lower())
        if index < 0:
            continue
        best_distance: int | None = None
        for term in slot_terms:
            term_norm = normalize(term)
            if not term_norm:
                continue
            for match in re.finditer(re.escape(term_norm), normalize(text)):
                distance = abs(index - match.start())
                if best_distance is None or distance < best_distance:
                    best_distance = distance
        if best_distance is not None:
            ranked.append((best_distance, index, cleaned))
    ranked.sort()
    return list(dict.fromkeys(identifier for _distance, _index, identifier in ranked))


def _answer_slot_class_tokens(
    frame: QueryFrame,
    target_terms: list[str],
) -> list[str]:
    excluded_tokens: set[str] = set()
    for term in [*target_terms, *frame.target_anchors, frame.requested_relation]:
        excluded_tokens.update(content_tokens(term))
    generic = {
        *ANSWER_SLOT_SKIP_TERMS,
        *STRUCTURAL_CHAIN_GENERIC_TERMS,
        "about",
        "despite",
        "implement",
        "implements",
        "implemented",
        "fix",
        "fixed",
        "listed",
    }
    values: list[str] = []
    for variable in frame.answer_variables:
        variable_tokens = [
            token
            for token in re.split(r"[^a-z0-9]+", normalize(variable))
            if len(token) >= 2
        ]
        for token in [*content_tokens(variable), *variable_tokens]:
            if len(token) < 2 or token in generic or token in excluded_tokens:
                continue
            values.append(token)
    return list(dict.fromkeys(values))


def _identifier_slot_alignment_bonus(
    value: str,
    evidence: Evidence,
    frame: QueryFrame,
    target_terms: list[str],
) -> float:
    class_tokens = _answer_slot_class_tokens(frame, target_terms)
    if not class_tokens:
        return 0.0
    value_text = clean_extracted_value(value)
    value_norm = normalize(value_text)
    if not value_norm:
        return 0.0
    value_tokens = _normalized_token_set(value_norm)
    bonus = 0.0
    for token in class_tokens:
        token_variants = set(expand_terms([token]))
        if token_variants & value_tokens:
            bonus = max(bonus, 9.0)
    evidence_text = evidence.text or ""
    lowered = evidence_text.lower()
    value_index = lowered.find(value_text.lower())
    if value_index < 0:
        return bonus
    for token in class_tokens:
        token_norm = normalize(token)
        if not token_norm:
            continue
        for match in re.finditer(rf"\b{re.escape(token_norm)}\b", lowered):
            if match.end() <= value_index:
                distance = value_index - match.end()
            elif value_index + len(value_text) <= match.start():
                distance = match.start() - (value_index + len(value_text))
            else:
                distance = 0
            if distance <= 18:
                bonus = max(bonus, 13.0 - min(distance, 12) * 0.5)
    return bonus


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
    target_terms: list[str] | None = None,
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
    if answer_slot_terms and _answer_slot_constraints(answer_slot_terms, target_terms):
        if not _answer_slot_label_matches(label_material, answer_slot_terms, target_terms):
            return []
    else:
        terms = list(dict.fromkeys([*relation_terms, *(answer_slot_terms or [])]))
        if terms and not _contains_any(label_material, terms):
            return []
    return [answer_text]


def _row_slot_label_material(row: dict[str, Any]) -> str:
    metadata = _relation_metadata(row)
    return normalize(
        " ".join(
            str(item or "")
            for item in [
                row.get("subject"),
                row.get("predicate"),
                metadata.get("column_header"),
                metadata.get("section_anchor"),
                metadata.get("record_path"),
            ]
        )
    )


def _structured_row_matches_answer_slot(
    row: dict[str, Any],
    answer_slot_terms: list[str] | None,
    target_terms: list[str] | None,
) -> bool:
    if not answer_slot_terms or not _answer_slot_constraints(answer_slot_terms, target_terms):
        return True
    relation_type = str(row.get("relation_type") or "")
    if relation_type not in {"label_value", "record_value", "table_cell"}:
        return True
    value_material = normalize(str(row.get("value") or row.get("object") or ""))
    if value_material and _answer_slot_label_matches(value_material, answer_slot_terms, target_terms):
        return True
    label_material = _row_slot_label_material(row)
    if not label_material:
        return True
    return _answer_slot_label_matches(label_material, answer_slot_terms, target_terms)


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



def _row_subject_answers_selector(
    row: dict[str, Any],
    expected: ExpectedAnswer,
    relation_terms: list[str],
    answer_slot_terms: list[str] | None,
) -> bool:
    relation_type = str(row.get("relation_type") or "")
    if relation_type not in {"table_cell", "record_value"}:
        return False
    subject_value = str(row.get("subject") or "")
    value = str(row.get("value") or row.get("object") or "")
    if not subject_value or not value:
        return False
    if not canonicalize_answer(expected, subject_value):
        return False
    if not answer_slot_terms:
        return False
    value_material = normalize(value)
    slot_material = normalize(" ".join(answer_slot_terms or []))
    relation_signal = [
        term for term in relation_terms
        if normalize(term) not in {"is", "are", "was", "were", "answer", "argument"}
        and not _has_term(slot_material, term)
    ]
    return bool(relation_signal and _contains_any(value_material, relation_signal))



def _locative_phrase_from_evidence(evidence: Evidence, target_terms: list[str], relation_terms: list[str]) -> str:
    if not _contains_any(" ".join(relation_terms), ["where"]):
        return ""
    text = clean_extracted_value(evidence.text)
    if target_terms and not _contains_any(normalize(text), target_terms):
        return ""
    match = re.search(
        r"\b(?:is|was|are|were|located|remains?)\s+"
        r"((?:on|in|under|behind|beside|near|inside|outside|above|below|at)\s+[^.;,]+)",
        text,
        re.I,
    )
    return clean_extracted_value(match.group(1)).strip() if match else ""

def _locative_answer_value(frame_row: dict[str, Any], value: str, relation_terms: list[str]) -> str:
    if not value or not _contains_any(" ".join(relation_terms), ["where"]):
        return value
    predicate = normalize(str(frame_row.get("predicate") or frame_row.get("trigger_surface") or ""))
    if predicate in {"on", "in", "under", "behind", "beside", "near", "inside", "outside", "above", "below", "at"}:
        value_norm = normalize(value)
        if value_norm and not value_norm.startswith(predicate + " "):
            return f"{predicate} {value}"
    return value

def _answer_values_from_relation(
    row: dict[str, Any],
    evidence: Evidence,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
    answer_slot_terms: list[str] | None = None,
    records: dict[str, Any] | None = None,
    query_frame: QueryFrame | None = None,
) -> list[str]:
    relation_type = str(row.get("relation_type") or "")
    if not _structured_row_matches_answer_slot(row, answer_slot_terms, target_terms):
        return []
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
        primary_values = _drs_condition_argument_values(
            row,
            records,
            expected,
            target_terms,
            relation_terms,
            answer_slot_terms,
            query_frame,
        )
        if not primary_values and expected.answer_type in {"content_phrase", "unknown"}:
            primary_values = [str(row.get("value") or "")]
    else:
        primary_values = [str(row.get(key) or "") for key in ["value", "object"]]
        subject_value = str(row.get("subject") or "")
        if _row_subject_answers_selector(row, expected, relation_terms, answer_slot_terms):
            # The row value matched the requested selector, so the row subject is
            # the answer.  Do not also return the selector value itself, e.g.
            # status=unpaid should answer INV-101, not unpaid.
            primary_values = [subject_value]
    fallback_values = (
        []
        if relation_type in {"semantic_argument", "semantic_frame", "drs_condition"}
        else [str(row.get("subject") or "")]
    )
    label_values = [
        split_value
        for value in [*primary_values, *fallback_values]
        for split_value in _matching_structural_label_values(value, relation_terms, answer_slot_terms, target_terms)
    ]
    if label_values:
        primary_values = label_values
        fallback_values = []
    url_requested = _unknown_query_requests_url(expected, relation_terms, answer_slot_terms)
    structural = expected.answer_type in {"url", "identifier", "file_path", "date_time", "count"} or url_requested
    primary_values = [
        value for value in primary_values
        if value
        and not _value_is_target(value, target_terms)
        and (structural or not _rejects_bound_target_value(expected, value, target_terms))
        and (structural or not _value_is_target(value, relation_terms))
    ]
    fallback_values = [
        value for value in fallback_values
        if value
        and not _value_is_target(value, target_terms)
        and not _rejects_bound_target_value(expected, value, target_terms)
        and (structural or not _value_is_target(value, relation_terms))
    ]
    if url_requested:
        return list(dict.fromkeys(url.rstrip(".,;)") for value in [*primary_values, *fallback_values] for url in urls(value)))
    if expected.answer_type == "identifier":
        slot_identifiers = [
            identifier
            for value in [*primary_values, *fallback_values]
            for identifier in _slot_adjacent_identifier_values(value, answer_slot_terms, target_terms, relation_terms)
        ]
        if slot_identifiers:
            return slot_identifiers
    compatible = [
        person_value
        for value in primary_values
        for person_value in _person_values_from_relation_text(value, expected)
    ]
    if not compatible:
        compatible = _compatible_values(expected, primary_values)
    if not compatible:
        compatible = [
            person_value
            for value in fallback_values
            for person_value in _person_values_from_relation_text(value, expected)
        ]
    if not compatible:
        compatible = _compatible_values(expected, fallback_values)
    if compatible or not structural:
        return compatible
    if relation_type == "drs_condition":
        return []
    return _compatible_values(expected, [evidence.text])


def _answer_values_from_frame(
    frame_row: dict[str, Any],
    args: list[dict[str, Any]],
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
    answer_slot_terms: list[str] | None = None,
    frame_type_material: str = "",
    evidence: Evidence | None = None,
    records: dict[str, Any] | None = None,
    query_frame: QueryFrame | None = None,
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
            # Some query DRS answer variables are the whole question clause
            # rather than a clean slot label.  If the model condition predicate
            # itself satisfies those slot words, keep the condition arguments
            # and let target/relation filtering below remove non-answer args.
            candidate_args = args
    values: list[str] = []
    for arg in candidate_args:
        surface = _locative_answer_value(frame_row, str(arg.get("surface") or ""), relation_terms)
        if surface:
            values.append(surface)
        if records is not None:
            for label in _identity_labels_for_referent(records, str(arg.get("referent_id") or ""), query_frame):
                identity_value = _locative_answer_value(frame_row, label, relation_terms)
                if identity_value:
                    values.append(identity_value)
    url_requested = _unknown_query_requests_url(expected, relation_terms, answer_slot_terms)
    structural = expected.answer_type in {"url", "identifier", "file_path", "date_time", "count"} or url_requested
    values = [
        value for value in values
        if value
        and not _value_is_target(value, target_terms)
        and (structural or not _rejects_bound_target_value(expected, value, target_terms))
        and (structural or not _value_is_target(value, relation_terms))
        and (
            expected.answer_type not in {"person", "actor", "organization"}
            or normalize(value) not in UNRESOLVED_PRONOUN_ANSWER_VALUES
        )
    ]
    if url_requested:
        return list(dict.fromkeys(url.rstrip(".,;)") for value in values for url in urls(value)))
    if evidence is not None:
        locative = _locative_phrase_from_evidence(evidence, target_terms, relation_terms)
        if locative:
            # For where-questions, the evidence sentence preserves articles and
            # prepositions better than decomposed frame arguments.  Prefer the
            # complete grounded locative phrase rather than letting a shorter
            # argument such as "on red desk" win tie-breaking.
            values = [locative]
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
    answer_slot_terms = _answer_slot_terms(frame, target_terms)
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
            evidence,
            records,
            frame,
        ):
            candidates.append((score, value, evidence, "frame_argument_binding"))
    return candidates


def _bind_relation_conditions(records: dict[str, Any], frame: QueryFrame, expected: ExpectedAnswer, target_terms: list[str], relation_terms: list[str]) -> list[tuple[float, str, Evidence, str]]:
    answer_slot_terms = _answer_slot_terms(frame, target_terms)
    candidates: list[tuple[float, str, Evidence, str]] = []
    for row in records.get("relations", []):
        if not _relation_scope_accessible(row, records, frame):
            continue
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        if _source_is_low_priority(evidence.rel_path, evidence.text) and not _structured_source_row(row):
            continue
        row_material = _relation_local_material(row, evidence, include_evidence=False, include_context=True, records=records)
        if str(row.get("relation_type") or "") == "drs_condition":
            if not _drs_condition_polarity_matches_query(row, records, frame):
                continue
            if not _has_specific_relation_hit(row_material, relation_terms, target_terms):
                continue
            if not _drs_condition_has_target_argument(row, records, target_terms, frame):
                continue
        evidence_material = normalize(" ".join([row_material, evidence.rel_path, evidence.text]))
        score = _split_match_score(evidence_material, row_material, target_terms, relation_terms)
        if score <= 0:
            continue
        for value in _answer_values_from_relation(
            row,
            evidence,
            expected,
            target_terms,
            relation_terms,
            answer_slot_terms,
            records,
            frame,
        ):
            candidates.append((score * float(row.get("confidence") or 0.7), value, evidence, "relation_condition_binding"))
    return candidates


def _bind_document_scoped_label_values(
    records: dict[str, Any],
    frame: QueryFrame,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
) -> list[tuple[float, str, Evidence, str]]:
    """Bind field values from a small source-local structural record cluster.

    This covers a general document pattern such as ``Name: X. Field: Y.`` where
    the target anchor and requested field are adjacent structural label/value
    rows rather than one table/object row.  It is intentionally conservative:
    the document must contain exactly one accessible structural row matching the
    target terms, avoiding multi-object documents where a field could belong to
    the wrong entity.
    """

    if not target_terms:
        return []
    answer_slot_terms = _answer_slot_terms(frame, target_terms)
    spans = _spans_by_id(records)
    target_document_ids = _document_context_target_document_ids(records, target_terms)

    def document_target_evidence(document_id: str) -> Evidence | None:
        if not document_id:
            return None
        partial_match: Evidence | None = None
        for span in records.get("source_spans", []):
            if str(span.get("document_id") or "") != document_id:
                continue
            evidence = _evidence_for_span(str(span.get("span_id") or ""), records)
            material = normalize(evidence.text)
            if _terms_match_material(target_terms, material):
                return evidence
            if partial_match is None and _contains_any(material, target_terms):
                partial_match = evidence
        if partial_match is not None:
            return partial_match
        doc = _docs_by_id(records).get(document_id, {})
        rel_path = str(doc.get("rel_path") or "")
        context = normalize(str((records.get("document_context_norm_by_rel_path") or {}).get(rel_path, "")))
        if rel_path and _terms_match_material(target_terms, context):
            return Evidence(rel_path, context, 0.64, source_kind="document")
        return None

    rows_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records.get("relations", []):
        if not _relation_scope_accessible(row, records, frame):
            continue
        if str(row.get("relation_type") or "") != "label_value":
            continue
        document_id = str(spans.get(str(row.get("source_span_id") or ""), {}).get("document_id") or "")
        rows_by_document[document_id].append(row)
    candidates: list[tuple[float, str, Evidence, str]] = []
    for _document_id, rows in rows_by_document.items():
        target_rows: list[dict[str, Any]] = []
        for row in rows:
            evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
            material = _relation_selector_material(row, evidence, include_evidence=True)
            if _contains_any(material, target_terms):
                target_rows.append(row)
        target_evidence: Evidence | None
        if len(target_rows) == 1:
            target_evidence = _evidence_for_span(str(target_rows[0].get("source_span_id") or ""), records)
        elif _document_id in target_document_ids and len(target_document_ids) == 1:
            target_evidence = document_target_evidence(_document_id)
        else:
            continue
        if target_evidence is None:
            continue
        for row in rows:
            if len(target_rows) == 1 and row is target_rows[0]:
                continue
            evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
            if _source_is_low_priority(evidence.rel_path, evidence.text) and not _structured_source_row(row):
                continue
            local_material = _relation_selector_material(row, evidence)
            relation_hit = _contains_any(local_material, relation_terms) or _contains_any(local_material, answer_slot_terms)
            if not relation_hit:
                continue
            for value in _answer_values_from_relation(row, evidence, expected, [], relation_terms, answer_slot_terms):
                candidates.append((6.5 * float(row.get("confidence") or 0.7), value, target_evidence, "document_scoped_label_binding"))
                candidates.append((6.5 * float(row.get("confidence") or 0.7), value, evidence, "document_scoped_label_binding"))
    return candidates


def _material_matches_compound_term_constraints(
    material: str,
    terms: list[str],
    target_terms: list[str] | None = None,
) -> bool:
    constraints = _answer_slot_constraints(terms, target_terms)
    if not constraints:
        return False
    material_norm = normalize(material)
    if not material_norm:
        return False
    material_tokens = _normalized_token_set(material_norm)
    for term, tokens in constraints:
        if term and _has_term(material_norm, term):
            return True
        if all(token in material_tokens or _slot_token_matches(material_norm, token) for token in tokens):
            return True
    return False


def _document_context_target_document_ids(records: dict[str, Any], target_terms: list[str]) -> set[str]:
    if not target_terms:
        return set()
    docs_by_rel_path = _docs_by_rel_path(records)
    document_ids: set[str] = set()
    for rel_path, material in (records.get("document_context_norm_by_rel_path") or {}).items():
        if not _terms_match_material(target_terms, normalize(str(material or ""))):
            continue
        document_id = str(docs_by_rel_path.get(str(rel_path), {}).get("document_id") or "")
        if document_id:
            document_ids.add(document_id)
    return document_ids


def _rows_document_id(rows: list[dict[str, Any]], records: dict[str, Any]) -> str:
    spans = _spans_by_id(records)
    document_ids = {
        str(spans.get(str(row.get("source_span_id") or ""), {}).get("document_id") or "")
        for row in rows
    }
    document_ids.discard("")
    return next(iter(document_ids)) if len(document_ids) == 1 else ""


def _document_scoped_row_selector_matches(
    material: str,
    relation_terms: list[str],
    answer_slot_terms: list[str],
    target_terms: list[str],
) -> bool:
    compound_terms = [*relation_terms, *answer_slot_terms]
    if _material_matches_compound_term_constraints(material, compound_terms, target_terms):
        return True
    if _answer_slot_constraints(compound_terms, target_terms):
        return False
    return _contains_any(material, list(dict.fromkeys(compound_terms)))


def _document_scoped_structural_row_candidates(
    records: dict[str, Any],
    frame: QueryFrame,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
) -> list[tuple[float, str, Evidence, str]]:
    """Bind rows in a table/object whose document carries the target anchor.

    Some sources introduce the entity once in prose and then provide a compact
    table of properties or participants.  The entity is document-scoped rather
    than repeated in each row.  This operator only activates when exactly one
    selected document carries the target terms, then uses ordinary structural
    row groups and compound selector terms to choose answer rows.
    """

    target_document_ids = _document_context_target_document_ids(records, target_terms)
    if len(target_document_ids) != 1:
        return []
    target_document_id = next(iter(target_document_ids))
    answer_slot_terms = _answer_slot_terms(frame, target_terms)
    candidates: list[tuple[float, str, Evidence, str]] = []
    for _group_id, rows in _record_groups(records).items():
        if not rows or not _rows_are_countable_structured_units(rows):
            continue
        if _rows_document_id(rows, records) != target_document_id:
            continue
        group_material = _group_material(rows, records, include_source_evidence=False)
        if not _document_scoped_row_selector_matches(
            group_material,
            relation_terms,
            answer_slot_terms,
            target_terms,
        ):
            continue
        for row in rows:
            if not _relation_scope_accessible(row, records, frame):
                continue
            evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
            local_material = _relation_selector_material(row, evidence)
            value_material = normalize(str(row.get("value") or row.get("object") or ""))
            row_slot_match = bool(
                answer_slot_terms
                and (
                    _answer_slot_label_matches(local_material, answer_slot_terms, target_terms)
                    or _answer_slot_label_matches(value_material, answer_slot_terms, target_terms)
                )
            )
            row_selector_match = _document_scoped_row_selector_matches(
                normalize(" ".join([local_material, value_material])),
                relation_terms,
                answer_slot_terms,
                target_terms,
            )
            if not row_slot_match and not row_selector_match:
                continue
            slot_terms = answer_slot_terms if row_slot_match else None
            for value in _answer_values_from_relation(row, evidence, expected, [], relation_terms, slot_terms):
                score = 8.0 + (3.0 if row_slot_match else 0.0) + (2.0 if row_selector_match else 0.0)
                candidates.append(
                    (
                        score * float(row.get("confidence") or 0.7),
                        value,
                        evidence,
                        "document_scoped_structural_row_binding",
                    )
                )
    return candidates


def _document_scoped_relation_value_candidates(
    records: dict[str, Any],
    frame: QueryFrame,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
) -> list[tuple[float, str, Evidence, str]]:
    target_document_ids = _document_context_target_document_ids(records, target_terms)
    if not target_document_ids:
        return []
    require_slot_aligned_identifier = len(target_document_ids) > 1
    if require_slot_aligned_identifier and expected.answer_type != "identifier":
        return []
    spans = _spans_by_id(records)
    answer_slot_terms = _answer_slot_terms(frame, target_terms)
    candidates: list[tuple[float, str, Evidence, str]] = []
    for row in records.get("relations", []):
        relation_type = str(row.get("relation_type") or "")
        if relation_type not in {"identifier", "label_value", "record_value", "table_cell"}:
            continue
        if not _relation_scope_accessible(row, records, frame):
            continue
        span_id = str(row.get("source_span_id") or "")
        if str(spans.get(span_id, {}).get("document_id") or "") not in target_document_ids:
            continue
        evidence = _evidence_for_span(span_id, records)
        if _source_is_low_priority(evidence.rel_path, evidence.text) and not _structured_source_row(row):
            continue
        local_material = _relation_selector_material(row, evidence, include_evidence=True)
        if not _document_scoped_row_selector_matches(
            local_material,
            relation_terms,
            answer_slot_terms,
            target_terms,
        ):
            continue
        for value in _answer_values_from_relation(row, evidence, expected, target_terms, relation_terms, answer_slot_terms):
            slot_bonus = (
                _identifier_slot_alignment_bonus(value, evidence, frame, target_terms)
                if expected.answer_type == "identifier"
                else 0.0
            )
            if require_slot_aligned_identifier and slot_bonus <= 0.0:
                continue
            candidates.append(
                (
                    7.0 * float(row.get("confidence") or 0.7) + slot_bonus,
                    value,
                    evidence,
                    "document_scoped_relation_value_binding",
                )
            )
    return candidates


def _visible_target_terms(frame: QueryFrame, question: str) -> list[str]:
    visible = {normalize(anchor) for anchor in visible_anchors(question)}
    values: list[str] = []
    for anchor in frame.target_anchors:
        norm = normalize(anchor)
        if norm and norm in visible:
            values.append(norm)
            if " " in norm:
                values.append(norm.replace(" ", "_"))
                values.append(norm.replace(" ", "-"))
    return list(dict.fromkeys(values))


def _structural_chain_term_groups(
    frame: QueryFrame,
    target_terms: list[str],
    visible_target_terms: list[str],
) -> list[list[str]]:
    visible_tokens: set[str] = set()
    for term in visible_target_terms:
        visible_tokens.update(content_tokens(term))
    visible_term_set = _normalized_term_set(tuple(visible_target_terms))
    raw_terms = [
        *frame.relation_terms,
        *frame.constraints,
        *_answer_slot_terms(frame, target_terms),
        *[term for term in target_terms if normalize(term) not in visible_term_set],
    ]
    groups: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for term in raw_terms:
        tokens = [
            token
            for token in content_tokens(term)
            if token not in STRUCTURAL_CHAIN_GENERIC_TERMS
            and token not in visible_tokens
            and token not in COUNT_AGGREGATION_SKIP_TERMS
        ]
        for token in tokens:
            variants = [
                variant
                for variant in expand_terms([token])
                if variant and variant not in STRUCTURAL_CHAIN_GENERIC_TERMS
            ]
            key = tuple(sorted(variants))
            if variants and key not in seen:
                groups.append(variants)
                seen.add(key)
    return groups


def _structural_chain_label_material(row: dict[str, Any]) -> str:
    metadata = _relation_metadata(row)
    return normalize(
        " ".join(
            str(item or "")
            for item in [
                row.get("subject"),
                row.get("predicate"),
                metadata.get("record_path"),
                metadata.get("row_key"),
                metadata.get("column_header"),
                metadata.get("section_anchor"),
            ]
        )
    )


def _structural_chain_rows(records: dict[str, Any], frame: QueryFrame) -> list[tuple[dict[str, Any], Evidence, str, str, str, str]]:
    rows: list[tuple[dict[str, Any], Evidence, str, str, str, str]] = []
    for row in records.get("relations", []):
        relation_type = str(row.get("relation_type") or "")
        if relation_type not in {"label_value", "record_value", "table_cell"}:
            continue
        if not _relation_scope_accessible(row, records, frame):
            continue
        if relation_type != "label_value" and not _structured_source_row(row):
            continue
        value = clean_extracted_value(str(row.get("value") or row.get("object") or ""))
        subject = clean_extracted_value(str(row.get("subject") or ""))
        if not subject or not value:
            continue
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        if _source_is_low_priority(evidence.rel_path, evidence.text) and relation_type != "label_value" and not _structured_source_row(row):
            continue
        label_material = _structural_chain_label_material(row)
        local_material = _relation_local_material(row, evidence, include_evidence=False, include_context=True, records=records)
        rows.append((row, evidence, normalize(subject), normalize(value), value, normalize(" ".join([label_material, local_material]))))
    return rows


def _structural_chain_group_hits(material: str, groups: list[list[str]]) -> frozenset[int]:
    return frozenset(index for index, group in enumerate(groups) if _material_matches_term_group(material, group))


def _structural_chain_frame_rows(records: dict[str, Any], frame: QueryFrame) -> list[tuple[dict[str, Any], Evidence, str, str, str, str]]:
    args_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arg in records.get("frame_arguments", []):
        args_by_frame[str(arg.get("frame_id") or "")].append(arg)
    spans = _spans_by_id(records)
    rows: list[tuple[dict[str, Any], Evidence, str, str, str, str]] = []
    for row in records.get("frames", []):
        source = str(row.get("source") or "")
        if source not in {"local_model", "local_model_drs"}:
            continue
        if not _context_accessible(str(row.get("context_id") or ""), records, frame):
            continue
        evidence = _evidence_for_span(str(row.get("span_id") or ""), records)
        arguments = [
            arg for arg in args_by_frame.get(str(row.get("frame_id") or ""), [])
            if clean_extracted_value(str(arg.get("surface") or ""))
        ]
        if len(arguments) < 2 or len(arguments) > 8:
            continue
        source_args = [
            arg for arg in arguments
            if normalize(str(arg.get("role") or "")) in STRUCTURAL_CHAIN_SOURCE_ARG_ROLES
        ]
        target_args = [
            arg for arg in arguments
            if normalize(str(arg.get("role") or "")) in STRUCTURAL_CHAIN_TARGET_ARG_ROLES
        ]
        if not source_args:
            continue
        if not target_args and len(arguments) == 2:
            target_args = [arg for arg in arguments if arg not in source_args]
        if not target_args:
            continue
        span = spans.get(str(row.get("span_id") or ""), {})
        edge_base = {
            "document_id": str(span.get("document_id") or ""),
            "source_span_id": str(row.get("span_id") or ""),
            "context_id": str(row.get("context_id") or ""),
            "relation_type": "semantic_frame",
            "predicate": str(row.get("predicate") or ""),
        }
        for source_arg in source_args:
            subject = clean_extracted_value(str(source_arg.get("surface") or ""))
            subject_norm = normalize(subject)
            if not subject_norm:
                continue
            for target_arg in target_args:
                value = clean_extracted_value(str(target_arg.get("surface") or ""))
                value_norm = normalize(value)
                if not value_norm or value_norm == subject_norm:
                    continue
                material = normalize(
                    " ".join(
                        [
                            str(row.get("predicate") or ""),
                            str(row.get("trigger_surface") or ""),
                            str(source_arg.get("role") or ""),
                            subject,
                            str(target_arg.get("role") or ""),
                            value,
                        ]
                    )
                )
                rows.append(({**edge_base, "subject": subject, "value": value}, evidence, subject_norm, value_norm, value, material))
    return rows


def _structural_chain_candidates(
    records: dict[str, Any],
    frame: QueryFrame,
    expected: ExpectedAnswer,
    target_terms: list[str],
) -> list[tuple[float, str, Evidence, str]]:
    visible_terms = _visible_target_terms(frame, frame.question_text)
    start_terms = visible_terms or target_terms
    if not start_terms:
        return []
    groups = _structural_chain_term_groups(frame, target_terms, visible_terms)
    if not groups:
        return []
    rows = _structural_chain_rows(records, frame)
    rows.extend(_structural_chain_frame_rows(records, frame))
    if not rows:
        return []
    rows_by_document: dict[str, list[tuple[dict[str, Any], Evidence, str, str, str, str]]] = defaultdict(list)
    for item in rows:
        rows_by_document[item[1].rel_path].append(item)
    candidates: list[tuple[float, str, Evidence, str]] = []
    required = frozenset(range(len(groups)))
    max_depth = 4
    for document_rows in rows_by_document.values():
        for index, (_row, evidence, subject_norm, value_norm, value_text, material) in enumerate(document_rows):
            if not any(term and _has_term(subject_norm, term) for term in start_terms):
                continue
            stack: list[tuple[int, str, str, frozenset[int], list[int], Evidence]] = [
                (
                    1,
                    value_norm,
                    value_text,
                    _structural_chain_group_hits(material, groups),
                    [index],
                    evidence,
                )
            ]
            while stack:
                depth, current_value_norm, current_value_text, covered, path, final_evidence = stack.pop()
                if required.issubset(covered):
                    for value in _compatible_values(expected, [current_value_text]):
                        if not _rejects_bound_target_value(expected, value, start_terms):
                            score = 34.0 + len(covered) * 5.0 - max(0, depth - 1) * 1.5
                            candidates.append((score, value, final_evidence, "structural_chain_drs_binding"))
                if depth >= max_depth or not current_value_norm:
                    continue
                for next_index, (_next_row, next_evidence, next_subject, next_value_norm, next_value_text, next_material) in enumerate(document_rows):
                    if next_index in path:
                        continue
                    if not _has_term(next_subject, current_value_norm):
                        continue
                    next_covered = frozenset([*covered, *_structural_chain_group_hits(next_material, groups)])
                    if next_covered == covered:
                        continue
                    stack.append((depth + 1, next_value_norm, next_value_text, next_covered, [*path, next_index], next_evidence))
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


def _group_material(
    rows: list[dict[str, Any]],
    records: dict[str, Any],
    *,
    include_document_context: bool = False,
    include_source_evidence: bool = True,
) -> str:
    parts: list[str] = []
    context_rel_paths: set[str] = set()
    for row in rows:
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        if include_source_evidence:
            parts.append(_relation_local_material(row, evidence, include_evidence=True))
        else:
            parts.append(_relation_local_material(row, evidence, include_evidence=False))
            if _structured_source_row(row) and evidence.rel_path:
                parts.append(evidence.rel_path)
        if include_document_context and evidence.rel_path not in context_rel_paths:
            context_rel_paths.add(evidence.rel_path)
            parts.append((records.get("document_context_norm_by_rel_path") or {}).get(evidence.rel_path, ""))
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
    answer_slot_terms = _answer_slot_terms(frame, target_terms)
    for _group_id, rows in _record_groups(records).items():
        if not rows:
            continue
        structured_group = _rows_are_countable_structured_units(rows)
        group_material = _group_material(rows, records, include_source_evidence=not structured_group)
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
            local_material = _relation_selector_material(row, evidence)
            relation_hits = sum(1 for term in relation_terms if _has_term(local_material, term))
            if relation_terms and relation_hits == 0:
                continue
            values = _answer_values_from_relation(row, evidence, expected, target_terms, relation_terms, answer_slot_terms)
            for value in values:
                value_hits = sum(1 for term in relation_terms if _has_term(normalize(value), term))
                score = 5.0 + target_hits * 5.0 + relation_hits * 6.0 + group_relation_hits * 1.5
                score += value_hits * 4.0
                score *= float(row.get("confidence") or 0.7)
                if expected.answer_type == "identifier":
                    score += _identifier_slot_alignment_bonus(value, evidence, frame, target_terms)
                candidates.append((score, value, evidence, "record_group_drs_binding"))
    return candidates


def _relation_term_groups_for_frame(frame: QueryFrame, target_terms: list[str] | None = None) -> list[list[str]]:
    groups: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add_group(variants: list[str]) -> None:
        variants = list(dict.fromkeys(variant for variant in variants if variant))
        if not variants:
            return
        variant_set = set(variants)
        if any(variant_set.issubset(set(group)) for group in groups):
            return
        groups[:] = [group for group in groups if not set(group).issubset(variant_set)]
        seen.clear()
        seen.update(tuple(sorted(group)) for group in groups)
        key = tuple(sorted(variants))
        if key not in seen:
            groups.append(variants)
            seen.add(key)

    raw_items = [*frame.relation_terms, *list(frame.constraints), *_query_terms(frame.requested_relation)]
    count_unit_tokens = _count_answer_unit_tokens(frame) if frame.aggregation == "count" else set()
    target_tokens: set[str] = set()
    for term in target_terms or []:
        target_tokens.update(content_tokens(term))
    generic = {
        "answer",
        "argument",
        "value",
        "values",
        "many",
        "much",
        "row",
        "rows",
        "entry",
        "entries",
        "have",
        "has",
        "had",
        "is",
        "are",
        "was",
        "were",
        "be",
    }
    for item in raw_items:
        item_norm = normalize(item)
        if not item_norm:
            continue
        if frame.aggregation == "count":
            if item_norm in COUNT_AGGREGATION_SKIP_TERMS or item_norm in generic:
                continue
            if item_norm in {"how many", "how much"} or item_norm.startswith("how many ") or item_norm.startswith("how much "):
                continue
            if item_norm.startswith("have ") or item_norm.startswith("has ") or item_norm.startswith("had "):
                item_norm = normalize(re.sub(r"^(?:have|has|had)\s+", "", item_norm))
            # Count queries need field/value groups, not whole surface clauses.
            # Requiring a row to contain "have status active" rejected exact
            # table rows that correctly contain only "status active".
            tokens = [
                token for token in content_tokens(item_norm)
                if token not in generic and token not in COUNT_AGGREGATION_SKIP_TERMS
                and token not in count_unit_tokens and token not in target_tokens
            ]
            if not tokens:
                continue
            for token in tokens:
                variants = [variant for variant in expand_terms([token]) if variant and variant not in generic]
                add_group(variants)
            continue
        variants = _compound_term_variants(item)
        if not variants:
            variants = [item_norm]
        variants = [variant for variant in expand_terms(variants) if variant and variant not in generic]
        add_group(variants)
    return groups


def _material_matches_all_term_groups(material: str, groups: list[list[str]]) -> bool:
    return all(any(_has_term(material, term) for term in group) for group in groups)


def _material_matches_term_group(material: str, group: list[str]) -> bool:
    return any(_has_term(material, term) for term in group)


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
    return bool(re.search(r"\b(?:rows?|entries|records?)\b", material))


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


def _rows_are_countable_structured_units(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        metadata = _relation_metadata(row)
        if str(row.get("relation_type") or "") == "record_value":
            return True
        if str(metadata.get("surface_format") or "") in {"delimited_table", "json", "json_like", "object_like"}:
            return True
    return False


def _countable_structured_rel_paths(records: dict[str, Any]) -> set[str]:
    rel_paths: set[str] = set()
    for rows in _record_groups(records).values():
        if not _rows_are_countable_structured_units(rows):
            continue
        span_ids = {str(row.get("source_span_id") or "") for row in rows}
        for span_id in span_ids:
            evidence = _evidence_for_span(span_id, records)
            if evidence.rel_path:
                rel_paths.add(evidence.rel_path)
    return rel_paths


def _row_local_count_match_rel_paths(
    records: dict[str, Any],
    target_terms: list[str],
    required_relation_groups: list[list[str]],
) -> tuple[set[str], dict[int, set[str]]]:
    target_rel_paths: set[str] = set()
    relation_group_rel_paths: dict[int, set[str]] = defaultdict(set)
    for rows in _record_groups(records).values():
        if not rows:
            continue
        rows_by_span: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_span[str(row.get("source_span_id") or "")].append(row)
        for span_id, span_rows in rows_by_span.items():
            if not span_rows or not _rows_are_countable_structured_units(span_rows):
                continue
            evidence = _evidence_for_span(span_id, records)
            local_material = _group_material(span_rows, records, include_source_evidence=False)
            if target_terms and _contains_any(local_material, target_terms):
                target_rel_paths.add(evidence.rel_path)
            for index, group in enumerate(required_relation_groups):
                if _material_matches_term_group(local_material, group):
                    relation_group_rel_paths[index].add(evidence.rel_path)
    return target_rel_paths, relation_group_rel_paths


def _count_matching_record_groups(
    records: dict[str, Any],
    frame: QueryFrame,
    target_terms: list[str],
    relation_terms: list[str],
) -> tuple[int, list[Evidence]]:
    groups = _record_groups(records)
    required_relation_groups = _relation_term_groups_for_frame(frame, target_terms)
    target_row_local_rel_paths, relation_group_row_local_rel_paths = _row_local_count_match_rel_paths(
        records,
        target_terms,
        required_relation_groups,
    )
    countable_rel_paths = _countable_structured_rel_paths(records)
    require_structured_unit = _frame_requests_row_units(frame)
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
            span_is_structured = _rows_are_countable_structured_units(span_rows)
            if require_structured_unit and not span_is_structured:
                continue
            evidence = _evidence_for_span(span_id, records)
            if evidence.rel_path in countable_rel_paths and not span_is_structured:
                continue
            if _source_is_low_priority(evidence.rel_path, evidence.text) and not any(_structured_source_row(row) for row in span_rows):
                continue
            span_material = _group_material(span_rows, records, include_source_evidence=False)
            scoped_material = ""
            if target_terms and not _contains_any(span_material, target_terms):
                if evidence.rel_path in target_row_local_rel_paths:
                    continue
                scoped_material = _group_material(span_rows, records, include_document_context=True, include_source_evidence=False)
                if not _contains_any(scoped_material, target_terms):
                    continue
            group_failed = False
            for index, group in enumerate(required_relation_groups):
                if _material_matches_term_group(span_material, group):
                    continue
                if not span_is_structured:
                    group_failed = True
                    break
                if evidence.rel_path in relation_group_row_local_rel_paths.get(index, set()):
                    group_failed = True
                    break
                if not scoped_material:
                    scoped_material = _group_material(span_rows, records, include_document_context=True, include_source_evidence=False)
                if not _material_matches_term_group(scoped_material, group):
                    group_failed = True
                    break
            if group_failed:
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


def _temporal_constraint_terms(frame: QueryFrame) -> list[str]:
    terms: list[str] = []
    for value in frame.constraints:
        for match in DATE_TIME_RE.finditer(str(value or "")):
            terms.append(normalize(match.group(0)))
    return list(dict.fromkeys(term for term in terms if term))


def _effective_temporal_scope(frame: QueryFrame) -> str:
    scope = normalize_temporal_scope(frame.temporal_scope)
    if scope in {"latest", "earliest"}:
        return scope
    for value in frame.constraints:
        scope = normalize_temporal_scope(str(value or ""))
        if scope in {"latest", "earliest"}:
            return scope
    return ""


def _temporal_row_matches_constraints(row: dict[str, Any], evidence: Evidence, constraint_terms: list[str]) -> bool:
    if not constraint_terms:
        return True
    material = normalize(" ".join([str(row.get("temporal_value") or ""), evidence.text]))
    return any(term and term in material for term in constraint_terms)


def _temporal_candidates(records: dict[str, Any], frame: QueryFrame, expected: ExpectedAnswer, target_terms: list[str], relation_terms: list[str]) -> list[tuple[float, str, Evidence, str]]:
    if expected.answer_type not in {"state", "date_time", "content_phrase", "unknown"}:
        return []
    temporal_scope = _effective_temporal_scope(frame)
    if temporal_scope not in {"latest", "earliest"} and expected.answer_type not in {"state", "date_time"}:
        return []
    referents = _referents_by_id(records)
    temporal_constraints = _temporal_constraint_terms(frame)
    rows: list[tuple[str, dict[str, Any], Evidence]] = []
    for row in records.get("temporal_edges", []):
        if not _context_accessible(str(row.get("context_id") or ""), records, frame):
            continue
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        if not _temporal_row_matches_constraints(row, evidence, temporal_constraints):
            continue
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
        if relation_terms and not _contains_any(material, relation_terms) and not temporal_constraints:
            continue
        rows.append((str(row.get("temporal_value") or ""), row, evidence))
    rows.sort(key=lambda item: item[0], reverse=temporal_scope != "earliest")
    candidates: list[tuple[float, str, Evidence, str]] = []
    selected_rows = rows[:3]
    if temporal_scope in {"latest", "earliest"} and rows:
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
    temporal_scope = _effective_temporal_scope(frame)
    if temporal_scope not in {"latest", "earliest"}:
        return []
    if expected.answer_type not in {"state", "date_time", "content_phrase", "unknown"}:
        return []
    answer_slot_terms = _answer_slot_terms(frame, target_terms)
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
    ordered.sort(key=lambda item: item[0], reverse=temporal_scope != "earliest")
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


def _selected_temporal_span_ids(
    records: dict[str, Any],
    frame: QueryFrame,
    target_terms: list[str],
    relation_terms: list[str],
) -> set[str]:
    temporal_scope = _effective_temporal_scope(frame)
    if temporal_scope not in {"latest", "earliest"} and not _temporal_constraint_terms(frame):
        return set()
    referents = _referents_by_id(records)
    temporal_constraints = _temporal_constraint_terms(frame)
    ordered: list[tuple[str, str, Evidence]] = []
    for row in records.get("temporal_edges", []):
        if not _context_accessible(str(row.get("context_id") or ""), records, frame):
            continue
        span_id = str(row.get("source_span_id") or "")
        if not span_id:
            continue
        evidence = _evidence_for_span(span_id, records)
        if _source_is_low_priority(evidence.rel_path, evidence.text):
            continue
        if not _temporal_row_matches_constraints(row, evidence, temporal_constraints):
            continue
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
        if relation_terms and not _contains_any(material, relation_terms) and not temporal_constraints:
            continue
        ordered.append((str(row.get("temporal_value") or ""), span_id, evidence))
    if not ordered:
        return set()
    ordered.sort(key=lambda item: item[0], reverse=temporal_scope != "earliest")
    if temporal_constraints:
        return {span_id for _time_value, span_id, _evidence in ordered}
    boundary_value = ordered[0][0]
    return {span_id for time_value, span_id, _evidence in ordered if time_value == boundary_value}


def _temporal_frame_argument_candidates(
    records: dict[str, Any],
    frame: QueryFrame,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
) -> list[tuple[float, str, Evidence, str]]:
    selected_span_ids = _selected_temporal_span_ids(records, frame, target_terms, relation_terms)
    if not selected_span_ids:
        return []
    args_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arg in records.get("frame_arguments", []):
        args_by_frame[str(arg.get("frame_id") or "")].append(arg)
    candidates: list[tuple[float, str, Evidence, str]] = []
    for row in records.get("frames", []):
        span_id = str(row.get("span_id") or "")
        if span_id not in selected_span_ids:
            continue
        if not _context_accessible(str(row.get("context_id") or ""), records, frame):
            continue
        evidence = _evidence_for_span(span_id, records)
        arg_text = " ".join(str(arg.get("surface") or "") for arg in args_by_frame.get(str(row.get("frame_id") or ""), []))
        material = normalize(" ".join([str(row.get("predicate") or ""), str(row.get("trigger_surface") or ""), arg_text, evidence.text]))
        if target_terms and not _contains_any(material, target_terms):
            continue
        if relation_terms and not _contains_any(material, relation_terms) and not _temporal_constraint_terms(frame):
            continue
        for value in _answer_values_from_frame(
            row,
            args_by_frame.get(str(row.get("frame_id") or ""), []),
            expected,
            target_terms,
            relation_terms,
            None,
            "",
            evidence,
        ):
            candidates.append((11.0, value, evidence, "temporal_frame_argument_binding"))
    return candidates


def _source_anchor_match_bonus(evidence: Evidence | None, target_terms: list[str] | None) -> float:
    if evidence is None or not target_terms:
        return 0.0
    source_material = normalize(evidence.rel_path)
    if not source_material:
        return 0.0
    strong_terms = [
        term
        for term in target_terms
        if term and (len(_normalized_token_set(term)) >= 2 or any(separator in term for separator in "_-/."))
    ]
    hits = sum(1 for term in strong_terms if _has_term(source_material, term))
    if hits <= 0:
        return 0.0
    # Source-local records are often named for the object they describe.  Treat a
    # path/stem anchor match as provenance, not as semantics, and cap its impact.
    return min(24.0, 14.0 + hits * 6.0)


def _choice_score(
    score: float,
    reason: str,
    expected: ExpectedAnswer,
    evidence: Evidence | None = None,
    target_terms: list[str] | None = None,
) -> float:
    if reason == "direct_label_slot_binding":
        score += 45.0
    if reason == "relation_label_value_binding":
        score += 22.0
    if reason == "frame_argument_binding":
        score += 3.0
    if reason == "relation_condition_binding" and expected.answer_type in {"content_phrase", "unknown"}:
        score += 7.0
    score += _source_anchor_match_bonus(evidence, target_terms)
    return score


CURRENT_STRUCTURED_STATE_VALUES = {
    "active",
    "current",
    "live",
    "valid",
}

STALE_STRUCTURED_STATE_VALUES = {
    "archived",
    "deprecated",
    "expired",
    "inactive",
    "obsolete",
    "old",
    "previous",
    "retired",
    "superseded",
}

STRUCTURED_STATE_SELECTORS = {"state", "status"}


def _frame_requests_explicit_structured_state(frame: QueryFrame) -> bool:
    material = normalize(
        " ".join(
            [
                frame.temporal_scope,
                frame.requested_relation,
                *frame.answer_variables,
                *frame.relation_terms,
                *frame.constraints,
                *frame.modality_requirements,
                *frame.scope_requirements,
            ]
        )
    )
    if normalize_temporal_scope(frame.temporal_scope) in {"latest", "earliest"}:
        return True
    markers = CURRENT_STRUCTURED_STATE_VALUES | STALE_STRUCTURED_STATE_VALUES
    return any(_has_term(material, marker) for marker in markers)


def _structured_row_state_rank(value: str, evidence: Evidence, records: dict[str, Any]) -> int:
    if not evidence.span_id:
        return 0
    value_norm = normalize(value)
    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records.get("relations", []):
        if str(row.get("source_span_id") or "") != evidence.span_id:
            continue
        if not _structured_source_row(row):
            continue
        metadata = _relation_metadata(row)
        group = str(metadata.get("record_group") or "")
        if not group:
            continue
        rows_by_group[group].append(row)
    ranks: list[int] = []
    for rows in rows_by_group.values():
        if value_norm and not _has_term(_group_material(rows, records, include_source_evidence=False), value_norm):
            continue
        group_rank = 0
        for row in rows:
            metadata = _relation_metadata(row)
            selector = normalize(str(row.get("predicate") or metadata.get("column_header") or ""))
            if selector not in STRUCTURED_STATE_SELECTORS:
                continue
            state_value = normalize(str(row.get("value") or row.get("object") or ""))
            if state_value in CURRENT_STRUCTURED_STATE_VALUES:
                group_rank = max(group_rank, 1)
            if state_value in STALE_STRUCTURED_STATE_VALUES:
                group_rank = min(group_rank, -1)
        if group_rank:
            ranks.append(group_rank)
    if 1 in ranks and -1 not in ranks:
        return 1
    if -1 in ranks and 1 not in ranks:
        return -1
    return 0


def _apply_structured_current_state_preference(
    candidates: list[tuple[float, str, Evidence, str]],
    records: dict[str, Any],
    frame: QueryFrame,
) -> list[tuple[float, str, Evidence, str]]:
    if not candidates or _frame_requests_explicit_structured_state(frame):
        return candidates
    ranks = [_structured_row_state_rank(value, evidence, records) for _score, value, evidence, _reason in candidates]
    if 1 not in ranks or -1 not in ranks:
        return candidates
    adjusted: list[tuple[float, str, Evidence, str]] = []
    for (score, value, evidence, reason), rank in zip(candidates, ranks):
        adjusted.append((score + rank * 18.0, value, evidence, reason))
    return adjusted


def _candidate_evidence_key(evidence: Evidence) -> tuple[str, str, int | None, str]:
    return (evidence.rel_path, evidence.span_id, evidence.chunk_order, evidence.text)


def _choose_answer(
    candidates: list[tuple[float, str, Evidence, str]],
    expected: ExpectedAnswer,
    target_terms: list[str] | None = None,
) -> Answer | None:
    scored: dict[str, dict[str, Any]] = {}
    for score, value, evidence, reason in candidates:
        canonical = canonicalize_answer(expected, value)
        if not canonical:
            continue
        score = _choice_score(score, reason, expected, evidence, target_terms)
        bucket = scored.setdefault(
            canonical,
            {"scores_by_evidence": {}, "evidence_by_key": {}, "reasons": []},
        )
        evidence_key = _candidate_evidence_key(evidence)
        previous_score = float(bucket["scores_by_evidence"].get(evidence_key, 0.0))
        if score > previous_score:
            bucket["scores_by_evidence"][evidence_key] = score
            bucket["evidence_by_key"][evidence_key] = evidence
        if reason not in bucket["reasons"]:
            bucket["reasons"].append(reason)
    if not scored:
        return None
    ordered = sorted(
        scored.items(),
        key=lambda item: (-sum(float(value) for value in item[1]["scores_by_evidence"].values()), len(item[0]), item[0]),
    )
    value, bucket = ordered[0]
    score = sum(float(value) for value in bucket["scores_by_evidence"].values())
    evidence = list(bucket["evidence_by_key"].values())[:4]
    reason = str((bucket["reasons"] or ["bounded_drs_binding"])[0])
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
        bucket = buckets.setdefault(
            canonical,
            {"score": 0.0, "scores_by_evidence": {}, "evidence": [], "evidence_by_key": {}, "reasons": set()},
        )
        evidence_key = _candidate_evidence_key(evidence)
        choice_score = _choice_score(score, reason, expected, evidence, target_terms)
        previous_score = float(bucket["scores_by_evidence"].get(evidence_key, 0.0))
        if choice_score > previous_score:
            bucket["scores_by_evidence"][evidence_key] = choice_score
            bucket["evidence_by_key"][evidence_key] = evidence
            bucket["score"] = sum(float(value) for value in bucket["scores_by_evidence"].values())
        bucket["reasons"].add(reason)
        bucket["evidence"] = list(bucket["evidence_by_key"].values())[:4]
    if len(buckets) < 2:
        return None
    ordered = sorted(buckets.items(), key=lambda item: (-float(item[1]["score"]), len(item[0]), item[0]))
    top_value, top_bucket = ordered[0]
    next_value, next_bucket = ordered[1]
    top_score = float(top_bucket["score"])
    next_score = float(next_bucket["score"])
    if top_score <= 0.0 or next_score < top_score * 0.85:
        return None
    if top_score - next_score >= max(10.0, top_score * 0.05):
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


def _has_unscoped_temporal_ambiguity(
    candidates: list[tuple[float, str, Evidence, str]],
    expected: ExpectedAnswer | None = None,
) -> bool:
    temporal_candidate_values = {
        normalize(value)
        for _score, value, _evidence, reason in candidates
        if reason in {"temporal_binding", "temporal_relation_binding"} and normalize(value)
    }
    if len(temporal_candidate_values) > 1:
        return True

    # Do not let unrelated dated evidence block a clear structural answer.
    # The guard exists to avoid guessing between competing dated states, not to
    # reject a dominant label/table/identifier candidate merely because other
    # selected chunks contain dates.
    if expected is not None:
        scored: dict[str, tuple[float, str]] = {}
        for score, value, _evidence, reason in candidates:
            canonical = canonicalize_answer(expected, value)
            if not canonical:
                continue
            choice = _choice_score(score, reason, expected, _evidence, None)
            prev = scored.get(canonical)
            if prev and prev[1] == "direct_label_slot_binding":
                merged_reason = prev[1]
            elif reason == "direct_label_slot_binding":
                merged_reason = reason
            elif prev:
                merged_reason = prev[1]
            else:
                merged_reason = reason
            scored[canonical] = (choice + (prev[0] if prev else 0.0), merged_reason)
        if scored:
            ordered = sorted(scored.items(), key=lambda item: (-item[1][0], len(item[0]), item[0]))
            top_value, (top_score, top_reason) = ordered[0]
            next_score = ordered[1][1][0] if len(ordered) > 1 else 0.0
            if top_reason == "direct_label_slot_binding" and top_score > next_score:
                return False
            if top_reason not in {"temporal_binding", "temporal_relation_binding"} and (
                len(ordered) == 1 or top_score >= max(8.0, next_score * 1.35)
            ):
                return False

    values_by_time: dict[str, set[str]] = defaultdict(set)
    for _score, value, evidence, _reason in candidates:
        match = DATE_TIME_RE.search(evidence.text)
        if match and value:
            values_by_time[match.group(0)].add(normalize(value))
    if len(values_by_time) < 2:
        return False
    distinct_values = {value for values in values_by_time.values() for value in values if value}
    return len(distinct_values) > 1



def _frame_arguments_by_condition_key(records: dict[str, Any]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    args_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arg in records.get("frame_arguments", []):
        args_by_frame[str(arg.get("frame_id") or "")].append(arg)
    values: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for frame_row in records.get("frames", []):
        key = (
            str(frame_row.get("span_id") or ""),
            normalize(str(frame_row.get("predicate") or "")),
            str(frame_row.get("context_id") or ""),
        )
        values[key].extend(args_by_frame.get(str(frame_row.get("frame_id") or ""), []))
    return values


def _model_surface_has_negative_polarity(value: str) -> bool:
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalize(value)) if token]
    return any(token in MODEL_NEGATION_TOKENS for token in tokens)


def _condition_row_is_negative(row: dict[str, Any], args: list[dict[str, Any]]) -> bool:
    if normalize(str(row.get("object") or "")) == "negative":
        return True
    model_surfaces = [
        str(row.get("predicate") or ""),
        str(row.get("value") or ""),
        *[
            str(arg.get(key) or "")
            for arg in args
            for key in ("surface", "value", "evidence_surface")
        ],
    ]
    return any(_model_surface_has_negative_polarity(surface) for surface in model_surfaces)


def _boolean_structural_terms(values: Iterable[str], target_terms: list[str]) -> list[str]:
    target = set(target_terms)
    terms: list[str] = []
    for value in values:
        for term in _compound_term_variants(value):
            if term and term not in target and term not in BOOLEAN_GENERIC_TERMS:
                terms.append(term)
        for token in content_tokens(value):
            if token and token not in target and token not in BOOLEAN_GENERIC_TERMS:
                terms.append(token)
    return list(dict.fromkeys(term for term in expand_terms(terms) if term and term not in target and term not in BOOLEAN_GENERIC_TERMS))


def _boolean_predicate_terms(frame: QueryFrame, target_terms: list[str]) -> list[str]:
    return _boolean_structural_terms([frame.requested_relation, *frame.relation_terms], target_terms)


def _boolean_constraint_terms(frame: QueryFrame, target_terms: list[str]) -> list[str]:
    return _boolean_structural_terms(frame.constraints, target_terms)


def _boolean_terms_covered(material: str, terms: list[str], *, require_all: bool) -> bool:
    if not terms:
        return True
    if require_all:
        return all(_has_term(material, term) for term in terms)
    return any(_has_term(material, term) for term in terms)


def _boolean_target_anchors_covered(material: str, frame: QueryFrame, target_terms: list[str]) -> bool:
    groups: list[list[str]] = []
    for anchor in frame.target_anchors:
        norm = normalize(anchor)
        if not norm:
            continue
        tokens = [token for token in content_tokens(norm) if token and token not in BOOLEAN_GENERIC_TERMS]
        if not tokens:
            continue
        groups.append(list(dict.fromkeys(token for token in expand_terms(tokens) if token)))
    if not groups and target_terms:
        for term in target_terms:
            tokens = [token for token in content_tokens(term) if token and token not in BOOLEAN_GENERIC_TERMS]
            if tokens:
                groups.append(list(dict.fromkeys(token for token in expand_terms(tokens) if token)))
    if not groups:
        return True
    for group in groups:
        if not all(_has_term(material, term) for term in group):
            return False
    return True


def _boolean_candidate_time_key(evidence: Evidence) -> str:
    match = DATE_TIME_RE.search(evidence.text)
    return match.group(0) if match else ""


def _select_temporal_boundary_candidates(
    candidates: list[tuple[float, str, Evidence, str]],
    frame: QueryFrame,
) -> list[tuple[float, str, Evidence, str]]:
    scope = _effective_temporal_scope(frame)
    if scope not in {"latest", "earliest"}:
        return candidates
    keyed = [(item, _boolean_candidate_time_key(item[2])) for item in candidates]
    keyed = [(item, key) for item, key in keyed if key]
    if not keyed:
        return candidates
    boundary = sorted({key for _item, key in keyed}, reverse=scope != "earliest")[0]
    return [item for item, key in keyed if key == boundary]


def _boolean_condition_candidates(
    records: dict[str, Any],
    frame: QueryFrame,
    target_terms: list[str],
    relation_terms: list[str],
) -> list[tuple[float, str, Evidence, str]]:
    args_by_key = _frame_arguments_by_condition_key(records)
    predicate_terms = _boolean_predicate_terms(frame, target_terms)
    constraint_terms = _boolean_constraint_terms(frame, target_terms)
    candidates: list[tuple[float, str, Evidence, str]] = []
    for row in records.get("relations", []):
        if str(row.get("relation_type") or "") != "drs_condition":
            continue
        if not _relation_scope_accessible(row, records, frame):
            continue
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        key = (
            str(row.get("source_span_id") or ""),
            normalize(str(row.get("predicate") or "")),
            str(row.get("context_id") or ""),
        )
        args = args_by_key.get(key, [])
        arg_material = normalize(" ".join(str(arg.get("surface") or "") for arg in args))
        row_material = normalize(
            " ".join(
                [
                    _relation_local_material(row, evidence, include_evidence=False, include_context=True, records=records),
                    arg_material,
                ]
            )
        )
        evidence_material = normalize(" ".join([row_material, evidence.rel_path, evidence.text]))
        score = _split_match_score(evidence_material, row_material, target_terms, relation_terms)
        if score <= 0:
            continue
        condition_true = not _condition_row_is_negative(row, args)
        if frame.negated:
            condition_true = not condition_true
        predicate_supported = _boolean_terms_covered(row_material, predicate_terms, require_all=False)
        target_supported = _boolean_target_anchors_covered(evidence_material, frame, target_terms)
        constraints_supported = _boolean_terms_covered(evidence_material, constraint_terms, require_all=True)
        if not target_supported:
            continue
        if condition_true and (not predicate_supported or not constraints_supported):
            continue
        prefix = "Yes" if condition_true else "No"
        evidence_text = clean_extracted_value(str(row.get("value") or "") or evidence.text).strip(" .;:")
        if not evidence_text:
            evidence_text = clean_extracted_value(evidence.text).strip(" .;:")
        answer = f"{prefix}; {evidence_text}." if evidence_text else prefix
        candidates.append((score * float(row.get("confidence") or 0.7), answer, evidence, "boolean_drs_condition_binding"))
    return _select_temporal_boundary_candidates(candidates, frame)


def _arithmetic_answer(
    records: dict[str, Any],
    frame: QueryFrame,
    expected: ExpectedAnswer,
    relation_terms: list[str],
) -> Answer | None:
    if expected.answer_type not in {"identifier", "count", "metadata_value", "unknown"}:
        return None
    material = normalize(" ".join([frame.question_text, *frame.answer_variables, *relation_terms]))
    if not any(op in material for op in (" plus ", " minus ", " times ", " multiplied by ", " divided by ")):
        return None
    match = re.search(
        r"(?:^|\s)(\d+)\s+(plus|minus|times|multiplied by|divided by)\s+(\d+)(?:\s|$)",
        material,
    )
    if not match:
        return None
    left = int(match.group(1)); op = match.group(2); right = int(match.group(3))
    if op == "plus":
        value = left + right
    elif op == "minus":
        value = left - right
    elif op in {"times", "multiplied by"}:
        value = left * right
    elif op == "divided by" and right:
        if left % right:
            return None
        value = left // right
    else:
        return None
    answer_text = str(value)
    evidence: list[Evidence] = []
    for span in records.get("source_spans", []):
        evidence_item = _evidence_for_span(str(span.get("span_id") or span.get("source_span_id") or ""), records)
        text = evidence_item.text
        text_norm = normalize(text)
        if str(left) in text_norm and str(right) in text_norm and answer_text in text_norm:
            evidence.append(evidence_item)
            break
    if not evidence:
        return None
    return Answer(answer_text, 0.9, evidence[:1], "deterministic arithmetic binding", expected.answer_type)


def _person_values_from_relation_text(value: str, expected: ExpectedAnswer) -> list[str]:
    if expected.answer_type not in {"person", "actor", "organization"}:
        return []
    values: list[str] = []
    for phrase in capitalized_phrases(value):
        if normalize(phrase) in UNRESOLVED_PRONOUN_ANSWER_VALUES:
            continue
        canonical = canonicalize_answer(expected, phrase)
        if canonical and canonical not in values:
            values.append(canonical)
    return values


def _direct_label_slot_candidates(
    records: dict[str, Any],
    frame: QueryFrame,
    expected: ExpectedAnswer,
    relation_terms: list[str],
    target_terms: list[str],
) -> list[tuple[float, str, Evidence, str]]:
    answer_slot_terms = _answer_slot_terms(frame, target_terms)
    if not answer_slot_terms:
        return []
    slot_material = normalize(" ".join(answer_slot_terms))
    candidates: list[tuple[float, str, Evidence, str]] = []
    for row in records.get("relations", []):
        if str(row.get("relation_type") or "") != "label_value":
            continue
        if not _relation_scope_accessible(row, records, frame):
            continue
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        if target_terms:
            material = _relation_local_material(row, evidence, include_evidence=True, include_context=True, records=records)
            if not _contains_any(material, target_terms):
                continue
        metadata = _relation_metadata(row)
        label_material = normalize(" ".join([str(row.get("subject") or ""), str(metadata.get("section_anchor") or "")]))
        if not _answer_slot_label_matches(label_material, answer_slot_terms, target_terms):
            continue
        value = str(row.get("value") or row.get("object") or "")
        values = _person_values_from_relation_text(value, expected) or _compatible_values(expected, [value])
        structural = expected.answer_type in {"url", "identifier", "file_path", "date_time", "count"}
        values = [
            item for item in values
            if structural
            or (
                not _value_is_target(item, target_terms)
                and not _rejects_bound_target_value(expected, item, target_terms)
                and not _value_is_target(item, relation_terms)
            )
        ]
        for item in values:
            slot_bonus = (
                _identifier_slot_alignment_bonus(item, evidence, frame, target_terms)
                if expected.answer_type == "identifier"
                else 0.0
            )
            candidates.append((7.5 + slot_bonus, item, evidence, "direct_label_slot_binding"))
    return candidates


def _relation_label_value_candidates(
    records: dict[str, Any],
    frame: QueryFrame,
    expected: ExpectedAnswer,
    relation_terms: list[str],
    target_terms: list[str],
) -> list[tuple[float, str, Evidence, str]]:
    candidates: list[tuple[float, str, Evidence, str]] = []
    answer_slot_terms = _answer_slot_terms(frame, target_terms)
    generic = {"is", "are", "was", "were", "answer", "argument", "who", "what", "which", "where"}
    relation_signal = [term for term in relation_terms if normalize(term) not in generic]
    strong_signal = [*content_tokens(frame.requested_relation)]
    for anchor in frame.target_anchors:
        anchor_terms = content_tokens(anchor)
        if 1 <= len(anchor_terms) <= 2:
            strong_signal.extend(anchor_terms)
    strong_signal = list(dict.fromkeys([term for term in strong_signal if term and term not in generic]))
    if not relation_signal:
        return candidates
    for row in records.get("relations", []):
        if str(row.get("relation_type") or "") != "label_value":
            continue
        if not _relation_scope_accessible(row, records, frame):
            continue
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        if answer_slot_terms and _answer_slot_constraints(answer_slot_terms, target_terms):
            label_material = _row_slot_label_material(row)
            if label_material and not _answer_slot_label_matches(label_material, answer_slot_terms, target_terms):
                continue
        material = _relation_selector_material(row, evidence, include_evidence=True)
        if target_terms and not _contains_any(material, target_terms):
            continue
        if strong_signal and not _contains_any(material, strong_signal):
            continue
        if not _contains_any(material, relation_signal):
            continue
        value = str(row.get("value") or row.get("object") or "")
        values = _person_values_from_relation_text(value, expected) or _compatible_values(expected, [value])
        structural = expected.answer_type in {"url", "identifier", "file_path", "date_time", "count"}
        values = [
            item for item in values
            if structural
            or (
                not _value_is_target(item, target_terms)
                and not _rejects_bound_target_value(expected, item, target_terms)
                and not _value_is_target(item, relation_signal)
            )
        ]
        for item in values:
            slot_bonus = (
                _identifier_slot_alignment_bonus(item, evidence, frame, target_terms)
                if expected.answer_type == "identifier"
                else 0.0
            )
            candidates.append((6.8 + slot_bonus, item, evidence, "relation_label_value_binding"))
    return candidates

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
    answer_slot_terms = _answer_slot_terms(frame, target_terms)
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
        boolean_relation_terms = _relation_terms(frame, question)
        boolean_candidates = _boolean_condition_candidates(records, frame, target_terms, boolean_relation_terms)
        answer = _with_supporting_evidence(_choose_answer(boolean_candidates, expected, target_terms), identity_expansion_evidence)
        if answer is not None:
            _attach_answer_provenance(diagnostics, records, answer)
            return answer, diagnostics
        _attach_no_answer_provenance(
            diagnostics,
            records,
            target_terms,
            boolean_relation_terms,
            boolean_candidates,
            expected,
            "boolean_not_bound_by_bounded_executor",
        )
        return None, diagnostics

    arithmetic_answer = _arithmetic_answer(records, frame, expected, relation_terms)
    if arithmetic_answer is not None:
        _attach_answer_provenance(diagnostics, records, arithmetic_answer)
        return arithmetic_answer, diagnostics

    candidates: list[tuple[float, str, Evidence, str]] = []
    candidates.extend(_direct_label_slot_candidates(records, frame, expected, relation_terms, target_terms))
    candidates.extend(_relation_label_value_candidates(records, frame, expected, relation_terms, target_terms))
    candidates.extend(_bind_record_groups(records, frame, expected, target_terms, relation_terms))
    candidates.extend(_structural_chain_candidates(records, frame, expected, target_terms))
    candidates.extend(_bind_frame_conditions(records, frame, expected, target_terms, relation_terms))
    candidates.extend(_bind_relation_conditions(records, frame, expected, target_terms, relation_terms))
    candidates.extend(_bind_document_scoped_label_values(records, frame, expected, target_terms, relation_terms))
    candidates.extend(_document_scoped_structural_row_candidates(records, frame, expected, target_terms, relation_terms))
    candidates.extend(_document_scoped_relation_value_candidates(records, frame, expected, target_terms, relation_terms))
    temporal_candidates = _temporal_candidates(records, frame, expected, target_terms, relation_terms)
    temporal_candidates.extend(_temporal_relation_candidates(records, frame, expected, target_terms, relation_terms))
    temporal_candidates.extend(_temporal_frame_argument_candidates(records, frame, expected, target_terms, relation_terms))
    effective_temporal_scope = _effective_temporal_scope(frame)
    if temporal_candidates and effective_temporal_scope in {"latest", "earliest"}:
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
        answer = _with_supporting_evidence(_choose_answer(temporal_candidates, expected, target_terms), identity_expansion_evidence)
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
    candidates = _apply_structured_current_state_preference(candidates, records, frame)

    if expected.answer_type == "count" and frame.aggregation == "count":
        group_count, group_evidence = _count_matching_record_groups(records, frame, target_terms, relation_terms)
        if group_count:
            answer = Answer(str(group_count), 0.86, group_evidence, "record-group aggregation DRS binding", "count")
            answer = _with_supporting_evidence(answer, identity_expansion_evidence)
            _attach_answer_provenance(diagnostics, records, answer)
            return answer, diagnostics
    if expected.answer_type == "count" and frame.aggregation == "count" and candidates:
        # Count over provenance-bearing bindings, not over raw extracted numeric
        # values.  Counting the values themselves caused row-count questions to
        # return unrelated numbers found in the corpus.
        provenance_keys: list[str] = []
        evidence: list[Evidence] = []
        for _score, _value, item_evidence, _reason in candidates:
            key = "|".join([item_evidence.rel_path, item_evidence.text])
            if key not in provenance_keys:
                provenance_keys.append(key)
                evidence.append(item_evidence)
        if provenance_keys:
            answer = Answer(str(len(provenance_keys)), 0.82, evidence[:4], "binding-provenance aggregation DRS binding", "count")
            answer = _with_supporting_evidence(answer, identity_expansion_evidence)
            _attach_answer_provenance(diagnostics, records, answer)
            return answer, diagnostics

    if not frame.temporal_scope and _has_unscoped_temporal_ambiguity(candidates, expected):
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

    answer = _with_supporting_evidence(_choose_answer(candidates, expected, target_terms), identity_expansion_evidence)
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
