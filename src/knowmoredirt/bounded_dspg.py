"""Bounded SQLite DSPG subgraph retrieval and execution.

The query planner produces a small generic plan. This module ranks source
documents and chunks, loads only the selected SQLite graph records, and returns
answers grounded in raw source spans. It does not depend on external input
schemas, external labels, or corpus-specific names.
"""

from __future__ import annotations

import re
from typing import Any

from .answer_types import ExpectedAnswer, canonicalize_answer, infer_expected_answer
from .extractors import identifiers, urls
from .model_planner import REFERENCE_PATTERNS, visible_named_anchors
from .models import Answer, Document, Evidence, Sentence
from .text import clean_extracted_value, content_tokens, is_low_semantic_noise, normalize


def _clean(value: str) -> str:
    return clean_extracted_value(value).strip(" .;:")


def _query_terms(text: str) -> list[str]:
    stop = {
        "what", "which", "where", "when", "find", "provide", "show", "give",
        "the", "and", "that", "this", "with", "from", "into", "only", "name",
        "names", "source", "document", "reference", "id", "ids",
        "identifier", "identifiers",
    }
    terms: list[str] = []
    for token in content_tokens(text):
        candidates = [token, *re.split(r"[-_]", token)]
        for candidate in candidates:
            if len(candidate) > 2 and candidate not in stop and candidate not in terms:
                terms.append(candidate)
    return terms


def _target_terms(plan: dict[str, Any], question: str) -> list[str]:
    target = str(plan.get("target_surface") or "")
    terms = _query_terms(target)
    if terms and all(term in {"id", "ids", "identifier", "identifiers", "reference", "references"} for term in terms):
        terms = [term for term in _query_terms(question) if term not in {"id", "ids", "identifier", "identifiers", "reference", "references"}]
    target_has_specific_anchor = bool(visible_named_anchors(target))
    question_anchors = visible_named_anchors(question)
    if question_anchors and (not terms or not target_has_specific_anchor or any(term in {"id", "ids", "identifier", "identifiers"} for term in terms)):
        terms = _query_terms(" ".join(question_anchors[:3]))
    for pattern in REFERENCE_PATTERNS:
        terms.extend(normalize(match.group(0)) for match in re.finditer(pattern, question, re.I))
        terms.extend(normalize(match.group(0)) for match in re.finditer(pattern, target, re.I))
    if not terms:
        terms = _query_terms(question)
    return list(dict.fromkeys(term for term in terms if term))


def _intent_cues(plan: dict[str, Any]) -> set[str]:
    role = normalize(str(plan.get("answer_role") or ""))
    intent = normalize(str(plan.get("intent") or ""))
    cues: set[str] = set()
    if role in {"author", "actor"}:
        cues.update({"author", "authored", "wrote", "drafted", "created", "compiled"})
    if role == "reviewer":
        cues.update({"review", "reviewed", "reviewer", "checked", "inspected"})
    if role == "approver":
        cues.update({"approve", "approved", "approver", "signed"})
    if role == "owner":
        cues.update({"owner", "owns", "owned", "responsible", "contact"})
    if role in {"reporter", "organization"}:
        cues.update({"reported", "requested", "raised", "flagged", "escalated", "claimed", "alleged", "account"})
    if intent in {"reference_lookup", "url_lookup", "file_lookup"}:
        cues.update({"id", "identifier", "reference", "url", "link", "file", "case"})
    if intent == "state_lookup":
        cues.update({"state", "status", "final", "current", "closed", "open", "fixed", "resolved"})
    if intent == "context_lookup":
        cues.update({"asserted", "reported", "quoted", "alleged", "valid", "effective", "measured", "date"})
    return cues


def _document_text(document: Document, sentences: list[Sentence]) -> str:
    metadata = " ".join(
        str(value)
        for value in [
            document.metadata.get("file_name", ""),
            document.metadata.get("stem", ""),
            document.metadata.get("suffix", ""),
            document.metadata.get("parent_rel_path", ""),
            " ".join(str(part) for part in document.metadata.get("path_parts", [])),
        ]
    )
    return normalize(f"{metadata} {' '.join(sentence.text for sentence in sentences[:40])}")


def _answer_type_bonus(expected: ExpectedAnswer, sentence_text: str) -> int:
    if expected.answer_type == "url":
        return 18 if urls(sentence_text) else 0
    if expected.answer_type == "identifier":
        return 12 if identifiers(sentence_text) else 0
    if expected.answer_type == "file_path":
        return 12 if any("." in item for item in identifiers(sentence_text)) else 0
    if expected.answer_type in {"person", "actor", "organization"}:
        return 6 if re.search(r"\b[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3}\b", sentence_text) else 0
    if expected.answer_type == "date_time":
        return 10 if re.search(r"\b\d{4}-\d{2}-\d{2}|\b\d{1,2}:\d{2}\b", sentence_text) else 0
    if expected.answer_type == "count":
        return 8 if re.search(r"\b\d+\b", sentence_text) else 0
    if expected.answer_type == "boolean":
        return 8 if re.search(r"\b(?:yes|no|not|no proof|unknown|unsupported|confirmed|denied)\b", normalize(sentence_text)) else 0
    return 0


def _rank_scope(
    documents: list[Document],
    sentences_by_document: dict[str, dict[int, Sentence]],
    question: str,
    plan: dict[str, Any],
    doc_limit: int,
    chunk_limit: int,
) -> tuple[list[str], list[tuple[str, int]], dict[str, Any]]:
    expected = infer_expected_answer(question)
    target_terms = _target_terms(plan, question)
    q_terms = _query_terms(question)
    cues = _intent_cues(plan)
    doc_scores: list[tuple[int, str, str]] = []
    for document in documents:
        search = _document_text(document, list(sentences_by_document.get(document.rel_path, {}).values()))
        target_hits = sum(1 for term in target_terms if term in search)
        q_hits = sum(1 for term in q_terms if term in search)
        if target_terms and not target_hits:
            continue
        score = target_hits * 12 + q_hits
        if is_low_semantic_noise(" ".join(sentence.text for sentence in sentences_by_document.get(document.rel_path, {}).values())) and expected.answer_type != "metadata_value":
            score = int(score * 0.2)
        if score:
            doc_scores.append((score, document.document_id, document.rel_path))
    doc_scores.sort(key=lambda item: (-item[0], item[2]))
    selected_docs = [document_id for _score, document_id, _rel_path in doc_scores[:doc_limit]]
    selected_set = set(selected_docs)
    chunk_scores: list[tuple[int, str, int, str]] = []
    fallback_scores: list[tuple[int, str, int, str]] = []
    for document in documents:
        if document.document_id not in selected_set:
            continue
        ordered = sentences_by_document.get(document.rel_path, {})
        doc_search = _document_text(document, list(ordered.values()))
        doc_has_target = any(term in doc_search for term in target_terms)
        for order, sentence in ordered.items():
            low = normalize(sentence.text)
            score = sum(20 for term in target_terms if term in low)
            score += sum(3 for term in q_terms if term in low)
            score += sum(8 for cue in cues if cue and cue in low)
            score += _answer_type_bonus(expected, sentence.text)
            if doc_has_target and any(cue and cue in low for cue in cues):
                score += 12
            if doc_has_target and (identifiers(sentence.text) or urls(sentence.text)):
                score += 4
            if is_low_semantic_noise(sentence.text) and expected.answer_type != "metadata_value":
                score = int(score * 0.15)
            if score:
                chunk_scores.append((score, document.document_id, order, document.rel_path))
            elif doc_has_target and re.search(r"\b[A-Za-z][A-Za-z0-9 _/-]{1,50}\s*[:=]", sentence.text):
                fallback_scores.append((1, document.document_id, order, document.rel_path))
    chunk_scores.sort(key=lambda item: (-item[0], item[3], item[2]))
    fallback_scores.sort(key=lambda item: (item[3], item[2]))
    selected: list[tuple[str, int]] = []
    per_doc: dict[str, int] = {}
    for _score, document_id, order, _rel_path in [*chunk_scores, *fallback_scores]:
        if len(selected) >= chunk_limit:
            break
        for nearby in (order - 1, order, order + 1):
            if nearby < 0 or per_doc.get(document_id, 0) >= 18:
                continue
            key = (document_id, nearby)
            if key not in selected:
                selected.append(key)
                per_doc[document_id] = per_doc.get(document_id, 0) + 1
    return selected_docs, selected[:chunk_limit], {
        "candidate_document_rows": len(doc_scores),
        "selected_document_count": len(selected_docs),
        "candidate_chunk_rows": len(chunk_scores),
        "selected_chunk_count": len(selected[:chunk_limit]),
        "target_terms": target_terms[:24],
        "intent_cues": sorted(cues),
    }


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


def _fetch_chunks(connection: Any, chunk_keys: list[tuple[str, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(0, len(chunk_keys), 120):
        group = chunk_keys[index:index + 120]
        clauses = " OR ".join("(document_id=? AND chunk_order=?)" for _ in group)
        params: list[Any] = []
        for document_id, order in group:
            params.extend([document_id, int(order)])
        if clauses:
            rows.extend(dict(row) for row in connection.execute(f"SELECT * FROM chunks WHERE {clauses}", params))
    return rows


def _load_records(store: Any, run_id: str, document_ids: list[str], chunk_keys: list[tuple[str, int]]) -> dict[str, Any]:
    connection = store.connection
    documents = _fetch_by_ids(connection, "documents", "document_id", document_ids)
    chunks = _fetch_chunks(connection, chunk_keys)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    spans = _fetch_by_ids(connection, "source_spans", "chunk_id", chunk_ids)
    span_ids = [span["span_id"] for span in spans]
    mentions = _fetch_by_ids(connection, "mentions", "span_id", span_ids)
    refs = _fetch_by_ids(connection, "mention_referents", "mention_id", [row["mention_id"] for row in mentions])
    referents = _fetch_by_ids(connection, "referents", "referent_id", [row["referent_id"] for row in refs])
    frames = _fetch_by_ids(connection, "frames", "span_id", span_ids)
    args = _fetch_by_ids(connection, "frame_arguments", "frame_id", [frame["frame_id"] for frame in frames])
    temporal = _fetch_by_ids(connection, "temporal_edges", "source_span_id", span_ids)
    relations = _fetch_by_ids(connection, "relations", "source_span_id", span_ids)
    contexts = [dict(row) for row in connection.execute("SELECT * FROM contexts WHERE run_id=?", (run_id,))]
    context_carriers = _fetch_by_ids(connection, "context_carriers", "document_id", document_ids)
    metadata_records = _fetch_by_ids(connection, "metadata_records", "document_id", document_ids)
    return {
        "documents": documents,
        "chunks": chunks,
        "source_spans": spans,
        "mentions": mentions,
        "mention_referents": refs,
        "referents": referents,
        "contexts": contexts,
        "frames": frames,
        "frame_arguments": args,
        "temporal_edges": temporal,
        "relations": relations,
        "context_carriers": context_carriers,
        "metadata_records": metadata_records,
        "record_counts": {
            "documents": len(documents), "chunks": len(chunks), "source_spans": len(spans),
            "mentions": len(mentions), "referents": len(referents), "frames": len(frames),
            "frame_arguments": len(args), "temporal_edges": len(temporal), "relations": len(relations),
            "context_carriers": len(context_carriers), "metadata_records": len(metadata_records),
        },
    }


def _evidence(span: dict[str, Any], chunks_by_id: dict[str, dict[str, Any]], docs_by_id: dict[str, dict[str, Any]]) -> Evidence:
    chunk = chunks_by_id.get(str(span.get("chunk_id")), {})
    doc = docs_by_id.get(str(span.get("document_id")), {})
    return Evidence(str(doc.get("rel_path") or doc.get("path") or ""), str(chunk.get("text") or span.get("surface") or ""), 0.75)


def _metadata_evidence(record: dict[str, Any], docs_by_id: dict[str, dict[str, Any]]) -> Evidence:
    doc = docs_by_id.get(str(record.get("document_id")), {})
    key = str(record.get("key") or "metadata")
    value = str(record.get("value") or "")
    return Evidence(str(doc.get("rel_path") or doc.get("path") or ""), f"metadata {key}: {value}", 0.7)


def _add(values: list[str], evidence: list[Evidence], value: str, ev: Evidence) -> None:
    value = _clean(value)
    if value and normalize(value) != "unknown" and value not in values:
        values.append(value)
        if ev.rel_path and ev.text:
            evidence.append(ev)


def _execute(records: dict[str, Any], plan: dict[str, Any]) -> tuple[list[str], list[Evidence], dict[str, Any]]:
    intent = normalize(str(plan.get("intent") or ""))
    role = normalize(str(plan.get("answer_role") or ""))
    terms = _target_terms(plan, str(plan.get("query_text") or ""))
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in records["chunks"]}
    docs_by_id = {str(document["document_id"]): document for document in records["documents"]}
    spans_by_id = {str(span["span_id"]): span for span in records["source_spans"]}
    contexts_by_id = {str(context["context_id"]): context for context in records["contexts"]}
    query_text = normalize(str(plan.get("query_text") or ""))
    values: list[str] = []
    evidence: list[Evidence] = []
    if intent == "context_lookup":
        metadata_key_by_token = {
            "size": "size_bytes",
            "hash": "content_hash",
            "suffix": "suffix",
            "extension": "suffix",
            "name": "file_name",
            "lines": "line_count",
            "line": "line_count",
            "words": "word_count",
            "encoding": "encoding",
        }
        wanted_keys = {key for token, key in metadata_key_by_token.items() if token in query_text}
        if wanted_keys:
            for record in records.get("metadata_records", []):
                if str(record.get("key") or "") not in wanted_keys:
                    continue
                document_material = normalize(str(docs_by_id.get(str(record.get("document_id")), {}).get("rel_path", "")))
                if terms and not any(term in document_material for term in terms):
                    continue
                _add(values, evidence, str(record.get("value") or ""), _metadata_evidence(record, docs_by_id))
            if values:
                return values[:8], evidence[:8], {"record_counts": records.get("record_counts", {})}
        wants_time = any(token in query_text for token in ["modified", "created", "time", "date", "valid", "effective", "measured"])
        if wants_time:
            for carrier in records.get("context_carriers", []):
                temporal_value = str(carrier.get("temporal_value") or "")
                if not temporal_value:
                    continue
                material = normalize(" ".join(str(carrier.get(key) or "") for key in ["carrier_kind", "carrier_surface", "temporal_value_type", "temporal_value"]))
                document_material = normalize(str(docs_by_id.get(str(carrier.get("document_id")), {}).get("rel_path", "")))
                if terms and not any(term in material or term in document_material for term in terms):
                    continue
                ev = Evidence(
                    str(docs_by_id.get(str(carrier.get("document_id")), {}).get("rel_path") or ""),
                    f"context {carrier.get('temporal_value_type')}: {temporal_value}",
                    0.72,
                )
                _add(values, evidence, temporal_value, ev)
            if values:
                return values[:8], evidence[:8], {"record_counts": records.get("record_counts", {})}
        for carrier in records.get("context_carriers", []):
            context = contexts_by_id.get(str(carrier.get("context_id")), {})
            kind = str(context.get("kind") or carrier.get("carrier_surface") or carrier.get("carrier_kind") or "")
            if not kind:
                continue
            span = spans_by_id.get(str(carrier.get("source_span_id")), {})
            ev = _evidence(span, chunks_by_id, docs_by_id)
            material = normalize(" ".join([kind, ev.text, str(carrier.get("carrier_surface") or "")]))
            if terms and not any(term in material for term in terms):
                continue
            _add(values, evidence, kind, ev)
        if values:
            return values[:8], evidence[:8], {"record_counts": records.get("record_counts", {})}
    for relation in records["relations"]:
        span = spans_by_id.get(str(relation.get("source_span_id")), {})
        ev = _evidence(span, chunks_by_id, docs_by_id)
        doc_material = normalize(str(docs_by_id.get(str(span.get("document_id")), {}).get("rel_path", "")))
        material = normalize(" ".join(str(relation.get(key) or "") for key in ["relation_type", "subject", "predicate", "object", "value"]) + " " + ev.text + " " + doc_material)
        has_target = any(term in material for term in terms)
        if terms and not has_target:
            continue
        predicate = normalize(str(relation.get("predicate") or ""))
        rel_type = normalize(str(relation.get("relation_type") or ""))
        if intent == "role_lookup":
            candidate = ""
            if rel_type in {"label_value", "record_value"} and role in normalize(str(relation.get("subject") or "")):
                candidate = str(relation.get("value") or "")
            elif role == "owner" and predicate in {"own", "owner", "manage", "responsible"}:
                candidate = str(relation.get("subject") or relation.get("value") or "")
            elif role == "reviewer" and predicate in {"review", "inspect", "check"}:
                candidate = str(relation.get("subject") or "")
            elif role == "approver" and predicate in {"approve", "sign"}:
                candidate = str(relation.get("subject") or "")
            elif role in {"author", "actor"} and predicate in {"author", "write", "draft", "create", "prepare"}:
                candidate = str(relation.get("subject") or "")
            elif role in {"reporter", "organization"} and predicate in {"report", "request", "allege", "claim", "state"}:
                candidate = str(relation.get("subject") or relation.get("value") or "")
            if candidate:
                _add(values, evidence, candidate, ev)
        elif intent in {"reference_lookup", "url_lookup", "file_lookup"}:
            candidates = urls(ev.text) if intent == "url_lookup" else identifiers(ev.text)
            if intent == "file_lookup":
                candidates = [item for item in identifiers(ev.text) if "." in item]
            for candidate in candidates:
                _add(values, evidence, candidate, ev)
        elif intent == "state_lookup":
            if rel_type == "status":
                _add(values, evidence, str(relation.get("value") or relation.get("predicate") or ""), ev)
        elif intent == "context_lookup":
            if rel_type in {"status", "temporal"}:
                _add(values, evidence, str(relation.get("value") or relation.get("predicate") or ""), ev)
    if intent == "state_lookup":
        for edge in records["temporal_edges"]:
            span = spans_by_id.get(str(edge.get("source_span_id")), {})
            ev = _evidence(span, chunks_by_id, docs_by_id)
            _add(values, evidence, str(edge.get("state_value") or edge.get("temporal_value") or ""), ev)
    if intent == "identity_lookup":
        for relation in records["relations"]:
            if normalize(str(relation.get("relation_type") or "")) != "identity":
                continue
            span = spans_by_id.get(str(relation.get("source_span_id")), {})
            ev = _evidence(span, chunks_by_id, docs_by_id)
            material = normalize(" ".join(str(relation.get(key) or "") for key in ["subject", "object", "value"]) + " " + ev.text)
            if terms and not any(term in material for term in terms):
                continue
            _add(values, evidence, str(relation.get("value") or relation.get("object") or "same"), ev)
    if intent == "grouped_search":
        seen_paths: set[str] = set()
        for chunk in records["chunks"][:24]:
            doc = docs_by_id.get(str(chunk.get("document_id")), {})
            rel_path = str(doc.get("rel_path") or "")
            if not rel_path or rel_path in seen_paths:
                continue
            material = normalize(" ".join([rel_path, str(chunk.get("text") or "")]))
            if terms and not any(term in material for term in terms):
                continue
            seen_paths.add(rel_path)
            _add(values, evidence, rel_path, Evidence(rel_path, str(chunk.get("text") or ""), 0.62))
    if intent == "role_lookup" and not values:
        desired: set[str] = set()
        if role == "reviewer":
            desired.update({"review", "inspect", "check", "test"})
        elif role == "approver":
            desired.update({"approve", "sign"})
        elif role == "owner":
            desired.update({"own", "manage"})
        elif role in {"author", "actor"}:
            desired.update({"author", "write", "draft", "create", "compile", "open", "fix", "merge"})
        elif role in {"reporter", "organization"}:
            desired.update({"report", "request", "allege", "claim", "state"})
        arguments_by_frame: dict[str, list[dict[str, Any]]] = {}
        mentions_by_id = {str(mention.get("mention_id")): mention for mention in records["mentions"]}
        for argument in records["frame_arguments"]:
            arguments_by_frame.setdefault(str(argument.get("frame_id")), []).append(argument)
        for frame in records["frames"]:
            predicate = normalize(str(frame.get("predicate_norm") or frame.get("predicate") or ""))
            if desired and predicate not in desired:
                continue
            span = spans_by_id.get(str(frame.get("span_id")), {})
            ev = _evidence(span, chunks_by_id, docs_by_id)
            material = normalize(" ".join([predicate, ev.text]))
            if terms and not any(term in material for term in terms):
                continue
            for argument in arguments_by_frame.get(str(frame.get("frame_id")), []):
                if normalize(str(argument.get("role") or "")) not in {"agent", "author", "reviewer", "approver", "speaker", "owner", "assignee", "requester"}:
                    continue
                mention = mentions_by_id.get(str(argument.get("mention_id") or ""))
                surface = str(argument.get("surface") or (mention or {}).get("surface") or "")
                if surface:
                    _add(values, evidence, surface, ev)
    return values[:8], evidence[:8], {"record_counts": records.get("record_counts", {})}


def execute_bounded_query(
    store: Any,
    run_id: str,
    documents: list[Document],
    sentences_by_document: dict[str, dict[int, Sentence]],
    question: str,
    plan: dict[str, Any],
    *,
    doc_limit: int = 40,
    chunk_limit: int = 240,
) -> tuple[Answer | None, dict[str, Any]]:
    enriched_plan = {**plan, "query_text": question}
    expected = infer_expected_answer(question)
    document_ids, chunk_keys, ranking = _rank_scope(documents, sentences_by_document, question, enriched_plan, doc_limit, chunk_limit)
    records = _load_records(store, run_id, document_ids, chunk_keys)
    values, evidence, execution = _execute(records, enriched_plan)
    diagnostics = {"ranking": ranking, "execution": execution}
    if not values or not evidence:
        return None, diagnostics
    answer_text = canonicalize_answer(expected, "; ".join(values))
    if not answer_text:
        diagnostics["execution"] = {**execution, "rejected_reason": "answer_type_incompatible", "expected_answer_type": expected.answer_type}
        return None, diagnostics
    return Answer(answer_text, 0.78, evidence, "bounded DSPG subgraph execution", expected.answer_type), diagnostics
