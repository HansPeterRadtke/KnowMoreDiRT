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
from typing import Any

from .answer_types import ExpectedAnswer, canonicalize_answer, infer_expected_answer, is_value_compatible
from .extractors import identifiers, urls
from .models import Answer, Document, Evidence, Sentence
from .query import QueryFrame, expand_terms, frame_from_mapping, plan_question
from .text import clean_extracted_value, content_tokens, is_low_semantic_noise, normalize

DATE_TIME_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2})?|\d{1,2}:\d{2})\b")
PATH_RE = re.compile(r"\b[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+\b|\b[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12}\b")
NEGATION_RE = re.compile(r"\b(?:no|not|never|without|denied|unsupported)\b", re.I)
LOW_VALIDITY_RE = re.compile(r"\b(?:archived|obsolete|superseded|stale|retired)\b", re.I)
INACCESSIBLE_CONTEXT_PREFIXES = ("modality:dream", "modality:fiction", "modality:hypothetical", "quality:")


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
        values.extend(_query_terms(anchor))
    return list(dict.fromkeys(values))


def _relation_terms(frame: QueryFrame, question: str) -> list[str]:
    target = set(_target_terms(frame, question))
    terms = list(frame.relation_terms) + _query_terms(frame.requested_relation) + list(frame.constraints)
    return list(dict.fromkeys(term for term in expand_terms(terms) if term and term not in target))


def _has_term(material: str, term: str) -> bool:
    if not term:
        return False
    if term in material:
        return True
    parts = re.split(r"[^a-z0-9]+", material)
    if term in parts:
        return True
    if len(term) >= 3 and any(part.startswith(term) or term.startswith(part) for part in parts if len(part) >= 3):
        return True
    if len(term) <= 4 and re.fullmatch(r"[a-z0-9_-]+", term):
        return re.search(rf"\b{re.escape(term)}\b", material) is not None
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
    parts = set(re.split(r"[/_.-]+", normalize(rel_path)))
    return bool(parts.intersection({"cache", "lock", "tmp", "temp", "transport", "hidden"})) or is_low_semantic_noise(text)


def _asks_about_source_structure(question: str) -> bool:
    q = normalize(question)
    return any(term in q for term in ["cache", "lock", "temporary", "metadata", "file", "folder", "path"])


def _rank_scope(
    documents: list[Document],
    sentences_by_document: dict[str, dict[int, Sentence]],
    question: str,
    frame: QueryFrame,
    doc_limit: int,
    chunk_limit: int,
) -> tuple[list[str], list[tuple[str, int]], dict[str, Any]]:
    target_terms = _target_terms(frame, question)
    relation_terms = _relation_terms(frame, question)
    all_terms = _query_terms(question)
    doc_scores: list[tuple[float, str, str]] = []
    for document in documents:
        sentences = list(sentences_by_document.get(document.rel_path, {}).values())
        material = _document_material(document, sentences)
        target_hits = sum(1 for term in target_terms if _has_term(material, term))
        relation_hits = sum(1 for term in relation_terms if _has_term(material, term))
        lexical_hits = sum(1 for term in all_terms if _has_term(material, term))
        if target_terms and not target_hits:
            continue
        score = target_hits * 16 + relation_hits * 8 + lexical_hits
        if _source_is_low_priority(document.rel_path, " ".join(sentence.text for sentence in sentences)) and not _asks_about_source_structure(question):
            score *= 0.2
        if score:
            doc_scores.append((score, document.document_id, document.rel_path))
    doc_scores.sort(key=lambda item: (-item[0], item[2]))
    selected_docs = [doc_id for _score, doc_id, _rel_path in doc_scores[:doc_limit]]
    selected_set = set(selected_docs)
    chunk_scores: list[tuple[float, str, int, str]] = []
    for document in documents:
        if document.document_id not in selected_set:
            continue
        ordered = sentences_by_document.get(document.rel_path, {})
        document_has_target = any(_has_term(_document_material(document, list(ordered.values())), term) for term in target_terms)
        for order, sentence in ordered.items():
            material = normalize(sentence.text)
            score = sum(22 for term in target_terms if _has_term(material, term))
            score += sum(11 for term in relation_terms if _has_term(material, term))
            score += sum(2 for term in all_terms if _has_term(material, term))
            if document_has_target and relation_terms and _contains_any(material, relation_terms):
                score += 12
            if _source_is_low_priority(sentence.rel_path, sentence.text) and not _asks_about_source_structure(question):
                score *= 0.15
            if score:
                chunk_scores.append((score, document.document_id, order, document.rel_path))
    chunk_scores.sort(key=lambda item: (-item[0], item[3], item[2]))
    selected_chunks: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for _score, document_id, order, _rel_path in chunk_scores:
        if len(selected_chunks) >= chunk_limit:
            break
        for nearby in (order - 2, order - 1, order, order + 1, order + 2):
            if nearby < 0:
                continue
            key = (document_id, nearby)
            if key not in seen:
                seen.add(key)
                selected_chunks.append(key)
                if len(selected_chunks) >= chunk_limit:
                    break
    return selected_docs, selected_chunks, {
        "candidate_document_rows": len(doc_scores),
        "selected_document_count": len(selected_docs),
        "candidate_chunk_rows": len(chunk_scores),
        "selected_chunk_count": len(selected_chunks),
        "target_terms": target_terms[:32],
        "relation_terms": relation_terms[:32],
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
    frames = _fetch_by_ids(connection, "frames", "span_id", span_ids)
    arguments = _fetch_by_ids(connection, "frame_arguments", "frame_id", [frame["frame_id"] for frame in frames])
    relations = _fetch_by_ids(connection, "relations", "source_span_id", span_ids)
    temporal = _fetch_by_ids(connection, "temporal_edges", "source_span_id", span_ids)
    metadata_records = _fetch_by_ids(connection, "metadata_records", "document_id", document_ids)
    contexts = [dict(row) for row in connection.execute("SELECT * FROM contexts WHERE run_id=?", (run_id,))]
    context_carriers = _fetch_by_ids(connection, "context_carriers", "document_id", document_ids)
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
            "contexts": len(contexts),
            "context_carriers": len(context_carriers),
        },
    }


def _docs_by_id(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("document_id")): row for row in records.get("documents", [])}


def _chunks_by_id(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("chunk_id")): row for row in records.get("chunks", [])}


def _spans_by_id(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("span_id")): row for row in records.get("source_spans", [])}


def _contexts_by_id(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("context_id")): row for row in records.get("contexts", [])}


def _evidence_for_span(span_id: str, records: dict[str, Any]) -> Evidence:
    span = _spans_by_id(records).get(span_id, {})
    chunk = _chunks_by_id(records).get(str(span.get("chunk_id")), {})
    doc = _docs_by_id(records).get(str(span.get("document_id")), {})
    return Evidence(str(doc.get("rel_path") or ""), str(chunk.get("text") or span.get("surface") or ""), 0.78)


def _metadata_evidence(record: dict[str, Any], records: dict[str, Any]) -> Evidence:
    doc = _docs_by_id(records).get(str(record.get("document_id")), {})
    return Evidence(str(doc.get("rel_path") or ""), f"metadata {record.get('key')}: {record.get('value')}", 0.72)


def _context_accessible(context_id: str, records: dict[str, Any], frame: QueryFrame) -> bool:
    context = _contexts_by_id(records).get(str(context_id), {})
    kind = normalize(str(context.get("kind") or "asserted"))
    if not kind:
        return True
    if kind.startswith(INACCESSIBLE_CONTEXT_PREFIXES) and not frame.negated:
        return False
    return True


def _relation_metadata(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("metadata_json") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _condition_material(row: dict[str, Any], evidence: Evidence, records: dict[str, Any]) -> str:
    metadata = _relation_metadata(row)
    fields = [
        row.get("relation_type"), row.get("subject"), row.get("predicate"), row.get("object"), row.get("value"),
        metadata.get("record_path"), metadata.get("record_group"), metadata.get("row_key"), metadata.get("column_header"), metadata.get("section_anchor"),
        evidence.rel_path, evidence.text,
        (records.get("document_context_norm_by_rel_path") or {}).get(evidence.rel_path, ""),
    ]
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


def _value_is_target(value: str, target_terms: list[str]) -> bool:
    material = normalize(value)
    return bool(target_terms) and any(_has_term(material, term) for term in target_terms)


def _answer_values_from_relation(
    row: dict[str, Any],
    evidence: Evidence,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
) -> list[str]:
    values = [str(row.get(key) or "") for key in ["value", "object", "subject"]]
    if expected.answer_type in {"url", "identifier", "file_path", "date_time"}:
        values.append(evidence.text)
    values = [
        value for value in values
        if value
        and not _value_is_target(value, target_terms)
        and (
            expected.answer_type in {"url", "identifier", "file_path", "date_time", "count"}
            or not _value_is_target(value, relation_terms)
        )
    ]
    return _compatible_values(expected, values)


def _answer_values_from_frame(
    frame_row: dict[str, Any],
    args: list[dict[str, Any]],
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
) -> list[str]:
    values = [str(arg.get("surface") or "") for arg in args]
    values = [
        value for value in values
        if value
        and not _value_is_target(value, target_terms)
        and (
            expected.answer_type in {"url", "identifier", "file_path", "date_time", "count"}
            or not _value_is_target(value, relation_terms)
        )
    ]
    return _compatible_values(expected, values)


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


def _negative_evidence(text: str) -> bool:
    return bool(NEGATION_RE.search(text or ""))


def _validity_penalty(text: str) -> float:
    return 0.5 if LOW_VALIDITY_RE.search(text or "") else 1.0


def _bind_frame_conditions(records: dict[str, Any], frame: QueryFrame, expected: ExpectedAnswer, target_terms: list[str], relation_terms: list[str]) -> list[tuple[float, str, Evidence, str]]:
    args_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arg in records.get("frame_arguments", []):
        args_by_frame[str(arg.get("frame_id"))].append(arg)
    candidates: list[tuple[float, str, Evidence, str]] = []
    for row in records.get("frames", []):
        if not _context_accessible(str(row.get("context_id") or ""), records, frame):
            continue
        evidence = _evidence_for_span(str(row.get("span_id") or ""), records)
        arg_text = " ".join(str(arg.get("surface") or "") for arg in args_by_frame.get(str(row.get("frame_id")), []))
        local_material = normalize(" ".join([str(row.get("predicate") or ""), str(row.get("trigger_surface") or ""), arg_text, evidence.text]))
        document_context = (records.get("document_context_norm_by_rel_path") or {}).get(evidence.rel_path, "")
        material = normalize(" ".join([local_material, document_context]))
        score = _split_match_score(material, local_material, target_terms, relation_terms)
        if score <= 0:
            continue
        if _negative_evidence(local_material) and expected.answer_type not in {"boolean", "content_phrase"}:
            continue
        for value in _answer_values_from_frame(row, args_by_frame.get(str(row.get("frame_id")), []), expected, target_terms, relation_terms):
            candidates.append((score * _validity_penalty(evidence.text), value, evidence, "frame_argument_binding"))
    return candidates


def _bind_relation_conditions(records: dict[str, Any], frame: QueryFrame, expected: ExpectedAnswer, target_terms: list[str], relation_terms: list[str]) -> list[tuple[float, str, Evidence, str]]:
    candidates: list[tuple[float, str, Evidence, str]] = []
    for row in records.get("relations", []):
        if not _context_accessible(str(row.get("context_id") or ""), records, frame):
            continue
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        material = _condition_material(row, evidence, records)
        local_material = normalize(
            " ".join(str(row.get(key) or "") for key in ["relation_type", "subject", "predicate", "object", "value"])
            + " "
            + evidence.text
        )
        score = _split_match_score(material, local_material, target_terms, relation_terms)
        if score <= 0:
            continue
        if _negative_evidence(local_material) and expected.answer_type not in {"boolean", "content_phrase"}:
            continue
        for value in _answer_values_from_relation(row, evidence, expected, target_terms, relation_terms):
            candidates.append((score * float(row.get("confidence") or 0.7) * _validity_penalty(evidence.text), value, evidence, "relation_condition_binding"))
    return candidates


def _bind_metadata(records: dict[str, Any], question: str, expected: ExpectedAnswer, target_terms: list[str], relation_terms: list[str]) -> list[tuple[float, str, Evidence, str]]:
    if not expected.allow_metadata_evidence:
        return []
    docs = _docs_by_id(records)
    candidates: list[tuple[float, str, Evidence, str]] = []
    for row in records.get("metadata_records", []):
        doc = docs.get(str(row.get("document_id")), {})
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
    rows: list[tuple[str, dict[str, Any], Evidence]] = []
    for row in records.get("temporal_edges", []):
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        material = normalize(" ".join([str(row.get("relation") or ""), str(row.get("temporal_value") or ""), str(row.get("state_value") or ""), evidence.text]))
        if target_terms and not _contains_any(material, target_terms):
            continue
        if relation_terms and not _contains_any(material, relation_terms):
            continue
        rows.append((str(row.get("temporal_value") or ""), row, evidence))
    rows.sort(key=lambda item: item[0], reverse=frame.temporal_scope != "earliest")
    candidates: list[tuple[float, str, Evidence, str]] = []
    limit = 1 if frame.temporal_scope in {"latest", "earliest"} else 3
    for _time_value, row, evidence in rows[:limit]:
        raw_values = [str(row.get("state_value") or ""), str(row.get("temporal_value") or "")]
        for value in _compatible_values(expected, raw_values):
            candidates.append((8.0, value, evidence, "temporal_binding"))
    return candidates


def _boolean_answer(records: dict[str, Any], frame: QueryFrame, target_terms: list[str], relation_terms: list[str]) -> Answer | None:
    support: list[Evidence] = []
    denial: list[Evidence] = []
    for row in records.get("relations", []):
        evidence = _evidence_for_span(str(row.get("source_span_id") or ""), records)
        material = _condition_material(row, evidence, records)
        if _match_score(material, target_terms, relation_terms) <= 0:
            continue
        (denial if _negative_evidence(material) else support).append(evidence)
    if denial:
        return Answer(f"No; {clean_extracted_value(denial[0].text)}.", 0.8, denial[:2], "boolean DRS binding", "boolean")
    if support:
        return Answer(f"Yes; {clean_extracted_value(support[0].text)}.", 0.72, support[:2], "boolean DRS binding", "boolean")
    return None


def _choose_answer(candidates: list[tuple[float, str, Evidence, str]], expected: ExpectedAnswer) -> Answer | None:
    scored: dict[str, tuple[float, list[Evidence], str]] = {}
    for score, value, evidence, reason in candidates:
        canonical = canonicalize_answer(expected, value)
        if not canonical:
            continue
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
    expected = infer_expected_answer(question)
    target_terms = _target_terms(frame, question)
    relation_terms = _relation_terms(frame, question)
    selected_docs, selected_chunks, ranking = _rank_scope(documents, sentences_by_document, question, frame, doc_limit, chunk_limit)
    records = _load_records(store, run_id, selected_docs, selected_chunks)
    diagnostics = {"ranking": ranking, "execution": {"record_counts": records["record_counts"], "query_frame": frame.as_dict()}}

    if expected.answer_type == "boolean":
        answer = _boolean_answer(records, frame, target_terms, relation_terms)
        return answer, diagnostics

    candidates: list[tuple[float, str, Evidence, str]] = []
    candidates.extend(_bind_frame_conditions(records, frame, expected, target_terms, relation_terms))
    candidates.extend(_bind_relation_conditions(records, frame, expected, target_terms, relation_terms))
    temporal_candidates = _temporal_candidates(records, frame, expected, target_terms, relation_terms)
    if temporal_candidates and frame.temporal_scope in {"latest", "earliest"}:
        return _choose_answer(temporal_candidates, expected), diagnostics
    candidates.extend(temporal_candidates)
    candidates.extend(_bind_metadata(records, question, expected, target_terms, relation_terms))
    candidates.extend(_bind_contexts(records, frame, expected, target_terms, relation_terms))

    if expected.answer_type == "count" and candidates:
        values = sorted({canonicalize_answer(expected, value) or value for _score, value, _evidence, _reason in candidates})
        evidence = [item[2] for item in candidates[:4]]
        return Answer(str(len(values)), 0.85, evidence, "aggregation DRS binding", "count"), diagnostics

    return _choose_answer(candidates, expected), diagnostics
