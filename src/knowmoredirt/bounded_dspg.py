"""Generic bounded DSPG retrieval and graph execution.

This module takes a relation-agnostic query frame, selects a compact SQLite
subgraph, and answers only from grounded relations/source spans.  It has no
content-domain intent switch: relation words from the question are treated as
data used for constraint matching.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .answer_types import ExpectedAnswer, canonicalize_answer, infer_expected_answer, is_value_compatible
from .extractors import capitalized_phrases, identifiers, urls
from .models import Answer, Document, Evidence, Sentence
from .query import QueryFrame, expand_terms, frame_from_mapping, plan_question
from .text import clean_extracted_value, content_tokens, is_low_semantic_noise, normalize


DATE_TIME_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2})?|\d{1,2}:\d{2})\b")
PATH_RE = re.compile(r"\b[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+\b|\b[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12}\b")
TRUTH_REQUIREMENT_TERMS = {"actually", "really", "proven", "proved", "confirmed", "fact", "factual"}
NONASSERTIVE_CONTEXT_RE = re.compile(
    r"\b(?:dream|dreamed|fiction|fictional|belief|believes|believed|allege|alleges|alleged|allegation|rumor|hypothetical)\b",
    re.I,
)
LOW_VALIDITY_RE = re.compile(r"\b(?:archived|obsolete|superseded|stale|old|retired)\b", re.I)
HIGH_VALIDITY_RE = re.compile(r"\b(?:active|current|ready|valid)\b", re.I)


def _clean(value: str) -> str:
    return clean_extracted_value(value).strip(" .;:")


def _query_terms(text: str) -> list[str]:
    stop = {
        "answer",
        "content",
        "document",
        "entity",
        "fact",
        "field",
        "folder",
        "item",
        "name",
        "note",
        "object",
        "record",
        "records",
        "row",
        "rows",
        "entry",
        "entries",
        "source",
        "text",
        "thing",
        "value",
    }
    terms: list[str] = []
    for token in content_tokens(text):
        for candidate in [token, *re.split(r"[-_]", token)]:
            if len(candidate) > 1 and candidate not in stop and candidate not in terms:
                terms.append(candidate)
    return expand_terms(terms)


def _count_constraint_terms(question: str, target_terms: list[str]) -> list[str]:
    structural = {
        "many",
        "count",
        "number",
        "row",
        "rows",
        "entry",
        "entries",
        "record",
        "records",
        "have",
        "has",
        "with",
    }
    terms: list[str] = []
    target_material = " ".join(target_terms)
    for token in content_tokens(question):
        if token in structural:
            continue
        if target_material and _has_term(target_material, token):
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _low_priority_source_path(rel_path: str) -> bool:
    path = normalize(rel_path)
    parts = re.split(r"[/_.-]+", path)
    return bool({"cache", "lock", "tmp", "temp", "transport", "hidden"}.intersection(parts))


def _asks_about_low_priority_source(question: str) -> bool:
    q = normalize(question)
    return any(term in q for term in ["cache", "lock", "temporary", "metadata", "file", "path"])


def _frame(plan: dict[str, Any] | QueryFrame | None, question: str) -> QueryFrame:
    if isinstance(plan, QueryFrame):
        return plan
    return frame_from_mapping(question, plan if isinstance(plan, dict) else None)


def _target_terms(frame: QueryFrame, question: str) -> list[str]:
    terms: list[str] = []
    for anchor in frame.target_anchors:
        anchor_norm = normalize(anchor)
        terms.append(anchor_norm)
        if " " in anchor_norm:
            terms.append(anchor_norm.replace(" ", "_"))
            terms.append(anchor_norm.replace(" ", "-"))
        if not re.search(r"\d|[-_:/@.]", anchor_norm):
            terms.extend(_query_terms(anchor))
    return list(dict.fromkeys(term for term in terms if term))


def _relation_terms(frame: QueryFrame, question: str) -> list[str]:
    terms = list(frame.relation_terms)
    terms.extend(_query_terms(frame.requested_relation))
    terms.extend(frame.constraints)
    target = set(_target_terms(frame, question))
    values = [
        term
        for term in expand_terms(terms)
        if term not in target and len(term) > 1
    ]
    return list(dict.fromkeys(values))


def _document_text(document: Document, sentences: list[Sentence]) -> str:
    metadata = " ".join(
        str(value)
        for value in [
            document.metadata.get("file_name", ""),
            document.metadata.get("stem", ""),
            document.metadata.get("suffix", ""),
            document.metadata.get("parent_rel_path", ""),
        ]
    )
    return normalize(f"{metadata} {' '.join(sentence.text for sentence in sentences[:80])}")


def _answer_type_bonus(expected: ExpectedAnswer, text: str) -> int:
    if expected.answer_type == "url":
        return 20 if urls(text) else 0
    if expected.answer_type == "identifier":
        return 14 if identifiers(text) else 0
    if expected.answer_type == "file_path":
        return 14 if PATH_RE.search(text) else 0
    if expected.answer_type in {"person", "actor", "organization"}:
        return 8 if re.search(r"\b[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3}\b", text) else 0
    if expected.answer_type == "date_time":
        return 12 if DATE_TIME_RE.search(text) else 0
    if expected.answer_type == "count":
        return 10 if re.search(r"\b\d+\b", text) else 0
    if expected.answer_type == "boolean":
        return 8 if re.search(r"\b(?:yes|no|not|no|never|denied|unsupported)\b", normalize(text)) else 0
    return 0


def _rank_scope(
    documents: list[Document],
    sentences_by_document: dict[str, dict[int, Sentence]],
    question: str,
    frame: QueryFrame,
    doc_limit: int,
    chunk_limit: int,
) -> tuple[list[str], list[tuple[str, int]], dict[str, Any]]:
    expected = infer_expected_answer(question)
    target_terms = _target_terms(frame, question)
    relation_terms = _relation_terms(frame, question)
    q_terms = _query_terms(question)
    doc_scores: list[tuple[int, str, str]] = []
    for document in documents:
        text = _document_text(document, list(sentences_by_document.get(document.rel_path, {}).values()))
        target_hits = sum(1 for term in target_terms if _has_term(text, term))
        relation_hits = sum(1 for term in relation_terms if _has_term(text, term))
        q_hits = sum(1 for term in q_terms if _has_term(text, term))
        if target_terms and not target_hits:
            continue
        score = target_hits * 16 + relation_hits * 6 + q_hits
        if (
            is_low_semantic_noise(" ".join(sentence.text for sentence in sentences_by_document.get(document.rel_path, {}).values()))
            or _low_priority_source_path(document.rel_path)
        ) and expected.answer_type != "metadata_value" and not _asks_about_low_priority_source(question):
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
        doc_text = _document_text(document, list(ordered.values()))
        doc_has_target = any(_has_term(doc_text, term) for term in target_terms)
        for order, sentence in ordered.items():
            low = normalize(sentence.text)
            score = sum(24 for term in target_terms if _has_term(low, term))
            score += sum(10 for term in relation_terms if _has_term(low, term))
            score += sum(2 for term in q_terms if _has_term(low, term))
            score += _answer_type_bonus(expected, sentence.text)
            if doc_has_target and any(_has_term(low, term) for term in relation_terms):
                score += 14
            if doc_has_target and _answer_type_bonus(expected, sentence.text):
                score += 6
            if (
                is_low_semantic_noise(sentence.text)
                or _low_priority_source_path(sentence.rel_path)
            ) and expected.answer_type != "metadata_value" and not _asks_about_low_priority_source(question):
                score = int(score * 0.15)
            if score:
                chunk_scores.append((score, document.document_id, order, document.rel_path))
            elif doc_has_target and re.search(r"\b[A-Za-z][A-Za-z0-9 _/-]{1,80}\s*[:=]", sentence.text):
                fallback_scores.append((1, document.document_id, order, document.rel_path))
    chunk_scores.sort(key=lambda item: (-item[0], item[3], item[2]))
    fallback_scores.sort(key=lambda item: (item[3], item[2]))
    selected: list[tuple[str, int]] = []
    per_doc: dict[str, int] = {}
    for _score, document_id, order, _rel_path in [*chunk_scores, *fallback_scores]:
        if len(selected) >= chunk_limit:
            break
        for nearby in (order - 2, order - 1, order, order + 1, order + 2):
            if nearby < 0 or per_doc.get(document_id, 0) >= 24:
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
    mentions = _fetch_by_ids(connection, "mentions", "span_id", span_ids)
    refs = _fetch_by_ids(connection, "mention_referents", "mention_id", [row["mention_id"] for row in mentions])
    referents = _fetch_by_ids(connection, "referents", "referent_id", [row["referent_id"] for row in refs])
    identity = _fetch_by_ids(connection, "identity_hypotheses", "left_referent_id", [row["referent_id"] for row in referents])
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
        "identity_hypotheses": identity,
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
            "identity_hypotheses": len(identity), "context_carriers": len(context_carriers), "metadata_records": len(metadata_records),
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


def _relation_metadata(relation: dict[str, Any]) -> dict[str, Any]:
    value = relation.get("metadata_json")
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _record_group_key(relation: dict[str, Any]) -> str:
    metadata = _relation_metadata(relation)
    group = str(metadata.get("record_group") or metadata.get("sentence_group") or "")
    if group:
        return group
    span_id = str(relation.get("source_span_id") or "")
    return f"span:{span_id}"


def _material_for_relation(relation: dict[str, Any], evidence: Evidence, document_context: str) -> str:
    metadata = _relation_metadata(relation)
    metadata_text = " ".join(
        str(metadata.get(key) or "")
        for key in ["record_group", "record_path", "row_key", "column_header", "section_anchor", "surface_format"]
    )
    return normalize(
        " ".join(str(relation.get(key) or "") for key in ["relation_type", "subject", "predicate", "object", "value"])
        + " "
        + evidence.text
        + " "
        + metadata_text
        + " "
        + document_context
    )


def _contains_any(material: str, terms: list[str]) -> bool:
    return any(_has_term(material, term) for term in terms)


def _has_explicit_non_target_anchor(relation: dict[str, Any], target_terms: list[str], extra_material: str = "") -> bool:
    metadata = _relation_metadata(relation)
    relation_type = normalize(str(relation.get("relation_type") or ""))
    fields = ["object"] if relation_type == "event" else ["subject", "object"]
    material = " ".join(str(relation.get(key) or "") for key in fields)
    material += " " + str(metadata.get("section_anchor") or "")
    material += " " + extra_material
    for phrase in capitalized_phrases(material):
        phrase_norm = normalize(phrase)
        if len(phrase.split()) >= 2 and not any(_has_term(phrase_norm, term) for term in target_terms):
            return True
    return False


def _has_term(material: str, term: str) -> bool:
    if not term:
        return False
    if len(term) <= 4 and re.fullmatch(r"[a-z0-9_-]+", term):
        return re.search(rf"\b{re.escape(term)}\b", material) is not None
    return term in material


def _focused_clause(text: str, preferred_terms: list[str]) -> str:
    clauses = [part.strip() for part in re.split(r"\b(?:but|however|although|though|while)\b|[.;]\s*", text, flags=re.I) if part.strip()]
    if not clauses:
        return text
    scored: list[tuple[int, int, str]] = []
    for index, clause in enumerate(clauses):
        material = normalize(clause)
        score = sum(1 for term in preferred_terms if _has_term(material, term))
        scored.append((score, -index, clause))
    scored.sort(reverse=True)
    return scored[0][2] if scored and scored[0][0] else text


def _requires_asserted_truth(relation_terms: list[str] | None) -> bool:
    return bool(set(relation_terms or []).intersection(TRUTH_REQUIREMENT_TERMS))


def _required_relation_constraints(relation_terms: list[str], expected: ExpectedAnswer) -> list[str]:
    structural = {
        "answer",
        "any",
        "belong",
        "belongs",
        "belonging",
        "find",
        "from",
        "give",
        "list",
        "many",
        "named",
        "number",
        "show",
        "which",
        "whose",
        "with",
        "url",
        "urls",
        "link",
        "links",
        "identifier",
        "identifiers",
        "id",
        "ids",
        "date",
        "time",
        "timestamp",
        "file",
        "path",
        "person",
        "people",
        "actor",
        "actors",
        "organization",
        "organizations",
        "state",
        "status",
        "value",
        "values",
        "count",
        "counts",
    }
    if expected.answer_type == "url":
        structural.update({"web", "site", "address"})
    if expected.answer_type == "identifier":
        structural.update({"code", "codes", "reference", "references"})
    structural_stems = {term[:5] for term in structural if len(term) >= 5}
    structural_stems.update({"act", "belong"})
    required: list[str] = []
    for term in relation_terms:
        if len(term) <= 2 or term in structural or term.endswith(("eror", "orer", "oror", "erer")):
            continue
        if any(term.startswith(stem) for stem in structural_stems):
            continue
        required.append(term)
    return required


def _is_nonassertive_material(text: str) -> bool:
    return bool(NONASSERTIVE_CONTEXT_RE.search(text or ""))


def _allows_document_target_prior(relation: dict[str, Any], evidence: Evidence, target_terms: list[str]) -> bool:
    relation_type = normalize(str(relation.get("relation_type") or ""))
    if relation_type == "event":
        return not _has_explicit_non_target_anchor(relation, target_terms)
    if relation_type != "label_value":
        return False
    if "|" in evidence.text or "\t" in evidence.text:
        return False
    prefix = re.split(r"[:=]", evidence.text, maxsplit=1)[0]
    return not any(len(phrase.split()) >= 2 for phrase in capitalized_phrases(prefix))


def _candidate_values(
    relation: dict[str, Any],
    evidence: Evidence,
    expected: ExpectedAnswer,
    target_terms: list[str],
    *,
    group_text: str = "",
    name_expansions: dict[str, str] | None = None,
    relation_terms: list[str] | None = None,
) -> list[str]:
    rel_type = normalize(str(relation.get("relation_type") or ""))
    local_relation_text = normalize(" ".join(str(relation.get(key) or "") for key in ["relation_type", "subject", "predicate", "object", "value"]))
    if expected.answer_type not in {"boolean", "metadata_value"} and re.search(r"\b(?:no|not|never|without|denied|unsupported)\b", local_relation_text):
        return []
    if expected.answer_type in {"person", "actor", "organization"} and rel_type == "event":
        predicate_text = normalize(str(relation.get("predicate") or ""))
        if relation_terms and not any(_has_term(predicate_text, term) for term in relation_terms):
            return []
        subject = _clean(str(relation.get("subject") or ""))
        object_value = _clean(str(relation.get("object") or relation.get("value") or ""))
        object_phrases = [
            phrase for phrase in capitalized_phrases(object_value)
            if is_value_compatible(expected, phrase)
            and not (target_terms and any(_has_term(normalize(phrase), term) for term in target_terms))
        ]
        if object_phrases and target_terms and any(_has_term(normalize(object_value), term) for term in target_terms):
            return [object_phrases[-1]]
        if target_terms and any(_has_term(normalize(object_value), term) for term in target_terms):
            if subject and is_value_compatible(expected, subject):
                if len(subject.split()) == 1:
                    subject = (name_expansions or {}).get(normalize(subject), subject)
                return [subject]
            focus_text = _focused_clause(evidence.text, [*target_terms, *(relation_terms or [])])
            evidence_phrases = [
                phrase for phrase in capitalized_phrases(focus_text)
                if is_value_compatible(expected, phrase)
                and normalize(phrase) != normalize(subject)
                and not any(_has_term(normalize(phrase), term) for term in target_terms)
            ]
            if evidence_phrases:
                return [evidence_phrases[-1]]
        if target_terms and subject and any(_has_term(normalize(subject), term) for term in target_terms) and object_value:
            for phrase in capitalized_phrases(object_value):
                parts = phrase.split()
                if len(parts) >= 2 or phrase.startswith(("Dr.", "Ms.", "Mr.", "Mrs.", "Prof.")):
                    return [phrase]
            return [object_value]
        if subject and len(subject.split()) == 1:
            honorific = re.search(rf"\b(?:Dr\.|Ms\.|Mr\.|Mrs\.|Prof\.)\s+{re.escape(subject)}\b", evidence.text)
            if honorific:
                subject = honorific.group(0)
            else:
                expanded = (name_expansions or {}).get(normalize(subject))
                if expanded:
                    subject = expanded
            for phrase in capitalized_phrases(evidence.text):
                if phrase.endswith(f" {subject}") or phrase == subject:
                    subject = phrase
                    break
        return [subject] if subject else []
    elif expected.answer_type in {"person", "actor", "organization"} and rel_type in {"label_value", "record_value", "table_cell", "semantic_argument"}:
        value = _clean(str(relation.get("value") or relation.get("object") or ""))
        if value and target_terms and any(_has_term(normalize(value), term) for term in target_terms):
            return []
        if value and len(value.split()) == 1:
            value = (name_expansions or {}).get(normalize(value), value)
        if value and not is_value_compatible(expected, value):
            phrases = [
                phrase
                for phrase in capitalized_phrases(value)
                if is_value_compatible(expected, phrase)
            ]
            if phrases:
                return [phrases[-1]]
        return [value] if value else []
    elif expected.answer_type == "state":
        fields = [
            str(relation.get("value") or ""),
            str(relation.get("object") or ""),
        ]
    elif expected.answer_type == "content_phrase" and rel_type in {"label_value", "record_value", "table_cell", "assertion"}:
        fields = [
            str(relation.get("value") or ""),
            str(relation.get("object") or ""),
        ]
    else:
        fields = [
            str(relation.get("value") or ""),
            str(relation.get("object") or ""),
            str(relation.get("subject") or ""),
        ]
    field_text = " ".join([field for field in fields if field])
    text = field_text or evidence.text
    if expected.answer_type == "url":
        return [value for value in urls(text) if "." in value.split("://", 1)[-1].split("/", 1)[0]]
    if expected.answer_type == "identifier":
        return identifiers(text)
    if expected.answer_type == "file_path":
        without_urls = text
        for url in urls(text):
            without_urls = without_urls.replace(url, " ")
        return [match.group(0).rstrip(".,;)") for match in PATH_RE.finditer(without_urls)]
    if expected.answer_type == "date_time":
        return [match.group(0) for match in DATE_TIME_RE.finditer(text)]
    if expected.answer_type == "boolean":
        if (
            relation_terms
            and _requires_asserted_truth(relation_terms)
            and _is_nonassertive_material(evidence.text)
            and not re.search(r"\b(?:no|not|never|without|denied|unsupported)\b", normalize(evidence.text))
        ):
            return []
        if re.search(r"\bno\s+final\s+decision\b|\bno\s+.+\s+decision\s+was\s+made\b", normalize(evidence.text)):
            return []
        if re.search(r"\b(?:no|not|never|without|denied|unsupported)\b", normalize(evidence.text)):
            return [f"No; {_clean(evidence.text)}."]
        return [f"Yes; {_clean(evidence.text)}."]
    if expected.answer_type in {"person", "actor", "organization"}:
        expanded: list[str] = []
        for field in fields:
            field_clean = _clean(field)
            if field_clean:
                if field_clean in {"Dr", "Ms", "Mr", "Mrs", "Prof"}:
                    honorific_match = re.search(rf"\b{re.escape(field_clean)}\.\s+[A-Z][A-Za-z'-]+", evidence.text)
                    if honorific_match:
                        expanded.append(honorific_match.group(0))
                expanded.append(field_clean)
                if is_value_compatible(expected, field_clean):
                    continue
            for phrase in capitalized_phrases(field):
                parts = phrase.split()
                if (
                    normalize(phrase) not in {"the", "a", "an"}
                    and (len(parts) >= 2 or phrase.startswith(("Dr.", "Ms.", "Mr.", "Mrs.", "Prof.")))
                    and not phrase.isupper()
                ):
                    expanded.append(phrase)
        fields = expanded
    values: list[str] = []
    for field in fields:
        field_clean = _clean(field)
        if not field_clean:
            continue
        field_norm = normalize(field_clean)
        if target_terms and any(_has_term(field_norm, term) for term in target_terms):
            continue
        values.append(field_clean)
    if not values and expected.answer_type in {"content_phrase", "state"}:
        values.append(_clean(str(relation.get("value") or relation.get("object") or evidence.text)))
    return values


def _score_relation(
    relation: dict[str, Any],
    evidence: Evidence,
    document_context: str,
    frame: QueryFrame,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
    group_material: str = "",
    name_expansions: dict[str, str] | None = None,
) -> tuple[int, list[str]]:
    metadata = _relation_metadata(relation)
    metadata_text = " ".join(
        str(metadata.get(key) or "")
        for key in ["record_path", "row_key", "column_header", "section_anchor", "surface_format"]
    )
    local_material = _material_for_relation(relation, evidence, "")
    local_key_material = normalize(" ".join(str(relation.get(key) or "") for key in ["relation_type", "subject", "predicate", "object"]))
    scoped_material = normalize(local_material + " " + group_material + " " + metadata_text)
    if (
        expected.answer_type != "boolean"
        and not frame.negated
        and re.search(r"\b(?:no|not|never|without|denied|unsupported)\b", scoped_material)
    ):
        return 0, []
    target_material = scoped_material
    target_hits = sum(1 for term in target_terms if _has_term(target_material, term))
    if target_terms and not target_hits:
        doc_target_hits = sum(1 for term in target_terms if _has_term(document_context, term))
        group_without_value = group_material
        for current_value in [str(relation.get("value") or ""), str(relation.get("object") or "")]:
            if current_value:
                group_without_value = group_without_value.replace(normalize(current_value), " ")
        if (
            doc_target_hits
            and _allows_document_target_prior(relation, evidence, target_terms)
            and not _has_explicit_non_target_anchor(relation, target_terms, group_without_value)
        ):
            target_hits = doc_target_hits
    relation_hits = sum(1 for term in relation_terms if _has_term(scoped_material, term))
    key_relation_hits = sum(1 for term in relation_terms if _has_term(local_key_material, term))
    if target_terms and not target_hits:
        return 0, []
    if relation_terms and not relation_hits:
        return 0, []
    required_relation_terms = _required_relation_constraints(relation_terms, expected)
    if required_relation_terms and not any(_has_term(scoped_material, term) for term in required_relation_terms):
        return 0, []
    structural_type_terms = {
        "url",
        "urls",
        "link",
        "links",
        "identifier",
        "identifiers",
        "id",
        "ids",
        "date",
        "time",
        "timestamp",
        "file",
        "path",
    }
    scoped_relation_terms = [
        term
        for term in relation_terms
        if term not in structural_type_terms and len(term) > 2
    ]
    if expected.answer_type in {"url", "identifier", "file_path", "date_time"} and scoped_relation_terms and not any(
        _has_term(scoped_material, term) for term in scoped_relation_terms
    ):
        return 0, []
    if (
        normalize(str(relation.get("relation_type") or "")) in {"label_value", "record_value", "table_cell"}
        and relation_terms
        and not key_relation_hits
        and expected.answer_type in {"content_phrase", "state", "person", "actor", "organization"}
    ):
        return 0, []
    values = [
        value
        for value in _candidate_values(
            relation,
            evidence,
            expected,
            target_terms,
            group_text=group_material,
            name_expansions=name_expansions,
            relation_terms=relation_terms,
        )
        if is_value_compatible(expected, value)
    ]
    if not values:
        return 0, []
    score = target_hits * 18 + relation_hits * 12 + key_relation_hits * 8 + _answer_type_bonus(expected, " ".join(values) + " " + evidence.text)
    if target_terms and target_hits == len(target_terms):
        score += 10
    if relation_terms and relation_hits == len(relation_terms):
        score += 10
    if group_material and group_material != normalize(evidence.text):
        score += 4
    if LOW_VALIDITY_RE.search(group_material) and not LOW_VALIDITY_RE.search(normalize(frame.question_text)):
        score -= 18
    if HIGH_VALIDITY_RE.search(group_material) and not LOW_VALIDITY_RE.search(group_material):
        score += 6
    if frame.temporal_scope == "latest" and DATE_TIME_RE.search(evidence.text):
        score += 8
    if normalize(str(relation.get("relation_type") or "")) in {"label_value", "record_value", "table_cell"}:
        score += 4
    return score, values


def _score_frame_bindings(
    records: dict[str, Any],
    spans_by_id: dict[str, dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    docs_by_id: dict[str, dict[str, Any]],
    document_context: dict[str, str],
    frame: QueryFrame,
    expected: ExpectedAnswer,
    target_terms: list[str],
    relation_terms: list[str],
) -> list[tuple[int, str, Evidence]]:
    """Bind answer variables against generic frame arguments.

    This is the DRT/DSPG counterpart to the relation-row matcher.  It does not
    know domain roles.  It asks whether a grounded condition's predicate and
    arguments satisfy the query frame, then returns compatible non-target
    arguments as candidate variable bindings.
    """

    args_by_frame: dict[str, list[dict[str, Any]]] = {}
    for argument in records.get("frame_arguments", []):
        args_by_frame.setdefault(str(argument.get("frame_id")), []).append(argument)
    contexts_by_id = {str(context.get("context_id")): context for context in records.get("contexts", [])}
    scored: list[tuple[int, str, Evidence]] = []
    for source_frame in records.get("frames", []):
        if str(source_frame.get("source") or "") != "local_model":
            continue
        span = spans_by_id.get(str(source_frame.get("span_id")), {})
        ev = _evidence(span, chunks_by_id, docs_by_id)
        doc_ctx = document_context.get(str(span.get("document_id")), "")
        arguments = args_by_frame.get(str(source_frame.get("frame_id")), [])
        argument_text = " ".join(str(argument.get("surface") or "") for argument in arguments)
        predicate_text = " ".join(
            str(source_frame.get(key) or "")
            for key in ["predicate", "trigger_surface", "source"]
        )
        context = contexts_by_id.get(str(source_frame.get("context_id")), {})
        context_kind = normalize(str(context.get("kind") or ""))
        local_material = normalize(" ".join([predicate_text, argument_text, ev.text, context_kind]))
        material = normalize(" ".join([local_material, doc_ctx]))
        if (
            expected.answer_type != "boolean"
            and not frame.negated
            and (context_kind.startswith("modality:") or _is_nonassertive_material(material))
            and _requires_asserted_truth(relation_terms)
        ):
            continue
        if (
            expected.answer_type != "boolean"
            and not frame.negated
            and re.search(r"\b(?:no|not|never|without|denied|unsupported)\b", local_material)
        ):
            continue
        target_hits = sum(1 for term in target_terms if _has_term(material, term))
        relation_hits = sum(1 for term in relation_terms if _has_term(local_material, term))
        if target_terms and not target_hits:
            continue
        if relation_terms and not relation_hits:
            continue
        ordered_arguments = sorted(
            arguments,
            key=lambda item: 0 if normalize(str(item.get("role") or "")) == "value" else 1,
        )
        has_value_argument = any(normalize(str(item.get("role") or "")) == "value" for item in ordered_arguments)
        for argument in ordered_arguments:
            role_norm = normalize(str(argument.get("role") or ""))
            if has_value_argument and expected.answer_type in {"state", "content_phrase", "metadata_value"} and role_norm != "value":
                continue
            value = _clean(str(argument.get("surface") or ""))
            if not value:
                continue
            value_norm = normalize(value)
            if target_terms and any(_has_term(value_norm, term) for term in target_terms):
                continue
            if not is_value_compatible(expected, value):
                continue
            score = 20 + target_hits * 18 + relation_hits * 14 + _answer_type_bonus(expected, value)
            if role_norm == "value":
                score += 18
            if str(source_frame.get("source") or "") == "local_model":
                score += 10
            scored.append((score, value, ev))
    return scored


def _execute(records: dict[str, Any], frame: QueryFrame, question: str) -> tuple[list[str], list[Evidence], dict[str, Any]]:
    expected = infer_expected_answer(question)
    target_terms = _target_terms(frame, question)
    relation_terms = _relation_terms(frame, question)
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in records["chunks"]}
    docs_by_id = {str(document["document_id"]): document for document in records["documents"]}
    spans_by_id = {str(span["span_id"]): span for span in records["source_spans"]}
    chunks_by_doc: dict[str, list[str]] = {}
    for chunk in records["chunks"]:
        chunks_by_doc.setdefault(str(chunk.get("document_id")), []).append(str(chunk.get("text") or ""))
    document_context = {
        document_id: normalize(str(docs_by_id.get(document_id, {}).get("rel_path") or "") + " " + " ".join(texts[:12]))
        for document_id, texts in chunks_by_doc.items()
    }
    expansion_candidates: dict[str, set[str]] = {}
    for referent in records.get("referents", []):
        label = _clean(str(referent.get("canonical_label") or ""))
        parts = label.split()
        if len(parts) >= 2 and all(part[:1].isupper() for part in parts):
            for part in parts:
                expansion_candidates.setdefault(normalize(part), set()).add(label)
    name_expansions = {
        token: next(iter(values))
        for token, values in expansion_candidates.items()
        if len(values) == 1
    }
    relation_evidence: dict[str, Evidence] = {}
    group_materials: dict[str, list[str]] = {}
    group_evidence: dict[str, list[Evidence]] = {}
    for relation in records["relations"]:
        span = spans_by_id.get(str(relation.get("source_span_id")), {})
        ev = _evidence(span, chunks_by_id, docs_by_id)
        relation_evidence[str(relation.get("relation_id"))] = ev
        group_key = _record_group_key(relation)
        metadata = _relation_metadata(relation)
        metadata_text = " ".join(
            str(metadata.get(key) or "")
            for key in ["record_path", "row_key", "column_header", "section_anchor", "surface_format"]
        )
        material = " ".join(
            str(relation.get(field) or "")
            for field in ["relation_type", "subject", "predicate", "object", "value"]
        ) + " " + metadata_text
        group_materials.setdefault(group_key, []).append(material)
        group_evidence.setdefault(group_key, []).append(ev)
    group_texts = {key: normalize(" ".join(values)) for key, values in group_materials.items()}

    if expected.answer_type == "metadata_value":
        metadata_hits: list[tuple[int, str, Evidence]] = []
        query_terms = _query_terms(question)
        for record in records.get("metadata_records", []):
            doc = docs_by_id.get(str(record.get("document_id")), {})
            doc_context = normalize(str(doc.get("rel_path") or "") + " " + document_context.get(str(record.get("document_id")), ""))
            if target_terms and not _contains_any(doc_context, target_terms):
                continue
            key = normalize(str(record.get("key") or ""))
            score = sum(10 for term in query_terms if term in key)
            if score:
                metadata_hits.append((score, str(record.get("value") or ""), _metadata_evidence(record, docs_by_id)))
        metadata_hits.sort(key=lambda item: -item[0])
        if metadata_hits:
            return [value for _score, value, _ev in metadata_hits[:8]], [ev for _score, _value, ev in metadata_hits[:8]], {"record_counts": records.get("record_counts", {})}

    if frame.aggregation == "count":
        seen: set[str] = set()
        evidence: list[Evidence] = []
        constraint_terms = _count_constraint_terms(question, target_terms)
        qnorm = normalize(question)
        structured_count_requested = bool(re.search(r"\b(?:row|rows|entry|entries|record|records)\b", qnorm))
        row_count_requested = bool(re.search(r"\b(?:row|rows)\b", qnorm))
        record_count_requested = bool(re.search(r"\b(?:record|records)\b", qnorm))
        for group_key, group_text in group_texts.items():
            evs = group_evidence.get(group_key, [])
            if row_count_requested and "table_cell" not in group_text:
                continue
            if record_count_requested and "record_value" not in group_text:
                continue
            if structured_count_requested and not any(marker in group_text for marker in ["table_cell", "record_value", "label_value"]):
                continue
            if target_terms and not _contains_any(group_text, target_terms):
                continue
            if constraint_terms and not all(_has_term(group_text, term) for term in constraint_terms):
                continue
            if group_key not in seen:
                seen.add(group_key)
                if evs:
                    evidence.append(evs[0])
        if seen:
            return [str(len(seen))], evidence[:8], {"record_counts": records.get("record_counts", {})}

    if "context" in normalize(question):
        contexts_by_id = {str(context["context_id"]): context for context in records["contexts"]}
        context_values: list[str] = []
        context_evidence: list[Evidence] = []
        for carrier in records.get("context_carriers", []):
            context = contexts_by_id.get(str(carrier.get("context_id")), {})
            value = str(context.get("kind") or carrier.get("carrier_surface") or "")
            span = spans_by_id.get(str(carrier.get("source_span_id")), {})
            ev = _evidence(span, chunks_by_id, docs_by_id)
            material = normalize(" ".join([value, str(carrier.get("carrier_surface") or ""), ev.text]))
            if target_terms and not _contains_any(material, target_terms):
                continue
            if value and normalize(value).startswith("quality:"):
                continue
            if value and value not in context_values:
                context_values.append(value)
                context_evidence.append(ev)
        if context_values:
            return context_values[:4], context_evidence[:4], {"record_counts": records.get("record_counts", {})}

    scored: list[tuple[int, str, Evidence]] = _score_frame_bindings(
        records,
        spans_by_id,
        chunks_by_id,
        docs_by_id,
        document_context,
        frame,
        expected,
        target_terms,
        relation_terms,
    )
    for relation in records["relations"]:
        span = spans_by_id.get(str(relation.get("source_span_id")), {})
        ev = relation_evidence.get(str(relation.get("relation_id"))) or _evidence(span, chunks_by_id, docs_by_id)
        doc_ctx = document_context.get(str(span.get("document_id")), "")
        group_text = group_texts.get(_record_group_key(relation), "")
        score, values = _score_relation(
            relation,
            ev,
            doc_ctx,
            frame,
            expected,
            target_terms,
            relation_terms,
            group_text,
            name_expansions,
        )
        if score and (_low_priority_source_path(ev.rel_path) or is_low_semantic_noise(ev.text)) and expected.answer_type != "metadata_value" and not _asks_about_low_priority_source(question):
            score = int(score * 0.1)
        for value in values:
            if score:
                scored.append((score, value, ev))

    if frame.temporal_scope == "latest":
        temporal_scored: list[tuple[str, str, Evidence]] = []
        for edge in records["temporal_edges"]:
            span = spans_by_id.get(str(edge.get("source_span_id")), {})
            ev = _evidence(span, chunks_by_id, docs_by_id)
            material = normalize(ev.text + " " + document_context.get(str(span.get("document_id")), ""))
            if target_terms and not _contains_any(material, target_terms):
                continue
            value = str(edge.get("state_value") or edge.get("temporal_value") or "")
            if value and is_value_compatible(expected, value):
                temporal_scored.append((str(edge.get("temporal_value") or ""), value, ev))
        temporal_scored.sort(key=lambda item: item[0], reverse=True)
        if temporal_scored:
            value, ev = temporal_scored[0][1], temporal_scored[0][2]
            return [value], [ev], {"record_counts": records.get("record_counts", {})}

    scored.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
    values: list[str] = []
    evidence: list[Evidence] = []
    allow_multiple = bool(re.search(r"\b(?:all|list|names|ids|urls|references|which ones)\b", normalize(question)))
    has_non_low_priority_source = any(
        not (_low_priority_source_path(ev.rel_path) or is_low_semantic_noise(ev.text))
        for _score, _value, ev in scored
    )
    for _score, value, ev in scored:
        if (
            has_non_low_priority_source
            and expected.answer_type != "metadata_value"
            and not _asks_about_low_priority_source(question)
            and (_low_priority_source_path(ev.rel_path) or is_low_semantic_noise(ev.text))
        ):
            continue
        value = _clean(value)
        if value and normalize(value) != "unknown" and value not in values:
            values.append(value)
            evidence.append(ev)
            if not allow_multiple or expected.answer_type in {"state", "date_time", "count", "boolean"}:
                break
        if len(values) >= 8:
            break
    return values, evidence, {"record_counts": records.get("record_counts", {})}


def execute_bounded_query(
    store: Any,
    run_id: str,
    documents: list[Document],
    sentences_by_document: dict[str, dict[int, Sentence]],
    question: str,
    plan: dict[str, Any] | QueryFrame | None,
    *,
    doc_limit: int = 40,
    chunk_limit: int = 240,
) -> tuple[Answer | None, dict[str, Any]]:
    frame = _frame(plan, question)
    expected = infer_expected_answer(question)
    document_ids, chunk_keys, ranking = _rank_scope(documents, sentences_by_document, question, frame, doc_limit, chunk_limit)
    records = _load_records(store, run_id, document_ids, chunk_keys)
    values, evidence, execution = _execute(records, frame, question)
    diagnostics = {"ranking": ranking, "execution": execution, "query_frame": frame.as_dict()}
    if not values or not evidence:
        return None, diagnostics
    answer_text = canonicalize_answer(expected, "; ".join(values))
    if not answer_text:
        diagnostics["execution"] = {**execution, "rejected_reason": "answer_type_incompatible", "expected_answer_type": expected.answer_type}
        return None, diagnostics
    return Answer(answer_text, 0.78, evidence, "bounded DSPG query-frame execution", expected.answer_type), diagnostics
