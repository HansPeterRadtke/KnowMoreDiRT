"""Raw-text ingestion into the internal DSPG store."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .drs import DiscourseArgument, DiscourseCondition, frame_from_model_dict
from .extractors import capitalized_phrases, identifiers, urls
from .models import Document, Sentence
from .model_planner import call_model_chunk_frames
from .relations import ExtractedRelation, extract_relations
from .scanner import scan_folder
from .semantic_cache import SemanticFrameCache
from .store import DSPGStore, stable_id
from .text import clean_extracted_value, normalize, text_quality_metrics, tokenize


DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2})?\b")
TABLE_SPLIT_RE = re.compile(r"\s*(?:\||\t)\s*")


def _timestamp_value(value: float) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(value)))
    except Exception:
        return str(value)


def _metadata_pairs(document: Document, quality: dict[str, object]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for key in [
        "file_name",
        "stem",
        "suffix",
        "suffixes",
        "parent_rel_path",
        "directory_depth",
        "hidden_file",
        "stat_mode",
        "permissions",
        "uid",
        "gid",
        "inode",
        "device",
        "atime",
        "mtime",
        "ctime",
        "line_count",
        "word_count",
        "mime_type",
        "encoding",
        "decode_errors",
        "read_mode",
        "symlink",
        "symlink_target",
    ]:
        if key in document.metadata:
            pairs.append((key, json.dumps(document.metadata[key], sort_keys=True) if isinstance(document.metadata[key], (list, dict)) else str(document.metadata[key])))
    pairs.extend(
        [
            ("size_bytes", str(document.size_bytes)),
            ("content_hash", document.sha256),
            ("char_count", str(len(document.text))),
            ("text_quality", str(quality.get("semantic_quality", ""))),
        ]
    )
    return [(key, value) for key, value in pairs if value != ""]


def mention_entity_type(surface: str) -> str:
    if re.fullmatch(r"https?://\S+", surface):
        return "url"
    if re.fullmatch(r"[A-Z][A-Z0-9]{1,9}-\d+[A-Z0-9-]*", surface):
        return "identifier"
    if re.fullmatch(r"[a-z][a-z0-9]{1,12}_[a-z0-9]{6,}", surface):
        return "identifier"
    if re.fullmatch(r"[0-9a-f]{8,16}", surface, re.I):
        return "commit"
    if "@" in surface and "." in surface:
        return "email"
    if len(surface.split()) >= 2:
        return "name"
    return "entity"


def context_kind_for_sentence(text: str) -> str:
    return "asserted"


def collect_mentions(sentence: Sentence) -> list[tuple[str, str, int, int]]:
    values: list[tuple[str, str, int, int]] = []
    for value in urls(sentence.text) + identifiers(sentence.text) + capitalized_phrases(sentence.text):
        start = sentence.text.find(value)
        if start < 0:
            continue
        entity_type = mention_entity_type(value)
        values.append((value, entity_type, sentence.char_start + start, sentence.char_start + start + len(value)))
    seen: set[tuple[str, int]] = set()
    unique: list[tuple[str, str, int, int]] = []
    for item in values:
        key = (item[0], item[2])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def frame_predicate(text: str) -> tuple[str, str] | None:
    for relation in extract_relations(text):
        if relation.predicate:
            return relation.predicate, relation.predicate
    return None


def temporal_state(text: str) -> tuple[str, str] | None:
    date_match = DATE_RE.search(text)
    if not date_match:
        return None
    relations = [
        relation
        for relation in extract_relations(text)
        if relation.relation_type in {"label_value", "record_value", "table_cell"} and relation.value
    ]
    if relations:
        return date_match.group(0), relations[-1].value
    return None


def _table_cells(text: str) -> list[str]:
    if "|" not in text and "\t" not in text:
        return []
    cells = [clean_extracted_value(cell) for cell in TABLE_SPLIT_RE.split(text)]
    return [cell for cell in cells if cell]


def _looks_like_table_header(cells: list[str]) -> bool:
    if len(cells) < 2:
        return False
    return all(re.search(r"[A-Za-z]", cell) for cell in cells) and not any(urls(cell) or identifiers(cell) for cell in cells)


def _is_structural_heading(text: str) -> bool:
    value = clean_extracted_value(text)
    if not value or ":" in value or "|" in value or "\t" in value:
        return False
    if urls(value) or identifiers(value):
        return False
    tokens = tokenize(value)
    if not 1 <= len(tokens) <= 8:
        return False
    phrases = capitalized_phrases(value)
    return bool(phrases) and len(value) <= 100


def _starts_new_structural_record(text: str) -> bool:
    if "|" in text or "\t" in text:
        return True
    if re.search(r"^\s*[\[{]", text):
        return True
    prefix = re.split(r"[:=]", text, maxsplit=1)[0]
    if any(len(phrase.split()) >= 2 for phrase in capitalized_phrases(prefix)):
        return True
    return False


def _relation_inherits_heading(text: str, relations: list[ExtractedRelation]) -> bool:
    if not relations:
        return False
    value = text.strip()
    if not value or _starts_new_structural_record(value):
        return False
    return all(relation.relation_type in {"label_value", "record_value"} for relation in relations)


def _label_heading_value(text: str) -> str:
    if "|" in text or "\t" in text or "://" in text:
        return ""
    relations = extract_relations(text)
    label_values = [relation for relation in relations if relation.relation_type == "label_value"]
    if len(label_values) != 1:
        return ""
    value = clean_extracted_value(label_values[0].value)
    if identifiers(value) or urls(value):
        return ""
    if any(len(phrase.split()) >= 2 and normalize(phrase) == normalize(value) for phrase in capitalized_phrases(value)):
        return value
    return ""


def _table_header_relations(sentence: Sentence, headers: list[str], cells: list[str]) -> list[ExtractedRelation]:
    if len(headers) < 2 or len(cells) != len(headers):
        return []
    row_key = cells[0]
    relations: list[ExtractedRelation] = []
    group = stable_id("table_row", sentence.document_id, sentence.order, row_key, "|".join(cells))
    for header, cell in zip(headers[1:], cells[1:]):
        if not header or not cell:
            continue
        relations.append(
            ExtractedRelation(
                relation_type="table_cell",
                predicate=normalize(header),
                subject=row_key,
                value=cell,
                confidence=0.82,
                metadata={
                    "record_group": group,
                    "row_key": row_key,
                    "column_header": header,
                    "surface_format": "delimited_table",
                },
            )
        )
    return relations


def _grounded_model_frames(
    sentence: Sentence,
    semantic_client: Any | None,
    semantic_cache: SemanticFrameCache | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if semantic_client is None:
        return [], {"source": "disabled"}
    quality = text_quality_metrics(sentence.text)
    if quality.get("low_semantic_noise"):
        return [], {"source": "skipped_noise"}
    if len(sentence.text) > 2600:
        return [], {"source": "skipped_long_chunk"}
    cached = semantic_cache.get(sentence.text) if semantic_cache else None
    if cached is not None:
        frames = [frame for frame in cached.get("frames", []) if isinstance(frame, dict)]
        return frames, {"source": "cache", "frame_count": len(frames)}
    result = call_model_chunk_frames(sentence.text, semantic_client, rel_path=sentence.rel_path)
    frames = [frame for frame in result.get("frames", []) if isinstance(frame, dict)] if result.get("accepted") else []
    if semantic_cache is not None and result.get("accepted"):
        semantic_cache.put(
            sentence.text,
            frames,
            {
                "rel_path": sentence.rel_path,
                "prompt_hash": result.get("prompt_hash"),
                "output_hash": result.get("output_hash"),
            },
        )
    return frames, result


def _condition_from_deterministic_relation(relation: ExtractedRelation, evidence_text: str) -> DiscourseCondition | None:
    predicate = relation.predicate or relation.relation_type
    if not predicate:
        return None
    arguments: list[DiscourseArgument] = []
    for role, value in [
        ("subject", relation.subject),
        ("object", relation.object),
        ("value", relation.value),
    ]:
        if value:
            arguments.append(DiscourseArgument(role=role, value=value, value_type="unknown"))
    if not arguments and relation.value:
        arguments.append(DiscourseArgument(role="value", value=relation.value, value_type="unknown"))
    return DiscourseCondition(
        predicate=predicate,
        arguments=tuple(arguments),
        frame_type=relation.relation_type,
        polarity="positive",
        modality="asserted",
        temporal_text="",
        evidence_text=evidence_text,
        confidence=relation.confidence,
        metadata=dict(relation.metadata),
    )


def ingest_folder(
    folder_path: str | Path,
    store: DSPGStore | None = None,
    *,
    semantic_client: Any | None = None,
    use_semantic_frames: bool = False,
    semantic_cache: SemanticFrameCache | None = None,
) -> tuple[DSPGStore, str, list[Document], list[Sentence]]:
    created_store = store is None
    store = store or DSPGStore(create_indexes=False)
    documents, sentences = scan_folder(folder_path)
    run_id = store.start_run(folder_path)

    sentence_by_id = {sentence.sentence_id: sentence for sentence in sentences}
    referent_cache: dict[tuple[str, str], str] = {}
    context_by_kind: dict[str, str] = {}

    for document in documents:
        quality = text_quality_metrics(document.text)
        store.execute(
            """
            INSERT INTO documents(
              document_id, run_id, path, rel_path, content_hash, size_bytes, mtime, ctime, char_count, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.document_id,
                run_id,
                str(document.path),
                document.rel_path,
                document.sha256,
                document.size_bytes,
                document.mtime,
                document.ctime,
                len(document.text),
                json.dumps({**document.metadata, "text_quality": quality}, sort_keys=True),
            ),
        )
        quality_kind = f"quality:{quality['semantic_quality']}"
        if quality_kind not in context_by_kind:
            context_id = stable_id("ctx", run_id, quality_kind)
            context_by_kind[quality_kind] = context_id
            store.execute(
                "INSERT INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (context_id, run_id, quality_kind, None, document.rel_path, quality_kind, 1.0),
            )
        quality_context_id = context_by_kind[quality_kind]
        store.execute(
            """
            INSERT OR IGNORE INTO context_carriers(
              carrier_id, run_id, context_id, document_id, source_span_id, carrier_kind, carrier_surface,
              temporal_value, temporal_value_type, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_id("carrier", run_id, document.document_id, "quality", quality_kind),
                run_id,
                quality_context_id,
                document.document_id,
                None,
                "source_quality",
                quality_kind,
                None,
                None,
                1.0,
            ),
        )
        for temporal_key, temporal_type in [("mtime", "file_modified_time"), ("ctime", "file_created_time")]:
            if temporal_key in document.metadata:
                temporal_value = _timestamp_value(float(document.metadata[temporal_key]))
                store.execute(
                    """
                    INSERT OR IGNORE INTO context_carriers(
                      carrier_id, run_id, context_id, document_id, source_span_id, carrier_kind, carrier_surface,
                      temporal_value, temporal_value_type, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("carrier", run_id, document.document_id, temporal_type),
                        run_id,
                        quality_context_id,
                        document.document_id,
                        None,
                        "filesystem_time",
                        temporal_key,
                        temporal_value,
                        temporal_type,
                        1.0,
                    ),
                )
        for key, value in _metadata_pairs(document, quality):
            store.execute(
                """
                INSERT OR IGNORE INTO metadata_records(
                  metadata_id, run_id, document_id, key, value, value_norm, source, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stable_id("meta", run_id, document.document_id, key, value),
                    run_id,
                    document.document_id,
                    key,
                    value,
                    normalize(value),
                    "filesystem" if key in {"size_bytes", "content_hash", "char_count"} or key in document.metadata else "analysis",
                    1.0,
                ),
            )

    table_headers_by_document: dict[str, list[str]] = {}
    section_anchor_by_document: dict[str, str] = {}
    section_group_by_document: dict[str, str] = {}

    for sentence in sentences:
        token_estimate = max(1, len(tokenize(sentence.text)))
        chunk_id = stable_id("chunk", sentence.sentence_id)
        store.execute(
            "INSERT INTO chunks(chunk_id, document_id, chunk_order, char_start, char_end, text, token_estimate) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chunk_id, sentence.document_id, sentence.order, sentence.char_start, sentence.char_end, sentence.text, token_estimate),
        )
        span_id = stable_id("span", sentence.sentence_id, "sentence")
        store.execute(
            "INSERT INTO source_spans(span_id, document_id, chunk_id, char_start, char_end, surface, surface_norm, span_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (span_id, sentence.document_id, chunk_id, sentence.char_start, sentence.char_end, sentence.text, normalize(sentence.text), "sentence"),
        )
        context_kind = context_kind_for_sentence(sentence.text)
        context_id = context_by_kind.get(context_kind)
        if context_id is None:
            context_id = stable_id("ctx", run_id, context_kind)
            context_by_kind[context_kind] = context_id
            store.execute(
                "INSERT INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (context_id, run_id, context_kind, None, None, context_kind, 1.0),
            )
        store.execute(
            """
            INSERT OR IGNORE INTO context_carriers(
              carrier_id, run_id, context_id, document_id, source_span_id, carrier_kind, carrier_surface,
              temporal_value, temporal_value_type, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_id("carrier", run_id, sentence.sentence_id, context_kind),
                run_id,
                context_id,
                sentence.document_id,
                span_id,
                "sentence_context",
                context_kind,
                None,
                None,
                0.9,
            ),
        )
        for applies_to_type, applies_to_id in [("chunk", chunk_id), ("source_span", span_id)]:
            store.execute(
                """
                INSERT OR IGNORE INTO context_assignments(
                  assignment_id, run_id, context_id, applies_to_type, applies_to_id, source_span_id, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stable_id("ctx_assign", run_id, context_id, applies_to_type, applies_to_id),
                    run_id,
                    context_id,
                    applies_to_type,
                    applies_to_id,
                    span_id,
                    0.9,
                ),
            )

        mentions_for_sentence: list[tuple[str, str, str]] = []
        for surface, entity_type, start, end in collect_mentions(sentence):
            mention_span_id = stable_id("span", sentence.sentence_id, surface, start)
            store.execute(
                "INSERT OR IGNORE INTO source_spans(span_id, document_id, chunk_id, char_start, char_end, surface, surface_norm, span_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (mention_span_id, sentence.document_id, chunk_id, start, end, surface, normalize(surface), "mention"),
            )
            mention_id = stable_id("men", run_id, mention_span_id, surface)
            store.execute(
                "INSERT OR IGNORE INTO mentions(mention_id, run_id, span_id, surface, surface_norm, mention_kind, entity_type, confidence, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (mention_id, run_id, mention_span_id, surface, normalize(surface), entity_type, entity_type, 1.0, "deterministic"),
            )
            referent_key = (normalize(surface), entity_type)
            referent_id = referent_cache.get(referent_key)
            if referent_id is None:
                referent_id = store.upsert_referent(run_id, surface, entity_type)
                referent_cache[referent_key] = referent_id
            store.execute(
                "INSERT OR IGNORE INTO mention_referents(mention_id, referent_id, link_status, confidence) VALUES (?, ?, ?, ?)",
                (mention_id, referent_id, "candidate", 1.0),
            )
            mentions_for_sentence.append((surface, mention_id, referent_id))

        frame_info = frame_predicate(sentence.text)
        if frame_info:
            predicate, trigger = frame_info
            frame_id = stable_id("frm", run_id, sentence.sentence_id, predicate, trigger)
            store.execute(
                "INSERT OR IGNORE INTO frames(frame_id, run_id, context_id, predicate, predicate_norm, trigger_surface, confidence, source, span_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (frame_id, run_id, context_id, predicate, normalize(predicate), trigger, 0.8, "deterministic", span_id),
            )
            for index, (surface, mention_id, referent_id) in enumerate(mentions_for_sentence[:4]):
                role = "agent" if index == 0 else "theme"
                store.execute(
                    "INSERT OR IGNORE INTO frame_arguments(argument_id, frame_id, role, mention_id, referent_id, surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (stable_id("arg", frame_id, role, mention_id), frame_id, role, mention_id, referent_id, surface, 0.7),
                )

        state_info = temporal_state(sentence.text)
        if state_info:
            temporal_value, state_value = state_info
            referent_id = mentions_for_sentence[0][2] if mentions_for_sentence else None
            store.execute(
                """
                INSERT OR IGNORE INTO temporal_edges(
                  edge_id, run_id, source_span_id, referent_id, context_id, relation, temporal_value, state_value, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stable_id("tmp", run_id, sentence.sentence_id, temporal_value, state_value),
                    run_id,
                    span_id,
                    referent_id,
                    context_id,
                    "state_at",
                    temporal_value,
                    state_value,
                    0.85,
                ),
            )
            store.execute(
                """
                INSERT OR IGNORE INTO context_carriers(
                  carrier_id, run_id, context_id, document_id, source_span_id, carrier_kind, carrier_surface,
                  temporal_value, temporal_value_type, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stable_id("carrier", run_id, sentence.sentence_id, "event_time", temporal_value),
                    run_id,
                    context_id,
                    sentence.document_id,
                    span_id,
                    "temporal_expression",
                    temporal_value,
                    temporal_value,
                    "event_time",
                    0.85,
                ),
            )

        if _is_structural_heading(sentence.text):
            section_anchor = clean_extracted_value(sentence.text)
            section_anchor_by_document[sentence.document_id] = section_anchor
            section_group_by_document[sentence.document_id] = stable_id("section_group", sentence.document_id, section_anchor)
        else:
            pending_label_heading = _label_heading_value(sentence.text)

        deterministic_relations = extract_relations(sentence.text)
        cells = _table_cells(sentence.text)
        if cells:
            current_header = table_headers_by_document.get(sentence.document_id)
            if current_header and len(cells) == len(current_header) and cells != current_header:
                deterministic_relations.extend(_table_header_relations(sentence, current_header, cells))
            elif _looks_like_table_header(cells):
                table_headers_by_document[sentence.document_id] = cells

        for relation in deterministic_relations:
            metadata = {
                **relation.metadata,
                "sentence_group": stable_id("sentence_group", sentence.sentence_id),
            }
            if "record_group" not in metadata:
                metadata["record_group"] = metadata["sentence_group"]
            if _relation_inherits_heading(sentence.text, deterministic_relations):
                section_group = section_group_by_document.get(sentence.document_id)
                section_anchor = section_anchor_by_document.get(sentence.document_id)
                if section_group and section_anchor:
                    metadata["record_group"] = section_group
                    metadata["section_anchor"] = section_anchor
            elif not _starts_new_structural_record(sentence.text):
                section_anchor = section_anchor_by_document.get(sentence.document_id)
                if section_anchor:
                    metadata["section_anchor"] = section_anchor
            elif "section_anchor" not in metadata:
                section_anchor = section_anchor_by_document.get(sentence.document_id)
                prefix = re.split(r"[:=|\t]", sentence.text, maxsplit=1)[0]
                if section_anchor and not any(len(phrase.split()) >= 2 for phrase in capitalized_phrases(prefix)):
                    metadata["section_anchor"] = section_anchor
            relation_id = stable_id(
                "rel",
                run_id,
                sentence.sentence_id,
                relation.relation_type,
                relation.predicate,
                relation.subject,
                relation.object,
                relation.value,
            )
            store.execute(
                """
                INSERT OR IGNORE INTO relations(
                  relation_id, run_id, relation_type, subject, subject_norm, predicate, predicate_norm,
                  object, object_norm, value, value_norm, source_span_id, context_id, confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation_id,
                    run_id,
                    relation.relation_type,
                    relation.subject,
                    normalize(relation.subject),
                    relation.predicate,
                    normalize(relation.predicate),
                    relation.object,
                    normalize(relation.object),
                    relation.value,
                    normalize(relation.value),
                    span_id,
                    context_id,
                    relation.confidence,
                    json.dumps(metadata, sort_keys=True),
                ),
            )
            condition = _condition_from_deterministic_relation(relation, sentence.text)
            if condition is not None and condition.arguments:
                condition_frame_id = stable_id(
                    "frm",
                    run_id,
                    sentence.sentence_id,
                    "condition",
                    relation.relation_type,
                    relation.predicate,
                    relation.subject,
                    relation.object,
                    relation.value,
                )
                store.execute(
                    "INSERT OR IGNORE INTO frames(frame_id, run_id, context_id, predicate, predicate_norm, trigger_surface, confidence, source, span_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        condition_frame_id,
                        run_id,
                        context_id,
                        condition.predicate,
                        normalize(condition.predicate),
                        condition.predicate,
                        condition.confidence,
                        "deterministic_relation",
                        span_id,
                    ),
                )
                for arg_index, argument in enumerate(condition.arguments):
                    arg_referent_id = store.upsert_referent(run_id, argument.value, argument.value_type)
                    store.execute(
                        "INSERT OR IGNORE INTO frame_arguments(argument_id, frame_id, role, mention_id, referent_id, surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            stable_id("arg", condition_frame_id, arg_index, argument.role, argument.value),
                            condition_frame_id,
                            argument.role,
                            None,
                            arg_referent_id,
                            argument.value,
                            condition.confidence,
                        ),
                    )
                    normalized_argument = normalize(argument.value)
                    for existing_surface, _mention_id, existing_referent_id in mentions_for_sentence:
                        if normalize(existing_surface) == normalized_argument and existing_referent_id != arg_referent_id:
                            store.execute(
                                """
                                INSERT OR IGNORE INTO identity_hypotheses(
                                  hypothesis_id, run_id, left_referent_id, right_referent_id,
                                  relation, evidence, confidence, source
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    stable_id("idh", run_id, existing_referent_id, arg_referent_id, sentence.sentence_id),
                                    run_id,
                                    existing_referent_id,
                                    arg_referent_id,
                                    "same_surface",
                                    argument.value,
                                    0.82,
                                    "deterministic_surface",
                                ),
                            )

        if not _is_structural_heading(sentence.text) and pending_label_heading:
            section_anchor_by_document[sentence.document_id] = pending_label_heading
            section_group_by_document[sentence.document_id] = stable_id("section_group", sentence.document_id, pending_label_heading)

        if use_semantic_frames and semantic_client is not None:
            model_frames, _frame_result = _grounded_model_frames(sentence, semantic_client, semantic_cache)
            for index, frame in enumerate(model_frames):
                condition = frame_from_model_dict(frame)
                if condition is None or condition.evidence_text not in sentence.text:
                    continue
                frame_type = condition.frame_type
                predicate = condition.predicate or frame_type
                evidence_text = condition.evidence_text
                modality = condition.modality
                polarity = condition.polarity
                temporal_text = condition.temporal_text
                semantic_context_id = context_id
                if modality != "asserted":
                    context_key = f"modality:{modality}"
                    semantic_context_id = context_by_kind.get(context_key)
                    if semantic_context_id is None:
                        semantic_context_id = stable_id("ctx", run_id, context_key)
                        context_by_kind[context_key] = semantic_context_id
                        store.execute(
                            "INSERT INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (semantic_context_id, run_id, context_key, context_id, None, evidence_text, condition.confidence),
                        )
                semantic_frame_id = stable_id("frm", run_id, sentence.sentence_id, "model", index, predicate, evidence_text)
                store.execute(
                    "INSERT OR IGNORE INTO frames(frame_id, run_id, context_id, predicate, predicate_norm, trigger_surface, confidence, source, span_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        semantic_frame_id,
                        run_id,
                        semantic_context_id,
                        predicate,
                        normalize(predicate),
                        predicate,
                        condition.confidence,
                        "local_model",
                        span_id,
                    ),
                )
                group = stable_id("semantic_group", semantic_frame_id)
                frame_metadata = {
                    "frame_type": frame_type,
                    "modality": modality,
                    "polarity": polarity,
                    "temporal_text": temporal_text,
                    "record_group": group,
                    "source": "local_model",
                }
                store.execute(
                    """
                    INSERT OR IGNORE INTO relations(
                      relation_id, run_id, relation_type, subject, subject_norm, predicate, predicate_norm,
                      object, object_norm, value, value_norm, source_span_id, context_id, confidence, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("rel", run_id, semantic_frame_id, "semantic_frame"),
                        run_id,
                        "semantic_frame",
                        frame_type,
                        normalize(frame_type),
                        predicate,
                        normalize(predicate),
                        "",
                        "",
                        evidence_text,
                        normalize(evidence_text),
                        span_id,
                        semantic_context_id,
                        condition.confidence,
                        json.dumps(frame_metadata, sort_keys=True),
                    ),
                )
                for arg_index, argument in enumerate(condition.arguments):
                    role = argument.role
                    surface = argument.value
                    arg_referent_id = store.upsert_referent(run_id, surface, argument.value_type)
                    store.execute(
                        "INSERT OR IGNORE INTO frame_arguments(argument_id, frame_id, role, mention_id, referent_id, surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            stable_id("arg", semantic_frame_id, arg_index, role, surface),
                            semantic_frame_id,
                            role,
                            None,
                            arg_referent_id,
                            surface,
                            condition.confidence,
                        ),
                    )
                    relation_metadata = {
                        **frame_metadata,
                        "argument_role": role,
                        "argument_value_type": argument.value_type,
                    }
                    store.execute(
                        """
                        INSERT OR IGNORE INTO relations(
                          relation_id, run_id, relation_type, subject, subject_norm, predicate, predicate_norm,
                          object, object_norm, value, value_norm, source_span_id, context_id, confidence, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            stable_id("rel", run_id, semantic_frame_id, "arg", arg_index, role, surface),
                            run_id,
                            "semantic_argument",
                            role,
                            normalize(role),
                            predicate,
                            normalize(predicate),
                            frame_type,
                            normalize(frame_type),
                            surface,
                            normalize(surface),
                            span_id,
                            semantic_context_id,
                            condition.confidence,
                            json.dumps(relation_metadata, sort_keys=True),
                        ),
                    )
                    normalized_argument = normalize(surface)
                    for existing_surface, _mention_id, existing_referent_id in mentions_for_sentence:
                        if normalize(existing_surface) == normalized_argument and existing_referent_id != arg_referent_id:
                            store.execute(
                                """
                                INSERT OR IGNORE INTO identity_hypotheses(
                                  hypothesis_id, run_id, left_referent_id, right_referent_id,
                                  relation, evidence, confidence, source
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    stable_id("idh", run_id, existing_referent_id, arg_referent_id, semantic_frame_id),
                                    run_id,
                                    existing_referent_id,
                                    arg_referent_id,
                                    "same_surface",
                                    surface,
                                    min(0.9, condition.confidence),
                                    "local_model_frame",
                                ),
                            )
                if temporal_text:
                    store.execute(
                        """
                        INSERT OR IGNORE INTO temporal_edges(
                          edge_id, run_id, source_span_id, referent_id, context_id, relation, temporal_value, state_value, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            stable_id("tmp", run_id, semantic_frame_id, temporal_text),
                            run_id,
                            span_id,
                            None,
                            semantic_context_id,
                            "frame_temporal_scope",
                            temporal_text,
                            "",
                            condition.confidence,
                        ),
                    )

    metrics = {
        "documents": len(documents),
        "sentences": len(sentences),
        **store.counts(),
    }
    if created_store:
        store.create_indexes()
    store.finish_run(run_id, metrics)
    return store, run_id, documents, sentences
