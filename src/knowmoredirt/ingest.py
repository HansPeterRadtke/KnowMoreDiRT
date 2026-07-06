"""Raw-text ingestion into the internal DSPG store."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .context_budget import context_relative_budget
from .drs import DiscourseArgument, DiscourseCondition, frame_from_model_dict
from .extractors import capitalized_phrases, identifiers, urls
from .model import LocalModelUnavailableError
from .models import Document, Sentence
from .model_planner import (
    call_model_chunk_drs,
    call_model_chunk_frames,
    chunk_drs_cache_context,
    chunk_frame_cache_context,
    default_chunk_drs_n_predict,
)
from .relations import ExtractedRelation, extract_relations, transcript_turn_parts
from .scanner import scan_folder
from .semantic_cache import SemanticFrameCache
from .store import DSPGStore, stable_id
from .text import clean_extracted_value, normalize, split_units, text_quality_metrics, tokenize


TABLE_SPLIT_RE = re.compile(r"\s*(?:\||\t)\s*")
PROGRESS_TRUE_VALUES = {"1", "true", "yes", "on"}
FIRST_PERSON_REFERENCE_NORMS = {"i", "me", "myself", "we", "us", "ourselves"}
STRUCTURAL_SPEAKER_LABEL_NORMS = {"author", "from", "sender", "speaker"}


def _progress_enabled() -> bool:
    return os.environ.get("KMD_PROGRESS", "").strip().lower() in PROGRESS_TRUE_VALUES or os.environ.get(
        "KMD_EVAL_PROGRESS", ""
    ).strip().lower() in PROGRESS_TRUE_VALUES


def _log_progress(message: str) -> None:
    if _progress_enabled():
        print(message, flush=True)


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in PROGRESS_TRUE_VALUES


def _identifier_bearing_discourse_span(text: str, quality: dict[str, object]) -> bool:
    if not (identifiers(text) or urls(text)):
        return False
    try:
        token_count = int(quality.get("token_count") or 0)
        char_count = int(quality.get("char_count") or 0)
        symbol_ratio = float(quality.get("symbol_ratio") or 0.0)
    except (TypeError, ValueError):
        return False
    return 5 <= token_count <= 40 and char_count <= 600 and symbol_ratio <= 0.18


def _model_semantic_skip_reason(quality: dict[str, object], text: str = "") -> str:
    if (
        str(quality.get("semantic_quality") or "") == "word_salad"
        and not bool(quality.get("low_semantic_noise"))
        and _identifier_bearing_discourse_span(text, quality)
    ):
        return ""
    if bool(quality.get("low_semantic_noise")) or str(quality.get("semantic_quality") or "") in {
        "base64_or_hex_blob",
        "multilingual_word_salad",
        "plausible_babble",
        "word_salad",
    }:
        return "skipped_noise"
    return ""


def _skip_model_semantics_for_quality(quality: dict[str, object], text: str = "") -> bool:
    return bool(_model_semantic_skip_reason(quality, text))


def _attempt_materialized(row: Any | None) -> bool:
    if row is None or not bool(row["accepted"]) or not bool(row["materialized"]):
        return False
    try:
        metadata = json.loads(str(row["metadata_json"] or "{}"))
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        return True
    inserted = metadata.get("materialized", {}).get("inserted", {}) if isinstance(metadata, dict) else {}
    if isinstance(inserted, dict) and "drs_conditions" in inserted:
        try:
            return int(inserted.get("drs_conditions") or 0) > 0
        except (TypeError, ValueError):
            return False
    return True


def _attempt_was_nonrequest_failure(row: Any | None) -> bool:
    if row is None:
        return False
    reason = str(row["reason"] or "")
    if reason in {"", "request_failed"} or bool(row["materialized"]):
        return False
    if bool(row["accepted"]):
        try:
            metadata = json.loads(str(row["metadata_json"] or "{}"))
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            metadata = {}
        materialized = metadata.get("materialized", {}) if isinstance(metadata, dict) else {}
        inserted = materialized.get("inserted", {}) if isinstance(materialized, dict) else {}
        if (
            reason == "materialized"
            or bool(materialized.get("accepted"))
            or (isinstance(inserted, dict) and any(int(value or 0) > 0 for value in inserted.values()))
        ):
            return False
    return True


def _raise_model_request_failed(result: dict[str, Any], operation: str) -> None:
    reason = str(result.get("reason") or "")
    fatal_reasons = {"request_failed", "invalid_json", "schema_validation_failed", "grounding_validation_failed"}
    if reason not in fatal_reasons:
        return
    _log_progress(
        "kmd-ingest model_structured_failure "
        f"operation={operation} "
        f"reason={reason} "
        f"error={str(result.get('error') or reason)[:300]}"
    )
    cache_context = result.get("cache_context") if isinstance(result.get("cache_context"), dict) else {}
    try:
        cache_context_text = json.dumps(cache_context, sort_keys=True, default=str)[:4000]
    except Exception:
        cache_context_text = str(cache_context)[:4000]
    raise LocalModelUnavailableError(
        "KnowMoreDiRT requires successful native structured local-model output during initialize(folder_path). "
        f"Local model structured output failed during {operation}: reason={reason}; error={result.get('error') or reason}. "
        f"cache_context={cache_context_text}",
        cache_context=cache_context,
    )


def _raise_model_materialization_failed(
    drs_result: dict[str, Any],
    materialized: dict[str, Any],
    operation: str,
) -> None:
    if not bool(drs_result.get("accepted")) or bool(materialized.get("accepted")):
        return
    reason = str(materialized.get("reason") or drs_result.get("reason") or "materialization_failed")
    errors = materialized.get("errors") if isinstance(materialized.get("errors"), list) else []
    grounding_failures = materialized.get("grounding_failures") if isinstance(materialized.get("grounding_failures"), list) else []
    _log_progress(
        "kmd-ingest model_materialization_failed "
        f"operation={operation} "
        f"reason={reason} "
        f"errors={str(errors[:5])[:300]} "
        f"grounding_failures={str(grounding_failures[:5])[:300]}"
    )
    cache_context = drs_result.get("cache_context") if isinstance(drs_result.get("cache_context"), dict) else {}
    try:
        cache_context_text = json.dumps(cache_context, sort_keys=True, default=str)[:4000]
    except Exception:
        cache_context_text = str(cache_context)[:4000]
    raise LocalModelUnavailableError(
        "KnowMoreDiRT requires model DRS output to materialize during initialize(folder_path). "
        f"Local model DRS materialization failed during {operation}: reason={reason}; "
        f"errors={errors[:20]}; grounding_failures={grounding_failures[:20]}. "
        f"cache_context={cache_context_text}",
        cache_context=cache_context,
    )


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


def _table_cells(text: str) -> list[str]:
    if "|" not in text and "\t" not in text:
        return []
    cells = [clean_extracted_value(cell) for cell in TABLE_SPLIT_RE.split(text)]
    return [cell for cell in cells if cell]


TABLE_HEADER_LABEL_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[_-][a-z][a-z0-9]*)+$")


def _is_table_header_label_identifier(cell: str) -> bool:
    return bool(TABLE_HEADER_LABEL_ID_RE.fullmatch(clean_extracted_value(cell).strip()))


def _looks_like_table_header(cells: list[str]) -> bool:
    if len(cells) < 2:
        return False
    if not all(re.search(r"[A-Za-z]", cell) for cell in cells):
        return False
    return not any(urls(cell) or (identifiers(cell) and not _is_table_header_label_identifier(cell)) for cell in cells)


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
    return _label_heading_value_from_relations(text, extract_relations(text))


def _label_heading_value_from_relations(text: str, relations: list[ExtractedRelation]) -> str:
    if "|" in text or "\t" in text or "://" in text:
        return ""
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
    skip_reason = _model_semantic_skip_reason(quality, sentence.text)
    if skip_reason:
        return [], {"source": skip_reason, "reason": skip_reason}
    cache_context = chunk_frame_cache_context(semantic_client, rel_path=sentence.rel_path, chunk_text=sentence.text)
    cached = semantic_cache.get(sentence.text, context=cache_context) if semantic_cache else None
    if cached is not None:
        frames = [frame for frame in cached.get("frames", []) if isinstance(frame, dict)]
        metadata = cached.get("metadata") if isinstance(cached.get("metadata"), dict) else {}
        accepted = bool(metadata.get("accepted", True))
        return frames, {
            "source": "cache",
            "frame_count": len(frames),
            "accepted": accepted,
            "reason": str(metadata.get("reason") or ""),
            "prompt_hash": metadata.get("prompt_hash"),
            "output_hash": metadata.get("output_hash"),
            "context_budget": metadata.get("context_budget"),
        }
    result = call_model_chunk_frames(sentence.text, semantic_client, rel_path=sentence.rel_path)
    _raise_model_request_failed(result, "chunk frame ingest")
    frames = [frame for frame in result.get("frames", []) if isinstance(frame, dict)] if result.get("accepted") else []
    cacheable_failure = result.get("reason") in {"invalid_json", "schema_validation_failed", "grounding_validation_failed"}
    if semantic_cache is not None and (result.get("accepted") or cacheable_failure):
        semantic_cache.put(
            sentence.text,
            frames,
            {
                "rel_path": sentence.rel_path,
                "accepted": bool(result.get("accepted")),
                "reason": str(result.get("reason") or ""),
                "prompt_hash": result.get("prompt_hash"),
                "output_hash": result.get("output_hash"),
                "context_budget": result.get("context_budget"),
            },
            context=cache_context,
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


def _link_first_person_referents_to_speaker_surface(
    store: DSPGStore,
    run_id: str,
    source_span_id: str,
    speaker_surface: str,
    evidence_surface: str,
    *,
    source: str,
    confidence: float,
) -> int:
    speaker = clean_extracted_value(speaker_surface)
    if not speaker:
        return 0
    context_row = store.execute(
        """
        SELECT context_id
        FROM context_assignments
        WHERE run_id=? AND applies_to_type='source_span' AND applies_to_id=?
        LIMIT 1
        """,
        (run_id, source_span_id),
    ).fetchone()
    context_id = str(context_row["context_id"] or "") if context_row is not None else ""
    speaker_ref = store.upsert_referent(run_id, speaker, mention_entity_type(speaker))
    rows = store.execute(
        """
        SELECT referent_id
        FROM drs_referents
        WHERE run_id=? AND source_span_id=?
        """,
        (run_id, source_span_id),
    ).fetchall()
    inserted = 0
    for row in rows:
        pronoun_ref = str(row["referent_id"] or "")
        if not pronoun_ref or pronoun_ref == speaker_ref:
            continue
        surface_row = store.execute(
            "SELECT canonical_label_norm FROM referents WHERE referent_id=?",
            (pronoun_ref,),
        ).fetchone()
        surface_norm = str(surface_row["canonical_label_norm"] or "") if surface_row is not None else ""
        if surface_norm not in FIRST_PERSON_REFERENCE_NORMS:
            continue
        store.execute(
            """
            INSERT OR IGNORE INTO identity_hypotheses(
              hypothesis_id, run_id, source_span_id, context_id, drs_box_id, box_external_id,
              left_referent_id, right_referent_id, relation, evidence, confidence, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_id("idh", run_id, source_span_id, source, speaker_ref, pronoun_ref),
                run_id,
                source_span_id,
                context_id,
                None,
                None,
                speaker_ref,
                pronoun_ref,
                "coreference",
                evidence_surface,
                confidence,
                source,
            ),
        )
        inserted += 1
    return inserted


def _link_labeled_turn_speaker_referents(
    store: DSPGStore,
    run_id: str,
    source_span_id: str,
    sentence: Sentence,
) -> int:
    speaker, _utterance = transcript_turn_parts(sentence.text)
    return _link_first_person_referents_to_speaker_surface(
        store,
        run_id,
        source_span_id,
        speaker,
        sentence.text,
        source="deterministic_speaker_turn",
        confidence=0.9,
    )


def _structural_speaker_surface_from_relations(relations: list[ExtractedRelation]) -> str:
    for relation in relations:
        if relation.relation_type != "label_value" or not relation.value:
            continue
        label_norm = normalize(relation.subject or relation.predicate)
        if label_norm in STRUCTURAL_SPEAKER_LABEL_NORMS:
            return clean_extracted_value(relation.value)
    return ""


def _scan_pack_unit_count() -> int:
    configured = os.environ.get("KMD_SCAN_PACK_MAX_UNITS", "").strip()
    if configured:
        try:
            return max(0, int(configured))
        except ValueError:
            pass
    return 0


def _scan_pack_unit_chars(semantic_client: Any | None) -> int:
    enabled = os.environ.get("KMD_SCAN_PACK_UNITS", "1").strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return 0
    configured = os.environ.get("KMD_SCAN_PACK_MAX_CHARS", "").strip()
    if configured:
        try:
            return max(0, int(configured))
        except ValueError:
            pass
    return _scan_unit_max_chars(semantic_client)


def _scan_unit_max_chars(semantic_client: Any | None) -> int:
    configured = os.environ.get("KMD_SCAN_UNIT_MAX_CHARS", "").strip()
    if configured:
        try:
            return max(0, int(configured))
        except ValueError:
            pass
    if semantic_client is not None and hasattr(semantic_client, "context_size"):
        try:
            context_size = int(semantic_client.context_size())
        except Exception:
            context_size = 0
        if context_size > 0:
            budget = context_relative_budget(
                context_size,
                output_ratio_names=("KMD_SCAN_UNIT_OUTPUT_RATIO", "KMD_CHUNK_DRS_OUTPUT_RATIO"),
                safety_ratio_names=("KMD_SCAN_UNIT_SAFETY_RATIO",),
                overhead_ratio_names=("KMD_SCAN_UNIT_OVERHEAD_RATIO",),
                chars_per_token_names=("KMD_SCAN_UNIT_CHARS_PER_TOKEN",),
            )
            return budget.safe_input_chars
    return 0




def _drs_adaptive_split_caps() -> list[int]:
    configured = os.environ.get("KMD_DRS_ADAPTIVE_SPLIT_UNIT_CAPS", "32,8,1").strip()
    caps: list[int] = []
    for item in configured.split(","):
        try:
            value = int(item.strip())
        except ValueError:
            continue
        if value > 0:
            caps.append(value)
    return caps or [32, 8, 1]


def _drs_adaptive_split_enabled() -> bool:
    return os.environ.get("KMD_DRS_ADAPTIVE_SPLIT", "1").strip().lower() not in {"0", "false", "no", "off"}


def _drs_adaptive_retryable_failure(result: dict[str, Any]) -> bool:
    return str(result.get("reason") or "") in {"invalid_json", "schema_validation_failed", "grounding_validation_failed"}


def _drs_adaptive_split_sentences(sentence: Sentence, *, depth: int) -> list[Sentence]:
    if not _drs_adaptive_split_enabled():
        return []
    caps = _drs_adaptive_split_caps()
    if depth >= len(caps):
        return []
    cap = max(1, caps[depth])
    units = split_units(sentence.text)
    if len(units) <= 1:
        return []
    groups: list[list[tuple[int, int, str]]] = []
    current: list[tuple[int, int, str]] = []
    for unit in units:
        if current and len(current) >= cap:
            groups.append(current)
            current = []
        current.append(unit)
    if current:
        groups.append(current)
    sub_sentences: list[Sentence] = []
    for index, group in enumerate(groups):
        rel_start = group[0][0]
        rel_end = group[-1][1]
        segment = sentence.text[rel_start:rel_end]
        stripped = segment.strip()
        if not stripped:
            continue
        leading = len(segment) - len(segment.lstrip())
        trailing = len(segment.rstrip())
        abs_start = sentence.char_start + rel_start + leading
        abs_end = sentence.char_start + rel_start + trailing
        if abs_start >= abs_end:
            continue
        sub_sentences.append(
            Sentence(
                sentence_id=stable_id("adaptive_sentence", sentence.sentence_id, depth, index, abs_start, abs_end),
                document_id=sentence.document_id,
                rel_path=sentence.rel_path,
                text=stripped,
                order=sentence.order * 100000 + (depth + 1) * 1000 + index,
                char_start=abs_start,
                char_end=abs_end,
            )
        )
    if len(sub_sentences) <= 1 and (not sub_sentences or sub_sentences[0].text == sentence.text):
        return []
    return sub_sentences


def _register_adaptive_drs_subspan(store: DSPGStore, sentence: Sentence) -> str:
    token_estimate = max(1, len(tokenize(sentence.text)))
    chunk_id = stable_id("chunk", sentence.sentence_id)
    store.execute(
        "INSERT OR IGNORE INTO chunks(chunk_id, document_id, chunk_order, char_start, char_end, text, token_estimate) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (chunk_id, sentence.document_id, sentence.order, sentence.char_start, sentence.char_end, sentence.text, token_estimate),
    )
    span_id = stable_id("span", sentence.sentence_id, "adaptive_drs_sentence")
    store.execute(
        "INSERT OR IGNORE INTO source_spans(span_id, document_id, chunk_id, char_start, char_end, surface, surface_norm, span_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            span_id,
            sentence.document_id,
            chunk_id,
            sentence.char_start,
            sentence.char_end,
            sentence.text,
            normalize(sentence.text),
            "adaptive_drs_sentence",
        ),
    )
    return span_id


def _ingest_adaptive_drs_split(
    store: DSPGStore,
    run_id: str,
    sentence: Sentence,
    semantic_client: Any,
    semantic_index: int,
    semantic_total: int,
    ingest_started: float,
    *,
    failure_result: dict[str, Any],
    adaptive_depth: int,
    refresh_empty_compact_legacy: bool,
    structural_speaker_surface: str,
    structural_speaker_evidence: str,
) -> int | None:
    sub_sentences = _drs_adaptive_split_sentences(sentence, depth=adaptive_depth)
    if not sub_sentences:
        return None
    _log_progress(
        "kmd-ingest drs_adaptive_split "
        f"source={sentence.rel_path}:{sentence.order} "
        f"reason={str(failure_result.get('reason') or '')} "
        f"depth={adaptive_depth} "
        f"subchunks={len(sub_sentences)} "
        f"elapsed={time.monotonic() - ingest_started:.1f}s"
    )
    for sub_sentence in sub_sentences:
        sub_span_id = _register_adaptive_drs_subspan(store, sub_sentence)
        semantic_index = _ingest_model_drs_for_sentence(
            store,
            run_id,
            sub_sentence,
            sub_span_id,
            semantic_client,
            semantic_index,
            semantic_total,
            ingest_started,
            refresh_empty_compact_legacy=refresh_empty_compact_legacy,
            structural_speaker_surface=structural_speaker_surface,
            structural_speaker_evidence=structural_speaker_evidence,
            adaptive_depth=adaptive_depth + 1,
        )
    return semantic_index

def _ingest_model_drs_for_sentence(
    store: DSPGStore,
    run_id: str,
    sentence: Sentence,
    span_id: str,
    semantic_client: Any,
    semantic_index: int,
    semantic_total: int,
    ingest_started: float,
    refresh_empty_compact_legacy: bool = False,
    structural_speaker_surface: str = "",
    structural_speaker_evidence: str = "",
    adaptive_depth: int = 0,
) -> int:
    semantic_index += 1
    skip_reason = (
        _model_semantic_skip_reason(text_quality_metrics(sentence.text), sentence.text)
        if _env_true("KMD_ALLOW_PREMODEL_SEMANTIC_SKIP")
        else ""
    )
    if skip_reason:
        _log_progress(
            "kmd-ingest drs_done "
            f"chunk={semantic_index}/{semantic_total} "
            f"source={sentence.rel_path}:{sentence.order} "
            "accepted=False "
            "materialized=False "
            f"reason={skip_reason} "
            "model_elapsed=0.0 "
            f"elapsed={time.monotonic() - ingest_started:.1f}s"
        )
        return semantic_index
    drs_n_predict = default_chunk_drs_n_predict(semantic_client, sentence.text)
    drs_cache_context = chunk_drs_cache_context(
        semantic_client,
        n_predict=drs_n_predict,
        rel_path=sentence.rel_path,
        chunk_text=sentence.text,
    )
    drs_cache_key = stable_id("drs_attempt_context", json.dumps(drs_cache_context, sort_keys=True, default=str))
    previous_attempt = store.execute(
        """
        SELECT accepted, materialized, reason, metadata_json
        FROM model_attempts
        WHERE run_id=? AND source_span_id=? AND task=? AND source=? AND cache_key=?
        LIMIT 1
        """,
        (run_id, span_id, "chunk_drs", "local_model_drs", drs_cache_key),
    ).fetchone()
    existing_drs = store.execute(
        """
        SELECT COUNT(*)
        FROM drs_conditions
        WHERE run_id=? AND source_span_id=? AND source='local_model_drs'
        """,
        (run_id, span_id),
    ).fetchone()[0]
    if existing_drs and _attempt_materialized(previous_attempt):
        _log_progress(
            "kmd-ingest drs_done "
            f"chunk={semantic_index}/{semantic_total} "
            f"source={sentence.rel_path}:{sentence.order} "
            "accepted=True "
            "materialized=True "
            "reason=already_materialized "
            "model_elapsed=0.0 "
            f"elapsed={time.monotonic() - ingest_started:.1f}s"
        )
        return semantic_index
    replaced: dict[str, int] = {}
    if existing_drs:
        replaced = store.delete_drs_materialization_for_span(
            run_id,
            span_id,
            source="local_model_drs",
        )
        inactive_attempts = store.deactivate_other_model_attempt_materializations(
            run_id,
            span_id,
            "chunk_drs",
            "local_model_drs",
            drs_cache_key,
        )
        if inactive_attempts:
            replaced["model_attempts"] = inactive_attempts
    if (
        _attempt_was_nonrequest_failure(previous_attempt)
        and not _env_true("KMD_DRS_RETRY_FAILED_ATTEMPTS")
    ):
        _log_progress(
            "kmd-ingest drs_done "
            f"chunk={semantic_index}/{semantic_total} "
            f"source={sentence.rel_path}:{sentence.order} "
            f"accepted={bool(previous_attempt['accepted'])} "
            f"materialized={bool(previous_attempt['materialized'])} "
            "reason=previous_attempt "
            f"replaced_prior_rows={sum(replaced.values()) if replaced else 0} "
            "model_elapsed=0.0 "
            f"elapsed={time.monotonic() - ingest_started:.1f}s"
        )
        return semantic_index
    _log_progress(
        "kmd-ingest drs_start "
        f"chunk={semantic_index}/{semantic_total} "
        f"source={sentence.rel_path}:{sentence.order} "
        f"elapsed={time.monotonic() - ingest_started:.1f}s"
    )
    drs_result = call_model_chunk_drs(
        sentence.text,
        semantic_client,
        rel_path=sentence.rel_path,
        n_predict=drs_n_predict,
        refresh_empty_compact_legacy=refresh_empty_compact_legacy,
    )
    if _drs_adaptive_retryable_failure(drs_result):
        adaptive_index = _ingest_adaptive_drs_split(
            store,
            run_id,
            sentence,
            semantic_client,
            semantic_index,
            semantic_total,
            ingest_started,
            failure_result=drs_result,
            adaptive_depth=adaptive_depth,
            refresh_empty_compact_legacy=refresh_empty_compact_legacy,
            structural_speaker_surface=structural_speaker_surface,
            structural_speaker_evidence=structural_speaker_evidence,
        )
        if adaptive_index is not None:
            _log_progress(
                "kmd-ingest drs_done "
                f"chunk={semantic_index}/{semantic_total} "
                f"source={sentence.rel_path}:{sentence.order} "
                "accepted=False "
                "materialized=True "
                f"reason=adaptive_split:{str(drs_result.get('reason') or '')} "
                f"model_elapsed={float(drs_result.get('elapsed') or 0.0):.1f}s "
                f"elapsed={time.monotonic() - ingest_started:.1f}s"
            )
            return adaptive_index
    _raise_model_request_failed(drs_result, "chunk DRS ingest")
    actual_drs_cache_context = (
        drs_result.get("cache_context") if isinstance(drs_result.get("cache_context"), dict) else drs_cache_context
    )
    actual_drs_cache_key = stable_id(
        "drs_attempt_context",
        json.dumps(actual_drs_cache_context, sort_keys=True, default=str),
    )
    materialized = {"accepted": False, "reason": "not_attempted", "inserted": {}}
    if drs_result.get("accepted") and isinstance(drs_result.get("drs"), dict):
        materialized = store.materialize_drs_payload(
            run_id,
            span_id,
            sentence.text,
            {"drs": drs_result["drs"]},
            source="local_model_drs",
        )
        _raise_model_materialization_failed(drs_result, materialized, "chunk DRS ingest")
        if materialized.get("accepted"):
            linked_speakers = _link_labeled_turn_speaker_referents(store, run_id, span_id, sentence)
            linked_structural_speakers = _link_first_person_referents_to_speaker_surface(
                store,
                run_id,
                span_id,
                structural_speaker_surface,
                structural_speaker_evidence,
                source="deterministic_structural_speaker",
                confidence=0.84,
            )
            if linked_speakers:
                materialized = {
                    **materialized,
                    "inserted": {
                        **dict(materialized.get("inserted") or {}),
                        "speaker_turn_identity_hypotheses": linked_speakers,
                    },
                }
            if linked_structural_speakers:
                materialized = {
                    **materialized,
                    "inserted": {
                        **dict(materialized.get("inserted") or {}),
                        "structural_speaker_identity_hypotheses": linked_structural_speakers,
                    },
                }
    store.execute(
        """
        INSERT OR REPLACE INTO model_attempts(
          attempt_id, run_id, source_span_id, task, source, cache_key, accepted, materialized,
          reason, prompt_hash, output_hash, elapsed, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            stable_id("attempt", run_id, span_id, "chunk_drs", "local_model_drs", drs_cache_key),
            run_id,
            span_id,
            "chunk_drs",
            "local_model_drs",
            drs_cache_key,
            int(bool(drs_result.get("accepted"))),
            int(bool(materialized.get("accepted"))),
            str(drs_result.get("reason") or materialized.get("reason") or ""),
            str(drs_result.get("prompt_hash") or ""),
            str(drs_result.get("output_hash") or ""),
            float(drs_result.get("elapsed") or 0.0),
            json.dumps(
                {
                    "cache_context": drs_cache_context,
                    "actual_cache_context": actual_drs_cache_context,
                    "actual_cache_key": actual_drs_cache_key,
                    "context_budget": drs_result.get("context_budget"),
                    "materialized": materialized,
                    "replaced_prior_rows": replaced,
                },
                sort_keys=True,
                default=str,
            ),
        ),
    )
    _log_progress(
        "kmd-ingest drs_done "
        f"chunk={semantic_index}/{semantic_total} "
        f"source={sentence.rel_path}:{sentence.order} "
        f"accepted={bool(drs_result.get('accepted'))} "
        f"materialized={bool(materialized.get('accepted'))} "
        f"reason={str(drs_result.get('reason') or materialized.get('reason') or '')} "
        f"model_elapsed={float(drs_result.get('elapsed') or 0.0):.1f}s "
        f"elapsed={time.monotonic() - ingest_started:.1f}s"
    )
    return semantic_index


def ingest_folder(
    folder_path: str | Path,
    store: DSPGStore | None = None,
    *,
    semantic_client: Any | None = None,
    use_semantic_frames: bool = False,
    use_drs_semantics: bool = False,
    semantic_cache: SemanticFrameCache | None = None,
) -> tuple[DSPGStore, str, list[Document], list[Sentence]]:
    created_store = store is None
    store = store or DSPGStore(create_indexes=False)
    ingest_started = time.monotonic()
    scan_unit_chars = _scan_unit_max_chars(semantic_client)
    scan_pack_chars = (
        _scan_pack_unit_chars(semantic_client)
        if semantic_client is not None and bool(use_semantic_frames or use_drs_semantics)
        else 0
    )
    documents, sentences = scan_folder(
        folder_path,
        max_unit_chars=scan_unit_chars,
        pack_unit_chars=scan_pack_chars,
        pack_unit_count=_scan_pack_unit_count() if scan_pack_chars else 0,
    )
    run_id = "" if created_store else store.latest_run_id(folder_path)
    if run_id:
        store.execute("UPDATE extraction_runs SET status=? WHERE run_id=?", ("running", run_id))
    else:
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
            ON CONFLICT(document_id) DO UPDATE SET
              run_id=excluded.run_id,
              path=excluded.path,
              rel_path=excluded.rel_path,
              content_hash=excluded.content_hash,
              size_bytes=excluded.size_bytes,
              mtime=excluded.mtime,
              ctime=excluded.ctime,
              char_count=excluded.char_count,
              metadata_json=excluded.metadata_json
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
                "INSERT OR IGNORE INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (context_id, run_id, quality_kind, None, document.rel_path, quality_kind, 1.0),
            )
        quality_context_id = context_by_kind[quality_kind]
        store.execute(
            """
            INSERT INTO context_carriers(
              carrier_id, run_id, context_id, document_id, source_span_id, carrier_kind, carrier_surface,
              temporal_value, temporal_value_type, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(carrier_id) DO UPDATE SET
              run_id=excluded.run_id,
              context_id=excluded.context_id,
              document_id=excluded.document_id,
              source_span_id=excluded.source_span_id,
              carrier_kind=excluded.carrier_kind,
              carrier_surface=excluded.carrier_surface,
              temporal_value=excluded.temporal_value,
              temporal_value_type=excluded.temporal_value_type,
              confidence=excluded.confidence
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
                    INSERT INTO context_carriers(
                      carrier_id, run_id, context_id, document_id, source_span_id, carrier_kind, carrier_surface,
                      temporal_value, temporal_value_type, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(carrier_id) DO UPDATE SET
                      run_id=excluded.run_id,
                      context_id=excluded.context_id,
                      document_id=excluded.document_id,
                      source_span_id=excluded.source_span_id,
                      carrier_kind=excluded.carrier_kind,
                      carrier_surface=excluded.carrier_surface,
                      temporal_value=excluded.temporal_value,
                      temporal_value_type=excluded.temporal_value_type,
                      confidence=excluded.confidence
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
        store.execute("DELETE FROM metadata_records WHERE run_id=? AND document_id=?", (run_id, document.document_id))
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
    structural_speaker_by_document: dict[str, tuple[str, str]] = {}
    semantic_passes = int(bool(use_semantic_frames and semantic_client is not None)) + int(
        bool(use_drs_semantics and semantic_client is not None)
    )
    semantic_total = len(sentences) * semantic_passes
    semantic_index = 0
    model_owned_semantics = (
        semantic_client is not None
        and bool(use_semantic_frames or use_drs_semantics)
        and not _env_true("KMD_ALLOW_DETERMINISTIC_SEMANTICS_WITH_LOCAL_MODEL")
    )
    deterministic_semantics_enabled = not model_owned_semantics

    for sentence in sentences:
        token_estimate = max(1, len(tokenize(sentence.text)))
        chunk_id = stable_id("chunk", sentence.sentence_id)
        store.execute(
            "INSERT OR IGNORE INTO chunks(chunk_id, document_id, chunk_order, char_start, char_end, text, token_estimate) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chunk_id, sentence.document_id, sentence.order, sentence.char_start, sentence.char_end, sentence.text, token_estimate),
        )
        span_id = stable_id("span", sentence.sentence_id, "sentence")
        store.execute(
            "INSERT OR IGNORE INTO source_spans(span_id, document_id, chunk_id, char_start, char_end, surface, surface_norm, span_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (span_id, sentence.document_id, chunk_id, sentence.char_start, sentence.char_end, sentence.text, normalize(sentence.text), "sentence"),
        )
        context_kind = "asserted" if model_owned_semantics else context_kind_for_sentence(sentence.text)
        context_id = context_by_kind.get(context_kind)
        if context_id is None:
            context_id = stable_id("ctx", run_id, context_kind)
            context_by_kind[context_kind] = context_id
            store.execute(
                "INSERT OR IGNORE INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
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
        if deterministic_semantics_enabled:
            try:
                max_mentions_per_chunk = max(0, int(os.environ.get("KMD_MENTIONS_MAX_PER_CHUNK", "128")))
            except ValueError:
                max_mentions_per_chunk = 128
            mention_candidates = collect_mentions(sentence)
            if max_mentions_per_chunk and len(mention_candidates) > max_mentions_per_chunk:
                mention_candidates = mention_candidates[:max_mentions_per_chunk]
            for surface, entity_type, start, end in mention_candidates:
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

        is_structural_heading = _is_structural_heading(sentence.text) if deterministic_semantics_enabled else False
        pending_label_heading = ""
        if is_structural_heading:
            section_anchor = clean_extracted_value(sentence.text)
            section_anchor_by_document[sentence.document_id] = section_anchor
            section_group_by_document[sentence.document_id] = stable_id("section_group", sentence.document_id, section_anchor)

        deterministic_relations = extract_relations(sentence.text) if deterministic_semantics_enabled else []
        cells = _table_cells(sentence.text) if deterministic_semantics_enabled else []
        if cells:
            current_header = table_headers_by_document.get(sentence.document_id)
            if current_header and len(cells) == len(current_header) and cells != current_header:
                deterministic_relations.extend(_table_header_relations(sentence, current_header, cells))
            elif _looks_like_table_header(cells):
                table_headers_by_document[sentence.document_id] = cells
                deterministic_relations = [
                    relation for relation in deterministic_relations if relation.relation_type != "table_cell"
                ]
        if not is_structural_heading:
            pending_label_heading = _label_heading_value_from_relations(sentence.text, deterministic_relations)
        structural_speaker_surface = _structural_speaker_surface_from_relations(deterministic_relations)
        if structural_speaker_surface:
            structural_speaker_by_document[sentence.document_id] = (structural_speaker_surface, sentence.text)
        active_structural_speaker = structural_speaker_by_document.get(sentence.document_id, ("", ""))

        temporal_values = [
            relation.value
            for relation in deterministic_relations
            if relation.relation_type == "temporal" and relation.value
        ]
        try:
            max_same_span_temporal_values = max(0, int(os.environ.get("KMD_TEMPORAL_SAME_SPAN_MAX_VALUES", "8")))
        except ValueError:
            max_same_span_temporal_values = 8
        try:
            max_same_span_temporal_edges = max(0, int(os.environ.get("KMD_TEMPORAL_SAME_SPAN_MAX_EDGES", "64")))
        except ValueError:
            max_same_span_temporal_edges = 64
        attachable_relation_count = sum(
            1
            for relation in deterministic_relations
            if relation.relation_type != "temporal" and relation.value
        )
        temporal_scope_values = temporal_values if len(temporal_values) <= max_same_span_temporal_values else []
        if len(temporal_scope_values) * attachable_relation_count > max_same_span_temporal_edges:
            temporal_scope_values = []
        try:
            max_deterministic_frames = max(0, int(os.environ.get("KMD_DETERMINISTIC_FRAMES_MAX_PER_CHUNK", "32")))
        except ValueError:
            max_deterministic_frames = 32
        deterministic_frame_count = 0
        relations_inherit_heading = _relation_inherits_heading(sentence.text, deterministic_relations)
        starts_new_structural_record = _starts_new_structural_record(sentence.text)
        active_section_anchor = section_anchor_by_document.get(sentence.document_id)
        prefix = re.split(r"[:=|\t]", sentence.text, maxsplit=1)[0]
        prefix_has_structural_phrase = any(len(phrase.split()) >= 2 for phrase in capitalized_phrases(prefix))
        for relation in deterministic_relations:
            metadata = {
                **relation.metadata,
                "sentence_group": stable_id("sentence_group", sentence.sentence_id),
            }
            if "record_group" not in metadata:
                metadata["record_group"] = metadata["sentence_group"]
            if relations_inherit_heading:
                section_group = section_group_by_document.get(sentence.document_id)
                if section_group and active_section_anchor:
                    metadata["record_group"] = section_group
                    metadata["section_anchor"] = active_section_anchor
            elif not starts_new_structural_record:
                if active_section_anchor:
                    metadata["section_anchor"] = active_section_anchor
            elif "section_anchor" not in metadata:
                if active_section_anchor and not prefix_has_structural_phrase:
                    metadata["section_anchor"] = active_section_anchor
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
            condition = (
                _condition_from_deterministic_relation(relation, sentence.text)
                if not max_deterministic_frames or deterministic_frame_count < max_deterministic_frames
                else None
            )
            if (
                condition is not None
                and condition.arguments
            ):
                deterministic_frame_count += 1
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
                        "INSERT OR IGNORE INTO frame_arguments(argument_id, frame_id, role, mention_id, referent_id, surface, value_type, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            stable_id("arg", condition_frame_id, arg_index, argument.role, argument.value),
                            condition_frame_id,
                            argument.role,
                            None,
                            arg_referent_id,
                            argument.value,
                            argument.value_type,
                            condition.confidence,
                        ),
                    )
                    normalized_argument = normalize(argument.value)
                    for existing_surface, _mention_id, existing_referent_id in mentions_for_sentence:
                        if normalize(existing_surface) == normalized_argument and existing_referent_id != arg_referent_id:
                            store.execute(
                                """
                                INSERT OR IGNORE INTO identity_hypotheses(
                                  hypothesis_id, run_id, source_span_id, context_id, drs_box_id, box_external_id,
                                  left_referent_id, right_referent_id,
                                  relation, evidence, confidence, source
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    stable_id("idh", run_id, existing_referent_id, arg_referent_id, sentence.sentence_id),
                                    run_id,
                                    span_id,
                                    context_id,
                                    None,
                                    None,
                                    existing_referent_id,
                                    arg_referent_id,
                                    "same_surface",
                                    argument.value,
                                    0.82,
                                    "deterministic_surface",
                                ),
                            )

            if temporal_scope_values and relation.relation_type != "temporal" and relation.value:
                # Pure DRT infrastructure: attach explicit same-span times to
                # structural conditions without interpreting the condition label.
                for temporal_value in temporal_scope_values:
                    store.execute(
                        """
                        INSERT OR IGNORE INTO temporal_edges(
                          edge_id, run_id, source_span_id, referent_id, context_id, relation, temporal_value, state_value, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            stable_id(
                                "tmp",
                                run_id,
                                sentence.sentence_id,
                                relation.relation_type,
                                relation.predicate,
                                relation.subject,
                                relation.object,
                                relation.value,
                                temporal_value,
                            ),
                            run_id,
                            span_id,
                            store.upsert_referent(run_id, relation.subject or relation.object or relation.value, "unknown"),
                            context_id,
                            relation.subject or relation.predicate or relation.relation_type,
                            temporal_value,
                            relation.value,
                            min(0.9, relation.confidence),
                        ),
                    )

        if (
            not _is_structural_heading(sentence.text)
            and pending_label_heading
            and not section_anchor_by_document.get(sentence.document_id)
        ):
            section_anchor_by_document[sentence.document_id] = pending_label_heading
            section_group_by_document[sentence.document_id] = stable_id("section_group", sentence.document_id, pending_label_heading)

        refresh_empty_compact_legacy = _env_true("KMD_DRS_REFRESH_EMPTY_STRUCTURAL_LEGACY") and any(
            relation.relation_type in {"label_value", "record_value"} and bool(relation.value)
            for relation in deterministic_relations
        )

        if use_semantic_frames and semantic_client is not None:
            semantic_index += 1
            frame_cache_context = chunk_frame_cache_context(
                semantic_client,
                rel_path=sentence.rel_path,
                chunk_text=sentence.text,
            )
            frame_cache_key = stable_id("frame_attempt_context", json.dumps(frame_cache_context, sort_keys=True, default=str))
            previous_attempt = store.execute(
                """
                SELECT accepted, materialized, reason, metadata_json
                FROM model_attempts
                WHERE run_id=? AND source_span_id=? AND task=? AND source=? AND cache_key=?
                LIMIT 1
                """,
                (run_id, span_id, "chunk_frames", "local_model", frame_cache_key),
            ).fetchone()
            existing_frames = store.execute(
                """
                SELECT COUNT(*)
                FROM frames
                WHERE run_id=? AND span_id=? AND source='local_model'
                """,
                (run_id, span_id),
            ).fetchone()[0]
            if existing_frames and _attempt_materialized(previous_attempt):
                _log_progress(
                    "kmd-ingest llm_done "
                    f"chunk={semantic_index}/{semantic_total} "
                    f"source={sentence.rel_path}:{sentence.order} "
                    "result=already_materialized "
                    "accepted=True "
                    "reason=already_materialized "
                    f"frames={int(existing_frames)} "
                    "model_elapsed=0.0 "
                    f"elapsed={time.monotonic() - ingest_started:.1f}s"
                )
                if use_drs_semantics and semantic_client is not None:
                    semantic_index = _ingest_model_drs_for_sentence(
                        store,
                        run_id,
                        sentence,
                        span_id,
                        semantic_client,
                        semantic_index,
                        semantic_total,
                        ingest_started,
                        refresh_empty_compact_legacy=refresh_empty_compact_legacy,
                        structural_speaker_surface=active_structural_speaker[0],
                        structural_speaker_evidence=f"{active_structural_speaker[1]}\n{sentence.text}".strip(),
                    )
                continue
            replaced_frames = {}
            if existing_frames:
                replaced_frames = store.delete_frame_materialization_for_span(
                    run_id,
                    span_id,
                    source="local_model",
                )
                inactive_attempts = store.deactivate_other_model_attempt_materializations(
                    run_id,
                    span_id,
                    "chunk_frames",
                    "local_model",
                    frame_cache_key,
                )
                if inactive_attempts:
                    replaced_frames["model_attempts"] = inactive_attempts
            if (
                _attempt_was_nonrequest_failure(previous_attempt)
                and not _env_true("KMD_FRAME_RETRY_FAILED_ATTEMPTS")
            ):
                _log_progress(
                    "kmd-ingest llm_done "
                    f"chunk={semantic_index}/{semantic_total} "
                    f"source={sentence.rel_path}:{sentence.order} "
                    "result=previous_attempt "
                    f"accepted={bool(previous_attempt['accepted'])} "
                    f"reason={str(previous_attempt['reason'] or 'previous_attempt')} "
                    "frames=0 "
                    f"replaced_prior_rows={sum(replaced_frames.values()) if replaced_frames else 0} "
                    "model_elapsed=0.0 "
                    f"elapsed={time.monotonic() - ingest_started:.1f}s"
                )
                if use_drs_semantics and semantic_client is not None:
                    semantic_index = _ingest_model_drs_for_sentence(
                        store,
                        run_id,
                        sentence,
                        span_id,
                        semantic_client,
                        semantic_index,
                        semantic_total,
                        ingest_started,
                        refresh_empty_compact_legacy=refresh_empty_compact_legacy,
                        structural_speaker_surface=active_structural_speaker[0],
                        structural_speaker_evidence=f"{active_structural_speaker[1]}\n{sentence.text}".strip(),
                    )
                continue
            _log_progress(
                "kmd-ingest llm_start "
                f"chunk={semantic_index}/{semantic_total} "
                f"source={sentence.rel_path}:{sentence.order} "
                f"elapsed={time.monotonic() - ingest_started:.1f}s"
            )
            model_frames, _frame_result = _grounded_model_frames(sentence, semantic_client, semantic_cache)
            result_source = str(_frame_result.get("fresh_or_cached") or _frame_result.get("source") or "fresh")
            accepted = bool(_frame_result.get("accepted")) if "accepted" in _frame_result else result_source == "cache"
            _log_progress(
                "kmd-ingest llm_done "
                f"chunk={semantic_index}/{semantic_total} "
                f"source={sentence.rel_path}:{sentence.order} "
                f"result={result_source} "
                f"accepted={accepted} "
                f"reason={str(_frame_result.get('reason') or '')} "
                f"frames={len(model_frames)} "
                f"model_elapsed={float(_frame_result.get('elapsed') or 0.0):.1f}s "
                f"elapsed={time.monotonic() - ingest_started:.1f}s"
            )
            inserted_model_frames = 0
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
                context_holder = clean_extracted_value(str(condition.metadata.get("context_holder") or ""))
                semantic_context_id = context_id
                if modality != "asserted":
                    context_key = "|".join(
                        [
                            "modality",
                            modality,
                            context_id,
                            normalize(context_holder),
                            normalize(evidence_text),
                        ]
                    )
                    semantic_context_id = context_by_kind.get(context_key)
                    if semantic_context_id is None:
                        semantic_context_id = stable_id("ctx", run_id, context_key)
                        context_by_kind[context_key] = semantic_context_id
                        store.execute(
                            "INSERT OR IGNORE INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (semantic_context_id, run_id, f"modality:{modality}", context_id, context_holder or None, evidence_text, condition.confidence),
                        )
                if polarity not in {"", "positive"}:
                    context_key = "|".join(
                        [
                            "polarity",
                            polarity,
                            semantic_context_id,
                            normalize(evidence_text),
                        ]
                    )
                    polarity_context_id = context_by_kind.get(context_key)
                    if polarity_context_id is None:
                        polarity_context_id = stable_id("ctx", run_id, context_key)
                        context_by_kind[context_key] = polarity_context_id
                        store.execute(
                            "INSERT OR IGNORE INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (polarity_context_id, run_id, f"polarity:{polarity}", semantic_context_id, None, evidence_text, condition.confidence),
                        )
                    semantic_context_id = polarity_context_id
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
                inserted_model_frames += 1
                group = stable_id("semantic_group", semantic_frame_id)
                frame_metadata = {
                    "frame_type": frame_type,
                    "modality": modality,
                    "polarity": polarity,
                    "context_holder": context_holder,
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
                        "INSERT OR IGNORE INTO frame_arguments(argument_id, frame_id, role, mention_id, referent_id, surface, value_type, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            stable_id("arg", semantic_frame_id, arg_index, role, surface),
                            semantic_frame_id,
                            role,
                            None,
                            arg_referent_id,
                            surface,
                            argument.value_type,
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
                                  hypothesis_id, run_id, source_span_id, context_id, drs_box_id, box_external_id,
                                  left_referent_id, right_referent_id,
                                  relation, evidence, confidence, source
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    stable_id("idh", run_id, existing_referent_id, arg_referent_id, semantic_frame_id),
                                    run_id,
                                    span_id,
                                    semantic_context_id,
                                    None,
                                    None,
                                    existing_referent_id,
                                    arg_referent_id,
                                    "same_surface",
                                    surface,
                                    min(0.9, condition.confidence),
                                    "local_model_frame",
                                ),
                            )
                for hypothesis_index, hypothesis in enumerate(condition.metadata.get("identity_hypotheses", [])):
                    if not isinstance(hypothesis, dict):
                        continue
                    left_text = clean_extracted_value(str(hypothesis.get("left_text") or ""))
                    right_text = clean_extracted_value(str(hypothesis.get("right_text") or ""))
                    identity_evidence = clean_extracted_value(str(hypothesis.get("evidence_text") or evidence_text))
                    if not left_text or not right_text or not identity_evidence:
                        continue
                    if left_text not in sentence.text or right_text not in sentence.text or identity_evidence not in sentence.text:
                        continue
                    left_ref = store.upsert_referent(run_id, left_text, "unknown")
                    right_ref = store.upsert_referent(run_id, right_text, "unknown")
                    store.execute(
                        """
                        INSERT OR IGNORE INTO identity_hypotheses(
                          hypothesis_id, run_id, source_span_id, context_id, drs_box_id, box_external_id,
                          left_referent_id, right_referent_id,
                          relation, evidence, confidence, source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            stable_id("idh", run_id, semantic_frame_id, "model_identity", hypothesis_index, left_text, right_text),
                            run_id,
                            span_id,
                            semantic_context_id,
                            None,
                            None,
                            left_ref,
                            right_ref,
                            clean_extracted_value(str(hypothesis.get("relation") or "same_referent")),
                            identity_evidence,
                            float(hypothesis.get("confidence") or condition.confidence),
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
            store.execute(
                """
                INSERT OR REPLACE INTO model_attempts(
                  attempt_id, run_id, source_span_id, task, source, cache_key, accepted, materialized,
                  reason, prompt_hash, output_hash, elapsed, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stable_id("attempt", run_id, span_id, "chunk_frames", "local_model", frame_cache_key),
                    run_id,
                    span_id,
                    "chunk_frames",
                    "local_model",
                    frame_cache_key,
                    int(accepted),
                    int(inserted_model_frames > 0),
                    str(_frame_result.get("reason") or ""),
                    str(_frame_result.get("prompt_hash") or ""),
                    str(_frame_result.get("output_hash") or ""),
                    float(_frame_result.get("elapsed") or 0.0),
                    json.dumps(
                        {
                            "cache_context": frame_cache_context,
                            "frame_count": len(model_frames),
                            "inserted_frame_count": inserted_model_frames,
                            "replaced_prior_rows": replaced_frames,
                            "context_budget": _frame_result.get("context_budget"),
                        },
                        sort_keys=True,
                        default=str,
                    ),
                ),
            )

        if use_drs_semantics and semantic_client is not None:
            semantic_index = _ingest_model_drs_for_sentence(
                store,
                run_id,
                sentence,
                span_id,
                semantic_client,
                semantic_index,
                semantic_total,
                ingest_started,
                refresh_empty_compact_legacy=refresh_empty_compact_legacy,
                structural_speaker_surface=active_structural_speaker[0],
                structural_speaker_evidence=f"{active_structural_speaker[1]}\n{sentence.text}".strip(),
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
