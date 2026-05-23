"""Raw-text ingestion into the internal DSPG store."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .extractors import capitalized_phrases, identifiers, urls
from .models import Document, Sentence
from .relations import extract_relations
from .scanner import scan_folder
from .store import DSPGStore, stable_id
from .text import normalize, text_quality_metrics, tokenize


DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2})?\b")
STATE_RE = re.compile(r"\b(?:state|status)\s*:\s*([A-Za-z0-9_-]+)", re.I)
GENERIC_VERB_RE = re.compile(r"\b([A-Za-z]{3,30}(?:ed|s|ing)?)\b")


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
    value = normalize(text)
    if "dream" in value or "woke up" in value:
        return "dreamed"
    if "fiction" in value or "homework" in value:
        return "fiction"
    if "alleges" in value or "allegation" in value or "plaintiff" in value:
        return "allegation"
    if "believes" in value or "argues" in value:
        return "believed"
    if "forwarded message" in value or value.startswith("from:"):
        return "reported"
    if "no proof" in value or "does not" in value or "not " in value:
        return "negated"
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
        if relation.relation_type in {"event", "assertion"} and relation.predicate:
            return relation.predicate, relation.metadata.get("surface_verb", relation.predicate) if isinstance(relation.metadata, dict) else relation.predicate
    match = GENERIC_VERB_RE.search(text)
    if match:
        verb = match.group(1).lower()
        return verb, verb
    return None


def temporal_state(text: str) -> tuple[str, str] | None:
    date_match = DATE_RE.search(text)
    state_match = STATE_RE.search(text)
    if date_match and state_match:
        return date_match.group(0), state_match.group(1)
    date_match = DATE_RE.search(text)
    if not date_match:
        return None
    return None


def ingest_folder(folder_path: str | Path, store: DSPGStore | None = None) -> tuple[DSPGStore, str, list[Document], list[Sentence]]:
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

        for relation in extract_relations(sentence.text):
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
                    json.dumps(relation.metadata, sort_keys=True),
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
