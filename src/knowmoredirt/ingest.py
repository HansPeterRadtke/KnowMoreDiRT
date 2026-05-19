"""Raw-text ingestion into the internal DSPG store."""

from __future__ import annotations

import re
from pathlib import Path

from .extractors import capitalized_phrases, identifiers, urls
from .models import Document, Sentence
from .scanner import scan_folder
from .store import DSPGStore, stable_id
from .text import normalize, tokenize


VERB_PREDICATES = {
    "drafted": "author",
    "authored": "author",
    "reviewed": "review",
    "review": "review",
    "approved": "approve",
    "merged": "merge",
    "opened": "open",
    "closed": "close",
    "reopened": "reopen",
    "reported": "report",
    "requested": "request",
    "fixed": "fix",
    "deleted": "delete",
    "believes": "believe",
    "alleges": "allege",
    "caused": "cause",
    "depends": "depend",
    "owns": "own",
    "owned": "own",
    "tested": "test",
    "manages": "manage",
}

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2})?\b")
STATE_RE = re.compile(r"\b(?:state|status)\s*:\s*([A-Za-z0-9_-]+)", re.I)


def mention_entity_type(surface: str) -> str:
    if re.fullmatch(r"https?://\S+", surface):
        return "url"
    if re.fullmatch(r"PR-\d+", surface):
        return "pr"
    if re.fullmatch(r"BUG-\d+", surface):
        return "bug"
    if re.fullmatch(r"SUP-\d+", surface):
        return "ticket"
    if re.fullmatch(r"[0-9a-f]{8,16}", surface, re.I):
        return "commit"
    if "@" in surface and "." in surface:
        return "email"
    if len(surface.split()) >= 2:
        return "name"
    return "artifact"


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
    for verb, predicate in VERB_PREDICATES.items():
        if re.search(rf"\b{re.escape(verb)}\b", text, re.I):
            return predicate, verb
    return None


def temporal_state(text: str) -> tuple[str, str] | None:
    date_match = DATE_RE.search(text)
    state_match = STATE_RE.search(text)
    if date_match and state_match:
        return date_match.group(0), state_match.group(1)
    date_match = DATE_RE.search(text)
    if not date_match:
        return None
    lowered = normalize(text)
    for trigger, state in [
        ("reopened", "reopened"),
        ("closed", "closed"),
        ("opened", "open"),
        ("fixed", "fixed"),
        ("regression", "regressed"),
    ]:
        if trigger in lowered:
            return date_match.group(0), state
    return None


def ingest_folder(folder_path: str | Path, store: DSPGStore | None = None) -> tuple[DSPGStore, str, list[Document], list[Sentence]]:
    store = store or DSPGStore()
    documents, sentences = scan_folder(folder_path)
    run_id = store.start_run(folder_path)

    sentence_by_id = {sentence.sentence_id: sentence for sentence in sentences}
    mention_by_surface: dict[tuple[str, str], str] = {}
    context_by_kind: dict[str, str] = {}

    for document in documents:
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
                "{}",
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
            referent_id = store.upsert_referent(run_id, surface, entity_type)
            store.execute(
                "INSERT OR IGNORE INTO mention_referents(mention_id, referent_id, link_status, confidence) VALUES (?, ?, ?, ?)",
                (mention_id, referent_id, "candidate", 1.0),
            )
            mention_by_surface[(sentence.sentence_id, surface)] = mention_id
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

    metrics = {
        "documents": len(documents),
        "sentences": len(sentences),
        **store.counts(),
    }
    store.finish_run(run_id, metrics)
    return store, run_id, documents, sentences
