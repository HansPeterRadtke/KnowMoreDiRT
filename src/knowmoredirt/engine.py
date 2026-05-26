"""KnowMoreDiRT raw-folder DRT/DSPG question-answering engine.

The engine initializes from one arbitrary folder path, reads all readable files
as raw text, builds grounded DSPG records, and answers questions by matching a
generic query frame against bounded discourse structures.  Optional local-model
use is restricted to query-frame refinement and evidence-only answer extraction.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .answer_types import (
    ExpectedAnswer,
    canonicalize_answer,
    infer_expected_answer,
    is_metadata_evidence_text,
)
from .bounded_dspg import execute_bounded_query
from .drs import frame_from_model_dict
from .extractors import capitalized_phrases
from .ingest import ingest_folder
from .index import LexicalIndex
from .model import LocalModelClient
from .model_planner import (
    ModelQueryTrace,
    call_model_answer_verification,
    call_model_chunk_frames,
    call_model_evidence_answer,
    call_model_identity_canonicalization,
    call_model_query_evidence_answer,
    call_model_query_plan,
    deterministic_plan as deterministic_query_frame,
    normalize_model_plan,
)
from .models import Answer, Evidence, Sentence
from .query import QueryFrame, frame_from_mapping, plan_question
from .semantic_cache import SemanticFrameCache
from .store import stable_id
from .text import content_tokens, is_low_semantic_noise, normalize


@dataclass
class EngineStats:
    document_count: int
    sentence_count: int


class KnowMoreDiRTEngine:
    """Internal session object backing the two-function public API."""

    def __init__(self, folder_path: str | Path) -> None:
        self.folder_path = Path(folder_path)
        self._use_local_model = os.environ.get("KMD_USE_LOCAL_MODEL", "").strip().lower() in {"1", "true", "yes", "on"}
        self._model_client = LocalModelClient() if self._use_local_model else None
        self.model_query_trace = ModelQueryTrace(enabled=self._use_local_model, prompt_hashes=[], response_hashes=[])
        self._semantic_cache = SemanticFrameCache() if self._use_local_model else None
        use_semantic_frames = self._use_local_model and os.environ.get("KMD_LLM_INGEST", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._log_progress(f"kmd-init start local_model={self._use_local_model} eager_llm_ingest={use_semantic_frames} root={self.folder_path}")
        self.store, self.run_id, self.documents, self.sentences = ingest_folder(
            self.folder_path,
            semantic_client=self._model_client if use_semantic_frames else None,
            use_semantic_frames=use_semantic_frames,
            semantic_cache=self._semantic_cache if use_semantic_frames else None,
        )
        self._log_progress(
            f"kmd-init indexed documents={len(self.documents)} chunks={len(self.sentences)} run_id={self.run_id}"
        )
        self.index = LexicalIndex(self.sentences)
        self.stats = EngineStats(len(self.documents), len(self.sentences))
        self._sentences_by_location = {
            (sentence.rel_path, sentence.order): sentence for sentence in self.sentences
        }
        self._sentences_by_document: dict[str, dict[int, Sentence]] = {}
        for sentence in self.sentences:
            self._sentences_by_document.setdefault(sentence.rel_path, {})[sentence.order] = sentence
        self._document_metadata_text = {
            document.rel_path: normalize(
                " ".join(
                    str(value)
                    for value in [
                        document.metadata.get("file_name", ""),
                        document.metadata.get("stem", ""),
                        document.metadata.get("suffix", ""),
                        document.metadata.get("parent_rel_path", ""),
                    ]
                )
            )
            for document in self.documents
        }
        self._low_semantic_noise_paths = {
            document.rel_path for document in self.documents if is_low_semantic_noise(document.text)
        }
        if use_semantic_frames:
            semantic_frame_rows = self.store.execute(
                "SELECT COUNT(*) FROM frames WHERE source='local_model'"
            ).fetchone()[0]
            self.model_query_trace.chunk_frame_call_count = int(semantic_frame_rows)
            self.model_query_trace.chunk_frame_parsed_count = int(semantic_frame_rows)
            self.model_query_trace.chunk_frame_accepted_count = int(semantic_frame_rows)
        self.last_answer: Answer | None = None
        self.last_bounded_diagnostics: dict[str, object] = {}

    def _progress_enabled(self) -> bool:
        return os.environ.get("KMD_PROGRESS", "").strip().lower() in {"1", "true", "yes", "on"} or os.environ.get(
            "KMD_EVAL_PROGRESS", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _log_progress(self, message: str) -> None:
        if self._progress_enabled():
            print(message, flush=True)

    def dspg_counts(self) -> dict[str, int]:
        return self.store.counts()

    def dspg_integrity(self) -> str:
        return self.store.integrity_check()

    def answer(self, question: str) -> Answer:
        text = str(question or "").strip()
        if not text:
            return Answer("unknown", reason="empty question")

        if self._use_local_model:
            model_answer = self._answer_with_local_model(text)
            if model_answer and normalize(model_answer.text) != "unknown":
                self.last_answer = model_answer
                return model_answer

        expected = infer_expected_answer(text)
        frame = plan_question(text)
        bounded = self._answer_with_bounded_dspg(text, frame, expected)
        if bounded and normalize(bounded.text) != "unknown":
            if self._use_local_model and not self._verify_with_local_model(text, frame, bounded, expected):
                bounded = None
            if bounded is None:
                pass
            else:
                self.last_answer = bounded
                return bounded

        answer = Answer("unknown", reason="no complete grounded DSPG match")
        self.last_answer = answer
        return answer

    def _expected_from_frame(self, frame: QueryFrame) -> ExpectedAnswer:
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
        answer_type = frame.answer_type if frame.answer_type in allowed else "content_phrase"
        return ExpectedAnswer(answer_type, allow_metadata_evidence=answer_type == "metadata_value")  # type: ignore[arg-type]

    def _verify_with_local_model(self, question: str, frame: QueryFrame, answer: Answer, expected: ExpectedAnswer) -> bool:
        if self._model_client is None:
            return True
        evidence_payload = self._evidence_payload(answer.evidence, limit=8)
        if not evidence_payload:
            return False
        discourse_frames = self._diagnostic_frames_for_answer(answer)
        trace = self.model_query_trace
        trace.verifier_call_count += 1
        result = call_model_answer_verification(
            question,
            frame.as_dict(),
            answer.text,
            evidence_payload,
            discourse_frames,
            self._model_client,
        )
        if result.get("prompt_hash"):
            trace.prompt_hashes = [*list(trace.prompt_hashes or []), str(result["prompt_hash"])][-20:]
        if result.get("output_hash"):
            trace.response_hashes = [*list(trace.response_hashes or []), str(result["output_hash"])][-20:]
        if not result.get("accepted"):
            trace.verifier_rejected_count += 1
            return False
        trace.verifier_parsed_count += 1
        entailed = bool(result.get("entailed"))
        proposed = str(result.get("answer") or "")
        span = str(result.get("evidence_span") or "")
        if not entailed or not proposed or (span and not any(span in item.get("text", "") for item in evidence_payload)):
            trace.verifier_rejected_count += 1
            return False
        canonical = canonicalize_answer(expected, proposed)
        if canonical and expected.answer_type in {"person", "actor"}:
            canonical = self._canonicalize_identity_with_local_model(question, canonical, answer.evidence) or canonical
        if canonical and normalize(canonical) != normalize(answer.text):
            answer.text = canonical
        trace.verifier_accepted_count += 1
        return True

    def _canonicalize_identity_with_local_model(self, question: str, value: str, evidence: list[Evidence]) -> str:
        if self._model_client is None or len(str(value).split()) != 1:
            return value
        token = normalize(value)
        fuller_candidates: list[str] = []
        for item in evidence:
            for phrase in capitalized_phrases(item.text):
                parts = normalize(phrase).split()
                if len(parts) > 1 and token in parts and phrase not in fuller_candidates:
                    fuller_candidates.append(phrase)
        if not fuller_candidates:
            return value
        evidence_payload = self._evidence_payload(evidence, limit=8)
        result = call_model_identity_canonicalization(
            question,
            value,
            fuller_candidates[:8],
            evidence_payload,
            self._model_client,
        )
        if result.get("prompt_hash"):
            self.model_query_trace.prompt_hashes = [*list(self.model_query_trace.prompt_hashes or []), str(result["prompt_hash"])][-20:]
        if result.get("output_hash"):
            self.model_query_trace.response_hashes = [*list(self.model_query_trace.response_hashes or []), str(result["output_hash"])][-20:]
        proposed = str(result.get("answer") or "")
        if result.get("accepted") and result.get("same_referent") and proposed in fuller_candidates:
            return proposed
        return value

    def _diagnostic_frames_for_answer(self, answer: Answer) -> list[dict[str, object]]:
        if not answer.evidence:
            return []
        rel_paths = list({evidence.rel_path for evidence in answer.evidence if evidence.rel_path})
        if not rel_paths:
            return []
        placeholders = ",".join("?" for _ in rel_paths[:8])
        rows = self.store.execute(
            f"""
            SELECT d.rel_path, f.predicate, f.trigger_surface, f.source, c.kind
            FROM frames f
            JOIN source_spans s ON s.span_id=f.span_id
            JOIN documents d ON d.document_id=s.document_id
            LEFT JOIN contexts c ON c.context_id=f.context_id
            WHERE d.rel_path IN ({placeholders})
            LIMIT 32
            """,
            tuple(rel_paths[:8]),
        ).fetchall()
        return [dict(row) for row in rows]

    def _evidence(self, sentence: Sentence, score: float = 1.0) -> Evidence:
        return Evidence(sentence.rel_path, sentence.text, score)

    def _evidence_window_text(self, evidence: Evidence, *, radius: int | None = None, max_chars: int | None = None) -> str:
        if radius is None:
            radius = int(os.environ.get("KMD_EVIDENCE_WINDOW_RADIUS", "3"))
        if max_chars is None:
            max_chars = int(os.environ.get("KMD_EVIDENCE_TEXT_CHARS", "1200"))
        sentences = self._sentences_by_document.get(evidence.rel_path, {})
        center_order: int | None = None
        for order, sentence in sentences.items():
            if sentence.text == evidence.text:
                center_order = order
                break
        if center_order is None:
            return evidence.text[:max_chars]
        parts = [
            sentences[order].text
            for order in range(center_order - radius, center_order + radius + 1)
            if order in sentences
        ]
        return "\n".join(parts)[:max_chars]

    def _evidence_payload(self, evidence: list[Evidence], *, limit: int = 8) -> list[dict[str, str]]:
        return [
            {"source": item.rel_path, "text": self._evidence_window_text(item)}
            for item in evidence[:limit]
            if item.rel_path and item.text
        ]

    def _matching_evidence(self, evidence: list[Evidence], evidence_span: str, proposed: str) -> list[Evidence]:
        matches: list[Evidence] = []
        for item in evidence:
            window = self._evidence_window_text(item)
            if evidence_span in window and (proposed in window or proposed.lower().startswith(("yes", "no"))):
                matches.append(item)
        return matches

    def _answer_with_local_model(self, question: str) -> Answer | None:
        if self._model_client is None:
            return None
        trace = self.model_query_trace
        trace.call_count += 1
        det = deterministic_query_frame(question)
        self._log_progress("kmd-answer model_plan_start")
        model = call_model_query_plan(question, self._model_client)
        if model.get("prompt_hash"):
            trace.prompt_hashes = [*list(trace.prompt_hashes or []), str(model["prompt_hash"])][-20:]
        if model.get("output_hash"):
            trace.response_hashes = [*list(trace.response_hashes or []), str(model["output_hash"])][-20:]
        if model.get("accepted"):
            trace.parsed_count += 1
            trace.accepted_count += 1
        plan = normalize_model_plan(question, model, det) if model.get("accepted") else det
        trace.last_plan = plan
        planned_frame = frame_from_mapping(question, plan)
        expected = self._expected_from_frame(planned_frame)
        self._materialize_question_semantics(question, planned_frame)
        self._log_progress("kmd-answer bounded_query_start")
        answer = self._answer_with_bounded_dspg(question, planned_frame, expected)
        if answer and normalize(answer.text) != "unknown":
            if self._verify_with_local_model(question, planned_frame, answer, expected):
                trace.model_answer_count += 1
                answer.reason = "local model query-frame execution"
                return answer
        self._log_progress("kmd-answer evidence_extraction_start")
        evidence_answer = self._answer_with_model_evidence_extraction(question, planned_frame, expected)
        if evidence_answer and normalize(evidence_answer.text) != "unknown":
            if self._verify_with_local_model(question, planned_frame, evidence_answer, expected):
                return evidence_answer
        direct = self._answer_with_model_query_evidence(question)
        if direct and normalize(direct.text) != "unknown":
            return direct
        return None

    def _lazy_semantic_frames_enabled(self) -> bool:
        return os.environ.get("KMD_LAZY_LLM_FRAMES", "1").strip().lower() not in {"0", "false", "no", "off"}

    def _materialize_question_semantics(self, question: str, frame: QueryFrame) -> None:
        if self._model_client is None or not self._lazy_semantic_frames_enabled():
            return
        limit = int(os.environ.get("KMD_LAZY_FRAME_SEARCH_LIMIT", "10"))
        chunk_limit = int(os.environ.get("KMD_LAZY_FRAME_CHUNK_LIMIT", "5"))
        required = list(frame.target_anchors) if frame.target_anchors else None
        candidates = self._search(question, limit=limit, required=required)
        if len(candidates) < min(3, limit) and required:
            candidates = self._search(question, limit=limit, required=None)
        target_terms = [normalize(anchor) for anchor in frame.target_anchors if normalize(anchor)]
        relation_terms = [normalize(term) for term in [frame.requested_relation, *frame.relation_terms, *frame.constraints] if normalize(term)]

        def materialization_rank(item: tuple[Sentence, float]) -> tuple[float, str, int]:
            sentence, score = item
            text = normalize(sentence.text)
            target_hits = sum(1 for term in target_terms if term and term in text)
            relation_hits = sum(1 for term in relation_terms if term and term in text)
            return (-(score + target_hits * 3.0 + relation_hits * 6.0), sentence.rel_path, sentence.order)

        candidates = sorted(candidates, key=materialization_rank)
        materialized = 0
        for sentence, _score in candidates[:chunk_limit]:
            materialized += self._materialize_sentence_semantics(sentence)
        if materialized:
            self.store.commit()
        self._log_progress(f"kmd-answer lazy_frames materialized={materialized} candidates={len(candidates)}")

    def _sentence_span_id(self, sentence: Sentence) -> str:
        return stable_id("span", sentence.sentence_id, "sentence")

    def _chunk_id(self, sentence: Sentence) -> str:
        return stable_id("chunk", sentence.sentence_id)

    def _sentence_context_id(self, sentence: Sentence) -> str:
        span_id = self._sentence_span_id(sentence)
        row = self.store.execute(
            """
            SELECT context_id
            FROM context_assignments
            WHERE run_id=? AND applies_to_type='source_span' AND applies_to_id=?
            LIMIT 1
            """,
            (self.run_id, span_id),
        ).fetchone()
        if row is not None:
            return str(row["context_id"])
        context_id = stable_id("ctx", self.run_id, "asserted")
        self.store.execute(
            "INSERT OR IGNORE INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (context_id, self.run_id, "asserted", None, None, "asserted", 1.0),
        )
        return context_id

    def _ensure_context(self, kind: str, parent_context_id: str, evidence_surface: str, confidence: float) -> str:
        context_id = stable_id("ctx", self.run_id, kind)
        self.store.execute(
            "INSERT OR IGNORE INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (context_id, self.run_id, kind, parent_context_id or None, None, evidence_surface, confidence),
        )
        return context_id

    def _mentions_for_sentence(self, sentence: Sentence) -> list[tuple[str, str, str]]:
        rows = self.store.execute(
            """
            SELECT m.surface, m.mention_id, mr.referent_id
            FROM mentions m
            JOIN mention_referents mr ON mr.mention_id=m.mention_id
            JOIN source_spans s ON s.span_id=m.span_id
            WHERE m.run_id=? AND s.chunk_id=?
            ORDER BY s.char_start, m.surface
            """,
            (self.run_id, self._chunk_id(sentence)),
        ).fetchall()
        return [(str(row["surface"]), str(row["mention_id"]), str(row["referent_id"])) for row in rows]

    def _cached_or_fresh_chunk_frames(self, sentence: Sentence) -> tuple[list[dict[str, object]], dict[str, object]]:
        if self._model_client is None:
            return [], {"source": "disabled"}
        if is_low_semantic_noise(sentence.text):
            return [], {"source": "skipped_noise"}
        if len(sentence.text) > int(os.environ.get("KMD_LAZY_FRAME_MAX_CHARS", "1800")):
            return [], {"source": "skipped_long_chunk"}
        cached = self._semantic_cache.get(sentence.text) if self._semantic_cache else None
        if cached is not None:
            frames = [frame for frame in cached.get("frames", []) if isinstance(frame, dict)]
            return frames, {"source": "cache", "frame_count": len(frames)}
        self._log_progress(f"kmd-llm-frame start {sentence.rel_path}:{sentence.order}")
        result = call_model_chunk_frames(sentence.text, self._model_client, rel_path=sentence.rel_path)
        frames = [frame for frame in result.get("frames", []) if isinstance(frame, dict)] if result.get("accepted") else []
        if self._semantic_cache is not None and result.get("accepted"):
            self._semantic_cache.put(
                sentence.text,
                frames,
                {
                    "rel_path": sentence.rel_path,
                    "prompt_hash": result.get("prompt_hash"),
                    "output_hash": result.get("output_hash"),
                },
            )
        self._log_progress(
            f"kmd-llm-frame done {sentence.rel_path}:{sentence.order} frames={len(frames)} source={result.get('fresh_or_cached', 'fresh')}"
        )
        return frames, result

    def _materialize_sentence_semantics(self, sentence: Sentence) -> int:
        span_id = self._sentence_span_id(sentence)
        existing = self.store.execute(
            "SELECT COUNT(*) FROM frames WHERE run_id=? AND span_id=? AND source='local_model'",
            (self.run_id, span_id),
        ).fetchone()[0]
        if existing:
            return 0
        model_frames, result = self._cached_or_fresh_chunk_frames(sentence)
        if result.get("prompt_hash"):
            self.model_query_trace.prompt_hashes = [*list(self.model_query_trace.prompt_hashes or []), str(result["prompt_hash"])][-20:]
        if result.get("output_hash"):
            self.model_query_trace.response_hashes = [*list(self.model_query_trace.response_hashes or []), str(result["output_hash"])][-20:]
        self.model_query_trace.chunk_frame_call_count += 0 if result.get("source") in {"cache", "skipped_noise", "skipped_long_chunk", "disabled"} else 1
        if not model_frames:
            return 0
        self.model_query_trace.chunk_frame_parsed_count += len(model_frames)
        context_id = self._sentence_context_id(sentence)
        mentions_for_sentence = self._mentions_for_sentence(sentence)
        inserted = 0
        for index, frame in enumerate(model_frames):
            condition = frame_from_model_dict(frame)
            if condition is None or condition.evidence_text not in sentence.text:
                continue
            predicate = condition.predicate or condition.frame_type
            semantic_context_id = context_id
            if condition.modality != "asserted":
                semantic_context_id = self._ensure_context(
                    f"modality:{condition.modality}",
                    context_id,
                    condition.evidence_text,
                    condition.confidence,
                )
            if condition.polarity not in {"", "positive"}:
                semantic_context_id = self._ensure_context(
                    f"polarity:{condition.polarity}",
                    semantic_context_id,
                    condition.evidence_text,
                    condition.confidence,
                )
            semantic_frame_id = stable_id("frm", self.run_id, sentence.sentence_id, "model", index, predicate, condition.evidence_text)
            self.store.execute(
                "INSERT OR IGNORE INTO frames(frame_id, run_id, context_id, predicate, predicate_norm, trigger_surface, confidence, source, span_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    semantic_frame_id,
                    self.run_id,
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
                "frame_type": condition.frame_type,
                "modality": condition.modality,
                "polarity": condition.polarity,
                "temporal_text": condition.temporal_text,
                "record_group": group,
                "source": "local_model",
            }
            self.store.execute(
                """
                INSERT OR IGNORE INTO relations(
                  relation_id, run_id, relation_type, subject, subject_norm, predicate, predicate_norm,
                  object, object_norm, value, value_norm, source_span_id, context_id, confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stable_id("rel", self.run_id, semantic_frame_id, "semantic_frame"),
                    self.run_id,
                    "semantic_frame",
                    condition.frame_type,
                    normalize(condition.frame_type),
                    predicate,
                    normalize(predicate),
                    "",
                    "",
                    condition.evidence_text,
                    normalize(condition.evidence_text),
                    span_id,
                    semantic_context_id,
                    condition.confidence,
                    json.dumps(frame_metadata, sort_keys=True),
                ),
            )
            for arg_index, argument in enumerate(condition.arguments):
                arg_referent_id = self.store.upsert_referent(self.run_id, argument.value, argument.value_type)
                self.store.execute(
                    "INSERT OR IGNORE INTO frame_arguments(argument_id, frame_id, role, mention_id, referent_id, surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        stable_id("arg", semantic_frame_id, arg_index, argument.role, argument.value),
                        semantic_frame_id,
                        argument.role,
                        None,
                        arg_referent_id,
                        argument.value,
                        condition.confidence,
                    ),
                )
                relation_metadata = {
                    **frame_metadata,
                    "argument_role": argument.role,
                    "argument_value_type": argument.value_type,
                }
                self.store.execute(
                    """
                    INSERT OR IGNORE INTO relations(
                      relation_id, run_id, relation_type, subject, subject_norm, predicate, predicate_norm,
                      object, object_norm, value, value_norm, source_span_id, context_id, confidence, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("rel", self.run_id, semantic_frame_id, "arg", arg_index, argument.role, argument.value),
                        self.run_id,
                        "semantic_argument",
                        argument.role,
                        normalize(argument.role),
                        predicate,
                        normalize(predicate),
                        condition.frame_type,
                        normalize(condition.frame_type),
                        argument.value,
                        normalize(argument.value),
                        span_id,
                        semantic_context_id,
                        condition.confidence,
                        json.dumps(relation_metadata, sort_keys=True),
                    ),
                )
                normalized_argument = normalize(argument.value)
                for existing_surface, _mention_id, existing_referent_id in mentions_for_sentence:
                    if normalize(existing_surface) == normalized_argument and existing_referent_id != arg_referent_id:
                        self.store.execute(
                            """
                            INSERT OR IGNORE INTO identity_hypotheses(
                              hypothesis_id, run_id, left_referent_id, right_referent_id,
                              relation, evidence, confidence, source
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                stable_id("idh", self.run_id, existing_referent_id, arg_referent_id, semantic_frame_id),
                                self.run_id,
                                existing_referent_id,
                                arg_referent_id,
                                "same_surface",
                                argument.value,
                                min(0.9, condition.confidence),
                                "local_model_frame",
                            ),
                        )
            if condition.temporal_text:
                self.store.execute(
                    """
                    INSERT OR IGNORE INTO temporal_edges(
                      edge_id, run_id, source_span_id, referent_id, context_id, relation, temporal_value, state_value, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("tmp", self.run_id, semantic_frame_id, condition.temporal_text),
                        self.run_id,
                        span_id,
                        None,
                        semantic_context_id,
                        "frame_temporal_scope",
                        condition.temporal_text,
                        "",
                        condition.confidence,
                    ),
                )
            inserted += 1
        self.model_query_trace.chunk_frame_accepted_count += inserted
        return inserted

    def _answer_with_model_query_evidence(self, question: str) -> Answer | None:
        if self._model_client is None:
            return None
        candidates = self._search(
            question,
            limit=int(os.environ.get("KMD_EVIDENCE_SEARCH_LIMIT", "18")),
            required=None,
        )
        if not candidates:
            return None
        evidence = [
            self._evidence(sentence, score)
            for sentence, score in candidates[: int(os.environ.get("KMD_EVIDENCE_PAYLOAD_LIMIT", "12"))]
        ]
        payload = self._evidence_payload(evidence, limit=len(evidence))
        if not payload:
            return None
        trace = self.model_query_trace
        model = call_model_query_evidence_answer(question, payload, self._model_client)
        if model.get("prompt_hash"):
            trace.prompt_hashes = [*list(trace.prompt_hashes or []), str(model["prompt_hash"])][-20:]
        if model.get("output_hash"):
            trace.response_hashes = [*list(trace.response_hashes or []), str(model["output_hash"])][-20:]
        if not model.get("accepted"):
            trace.evidence_rejected_count += 1
            return None
        trace.evidence_call_count += 1
        trace.evidence_parsed_count += 1
        if not model.get("sufficient_evidence"):
            trace.evidence_rejected_count += 1
            return None
        proposed = str(model.get("answer") or "")
        evidence_span = str(model.get("evidence_span") or "")
        answer_type = str(model.get("answer_type") or "content_phrase")
        frame = frame_from_mapping(question, model.get("query_frame") if isinstance(model.get("query_frame"), dict) else None)
        expected = self._expected_from_frame(frame)
        if answer_type:
            expected = ExpectedAnswer(answer_type if answer_type in {
                "person", "actor", "organization", "identifier", "url", "file_path", "count",
                "state", "date_time", "boolean", "content_phrase", "metadata_value", "unknown",
            } else expected.answer_type, allow_metadata_evidence=answer_type == "metadata_value")  # type: ignore[arg-type]
        if not proposed or not evidence_span:
            trace.evidence_rejected_count += 1
            return None
        matching = self._matching_evidence(evidence, evidence_span, proposed)
        if not matching:
            trace.evidence_rejected_count += 1
            return None
        answer = Answer(proposed, 0.78, matching[:3], "local model query-DRS evidence verification", expected.answer_type)
        finalized = self._finalize_answer(answer, expected, "local model query-DRS evidence verification")
        if not finalized:
            trace.evidence_rejected_count += 1
            return None
        trace.evidence_accepted_count += 1
        trace.model_answer_count += 1
        return finalized

    def _answer_with_model_evidence_extraction(
        self,
        question: str,
        frame: QueryFrame,
        expected: ExpectedAnswer | None = None,
    ) -> Answer | None:
        if self._model_client is None:
            return None
        expected = expected or infer_expected_answer(question)
        required = list(frame.target_anchors) if frame.target_anchors else None
        candidates = self._search(question, limit=int(os.environ.get("KMD_EVIDENCE_SEARCH_LIMIT", "18")), required=required)
        if len(candidates) < 4 and required:
            candidates = self._search(question, limit=int(os.environ.get("KMD_EVIDENCE_SEARCH_LIMIT", "18")), required=None)
        if not candidates:
            return None
        evidence = [self._evidence(sentence, score) for sentence, score in candidates[: int(os.environ.get("KMD_EVIDENCE_PAYLOAD_LIMIT", "12"))]]
        payload = self._evidence_payload(evidence, limit=len(evidence))
        if not payload:
            return None
        trace = self.model_query_trace
        trace.evidence_call_count += 1
        model = call_model_evidence_answer(question, expected.answer_type, payload, self._model_client)
        if model.get("prompt_hash"):
            trace.prompt_hashes = [*list(trace.prompt_hashes or []), str(model["prompt_hash"])][-20:]
        if model.get("output_hash"):
            trace.response_hashes = [*list(trace.response_hashes or []), str(model["output_hash"])][-20:]
        if not model.get("accepted"):
            trace.evidence_rejected_count += 1
            return None
        trace.evidence_parsed_count += 1
        if not model.get("sufficient_evidence"):
            trace.evidence_rejected_count += 1
            return None
        proposed = str(model.get("answer") or "")
        evidence_span = str(model.get("evidence_span") or "")
        if not proposed or not evidence_span:
            trace.evidence_rejected_count += 1
            return None
        matching = self._matching_evidence(evidence, evidence_span, proposed)
        if not matching:
            trace.evidence_rejected_count += 1
            return None
        answer = Answer(
            proposed,
            0.74,
            matching[:3],
            "local model bounded evidence extraction",
            str(model.get("answer_type") or "unknown"),
        )
        finalized = self._finalize_answer(answer, expected, "local model bounded evidence extraction")
        if not finalized:
            trace.evidence_rejected_count += 1
            return None
        trace.evidence_accepted_count += 1
        trace.model_answer_count += 1
        return finalized

    def _answer_with_bounded_dspg(self, question: str, frame: QueryFrame, expected: ExpectedAnswer) -> Answer | None:
        bounded_answer, diagnostics = execute_bounded_query(
            self.store,
            self.run_id,
            self.documents,
            self._sentences_by_document,
            question,
            frame,
        )
        self.last_bounded_diagnostics = diagnostics
        if not bounded_answer:
            return None
        return self._finalize_answer(bounded_answer, expected, "bounded DSPG query-frame execution")

    def _answer_has_source_grounding(self, answer: Answer) -> bool:
        if normalize(answer.text) == "unknown":
            return True
        return any(evidence.rel_path and evidence.text for evidence in answer.evidence)

    def _finalize_answer(self, answer: Answer, expected: ExpectedAnswer, source: str) -> Answer | None:
        if normalize(answer.text) == "unknown":
            return answer
        if not self._answer_has_source_grounding(answer):
            return None
        if any(is_metadata_evidence_text(evidence.text) for evidence in answer.evidence) and not expected.allow_metadata_evidence:
            return None
        canonical = canonicalize_answer(expected, answer.text)
        if canonical and expected.answer_type in {"person", "actor"} and answer.evidence:
            canonical = self._expand_person_from_evidence(canonical, answer.evidence) or canonical
        if not canonical:
            return None
        return Answer(canonical, answer.confidence, answer.evidence, source, expected.answer_type)

    def _expand_person_from_evidence(self, value: str, evidence: list[Evidence]) -> str:
        if len(str(value).split()) != 1:
            return value
        target = normalize(value)
        for item in evidence:
            for phrase in capitalized_phrases(item.text):
                if normalize(phrase).split()[-1:] == [target]:
                    return phrase
        return value

    def _search(self, question: str, limit: int = 12, required: list[str] | None = None) -> list[tuple[Sentence, float]]:
        frame = plan_question(question)
        combined: dict[str, tuple[Sentence, float]] = {}
        for sentence, score in self.index.search(question, limit=limit, required=required):
            combined[sentence.sentence_id] = (sentence, score)

        anchors = list(frame.target_anchors)
        relation_terms = list(frame.relation_terms)
        for row in self.store.referent_candidate_chunks(self.run_id, anchors, limit=limit):
            sentence = self._sentences_by_location.get((str(row["rel_path"]), int(row["chunk_order"])))
            if sentence:
                previous = combined.get(sentence.sentence_id, (sentence, 0.0))[1]
                combined[sentence.sentence_id] = (sentence, previous + 2.0)
        for row in self.store.frame_candidate_chunks(self.run_id, relation_terms, anchors, limit=limit):
            sentence = self._sentences_by_location.get((str(row["rel_path"]), int(row["chunk_order"])))
            if sentence:
                previous = combined.get(sentence.sentence_id, (sentence, 0.0))[1]
                combined[sentence.sentence_id] = (sentence, previous + 2.5)
        for row in self.store.relation_candidate_chunks(self.run_id, relation_terms, anchors, limit=limit):
            sentence = self._sentences_by_location.get((str(row["rel_path"]), int(row["chunk_order"])))
            if sentence:
                previous = combined.get(sentence.sentence_id, (sentence, 0.0))[1]
                combined[sentence.sentence_id] = (sentence, previous + 2.5)
        for sentence, score in self._metadata_bounded_candidates(question, limit=max(limit * 2, 24)):
            previous = combined.get(sentence.sentence_id, (sentence, 0.0))[1]
            combined[sentence.sentence_id] = (sentence, max(previous, score))

        seed_items = list(combined.values())
        for sentence, score in seed_items:
            document_sentences = self._sentences_by_document.get(sentence.rel_path, {})
            for offset in range(-4, 5):
                if offset == 0:
                    continue
                neighbor = document_sentences.get(sentence.order + offset)
                if neighbor:
                    previous = combined.get(neighbor.sentence_id, (neighbor, 0.0))[1]
                    combined[neighbor.sentence_id] = (neighbor, max(previous, score * 0.55))

        adjusted: list[tuple[Sentence, float]] = []
        target_terms = [normalize(anchor) for anchor in frame.target_anchors if normalize(anchor)]
        relation_terms = [normalize(term) for term in [frame.requested_relation, *frame.relation_terms, *frame.constraints] if normalize(term)]
        for sentence, score in combined.values():
            text_norm = normalize(sentence.text)
            score += sum(2.0 for term in target_terms if term and term in text_norm)
            score += sum(4.0 for term in relation_terms if term and term in text_norm)
            if sentence.rel_path in self._low_semantic_noise_paths:
                score *= 0.15
            if self._is_low_priority_source_path(sentence.rel_path) and not self._asks_about_low_priority_source(question):
                score *= 0.05
            adjusted.append((sentence, score))
        scored = sorted(adjusted, key=lambda item: (-item[1], item[0].rel_path, item[0].order))
        return scored[:limit]

    def _metadata_bounded_candidates(self, question: str, limit: int = 24) -> list[tuple[Sentence, float]]:
        query_tokens = [
            token for token in content_tokens(question)
            if len(token) > 3 and token not in {"file", "folder", "document", "object", "source"}
        ]
        if not query_tokens:
            return []
        doc_scores: list[tuple[float, str]] = []
        score_by_doc: dict[str, float] = {}
        for rel_path, metadata_text in self._document_metadata_text.items():
            score = sum(4.0 for token in query_tokens if token in metadata_text)
            if score:
                doc_scores.append((score, rel_path))
                score_by_doc[rel_path] = score
        doc_scores.sort(key=lambda item: (-item[0], item[1]))
        selected_docs = {rel_path for _, rel_path in doc_scores[:8]}
        candidates: list[tuple[Sentence, float]] = []
        for rel_path in selected_docs:
            for sentence in self._sentences_by_document.get(rel_path, {}).values():
                text_norm = normalize(sentence.text)
                token_hits = sum(1 for token in query_tokens if token in text_norm)
                if token_hits:
                    candidates.append((sentence, score_by_doc.get(sentence.rel_path, 0.0) + token_hits))
        candidates.sort(key=lambda item: (-item[1], item[0].rel_path, item[0].order))
        return candidates[:limit]

    def _is_low_priority_source_path(self, rel_path: str) -> bool:
        parts = re.split(r"[/_.-]+", normalize(rel_path))
        return bool({"cache", "lock", "tmp", "temp", "transport", "hidden"}.intersection(parts))

    def _asks_about_low_priority_source(self, question: str) -> bool:
        qnorm = normalize(question)
        return any(term in qnorm for term in ["cache", "lock", "temporary", "metadata", "file", "path"])

    def _target_anchors(self, question: str) -> list[str]:
        return capitalized_phrases(question)
