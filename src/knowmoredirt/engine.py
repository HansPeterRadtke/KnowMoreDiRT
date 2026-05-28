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
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .answer_types import (
    ExpectedAnswer,
    answer_parts,
    canonicalize_answer,
    classify_value,
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
    call_model_answer_canonicalization,
    call_model_answer_verification,
    call_model_chunk_frames,
    call_model_evidence_answer,
    call_model_identity_canonicalization,
    call_model_query_drs,
    call_model_query_evidence_answer,
    call_model_query_plan,
    chunk_frame_cache_context,
    deterministic_plan as deterministic_query_frame,
    normalize_model_plan,
    query_frame_from_query_drs,
)
from .models import Answer, Evidence, Sentence
from .query import QueryFrame, frame_from_mapping, plan_question, term_variants
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
        self._use_local_model = self._should_use_local_model()
        self._model_client = LocalModelClient() if self._use_local_model else None
        self.model_query_trace = ModelQueryTrace(enabled=self._use_local_model, prompt_hashes=[], response_hashes=[])
        self._semantic_cache = SemanticFrameCache() if self._use_local_model else None
        llm_ingest_setting = os.environ.get("KMD_LLM_INGEST", "1").strip().lower()
        use_semantic_frames = self._use_local_model and llm_ingest_setting not in {"0", "false", "no", "off"}
        drs_ingest_setting = os.environ.get("KMD_LLM_DRS_INGEST", "0").strip().lower()
        use_drs_semantics = self._use_local_model and drs_ingest_setting in {"1", "true", "yes", "on"}
        self._log_progress(
            "kmd-init start "
            f"local_model={self._use_local_model} "
            f"eager_llm_ingest={use_semantic_frames} "
            f"drs_ingest={use_drs_semantics} "
            f"root={self.folder_path}"
        )
        self.store, self.run_id, self.documents, self.sentences = ingest_folder(
            self.folder_path,
            semantic_client=self._model_client if use_semantic_frames or use_drs_semantics else None,
            use_semantic_frames=use_semantic_frames,
            use_drs_semantics=use_drs_semantics,
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

    def _should_use_local_model(self) -> bool:
        requested = os.environ.get("KMD_USE_LOCAL_MODEL", "").strip().lower()
        if requested in {"0", "false", "no", "off"}:
            return False
        if requested in {"1", "true", "yes", "on"}:
            return True
        if os.environ.get("KMD_AUTO_LOCAL_MODEL", "1").strip().lower() in {"0", "false", "no", "off"}:
            return False
        endpoint = os.environ.get("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:14829/v1").rstrip("/")
        if not (
            endpoint.startswith("http://127.0.0.1:")
            or endpoint.startswith("http://localhost:")
            or endpoint.startswith("http://[::1]:")
        ):
            return False
        models_url = endpoint + "/models" if endpoint.endswith("/v1") else endpoint + "/v1/models"
        try:
            with urllib.request.urlopen(models_url, timeout=float(os.environ.get("KMD_MODEL_PROBE_TIMEOUT", "1.5"))) as response:
                return response.status == 200
        except Exception:
            return False

    def _progress_enabled(self) -> bool:
        return os.environ.get("KMD_PROGRESS", "").strip().lower() in {"1", "true", "yes", "on"} or os.environ.get(
            "KMD_EVAL_PROGRESS", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _log_progress(self, message: str) -> None:
        if self._progress_enabled():
            print(message, flush=True)

    def _record_model_result(self, result: dict[str, object]) -> None:
        trace = self.model_query_trace
        if result.get("fresh_or_cached") == "cache" or result.get("source") == "cache":
            trace.cache_hit_count += 1
        try:
            trace.time_spent_seconds += float(result.get("elapsed") or 0.0)
        except (TypeError, ValueError):
            pass
        if result.get("accepted") is False:
            trace.rejected_output_count += 1
            reason = str(result.get("reason") or "")
            if reason == "invalid_json":
                trace.invalid_json_count += 1
            elif reason == "schema_validation_failed":
                trace.schema_rejection_count += 1
            elif reason == "grounding_validation_failed":
                trace.grounding_rejection_count += 1

    def _fallback_model_client(self) -> LocalModelClient | None:
        if self._model_client is None:
            return None
        timeout = float(os.environ.get("KMD_FALLBACK_MODEL_TIMEOUT_SECONDS", "35"))
        if timeout <= 0 or abs(timeout - float(getattr(self._model_client, "timeout_seconds", timeout))) < 0.001:
            return self._model_client
        return LocalModelClient(endpoint=self._model_client.endpoint, timeout_seconds=timeout)

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
            if model_answer:
                self.last_answer = model_answer
                return model_answer
            answer = Answer("unknown", reason="local model DRT path found no complete grounded answer")
            self.last_answer = answer
            return answer

        frame = plan_question(text)
        expected = self._expected_from_frame(frame)
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
        answer_type = frame.answer_type if frame.answer_type in allowed else "unknown"
        return ExpectedAnswer(answer_type, allow_metadata_evidence=answer_type == "metadata_value")  # type: ignore[arg-type]

    def _verify_with_local_model(self, question: str, frame: QueryFrame, answer: Answer, expected: ExpectedAnswer) -> bool:
        if self._model_client is None:
            return True
        evidence_payload = self._evidence_payload(answer.evidence, limit=8)
        if not evidence_payload:
            return False
        discourse_frames = self._diagnostic_frames_for_answer(answer)
        trace = self.model_query_trace
        candidate_answers = [answer.text]
        canonical_candidate = self._canonicalize_model_answer_with_local_model(question, answer.text, expected, answer.evidence)
        if canonical_candidate and normalize(canonical_candidate) != normalize(answer.text):
            candidate_answers.insert(0, canonical_candidate)
        seen_candidates: set[str] = set()
        for candidate_answer in candidate_answers:
            candidate_key = normalize(candidate_answer)
            if not candidate_key or candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)
            trace.verifier_call_count += 1
            result = call_model_answer_verification(
                question,
                frame.as_dict(),
                candidate_answer,
                evidence_payload,
                discourse_frames,
                self._model_client,
            )
            self._record_model_result(result)
            if result.get("prompt_hash"):
                trace.prompt_hashes = [*list(trace.prompt_hashes or []), str(result["prompt_hash"])][-20:]
            if result.get("output_hash"):
                trace.response_hashes = [*list(trace.response_hashes or []), str(result["output_hash"])][-20:]
            if not result.get("accepted"):
                trace.verifier_rejected_count += 1
                continue
            trace.verifier_parsed_count += 1
            entailed = bool(result.get("entailed"))
            proposed = str(result.get("answer") or "")
            span = str(result.get("evidence_span") or "")
            if not entailed or not proposed or (span and not any(span in item.get("text", "") for item in evidence_payload)):
                trace.verifier_rejected_count += 1
                continue
            canonical = canonicalize_answer(expected, proposed)
            if not canonical:
                trace.verifier_rejected_count += 1
                continue
            if canonical and expected.answer_type in {"person", "actor"}:
                if len(str(canonical).split()) == 1:
                    canonical = self._canonicalize_identity_with_local_model(question, canonical, answer.evidence) or canonical
            if canonical and normalize(canonical) != normalize(answer.text):
                answer.text = canonical
            trace.verifier_accepted_count += 1
            return True
        return False

    def _canonicalize_identity_with_local_model(self, question: str, value: str, evidence: list[Evidence]) -> str:
        if self._model_client is None or len(str(value).split()) != 1:
            return value
        token = normalize(value)
        fuller_candidates: list[str] = []
        for item in evidence:
            for phrase in capitalized_phrases(item.text):
                parts = normalize(phrase).split()
                if len(parts) > 1 and parts[0] == token and phrase not in fuller_candidates:
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
        self._record_model_result(result)
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

    def _discourse_payload_for_evidence(self, evidence: list[Evidence], *, limit: int | None = None) -> list[dict[str, object]]:
        if limit is None:
            limit = int(os.environ.get("KMD_DISCOURSE_PAYLOAD_LIMIT", "32"))
        rel_paths = list(dict.fromkeys(item.rel_path for item in evidence if item.rel_path))
        if not rel_paths:
            return []
        per_kind_limit = max(8, limit // 2)
        placeholders = ",".join("?" for _ in rel_paths[:8])
        frame_rows = self.store.execute(
            f"""
            SELECT
              d.rel_path,
              c.chunk_order,
              f.predicate,
              f.trigger_surface,
              f.source,
              ctx.kind AS context_kind,
              fa.role,
              fa.surface,
              fa.confidence
            FROM frames f
            JOIN source_spans s ON s.span_id=f.span_id
            JOIN chunks c ON c.chunk_id=s.chunk_id
            JOIN documents d ON d.document_id=s.document_id
            LEFT JOIN contexts ctx ON ctx.context_id=f.context_id
            LEFT JOIN frame_arguments fa ON fa.frame_id=f.frame_id
            WHERE d.rel_path IN ({placeholders})
            LIMIT ?
            """,
            (*rel_paths[:8], per_kind_limit),
        ).fetchall()
        relation_rows = self.store.execute(
            f"""
            SELECT
              d.rel_path,
              c.chunk_order,
              r.relation_type,
              r.subject,
              r.predicate,
              r.object,
              r.value,
              ctx.kind AS context_kind,
              r.confidence
            FROM relations r
            JOIN source_spans s ON s.span_id=r.source_span_id
            JOIN chunks c ON c.chunk_id=s.chunk_id
            JOIN documents d ON d.document_id=s.document_id
            LEFT JOIN contexts ctx ON ctx.context_id=r.context_id
            WHERE d.rel_path IN ({placeholders})
            LIMIT ?
            """,
            (*rel_paths[:8], per_kind_limit),
        ).fetchall()
        records: list[dict[str, object]] = []
        records.extend({"record_kind": "frame", **dict(row)} for row in frame_rows)
        records.extend({"record_kind": "condition", **dict(row)} for row in relation_rows)
        return records[:limit]

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
        proposed_clean = str(proposed or "").strip().strip(" .;:,")
        for item in evidence:
            window = self._evidence_window_text(item)
            if evidence_span in window and (
                proposed in window
                or (proposed_clean and proposed_clean in window)
                or self._is_boolean_text(proposed)
            ):
                matches.append(item)
        return matches

    def _answer_with_local_model(self, question: str) -> Answer | None:
        if self._model_client is None:
            return None
        trace = self.model_query_trace
        trace.call_count += 1
        det = deterministic_query_frame(question)
        self._log_progress("kmd-answer model_plan_start")
        use_query_drs_plan = os.environ.get("KMD_QUERY_DRS_PLAN", "0").strip().lower() in {"1", "true", "yes", "on"}
        if use_query_drs_plan:
            model = call_model_query_drs(question, self._model_client)
            if model.get("accepted"):
                query_drs_model = model
                projected = query_frame_from_query_drs(
                    question,
                    model.get("query_drs") if isinstance(model.get("query_drs"), dict) else None,
                )
                if projected is not None:
                    model = {
                        **projected,
                        "accepted": True,
                        "query_drs": query_drs_model.get("query_drs"),
                        "source": "model_query_drs",
                        "prompt_hash": query_drs_model.get("prompt_hash"),
                        "output_hash": query_drs_model.get("output_hash"),
                        "elapsed": query_drs_model.get("elapsed"),
                    }
        else:
            model = call_model_query_plan(question, self._model_client)
        self._record_model_result(model)
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
            if planned_frame.aggregation in {"list", "set"} and expected.answer_type == "content_phrase":
                answer = None
            elif (
                expected.answer_type == "count"
                and planned_frame.aggregation == "count"
                and answer.reason == "bounded DSPG query-frame execution"
            ):
                trace.model_answer_count += 1
                answer.reason = "local model query-frame count aggregation"
                return answer
            elif (
                planned_frame.temporal_scope in {"latest", "earliest"}
                and answer.reason == "bounded DSPG query-frame execution"
                and answer.evidence
            ):
                trace.model_answer_count += 1
                answer.reason = "local model query-frame temporal binding"
                return answer
        if answer and normalize(answer.text) != "unknown":
            if self._verify_with_local_model(question, planned_frame, answer, expected):
                trace.model_answer_count += 1
                answer.reason = "local model query-frame execution"
                return answer
        self._log_progress("kmd-answer evidence_extraction_start")
        evidence_answer = self._answer_with_model_evidence_extraction(question, planned_frame, expected)
        if evidence_answer and normalize(evidence_answer.text) != "unknown":
            if expected.answer_type != "boolean" or self._verify_with_local_model(question, planned_frame, evidence_answer, expected):
                return evidence_answer
        direct = self._answer_with_model_query_evidence(question, expected)
        if direct:
            return direct
        return None

    def _lazy_semantic_frames_enabled(self) -> bool:
        return os.environ.get("KMD_LAZY_LLM_FRAMES", "0").strip().lower() in {"1", "true", "yes", "on"}

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

    def _ensure_context(
        self,
        kind: str,
        parent_context_id: str,
        evidence_surface: str,
        confidence: float,
        holder_surface: str = "",
    ) -> str:
        context_id = stable_id(
            "ctx",
            self.run_id,
            kind,
            parent_context_id,
            normalize(holder_surface),
            normalize(evidence_surface),
        )
        self.store.execute(
            "INSERT OR IGNORE INTO contexts(context_id, run_id, kind, parent_context_id, holder_surface, evidence_surface, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (context_id, self.run_id, kind, parent_context_id or None, holder_surface or None, evidence_surface, confidence),
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
        cache_context = chunk_frame_cache_context(self._model_client)
        cached = self._semantic_cache.get(sentence.text, context=cache_context) if self._semantic_cache else None
        if cached is not None:
            frames = [frame for frame in cached.get("frames", []) if isinstance(frame, dict)]
            metadata = cached.get("metadata") if isinstance(cached.get("metadata"), dict) else {}
            return frames, {
                "source": "cache",
                "frame_count": len(frames),
                "accepted": bool(metadata.get("accepted", True)),
                "reason": str(metadata.get("reason") or ""),
            }
        self._log_progress(f"kmd-llm-frame start {sentence.rel_path}:{sentence.order}")
        result = call_model_chunk_frames(sentence.text, self._model_client, rel_path=sentence.rel_path)
        frames = [frame for frame in result.get("frames", []) if isinstance(frame, dict)] if result.get("accepted") else []
        cacheable_failure = result.get("reason") in {"invalid_json", "schema_validation_failed", "grounding_validation_failed"}
        if self._semantic_cache is not None and (result.get("accepted") or cacheable_failure):
            self._semantic_cache.put(
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
        self._record_model_result(result)
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
            context_holder = str(condition.metadata.get("context_holder") or "").strip()
            semantic_context_id = context_id
            if condition.modality != "asserted":
                semantic_context_id = self._ensure_context(
                    f"modality:{condition.modality}",
                    context_id,
                    condition.evidence_text,
                    condition.confidence,
                    context_holder,
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
                "context_holder": context_holder,
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
                    "INSERT OR IGNORE INTO frame_arguments(argument_id, frame_id, role, mention_id, referent_id, surface, value_type, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        stable_id("arg", semantic_frame_id, arg_index, argument.role, argument.value),
                        semantic_frame_id,
                        argument.role,
                        None,
                        arg_referent_id,
                        argument.value,
                        argument.value_type,
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
            for hypothesis_index, hypothesis in enumerate(condition.metadata.get("identity_hypotheses", [])):
                if not isinstance(hypothesis, dict):
                    continue
                left_text = str(hypothesis.get("left_text") or "").strip()
                right_text = str(hypothesis.get("right_text") or "").strip()
                identity_evidence = str(hypothesis.get("evidence_text") or condition.evidence_text).strip()
                if not left_text or not right_text or not identity_evidence:
                    continue
                if left_text not in sentence.text or right_text not in sentence.text or identity_evidence not in sentence.text:
                    continue
                left_ref = self.store.upsert_referent(self.run_id, left_text, "unknown")
                right_ref = self.store.upsert_referent(self.run_id, right_text, "unknown")
                self.store.execute(
                    """
                    INSERT OR IGNORE INTO identity_hypotheses(
                      hypothesis_id, run_id, left_referent_id, right_referent_id,
                      relation, evidence, confidence, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("idh", self.run_id, semantic_frame_id, "model_identity", hypothesis_index, left_text, right_text),
                        self.run_id,
                        left_ref,
                        right_ref,
                        str(hypothesis.get("relation") or "same_referent").strip() or "same_referent",
                        identity_evidence,
                        float(hypothesis.get("confidence") or condition.confidence),
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

    def _answer_with_model_query_evidence(self, question: str, expected_hint: ExpectedAnswer | None = None) -> Answer | None:
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
            for sentence, score in candidates[: int(os.environ.get("KMD_EVIDENCE_PAYLOAD_LIMIT", "10"))]
        ]
        payload = self._evidence_payload(evidence, limit=len(evidence))
        if not payload:
            return None
        trace = self.model_query_trace
        discourse_payload = self._discourse_payload_for_evidence(evidence)
        fallback_client = self._fallback_model_client()
        if fallback_client is None:
            return None
        model = call_model_query_evidence_answer(question, payload, fallback_client, discourse_records=discourse_payload)
        self._record_model_result(model)
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
            return Answer("unknown", reason="local model query-DRS insufficient evidence")
        proposed = str(model.get("answer") or "")
        evidence_span = str(model.get("evidence_span") or "")
        answer_type = str(model.get("answer_type") or "content_phrase")
        frame = frame_from_mapping(question, model.get("query_frame") if isinstance(model.get("query_frame"), dict) else None)
        expected = expected_hint if expected_hint and expected_hint.answer_type != "unknown" else self._expected_from_frame(frame)
        if answer_type:
            direct_expected = ExpectedAnswer(answer_type if answer_type in {
                "person", "actor", "organization", "identifier", "url", "file_path", "count",
                "state", "date_time", "boolean", "content_phrase", "metadata_value", "unknown",
            } else expected.answer_type, allow_metadata_evidence=answer_type == "metadata_value")  # type: ignore[arg-type]
            if expected.answer_type == "unknown" and direct_expected.answer_type != "unknown":
                expected = direct_expected
        if not proposed:
            trace.evidence_rejected_count += 1
            return None
        proposed = self._shortest_model_answer_value(proposed, answer_type, frame)
        if not evidence_span:
            trace.evidence_rejected_count += 1
            return None
        else:
            if self._is_boolean_text(proposed) and not self._boolean_answer_has_target_grounding(frame, evidence_span):
                trace.evidence_rejected_count += 1
                return Answer("unknown", reason="local model boolean answer lacked target grounding")
            matching = self._matching_evidence(evidence, evidence_span, proposed)
            if not matching and classify_value(proposed) == "count":
                matching = [item for item in evidence if evidence_span in self._evidence_window_text(item)]
            if not matching:
                trace.evidence_rejected_count += 1
                return None
        support = list(matching[:3])
        if expected.answer_type in {"person", "actor"} or classify_value(proposed) == "person":
            proposed_norm = normalize(proposed)
            for item in evidence:
                if item not in support and proposed_norm and proposed_norm in normalize(self._evidence_window_text(item)):
                    support.append(item)
                if len(support) >= 6:
                    break
        answer = Answer(proposed, 0.78, support[:6], "local model query-DRS evidence verification", expected.answer_type)
        finalized = self._finalize_answer(question, answer, expected, "local model query-DRS evidence verification")
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
        expected = expected or self._expected_from_frame(frame)
        required = list(frame.target_anchors) if frame.target_anchors else None
        candidates = self._search(question, limit=int(os.environ.get("KMD_EVIDENCE_SEARCH_LIMIT", "18")), required=required)
        if len(candidates) < 4 and required:
            candidates = self._search(question, limit=int(os.environ.get("KMD_EVIDENCE_SEARCH_LIMIT", "18")), required=None)
        if not candidates:
            return None
        evidence = [self._evidence(sentence, score) for sentence, score in candidates[: int(os.environ.get("KMD_EVIDENCE_PAYLOAD_LIMIT", "10"))]]
        payload = self._evidence_payload(evidence, limit=len(evidence))
        if not payload:
            return None
        trace = self.model_query_trace
        trace.evidence_call_count += 1
        fallback_client = self._fallback_model_client()
        if fallback_client is None:
            return None
        model = call_model_evidence_answer(question, expected.answer_type, payload, fallback_client)
        self._record_model_result(model)
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
        model_answer_type = str(model.get("answer_type") or "unknown")
        if model_answer_type in {
            "person", "actor", "organization", "identifier", "url", "file_path", "count",
            "state", "date_time", "boolean", "content_phrase", "metadata_value",
        }:
            model_expected = ExpectedAnswer(model_answer_type, allow_metadata_evidence=model_answer_type == "metadata_value")  # type: ignore[arg-type]
            if expected.answer_type == "unknown":
                expected = model_expected
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
        finalized = self._finalize_answer(question, answer, expected, "local model bounded evidence extraction")
        if not finalized:
            trace.evidence_rejected_count += 1
            return None
        trace.evidence_accepted_count += 1
        trace.model_answer_count += 1
        return finalized

    def _shortest_model_answer_value(self, proposed: str, answer_type: str, frame: QueryFrame) -> str:
        text = str(proposed or "").strip()
        if not text:
            return text
        if answer_type == "boolean" or self._is_boolean_text(text):
            return text
        parts = answer_parts(text)
        if len(parts) > 1 and parts[0]:
            text = parts[0]
        return text

    def _is_boolean_text(self, value: str) -> bool:
        return re.match(r"^(yes|no)(?:$|[;,:.!?]\s+)", normalize(value)) is not None

    def _boolean_answer_has_target_grounding(self, frame: QueryFrame, evidence_span: str) -> bool:
        anchors = [normalize(anchor) for anchor in frame.target_anchors if normalize(anchor)]
        if not anchors:
            return True
        if "\n" in str(evidence_span or "").strip():
            return False
        span_norm = normalize(evidence_span)
        return all(self._anchor_has_grounded_token(anchor, span_norm) for anchor in anchors)

    def _bounded_evidence_covers_targets(self, frame: QueryFrame, evidence: list[Evidence]) -> bool:
        anchors = [normalize(anchor) for anchor in frame.target_anchors if normalize(anchor)]
        if not anchors:
            return True
        material = normalize("\n".join(self._evidence_window_text(item) for item in evidence[:6]))
        return all(self._anchor_has_grounded_token(anchor, material) for anchor in anchors)

    def _anchor_has_grounded_token(self, anchor: str, material_norm: str) -> bool:
        tokens = [token for token in content_tokens(anchor) if len(token) > 2]
        if not tokens:
            return normalize(anchor) in material_norm
        material_tokens = set(content_tokens(material_norm))
        expanded_material = set(material_tokens)
        for token in material_tokens:
            expanded_material.update(term_variants(token))
        for token in tokens:
            if token in expanded_material:
                return True
            if any(variant in expanded_material for variant in term_variants(token)):
                return True
        return False

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
        return self._finalize_answer(question, bounded_answer, expected, "bounded DSPG query-frame execution")

    def _answer_has_source_grounding(self, answer: Answer) -> bool:
        if normalize(answer.text) == "unknown":
            return True
        return any(evidence.rel_path and evidence.text for evidence in answer.evidence)

    def _finalize_answer(self, question: str, answer: Answer, expected: ExpectedAnswer, source: str) -> Answer | None:
        if normalize(answer.text) == "unknown":
            return answer
        has_metadata_evidence = any(is_metadata_evidence_text(evidence.text) for evidence in answer.evidence)
        if expected.answer_type == "unknown":
            model_type = answer.answer_type if answer.answer_type not in {"", "unknown"} else classify_value(answer.text)
            if model_type != "unknown":
                expected = ExpectedAnswer(model_type, allow_metadata_evidence=has_metadata_evidence or model_type == "metadata_value")  # type: ignore[arg-type]
        if expected.answer_type == "content_phrase" and source.startswith("local model"):
            structural_type = classify_value(answer.text)
            if structural_type in {"url", "identifier", "file_path", "date_time", "count"}:
                expected = ExpectedAnswer(structural_type)  # type: ignore[arg-type]
        if not self._answer_has_source_grounding(answer):
            return None
        if has_metadata_evidence and not expected.allow_metadata_evidence:
            return None
        canonical = canonicalize_answer(expected, answer.text)
        if canonical and source.startswith("local model") and expected.answer_type in {"boolean", "content_phrase", "state", "metadata_value"}:
            canonical = self._canonicalize_model_answer_with_local_model(question, canonical, expected, answer.evidence) or canonical
        if (
            canonical
            and source.startswith("local model")
            and expected.answer_type in {"person", "actor"}
            and len(str(canonical).split()) == 1
            and answer.evidence
        ):
            canonical = self._canonicalize_identity_with_local_model(question, canonical, answer.evidence) or canonical
        if not canonical:
            return None
        return Answer(canonical, answer.confidence, answer.evidence, source, expected.answer_type)

    def _canonicalize_model_answer_with_local_model(
        self,
        question: str,
        value: str,
        expected: ExpectedAnswer,
        evidence: list[Evidence],
    ) -> str:
        if self._model_client is None:
            return value
        if expected.answer_type not in {"person", "actor", "organization", "boolean", "content_phrase", "state", "metadata_value"}:
            return value
        if len(str(value).split()) < 2:
            return value
        evidence_payload = self._evidence_payload(evidence, limit=6)
        if not evidence_payload:
            return value
        trace = self.model_query_trace
        trace.canonicalization_call_count += 1
        result = call_model_answer_canonicalization(
            question,
            value,
            expected.answer_type,
            evidence_payload,
            self._model_client,
        )
        self._record_model_result(result)
        if result.get("prompt_hash"):
            trace.prompt_hashes = [*list(trace.prompt_hashes or []), str(result["prompt_hash"])][-20:]
        if result.get("output_hash"):
            trace.response_hashes = [*list(trace.response_hashes or []), str(result["output_hash"])][-20:]
        if not result.get("accepted"):
            trace.canonicalization_rejected_count += 1
            return value
        proposed = str(result.get("answer") or "")
        canonical = canonicalize_answer(expected, proposed)
        if not canonical:
            trace.canonicalization_rejected_count += 1
            return value
        trace.canonicalization_accepted_count += 1
        return canonical

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

    def _target_anchors(self, question: str) -> list[str]:
        return capitalized_phrases(question)
