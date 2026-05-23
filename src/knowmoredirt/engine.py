"""KnowMoreDiRT raw-folder DRT/DSPG question-answering engine.

The engine initializes from one arbitrary folder path, reads all readable files
as raw text, builds grounded DSPG records, and answers questions by matching a
generic query frame against bounded discourse structures.  Optional local-model
use is restricted to query-frame refinement and evidence-only answer extraction.
"""

from __future__ import annotations

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
from .extractors import capitalized_phrases
from .ingest import ingest_folder
from .index import LexicalIndex
from .model import LocalModelClient
from .model_planner import (
    ModelQueryTrace,
    call_model_evidence_answer,
    call_model_query_plan,
    deterministic_plan as deterministic_query_frame,
    normalize_model_plan,
)
from .models import Answer, Evidence, Sentence
from .query import QueryFrame, frame_from_mapping, plan_question
from .text import content_tokens, is_low_semantic_noise, normalize


@dataclass
class EngineStats:
    document_count: int
    sentence_count: int


class KnowMoreDiRTEngine:
    """Internal session object backing the two-function public API."""

    def __init__(self, folder_path: str | Path) -> None:
        self.folder_path = Path(folder_path)
        self.store, self.run_id, self.documents, self.sentences = ingest_folder(self.folder_path)
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
        self._use_local_model = os.environ.get("KMD_USE_LOCAL_MODEL", "").strip().lower() in {"1", "true", "yes", "on"}
        self._model_client = LocalModelClient() if self._use_local_model else None
        self.model_query_trace = ModelQueryTrace(enabled=self._use_local_model, prompt_hashes=[], response_hashes=[])
        self.last_answer: Answer | None = None
        self.last_bounded_diagnostics: dict[str, object] = {}

    def dspg_counts(self) -> dict[str, int]:
        return self.store.counts()

    def dspg_integrity(self) -> str:
        return self.store.integrity_check()

    def answer(self, question: str) -> Answer:
        text = str(question or "").strip()
        if not text:
            return Answer("unknown", reason="empty question")
        expected = infer_expected_answer(text)
        frame = plan_question(text)

        if self._use_local_model:
            model_answer = self._answer_with_local_model(text, expected, frame)
            if model_answer and normalize(model_answer.text) != "unknown":
                self.last_answer = model_answer
                return model_answer

        bounded = self._answer_with_bounded_dspg(text, frame, expected)
        if bounded and normalize(bounded.text) != "unknown":
            self.last_answer = bounded
            return bounded

        if self._use_local_model:
            evidence_answer = self._answer_with_model_evidence_extraction(text, frame.as_dict())
            if evidence_answer and normalize(evidence_answer.text) != "unknown":
                self.last_answer = evidence_answer
                return evidence_answer

        answer = Answer("unknown", reason="no complete grounded DSPG match")
        self.last_answer = answer
        return answer

    def _evidence(self, sentence: Sentence, score: float = 1.0) -> Evidence:
        return Evidence(sentence.rel_path, sentence.text, score)

    def _answer_with_local_model(self, question: str, expected: ExpectedAnswer, frame: QueryFrame) -> Answer | None:
        if self._model_client is None:
            return None
        trace = self.model_query_trace
        trace.call_count += 1
        det = deterministic_query_frame(question)
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
        answer = self._answer_with_bounded_dspg(question, planned_frame, expected)
        if answer and normalize(answer.text) != "unknown":
            trace.model_answer_count += 1
            answer.reason = "local model query-frame execution"
            return answer
        return None

    def _answer_with_model_evidence_extraction(self, question: str, plan: dict[str, object]) -> Answer | None:
        if self._model_client is None:
            return None
        expected = infer_expected_answer(question)
        candidates = self._search(question, limit=10)
        if not candidates:
            return None
        evidence = [self._evidence(sentence, score) for sentence, score in candidates[:10]]
        payload = [
            {"source": item.rel_path, "text": item.text[:1200]}
            for item in evidence
            if item.rel_path and item.text
        ]
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
        matching = [item for item in evidence if evidence_span in item.text and proposed in item.text]
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
        if not canonical:
            return None
        return Answer(canonical, answer.confidence, answer.evidence, answer.reason or source, expected.answer_type)

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
            for offset in [-2, -1, 1, 2]:
                neighbor = document_sentences.get(sentence.order + offset)
                if neighbor:
                    previous = combined.get(neighbor.sentence_id, (neighbor, 0.0))[1]
                    combined[neighbor.sentence_id] = (neighbor, max(previous, score * 0.55))

        adjusted: list[tuple[Sentence, float]] = []
        for sentence, score in combined.values():
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
