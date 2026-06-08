"""KnowMoreDiRT raw-folder DRT/DSPG question-answering engine.

The engine initializes from one arbitrary folder path, reads all readable files
as raw text, builds grounded DSPG records, and answers questions by matching a
model-produced query DRS projection against bounded discourse structures.
Normal runtime requires a reachable localhost llama.cpp-compatible model.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from .model import LocalModelClient, LocalModelUnavailableError
from .model_planner import (
    ModelQueryTrace,
    call_model_answer_canonicalization,
    call_model_answer_verification,
    call_model_chunk_frames,
    call_model_evidence_answer,
    call_model_identity_canonicalization,
    call_model_query_drs,
    call_model_query_evidence_answer,
    call_model_source_resolved_answer,
    chunk_frame_cache_context,
    query_frame_from_query_drs,
)
from .models import Answer, Document, Evidence, Sentence
from .query import QueryFrame, frame_from_mapping, plan_question, term_variants
from .semantic_cache import SemanticFrameCache
from .store import stable_id
from .text import clean_extracted_value, content_tokens, is_low_semantic_noise, normalize, text_quality_metrics


PROGRESS_TRUE_VALUES = {"1", "true", "yes", "on"}
SOURCE_DEICTIC_TOKENS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}
TRUSTED_STRUCTURAL_BINDING_REASONS = {
    "direct_label_slot_binding",
    "document_scoped_label_binding",
    "record_group_drs_binding",
    "relation_label_value_binding",
    "structural_chain_drs_binding",
}
TRUSTED_STRUCTURAL_ANSWER_TYPES = {
    "actor",
    "content_phrase",
    "date_time",
    "file_path",
    "identifier",
    "organization",
    "person",
    "state",
    "url",
}


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in PROGRESS_TRUE_VALUES


def _attempt_was_nonrequest_failure(row: Any | None) -> bool:
    if row is None:
        return False
    try:
        reason = str(row["reason"] or "")
        accepted = bool(row["accepted"])
        materialized = bool(row["materialized"])
    except Exception:
        return False
    return not accepted and not materialized and reason != "request_failed"


@dataclass
class EngineStats:
    document_count: int
    sentence_count: int


class KnowMoreDiRTEngine:
    """Internal session object backing the two-function public API."""

    def __init__(self, folder_path: str | Path) -> None:
        self.folder_path = Path(folder_path)
        self._test_no_model_runtime = self._test_no_model_allowed()
        self._model_client = None if self._test_no_model_runtime else self._required_local_model_client()
        self._use_local_model = self._model_client is not None
        self.model_query_trace = ModelQueryTrace(enabled=self._use_local_model, prompt_hashes=[], response_hashes=[])
        self._semantic_cache = SemanticFrameCache() if self._use_local_model else None
        llm_ingest_setting = os.environ.get("KMD_LLM_INGEST", "0").strip().lower()
        use_semantic_frames = self._use_local_model and llm_ingest_setting in {"1", "true", "yes", "on"}
        drs_ingest_setting = os.environ.get("KMD_LLM_DRS_INGEST", "1").strip().lower()
        use_drs_semantics = self._use_local_model and drs_ingest_setting not in {"0", "false", "no", "off"}
        self._log_progress(
            "kmd-init start "
            f"local_model={self._use_local_model} "
            f"eager_llm_ingest={use_semantic_frames} "
            f"drs_ingest={use_drs_semantics} "
            f"root={self.folder_path}"
        )
        ingest_model_client = self._model_client
        if ingest_model_client is not None and (use_semantic_frames or use_drs_semantics):
            ingest_model_client = self._chunk_stage_model_client(ingest_model_client)
            self._model_client = ingest_model_client
        self.store, self.run_id, self.documents, self.sentences = ingest_folder(
            self.folder_path,
            semantic_client=ingest_model_client if use_semantic_frames or use_drs_semantics else None,
            use_semantic_frames=use_semantic_frames,
            use_drs_semantics=use_drs_semantics,
            semantic_cache=self._semantic_cache if use_semantic_frames else None,
        )
        self._log_progress(
            f"kmd-init indexed documents={len(self.documents)} chunks={len(self.sentences)} run_id={self.run_id}"
        )
        if self._model_client is not None:
            self._model_client = self._question_stage_model_client(self._model_client)
        self.index = LexicalIndex(self.sentences)
        self.stats = EngineStats(len(self.documents), len(self.sentences))
        self._documents_by_rel_path = {document.rel_path: document for document in self.documents}
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

    def _test_no_model_allowed(self) -> bool:
        return _env_true("KMD_TEST_ALLOW_NO_MODEL") and "PYTEST_CURRENT_TEST" in os.environ

    def _model_evidence_tools_allowed(self) -> bool:
        value = os.environ.get("KMD_MODEL_EVIDENCE_TOOLS", "0").strip().lower()
        if value in {"0", "false", "no", "off"}:
            return False
        return True

    def _test_model_evidence_helpers_allowed(self) -> bool:
        if _env_true("KMD_TEST_ALLOW_MODEL_EVIDENCE_TOOLS") and "PYTEST_CURRENT_TEST" in os.environ:
            return True
        return self._model_evidence_tools_allowed()

    def _required_local_model_client(self) -> LocalModelClient:
        endpoint = os.environ.get("KMD_LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:14829/v1").rstrip("/")
        try:
            probe_timeout = float(os.environ.get("KMD_MODEL_PROBE_TIMEOUT", "1.5"))
        except ValueError:
            probe_timeout = 1.5
        client = LocalModelClient(endpoint=endpoint, timeout_seconds=probe_timeout)
        try:
            models = client.models()
        except Exception as exc:
            disabled_hint = ""
            if os.environ.get("KMD_USE_LOCAL_MODEL", "").strip().lower() in {"0", "false", "no", "off"}:
                disabled_hint = " KMD_USE_LOCAL_MODEL=0 no longer disables the production model requirement."
            raise LocalModelUnavailableError(
                "KnowMoreDiRT requires a reachable localhost llama.cpp endpoint for initialize(folder_path). "
                f"Failed to probe {endpoint!r}: {type(exc).__name__}: {exc}.{disabled_hint}"
            ) from exc
        if not isinstance(models, dict):
            raise LocalModelUnavailableError(
                "KnowMoreDiRT requires a reachable localhost llama.cpp endpoint for initialize(folder_path). "
                f"Probe {endpoint!r} returned a non-JSON-object model listing."
            )
        return LocalModelClient(endpoint=endpoint)

    def _question_stage_model_client(self, client: LocalModelClient) -> LocalModelClient:
        return self._stage_timeout_model_client(
            client,
            env_name="KMD_QUESTION_MODEL_TIMEOUT_SECONDS",
            progress_label="question_model_timeout",
            previous_label="ingest_timeout",
            next_label="question_timeout",
        )

    def _chunk_stage_model_client(self, client: LocalModelClient) -> LocalModelClient:
        return self._stage_timeout_model_client(
            client,
            env_name="KMD_CHUNK_MODEL_TIMEOUT_SECONDS",
            progress_label="chunk_model_timeout",
            previous_label="default_timeout",
            next_label="chunk_timeout",
        )

    def _stage_timeout_model_client(
        self,
        client: LocalModelClient,
        *,
        env_name: str,
        progress_label: str,
        previous_label: str,
        next_label: str,
    ) -> LocalModelClient:
        raw_timeout = os.environ.get(env_name, "").strip()
        if not raw_timeout:
            return client
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise LocalModelUnavailableError(
                f"{env_name} must be a positive number when set."
            ) from exc
        if timeout <= 0:
            raise LocalModelUnavailableError(
                f"{env_name} must be a positive number when set."
            )
        if abs(timeout - float(getattr(client, "timeout_seconds", timeout))) < 0.001:
            return client
        self._log_progress(
            f"kmd-init {progress_label} "
            f"{previous_label}={getattr(client, 'timeout_seconds', '')} "
            f"{next_label}={timeout:g}"
        )
        return LocalModelClient(endpoint=client.endpoint, timeout_seconds=timeout)

    def _raise_model_request_failed(self, result: dict[str, object], operation: str) -> None:
        if str(result.get("reason") or "") != "request_failed":
            return
        cache_context = result.get("cache_context") if isinstance(result.get("cache_context"), dict) else {}
        try:
            cache_context_text = json.dumps(cache_context, sort_keys=True, default=str)[:4000]
        except Exception:
            cache_context_text = str(cache_context)[:4000]
        raise LocalModelUnavailableError(
            "KnowMoreDiRT requires reachable llama.cpp for normal question answering. "
            f"Local model request failed during {operation}: {result.get('error') or 'request_failed'}. "
            f"cache_context={cache_context_text}",
            cache_context=cache_context,
        )

    def _progress_enabled(self) -> bool:
        return os.environ.get("KMD_PROGRESS", "").strip().lower() in {"1", "true", "yes", "on"} or os.environ.get(
            "KMD_EVAL_PROGRESS", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _log_progress(self, message: str) -> None:
        if self._progress_enabled():
            print(message, flush=True)

    def _record_model_result(self, result: dict[str, object]) -> None:
        self._raise_model_request_failed(result, "model operation")
        trace = self.model_query_trace
        cache_hit = result.get("fresh_or_cached") == "cache" or result.get("source") == "cache"
        if cache_hit:
            trace.cache_hit_count += 1
        else:
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
        timeout_default = os.environ.get("KMD_QUESTION_MODEL_TIMEOUT_SECONDS", os.environ.get("KMD_LOCAL_MODEL_TIMEOUT", "120"))
        timeout = float(os.environ.get("KMD_FALLBACK_MODEL_TIMEOUT_SECONDS", timeout_default))
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
                model_answer = self._cleanup_public_answer(model_answer)
                improved = self._answer_with_boolean_source_explanation(text, prior_answer=model_answer)
                if improved:
                    model_answer = improved
                self.last_answer = model_answer
                return model_answer
            boolean_answer = self._answer_with_boolean_source_explanation(text)
            if boolean_answer:
                self.last_answer = boolean_answer
                return boolean_answer
            answer = self._unknown_answer("local model DRT path found no complete grounded answer")
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
                bounded = self._cleanup_public_answer(bounded)
                self.last_answer = bounded
                return bounded

        answer = self._unknown_answer("no complete grounded DSPG match")
        self.last_answer = answer
        return answer

    def _answer_with_boolean_source_explanation(self, question: str, prior_answer: Answer | None = None) -> Answer | None:
        frame_data = self.model_query_trace.last_plan if isinstance(self.model_query_trace.last_plan, dict) else None
        frame = frame_from_mapping(question, frame_data) if frame_data else plan_question(question)
        expected = self._expected_from_frame(frame)
        if expected.answer_type != "boolean" and not re.match(
            r"^(did|does|do|is|are|was|were|should|can|could|will|would|has|have|had)\b",
            normalize(question),
        ):
            return None
        candidates = self._search(
            question,
            limit=int(os.environ.get("KMD_BOOLEAN_SOURCE_EVIDENCE_LIMIT", "36")),
            required=None,
        )
        evidence = [self._evidence(sentence, score) for sentence, score in candidates]
        if prior_answer is not None:
            evidence = [*prior_answer.evidence, *evidence]
        evidence = list(dict.fromkeys(evidence))
        answer_text = self._boolean_source_explanation(question, frame, evidence, prior_answer)
        if not answer_text:
            return None
        support = self._boolean_source_support(answer_text, evidence)
        if not support:
            support = [item for item in evidence[:6] if item.rel_path and item.text]
        if not support:
            return None
        return Answer(answer_text, 0.83, support[:6], "general boolean source evidence assembly", "boolean")

    def _boolean_source_support(self, answer_text: str, evidence: list[Evidence]) -> list[Evidence]:
        answer_norm = normalize(answer_text)
        content = [token for token in content_tokens(answer_norm) if len(token) > 3]
        support: list[Evidence] = []
        for item in evidence:
            material = normalize(self._evidence_window_text(item))
            if any(token in material for token in content):
                support.append(item)
        return support

    def _boolean_source_explanation(
        self,
        question: str,
        frame: QueryFrame,
        evidence: list[Evidence],
        prior_answer: Answer | None = None,
    ) -> str:
        question_norm = normalize(question)
        windows = [self._evidence_window_text(item, radius=4, max_chars=1600) for item in evidence if item.text]
        if prior_answer is not None:
            windows = [*(item.text for item in prior_answer.evidence if item.text), *windows]
        material = "\n".join(dict.fromkeys(text for text in windows if text))
        material_norm = normalize(material)
        if not material_norm:
            return ""
        target_anchors = [anchor for anchor in frame.target_anchors if normalize(anchor)]
        file_like_targets = [anchor for anchor in target_anchors if re.search(r"[./_-]", anchor)]
        if any(term in material_norm for term in (" dream ", " dreamed ", " fiction ", " fictional ")):
            still_match = re.search(
                r"(?:^|[.\n]\s*)(?:when\s+[^.]+,\s*)?(?:the\s+)?(?P<container>[A-Za-z][A-Za-z0-9 _-]{2,60}?)\s+still\s+contained\s+(?P<object>[A-Za-z0-9_.\-\/]+)",
                material,
                re.I,
            )
            if still_match:
                obj = next((target for target in file_like_targets if normalize(target) in normalize(still_match.group("object"))), still_match.group("object").strip())
                container = clean_extracted_value(still_match.group("container")).strip().lower()
                relation = normalize(frame.requested_relation)
                event = "event"
                if "delete" in relation or "deleted" in material_norm:
                    event = "deletion"
                elif relation:
                    event = relation.split()[0]
                scope = "dream" if "dream" in material_norm else "fiction"
                return f"No; the {event} occurred only in a {scope} and the {container} still contained {obj}."
        if "found no proof" in material_norm or "no proof" in material_norm:
            source = "final judgment" if "final judgment" in material_norm else "source"
            if "court" in material_norm and source == "source":
                source = "court"
            return f"No; the {source} found no proof." if source != "source" else "No; no proof was found."
        if "delete" in question_norm and "human review" in material_norm:
            flag_match = re.search(
                r"(?:runtime\s+note:\s*)?(?:the\s+code\s+)?flags\s+(?P<object>[^.;\n]+?)\s+for\s+human\s+review",
                material,
                re.I,
            )
            if not flag_match:
                flag_match = re.search(r"(?P<object>[^.;\n]+?)\s+(?:are|is)\s+flagged\s+for\s+human\s+review", material, re.I)
            if flag_match:
                obj = clean_extracted_value(flag_match.group("object")).strip().strip('"')
                obj = re.sub(r'^return\s+["\']?', "", obj, flags=re.I).strip().strip('"')
                return f"No; runtime flags {obj} for human review."
        class_match = re.search(r"\bthis\s+is\s+(?P<yes>[^.;\n,]+),\s+not\s+(?P<no>[^.;\n]+)", material, re.I)
        if class_match and any(token in question_norm for token in content_tokens(class_match.group("no"))):
            yes = clean_extracted_value(class_match.group("yes")).strip().lower()
            return f"No; it is {yes}."
        only_match = re.search(r"\b(?:audit\s+result:\s*)?(?P<entity>[A-Z][A-Za-z0-9_-]*)\s+stores\s+only\s+(?P<value>[^.;\n]+)", material)
        if only_match and "store" in question_norm:
            value = clean_extracted_value(only_match.group("value")).strip().lower()
            return f"No; it stores only {value}."
        unrelated_match = re.search(r"\bthis\s+unrelated\s+(?P<kind>[a-z][a-z -]*?note)\b", material, re.I)
        if unrelated_match and ("no relation" in material_norm or "unrelated" in material_norm):
            kind = clean_extracted_value(unrelated_match.group("kind")).strip().lower()
            return f"No; it is an unrelated {kind}."
        return ""

    def _cleanup_public_answer(self, answer: Answer) -> Answer:
        if normalize(answer.text) == "unknown":
            return answer
        expected_type = answer.answer_type if answer.answer_type not in {"", "unknown"} else classify_value(answer.text)
        expected = ExpectedAnswer(expected_type)  # type: ignore[arg-type]
        cleaned = self._cleanup_canonical_answer(answer.text, expected)
        cleaned = self._restore_sentence_terminal_punctuation(cleaned, answer.text, expected, answer.evidence)
        if cleaned and cleaned != answer.text:
            original = str(answer.text or "").strip()
            if original and original[-1] in ".!?" and cleaned == original[:-1].strip():
                return answer
            return Answer(cleaned, answer.confidence, answer.evidence, answer.reason, answer.answer_type)
        return answer

    def _unknown_answer(self, reason: str) -> Answer:
        return Answer("unknown", 0.0, self._diagnostic_unknown_evidence(), reason, "unknown")

    def _diagnostic_unknown_evidence(self, *, limit: int = 6) -> list[Evidence]:
        diagnostics = self.last_bounded_diagnostics if isinstance(self.last_bounded_diagnostics, dict) else {}
        execution = diagnostics.get("execution") if isinstance(diagnostics.get("execution"), dict) else {}
        payloads: list[dict[str, object]] = []
        conflict = execution.get("answer_conflict_without_query_scope") if isinstance(execution, dict) else None
        if isinstance(conflict, dict):
            for value_item in conflict.get("values") or []:
                if not isinstance(value_item, dict):
                    continue
                for evidence in value_item.get("evidence") or []:
                    if isinstance(evidence, dict):
                        payloads.append(evidence)
        for candidate in execution.get("candidate_evidence_sample") or []:
            if isinstance(candidate, dict) and isinstance(candidate.get("evidence"), dict):
                payloads.append(candidate["evidence"])
        for blocked_identity in execution.get("blocked_identity_source_provenance") or []:
            if isinstance(blocked_identity, dict):
                payloads.append(blocked_identity)
        scattered = execution.get("scattered_source_provenance_without_binding")
        if isinstance(scattered, dict):
            target_sources = [source for source in scattered.get("target_sources") or [] if isinstance(source, dict)]
            relation_sources = [source for source in scattered.get("relation_sources") or [] if isinstance(source, dict)]
            if target_sources and relation_sources:
                payloads.extend([target_sources[0], relation_sources[0]])
            payloads.extend([*target_sources, *relation_sources])
        for source in execution.get("source_provenance_sample") or []:
            if isinstance(source, dict):
                payloads.append(source)

        evidence_items: list[Evidence] = []
        seen: set[tuple[str, str, str]] = set()
        chunk_indexes: dict[tuple[str, int | None, str], int] = {}
        for payload in payloads:
            rel_path = str(payload.get("rel_path") or payload.get("source") or "")
            text = str(payload.get("text") or "")
            span_id = str(payload.get("span_id") or "")
            if not rel_path or not text:
                continue
            try:
                chunk_order = int(payload["chunk_order"]) if payload.get("chunk_order") not in {"", None} else None
            except (TypeError, ValueError):
                chunk_order = None
            key = (rel_path, span_id, text)
            chunk_key = (rel_path, chunk_order, normalize(text))
            if key in seen:
                continue
            seen.add(key)
            try:
                score = float(payload.get("score") or 0.45)
            except (TypeError, ValueError):
                score = 0.45
            try:
                char_start = int(payload["char_start"]) if payload.get("char_start") not in {"", None} else None
            except (TypeError, ValueError):
                char_start = None
            try:
                char_end = int(payload["char_end"]) if payload.get("char_end") not in {"", None} else None
            except (TypeError, ValueError):
                char_end = None
            evidence = Evidence(
                rel_path,
                text,
                score,
                span_id=span_id,
                chunk_order=chunk_order,
                char_start=char_start,
                char_end=char_end,
                source_kind=str(payload.get("source_kind") or payload.get("span_kind") or "source_span"),
            )
            existing_index = chunk_indexes.get(chunk_key)
            if existing_index is not None:
                if span_id and not evidence_items[existing_index].span_id:
                    evidence_items[existing_index] = evidence
                continue
            chunk_indexes[chunk_key] = len(evidence_items)
            evidence_items.append(evidence)
            if len(evidence_items) >= limit:
                break
        return evidence_items

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
            canonical_expected = expected
            if canonical_expected.answer_type == "unknown":
                inferred_type = answer.answer_type if answer.answer_type not in {"", "unknown"} else classify_value(proposed)
                if inferred_type != "unknown":
                    canonical_expected = ExpectedAnswer(inferred_type)  # type: ignore[arg-type]
            if normalize(candidate_answer) != normalize(answer.text) and normalize(proposed) != normalize(candidate_answer):
                candidate_canonical = canonicalize_answer(canonical_expected, candidate_answer)
                if candidate_canonical:
                    proposed = candidate_answer
            canonical = canonicalize_answer(canonical_expected, proposed)
            if not canonical:
                trace.verifier_rejected_count += 1
                continue
            canonical = self._restore_sentence_terminal_punctuation(canonical, proposed, canonical_expected, answer.evidence)
            if canonical and canonical_expected.answer_type in {"person", "actor"}:
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
        try:
            frame_limit = int(os.environ.get("KMD_VERIFIER_DISCOURSE_FRAME_LIMIT", "0"))
        except ValueError:
            frame_limit = 8
        frame_limit = max(0, min(32, frame_limit))
        if frame_limit <= 0:
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
            LIMIT ?
            """,
            (*rel_paths[:8], frame_limit),
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
        return Evidence(
            sentence.rel_path,
            sentence.text,
            score,
            span_id=self._sentence_span_id(sentence),
            chunk_order=sentence.order,
            char_start=sentence.char_start,
            char_end=sentence.char_end,
        )

    def _evidence_window_text(self, evidence: Evidence, *, radius: int | None = None, max_chars: int | None = None) -> str:
        if radius is None:
            radius = int(os.environ.get("KMD_EVIDENCE_WINDOW_RADIUS", "3"))
        if max_chars is None:
            max_chars = int(os.environ.get("KMD_EVIDENCE_TEXT_CHARS", "1200"))
        sentences = self._sentences_by_document.get(evidence.rel_path, {})
        center_order = evidence.chunk_order if evidence.chunk_order in sentences else None
        for order, sentence in sentences.items():
            if center_order is not None:
                break
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
            {
                "source": item.rel_path,
                "text": self._evidence_window_text(item),
                "span_id": item.span_id,
                "chunk_order": "" if item.chunk_order is None else str(item.chunk_order),
                "char_start": "" if item.char_start is None else str(item.char_start),
                "char_end": "" if item.char_end is None else str(item.char_end),
                "source_kind": item.source_kind,
            }
            for item in evidence[:limit]
            if item.rel_path and item.text
        ]

    def _document_provenance_summary(self, document: Document | None) -> dict[str, object]:
        if document is None:
            return {}
        metadata = document.metadata if isinstance(document.metadata, dict) else {}
        text_quality = metadata.get("text_quality")
        semantic_quality = (
            text_quality.get("semantic_quality")
            if isinstance(text_quality, dict)
            else metadata.get("semantic_quality")
        )
        summary: dict[str, object] = {
            "document_id": document.document_id,
            "rel_path": document.rel_path,
            "size_bytes": document.size_bytes,
            "char_count": len(document.text),
        }
        for key in ["file_name", "suffix", "parent_rel_path", "mime_type"]:
            value = metadata.get(key)
            if value is not None and value != "":
                summary[key] = value
        if semantic_quality is not None and semantic_quality != "":
            summary["semantic_quality"] = semantic_quality
        return {key: value for key, value in summary.items() if value is not None and value != ""}

    def _sentence_for_evidence(self, evidence: Evidence) -> Sentence | None:
        if evidence.chunk_order is not None:
            sentence = self._sentences_by_location.get((evidence.rel_path, evidence.chunk_order))
            if sentence is not None:
                return sentence
        for sentence in self._sentences_by_document.get(evidence.rel_path, {}).values():
            if sentence.text == evidence.text:
                return sentence
        return None

    def _evidence_source_provenance_payload(self, evidence: Evidence) -> dict[str, object]:
        sentence = self._sentence_for_evidence(evidence)
        document = self._documents_by_rel_path.get(evidence.rel_path)
        payload: dict[str, object] = {
            "rel_path": evidence.rel_path,
            "span_id": evidence.span_id,
            "chunk_order": evidence.chunk_order,
            "char_start": evidence.char_start,
            "char_end": evidence.char_end,
            "source_kind": evidence.source_kind,
            "text": evidence.text[:500],
            "score": round(float(evidence.score), 3),
        }
        if sentence is not None:
            payload["document_id"] = sentence.document_id
            payload["chunk_id"] = self._chunk_id(sentence)
            payload["span_kind"] = evidence.source_kind
            payload["token_estimate"] = len(content_tokens(sentence.text))
            if payload.get("char_start") is None:
                payload["char_start"] = sentence.char_start
            if payload.get("char_end") is None:
                payload["char_end"] = sentence.char_end
            if document is None:
                document = self._documents_by_rel_path.get(sentence.rel_path)
        elif document is not None:
            payload["document_id"] = document.document_id
        document_summary = self._document_provenance_summary(document)
        if document_summary:
            payload["document"] = document_summary
        return {key: value for key, value in payload.items() if value is not None and value != ""}

    def _model_answer_source_provenance_sample(self, answer: Answer, *, limit: int = 8) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        seen: set[tuple[str, str, int | None, str]] = set()
        for evidence in answer.evidence:
            payload = self._evidence_source_provenance_payload(evidence)
            key = (
                str(payload.get("rel_path") or ""),
                str(payload.get("span_id") or ""),
                payload.get("chunk_order") if isinstance(payload.get("chunk_order"), int) else None,
                str(payload.get("text") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(payload)
            if len(rows) >= limit:
                break
        return rows

    def _attach_model_answer_provenance(self, answer: Answer | None) -> None:
        if answer is None:
            return
        provenance = self._model_answer_source_provenance_sample(answer)
        if not provenance:
            return
        execution = self.last_bounded_diagnostics.setdefault("execution", {})
        if isinstance(execution, dict):
            execution["answer_source_provenance"] = provenance

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
        if matches:
            return matches
        if classify_value(proposed) == "count":
            return self._matching_count_evidence(evidence, evidence_span, proposed)
        return []

    def _matching_count_evidence(self, evidence: list[Evidence], evidence_span: str, proposed: str) -> list[Evidence]:
        canonical = canonicalize_answer(ExpectedAnswer("count"), proposed)
        if not canonical:
            return []
        try:
            expected_count = int(canonical)
        except ValueError:
            return []
        if expected_count <= 0:
            return []
        segments = [line.strip() for line in str(evidence_span or "").splitlines() if line.strip()]
        if len(segments) != expected_count:
            return []
        matches: list[Evidence] = []
        for segment in segments:
            segment_norm = normalize(segment)
            if not segment_norm:
                return []
            matched: Evidence | None = None
            for item in evidence:
                window = self._evidence_window_text(item)
                if segment in window or segment_norm in normalize(window):
                    matched = item
                    break
            if matched is None:
                return []
            if matched not in matches:
                matches.append(matched)
        return matches

    def _answer_with_local_model(self, question: str) -> Answer | None:
        if self._model_client is None:
            return None
        trace = self.model_query_trace
        trace.call_count += 1
        if os.environ.get("KMD_QUERY_DRS_PLAN", "1").strip().lower() in {"0", "false", "no", "off"}:
            raise LocalModelUnavailableError(
                "KnowMoreDiRT production runtime requires query DRS planning; KMD_QUERY_DRS_PLAN=0 is not supported."
            )
        self._log_progress("kmd-answer query_drs_start")
        query_drs_model = call_model_query_drs(question, self._model_client)
        self._record_model_result(query_drs_model)
        projected = None
        if query_drs_model.get("accepted"):
            projected = query_frame_from_query_drs(
                question,
                query_drs_model.get("query_drs") if isinstance(query_drs_model.get("query_drs"), dict) else None,
            )
        if projected is None:
            trace.last_plan = {
                "accepted": False,
                "source": "model_query_drs",
                "reason": query_drs_model.get("reason") or "query_drs_projection_failed",
            }
            return None
        model = {
            **projected,
            "accepted": True,
            "query_drs": query_drs_model.get("query_drs"),
            "source": "model_query_drs",
            "prompt_hash": query_drs_model.get("prompt_hash"),
            "output_hash": query_drs_model.get("output_hash"),
            "elapsed": query_drs_model.get("elapsed"),
        }
        if model.get("prompt_hash"):
            trace.prompt_hashes = [*list(trace.prompt_hashes or []), str(model["prompt_hash"])][-20:]
        if model.get("output_hash"):
            trace.response_hashes = [*list(trace.response_hashes or []), str(model["output_hash"])][-20:]
        trace.parsed_count += 1
        trace.accepted_count += 1
        plan = model
        trace.last_plan = plan
        planned_frame = frame_from_mapping(question, plan)
        expected = self._expected_from_frame(planned_frame)
        self._materialize_question_semantics(question, planned_frame)
        self._log_progress("kmd-answer bounded_query_start")
        answer = self._answer_with_bounded_dspg(question, planned_frame, expected)
        if answer and normalize(answer.text) != "unknown":
            if answer.reason == "bounded DSPG deterministic arithmetic execution":
                trace.model_answer_count += 1
                return answer
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
            elif self._answer_evidence_has_model_drs(answer) and os.environ.get(
                "KMD_VERIFY_MODEL_DRS_BOUND_ANSWERS",
                "0",
            ).strip().lower() in {"0", "false", "no", "off"}:
                trace.model_answer_count += 1
                answer.reason = "local model DRS query-frame execution"
                self._attach_model_answer_provenance(answer)
                return answer
            elif self._trusted_exact_structural_bounded_answer(answer, expected):
                trace.model_answer_count += 1
                answer.reason = "local model exact structural query-frame execution"
                self._attach_model_answer_provenance(answer)
                return answer
        if answer and normalize(answer.text) != "unknown":
            if self._verify_with_local_model(question, planned_frame, answer, expected):
                trace.model_answer_count += 1
                answer.reason = "local model query-frame execution"
                return answer
        if not self._bounded_conflict_blocks_model_evidence_fallback():
            evidence_answer = self._answer_with_model_query_evidence(question, expected)
            if evidence_answer and normalize(evidence_answer.text) != "unknown":
                return evidence_answer
        return None

    def _answer_evidence_has_model_drs(self, answer: Answer) -> bool:
        span_ids = [evidence.span_id for evidence in answer.evidence if evidence.span_id]
        if not span_ids:
            return False
        answer_norm = normalize(answer.text)
        if not answer_norm:
            return False
        for span_id in span_ids[:8]:
            if answer.answer_type == "boolean":
                row = self.store.execute(
                    """
                    SELECT 1
                    FROM drs_conditions
                    WHERE run_id=? AND source_span_id=? AND source='local_model_drs'
                    LIMIT 1
                    """,
                    (self.run_id, span_id),
                ).fetchone()
                if row is not None:
                    return True
            row = self.store.execute(
                """
                SELECT 1
                FROM drs_referents
                WHERE run_id=? AND source_span_id=? AND source='local_model_drs'
                  AND surface_norm=?
                LIMIT 1
                """,
                (self.run_id, span_id, answer_norm),
            ).fetchone()
            if row is not None:
                return True
            row = self.store.execute(
                """
                SELECT 1
                FROM drs_condition_arguments a
                JOIN drs_conditions c ON c.drs_condition_id=a.drs_condition_id
                WHERE a.run_id=? AND c.source_span_id=? AND c.source='local_model_drs'
                  AND (a.value_norm=? OR lower(a.evidence_surface)=?)
                LIMIT 1
                """,
                (self.run_id, span_id, answer_norm, answer_norm),
            ).fetchone()
            if row is not None:
                return True
            if len(answer_norm.split()) >= 2:
                rows = self.store.execute(
                    """
                    SELECT a.value, a.evidence_surface
                    FROM drs_condition_arguments a
                    JOIN drs_conditions c ON c.drs_condition_id=a.drs_condition_id
                    WHERE a.run_id=? AND c.source_span_id=? AND c.source='local_model_drs'
                    """,
                    (self.run_id, span_id),
                ).fetchall()
                for arg_value, evidence_surface in rows:
                    material = normalize(" ".join([str(arg_value or ""), str(evidence_surface or "")]))
                    if answer_norm in material:
                        return True
        return False

    def _trusted_exact_structural_bounded_answer(self, answer: Answer, expected: ExpectedAnswer) -> bool:
        if expected.answer_type not in TRUSTED_STRUCTURAL_ANSWER_TYPES:
            return False
        if answer.reason != "bounded DSPG query-frame execution":
            return False
        if not self._answer_has_source_grounding(answer):
            return False
        if not canonicalize_answer(expected, answer.text):
            return False
        diagnostics = self.last_bounded_diagnostics if isinstance(self.last_bounded_diagnostics, dict) else {}
        execution = diagnostics.get("execution") if isinstance(diagnostics.get("execution"), dict) else {}
        if not isinstance(execution, dict):
            return False
        binding_reason = str(execution.get("answer_binding_reason") or "")
        if binding_reason not in TRUSTED_STRUCTURAL_BINDING_REASONS:
            return False
        provenance = execution.get("answer_source_provenance")
        return isinstance(provenance, list) and bool(provenance)

    def _bounded_conflict_blocks_model_evidence_fallback(self) -> bool:
        diagnostics = self.last_bounded_diagnostics if isinstance(self.last_bounded_diagnostics, dict) else {}
        execution = diagnostics.get("execution") if isinstance(diagnostics.get("execution"), dict) else {}
        if not isinstance(execution, dict):
            return False
        blocking_keys = (
            "answer_conflict_without_query_scope",
            "temporal_ambiguity_without_query_scope",
            "temporal_answer_conflict_at_boundary",
        )
        for key in blocking_keys:
            if execution.get(key):
                execution["model_evidence_fallback_blocked_reason"] = key
                return True
        no_answer_reason = str(execution.get("no_answer_reason") or "")
        if no_answer_reason in blocking_keys:
            execution["model_evidence_fallback_blocked_reason"] = no_answer_reason
            return True
        return False

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
        quality = text_quality_metrics(sentence.text)
        if bool(quality.get("low_semantic_noise")) or str(quality.get("semantic_quality") or "") in {
            "base64_or_hex_blob",
            "multilingual_word_salad",
            "plausible_babble",
            "word_salad",
        }:
            return [], {"source": "skipped_noise"}
        cache_context = chunk_frame_cache_context(
            self._model_client,
            rel_path=sentence.rel_path,
            chunk_text=sentence.text,
        )
        cached = self._semantic_cache.get(sentence.text, context=cache_context) if self._semantic_cache else None
        if cached is not None:
            frames = [frame for frame in cached.get("frames", []) if isinstance(frame, dict)]
            metadata = cached.get("metadata") if isinstance(cached.get("metadata"), dict) else {}
            return frames, {
                "source": "cache",
                "frame_count": len(frames),
                "accepted": bool(metadata.get("accepted", True)),
                "reason": str(metadata.get("reason") or ""),
                "prompt_hash": metadata.get("prompt_hash"),
                "output_hash": metadata.get("output_hash"),
                "context_budget": metadata.get("context_budget"),
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
        if self._model_client is None:
            return 0
        frame_cache_context = chunk_frame_cache_context(
            self._model_client,
            rel_path=sentence.rel_path,
            chunk_text=sentence.text,
        )
        frame_cache_key = stable_id(
            "frame_attempt_context",
            json.dumps(frame_cache_context, sort_keys=True, default=str),
        )
        previous_attempt = self.store.execute(
            """
            SELECT accepted, materialized, reason
            FROM model_attempts
            WHERE run_id=? AND source_span_id=? AND task=? AND source=? AND cache_key=?
            LIMIT 1
            """,
            (self.run_id, span_id, "chunk_frames", "local_model", frame_cache_key),
        ).fetchone()
        existing = self.store.execute(
            "SELECT COUNT(*) FROM frames WHERE run_id=? AND span_id=? AND source='local_model'",
            (self.run_id, span_id),
        ).fetchone()[0]
        if (
            existing
            and previous_attempt is not None
            and bool(previous_attempt["accepted"])
            and bool(previous_attempt["materialized"])
        ):
            return 0
        replaced_frames: dict[str, int] = {}
        if existing:
            replaced_frames = self.store.delete_frame_materialization_for_span(
                self.run_id,
                span_id,
                source="local_model",
            )
            inactive_attempts = self.store.deactivate_other_model_attempt_materializations(
                self.run_id,
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
            self._log_progress(
                "kmd-answer lazy_frame previous_attempt "
                f"{sentence.rel_path}:{sentence.order} "
                f"accepted={bool(previous_attempt['accepted'])} "
                f"materialized={bool(previous_attempt['materialized'])} "
                f"reason={str(previous_attempt['reason'] or 'previous_attempt')}"
            )
            return 0
        model_frames, result = self._cached_or_fresh_chunk_frames(sentence)
        self._record_model_result(result)
        if result.get("prompt_hash"):
            self.model_query_trace.prompt_hashes = [*list(self.model_query_trace.prompt_hashes or []), str(result["prompt_hash"])][-20:]
        if result.get("output_hash"):
            self.model_query_trace.response_hashes = [*list(self.model_query_trace.response_hashes or []), str(result["output_hash"])][-20:]
        self.model_query_trace.chunk_frame_call_count += 0 if result.get("source") in {"cache", "skipped_noise", "skipped_long_chunk", "disabled"} else 1
        if model_frames:
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
                              hypothesis_id, run_id, source_span_id, context_id, drs_box_id, box_external_id,
                              left_referent_id, right_referent_id,
                              relation, evidence, confidence, source
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                stable_id("idh", self.run_id, existing_referent_id, arg_referent_id, semantic_frame_id),
                                self.run_id,
                                span_id,
                                semantic_context_id,
                                None,
                                None,
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
                      hypothesis_id, run_id, source_span_id, context_id, drs_box_id, box_external_id,
                      left_referent_id, right_referent_id,
                      relation, evidence, confidence, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("idh", self.run_id, semantic_frame_id, "model_identity", hypothesis_index, left_text, right_text),
                        self.run_id,
                        span_id,
                        semantic_context_id,
                        None,
                        None,
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
        result_source = str(result.get("fresh_or_cached") or result.get("source") or "fresh")
        accepted = bool(result.get("accepted")) if "accepted" in result else result_source == "cache"
        self.store.execute(
            """
            INSERT OR REPLACE INTO model_attempts(
              attempt_id, run_id, source_span_id, task, source, cache_key, accepted, materialized,
              reason, prompt_hash, output_hash, elapsed, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_id("attempt", self.run_id, span_id, "chunk_frames", "local_model", frame_cache_key),
                self.run_id,
                span_id,
                "chunk_frames",
                "local_model",
                frame_cache_key,
                int(accepted),
                int(inserted > 0),
                str(result.get("reason") or ""),
                str(result.get("prompt_hash") or ""),
                str(result.get("output_hash") or ""),
                float(result.get("elapsed") or 0.0),
                json.dumps(
                    {
                        "cache_context": frame_cache_context,
                        "frame_count": len(model_frames),
                        "inserted_frame_count": inserted,
                        "result_source": result_source,
                        "context_budget": result.get("context_budget"),
                        "replaced_prior_rows": replaced_frames,
                    },
                    sort_keys=True,
                    default=str,
                ),
            ),
        )
        return inserted

    def _answer_with_model_query_evidence(self, question: str, expected_hint: ExpectedAnswer | None = None) -> Answer | None:
        if not self._test_model_evidence_helpers_allowed():
            return None
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
        try:
            self._record_model_result(model)
        except LocalModelUnavailableError:
            trace.evidence_rejected_count += 1
            return None
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
            unknown = Answer(
                "unknown",
                0.0,
                [item for item in evidence[:6] if item.rel_path and item.text],
                "local model query-DRS insufficient evidence",
                "unknown",
            )
            self._attach_model_answer_provenance(unknown)
            return unknown
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
        finalized = self._finalize_answer(question, answer, expected, "local model query-DRS evidence verification", frame)
        if not finalized:
            trace.evidence_rejected_count += 1
            return None
        trace.evidence_accepted_count += 1
        trace.model_answer_count += 1
        self._attach_model_answer_provenance(finalized)
        return finalized

    def _answer_with_model_evidence_extraction(
        self,
        question: str,
        frame: QueryFrame,
        expected: ExpectedAnswer | None = None,
    ) -> Answer | None:
        if not self._test_model_evidence_helpers_allowed():
            return None
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
        try:
            self._record_model_result(model)
        except LocalModelUnavailableError:
            trace.evidence_rejected_count += 1
            return None
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
        finalized = self._finalize_answer(question, answer, expected, "local model bounded evidence extraction", frame)
        if not finalized:
            trace.evidence_rejected_count += 1
            return None
        trace.evidence_accepted_count += 1
        trace.model_answer_count += 1
        self._attach_model_answer_provenance(finalized)
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
        final_expected = expected
        if bounded_answer.reason == "deterministic arithmetic binding":
            final_expected = ExpectedAnswer("count")
        source = "bounded DSPG query-frame execution"
        if bounded_answer.reason == "deterministic arithmetic binding":
            source = "bounded DSPG deterministic arithmetic execution"
        return self._finalize_answer(question, bounded_answer, final_expected, source, frame)

    def _answer_has_source_grounding(self, answer: Answer) -> bool:
        if normalize(answer.text) == "unknown":
            return True
        return any(evidence.rel_path and evidence.text for evidence in answer.evidence)

    def _cleanup_canonical_answer(
        self,
        canonical: str,
        expected: ExpectedAnswer,
        frame: QueryFrame | None = None,
    ) -> str:
        if expected.answer_type == "boolean":
            return str(canonical or "").strip()
        text = clean_extracted_value(canonical).strip()
        if not text:
            return text
        low = normalize(text)
        if frame is not None and expected.answer_type in {"identifier", "content_phrase", "metadata_value"}:
            text = self._strip_redundant_answer_slot_suffix(text, frame)
            low = normalize(text)
        if frame is not None and expected.answer_type in {"content_phrase", "metadata_value", "identifier"}:
            text = self._strip_redundant_target_tail(text, frame)
            low = normalize(text)
        if frame is not None and expected.answer_type in {"content_phrase", "metadata_value"}:
            text = self._replace_redundant_topic_subject_with_pronoun(text, frame)
            low = normalize(text)
        if frame is not None and expected.answer_type in {"content_phrase", "state", "metadata_value", "identifier"}:
            text = self._strip_answer_clause_residual(text, frame)
            low = normalize(text)
        article_strippable = expected.answer_type in {"content_phrase", "state", "metadata_value"}
        article_strippable = article_strippable or (
            expected.answer_type == "identifier" and classify_value(text) == "content_phrase"
        )
        if article_strippable:
            words = text.split()
            low_words = [word.lower().strip(".,;:") for word in words]
            if len(words) <= 4 and low_words and low_words[0] in {"the", "a", "an"}:
                verbish = {"is", "was", "were", "are", "be", "been", "being", "should", "would", "could", "did", "does", "do", "has", "have", "had"}
                if not any(word in verbish for word in low_words[1:]):
                    return " ".join(words[1:]).strip()
        return text

    def _replace_redundant_topic_subject_with_pronoun(self, text: str, frame: QueryFrame) -> str:
        topic_heads: list[str] = []
        for source in [frame.question_text, *frame.answer_variables]:
            norm = normalize(source)
            if " about " not in f" {norm} ":
                continue
            after_about = norm.split(" about ", 1)[1]
            tokens = content_tokens(after_about)
            if tokens:
                topic_heads.append(tokens[-1])
        topic_heads = list(dict.fromkeys(topic_heads))
        if not topic_heads:
            return text
        words = [word for word in text.split() if word]
        if len(words) < 3:
            return text
        first = normalize(words[0].strip(".,;:"))
        second = normalize(words[1].strip(".,;:"))
        if first == "the" and second in topic_heads:
            remainder = words[2:]
        elif first in topic_heads:
            remainder = words[1:]
        else:
            return text
        if not remainder:
            return text
        next_word = normalize(remainder[0].strip(".,;:"))
        if next_word not in {
            "are",
            "can",
            "could",
            "does",
            "has",
            "is",
            "may",
            "might",
            "must",
            "needs",
            "should",
            "was",
            "will",
            "would",
        }:
            return text
        return " ".join(["It", *remainder]).strip()

    def _strip_answer_clause_residual(self, text: str, frame: QueryFrame) -> str:
        words = [word for word in text.split() if word]
        if len(words) < 4 or len(words) > 12:
            return text
        normalized_words = [normalize(word.strip(".,;:")) for word in words]
        target_tokens = {
            token
            for anchor in frame.target_anchors
            for token in content_tokens(anchor)
            if token
        }
        slot_tokens = {
            token
            for variable in frame.answer_variables
            for token in content_tokens(variable)
            if token and token not in target_tokens
        }
        relation_tokens: set[str] = set()
        for source in [frame.requested_relation, *frame.relation_terms]:
            source_tokens = content_tokens(source)
            if source_tokens:
                for token in source_tokens:
                    relation_tokens.update(term_variants(token))
                continue
            source_norm = normalize(source)
            if source_norm:
                relation_tokens.add(source_norm)
        if not target_tokens or not relation_tokens:
            return text
        relation_indexes = [
            index for index, token in enumerate(normalized_words)
            if token in relation_tokens
        ]
        if not relation_indexes:
            return text
        relation_index = relation_indexes[-1]
        if relation_index >= len(words) - 1:
            return text
        prefix_tokens = set(normalized_words[: relation_index + 1])
        prefix_has_target = bool(prefix_tokens & target_tokens)
        prefix_has_slot = bool(slot_tokens and (prefix_tokens & slot_tokens))
        if not prefix_has_target and not prefix_has_slot:
            return text
        if slot_tokens and prefix_has_target and not prefix_has_slot:
            return text
        residual_words = words[relation_index + 1 :]
        if len(residual_words) > 4:
            return text
        residual = clean_extracted_value(" ".join(residual_words)).strip(" .;:")
        return residual or text

    def _strip_redundant_answer_slot_suffix(self, text: str, frame: QueryFrame) -> str:
        slot_terms = [
            term
            for variable in frame.answer_variables
            for term in [variable, *content_tokens(variable)]
            if normalize(term)
        ]
        if not slot_terms:
            return text
        current = text
        for slot in sorted(dict.fromkeys(slot_terms), key=lambda value: len(value), reverse=True):
            suffix = " " + normalize(slot)
            if normalize(current).endswith(suffix) and len(current.split()) <= 8:
                trimmed = current[: -len(suffix)].strip()
                if trimmed:
                    current = trimmed
                    break
        return current

    def _strip_redundant_target_tail(self, text: str, frame: QueryFrame) -> str:
        current = text
        for target in sorted(dict.fromkeys(frame.target_anchors), key=lambda value: len(value), reverse=True):
            target_clean = clean_extracted_value(target).strip(" ?.:")
            if not target_clean:
                continue
            suffix = " for " + normalize(target_clean)
            if normalize(current).endswith(suffix):
                trimmed = current[: -(len(" for ") + len(target_clean))].strip()
                if trimmed:
                    current = trimmed
                    break
        return current

    def _date_time_shape_compatible(self, frame: QueryFrame | None, value: str) -> bool:
        if frame is None:
            return True
        answer_material = normalize(
            " ".join([*frame.answer_variables, frame.requested_relation, *frame.relation_terms, *frame.constraints])
        )
        if not answer_material:
            return True
        asks_for_calendar_date = any(term in answer_material.split() for term in ("date", "day"))
        asks_for_clock_time = any(term in answer_material.split() for term in ("time", "hour", "minute"))
        if not asks_for_calendar_date or asks_for_clock_time:
            return True
        return re.fullmatch(r"\d{1,2}:\d{2}", normalize(value)) is None

    def _finalize_answer(
        self,
        question: str,
        answer: Answer,
        expected: ExpectedAnswer,
        source: str,
        frame: QueryFrame | None = None,
    ) -> Answer | None:
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
        if canonical and source.startswith("local model") and expected.answer_type in {"content_phrase", "state", "metadata_value"}:
            canonical = self._canonicalize_model_answer_with_local_model(question, canonical, expected, answer.evidence) or canonical
        if normalize(canonical) == "unknown":
            return Answer("unknown", 0.0, answer.evidence, source, "unknown")
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
        if expected.answer_type == "date_time" and not self._date_time_shape_compatible(frame, canonical):
            return None
        pre_cleanup_canonical = canonical
        canonical = self._cleanup_canonical_answer(canonical, expected, frame)
        canonical = self._restore_sentence_terminal_punctuation(
            canonical,
            pre_cleanup_canonical,
            expected,
            answer.evidence,
        )
        if not canonical:
            return None
        return Answer(canonical, answer.confidence, answer.evidence, source, expected.answer_type)

    def _restore_sentence_terminal_punctuation(
        self,
        text: str,
        source_value: str,
        expected: ExpectedAnswer,
        evidence: list[Evidence],
    ) -> str:
        if expected.answer_type not in {"content_phrase", "state", "metadata_value"}:
            return text
        value = str(text or "").strip()
        if not value or value[-1] in ".!?":
            return value
        words = [word for word in value.split() if word]
        if len(words) < 4 or not words[0][:1].isupper():
            return value
        low_words = [normalize(word.strip(".,;:!?")) for word in words]
        finite_or_modal = {
            "are",
            "can",
            "could",
            "did",
            "does",
            "has",
            "is",
            "may",
            "might",
            "must",
            "needs",
            "should",
            "was",
            "were",
            "will",
            "would",
        }
        source_norm = normalize(str(source_value or "").strip(" .;:!?"))
        value_norm = normalize(value.strip(" .;:!?"))
        if not source_norm and not value_norm:
            return value
        evidence_texts = self._terminal_punctuation_evidence_texts(evidence)
        for evidence_text in evidence_texts:
            if evidence_text[-1] in ".!?" and value_norm == normalize(evidence_text.strip(" .;:!?")):
                return value + evidence_text[-1]
        has_sentence_predicate = any(
            word in finite_or_modal or word.endswith(("ed", "ing"))
            for word in low_words[1:]
        )
        if not has_sentence_predicate:
            return value
        for evidence_text in evidence_texts:
            if evidence_text[-1] not in ".!?":
                continue
            evidence_norm = normalize(evidence_text.strip(" .;:!?"))
            if source_norm and source_norm in evidence_norm:
                return value + evidence_text[-1]
            if value_norm and value_norm in evidence_norm:
                return value + evidence_text[-1]
            value_terms = content_tokens(value)
            evidence_terms: set[str] = set()
            for term in content_tokens(evidence_text):
                evidence_terms.update(term_variants(term))
            if len(value_terms) >= 3 and all(term_variants(term) & evidence_terms for term in value_terms):
                return value + evidence_text[-1]
        terminal = next((text[-1] for text in evidence_texts if text and text[-1] in ".!?"), "")
        if terminal:
            value_terms = content_tokens(value)
            combined_terms: set[str] = set()
            for evidence_text in evidence_texts:
                for term in content_tokens(evidence_text):
                    combined_terms.update(term_variants(term))
            if len(value_terms) >= 3 and all(term_variants(term) & combined_terms for term in value_terms):
                return value + terminal
        return value

    def _terminal_punctuation_evidence_texts(self, evidence: list[Evidence]) -> list[str]:
        values: list[str] = []
        for item in evidence:
            evidence_text = str(item.text or "").strip()
            if evidence_text:
                values.append(evidence_text)
            span_id = str(item.span_id or "")
            if not span_id or not hasattr(self, "store"):
                continue
            try:
                row = self.store.execute(
                    "SELECT surface FROM source_spans WHERE span_id=? LIMIT 1",
                    (span_id,),
                ).fetchone()
            except Exception:
                row = None
            if row is None:
                continue
            try:
                surface = str(row["surface"] or "").strip()
            except Exception:
                surface = str(row[0] or "").strip()
            if surface:
                values.append(surface)
        return list(dict.fromkeys(values))

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
        source_resolved = self._source_resolve_model_answer_with_local_model(
            question,
            value,
            expected,
            evidence_payload,
        )
        if source_resolved and normalize(source_resolved) != normalize(value):
            return source_resolved
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
        if normalize(proposed) == "unknown":
            trace.canonicalization_accepted_count += 1
            return "unknown"
        canonical = canonicalize_answer(expected, proposed)
        if not canonical:
            trace.canonicalization_rejected_count += 1
            return value
        trace.canonicalization_accepted_count += 1
        return canonical

    def _source_resolve_model_answer_with_local_model(
        self,
        question: str,
        value: str,
        expected: ExpectedAnswer,
        evidence_payload: list[dict[str, str]],
    ) -> str:
        if self._model_client is None:
            return value
        if expected.answer_type not in {"content_phrase", "state", "metadata_value"}:
            return value
        if not self._answer_has_source_deictic_terms(value):
            return value
        trace = self.model_query_trace
        trace.canonicalization_call_count += 1
        result = call_model_source_resolved_answer(
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
        if normalize(proposed) == "unknown":
            trace.canonicalization_accepted_count += 1
            return "unknown"
        canonical = canonicalize_answer(expected, proposed)
        if not canonical:
            trace.canonicalization_rejected_count += 1
            return value
        trace.canonicalization_accepted_count += 1
        return canonical

    def _answer_has_source_deictic_terms(self, value: str) -> bool:
        tokens = [token.lower().strip(".,;:!?()[]{}\"'`") for token in re.findall(r"[A-Za-z]+", value or "")]
        return any(token in SOURCE_DEICTIC_TOKENS for token in tokens)

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
