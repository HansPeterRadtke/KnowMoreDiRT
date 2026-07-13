"""Staged model-owned semantics over generic deterministic tools."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .catalog import SourceCatalog
from .model import ModelError, StrictModelClient
from .models import Answer, ToolResult
from .schemas import (
    dataset_profile_schema,
    event_fact_verdict_schema,
    grounded_answer_schema,
    numeric_value_repair_schema,
    query_program_schema,
    semantic_contract_schema,
    tool_extraction_schema,
)
from .tools import ToolExecutor, expand_step

_TOOL_DESCRIPTIONS = {
    "sample_records": "Return a bounded sample from one collection.",
    "search_records": "Search a collection or prior record set with literal model-selected terms; all, any, and phrase modes are available.",
    "expand_source_context": "Replace matched fragments with coherent preferred records from the same source files.",
    "filter_records": "Filter prior records using explicit field paths and comparisons.",
    "project_values": "Project values from explicit structured field paths while retaining provenance.",
    "extract_values": (
        "Extract values from prior records using a model-selected generic extractor: field, after_label, "
        "after_phrase, before_phrase, between_phrases, regex, url, identifier, date_time, number, or "
        "event_series. occurrence can select first, last, all, latest_by_time, or earliest_by_time."
    ),
    "model_extract": (
        "Apply a bounded strict model extraction to prior evidence records under the immutable semantic contract. "
        "Use for relations, negation, temporal interpretation, epistemic scope, or text patterns that deterministic "
        "extractors cannot express safely. For where or location requests, an explicitly associated URL, URI, path, "
        "address, directory, shelf, room, or similar locator is a valid answer value."
    ),
    "join_records": "Join two prior record sets using explicit field paths.",
    "union_values": "Combine values from prior steps.",
    "intersect_values": "Intersect values from prior steps.",
    "sort_records": "Sort records by an explicit field path.",
    "aggregate_values": "Count, deduplicate, or calculate an aggregate over prior results.",
    "calculate": "Execute explicit model-selected arithmetic over supplied numbers or prior numeric values.",
}


class ProgramValidationError(ValueError):
    pass


class KnowMoreDiRTEngine:
    def __init__(self, folder_path: str | Path, model: Any | None = None):
        self.folder_path = Path(folder_path).resolve()
        self.catalog = SourceCatalog(self.folder_path)
        self.model = model or StrictModelClient()
        self.executor = ToolExecutor(self.catalog)
        self.last_answer: Answer | None = None
        self.model_query_trace: dict[str, Any] = {}
        self._dataset_profile: dict[str, Any] | None = None

    def answer(self, question: str) -> Answer:
        question = str(question or "").strip()
        if not question:
            answer = Answer("unknown", diagnostics={"reason": "empty_question"})
            self.last_answer = answer
            return answer
        try:
            profile = self._profile_dataset()
            contract = self._parse_semantics(question)
            program = self._compile_program(profile, contract)
            results = self._execute_program(contract, program)
            if self._needs_execution_repair(program, results):
                program = self._repair_program_after_execution(profile, contract, program, results)
                results = self._execute_program(contract, program)
            if self._needs_execution_repair(program, results):
                fallback = self._fallback_model_extract_program(program, results)
                if fallback is not None:
                    program = fallback
                    results = self._execute_program(contract, program)
            if self._needs_list_cardinality_fallback(contract, program, results):
                fallback = self._fallback_model_extract_program(program, results)
                if fallback is not None:
                    program = fallback
                    results = self._execute_program(contract, program)
            direct = self._direct_structural_answer(contract, program, results)
            if direct is not None:
                answer = direct
            else:
                grounded = self._ground(contract, program, results)
                answer = self._answer_from_grounded(grounded, results)
        except (ModelError, ProgramValidationError, ValueError, KeyError) as exc:
            answer = Answer("unknown", diagnostics={"reason": type(exc).__name__, "error": str(exc)})
        if "contract" in locals():
            answer = self._canonicalize_final_answer(contract, answer)
        self.last_answer = answer
        return answer

    @staticmethod
    def _text_answer_denotes_absent_requested_decision(
        contract: dict[str, Any],
        text: str,
    ) -> bool:
        slot_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                str(contract.get("answer_slot", "")).lower().replace("_", " "),
            )
        )
        if not slot_tokens.intersection({"decision", "choice", "determination", "resolution"}):
            return False
        normalized = re.sub(r"\s+", " ", str(text).strip().lower())
        return bool(
            re.search(
                r"\b(?:no|without)\s+(?:final\s+)?(?:decision|choice|determination|resolution)\b"
                r"|\b(?:decision|choice|determination|resolution)\s+(?:was\s+)?not\s+(?:made|reached|recorded)\b"
                r"|\b(?:undecided|decision pending|pending decision)\b",
                normalized,
            )
        )

    @classmethod
    def _canonicalize_final_answer(
        cls,
        contract: dict[str, Any],
        answer: Answer,
    ) -> Answer:
        text = str(answer.text).strip()
        if (
            not text
            or text.lower() == "unknown"
            or contract.get("answer_shape") != "text"
            or text.lower().startswith(("yes;", "no;"))
            or ";" in text
        ):
            return answer
        if cls._text_answer_denotes_absent_requested_decision(contract, text):
            return Answer(
                "unknown",
                evidence=answer.evidence,
                diagnostics={
                    **answer.diagnostics,
                    "absence_canonicalized": True,
                    "reason": "requested decision value does not exist",
                },
            )
        canonical = cls._canonicalize_extracted_value(contract, text)
        if canonical == text or not canonical:
            return answer
        return Answer(
            canonical,
            evidence=answer.evidence,
            diagnostics={**answer.diagnostics, "surface_canonicalized": True},
        )

    def _execute_program(
        self,
        contract: dict[str, Any],
        program: dict[str, Any],
    ) -> dict[int, ToolResult]:
        return self.executor.execute(
            program["steps"],
            semantic_extractor=lambda step_id, step, results: self._model_extract(
                contract,
                step_id,
                step,
                results,
            ),
        )

    @staticmethod
    def _enforce_extraction_status_invariant(extraction: dict[str, Any]) -> dict[str, Any]:
        has_values = any(str(value).strip() for value in extraction.get("values", []))
        return {
            **extraction,
            "status": "extracted" if has_values else "unknown",
        }

    @classmethod
    def _needs_mixed_epistemic_correction_repair(
        cls,
        contract: dict[str, Any],
        extraction: dict[str, Any],
        records: list[Any],
    ) -> bool:
        values = {str(value).strip().lower() for value in extraction.get("values", [])}
        return (
            contract.get("answer_shape") == "boolean"
            and extraction.get("status") == "extracted"
            and bool(values.intersection({"no", "false"}))
            and cls._mixed_epistemic_evidence(records)
            and (
                extraction.get("evidence_relation") in {"nonactual_content", "state_only", "unknown"}
                or cls._reason_explicit_false(str(extraction.get("reason", "")))
            )
        )

    @classmethod
    def _evidence_has_explicit_entity_ambiguity(cls, records: list[Any]) -> bool:
        ambiguity = re.compile(
            r"(?i)\b(?:"
            r"does\s+not\s+say\s+which|doesn't\s+say\s+which|not\s+clear\s+which|"
            r"unclear\s+which|ambiguous|cannot\s+determine|can't\s+determine|"
            r"until\s+[^.\n]{0,80}\s+clarified|keep\s+[^.\n]{0,100}\s+separate|"
            r"do\s+not\s+merge|don't\s+merge|not\s+enough\s+information"
            r")\b"
        )
        for record in records:
            text = (
                str(getattr(record, "text", ""))
                + "\n"
                + json.dumps(getattr(record, "data", {}), ensure_ascii=False, default=str)
            )
            if ambiguity.search(text):
                return True
        return False

    @classmethod
    def _needs_entity_ambiguity_repair(
        cls,
        contract: dict[str, Any],
        extraction: dict[str, Any],
        records: list[Any],
    ) -> bool:
        return (
            contract.get("semantic_kind") == "entity_attribute"
            and extraction.get("status") == "extracted"
            and any(str(value).strip() for value in extraction.get("values", []))
            and cls._evidence_has_explicit_entity_ambiguity(records)
        )

    @staticmethod
    def _needs_extraction_consistency_repair(extraction: dict[str, Any]) -> bool:
        has_values = any(str(value).strip() for value in extraction.get("values", []))
        return (
            extraction.get("status") == "unknown" and has_values
        ) or (
            extraction.get("status") == "extracted" and not has_values
        )

    @classmethod
    def _has_explicit_alternative_behavior(cls, records: list[Any]) -> bool:
        for record in records:
            text = (
                str(getattr(record, "text", ""))
                + "\n"
                + json.dumps(getattr(record, "data", {}), ensure_ascii=False, default=str)
            ).lower()
            has_explicit_negation = bool(
                re.search(r"\b(?:does|do|did|is|are|was|were|will|would|should|can|could)\s+not\b", text)
                or re.search(r"\bnot\s+(?:delete|remove|erase|drop|discard|send|store|retain|flag|route|mark|queue)\b", text)
            )
            has_separate_behavior = bool(
                re.search(
                    r"\b(?:flags?|routes?|queues?|retains?|preserves?|marks?|stores?|keeps?|returns?|sends?)\b",
                    text,
                )
            )
            if has_explicit_negation and has_separate_behavior:
                return True
        return False

    @classmethod
    def _needs_negative_alternative_repair(
        cls,
        contract: dict[str, Any],
        extraction: dict[str, Any],
        records: list[Any],
    ) -> bool:
        values = {str(value).strip().lower() for value in extraction.get("values", [])}
        return (
            contract.get("semantic_kind") == "event_fact"
            and contract.get("answer_shape") == "boolean"
            and extraction.get("status") == "extracted"
            and bool(values.intersection({"no", "false"}))
            and not cls._mixed_epistemic_evidence(records)
            and not cls._contract_asks_proof_status(contract)
            and cls._has_explicit_alternative_behavior(records)
        )

    @staticmethod
    def _needs_event_fact_repair(
        contract: dict[str, Any],
        extraction: dict[str, Any],
    ) -> bool:
        return (
            extraction.get("status") in {"extracted", "unknown"}
            and contract.get("semantic_kind") == "event_fact"
        )

    @classmethod
    def _needs_negative_correction_repair(
        cls,
        contract: dict[str, Any],
        extraction: dict[str, Any],
        records: list[Any],
    ) -> bool:
        values = {str(value).strip().lower() for value in extraction.get("values", [])}
        return (
            contract.get("answer_shape") == "boolean"
            and extraction.get("status") == "extracted"
            and bool(values.intersection({"no", "false"}))
            and extraction.get("evidence_relation") == "direct_contradiction"
            and not cls._mixed_epistemic_evidence(records)
            and not cls._contract_asks_proof_status(contract)
        )

    @staticmethod
    def _extraction_from_event_fact_verdict(
        contract: dict[str, Any],
        verdict: dict[str, Any],
    ) -> dict[str, Any]:
        decision = verdict["verdict"]
        evidence_ids = list(verdict.get("evidence_record_ids", []))
        correction = str(verdict.get("correction_clause", "")).strip()
        reason = str(verdict.get("reason", "")).strip()
        if decision == "supports":
            return {
                "contract_id": contract["contract_id"],
                "status": "extracted",
                "values": ["yes"],
                "answer_shape": contract["answer_shape"],
                "evidence_record_ids": evidence_ids,
                "evidence_relation": "direct_support",
                "reason": reason,
            }
        if decision == "contradicts":
            return {
                "contract_id": contract["contract_id"],
                "status": "extracted",
                "values": ["no"],
                "answer_shape": contract["answer_shape"],
                "evidence_record_ids": evidence_ids,
                "evidence_relation": "direct_contradiction",
                "reason": correction or reason,
            }
        return {
            "contract_id": contract["contract_id"],
            "status": "unknown",
            "values": [],
            "answer_shape": contract["answer_shape"],
            "evidence_record_ids": evidence_ids,
            "evidence_relation": "absence",
            "reason": reason,
        }

    @staticmethod
    def _needs_classification_repair(
        contract: dict[str, Any],
        extraction: dict[str, Any],
    ) -> bool:
        return (
            extraction.get("status") == "extracted"
            and contract.get("semantic_kind") == "source_classification"
        )

    @staticmethod
    def _needs_discourse_repair(
        contract: dict[str, Any],
        extraction: dict[str, Any],
    ) -> bool:
        if extraction.get("status") != "extracted":
            return False
        mode = contract.get("epistemic_mode")
        if mode not in {"quoted", "reported"}:
            return False
        values = " ".join(str(value) for value in extraction.get("values", []))
        if mode == "quoted":
            return bool(re.search(r"\b(?:I|me|my|mine|we|our|ours)\b", values, flags=re.IGNORECASE))
        slot = str(contract.get("answer_slot", "")).lower()
        return any(marker in slot for marker in ("belief", "report", "claim", "message", "content", "statement"))

    @staticmethod
    def _record_field_keys(record: Any) -> set[str]:
        keys: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    normalized = " ".join(
                        re.findall(r"[a-z0-9]+", str(key).lower().replace("_", " "))
                    )
                    if normalized:
                        keys.add(normalized)
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(getattr(record, "data", {}))
        return keys

    @classmethod
    def _record_field_key_tokens(cls, record: Any) -> set[str]:
        return {
            token
            for key in cls._record_field_keys(record)
            for token in key.split()
        }

    @classmethod
    def _prefer_structured_answer_slot_records(
        cls,
        contract: dict[str, Any],
        records: list[Any],
    ) -> list[Any]:
        slot_tokens = {
            token
            for token in re.findall(
                r"[a-z0-9]+",
                str(contract.get("answer_slot", "")).lower().replace("_", " "),
            )
            if token not in {"value", "answer", "text", "result", "content", "boolean"}
        }
        if not slot_tokens:
            return records
        slot_phrase = " ".join(sorted(slot_tokens))
        exact = [
            record
            for record in records
            if any(
                " ".join(sorted(key.split())) == slot_phrase
                for key in cls._record_field_keys(record)
            )
        ]
        if exact:
            return exact
        matched = [
            record
            for record in records
            if slot_tokens.issubset(cls._record_field_key_tokens(record))
        ]
        return matched or records

    @classmethod
    def _contract_target_tokens(cls, contract: dict[str, Any]) -> set[str]:
        target_tokens = {
            token
            for phrase in contract.get("target_phrases", [])
            for token in cls._content_tokens(str(phrase))
        }
        relation_tokens = {
            token
            for phrase in contract.get("relation_phrases", [])
            for token in cls._content_tokens(str(phrase))
        }
        slot_tokens = cls._content_tokens(
            str(contract.get("answer_slot", "")).replace("_", " ")
        )
        reduced = target_tokens - relation_tokens - slot_tokens
        return reduced or target_tokens

    @classmethod
    def _localized_record_view(
        cls,
        record: Any,
        contract: dict[str, Any],
        max_chars: int = 1400,
    ) -> dict[str, Any]:
        view = record.model_view(max_chars)
        if contract.get("answer_shape") == "list":
            return view
        if (
            contract.get("answer_shape") == "number"
            and set(
                re.findall(
                    r"[a-z0-9]+",
                    str(contract.get("answer_slot", "")).lower().replace("_", " "),
                )
            ).intersection({"count", "number", "total", "quantity"})
        ):
            return view
        if (
            contract.get("answer_shape") == "boolean"
            and cls._target_mixed_epistemic_evidence(record, contract)
        ):
            return view
        data = getattr(record, "data", {})
        raw_text = str(getattr(record, "text", "") or "")
        structured_text_needs_scope_localization = bool(
            isinstance(data, dict)
            and isinstance(data.get("label_records"), list)
            and "\n" in raw_text
        )
        slot_tokens = {
            token
            for token in re.findall(
                r"[a-z0-9]+",
                str(contract.get("answer_slot", "")).lower().replace("_", " "),
            )
            if token not in {"value", "answer", "text", "result", "content", "boolean"}
        }
        target_tokens = cls._contract_target_tokens(contract)
        document_field_slot = bool(
            slot_tokens.intersection(
                {"summary", "explanation", "description", "note", "message", "statement", "content"}
            )
        )
        if (
            isinstance(data, dict)
            and slot_tokens
            and (not structured_text_needs_scope_localization or document_field_slot)
        ):
            answer_fields: dict[str, Any] = {}
            context_fields: dict[str, Any] = {}
            temporal_slot = bool(
                slot_tokens.intersection({"date", "time", "timestamp", "datetime", "when"})
            )
            temporal_role_tokens = set()
            for phrase in [
                *contract.get("target_phrases", []),
                *contract.get("scope_phrases", []),
                *contract.get("relation_phrases", []),
            ]:
                temporal_role_tokens.update(cls._content_tokens(str(phrase)))
            temporal_role_tokens.difference_update(
                {"date", "time", "timestamp", "datetime", "measurement"}
            )
            temporal_role_fields: list[tuple[int, str, Any]] = []

            def visit_structured(value: Any, path: str = "") -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key in {"text", "source"}:
                            continue
                        child_path = f"{path}.{key}" if path else str(key)
                        visit_structured(child, child_path)
                    return
                if isinstance(value, list):
                    for index, child in enumerate(value[:100]):
                        visit_structured(child, f"{path}[{index}]")
                    return
                if not path:
                    return
                key_tokens = set(
                    re.findall(r"[a-z0-9]+", path.lower().replace("_", " "))
                )
                value_tokens = cls._content_tokens(str(value))
                if temporal_slot and re.search(
                    r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b",
                    str(value),
                ):
                    role_score = len(key_tokens.intersection(temporal_role_tokens))
                    if role_score:
                        temporal_role_fields.append((role_score, path, value))
                if key_tokens.intersection(slot_tokens):
                    answer_fields[path] = value
                elif target_tokens and len(value_tokens.intersection(target_tokens)) >= max(
                    1, (2 * len(target_tokens) + 2) // 3
                ):
                    context_fields[path] = value

            visit_structured(data)
            if temporal_role_fields:
                best_score = max(score for score, _, _ in temporal_role_fields)
                best_temporal = [
                    (path, value)
                    for score, path, value in temporal_role_fields
                    if score == best_score
                ]
                if len(best_temporal) == 1:
                    path, value = best_temporal[0]
                    selected = {path: value}
                    localized = f"{path}: {value}"
                    source = data.get("source", {})
                    view["text"] = localized[:max_chars]
                    view["excerpt"] = localized[:max_chars]
                    view["data"] = {**selected, "source": source}
                    return view
            if answer_fields:
                selected = {**context_fields, **answer_fields}
                localized = "\n".join(f"{key}: {value}" for key, value in selected.items())
                source = data.get("source", {})
                view["text"] = localized[:max_chars]
                view["excerpt"] = localized[:max_chars]
                view["data"] = {**selected, "source": source}
                return view
        text = raw_text
        blocks = [
            block.strip()
            for block in re.split(r"\n\s*\n", text)
            if block.strip()
        ]
        target_tokens = cls._contract_target_tokens(contract)
        if not target_tokens:
            return view
        if len(blocks) >= 2:
            required = max(1, (2 * len(target_tokens) + 2) // 3)
            matched = [
                block
                for block in blocks
                if len(cls._content_tokens(block) & target_tokens) >= required
            ]
            if not matched or len(matched) == len(blocks):
                return view
            localized = "\n\n".join(matched)
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) < 2:
                return view
            temporal_mode = str(contract.get("temporal_mode", "none"))
            if temporal_mode == "after":
                scope_tokens = {
                    cls._relation_stem(token)
                    for phrase in contract.get("scope_phrases", [])
                    for token in cls._content_tokens(str(phrase))
                }
                result_tokens = {
                    cls._relation_stem(token)
                    for token in target_tokens
                }
                result_tokens.update(
                    cls._relation_stem(token)
                    for token in cls._content_tokens(
                        str(contract.get("answer_slot", "")).replace("_", " ")
                    )
                )
                result_tokens.difference_update(
                    {"what", "which", "who", "answer", "value", "result", "remain"}
                )
                line_stem_sets = [
                    {
                        cls._relation_stem(token)
                        for token in cls._content_tokens(line)
                    }
                    for line in lines
                ]
                anchors = [
                    index
                    for index, stems in enumerate(line_stem_sets)
                    if scope_tokens.intersection(stems)
                ]
                if anchors:
                    anchor = anchors[-1]
                    anchor_score = len(result_tokens.intersection(line_stem_sets[anchor]))
                    selected_after: int | None = anchor if anchor_score else None
                    if selected_after is None:
                        for candidate in range(anchor + 1, len(lines)):
                            if result_tokens.intersection(line_stem_sets[candidate]):
                                selected_after = candidate
                                break
                    if selected_after is not None:
                        selected_indexes = {selected_after}
                        if selected_after != anchor:
                            selected_indexes.add(anchor)
                        localized = "\n".join(
                            lines[index] for index in sorted(selected_indexes)
                        )
                        view["text"] = localized[:max_chars]
                        view["excerpt"] = localized[:max_chars]
                        source = getattr(record, "data", {}).get("source", {})
                        view["data"] = {"text": localized[:max_chars], "source": source}
                        return view
            relation_stems = {
                cls._relation_stem(token)
                for phrase in contract.get("relation_phrases", [])
                for token in re.findall(r"[a-z0-9]+", str(phrase).lower())
                if token not in {
                    "about", "regarding", "concerning", "according", "to", "by", "of", "the"
                }
            }
            scores = []
            for line in lines:
                line_tokens = cls._content_tokens(line)
                line_stems = {
                    cls._relation_stem(token)
                    for token in re.findall(r"[a-z0-9]+", line.lower())
                }
                scores.append(
                    2 * len(line_tokens & target_tokens)
                    + 3 * len(line_stems & relation_stems)
                )
            best_score = max(scores, default=0)
            if best_score <= 0:
                return view
            best_indexes = [index for index, score in enumerate(scores) if score == best_score]
            if temporal_mode in {"current", "latest", "final", "earliest"}:
                selected_indexes = set(best_indexes)
                best_index = best_indexes[0]
            elif temporal_mode == "after":
                best_index = best_indexes[-1]
                selected_indexes = {best_index}
            else:
                best_index = best_indexes[0]
                selected_indexes = {best_index}
            label_pattern = re.compile(r"^(?P<label>[A-Za-z][A-Za-z0-9 _./-]{0,79}):\s*")
            role_tokens = set(target_tokens)
            role_tokens.update(
                token
                for phrase in contract.get("relation_phrases", [])
                for token in re.findall(r"[a-z0-9]+", str(phrase).lower())
            )
            role_tokens.update(
                re.findall(
                    r"[a-z0-9]+",
                    str(contract.get("answer_slot", "")).lower().replace("_", " "),
                )
            )
            labeled_candidates: list[tuple[int, int]] = []
            for candidate in range(best_index + 1, min(len(lines), best_index + 8)):
                match = label_pattern.match(lines[candidate])
                if not match:
                    continue
                label_tokens = set(re.findall(r"[a-z0-9]+", match.group("label").lower()))
                labeled_candidates.append((len(label_tokens.intersection(role_tokens)), candidate))
            if len(selected_indexes) == 1 and labeled_candidates:
                label_score, label_index = max(labeled_candidates, key=lambda item: (item[0], -item[1]))
                if label_score > 0:
                    selected_indexes.add(label_index)
                elif best_index + 1 < len(lines) and label_pattern.match(lines[best_index + 1]):
                    selected_indexes.add(best_index + 1)
            if len(selected_indexes) == 1 and best_index > 0 and label_pattern.match(lines[best_index]):
                selected_indexes.add(best_index - 1)
            if len(selected_indexes) == 1 and slot_tokens:
                slot_candidates: list[tuple[int, int, int]] = []
                for candidate, line in enumerate(lines):
                    if candidate in selected_indexes:
                        continue
                    raw_line_tokens = set(re.findall(r"[a-z0-9]+", line.lower()))
                    overlap = len(raw_line_tokens.intersection(slot_tokens))
                    if overlap:
                        slot_candidates.append((overlap, -abs(candidate - best_index), candidate))
                if slot_candidates:
                    overlap, _, candidate = max(slot_candidates)
                    if overlap >= min(2, len(slot_tokens)):
                        selected_indexes.add(candidate)
            localized = "\n".join(lines[index] for index in sorted(selected_indexes))
            if not localized or localized == "\n".join(lines):
                return view
        if len(localized) > max_chars:
            localized = localized[:max_chars]
        view["text"] = localized
        view["excerpt"] = localized
        source = getattr(record, "data", {}).get("source", {})
        view["data"] = {"text": localized, "source": source}
        return view

    @classmethod
    def _select_reported_clause(
        cls,
        value: str,
        contract: dict[str, Any],
    ) -> str:
        text = str(value).strip()
        if (
            contract.get("semantic_kind") != "reported_content"
            or contract.get("answer_shape") != "text"
            or contract.get("compound_request")
            or ";" not in text
        ):
            return text
        clauses = [clause.strip() for clause in re.split(r"\s*;\s*", text) if clause.strip()]
        if len(clauses) < 2:
            return text
        target_tokens = cls._contract_target_tokens(contract)
        if not target_tokens:
            return clauses[0]
        ranked = sorted(
            enumerate(clauses),
            key=lambda item: (
                len(cls._content_tokens(item[1]).intersection(target_tokens)),
                -item[0],
            ),
            reverse=True,
        )
        best = ranked[0][1]
        return re.sub(r"[.;]+$", "", best.strip())

    @staticmethod
    def _canonicalize_extracted_value(
        contract: dict[str, Any],
        value: str,
    ) -> str:
        text = str(value).strip()
        slot = str(contract.get("answer_slot", "")).lower().replace("_", " ").strip()
        slot_tokens = set(re.findall(r"[a-z0-9]+", slot))
        if slot:
            slot_label = r"[ _-]+".join(re.escape(token) for token in slot.split())
            text = re.sub(rf"(?i)^{slot_label}\s*:\s*", "", text).strip()
        if "scale" in slot_tokens:
            scale_match = re.search(
                r"(?i)\b([A-G](?:#|b)?\s+(?:major|minor|dorian|phrygian|lydian|mixolydian|aeolian|locrian|pentatonic|blues|chromatic))\s+scale\b",
                text,
            )
            if scale_match:
                text = scale_match.group(1)
        if slot and len(slot_tokens) == 1:
            token = next(iter(slot_tokens))
            text = re.sub(rf"(?i)\s+{re.escape(token)}[.,;:]*$", "", text).strip()
        if slot_tokens.intersection({
            "item", "object", "component", "part", "thing", "artifact", "device", "cause"
        }):
            text = re.sub(r"(?i)^(?:a|an|the)\s+", "", text).strip()
        person_role_tokens = {
            "actor", "recorder", "reviewer", "approver", "author", "owner",
            "inspector", "witness", "researcher", "speaker", "teacher", "coach",
            "sender", "recipient", "reporter", "editor", "maintainer", "operator",
        }
        question = str(contract.get("question", "")).strip().lower()
        relation_tokens = {
            token
            for phrase in contract.get("relation_phrases", [])
            for token in re.findall(r"[a-z0-9]+", str(phrase).lower())
        }
        who_action_contract = bool(
            question.startswith("who ")
            and relation_tokens.difference({"is", "are", "was", "were", "be"})
            and not slot_tokens.intersection({"title", "rank", "role"})
        )
        if slot_tokens.intersection(person_role_tokens) or who_action_contract:
            occupational_prefixes = (
                "Officer", "Farmer", "Inspector", "Technician", "Engineer",
                "Mechanic", "Detective", "Agent", "Captain", "Lieutenant",
                "Sergeant", "Constable", "Clerk", "Coordinator", "Manager",
            )
            prefix_pattern = "|".join(re.escape(item) for item in occupational_prefixes)
            text = re.sub(rf"^(?:{prefix_pattern})\s+", "", text).strip()
        document_field_slot_tokens = {"summary", "explanation", "description"}
        if slot_tokens.intersection(document_field_slot_tokens):
            text = re.sub(r"[.;:]+$", "", text.strip())
        content_slot_tokens = {
            "result", "summary", "explanation", "statement", "content", "finding", "note"
        }
        if slot_tokens.intersection(content_slot_tokens):
            question_words = {"what", "who", "where", "when", "why", "how", "which"}
            for phrase in sorted(
                (str(item).strip() for item in contract.get("target_phrases", [])),
                key=len,
                reverse=True,
            ):
                phrase_tokens = set(re.findall(r"[a-z0-9]+", phrase.lower()))
                if not phrase_tokens or phrase_tokens.intersection(question_words):
                    continue
                if phrase_tokens.issubset(slot_tokens):
                    continue
                text = re.sub(
                    rf"(?i)\s+for\s+{re.escape(phrase)}[.,;:]*$",
                    "",
                    text,
                ).strip()
        return text

    @staticmethod
    def _value_matches_contract_type(
        contract: dict[str, Any],
        value: str,
    ) -> bool:
        text = str(value).strip()
        if not text:
            return False
        slot_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                str(contract.get("answer_slot", "")).lower().replace("_", " "),
            )
        )
        locator_tokens = {
            "url", "uri", "link", "href", "location", "storage", "path",
            "address", "directory", "file", "endpoint", "website", "web",
        }
        contains_locator = bool(
            re.search(r"(?i)\b(?:https?://|file://|s3://|ftp://)\S+", text)
            or re.search(r"(?i)\bURL[- ]?ONLY\b", text)
        )
        heterogeneous_list_tokens = {
            "artifact", "resource", "dependency", "dependencies", "reference",
            "references", "requirement", "requirements", "input", "inputs",
        }
        locator_allowed_by_list_slot = bool(
            contract.get("answer_shape") == "list"
            and slot_tokens.intersection(heterogeneous_list_tokens)
        )
        if (
            contains_locator
            and not slot_tokens.intersection(locator_tokens)
            and not locator_allowed_by_list_slot
        ):
            return False
        person_tokens = {
            "reviewer", "approver", "author", "inspector", "witness", "researcher",
            "speaker", "person", "doctor", "teacher", "patient", "recipient", "sender",
            "owner", "actor",
        }
        named_entity_tokens = {
            "customer", "organization", "company", "vendor", "partner", "project",
            "product", "team", "group", "client", "account_holder",
        }
        identifier_tokens = {
            "id", "ids", "identifier", "identifiers", "code", "codes",
            "reference", "references", "account", "accounts", "token", "tokens",
        }
        if (
            slot_tokens.intersection(person_tokens | named_entity_tokens)
            and not slot_tokens.intersection(identifier_tokens)
            and re.fullmatch(r"[A-Z]{2,}(?:[-_][A-Z0-9]+)+", text)
        ):
            return False
        return True

    @classmethod
    def _mixed_epistemic_correction_sentence(
        cls,
        contract: dict[str, Any],
        evidence_views: list[dict[str, Any]],
    ) -> str:
        relation_tokens = [
            token
            for phrase in contract.get("relation_phrases", [])
            for token in re.findall(r"[a-z0-9]+", str(phrase).lower())
            if token not in {"is", "are", "was", "were", "did", "does", "do", "really"}
        ]
        if not relation_tokens:
            relation_tokens = [
                token
                for token in re.findall(
                    r"[a-z0-9]+",
                    str(contract.get("answer_slot", "")).lower().replace("_", " "),
                )
                if token not in {
                    "did", "does", "do", "was", "were", "is", "are", "has", "have",
                    "value", "answer", "boolean", "status", "event",
                }
            ]
        relation = cls._relation_stem(relation_tokens[-1]) if relation_tokens else "event"
        nominalizations = {
            "delete": "deletion",
            "remove": "removal",
            "ship": "shipment",
            "approve": "approval",
            "decide": "decision",
            "change": "change",
            "reroute": "reroute",
            "install": "installation",
        }
        event_noun = nominalizations.get(relation, relation if relation.endswith("ion") else "event")
        waking_markers = re.compile(
            r"(?i)\b(?:when\s+i\s+woke\s+up|upon\s+waking|in\s+reality|actually|verified|confirmed)\b"
        )
        for view in evidence_views:
            text = str(view.get("excerpt", "")) or str(view.get("text", ""))
            for sentence in [
                item.strip()
                for item in re.split(r"(?<=[.!?])\s+|\n+", text)
                if item.strip()
            ]:
                if not waking_markers.search(sentence):
                    continue
                clause = re.sub(
                    r"(?i)^(?:when\s+i\s+woke\s+up|upon\s+waking|in\s+reality|actually|verified|confirmed)\s*[:,]?\s*",
                    "",
                    sentence,
                ).strip()
                clause = re.sub(r"[.!?]+$", "", clause).strip()
                if clause:
                    clause = clause[0].lower() + clause[1:]
                    return f"the {event_noun} occurred only in a dream and {clause}"
        return ""

    @staticmethod
    def _normalize_negative_correction_clause(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value).strip())
        text = re.sub(r"(?i)^(?:no|false)\s*[;,:-]\s*", "", text).strip()
        operational = re.fullmatch(
            r"(?i)(?P<subject>(?:the\s+)?[a-z][a-z ]*?)\s+flags\s+"
            r"(?P<object>[^;,.]+);\s*it\s+(?:sends|routes)\s+them\s+for\s+"
            r"(?P<purpose>[^.]+)\.?",
            text,
        )
        if operational:
            subject = re.sub(
                r"(?i)^the\s+(?=(?:runtime|code|system|service)\b)",
                "",
                operational.group("subject").strip(),
            )
            text = (
                f"{subject} flags {operational.group('object').strip()} "
                f"for {operational.group('purpose').strip()}."
            )
        text = re.sub(
            r"(?i)^the\s+(?=(?:runtime|code|system|service)\b)",
            "",
            text,
        )
        return text

    @staticmethod
    def _normalize_contract_bound_correction_surface(
        contract: dict[str, Any],
        value: str,
    ) -> str:
        text = re.sub(r"\s+", " ", str(value).strip())
        text = re.sub(
            r"(?i)^(?:the\s+)?(?:"
            r"(?:teacher|audit|source|document|report)\s+(?:note|report|document|result)"
            r"|teacher|audit|source|document|report|note"
            r")\s+(?:(?:explicitly|directly|clearly|specifically)\s+)?"
            r"(?:indicat(?:e|es|ed)|stat(?:e|es|ed)|say|says|said|report(?:s|ed)?|confirm(?:s|ed)?)\s+",
            "",
            text,
        ).strip()
        operational_noun = ""
        for phrase in contract.get("target_phrases", []):
            phrase_tokens = set(re.findall(r"[a-z0-9]+", str(phrase).lower()))
            for candidate in ("runtime", "code", "system", "service"):
                if candidate in phrase_tokens:
                    operational_noun = candidate
                    break
            if operational_noun:
                break
        if operational_noun:
            text = re.sub(
                r"(?i)^(?:the\s+)?(?:runtime|code|system|service)\b",
                operational_noun,
                text,
                count=1,
            )
        text = re.sub(r"(?i)\s+instead\.?$", ".", text).strip()
        for phrase in sorted(
            (str(item).strip() for item in contract.get("target_phrases", [])),
            key=len,
            reverse=True,
        ):
            if not phrase or len(re.findall(r"[A-Za-z0-9]+", phrase)) > 4:
                continue
            if not re.search(r"[A-Z]", phrase):
                continue
            replaced = re.sub(
                rf"(?i)^{re.escape(phrase)}\b",
                "it",
                text,
                count=1,
            )
            if replaced != text:
                text = replaced
                break
        return text

    @classmethod
    def _direct_document_classification_correction(
        cls,
        contract: dict[str, Any],
        evidence_views: list[dict[str, Any]],
    ) -> str:
        target_tokens = cls._contract_target_tokens(contract)
        if not target_tokens:
            return ""
        required_overlap = max(1, min(2, len(target_tokens)))
        document_nouns = (
            "note", "document", "report", "record", "file", "memo", "story",
            "draft", "transcript", "drawing", "sketch", "article", "entry",
        )
        noun_pattern = "|".join(document_nouns)
        pattern = re.compile(
            rf"(?i)\bthis\s+(?P<classification>(?:[a-z][a-z-]*\s+){{1,5}}(?:{noun_pattern}))\b"
        )
        for view in evidence_views:
            text = str(view.get("excerpt", "")) or str(view.get("text", ""))
            if len(cls._content_tokens(text).intersection(target_tokens)) < required_overlap:
                continue
            match = pattern.search(text)
            if match:
                classification = re.sub(
                    r"\s+", " ", match.group("classification").strip().lower()
                )
                return f"it is an {classification}" if not classification.startswith(("a ", "an ", "the ")) else f"it is {classification}"
        return ""

    @classmethod
    def _explicit_negative_finding_sentence(
        cls,
        contract: dict[str, Any],
        evidence_views: list[dict[str, Any]],
    ) -> str:
        target_tokens = cls._contract_target_tokens(contract)
        if not target_tokens:
            return ""
        required_target_overlap = max(1, min(2, len(target_tokens)))
        finding_pattern = re.compile(
            r"(?:"
            r"\b(?:inspection|test|scan|review|examination|audit|check|analysis)\b.{0,100}"
            r"\b(?:found|observed|detected|identified|revealed|showed)\b.{0,30}\bno\b"
            r"|\b(?:found|observed|detected|identified|revealed|showed)\s+no\b"
            r"|\bno\b.{0,80}\b(?:was|were)\s+(?:found|observed|detected|identified)\b"
            r")",
            flags=re.IGNORECASE,
        )
        nonproof_objects = {"evidence", "proof", "confirmation", "support", "documentation"}
        for view in evidence_views:
            text = str(view.get("excerpt", "")) or str(view.get("text", ""))
            sentences = [
                item.strip()
                for item in re.split(r"(?<=[.!?])\s+|\n+", text)
                if item.strip()
            ]
            for sentence in sentences:
                sentence_tokens = cls._content_tokens(sentence)
                if len(sentence_tokens.intersection(target_tokens)) < required_target_overlap:
                    continue
                if sentence_tokens.intersection(nonproof_objects):
                    continue
                if finding_pattern.search(sentence):
                    return sentence
        return ""

    @classmethod
    def _has_target_bound_source_classification(
        cls,
        contract: dict[str, Any],
        evidence_views: list[dict[str, Any]],
    ) -> bool:
        target_tokens = cls._contract_target_tokens(contract)
        if not target_tokens:
            return False
        class_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                str(contract.get("answer_slot", "")).lower().replace("_", " "),
            )
        )
        class_tokens.update(
            {
                "real", "fiction", "fictional", "fantasy", "imaginary", "history",
                "historical", "document", "record", "report", "engineering", "homework",
                "story", "lore", "official", "draft", "transcript",
            }
        )
        required_target_overlap = max(1, min(2, len(target_tokens)))
        for view in evidence_views:
            text = str(view.get("excerpt", "")) or str(view.get("text", ""))
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            for index, line in enumerate(lines):
                windows = [line]
                if index + 1 < len(lines):
                    windows.append(line + " " + lines[index + 1])
                for window in windows:
                    raw_tokens = set(re.findall(r"[a-z0-9]+", window.lower()))
                    if (
                        len(cls._content_tokens(window).intersection(target_tokens))
                        >= required_target_overlap
                        and raw_tokens.intersection(class_tokens)
                        and re.search(
                            r"\b(?:is|are|was|were|classified|marked|labeled|treated|considered|not|fictional|fantasy|real)\b",
                            window,
                            flags=re.IGNORECASE,
                        )
                    ):
                        return True
        return False

    @classmethod
    def _explicit_relation_actor_candidates(
        cls,
        contract: dict[str, Any],
        evidence_views: list[dict[str, Any]],
    ) -> list[str]:
        slot_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                str(contract.get("answer_slot", "")).lower().replace("_", " "),
            )
        )
        person_slots = {
            "reviewer", "approver", "author", "actor", "owner", "inspector",
            "witness", "researcher", "speaker", "person", "teacher", "doctor",
        }
        if not slot_tokens.intersection(person_slots):
            return []
        relation_tokens = {
            token
            for phrase in contract.get("relation_phrases", [])
            for token in re.findall(r"[a-z0-9]+", str(phrase).lower())
            if token not in {"is", "are", "was", "were", "did", "does", "do", "by", "of", "the"}
        }
        role_nouns = {
            "owner", "reviewer", "approver", "author", "inspector", "witness",
            "researcher", "speaker", "person", "teacher", "doctor",
        }
        if relation_tokens and relation_tokens.issubset(role_nouns):
            return []
        if not relation_tokens:
            return []
        relation_stems = {cls._relation_stem(token) for token in relation_tokens}
        if not relation_stems:
            return []
        pattern = re.compile(
            r"(?<![A-Za-z])(?P<name>[A-Z][A-Za-z'-]+(?:[ \t]+[A-Z][A-Za-z'-]+){0,2})"
            r"[ \t]+(?P<verb>[A-Za-z]+)\b"
        )
        candidates: list[str] = []
        seen: set[str] = set()
        for view in evidence_views:
            text = str(view.get("excerpt", "")) or str(view.get("text", ""))
            text = re.sub(r"(?m)^\s*\[[^\]]+\]\s*(?:Correction:\s*)?", "", text)
            for match in pattern.finditer(text):
                if cls._relation_stem(match.group("verb").lower()) not in relation_stems:
                    continue
                name = match.group("name").strip()
                if name.lower() in {"correction", "meeting transcript", "final note"}:
                    continue
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    candidates.append(name)
        return candidates

    @classmethod
    def _needs_actor_role_repair(
        cls,
        contract: dict[str, Any],
        extraction: dict[str, Any],
        evidence_views: list[dict[str, Any]],
    ) -> bool:
        if extraction.get("status") != "extracted":
            return False
        candidates = cls._explicit_relation_actor_candidates(contract, evidence_views)
        if not candidates:
            return False
        candidate_tokens = {
            token.lower()
            for candidate in candidates
            for token in re.findall(r"[A-Za-z][A-Za-z'-]+", candidate)
        }
        for value in extraction.get("values", []):
            value_tokens = {
                token.lower()
                for token in re.findall(r"[A-Za-z][A-Za-z'-]+", str(value))
            }
            if value_tokens and value_tokens.intersection(candidate_tokens):
                return False
        return True

    @staticmethod
    def _preserve_source_surface_name(contract: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(contract.get("question", "")),
                str(contract.get("intent_summary", "")),
                *[str(item) for item in contract.get("scope_phrases", [])],
            ]
        ).lower()
        return bool(
            re.search(
                r"\b(?:according to|top[- ]level note|quoted|forwarded|source says|note says|message says)\b",
                text,
            )
        )

    @staticmethod
    def _unique_full_name_expansion(
        value: str,
        evidence_views: list[dict[str, Any]],
    ) -> str:
        short = str(value).strip()
        short_tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", short)
        if len(short_tokens) != 1:
            return ""
        first = short_tokens[0]
        pattern = re.compile(
            rf"(?<![A-Za-z]){re.escape(first)}(?:[ \t]+[A-Z][A-Za-z'-]+){{1,2}}(?![A-Za-z])"
        )
        matches: list[str] = []
        seen: set[str] = set()
        for view in evidence_views:
            text = (
                str(view.get("excerpt", ""))
                + "\n"
                + str(view.get("text", ""))
                + "\n"
                + json.dumps(view.get("data", {}), ensure_ascii=False, default=str)
            )
            for match in pattern.finditer(text):
                candidate = re.sub(r"[ \t]+", " ", match.group(0)).strip()
                key = candidate.lower()
                if key not in seen:
                    seen.add(key)
                    matches.append(candidate)
        return matches[0] if len(matches) == 1 else ""

    @classmethod
    def _extract_explicit_actor_relation(
        cls,
        contract: dict[str, Any],
        evidence_views: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        slot_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                str(contract.get("answer_slot", "")).lower().replace("_", " "),
            )
        )
        if not slot_tokens.intersection({"actor", "person", "name", "author", "writer", "speaker"}):
            return []
        relation_stems = cls._entity_relation_stems(contract)
        author_stems = {
            cls._relation_stem(token)
            for token in ("write", "wrote", "written", "author", "authored", "provide", "provided", "submit", "submitted")
        }
        if not relation_stems.intersection(author_stems):
            return []
        relation_words = r"(?:wrote|writes|written|authored|authors|provided|provides|submitted|submits)"
        titled_name = r"(?:Mr|Ms|Mrs|Miss|Dr|Prof)\.?\s+[A-Z][A-Za-z'-]+"
        plain_name = r"[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,2}"
        pattern = re.compile(
            rf"(?<![A-Za-z])(?P<name>{titled_name}|{plain_name})\s+{relation_words}\b"
        )
        output: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for view in evidence_views:
            record_id = str(view.get("record_id", ""))
            text = str(view.get("excerpt", "")) or str(view.get("text", ""))
            for match in pattern.finditer(text):
                name = match.group("name").strip()
                key = (name, record_id)
                if key not in seen:
                    seen.add(key)
                    output.append(key)
        return output

    @staticmethod
    def _record_is_cache_like(record: Any) -> bool:
        path = str(getattr(record, "source_path", "")).lower()
        data = getattr(record, "data", {})
        source = data.get("source", {}) if isinstance(data, dict) else {}
        metadata = " ".join(
            [
                path,
                str(source.get("file_name", "")),
                str(source.get("file_stem", "")),
                str(source.get("representation", "")),
            ]
        ).lower()
        metadata_tokens = set(re.findall(r"[a-z0-9]+", metadata))
        if metadata_tokens.intersection({"cache", "cached", "tmp", "temp", "lock"}):
            return True
        text = str(getattr(record, "text", "")).strip().lower()
        return bool(
            re.search(r"\bcache(?:d)?\s+only\b|\bnot\s+a\s+semantic\s+record\b", text[:500])
        )

    @classmethod
    def _apply_source_scope(
        cls,
        records: list[Any],
        contract: dict[str, Any],
    ) -> list[Any]:
        scope = str(contract.get("source_scope", "any"))
        if scope in {"non_cache", "semantic_only"}:
            return [record for record in records if not cls._record_is_cache_like(record)]
        if scope == "cache_only":
            return [record for record in records if cls._record_is_cache_like(record)]
        return records

    @staticmethod
    def _has_explicit_authority_evidence(
        values: list[str],
        evidence_views: list[dict[str, Any]],
    ) -> bool:
        authority = re.compile(
            r"\b(?:official|authoritative|canonical|verified|approve|approved|approver|approval|authenticated|confirmed|confirmation)\b",
            flags=re.IGNORECASE,
        )
        for view in evidence_views:
            text = (
                str(view.get("excerpt", ""))
                + "\n"
                + str(view.get("text", ""))
                + "\n"
                + json.dumps(view.get("data", {}), ensure_ascii=False, default=str)
            )
            for value in values:
                value_text = str(value).strip()
                if not value_text:
                    continue
                index = text.lower().find(value_text.lower())
                if index < 0:
                    continue
                window = text[max(0, index - 220) : index + len(value_text) + 220]
                if authority.search(window):
                    return True
        return False

    def _model_extract(
        self,
        contract: dict[str, Any],
        step_id: str,
        step: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> ToolResult:
        records_by_id: dict[str, Any] = {}
        prior_values: list[Any] = []
        for ref in step["inputs"]:
            result = results[ref]
            for record in result.records:
                records_by_id.setdefault(record.record_id, record)
            prior_values.extend(result.values)
            if result.scalar is not None:
                prior_values.append(result.scalar)
        scoped_records = self._apply_source_scope(
            list(records_by_id.values()),
            contract,
        )
        records = self._prefer_structured_answer_slot_records(
            contract,
            scoped_records,
        )[:12]
        records_by_id = {record.record_id: record for record in records}
        if not records and not prior_values:
            return ToolResult(
                step_id,
                "values",
                diagnostics={"status": "unknown", "reason": "no input material"},
            )
        schema = tool_extraction_schema(contract["contract_id"])
        localized_views = [
            self._localized_record_view(record, contract, 1400)
            for record in records
        ]
        name_expansion_views = [record.model_view(1400) for record in records]
        localized_views_by_id = {
            view["record_id"]: view
            for view in localized_views
        }
        evidence_payload = {
            "records": localized_views,
            "prior_values": prior_values[:30],
            "hints": {
                "terms": step["terms"],
                "fields": step["fields"],
            },
        }
        prompt = (
            "Extract the exact answer-slot value or values from bounded evidence under the immutable semantic "
            "contract. Use only the supplied records and prior values. Preserve target, scope, grammatical actor role, "
            "relation, polarity, temporal mode, epistemic mode, and requested cardinality. For where, location, or storage "
            "requests, an explicitly associated URL, URI, path, address, directory, shelf, room, or similar locator "
            "satisfies the answer slot. For 'who performed X', return "
            "only the explicit actor performing X, not all participants or the person whose statement was opposed. For "
            "current/latest/final requests, compare explicit dated or ordered "
            "events. For allegations, dreams, fiction, quotations, and hypotheticals, extract only the relation the "
            "contract requests. If epistemic_mode is dream, fictional, or hypothetical and the nonactual content itself "
            "is asked to have caused an actual external event, return unknown unless the evidence explicitly asserts that "
            "real-world causal effect. Evidence that the object remained unchanged establishes its state, but by itself does "
            "not establish that a dream or fictional content performed a real causal action or its explicit negation. When "
            "the question asks whether a document, story, drawing, message, or source itself "
            "should be treated as real, fictional, a report, or a record, an explicit source classification such as "
            "'fiction homework, not an engineering record' may directly establish no. For questions asking what a named speaker said, wrote, or sent in a quoted or forwarded "
            "first-person message, resolve first-person pronouns to that explicitly identified speaker and return a concise "
            "third-person reported statement; never return an unattributed 'I'. For reported content, corrections, claims, "
            "or statements about a specified proposition, return only the clause directly about that requested target and "
            "relation; exclude adjacent clauses about other corrections and do not invent or carry over a speaker from a "
            "different field or line. A target-identifying line followed immediately by a labeled field such as teacher "
            "feedback, reviewer note, owner, or status belongs to the same coherent record unless the source explicitly "
            "starts a new record; an explicit 'NAME wrote/authored/provided' phrase in that field binds NAME as the actor. "
            "For questions asking what someone believes "
            "about a target, return the complete attributed proposition content rather than treating it as established fact; "
            "preserve modal words such as should, negation, quantities, units, and intervals. A dream or hypothetical alone cannot establish factual false, but when the same coherent "
            "evidence explicitly states that the event occurred only in the dream and gives a waking, observed, verified, "
            "or inspected state that contradicts it, answer no. Absence alone does not prove a boolean false. Return unknown with no values when the evidence "
            "does not establish the answer. For meaning, definition, or translation requests, mere adjacency in word "
            "salad is not evidence; require an explicit definitional marker such as means, translation, definition, a "
            "labeled field, or an equivalent stated relation. Preserve source units for temperatures, durations, distances, sizes, and "
            "other measured quantities even when the contract answer shape is number. Enforce source_scope exactly: "
            "only the supplied records survive that provenance scope. When authority_mode is explicit_official, return a "
            "value only if the supplied evidence explicitly labels or asserts that value as official, authoritative, "
            "canonical, verified, approved, or authenticated; an ordinary field label alone is insufficient. Return minimal values only, "
            "without role prefixes or explanations, and cite only supplied record IDs.\n"
            f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
        )
        payload = self.model.complete_json("tool_extract", prompt, schema, max_tokens=512)
        extraction = payload["tool_extraction"]
        if (
            extraction.get("status") == "unknown"
            and contract["answer_shape"] == "boolean"
            and contract["epistemic_mode"] == "asserted"
            and self._mixed_epistemic_evidence(records)
        ):
            repair_prompt = (
                "Re-adjudicate a prior strict boolean extraction over mixed epistemic evidence. The prior result was "
                "unknown. Preserve the immutable contract and use only supplied evidence. A dream or hypothetical alone "
                "is unknown, but when the same coherent record explicitly supplies a waking, observed, verified, or "
                "inspected state that contradicts the dreamed event, that explicit contradiction may establish no. Return "
                "the complete replacement extraction with minimal yes/no value and cited record IDs.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
            )
            repaired_payload = self.model.complete_json(
                "tool_extract_epistemic_repair",
                repair_prompt,
                schema,
                max_tokens=512,
            )
            extraction = repaired_payload["tool_extraction"]
        if self._needs_discourse_repair(contract, extraction):
            discourse_slot_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    str(contract.get("answer_slot", "")).lower().replace("_", " "),
                )
            )
            slot_shape_instruction = (
                "The answer slot denotes a requested argument such as an item, object, component, part, device, artifact, "
                "file, or person. Return only that explicit argument phrase, not the complete reported clause. "
                if discourse_slot_tokens.intersection(
                    {"item", "object", "component", "part", "device", "artifact", "file", "person"}
                )
                else "The answer slot denotes proposition content; return the complete requested proposition rather than an unrelated argument. "
            )
            reference_instruction = (
                "For a reported belief or claim about an already named singular target, return the proposition as a "
                "complete sentence using a capitalized referential pronoun such as 'It' rather than repeating a generic "
                "target noun, unless that would be ambiguous. Preserve modal force and end the sentence with punctuation. "
                if contract.get("epistemic_mode") == "reported"
                else "For quoted or forwarded content, identify the explicit speaker rather than replacing the speaker with a generic pronoun. "
            )
            reporting_tense_instruction = (
                "The immutable question uses a past reporting frame. Use natural reported-speech backshift for a simple "
                "present reporting predicate when appropriate (for example, plans to -> planned to), while preserving "
                "future-relative words such as tomorrow, today, later, and modal meaning. "
                if contract.get("reporting_tense") == "past"
                else "Follow the question's present reporting frame and do not introduce an unnecessary past-tense backshift. "
            )
            discourse_prompt = (
                "Normalize a strict attributed-content extraction without changing its proposition or evidence. Use only "
                "the supplied contract, prior extraction, and evidence. For an explicitly quoted or forwarded first-person "
                "message, resolve first-person pronouns to the explicitly identified speaker and return a concise "
                "third-person reported proposition, never an unattributed 'I'. "
                + reporting_tense_instruction
                + slot_shape_instruction
                + reference_instruction
                + "For a belief or reported-content question, return only the requested attributed content, not a claim that it is established fact. When the immutable question asks what someone "
                "did say, write, or report in a past message, align ordinary reporting tense with that frame: first-person "
                "present 'I plan' becomes third-person past '<speaker> planned', not present '<speaker> plans'. Preserve "
                "modal words, negation, quantities, units, intervals, filenames, identifiers, and temporal qualifiers. "
                "Return the complete replacement extraction "
                "with the same cited evidence.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
            )
            discourse_payload = self.model.complete_json(
                "tool_extract_discourse_repair",
                discourse_prompt,
                schema,
                max_tokens=512,
            )
            extraction = discourse_payload["tool_extraction"]
        if self._needs_event_fact_repair(contract, extraction):
            event_prompt = (
                "Normalize a strict model-owned event_fact extraction without changing its evidence. Examine the "
                "immutable question and choose the requested surface form. For a polar yes/no proposition, return a "
                "minimal yes or no and retain the explicit corrective proposition when useful. The prior extraction may "
                "be unknown; re-adjudicate it from the supplied coherent evidence rather than preserving that status. "
                "For dream, fictional, or hypothetical epistemic modes, do not infer a real causal yes or no merely from "
                "the affected object's later state; require an explicit real-world causal assertion, otherwise preserve "
                "unknown. For an open event question, "
                "return the event content itself. A title or heading may establish the subject of the following coherent "
                "document body; a body-level statement about 'this note' or 'this document' applies to that titled subject "
                "unless the source explicitly narrows it. When evidence explicitly contradicts the queried proposition, answer no "
                "and include the grounded correction. Preserve negation, quantities, units, identifiers, and cited evidence. "
                "Return the complete replacement extraction.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
            )
            event_payload = self.model.complete_json(
                "tool_extract_event_fact_repair",
                event_prompt,
                schema,
                max_tokens=512,
            )
            extraction = event_payload["tool_extraction"]
        if (
            contract.get("semantic_kind") == "event_fact"
            and contract.get("answer_shape") == "boolean"
            and extraction.get("status") == "unknown"
        ):
            verdict_prompt = (
                "Adjudicate the immutable yes/no event or relation proposition from only the supplied coherent evidence. "
                "Return supports only when the proposition is explicitly established, contradicts only when the evidence "
                "explicitly rules it out, and insufficient when it merely fails to establish it. Absence is not "
                "contradiction. A document title or heading may bind its subject to the following coherent body; a body "
                "reference such as 'this note', 'this document', or a document-level classification applies to that titled "
                "subject unless explicitly narrowed. A document-level universal negative such as no relation to any member "
                "of a requested category can contradict membership in that category. For contradicts, correction_clause "
                "must be one concise grounded clause suitable immediately after 'No;' and should state the explicit "
                "classification or contrary fact, not discuss evidence or reasoning. For supports or insufficient, leave "
                "correction_clause empty. Cite only supplied record IDs.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
            )
            verdict_payload = self.model.complete_json(
                "event_fact_verdict",
                verdict_prompt,
                event_fact_verdict_schema(contract["contract_id"]),
                max_tokens=512,
            )
            verdict = verdict_payload["event_fact_verdict"]
            if verdict["contract_id"] != contract["contract_id"]:
                raise ProgramValidationError("event fact verdict contract mismatch")
            if not set(verdict["evidence_record_ids"]).issubset(set(records_by_id)):
                raise ProgramValidationError("event fact verdict cites unavailable evidence")
            if (
                verdict["scope_binding"] == "none"
                and records
            ):
                scope_prompt = "\n".join(
                    [
                        (
                            "Re-adjudicate the prior event-fact verdict with explicit document-scope resolution. Treat each "
                            "supplied record as one coherent document, not as disconnected sentences. First identify whether "
                            "its title, heading, or opening subject names the entity in the immutable proposition. Then "
                            "resolve body-level deictic phrases such as 'this note', 'this document', 'this report', or a "
                            "classification noun back to that titled subject. If the bound body explicitly classifies the "
                            "subject as unrelated to the requested category and also states a universal negative relation to "
                            "any member of that category, the proposition that the subject is a member of the category is "
                            "contradicted, not merely unsupported. Conversely, if there is no such binding or explicit "
                            "negative classification or relation, preserve insufficient. Do not infer from absence alone. For "
                            "contradicts, correction_clause must be a short grounded clause suitable after 'No;'. When "
                            "both a direct deictic classification and a category-negative relation are present, prefer the "
                            "direct classification by rewriting 'this [classification]' as 'it is [classification]' rather "
                            "than mentioning incidental content or only the category relation. Cite only supplied record IDs."
                        ),
                        f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}",
                        f"Prior verdict: {json.dumps(verdict, ensure_ascii=False)}",
                        f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}",
                    ]
                )
                scope_payload = self.model.complete_json(
                    "event_fact_scope_repair",
                    scope_prompt,
                    event_fact_verdict_schema(contract["contract_id"]),
                    max_tokens=512,
                )
                verdict = scope_payload["event_fact_verdict"]
                if verdict["contract_id"] != contract["contract_id"]:
                    raise ProgramValidationError("event fact scope verdict contract mismatch")
                if not set(verdict["evidence_record_ids"]).issubset(set(records_by_id)):
                    raise ProgramValidationError("event fact scope verdict cites unavailable evidence")
            extraction = self._extraction_from_event_fact_verdict(contract, verdict)
        if self._needs_mixed_epistemic_correction_repair(contract, extraction, records):
            mixed_prompt = (
                "Normalize a correct negative judgment over mixed nonactual and waking evidence. Preserve the immutable "
                "contract, value no, and evidence citations. Set evidence_relation to direct_contradiction. Put in reason "
                "only a concise grounded correction clause suitable after 'No;': state that the queried event occurred only "
                "in the dream or hypothetical and then mirror the explicit waking or verified state. Do not begin with "
                "'No', 'the evidence', or an explanation of your reasoning. Preserve exact filenames, identifiers, and "
                "objects. Return the complete replacement extraction.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
            )
            mixed_payload = self.model.complete_json(
                "tool_extract_mixed_epistemic_correction",
                mixed_prompt,
                schema,
                max_tokens=512,
            )
            extraction = mixed_payload["tool_extraction"]
        if self._needs_negative_alternative_repair(contract, extraction, records):
            alternative_prompt = (
                "Normalize a correct negative event judgment whose evidence explicitly states the actual alternative "
                "behavior. Preserve value no, the immutable contract, and evidence citations. Set evidence_relation to "
                "direct_contradiction. Put in reason only a concise grounded correction clause suitable after 'No;'. State "
                "what the relevant runtime, system, service, process, or actor actually does, including an explicit purpose "
                "or destination when present. Copy the source's positive action verb and purpose phrase exactly; do not "
                "substitute a different verb, invent a second action, add 'instead', or coordinate two behaviors. When one "
                "clause supplies the agent and another supplies a passive positive predicate, combine them into one concise "
                "active clause without changing the predicate. Preserve exact objects, identifiers, quantities, and "
                "qualifiers. Do not begin with 'No', 'false', 'the evidence', or a reasoning preamble, and do not repeat the "
                "negated queried action. Return the complete replacement extraction.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Localized evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}\n"
                f"Bounded full-record views: {json.dumps(name_expansion_views, ensure_ascii=False, default=str)}"
            )
            alternative_payload = self.model.complete_json(
                "tool_extract_negative_alternative_repair",
                alternative_prompt,
                schema,
                max_tokens=512,
            )
            extraction = alternative_payload["tool_extraction"]
        if self._needs_classification_repair(contract, extraction):
            classification_prompt = (
                "Normalize a strict source-classification extraction without changing its evidence. The model-owned "
                "semantic_kind is source_classification. Examine the immutable question and determine its requested "
                "surface form. When it asks whether a source should be treated as, considered, or classified as a type, "
                "return a minimal yes or no followed by the explicit source classification reason when stated. When it "
                "asks an open category question, return the category itself. Preserve cited evidence and never infer beyond "
                "the supplied source. Return the complete replacement extraction.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
            )
            classification_payload = self.model.complete_json(
                "tool_extract_classification_repair",
                classification_prompt,
                schema,
                max_tokens=512,
            )
            extraction = classification_payload["tool_extraction"]
        if self._needs_negative_correction_repair(contract, extraction, records):
            correction_prompt = (
                "Normalize only the corrective clause of a grounded negative boolean extraction. Preserve contract_id, "
                "status extracted, value no, answer shape, direct_contradiction, and the same evidence citations. Put in "
                "reason one concise complete clause suitable immediately after 'No;'. Do not begin with 'No', 'the "
                "evidence', 'the source', or an explanation of reasoning. When the source explicitly states an affirmative "
                "alternative behavior, report that alternative and omit redundant restatement of the rejected action. "
                "Prefer the concise common-noun operational subject explicitly supported by the target and source (for "
                "example, a product runtime or the final judgment) rather than an unnecessary proper-name prefix. Preserve "
                "identifiers, filenames, quantities, negation, exclusivity words such as 'only', and purpose phrases "
                "such as 'for human review'. Copy the explicit alternative predicate closely rather than weakening or "
                "generalizing it. End with "
                "terminal punctuation. Return the complete replacement extraction.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
            )
            correction_payload = self.model.complete_json(
                "tool_extract_negative_correction",
                correction_prompt,
                schema,
                max_tokens=512,
            )
            extraction = correction_payload["tool_extraction"]
        if (
            contract.get("answer_shape") == "number"
            and (
                extraction.get("status") == "unknown"
                or self._needs_extraction_consistency_repair(extraction)
            )
        ):
            numeric_prompt = (
                "Repair a structurally malformed numeric extraction using only the immutable contract and supplied "
                "evidence. Decide whether one explicit grounded numeric answer is established. When established, set "
                "status extracted and put that number in value as a JSON number. Count explicit rows or items only when "
                "the contract asks for a count; do not leave the number only in reason. When not established, set status "
                "unknown and value 0. Cite only supplied record IDs and return the complete strict repair.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
            )
            numeric_payload = self.model.complete_json(
                "numeric_value_repair",
                numeric_prompt,
                numeric_value_repair_schema(contract["contract_id"]),
                max_tokens=384,
            )
            numeric_repair = numeric_payload["numeric_value_repair"]
            if numeric_repair["contract_id"] != contract["contract_id"]:
                raise ProgramValidationError("numeric repair contract mismatch")
            if not set(numeric_repair["evidence_record_ids"]).issubset(set(records_by_id)):
                raise ProgramValidationError("numeric repair cites unavailable evidence")
            if numeric_repair["status"] == "extracted":
                numeric_value = float(numeric_repair["value"])
                rendered_value = (
                    str(int(numeric_value)) if numeric_value.is_integer() else str(numeric_value)
                )
                extraction = {
                    "contract_id": contract["contract_id"],
                    "status": "extracted",
                    "values": [rendered_value],
                    "answer_shape": contract["answer_shape"],
                    "evidence_record_ids": list(numeric_repair["evidence_record_ids"]),
                    "evidence_relation": numeric_repair["evidence_relation"],
                    "reason": numeric_repair["reason"],
                }
            else:
                extraction = {
                    "contract_id": contract["contract_id"],
                    "status": "unknown",
                    "values": [],
                    "answer_shape": contract["answer_shape"],
                    "evidence_record_ids": list(numeric_repair["evidence_record_ids"]),
                    "evidence_relation": numeric_repair["evidence_relation"],
                    "reason": numeric_repair["reason"],
                }
        if self._needs_extraction_consistency_repair(extraction):
            consistency_prompt = (
                "Repair a structurally inconsistent strict extraction. Use only the immutable contract and supplied "
                "evidence. Status must be extracted when grounded values are returned, and unknown when values is empty. "
                "For answer_shape number, when the evidence or prior reason explicitly establishes a count or numeric value, "
                "put that decimal numeral in values; never leave the count only in reason. Re-adjudicate the conflict without "
                "guessing, preserve correct values and citations, and return the complete replacement extraction.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
            )
            consistency_payload = self.model.complete_json(
                "tool_extract_consistency_repair",
                consistency_prompt,
                schema,
                max_tokens=512,
            )
            extraction = consistency_payload["tool_extraction"]
        extraction = self._enforce_extraction_status_invariant(extraction)
        if (
            extraction.get("status") == "extracted"
            and any(
                not self._value_matches_contract_type(contract, str(value))
                for value in extraction.get("values", [])
                if str(value).strip()
            )
        ):
            type_prompt = (
                "Repair an extraction whose value has the wrong surface type for the immutable answer slot. Use only the "
                "supplied evidence. When the slot asks for a named entity such as a customer, organization, person, team, "
                "product, or project and does not ask for an id, code, token, account, or reference, return the explicit "
                "human-readable name bound to the requested relation rather than an identifier. When the slot asks for an "
                "identifier or locator, preserve that identifier or locator. Return unknown only when no correctly typed "
                "value is explicit. Return the complete replacement extraction.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
            )
            type_payload = self.model.complete_json(
                "tool_extract_type_repair",
                type_prompt,
                schema,
                max_tokens=512,
            )
            extraction = type_payload["tool_extraction"]
        if self._needs_entity_ambiguity_repair(contract, extraction, records):
            ambiguity_prompt = (
                "Re-adjudicate a strict entity selection when the bounded evidence explicitly signals ambiguity, unresolved "
                "identity, a requirement to keep entities separate, or a need for clarification. Return the selected entity "
                "only if the immutable requested relation uniquely and explicitly binds that entity. Do not infer a merge, "
                "equivalence, ownership, role, or identity from mere co-occurrence, a nearby name, or a statement that the "
                "entities must remain separate. If the evidence says it does not identify which entity, is unclear, is "
                "ambiguous, or awaits clarification, return status unknown with no values. Preserve cited evidence and return "
                "the complete replacement extraction.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
            )
            ambiguity_payload = self.model.complete_json(
                "tool_extract_entity_ambiguity_repair",
                ambiguity_prompt,
                schema,
                max_tokens=512,
            )
            extraction = ambiguity_payload["tool_extraction"]
        if self._needs_actor_role_repair(contract, extraction, localized_views):
            actor_candidates = self._explicit_relation_actor_candidates(contract, localized_views)
            actor_prompt = (
                "Repair a person-role extraction that selected a nearby speaker or participant instead of the grammatical "
                "actor of the requested relation. Use only the immutable contract and supplied evidence. The explicit "
                f"grammatical-subject candidates are {json.dumps(actor_candidates, ensure_ascii=False)}. Return the fullest "
                "explicit name in the evidence that unambiguously refers to the correct candidate, not a transcript speaker "
                "label, author of another clause, owner of another relation, or nearby participant. Preserve evidence "
                "citations and return the complete replacement extraction.\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                f"Evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}"
            )
            actor_payload = self.model.complete_json(
                "tool_extract_actor_role_repair",
                actor_prompt,
                schema,
                max_tokens=512,
            )
            extraction = actor_payload["tool_extraction"]
        if (
            extraction.get("status") == "extracted"
            and len(extraction.get("values", [])) == 1
            and not self._preserve_source_surface_name(contract)
        ):
            full_name = self._unique_full_name_expansion(
                str(extraction["values"][0]),
                name_expansion_views,
            )
            if full_name:
                full_name_prompt = (
                    "Normalize a person extraction to the unique fullest explicit name in the bounded evidence. The prior "
                    f"short value is {json.dumps(str(extraction['values'][0]), ensure_ascii=False)} and the unique explicit "
                    f"full-name expansion is {json.dumps(full_name, ensure_ascii=False)}. Preserve the same person, relation, "
                    "answer shape, and evidence citations; do not change to another participant. Return the complete "
                    "replacement extraction.\n"
                    f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                    f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
                    f"Localized evidence: {json.dumps(evidence_payload, ensure_ascii=False, default=str)}\n"
                    f"Bounded full-record views for name expansion: {json.dumps(name_expansion_views, ensure_ascii=False, default=str)}"
                )
                full_name_payload = self.model.complete_json(
                    "tool_extract_full_name_repair",
                    full_name_prompt,
                    schema,
                    max_tokens=512,
                )
                extraction = full_name_payload["tool_extraction"]
        if extraction.get("status") == "unknown":
            explicit_actors = self._extract_explicit_actor_relation(contract, localized_views)
            if explicit_actors:
                extraction = {
                    **extraction,
                    "status": "extracted",
                    "values": [value for value, _ in explicit_actors],
                    "evidence_record_ids": list(dict.fromkeys(record_id for _, record_id in explicit_actors if record_id)),
                    "evidence_relation": "direct_support",
                    "reason": "A model-declared author relation is explicitly stated in the bounded evidence.",
                }
        if extraction["contract_id"] != contract["contract_id"]:
            raise ProgramValidationError("tool extraction contract mismatch")
        if extraction["answer_shape"] != contract["answer_shape"]:
            raise ProgramValidationError("tool extraction answer shape mismatch")
        available_ids = set(records_by_id)
        evidence_ids = set(extraction["evidence_record_ids"])
        if not evidence_ids.issubset(available_ids):
            raise ProgramValidationError("tool extraction cites unavailable evidence")
        values = [
            self._canonicalize_extracted_value(contract, str(value))
            for value in extraction["values"]
            if str(value).strip()
        ]
        if (
            extraction.get("status") == "unknown"
            and contract["answer_shape"] == "boolean"
            and self._mixed_epistemic_evidence(records)
            and self._reason_explicit_false(extraction.get("reason", ""))
        ):
            extraction = {**extraction, "status": "extracted", "values": ["no"]}
            values = ["no"]
        if (
            contract["answer_shape"] == "boolean"
            and self._reason_is_nonproof(extraction.get("reason", ""))
        ):
            if self._contract_asks_proof_status(contract):
                extraction = {
                    **extraction,
                    "status": "extracted",
                    "values": ["no"],
                    "evidence_relation": "direct_contradiction",
                    "reason": "The requested proof status is explicitly negative.",
                }
                values = ["no"]
            else:
                extraction = {
                    **extraction,
                    "status": "unknown",
                    "values": [],
                    "evidence_relation": "absence",
                    "reason": "The evidence states only that the proposition was unproven or unconfirmed.",
                }
                values = []
        if contract["answer_shape"] == "boolean":
            values = [
                "yes" if value.lower() == "true" else "no" if value.lower() == "false" else value
                for value in values
            ]
        if contract.get("semantic_kind") == "reported_content":
            values = [self._select_reported_clause(value, contract) for value in values]
        if values and not all(self._value_matches_contract_type(contract, value) for value in values):
            extraction = {
                **extraction,
                "status": "unknown",
                "values": [],
                "evidence_relation": "type_mismatch",
                "reason": "The extracted value type does not match the requested answer slot.",
            }
            values = []
        identifier_slot_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                str(contract.get("answer_slot", "")).lower().replace("_", " "),
            )
        )
        if identifier_slot_tokens.intersection(
            {"id", "identifier", "code", "reference", "account"}
        ):
            values = [re.sub(r"[.,;:]+$", "", value.strip()) for value in values]
        values = [
            re.sub(r"[.,;:]+$", "", value.strip())
            if re.match(r"^https?://", value.strip(), flags=re.IGNORECASE)
            else value
            for value in values
        ]
        entity_name_slot_tokens = {
            "organization", "owner", "reviewer", "approver", "actor", "person",
            "name", "witness", "inspector", "researcher", "speaker", "author",
        }
        if (
            contract.get("semantic_kind") == "entity_attribute"
            and identifier_slot_tokens.intersection(entity_name_slot_tokens)
        ):
            values = [re.sub(r"[.,;:]+$", "", value.strip()) for value in values]
        cited_records = [records_by_id[item] for item in extraction["evidence_record_ids"] if item in records_by_id]
        cited_views = [
            localized_views_by_id[item]
            for item in extraction["evidence_record_ids"]
            if item in localized_views_by_id
        ]
        corrective_sentence = ""
        if (
            contract.get("answer_shape") == "boolean"
            and any(value.strip().lower() in {"no", "false"} for value in values)
        ):
            if (
                extraction.get("evidence_relation") == "direct_contradiction"
                and self._mixed_epistemic_evidence(cited_records)
            ):
                corrective_sentence = self._mixed_epistemic_correction_sentence(
                    contract,
                    cited_views,
                ) or str(extraction.get("reason", "")).strip()
            if not corrective_sentence:
                corrective_sentence = self._direct_document_classification_correction(
                    contract,
                    cited_views,
                )
            if not corrective_sentence and self._contract_asks_proof_status(contract):
                corrective_sentence = self._proof_status_correction_sentence(
                    contract,
                    cited_records,
                )
            if (
                not corrective_sentence
                and extraction.get("evidence_relation") == "direct_contradiction"
                and self._has_explicit_alternative_behavior(cited_records)
            ):
                corrective_sentence = str(extraction.get("reason", "")).strip()
            if not corrective_sentence:
                corrective_sentence = self._explicit_negative_finding_sentence(contract, cited_views)
            if (
                not corrective_sentence
                and extraction.get("evidence_relation") == "direct_contradiction"
            ):
                corrective_sentence = self._normalize_negative_correction_clause(
                    str(extraction.get("reason", ""))
                )
            if corrective_sentence:
                extraction = {
                    **extraction,
                    "evidence_relation": "direct_contradiction",
                    "reason": f"The scoped evidence explicitly reports a negative finding: {corrective_sentence}",
                }
        values = self._expand_temporal_values(values, cited_records, contract["temporal_mode"])
        if (
            values
            and contract.get("semantic_kind") == "source_classification"
            and contract.get("answer_shape") == "boolean"
            and not self._has_target_bound_source_classification(contract, cited_views)
        ):
            extraction = {
                **extraction,
                "status": "unknown",
                "values": [],
                "evidence_relation": "absence",
                "reason": "The localized evidence does not explicitly bind the requested source classification to the named target.",
            }
            values = []
        if (
            values
            and contract.get("authority_mode") == "explicit_official"
            and not self._has_explicit_authority_evidence(values, cited_views)
        ):
            extraction = {
                **extraction,
                "status": "unknown",
                "values": [],
                "evidence_relation": "absence",
                "reason": "The scoped evidence does not explicitly establish the value as official or authoritative.",
            }
            values = []
        if (
            contract.get("answer_shape") == "boolean"
            and extraction.get("evidence_relation") in {
                "absence", "nonactual_content", "unknown"
            }
            and contract.get("semantic_kind") != "source_classification"
        ):
            extraction = {
                **extraction,
                "status": "unknown",
                "values": [],
                "reason": "The model classified the evidence as nonactual, absent, or otherwise insufficient for a boolean fact judgment.",
            }
            values = []
        if (
            contract.get("world_scope") == "nonactual_external_effect"
            and extraction.get("evidence_relation") in {
                "state_only", "absence", "nonactual_content", "unknown"
            }
        ):
            extraction = {
                **extraction,
                "status": "unknown",
                "values": [],
                "reason": "The model classified the evidence as insufficient for a real external causal effect.",
            }
            values = []
        if (
            values
            and contract.get("semantic_kind") == "entity_attribute"
            and contract.get("answer_shape") != "number"
            and not all(
                self._value_has_explicit_entity_relation(contract, value, cited_views)
                for value in values
            )
        ):
            extraction = {
                **extraction,
                "status": "unknown",
                "values": [],
                "reason": "The localized evidence does not explicitly bind the value to the requested entity relation.",
            }
            values = []
        if (
            values
            and self._is_definition_request(contract)
            and not self._has_explicit_definition_evidence(contract, values, cited_records)
        ):
            extraction = {
                **extraction,
                "status": "unknown",
                "values": [],
                "reason": "The evidence contains no explicit definition or translation relation.",
            }
            values = []
        if (
            contract["answer_shape"] == "boolean"
            and any(value.lower() in {"false", "no"} for value in values)
            and contract["epistemic_mode"] == "asserted"
            and contract["semantic_kind"] != "source_classification"
            and self._fiction_only_boolean_evidence(cited_records)
        ):
            extraction = {
                **extraction,
                "status": "unknown",
                "values": [],
                "reason": "Fictional or hypothetical evidence alone cannot establish an asserted factual false.",
            }
            values = []
        if values and all(self._unknown_like_value(value) for value in values):
            extraction = {**extraction, "status": "unknown", "values": []}
            values = []
        if extraction["status"] == "unknown" and contract.get("answer_shape") == "boolean":
            negative_finding = self._explicit_negative_finding_sentence(contract, cited_views)
            if negative_finding:
                extraction = {
                    **extraction,
                    "status": "extracted",
                    "values": ["no"],
                    "evidence_relation": "direct_contradiction",
                    "reason": f"The scoped evidence explicitly reports a negative finding: {negative_finding}",
                }
                values = ["no"]
                corrective_sentence = negative_finding
        if extraction["status"] == "unknown":
            if values:
                raise ProgramValidationError("unknown tool extraction cannot contain values")
            return ToolResult(
                step_id,
                "values",
                records=[records_by_id[item] for item in evidence_ids if item in records_by_id],
                diagnostics={"status": "unknown", "reason": extraction["reason"]},
            )
        if not values:
            raise ProgramValidationError("extracted status requires values")
        if records and not evidence_ids:
            raise ProgramValidationError("model extraction over records requires evidence citations")
        return ToolResult(
            step_id,
            "values",
            values=values[: max(1, min(step["limit"] or 20, 20))],
            records=cited_records,
            diagnostics={
                "status": "extracted",
                "reason": extraction["reason"],
                "evidence_record_ids": extraction["evidence_record_ids"],
                **({"corrective_sentence": corrective_sentence} if corrective_sentence else {}),
            },
        )

    def _profile_dataset(self) -> dict[str, Any]:
        if self._dataset_profile is not None:
            return self._dataset_profile
        fingerprint = self.catalog.fingerprint
        schema = dataset_profile_schema(fingerprint)
        prompt = (
            "Profile the structure and semantics of an unfamiliar raw-folder dataset. This is dataset-level "
            "schema induction, not question answering. Describe at most six useful collections. Use only exact "
            "collection and field paths shown in the catalog. Identify coherent record granularity, identity fields, "
            "temporal fields, text fields, and suitable generic operations. Do not invent facts or benchmark-specific "
            "intents. Prefer coherent logical records over line fragments when both exist.\n"
            f"Dataset fingerprint: {fingerprint}\n"
            f"Catalog: {self.catalog.summary(7000)}"
        )
        payload = self.model.complete_json("dataset_profile", prompt, schema, max_tokens=1024)
        profile = payload["dataset_profile"]
        valid_collections: list[dict[str, Any]] = []
        for item in profile["collections"]:
            path = item["collection_path"]
            if path in {"all_records", "all_representations"} or self.catalog.collection_records(path):
                valid_collections.append(item)
        universal = {
            "collection_path": "all_records",
            "purpose": "Universal coherent evidence records across every source file; use for evidence-first retrieval when no narrow collection explicitly proves the requested relation.",
            "record_granularity": "record",
            "identity_fields": [],
            "temporal_fields": [],
            "text_fields": ["text", "source.path", "source.file_name", "source.file_stem"],
            "extraction_notes": "Search exact source substrings, then project real fields or use generic extract_values for text.",
        }
        valid_collections = [item for item in valid_collections if item["collection_path"] != "all_records"]
        profile = {**profile, "collections": [universal, *valid_collections]}
        self._dataset_profile = profile
        return profile

    def _contract_id(self, question: str) -> str:
        material = f"semantic-contract-v2\0{self.catalog.fingerprint}\0{question}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    def _parse_semantics(self, question: str) -> dict[str, Any]:
        contract_id = self._contract_id(question)
        schema = semantic_contract_schema(question, contract_id)
        prompt = (
            "Create an immutable semantic contract for the question independently of any dataset. Preserve exact "
            "targets, scopes, grammatical roles, relations, constraints, negation, temporal requirements, epistemic "
            "status, answer cardinality, semantic_kind, world_scope, source_scope, and authority_mode. Set source_scope "
            "to any unless the wording explicitly selects or excludes a provenance class. Use non_cache when the question "
            "says despite, ignore, exclude, or do not use a cache; use cache_only when it specifically asks for a hidden or "
            "cached value; use semantic_only when it explicitly asks for the semantic or meaningful record. Set "
            "authority_mode to explicit_official only when the requested answer must itself be explicitly official, "
            "authoritative, canonical, verified, or approved; otherwise use any. A cache-only value is not official merely "
            "because another non-cache source supplies an official value. Use asserted_world for ordinary real-world "
            "facts, reported_content for beliefs or attributed claims, nonactual_internal for what happened inside a dream, "
            "fiction, or hypothetical scenario, nonactual_external_effect when nonactual content itself is alleged to have "
            "caused an external real-world effect, and source_metadata for classification of the source itself. Apply this "
            "precedence strictly: when the grammatical actor is a dream, fictional story, imagined scenario, hypothesis, "
            "rumor, belief, or claim and the predicate asks whether it actually or really caused an external event, "
            "world_scope MUST be nonactual_external_effect and epistemic_mode MUST reflect that nonactual source. Words "
            "such as actually or really do not convert the nonactual actor into asserted_world. For example, 'Did an "
            "imagined scenario actually erase an external file?' is nonactual_external_effect, whereas 'Was the file erased "
            "inside the imagined scenario?' is nonactual_internal. Conversely, when a dream or fictional episode is only a "
            "temporal anchor and the question asks what real object or state remains before or after it, world_scope MUST be "
            "asserted_world; for example, 'What remains installed after the dream?' asks for the later real inventory, not "
            "an event inside the dream. Use source_classification only when the requested answer "
            "classifies the source, document, drawing, story, message, report, or record itself; use event_fact for whether "
            "an event occurred; reported_content for speech, belief, claims, or quoted content; definition for meaning or "
            "translation; calculation for arithmetic; and entity_attribute for ordinary values. For reported_content, "
            "distinguish proposition questions from argument questions. A form like 'What did a person say about a topic?' "
            "asks for proposition content. A form like 'What did a person say broke, snapped, failed, arrived, or was "
            "missing?' asks for the explicit argument filling that predicate, not the whole clause; use a narrow answer_slot "
            "such as broken_item, snapped_item, failed_component, arrived_object, or missing_artifact with answer_shape text. "
            "When the question is "
            "polar—asking whether, should, does, did, is, was, can, or another yes/no proposition—use answer_shape "
            "boolean even when an explanatory correction is available. A question of the form "
            "'Who performed relation X?' asks for the actor who "
            "explicitly performs X, not every participant, target, opponent, or mentioned person. Use answer_shape text "
            "for one requested actor or value; use list only when the wording explicitly requests multiple values or a "
            "set. Do not broaden singular wording into parties, participants, or groups. Distinguish asserted facts from "
            "allegations, fiction, dreams, quotations, and hypotheticals. When a dream, fictional narrative, imagined "
            "scenario, or hypothesis is itself the grammatical subject of an alleged real-world causal action, set "
            "epistemic_mode to dream, fictional, or hypothetical as appropriate; do not silently convert the question to "
            "the current state of the affected object. A boolean answer requires explicit evidence for the proposition or "
            "its explicit negation; absence, fiction, or a dream alone does not prove false. "
            "Unknown translations or unstated facts require explicit evidence before answering. Questions asking what "
            "a named person said, wrote, reported, claimed, or believed require attributed content: preserve that person's "
            "speaker role, modal words, negation, quantities, units, and temporal qualifiers. Use epistemic_mode reported "
            "for beliefs or reported claims and quoted for explicitly quoted or forwarded message content. Set reporting_tense "
            "to past when the question itself frames the reporting act in the past (for example what someone did say, write, "
            "report, or claim), present when the question frames the reporting act in the present, and none when reporting "
            "tense is not applicable. This field controls only natural reported-speech surface tense and must not alter the "
            "underlying proposition or its temporal qualifiers. Use visible "
            "question phrases in the phrase arrays.\n"
            f"Question: {question}\n"
            f"Contract ID: {contract_id}"
        )
        payload = self.model.complete_json("semantic_contract", prompt, schema, max_tokens=1024)
        contract = self._normalize_contract(payload["semantic_contract"])
        self._validate_contract(question, contract)
        return contract

    @staticmethod
    def _normalize_contract(contract: dict[str, Any]) -> dict[str, Any]:
        """Repair internal inconsistencies using only the model-owned contract text."""
        normalized = dict(contract)
        normalized.setdefault("reporting_tense", "none")
        question_text = str(contract.get("question", "")).strip()
        source_nouns = {
            "text", "file", "record", "document", "note", "scan", "table", "calendar",
            "log", "json", "csv", "tsv", "blob", "memo", "report", "dataset", "folder",
            "corpus", "transcript", "correction",
        }
        source_match = re.search(
            r"(?i)\b(?:in|from|inside|within|according to)\s+(?:the\s+)?([^?.,;]{1,90})",
            question_text,
        )
        if source_match:
            source_phrase = source_match.group(1).strip()
            source_tokens = set(re.findall(r"[a-z0-9]+", source_phrase.lower()))
            if source_tokens.intersection(source_nouns):
                scopes = [str(item) for item in normalized.get("scope_phrases", [])]
                if source_phrase.lower() not in {item.lower() for item in scopes}:
                    normalized["scope_phrases"] = [*scopes, source_phrase]
        if not contract.get("relation_phrases"):
            if re.search(r"(?i)\b(?:also called|also known as|known as|nicknamed|alias(?:ed)? as)\b", question_text):
                normalized["relation_phrases"] = ["also called"]
            else:
                relation_match = re.match(r"(?i)^who\s+([a-z]+)\b", question_text)
                if relation_match and relation_match.group(1).lower() not in {
                    "is", "are", "was", "were", "does", "do", "did", "has", "have", "had",
                }:
                    normalized["relation_phrases"] = [relation_match.group(1)]
        semantic_text = " ".join(
            [
                str(contract.get("intent_summary", "")),
                str(contract.get("answer_slot", "")),
                *[str(item) for item in contract.get("target_phrases", [])],
                *[str(item) for item in contract.get("scope_phrases", [])],
                *[str(item) for item in contract.get("relation_phrases", [])],
                *[str(item) for item in contract.get("constraint_phrases", [])],
            ]
        ).lower()
        excludes_cache = bool(
            re.search(
                r"\b(?:ignore|ignoring|exclude|excluding|despite|without|non[- ]?cache)\b"
                r".{0,80}\b(?:cache|cached)\b",
                semantic_text,
            )
            or re.search(
                r"\b(?:cache|cached)\b.{0,80}\b(?:ignore|ignoring|exclude|excluding)\b",
                semantic_text,
            )
        )
        requests_cache = bool(
            re.search(
                r"\b(?:hidden|cached|cache)\b.{0,50}\b(?:url|value|record|entry|field|source)\b",
                semantic_text,
            )
            or re.search(
                r"\b(?:url|value|record|entry|field|source)\b.{0,50}\b(?:cache|cached)\b",
                semantic_text,
            )
        )
        requests_semantic = bool(
            re.search(
                r"\b(?:semantic|meaningful|authoritative)\b.{0,40}\b(?:record|source|document|entry)\b",
                semantic_text,
            )
        )
        if excludes_cache:
            normalized["source_scope"] = "non_cache"
        elif requests_cache:
            normalized["source_scope"] = "cache_only"
        elif requests_semantic and normalized.get("source_scope") in {None, "", "any", "unknown"}:
            normalized["source_scope"] = "semantic_only"
        if (
            re.search(
                r"\b(?:official|authoritative|canonical|verified|approved|authenticated)\b",
                semantic_text,
            )
            and normalized.get("authority_mode") in {None, "", "any", "unknown"}
        ):
            normalized["authority_mode"] = "explicit_official"
        normalized.setdefault("source_scope", "any")
        normalized.setdefault("authority_mode", "any")
        return normalized

    def _validate_contract(self, question: str, contract: dict[str, Any]) -> None:
        if contract["question"] != question:
            raise ProgramValidationError("semantic contract question mismatch")
        rendered = json.dumps(contract, ensure_ascii=False).lower()
        anchors = set(
            re.findall(
                r"\b[A-Z][A-Za-z0-9_-]{2,}\b|https?://\S+|\b[A-Za-z]+-\d+\b|\b\d+(?:\.\d+)?\b",
                question,
            )
        )
        missing = [anchor for anchor in sorted(anchors) if anchor.lower() not in rendered]
        if missing:
            raise ProgramValidationError(f"semantic contract omitted explicit anchors: {missing}")
        if not contract["intent_summary"].strip() or not contract["answer_slot"].strip():
            raise ProgramValidationError("semantic contract is incomplete")

    def _compile_program(
        self,
        profile: dict[str, Any],
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        contract_id = contract["contract_id"]
        schema = query_program_schema(contract_id)
        prompt = (
            'Compile the immutable semantic contract into a complete generic tool program. Do not answer the '
            'question or reinterpret its meaning. Prefer coherent logical records. For ambiguous relations such as '
            'owner, reviewer, state, author, or status, first retrieve evidence containing both the explicit target and '
            'relation; never select a same-named field from an unrelated record. Use all_records unless a narrow '
            'collection explicitly contains both. Search terms must be literal source substrings. When multiple terms '
            'jointly identify one target, use all matching, not any. Use project_values only for actual structured answer '
            'fields. Use extract_values for labels, spans, regex captures, URLs, identifiers, dates, numbers, or event '
            'series. Each compact step contains tool, inputs, collection, terms, fields, filters, arguments, and limit. '
            'The mode argument is only for search and must be all, any, or phrase. Extraction uses an extractor argument '
            'such as field, after_label, after_phrase, before_phrase, regex, url, identifier, date_time, number, or '
            'event_series. Add label, start_phrase, pattern, value_group, time_group, occurrence, value_kind, and '
            'strip_chars only when needed. A field extractor must name an actual answer field, never generic text. For '
            'current, latest, or final values, use explicit temporal evidence. For negation, allegations, fiction, dreams, '
            'or quotations, retrieve coherent context for grounded reasoning. Use at most five steps.\n'
            f"Tools: {json.dumps(_TOOL_DESCRIPTIONS, ensure_ascii=False)}\n"
            f"Dataset profile: {json.dumps(profile, ensure_ascii=False)}\n"
            f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
            f"Catalog: {self.catalog.summary(6500)}"
        )
        payload = self.model.complete_json("query_program", prompt, schema, max_tokens=1536)
        program = self._normalize_program(payload["query_program"], contract)
        try:
            self._validate_program(contract, program)
        except ProgramValidationError as exc:
            repair_prompt = (
                'Repair the structurally invalid generic tool program. Return a complete replacement, not an answer. '
                'Preserve the immutable semantic contract. Use only valid catalog paths and earlier step indexes. The '
                'primary multi-term target search must use all or phrase matching. Use all_records when a narrow '
                'collection does not explicitly contain both target and relation. Never project generic text as an answer; '
                'choose a real structured answer field or a text extractor with explicit extractor arguments. Use mode only '
                'for search and extractor only for extraction.\n'
                f"Validation error: {exc}\n"
                f"Rejected program: {json.dumps(program, ensure_ascii=False)}\n"
                f"Dataset profile: {json.dumps(profile, ensure_ascii=False)}\n"
                f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
                f"Catalog: {self.catalog.summary(6500)}"
            )
            repaired = self.model.complete_json("query_program_repair", repair_prompt, schema, max_tokens=1536)
            program = self._normalize_program(repaired["query_program"], contract)
            self._validate_program(contract, program)
        self.model_query_trace = {
            "dataset_profile": profile,
            "semantic_contract": contract,
            "program": program,
            "dataset_fingerprint": self.catalog.fingerprint,
        }
        return program

    @staticmethod
    def _normalize_program(
        program: dict[str, Any],
        contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply structural normalization and consume explicit model-owned contract scope."""
        normalized = {
            "contract_id": program["contract_id"],
            "steps": [dict(step) for step in program["steps"]],
        }
        count_request = False
        superlative_count_request = False
        if contract is not None and contract.get("semantic_kind") == "calculation":
            question_text = str(contract.get("question", "")).lower()
            fallback_text = " ".join(
                [
                    *[str(item) for item in contract.get("target_phrases", [])],
                    *[str(item) for item in contract.get("relation_phrases", [])],
                ]
            ).lower()
            calculation_text = question_text or fallback_text
            numbers = [float(item) for item in re.findall(r"(?<![a-z0-9_.-])-?\d+(?:\.\d+)?", calculation_text)]
            if len(numbers) < 2:
                calculation_text = fallback_text
                numbers = [float(item) for item in re.findall(r"(?<![a-z0-9_.-])-?\d+(?:\.\d+)?", calculation_text)]
            operation = ""
            operation_patterns = [
                ("add", r"\b(?:plus|add|added to|sum of)\b"),
                ("subtract", r"\b(?:minus|subtract|subtracted from|difference between)\b"),
                ("multiply", r"\b(?:times|multiply|multiplied by|product of)\b"),
                ("divide", r"\b(?:divide|divided by|quotient of)\b"),
            ]
            for candidate, pattern in operation_patterns:
                if re.search(pattern, calculation_text):
                    operation = candidate
                    break
            if operation and len(numbers) >= 2:
                normalized["steps"] = [
                    {
                        "tool": "calculate",
                        "inputs": [],
                        "collection": "",
                        "terms": [],
                        "fields": [],
                        "filters": [],
                        "arguments": [
                            {
                                "name": "operation",
                                "value": operation,
                                "values": [],
                                "numbers": numbers[:8],
                            }
                        ],
                        "limit": 1,
                    }
                ]
        if contract is not None:
            slot_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    str(contract.get("answer_slot", "")).lower().replace("_", " "),
                )
            )
            intent_text = str(contract.get("intent_summary", "")).lower()
            contract_phrase_text = " ".join(
                [
                    intent_text,
                    *[str(item).lower() for item in contract.get("target_phrases", [])],
                    *[str(item).lower() for item in contract.get("constraint_phrases", [])],
                ]
            )
            count_request = bool(
                slot_tokens.intersection({"count", "number", "total"})
                or intent_text.startswith("count ")
                or "count the number" in intent_text
            )
            superlative_count_request = bool(
                not count_request
                and (
                    "highest count" in contract_phrase_text
                    or "most " in contract_phrase_text
                    or "largest number" in contract_phrase_text
                )
                and re.search(r"\b(?:row|rows|entry|entries|record|records)\b", contract_phrase_text)
            )
            if count_request:
                constraint_field = ""
                constraint_value = ""
                condition_phrases = [
                    *contract.get("constraint_phrases", []),
                    *contract.get("relation_phrases", []),
                    *contract.get("scope_phrases", []),
                    *contract.get("target_phrases", []),
                    str(contract.get("question", "")),
                ]
                for phrase in condition_phrases:
                    text = str(phrase).lower()
                    match = re.search(
                        r"\b(status|state|owner|reviewer|phase|condition|category|type)\b"
                        r"\s*(?:is|are|was|were|equals?|=|:)?\s*([a-z0-9_-]+)",
                        text,
                    )
                    if match and match.group(2) not in {
                        "in", "on", "at", "of", "for", "from", "to", "by", "with",
                        "within", "into", "the", "a", "an",
                    }:
                        constraint_field = match.group(1)
                        constraint_value = match.group(2)
                        break
                if not constraint_field:
                    for phrase in condition_phrases:
                        text = str(phrase).lower().replace("_", " ")
                        match = re.search(
                            r"(?:\b(?:have|has|had|with)\s+)?"
                            r"\b(?P<value>[a-z0-9_-]+)\s+"
                            r"(?P<field>[a-z0-9_-]+\s+status)\b",
                            text,
                        )
                        if not match:
                            continue
                        value = match.group("value")
                        if value in {
                            "in", "on", "at", "of", "for", "from", "to", "by", "with",
                            "within", "into", "the", "a", "an", "many", "how",
                        }:
                            continue
                        constraint_field = re.sub(
                            r"[^a-z0-9]+",
                            "_",
                            match.group("field"),
                        ).strip("_")
                        constraint_value = value
                        break
                if not constraint_field:
                    status_values = {
                        "active", "archived", "blocked", "open", "closed", "ready",
                        "paused", "waiting", "draft", "released", "approved", "stable",
                        "review", "monitoring", "testing", "reopened",
                    }
                    for phrase in condition_phrases:
                        text = str(phrase).lower()
                        match = re.search(
                            r"\b(?:is|are|was|were|be|being)\s+([a-z0-9_-]+)\b",
                            text,
                        )
                        if match and match.group(1) in status_values:
                            constraint_field = "status"
                            constraint_value = match.group(1)
                            break
                        match = re.search(
                            r"\b(active|archived|blocked|open|closed|ready|paused|waiting|draft|released|approved|stable)\b"
                            r"\s+(?:row|rows|entry|entries|record|records)\b",
                            text,
                        )
                        if match:
                            constraint_field = "state"
                            constraint_value = match.group(1)
                            break
                target_tokens = set()
                for phrase in contract.get("target_phrases", []):
                    target_tokens.update(KnowMoreDiRTEngine._content_tokens(str(phrase)))
                for phrase in condition_phrases:
                    if constraint_field and constraint_value:
                        phrase_tokens = KnowMoreDiRTEngine._content_tokens(str(phrase))
                        if constraint_field in phrase_tokens or constraint_value in phrase_tokens:
                            target_tokens.discard(constraint_field)
                            target_tokens.discard(constraint_value)
                target_tokens.difference_update(
                    {
                        "many", "row", "rows", "entry", "entries", "have", "has",
                        "count", "number", "total", "status", "state", "active",
                        "archived", "blocked", "open", "closed", "ready", "paused",
                        "waiting", "draft", "released", "approved", "stable",
                    }
                )
                target_tokens.difference_update(
                    {
                        "contact", "contacts", "item", "items", "customer", "customers",
                        "person", "people", "file", "files", "artifact", "artifacts",
                        "ticket", "tickets", "issue", "issues",
                    }
                )
                search_terms = sorted(target_tokens)
                if not search_terms and constraint_value:
                    search_terms = [constraint_value]
                if search_terms and constraint_field and constraint_value:
                    normalized["steps"] = [
                        {
                            "tool": "search_records",
                            "inputs": [],
                            "collection": "all_records",
                            "terms": search_terms,
                            "fields": [],
                            "filters": [],
                            "arguments": [
                                {
                                    "name": "mode",
                                    "value": "all",
                                    "values": [],
                                    "numbers": [],
                                }
                            ],
                            "limit": 5000,
                        },
                        {
                            "tool": "filter_records",
                            "inputs": [0],
                            "collection": "",
                            "terms": [],
                            "fields": [],
                            "filters": [
                                {
                                    "field_path": constraint_field,
                                    "operator": "equals",
                                    "value": constraint_value,
                                    "values": [],
                                }
                            ],
                            "arguments": [],
                            "limit": 5000,
                        },
                        {
                            "tool": "aggregate_values",
                            "inputs": [1],
                            "collection": "",
                            "terms": [],
                            "fields": [],
                            "filters": [],
                            "arguments": [
                                {
                                    "name": "aggregate",
                                    "value": "count",
                                    "values": [],
                                    "numbers": [],
                                },
                                {
                                    "name": "distinct",
                                    "value": "false",
                                    "values": [],
                                    "numbers": [],
                                },
                            ],
                            "limit": 1,
                        },
                    ]
                elif search_terms:
                    counted_kind = " ".join(
                        token
                        for token in sorted(slot_tokens)
                        if token not in {"count", "number", "total"}
                    ) or "items"
                    normalized["steps"] = [
                        {
                            "tool": "search_records",
                            "inputs": [],
                            "collection": "all_records",
                            "terms": search_terms,
                            "fields": [],
                            "filters": [],
                            "arguments": [
                                {
                                    "name": "mode",
                                    "value": "all",
                                    "values": [],
                                    "numbers": [],
                                }
                            ],
                            "limit": 20,
                        },
                        {
                            "tool": "expand_source_context",
                            "inputs": [0],
                            "collection": "",
                            "terms": [],
                            "fields": [],
                            "filters": [],
                            "arguments": [],
                            "limit": 5000,
                        },
                        {
                            "tool": "model_extract",
                            "inputs": [1],
                            "collection": "",
                            "terms": [f"count explicit matching {counted_kind}"],
                            "fields": [],
                            "filters": [],
                            "arguments": [],
                            "limit": 1,
                        },
                    ]
            if superlative_count_request:
                condition_phrases = [
                    *contract.get("constraint_phrases", []),
                    *contract.get("relation_phrases", []),
                    *contract.get("scope_phrases", []),
                    *contract.get("target_phrases", []),
                    str(contract.get("question", "")),
                ]
                constraint_field = ""
                constraint_value = ""
                for phrase in condition_phrases:
                    text = str(phrase).lower()
                    match = re.search(
                        r"\b(status|state|phase|condition|category|type)\b"
                        r"\s*(?:is|are|was|were|equals?|=|:)?\s*([a-z0-9_-]+)",
                        text,
                    )
                    if match:
                        constraint_field = match.group(1)
                        constraint_value = match.group(2)
                        break
                    match = re.search(
                        r"\b(active|archived|blocked|open|closed|ready|paused|waiting|draft|released|approved|stable)\b"
                        r"\s+(?:row|rows|entry|entries|record|records)\b",
                        text,
                    )
                    if match:
                        constraint_field = "state"
                        constraint_value = match.group(1)
                        break
                if constraint_field and constraint_value:
                    normalized["steps"] = [
                        {
                            "tool": "search_records",
                            "inputs": [],
                            "collection": "all_records",
                            "terms": [constraint_value],
                            "fields": [],
                            "filters": [],
                            "arguments": [
                                {
                                    "name": "mode",
                                    "value": "all",
                                    "values": [],
                                    "numbers": [],
                                }
                            ],
                            "limit": 5000,
                        },
                        {
                            "tool": "filter_records",
                            "inputs": [0],
                            "collection": "",
                            "terms": [],
                            "fields": [],
                            "filters": [
                                {
                                    "field_path": constraint_field,
                                    "operator": "equals",
                                    "value": constraint_value,
                                    "values": [],
                                }
                            ],
                            "arguments": [],
                            "limit": 5000,
                        },
                        {
                            "tool": "project_values",
                            "inputs": [1],
                            "collection": "",
                            "terms": [],
                            "fields": [str(contract.get("answer_slot", ""))],
                            "filters": [],
                            "arguments": [
                                {
                                    "name": "distinct",
                                    "value": "false",
                                    "values": [],
                                    "numbers": [],
                                }
                            ],
                            "limit": 5000,
                        },
                        {
                            "tool": "aggregate_values",
                            "inputs": [2],
                            "collection": "",
                            "terms": [],
                            "fields": [],
                            "filters": [],
                            "arguments": [
                                {
                                    "name": "aggregate",
                                    "value": "mode",
                                    "values": [],
                                    "numbers": [],
                                }
                            ],
                            "limit": 1,
                        },
                    ]
        if contract is not None:
            first_search = next(
                (step for step in normalized["steps"] if step.get("tool") == "search_records"),
                None,
            )
            if first_search is not None:
                person_slot_tokens = set(
                    re.findall(
                        r"[a-z0-9]+",
                        str(contract.get("answer_slot", "")).lower().replace("_", " "),
                    )
                )
                if person_slot_tokens.intersection(
                    {
                        "owner", "reviewer", "approver", "author", "actor", "person",
                        "speaker", "inspector", "witness", "researcher", "doctor",
                        "teacher", "recipient", "sender",
                    }
                ):
                    first_search["limit"] = max(int(first_search.get("limit", 0)), 20)
                first_search["terms"] = [
                    term for term in first_search.get("terms", [])
                    if KnowMoreDiRTEngine._content_tokens(term)
                ]
                existing_text = " ".join(
                    [first_search.get("collection", ""), *first_search.get("terms", [])]
                )
                existing_tokens = KnowMoreDiRTEngine._content_tokens(existing_text)
                terms = list(first_search.get("terms", []))
                quantitative_scope_tokens = {
                    "how", "many", "much", "count", "number", "total", "quantity"
                }
                answer_slot_tokens = KnowMoreDiRTEngine._content_tokens(
                    str(contract.get("answer_slot", "")).replace("_", " ")
                )
                relation_scope_tokens = {
                    token
                    for phrase in contract.get("relation_phrases", [])
                    for token in KnowMoreDiRTEngine._content_tokens(str(phrase))
                }
                source_scope_tokens = {
                    "cache", "cached", "hidden", "file", "record", "semantic",
                    "meaningful", "despite", "ignore", "ignoring", "exclude",
                    "excluding", "official", "authoritative", "canonical", "verified",
                }
                for scope_phrase in contract.get("scope_phrases", []):
                    scope_tokens = KnowMoreDiRTEngine._content_tokens(scope_phrase)
                    raw_scope_tokens = set(
                        re.findall(r"[a-z0-9]+", str(scope_phrase).lower())
                    )
                    if scope_tokens and scope_tokens.issubset(quantitative_scope_tokens):
                        continue
                    if scope_tokens and scope_tokens.issubset(answer_slot_tokens):
                        continue
                    if scope_tokens and scope_tokens.issubset(relation_scope_tokens):
                        continue
                    if (
                        contract.get("source_scope") not in {None, "", "any", "unknown"}
                        and raw_scope_tokens.intersection(source_scope_tokens)
                    ):
                        continue
                    distinctive = {
                        token for token in scope_tokens
                        if token not in {"text", "record", "document", "file", "note", "data"}
                    } or scope_tokens
                    if distinctive and not existing_tokens.intersection(distinctive):
                        terms.append(scope_phrase)
                        existing_tokens.update(scope_tokens)
                first_search["terms"] = terms
        generic_fields = {"text", "source.path", "source.file_name", "source.file_stem"}
        for index, compact_step in enumerate(normalized["steps"]):
            expanded = expand_step(compact_step)
            operation_aliases = {
                "addition": "add", "plus": "add", "sum": "add",
                "subtraction": "subtract", "minus": "subtract", "difference": "subtract",
                "multiplication": "multiply", "times": "multiply", "product": "multiply",
                "division": "divide", "quotient": "divide",
            }
            if expanded["tool"] == "calculate" and expanded["operation"] in operation_aliases:
                canonical = operation_aliases[expanded["operation"]]
                arguments = [dict(argument) for argument in compact_step.get("arguments", [])]
                found = False
                for argument in arguments:
                    if argument.get("name") == "operation":
                        argument["value"] = canonical
                        found = True
                if not found:
                    arguments.append(
                        {
                            "name": "operation",
                            "value": canonical,
                            "values": [],
                            "numbers": list(expanded["numbers"]),
                        }
                    )
                normalized["steps"][index] = {**compact_step, "arguments": arguments}
                expanded = expand_step(normalized["steps"][index])
            if expanded["tool"] == "join_records" and len(expanded["inputs"]) == 1:
                normalized["steps"][index] = {
                    "tool": "expand_source_context",
                    "inputs": list(expanded["inputs"]),
                    "collection": "",
                    "terms": [],
                    "fields": [],
                    "filters": [],
                    "arguments": [],
                    "limit": max(20, expanded["limit"]),
                }
                expanded = expand_step(normalized["steps"][index])
            identifier_slot_tokens = set()
            if contract is not None:
                identifier_slot_tokens = set(
                    re.findall(
                        r"[a-z0-9]+",
                        str(contract.get("answer_slot", "")).lower().replace("_", " "),
                    )
                )
            temporal_role_extraction = bool(
                contract is not None
                and expanded["tool"] == "extract_values"
                and expanded["extractor"] == "date_time"
                and contract.get("semantic_kind") in {"event_fact", "entity_attribute"}
                and (
                    contract.get("relation_phrases")
                    or contract.get("scope_phrases")
                    or contract.get("target_phrases")
                )
            )
            target_bound_locator_extraction = bool(
                contract is not None
                and contract.get("semantic_kind") == "entity_attribute"
                and expanded["tool"] == "extract_values"
                and (
                    (
                        expanded["extractor"] == "url"
                        and identifier_slot_tokens.intersection(
                            {"url", "uri", "link", "report", "manual", "runbook", "guide", "warranty"}
                        )
                    )
                    or (
                        expanded["extractor"] == "identifier"
                        and identifier_slot_tokens.intersection(
                            {"id", "identifier", "code", "reference", "account"}
                        )
                    )
                )
            )
            if (
                temporal_role_extraction
                or target_bound_locator_extraction
                or (
                    contract is not None
                    and contract.get("semantic_kind") == "entity_attribute"
                    and expanded["tool"] == "extract_values"
                    and expanded["extractor"] in {
                        "after_label", "after_phrase", "before_phrase", "between_phrases", "regex"
                    }
                    and identifier_slot_tokens.intersection(
                        {"id", "identifier", "code", "reference", "account"}
                    )
                )
            ):
                normalized["steps"][index] = {
                    "tool": "model_extract",
                    "inputs": list(expanded["inputs"]),
                    "collection": expanded["collection"],
                    "terms": list(expanded["terms"]),
                    "fields": list(expanded["fields"]),
                    "filters": [],
                    "arguments": [],
                    "limit": expanded["limit"],
                }
                expanded = expand_step(normalized["steps"][index])
            if (
                expanded["tool"] == "extract_values"
                and expanded["extractor"] == "field"
                and set(expanded["fields"]).issubset(generic_fields)
            ):
                normalized["steps"][index] = {
                    "tool": "model_extract",
                    "inputs": list(expanded["inputs"]),
                    "collection": expanded["collection"],
                    "terms": list(expanded["terms"]),
                    "fields": list(expanded["fields"]),
                    "filters": [],
                    "arguments": [],
                    "limit": expanded["limit"],
                }
            if normalized["steps"][index].get("tool") == "model_extract":
                if contract is not None and contract.get("answer_shape") == "list":
                    normalized["steps"][index]["limit"] = max(
                        int(normalized["steps"][index].get("limit", 0)),
                        20,
                    )
                normalized["steps"] = normalized["steps"][: index + 1]
                break
        if (
            contract is not None
            and contract.get("semantic_kind") == "entity_attribute"
            and not count_request
            and not superlative_count_request
            and any(step.get("tool") == "model_extract" for step in normalized["steps"])
        ):
            first_search = next(
                (step for step in normalized["steps"] if step.get("tool") == "search_records"),
                None,
            )
            if first_search is not None:
                target_tokens = set()
                for phrase in contract.get("target_phrases", []):
                    target_tokens.update(KnowMoreDiRTEngine._content_tokens(str(phrase)))
                for phrase in contract.get("scope_phrases", []):
                    target_tokens.update(KnowMoreDiRTEngine._content_tokens(str(phrase)))
                relation_tokens = set()
                for phrase in contract.get("relation_phrases", []):
                    relation_tokens.update(KnowMoreDiRTEngine._content_tokens(str(phrase)))
                slot_tokens = set(
                    re.findall(
                        r"[a-z0-9]+",
                        str(contract.get("answer_slot", "")).lower().replace("_", " "),
                    )
                )
                target_tokens.difference_update(relation_tokens | slot_tokens)
                target_tokens.difference_update(
                    {
                        "owns", "owned", "owner", "reviewer", "reviewed",
                        "identify", "identifies", "identified", "identifier",
                        "belong", "belongs", "belonged", "associate", "associated",
                        "attribute", "value", "cache", "cached", "hidden", "official",
                        "semantic", "record", "file", "despite", "ignore", "ignoring",
                    }
                )
                if target_tokens:
                    first_search["terms"] = sorted(target_tokens)
                first_search["collection"] = "all_records"
                first_search["fields"] = []
                first_search["arguments"] = [
                    {
                        "name": "mode",
                        "value": "all",
                        "values": [],
                        "numbers": [],
                    }
                ]
                first_search["limit"] = max(int(first_search.get("limit", 0)), 20)
        return normalized

    def _valid_collection(self, collection: str) -> bool:
        if not collection:
            return True
        if collection in {"all_records", "all_representations"}:
            return True
        if collection in self.catalog.collections:
            return True
        return bool(self.catalog.collection_records(collection))

    @staticmethod
    def _content_tokens(value: str) -> set[str]:
        ignored = {
            "a", "an", "the", "in", "on", "at", "of", "for", "to", "from", "with",
            "who", "what", "which", "where", "when", "how", "is", "are", "was", "were", "did", "does", "do",
            "this", "that", "provided", "mentioned", "listed", "shown", "stated", "corpus",
            "folder", "collection", "source", "sources", "record", "records",
            "semantic", "meaningful", "credible",
            "reliable", "authoritative", "trustworthy", "trusted", "valid", "clean",
            "recorded", "described", "called", "named", "mean", "means", "translated",
            "translation", "stored", "located", "kept", "found", "location",
            "really", "actually", "factually", "truly", "real", "about", "regarding",
            "concerning", "according", "per", "says", "said", "believes", "believed",
            "reported", "wrote", "written", "forwarded", "quoted",
            "row", "rows", "entry", "entries",
            "after", "before", "during", "since", "until", "following", "preceding",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.lower())
            if token not in ignored and len(token) > 1
        }

    @staticmethod
    def _is_definition_request(contract: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(contract.get("answer_slot", "")),
                str(contract.get("intent_summary", "")),
                *[str(item) for item in contract.get("relation_phrases", [])],
            ]
        ).lower()
        return bool(
            re.search(
                r"\b(?:meaning|definition|translation|translate|translated)\b",
                text,
            )
        )

    @staticmethod
    def _has_explicit_definition_evidence(
        contract: dict[str, Any],
        values: list[str],
        records: list[Any],
    ) -> bool:
        if not values or not records:
            return False
        targets: list[str] = []
        for item in contract.get("target_phrases", []):
            phrase = str(item).strip().lower()
            if not phrase:
                continue
            targets.append(phrase)
            targets.extend(sorted(KnowMoreDiRTEngine._content_tokens(phrase)))
        targets = list(dict.fromkeys(targets))
        word_relation = r"(?:means?|meaning(?:\s+is)?|definition(?:\s+is)?|translat(?:es?|ion)(?:\s+is|\s+to)?|refers?\s+to)"
        for record in records:
            text = (
                str(getattr(record, "text", ""))
                + "\n"
                + json.dumps(getattr(record, "data", {}), ensure_ascii=False, default=str)
            ).lower()
            for target in targets:
                if target not in text:
                    continue
                for value in values:
                    value_text = str(value).strip().lower()
                    if not value_text or value_text not in text:
                        continue
                    target_pattern = re.escape(target)
                    value_pattern = re.escape(value_text)
                    if re.search(
                        rf"{target_pattern}.{{0,80}}{word_relation}.{{0,80}}{value_pattern}",
                        text,
                        flags=re.IGNORECASE | re.DOTALL,
                    ):
                        return True
                    if re.search(
                        rf"{target_pattern}\s*(?::=|=|:|—|-)\s*[\"']?{value_pattern}",
                        text,
                        flags=re.IGNORECASE,
                    ):
                        return True
                    if re.search(
                        rf"(?:meaning|definition|translation).{{0,80}}{value_pattern}",
                        text,
                        flags=re.IGNORECASE | re.DOTALL,
                    ):
                        return True
        return False

    @staticmethod
    def _expand_temporal_values(
        values: list[str],
        records: list[Any],
        temporal_mode: str,
    ) -> list[str]:
        if temporal_mode != "at_time" or not values or not records:
            return values
        evidence_text = "\n".join(str(getattr(record, "text", "")) for record in records)
        full_datetimes = re.findall(
            r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?\b",
            evidence_text,
        )
        output: list[str] = []
        for value in values:
            stripped = value.strip()
            if re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?", stripped):
                matches = [item for item in full_datetimes if item.endswith(stripped)]
                output.append(matches[0] if len(matches) == 1 else value)
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
                matches = [item for item in full_datetimes if item.startswith(stripped)]
                output.append(matches[0] if len(matches) == 1 else value)
            else:
                output.append(value)
        return output

    @classmethod
    def _target_mixed_epistemic_evidence(
        cls,
        record: Any,
        contract: dict[str, Any],
    ) -> bool:
        target_tokens = cls._contract_target_tokens(contract)
        if not target_tokens:
            return False
        fiction_markers = {
            "fiction", "fictional", "fantasy", "dream", "dreamed", "hypothetical",
            "imagined", "imaginary", "made-up", "myth", "legend", "story",
        }
        reality_markers = {
            "in reality", "real-world", "when i woke", "woke up", "still exists",
            "still existed", "still contains", "still contained", "verified", "confirmed",
            "inspection", "incident report", "actual state", "observed state",
        }
        text = str(getattr(record, "text", "") or "")
        segments = [
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+|\n+", text)
            if item.strip()
        ]
        fiction_targets: list[set[str]] = []
        reality_targets: list[set[str]] = []
        for segment in segments:
            lowered = segment.lower()
            overlap = cls._content_tokens(segment).intersection(target_tokens)
            if not overlap:
                continue
            if any(marker in lowered for marker in fiction_markers):
                fiction_targets.append(overlap)
            if any(marker in lowered for marker in reality_markers):
                reality_targets.append(overlap)
        return any(left.intersection(right) for left in fiction_targets for right in reality_targets)

    @staticmethod
    def _mixed_epistemic_evidence(records: list[Any]) -> bool:
        fiction_markers = {
            "fiction", "fictional", "fantasy", "dream", "dreamed", "hypothetical",
            "imagined", "imaginary", "made-up", "myth", "legend", "story",
        }
        reality_markers = {
            "in reality", "real-world", "when i woke", "woke up", "still exists",
            "still existed", "still contains", "still contained", "verified", "confirmed",
            "inspection", "incident report", "actual state", "observed state",
        }
        for record in records:
            text = (
                str(getattr(record, "text", ""))
                + " "
                + json.dumps(getattr(record, "data", {}), ensure_ascii=False, default=str)
            ).lower()
            if (
                any(marker in text for marker in fiction_markers)
                and any(marker in text for marker in reality_markers)
            ):
                return True
        return False

    @staticmethod
    def _fiction_only_boolean_evidence(records: list[Any]) -> bool:
        if not records:
            return False
        fiction_markers = {
            "fiction", "fictional", "fantasy", "dream", "dreamed", "hypothetical",
            "imagined", "imaginary", "made-up", "myth", "legend", "story",
        }
        reality_markers = {
            "in reality", "real-world", "when i woke", "woke up", "still exists",
            "still existed", "still contains", "still contained", "verified", "confirmed",
            "inspection", "incident report", "actual state", "observed state",
        }
        def fiction_only(record: Any) -> bool:
            text = (
                str(getattr(record, "text", ""))
                + " "
                + json.dumps(getattr(record, "data", {}), ensure_ascii=False, default=str)
            ).lower()
            return (
                any(marker in text for marker in fiction_markers)
                and not any(marker in text for marker in reality_markers)
            )
        return all(fiction_only(record) for record in records)

    @staticmethod
    def _relation_stem(token: str) -> str:
        value = str(token).lower()
        aliases = {
            "status": "state",
            "condition": "state",
            "phase": "state",
            "own": "own",
            "owns": "own",
            "owned": "own",
            "owner": "own",
            "ownership": "own",
            "bought": "buy",
            "purchase": "buy",
            "purchased": "buy",
            "purchases": "buy",
            "buying": "buy",
            "found": "find",
            "finding": "find",
        }
        if value in aliases:
            return aliases[value]
        for suffix in ("ations", "ation", "ions", "ion", "ing", "ers", "ors", "ed", "er", "or", "es", "s"):
            if value.endswith(suffix) and len(value) - len(suffix) >= 4:
                value = value[: -len(suffix)]
                break
        return aliases.get(value, value)

    @classmethod
    def _entity_relation_stems(cls, contract: dict[str, Any]) -> set[str]:
        phrases = [
            *[str(item) for item in contract.get("relation_phrases", [])],
            *[str(item) for item in contract.get("constraint_phrases", [])],
        ]
        ignored = {
            "not", "no", "without", "except", "excluding", "exclude",
            "with", "to", "from", "by", "of", "for", "in", "on", "at",
            "into", "about", "according", "per", "the", "a", "an", "and",
            "or", "is", "are", "was", "were", "be", "been", "being",
            "did", "does", "do", "has", "have", "had", "should", "would",
            "could", "can", "may", "might", "must",
            "named", "mentioned", "listed", "shown", "stated", "provided",
            "recorded", "described", "identified",
        }
        relation_tokens: set[str] = set()
        for phrase in phrases:
            raw_tokens = re.findall(r"[A-Za-z0-9]+", phrase)
            for index, raw in enumerate(raw_tokens):
                token = raw.lower()
                if token in ignored:
                    continue
                # Later capitalized tokens in a relation phrase are normally named
                # arguments (for example, "merged with Morgan Hale"), not predicates.
                if index > 0 and raw[:1].isupper():
                    continue
                relation_tokens.add(token)
        if relation_tokens:
            return {cls._relation_stem(token) for token in relation_tokens if token}

        slot_tokens = {
            token
            for token in re.findall(
                r"[a-z0-9]+",
                str(contract.get("answer_slot", "")).lower().replace("_", " "),
            )
            if token not in {
                "person", "name", "value", "answer", "content", "entity",
                "what", "who", "where", "when", "which", "how",
                "current", "final", "latest", "earliest", "present", "active",
            }
        }
        return {cls._relation_stem(token) for token in slot_tokens if token}

    @classmethod
    def _value_has_explicit_entity_relation(
        cls,
        contract: dict[str, Any],
        value: str,
        evidence_views: list[dict[str, Any]],
    ) -> bool:
        if contract.get("semantic_kind") != "entity_attribute":
            return True
        relation_stems = cls._entity_relation_stems(contract)
        if not relation_stems:
            return True
        normalized_value = re.sub(r"\s+", " ", str(value).strip().lower())
        if not normalized_value:
            return False
        slot_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                str(contract.get("answer_slot", "")).lower().replace("_", " "),
            )
        )
        locator_slot = bool(
            slot_tokens.intersection(
                {"location", "storage", "url", "uri", "path", "address", "directory", "shelf", "room"}
            )
        )
        locator_value = bool(
            re.match(r"^(?:https?://|file://|/|[a-z]:\\)", normalized_value)
        )
        spatial_slot = bool(
            slot_tokens.intersection(
                {"location", "place", "position", "where", "room", "shelf"}
            )
        )
        target_tokens = cls._contract_target_tokens(contract)
        slot_label_tokens = [
            token
            for token in re.findall(
                r"[a-z0-9]+",
                str(contract.get("answer_slot", "")).lower().replace("_", " "),
            )
            if token not in {"value", "answer", "content", "text"}
        ]
        slot_label_pattern = (
            r"[ _-]+".join(re.escape(token) for token in slot_label_tokens)
            if slot_label_tokens
            else ""
        )
        for view in evidence_views:
            text = (
                str(view.get("excerpt", ""))
                + "\n"
                + str(view.get("text", ""))
                + "\n"
                + json.dumps(view.get("data", {}), ensure_ascii=False, default=str)
            ).lower()
            if slot_label_pattern:
                labeled_values = [
                    (
                        re.sub(r"\s+", " ", match.group("label").strip().lower()),
                        re.sub(r"\s+", " ", match.group("value").strip().lower()),
                    )
                    for match in re.finditer(
                        r"(?im)^(?P<label>[a-z][a-z0-9 _./-]{0,79})\s*[:=]\s*(?P<value>[^\n]+)$",
                        text,
                    )
                ]
                matching_slot_values = [
                    field_value
                    for label, field_value in labeled_values
                    if re.fullmatch(slot_label_pattern, label, flags=re.IGNORECASE)
                ]
                if matching_slot_values:
                    return any(
                        normalized_value == re.sub(r"[.;:]+$", "", field_value).strip()
                        or normalized_value in field_value
                        for field_value in matching_slot_values
                    )
            if locator_slot and locator_value and normalized_value in text:
                value_index = text.find(normalized_value)
                locator_window = text[max(0, value_index - 220) : value_index + len(normalized_value) + 120]
                window_tokens = cls._content_tokens(locator_window)
                required_target_overlap = max(1, min(2, len(target_tokens)))
                if len(window_tokens.intersection(target_tokens)) >= required_target_overlap:
                    return True
            if spatial_slot and normalized_value in text:
                required_target_overlap = max(1, min(2, len(target_tokens)))
                for sentence in [
                    item.strip()
                    for item in re.split(r"(?<=[.!?])\s+|\n+", text)
                    if item.strip()
                ]:
                    if normalized_value not in sentence:
                        continue
                    if len(cls._content_tokens(sentence).intersection(target_tokens)) < required_target_overlap:
                        continue
                    spatial_binding = bool(
                        re.search(
                            r"\b(?:is|are|was|were|sits?|stands?|lies?|located|kept|stored|placed)\b"
                            r".{0,80}\b(?:on|in|at|under|over|above|below|behind|beside|near|inside|outside|left|right|between|within)\b",
                            sentence,
                            flags=re.IGNORECASE,
                        )
                        or re.match(
                            r"^(?:on|in|at|under|over|above|below|behind|beside|near|inside|outside|left|right|between|within)\b",
                            normalized_value,
                            flags=re.IGNORECASE,
                        )
                    )
                    if spatial_binding:
                        return True
            start = 0
            while True:
                index = text.find(normalized_value, start)
                if index < 0:
                    break
                window = text[max(0, index - 140) : index + len(normalized_value) + 140]
                window_stems = {
                    cls._relation_stem(token)
                    for token in re.findall(r"[a-z0-9]+", window)
                }
                if relation_stems & window_stems:
                    return True
                start = index + max(1, len(normalized_value))
        return False

    @staticmethod
    def _contract_asks_proof_status(contract: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(contract.get("question", "")),
                str(contract.get("answer_slot", "")).replace("_", " "),
                *[str(item) for item in contract.get("constraint_phrases", [])],
                *[str(item) for item in contract.get("relation_phrases", [])],
            ]
        ).lower()
        return bool(re.search(r"\b(?:proof|prove|proved|proven|established|confirmed)\b", text))

    @classmethod
    def _proof_status_correction_sentence(
        cls,
        contract: dict[str, Any],
        records: list[Any],
    ) -> str:
        if not cls._contract_asks_proof_status(contract):
            return ""
        text = "\n".join(str(getattr(record, "text", "")) for record in records)
        if not re.search(r"(?i)\b(?:no|without)\s+proof\b|\bnot\s+proven\b", text):
            return ""
        if re.search(r"(?i)\bfinal\s+judgment\b", text):
            return "the final judgment found no proof"
        if re.search(r"(?i)\b(?:court|tribunal|panel)\b", text):
            return "the court found no proof"
        return "the evidence contained no proof"

    @staticmethod
    def _reason_is_nonproof(reason: str) -> bool:
        text = re.sub(r"\s+", " ", str(reason).strip().lower())
        markers = (
            "not proven", "unproven", "not confirmed", "unconfirmed",
            "no proof", "insufficient proof", "insufficient evidence",
            "lack of proof", "lack of evidence", "not established as fact",
            "no decision was made", "no final decision", "not decided",
            "undecided", "decision pending", "pending decision",
            "no confirmation", "not adopted as a plan",
        )
        if any(marker in text for marker in markers):
            return True
        return bool(
            re.search(
                r"\bno\b.{0,80}\b(?:decision|confirmation|approval|determination)\b"
                r".{0,30}\b(?:was|were|has been|had been)?\s*(?:made|reached|given|recorded|issued|confirmed)?\b",
                text,
            )
        )

    @staticmethod
    def _reason_explicit_false(reason: str) -> bool:
        text = re.sub(r"\s+", " ", str(reason).strip().lower())
        markers = (
            "did not occur", "didn't occur", "did not happen", "didn't happen",
            "did not take place", "not occur in reality", "never occurred",
            "event was false", "claim was false", "proposition is false",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _unknown_like_value(value: str) -> bool:
        text = re.sub(r"\s+", " ", str(value).strip().lower())
        markers = {
            "unknown", "not known", "unavailable", "not available",
            "has no stated translation", "no stated translation",
            "no translation is stated", "not stated", "not specified",
            "not provided", "cannot be determined", "insufficient evidence",
        }
        if text in markers or any(marker in text for marker in markers if len(marker) > 8):
            return True
        return bool(
            re.search(
                r"\bno\b.{0,100}\b(?:is|are|was|were)\s+(?:stated|provided|specified|listed|recorded|given|available)\b",
                text,
            )
        )

    def _validate_program(self, contract: dict[str, Any], program: dict[str, Any]) -> None:
        if program["contract_id"] != contract["contract_id"]:
            raise ProgramValidationError("program contract mismatch")
        steps = program["steps"]
        if not steps:
            raise ProgramValidationError("query program has no steps")
        first_search = next(
            (expand_step(step) for step in steps if step.get("tool") == "search_records"),
            None,
        )
        if first_search is not None and contract["scope_phrases"]:
            retrieval_text = " ".join(
                [first_search["collection"], *first_search["terms"], *first_search["fields"]]
            )
            retrieval_tokens = self._content_tokens(retrieval_text)
            relation_tokens = {
                token
                for phrase in contract.get("relation_phrases", [])
                for token in self._content_tokens(str(phrase))
            }
            answer_slot_tokens = self._content_tokens(
                str(contract.get("answer_slot", "")).replace("_", " ")
            )
            polarity_markers = {"not", "no", "without", "except", "excluding", "exclude"}
            quantitative_scope_tokens = {
                "how", "many", "much", "count", "number", "total", "quantity"
            }
            source_scope_tokens = {
                "cache", "cached", "hidden", "file", "record", "semantic",
                "meaningful", "despite", "ignore", "ignoring", "exclude",
                "excluding", "official", "authoritative", "canonical", "verified",
            }
            for scope_phrase in contract["scope_phrases"]:
                scope_tokens = self._content_tokens(scope_phrase)
                raw_scope_tokens = set(re.findall(r"[a-z0-9]+", str(scope_phrase).lower()))
                semantic_operator_scope = bool(
                    (scope_tokens and scope_tokens.issubset(relation_tokens))
                    or (scope_tokens and scope_tokens.issubset(answer_slot_tokens))
                    or (scope_tokens and scope_tokens.issubset(quantitative_scope_tokens))
                    or (
                        contract.get("source_scope") not in {None, "", "any", "unknown"}
                        and raw_scope_tokens.intersection(source_scope_tokens)
                    )
                    or (
                        contract.get("polarity") == "negative"
                        and raw_scope_tokens.intersection(polarity_markers)
                    )
                )
                if semantic_operator_scope:
                    continue
                distinctive = {
                    token for token in scope_tokens
                    if token not in {"text", "record", "document", "file", "note", "data"}
                } or scope_tokens
                if distinctive and not retrieval_tokens.intersection(distinctive):
                    raise ProgramValidationError(
                        f"primary retrieval omitted semantic scope phrase {scope_phrase!r}"
                    )
        for index, compact_step in enumerate(steps):
            step = expand_step(compact_step)
            for ref in step["inputs"]:
                if ref < 0 or ref >= index:
                    raise ProgramValidationError(f"step {index} references unavailable prior step {ref}")
            if not self._valid_collection(step["collection"]):
                raise ProgramValidationError(f"unknown collection: {step['collection']}")
            if step["limit"] < 0:
                raise ProgramValidationError("step limit cannot be negative")
            tool = step["tool"]
            valid_modes = {"none", "all", "any", "phrase"}
            valid_directions = {"none", "ascending", "descending"}
            valid_aggregates = {"none", "count", "min", "max", "sum", "average", "distinct", "mode"}
            valid_operations = {"none", "add", "subtract", "multiply", "divide"}
            valid_extractors = {
                "none", "field", "after_label", "after_phrase", "before_phrase",
                "between_phrases", "regex", "url", "identifier", "date_time",
                "number", "event_series",
            }
            valid_occurrences = {
                "none", "first", "last", "all", "latest_by_time", "earliest_by_time",
            }
            if step["mode"] not in valid_modes:
                raise ProgramValidationError(f"step {index} has invalid mode {step['mode']!r}")
            if step["direction"] not in valid_directions:
                raise ProgramValidationError(f"step {index} has invalid direction {step['direction']!r}")
            if step["aggregate"] not in valid_aggregates:
                raise ProgramValidationError(f"step {index} has invalid aggregate {step['aggregate']!r}")
            if step["operation"] not in valid_operations:
                raise ProgramValidationError(f"step {index} has invalid operation {step['operation']!r}")
            if step["extractor"] not in valid_extractors:
                raise ProgramValidationError(f"step {index} has invalid extractor {step['extractor']!r}")
            if step["occurrence"] not in valid_occurrences:
                raise ProgramValidationError(f"step {index} has invalid occurrence {step['occurrence']!r}")
            if tool == "search_records":
                if not step["terms"]:
                    raise ProgramValidationError(f"search step {index} has no terms")
                if step["mode"] == "none":
                    raise ProgramValidationError(f"search step {index} has no match mode")
                if step["mode"] == "phrase" and len(step["terms"]) != 1:
                    raise ProgramValidationError(
                        f"phrase search step {index} must contain one complete phrase"
                    )
                if index == 0 and len(step["terms"]) > 1 and step["mode"] == "any" and contract["target_phrases"]:
                    raise ProgramValidationError(
                        "the primary target search has multiple terms and must use all or phrase matching"
                    )
            elif tool == "expand_source_context" and not step["inputs"]:
                raise ProgramValidationError(f"context expansion step {index} has no input")
            elif tool == "filter_records" and (not step["inputs"] or not step["filters"]):
                raise ProgramValidationError(f"filter step {index} is incomplete")
            elif tool == "project_values" and (not step["inputs"] or not step["fields"]):
                raise ProgramValidationError(f"projection step {index} is incomplete")
            elif tool == "extract_values":
                if not step["inputs"] or step["extractor"] == "none":
                    raise ProgramValidationError(f"extraction step {index} is incomplete")
                extractor = step["extractor"]
                if extractor == "field" and not step["fields"]:
                    raise ProgramValidationError(f"field extraction step {index} has no fields")
                if extractor == "field" and step["fields"]:
                    generic_fields = {"text", "source.path", "source.file_name", "source.file_stem"}
                    if set(step["fields"]).issubset(generic_fields):
                        slot_tokens = set(re.findall(r"[a-z0-9]+", contract["answer_slot"].lower()))
                        field_tokens = set(
                            token
                            for field in step["fields"]
                            for token in re.findall(r"[a-z0-9]+", field.lower())
                        )
                        if not slot_tokens.intersection(field_tokens):
                            raise ProgramValidationError(
                                f"field extraction step {index} projects generic text instead of the answer slot; use a structured answer field or a text extractor"
                            )
                if extractor == "after_label" and not step["label"]:
                    raise ProgramValidationError(f"label extraction step {index} has no label")
                if extractor in {"regex", "event_series"} and not step["pattern"]:
                    raise ProgramValidationError(f"regex extraction step {index} has no pattern")
                if extractor == "event_series" and (
                    not step["value_group"]
                    or not step["time_group"]
                    or step["occurrence"] not in {"latest_by_time", "earliest_by_time", "all"}
                ):
                    raise ProgramValidationError(f"event extraction step {index} is incomplete")
                if (
                    contract["answer_shape"] == "number"
                    and index == len(steps) - 1
                    and extractor != "none"
                ):
                    raise ProgramValidationError(
                        "evidence-backed numeric or measured answers must use model_extract so source units are preserved; use calculate only for arithmetic"
                    )
            elif tool == "model_extract":
                if not step["inputs"]:
                    raise ProgramValidationError(f"model extraction step {index} has no evidence input")
                if index != len(steps) - 1:
                    raise ProgramValidationError(
                        f"model extraction step {index} must be the final answer-producing step"
                    )
            elif tool == "join_records" and (
                len(step["inputs"]) != 2 or not step["left_field"] or not step["right_field"]
            ):
                raise ProgramValidationError(f"join step {index} is incomplete")
            elif tool == "calculate" and step["operation"] == "none":
                raise ProgramValidationError(f"calculate step {index} has no operation")

    @staticmethod
    def _result_has_material(result: ToolResult) -> bool:
        if result.kind == "records":
            return bool(result.records)
        if result.kind == "values":
            return bool(result.values)
        if result.kind == "scalar":
            return result.scalar is not None
        return bool(result.records or result.values or result.scalar is not None)

    def _needs_execution_repair(
        self,
        program: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> bool:
        final_result = results[len(program["steps"]) - 1]
        if final_result.diagnostics.get("status") == "unknown":
            return final_result.diagnostics.get("reason") == "no input material"
        return not self._result_has_material(final_result)

    def _repair_program_after_execution(
        self,
        profile: dict[str, Any],
        contract: dict[str, Any],
        program: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> dict[str, Any]:
        diagnostics = [
            {
                "step": index,
                "tool": step["tool"],
                "collection": step["collection"],
                "record_count": len(results[index].records),
                "value_count": len(results[index].values),
                "has_scalar": results[index].scalar is not None,
                "diagnostics": results[index].diagnostics,
            }
            for index, step in enumerate(program["steps"])
        ]
        schema = query_program_schema(contract["contract_id"])
        prompt = (
            'Repair a valid generic tool program whose final step produced no material. Return a complete replacement, '
            'not an answer. Preserve the immutable semantic contract. When a narrow collection is empty or ambiguous, '
            'search all_records with separate literal target and relation terms using all matching. Remove unsupported '
            'filters and fields. Never substitute an unrelated same-named field. Use project_values only for real answer '
            'fields, extract_values with explicit extractor arguments, or model_extract over the retrieved evidence.\n'
            f"Rejected program: {json.dumps(program, ensure_ascii=False)}\n"
            f"Execution diagnostics: {json.dumps(diagnostics, ensure_ascii=False, default=str)}\n"
            f"Dataset profile: {json.dumps(profile, ensure_ascii=False)}\n"
            f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
            f"Catalog: {self.catalog.summary(6500)}"
        )
        payload = self.model.complete_json(
            "query_program_execution_repair",
            prompt,
            schema,
            max_tokens=1536,
        )
        repaired = self._normalize_program(payload["query_program"], contract)
        self._validate_program(contract, repaired)
        self.model_query_trace = {
            "dataset_profile": profile,
            "semantic_contract": contract,
            "program": repaired,
            "repaired_after_empty_execution": True,
            "dataset_fingerprint": self.catalog.fingerprint,
        }
        return repaired

    def _fallback_model_extract_program(
        self,
        program: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> dict[str, Any] | None:
        """Use bounded semantic extraction when deterministic extraction failed over real evidence."""
        evidence_index: int | None = None
        for index in range(len(program["steps"]) - 1, -1, -1):
            if results[index].records:
                evidence_index = index
                break
        if evidence_index is None:
            return None
        steps = list(program["steps"][: evidence_index + 1])
        steps.append(
            {
                "tool": "model_extract",
                "inputs": [evidence_index],
                "collection": "",
                "terms": [],
                "fields": [],
                "filters": [],
                "arguments": [],
                "limit": 20,
            }
        )
        fallback = {"contract_id": program["contract_id"], "steps": steps}
        self.model_query_trace = {
            **self.model_query_trace,
            "program": fallback,
            "fallback_to_model_extract": True,
        }
        return fallback

    @staticmethod
    def _needs_list_cardinality_fallback(
        contract: dict[str, Any],
        program: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> bool:
        if contract.get("answer_shape") != "list":
            return False
        final_index = len(program["steps"]) - 1
        final_step = expand_step(program["steps"][final_index])
        final_result = results[final_index]
        if final_step["tool"] == "model_extract":
            return False
        material_count = len(final_result.values)
        if final_result.scalar is not None:
            material_count = 1
        return material_count <= 1

    def _dependency_closure(
        self,
        program: dict[str, Any],
        final_index: int,
    ) -> list[int]:
        selected: set[int] = set()
        pending = [final_index]
        while pending:
            index = pending.pop()
            if index in selected:
                continue
            selected.add(index)
            pending.extend(program["steps"][index]["inputs"])
        return sorted(selected)

    @staticmethod
    def _format_list_values(values: list[str]) -> str:
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        identifier_pattern = re.compile(r"^[A-Z]{2,}(?:-[A-Z0-9]+)+$")
        if all(identifier_pattern.fullmatch(value) for value in cleaned):
            return "; ".join(cleaned)
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"

    def _direct_structural_answer(
        self,
        contract: dict[str, Any],
        program: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> Answer | None:
        final_index = len(program["steps"]) - 1
        final_step = expand_step(program["steps"][final_index])
        final = results[final_index]
        answer_shape = contract["answer_shape"]
        if final_step["tool"] == "model_extract" and final.diagnostics.get("status") == "unknown":
            evidence = tuple(record.model_view() for record in final.records)
            return Answer(
                "unknown",
                evidence=evidence,
                diagnostics={
                    "derivation": "contract_bound_model_extraction_unknown",
                    "trace": self.model_query_trace,
                },
            )
        corrective_sentence = str(final.diagnostics.get("corrective_sentence", "")).strip()
        if (
            final_step["tool"] == "model_extract"
            and answer_shape == "boolean"
            and corrective_sentence
            and any(str(value).strip().lower() in {"no", "false"} for value in final.values)
        ):
            correction = re.sub(r"^\[[^\]]+\]\s*", "", corrective_sentence).strip()
            correction = re.sub(
                r"(?i)^(?:no|false)\s*(?:[;:,.!?-]+\s*|$)",
                "",
                correction,
            ).strip()
            if contract.get("semantic_kind") == "source_classification":
                correction = re.sub(
                    r"(?i)^(?:(?:the\s+)?(?:teacher\s+note|source|document|note|record))\s+"
                    r"(?:indicates|states|says|reports|classifies|labels|marks|shows)\s+(?:that\s+)?",
                    "",
                    correction,
                    count=1,
                ).strip()
            document_classification = self._direct_document_classification_correction(
                contract,
                [record.model_view() for record in final.records],
            )
            if document_classification:
                correction = document_classification
            correction = self._normalize_contract_bound_correction_surface(
                contract,
                correction,
            )
            declared_target_tokens = {
                token
                for phrase in contract.get("target_phrases", [])
                for token in re.findall(r"[a-z0-9]+", str(phrase).lower())
            }
            for subject_type in (
                "runtime", "system", "service", "process", "code", "application", "worker", "job"
            ):
                if subject_type in declared_target_tokens:
                    correction = re.sub(
                        rf"(?i)^the\s+{re.escape(subject_type)}\b",
                        subject_type,
                        correction,
                        count=1,
                    ).strip()
                    break
            if correction:
                correction = correction[0].lower() + correction[1:]
            if not correction.endswith((".", "!", "?")):
                correction += "."
            evidence_by_id = {
                record.record_id: record.model_view()
                for index in self._dependency_closure(program, final_index)
                for record in results[index].records
            }
            return Answer(
                f"No; {correction}",
                evidence=tuple(evidence_by_id.values()),
                diagnostics={
                    "derivation": "explicit_negative_finding",
                    "trace": self.model_query_trace,
                },
            )
        if (
            final_step["tool"] == "extract_values"
            and final_step["extractor"] == "field"
            and set(final_step["fields"]).issubset({"text", "source.path", "source.file_name", "source.file_stem"})
        ):
            return None
        if final.scalar is not None and (
            answer_shape == "number"
            or final_step["tool"] in {"calculate", "aggregate_values"}
        ):
            if isinstance(final.scalar, (int, float)) and not isinstance(final.scalar, bool):
                number = float(final.scalar)
                text = str(int(number)) if number.is_integer() else str(number)
            elif answer_shape == "number" or final_step["tool"] == "calculate":
                number = float(final.scalar)
                text = str(int(number)) if number.is_integer() else str(number)
            else:
                text = str(final.scalar).strip()
                if not text:
                    return None
        elif (
            final.values
            and final_step["tool"] in {
                "project_values", "extract_values", "model_extract", "union_values", "intersect_values"
            }
            and answer_shape != "boolean"
        ):
            values = [str(value).strip() for value in final.values if str(value).strip()]
            if not values:
                return None
            text = self._format_list_values(values) if answer_shape == "list" else (
                "; ".join(values) if len(values) > 1 else values[0]
            )
        else:
            return None
        evidence_by_id = {
            record.record_id: record.model_view()
            for index in self._dependency_closure(program, final_index)
            for record in results[index].records
        }
        return Answer(
            text,
            evidence=tuple(evidence_by_id.values()),
            diagnostics={
                "derivation": "deterministic_execution_of_model_selected_tool",
                "trace": self.model_query_trace,
            },
        )

    def _ground(
        self,
        contract: dict[str, Any],
        program: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> dict[str, Any]:
        selected_ids = self._dependency_closure(program, len(program["steps"]) - 1)
        record_map: dict[str, Any] = {}
        step_views: list[dict[str, Any]] = []
        for step_id in selected_ids:
            result = results[step_id]
            for record in result.records[:16]:
                record_map.setdefault(record.record_id, record)
            diagnostics = dict(result.diagnostics)
            if "evidence" in diagnostics:
                diagnostics["evidence"] = diagnostics["evidence"][:30]
            step_views.append(
                {
                    "step_id": step_id,
                    "kind": result.kind,
                    "values": result.values[:30],
                    "scalar": result.scalar,
                    "diagnostics": diagnostics,
                    "record_ids": [record.record_id for record in result.records[:16]],
                }
            )
        observations = {
            "steps": step_views,
            "records": [record.model_view(1200) for record in list(record_map.values())[:16]],
        }
        schema = grounded_answer_schema(contract["contract_id"])
        prompt = (
            "Act as the generic grounded extraction and formatting tool for an immutable semantic contract. Use only "
            "the supplied tool results. Do not reinterpret the contract or use outside knowledge. Every target, "
            "scope, relation, polarity, temporal requirement, and epistemic requirement must be established by the "
            "same coherent source record or an explicit model-planned join. For current/latest/final questions, use "
            "the latest applicable dated or ordered event. For allegations, dreams, fiction, quotations, and "
            "hypotheticals, answer only the relation requested by the contract. For booleans, absence or fictional "
            "context does not prove false; false requires explicit negation of the proposition. When evidence does "
            "not explicitly establish the requested answer, return status unknown and an empty answer. Never put an "
            "explanation of missing information in the answer field. For answered status, return the minimal value "
            "only: omit role prefixes unless they are part of the value, preserve source units for measured quantities, "
            "preserve complete URLs and identifiers, and "
            "cite only supplied record IDs.\n"
            f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
            f"Model-owned query program: {json.dumps(program, ensure_ascii=False)}\n"
            f"Tool results: {json.dumps(observations, ensure_ascii=False, default=str)}"
        )
        payload = self.model.complete_json("grounded_answer", prompt, schema, max_tokens=512)
        grounded = self._normalize_grounded(payload["grounded_answer"])
        available_ids = {record.record_id for result in results.values() for record in result.records}
        self._validate_grounded(contract, grounded, available_ids, record_map, results)
        return grounded

    @staticmethod
    def _normalize_grounded(grounded: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(grounded)
        answer = str(normalized.get("answer") or "").strip()
        if normalized.get("answer_shape") == "boolean":
            if answer.lower() == "true":
                normalized["answer"] = "yes"
            elif answer.lower() == "false":
                normalized["answer"] = "no"
            answer = str(normalized.get("answer") or "").strip()
        if normalized.get("answer_shape") == "number" and answer:
            match = re.fullmatch(r"[\s,;:=-]*([+-]?\d+(?:\.\d+)?)[\s,;:=-]*", answer)
            if match:
                number = float(match.group(1))
                normalized["answer"] = str(int(number)) if number.is_integer() else str(number)
        return normalized

    @staticmethod
    def _validate_grounded(
        contract: dict[str, Any],
        grounded: dict[str, Any],
        available_record_ids: set[str],
        record_map: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> None:
        if grounded["contract_id"] != contract["contract_id"]:
            raise ProgramValidationError("grounded answer contract mismatch")
        status = grounded["status"]
        answer = grounded["answer"].strip()
        evidence_ids = set(grounded["evidence_record_ids"])
        if not evidence_ids.issubset(available_record_ids):
            raise ProgramValidationError("grounded answer cites unknown evidence records")
        if status == "unknown" and answer:
            raise ProgramValidationError("unknown status cannot contain an answer")
        if status == "answered" and not answer:
            raise ProgramValidationError("answered status requires an answer")
        derived_values = [
            value
            for result in results.values()
            for value in ([result.scalar] if result.scalar is not None else result.values)
        ]
        if status == "answered" and not evidence_ids and answer not in {str(value) for value in derived_values}:
            raise ProgramValidationError("answered status requires evidence or an exact derived value")
        evidence_text = "\n".join(
            record_map[record_id].text
            + "\n"
            + json.dumps(record_map[record_id].data, ensure_ascii=False, default=str)
            for record_id in evidence_ids
            if record_id in record_map
        )
        shape = grounded["answer_shape"]
        if status == "answered" and shape == "url":
            if not re.fullmatch(r"https?://\S+", answer) or answer not in evidence_text:
                raise ProgramValidationError("URL answer must be copied exactly from evidence")
        if status == "answered" and shape == "identifier" and answer not in evidence_text:
            raise ProgramValidationError("identifier answer must be copied exactly from evidence")
        if status == "answered" and shape == "date_time" and answer not in evidence_text:
            raise ProgramValidationError("date answer must be copied exactly from evidence")
        if status == "answered" and shape == "boolean" and answer.lower() not in {
            "yes",
            "no",
            "true",
            "false",
        }:
            raise ProgramValidationError("boolean answer must be yes, no, true, or false")

    def _answer_from_grounded(
        self,
        grounded: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> Answer:
        if grounded["status"] == "unknown":
            return Answer("unknown", diagnostics={"grounded": grounded, "trace": self.model_query_trace})
        evidence_by_id = {
            record.record_id: record.model_view()
            for result in results.values()
            for record in result.records
        }
        evidence = tuple(
            evidence_by_id[item]
            for item in grounded["evidence_record_ids"]
            if item in evidence_by_id
        )
        return Answer(
            grounded["answer"].strip(),
            evidence=evidence,
            diagnostics={"grounded": grounded, "trace": self.model_query_trace},
        )

    def dspg_counts(self) -> dict[str, int]:
        return {
            "records": len(self.catalog.records),
            "preferred_records": len(self.catalog.preferred_records()),
            "collections": len(self.catalog.collections),
        }

    def dspg_integrity(self) -> str:
        return "ok" if self.catalog.records else "empty"
