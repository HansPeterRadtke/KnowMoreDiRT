"""Model-owned semantic reasoning over generic deterministic retrieval tools."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .catalog import SourceCatalog
from .model import ModelError, StrictModelClient
from .models import Answer, SourceRecord, ToolResult
from .schemas import (
    dataset_profile_schema,
    event_fact_verdict_schema,
    evidence_review_schema,
    grounded_answer_schema,
    numeric_value_repair_schema,
    query_program_schema,
    semantic_contract_schema,
    tool_extraction_schema,
)
from .tools import ToolExecutor, expand_step

_TOOL_DESCRIPTIONS = {
    "sample_records": "Return a bounded sample from one collection.",
    "search_records": "Search a collection or prior record set using literal model-selected terms.",
    "expand_source_context": "Replace matched fragments with coherent preferred records from the same source files.",
    "filter_records": "Filter prior records using explicit field paths and comparisons.",
    "project_values": "Project values from explicit structured field paths while retaining provenance.",
    "extract_values": "Extract values with a model-selected field, delimiter, regular expression, URL, identifier, date, number, or event-series extractor.",
    "model_extract": "Apply strict model extraction to bounded prior evidence under the immutable semantic contract.",
    "join_records": "Join two prior record sets using explicit field paths.",
    "union_values": "Combine values from prior steps.",
    "intersect_values": "Intersect values from prior steps.",
    "sort_records": "Sort records by an explicit field path.",
    "aggregate_values": "Count, deduplicate, or aggregate prior values.",
    "calculate": "Execute explicit arithmetic over supplied numbers or prior numeric values.",
}

_RECORD_INPUT_TOOLS = {
    "search_records",
    "expand_source_context",
    "filter_records",
    "project_values",
    "extract_values",
    "model_extract",
    "join_records",
    "sort_records",
}
_VALUE_INPUT_TOOLS = {"union_values", "intersect_values", "aggregate_values", "calculate"}

_TOOL_ARGUMENT_GUIDE = {
    "search_records": "arguments: mode=all|any|phrase; use any for alternative clues and all only for required conjunctions",
    "extract_values": "extractor must be field|after_label|after_phrase|before_phrase|between_phrases|regex|url|identifier|date_time|number|event_series; add matching label/start_phrase/end_phrase/pattern/value_group/time_group/occurrence/value_kind/strip_chars/distinct parameters",
    "join_records": "arguments: left_field and right_field",
    "sort_records": "arguments: sort_field and direction=ascending|descending",
    "aggregate_values": "arguments: aggregate=count|distinct|sum|average|min|max|mode and optionally distinct=true",
    "calculate": "one operation argument with value add|subtract|multiply|divide and numbers containing explicit operands",
}


class ProgramValidationError(ValueError):
    """Raised when model output violates structural execution invariants."""


class KnowMoreDiRTEngine:
    """Answer questions from an unfamiliar folder without Python-owned domain semantics."""

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
            answer = self._event_fact_answer(contract, program, results)
            if answer is None:
                answer = self._deterministic_terminal_answer(contract, program, results)
            if answer is None:
                answer = self._terminal_record_answer(contract, program, results)
            if answer is None:
                answer = self._review_until_complete(profile, contract, program, results)
        except (ModelError, ProgramValidationError, ValueError, KeyError, TypeError) as exc:
            answer = Answer(
                "unknown",
                diagnostics={"reason": type(exc).__name__, "error": str(exc), "trace": self.model_query_trace},
            )
        self.last_answer = answer
        return answer

    def _profile_dataset(self) -> dict[str, Any]:
        if self._dataset_profile is not None:
            return self._dataset_profile
        fingerprint = self.catalog.fingerprint
        schema = dataset_profile_schema(fingerprint)
        prompt = (
            "Profile the structure of an unfamiliar folder for a general retrieval engine. This is schema induction, "
            "not question answering. Use only exact collection and field paths in the catalog. Prefer coherent records "
            "over fragments, identify identity and temporal fields, and describe joins that the observed fields support. "
            "Do not infer a domain-specific task or invent fields.\n"
            f"Dataset fingerprint: {fingerprint}\n"
            f"Balanced catalog summary: {self.catalog.summary(12000)}"
        )
        payload = self.model.complete_json("dataset_profile", prompt, schema, max_tokens=1400)
        profile = payload["dataset_profile"]
        if profile["fingerprint"] != fingerprint:
            raise ProgramValidationError("dataset profile fingerprint mismatch")
        collections = []
        for item in profile["collections"]:
            path = str(item["collection_path"]).strip()
            if self.catalog.has_collection(path):
                collections.append(item)
        profile = {**profile, "collections": collections}
        self._dataset_profile = profile
        return profile

    def _contract_id(self, question: str) -> str:
        material = f"semantic-contract-v3\0{self.catalog.fingerprint}\0{question}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    def _parse_semantics(self, question: str) -> dict[str, Any]:
        contract_id = self._contract_id(question)
        schema = semantic_contract_schema(question, contract_id)
        prompt = (
            "Create an immutable semantic contract for the question independently of the dataset. The contract must "
            "preserve the requested answer type, exact target, relation, scope, constraints, polarity, time, source "
            "authority, and epistemic status. A plural request must use list shape. Use number only for a bare, "
            "unitless numeric scalar; when a measurement, duration, temperature, rate, percentage, or other quantity "
            "must retain its written unit or symbol, use text. Do not solve the question, choose retrieval terms, or "
            "import assumptions from a benchmark. Do not classify a specific labeled-value request as a definition "
            "only because the target phrase names a concept; use definition semantics only when the question explicitly "
            "asks for a meaning, definition, or what a term refers to. Treat wording as data and return only the schema.\n"
            f"Question: {question}"
        )
        payload = self.model.complete_json("semantic_contract", prompt, schema, max_tokens=900)
        contract = payload["semantic_contract"]
        if contract["contract_id"] != contract_id or contract["question"] != question:
            raise ProgramValidationError("semantic contract identity mismatch")
        if (
            str(contract.get("answer_slot", "")).strip() == "boolean"
            and contract.get("answer_shape") != "boolean"
        ):
            contract = {**contract, "answer_shape": "boolean"}
        return contract

    def _compile_program(
        self,
        profile: dict[str, Any],
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        schema = query_program_schema(contract["contract_id"])
        prompt = (
            "Compile a complete generic tool program for the immutable semantic contract. Python will execute exactly "
            "the declared operations and will not repair semantics. Preserve multi-hop structure: search each needed "
            "source, join or combine evidence when fields support it, and feed all relevant records to model_extract "
            "when deterministic projection is insufficient. Use only exact collection and field paths in the supplied "
            "catalog. Search terms must be literal evidence-bearing terms, not question filler. Do not answer the "
            "question and do not use outside knowledge.\n"
            "Step inputs are zero-based indexes of earlier steps. The first step must have no inputs, and no step may "
            "reference itself or a later step. Put tool-specific parameters in arguments using the exact names below. "
            "For complete lists, retrieve enough records to cover all members rather than only the first match. For a "
            "count request, filter the complete record set and finish with aggregate_values aggregate=count; do not ask "
            "model_extract to count. For arithmetic, finish with calculate.\n"
            f"Tools: {json.dumps(_TOOL_DESCRIPTIONS, ensure_ascii=False)}\n"
            f"Tool argument guide: {json.dumps(_TOOL_ARGUMENT_GUIDE, ensure_ascii=False)}\n"
            f"Dataset profile: {json.dumps(profile, ensure_ascii=False)}\n"
            f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
            f"Question-focused catalog: {self.catalog.summary(9000, query=contract['question'])}"
        )
        payload = self.model.complete_json("query_program", prompt, schema, max_tokens=1800)
        program = self._normalize_program(payload["query_program"])
        program = self._bind_root_searches_to_contract(program, contract)
        self._validate_program(contract, program)
        self.model_query_trace = {
            "dataset_profile": profile,
            "semantic_contract": contract,
            "program": program,
            "dataset_fingerprint": self.catalog.fingerprint,
            "reviews": [],
        }
        return program

    @staticmethod
    def _normalize_program(program: dict[str, Any]) -> dict[str, Any]:
        """Apply only bounded structural normalization to a model-owned program."""
        steps: list[dict[str, Any]] = []
        for index, raw in enumerate(list(program.get("steps", []))[:8]):
            inputs: list[int] = []
            for value in list(raw.get("inputs", []))[:8]:
                try:
                    ref = int(value)
                except (TypeError, ValueError):
                    continue
                # Small models often label the current output instead of the prior
                # output. Repair that generic zero/one-based indexing error only.
                if ref == index and index > 0:
                    ref = index - 1
                if 0 <= ref < index and ref not in inputs:
                    inputs.append(ref)
            step = {
                "tool": str(raw.get("tool", "")).strip(),
                "inputs": inputs,
                "collection": str(raw.get("collection", "")).strip(),
                "terms": [str(value).strip() for value in raw.get("terms", []) if str(value).strip()][:12],
                "fields": [str(value).strip() for value in raw.get("fields", []) if str(value).strip()][:12],
                "filters": list(raw.get("filters", []))[:8],
                "arguments": list(raw.get("arguments", []))[:12],
                "limit": max(1, min(int(raw.get("limit", 20)), 5000)),
            }
            if step["tool"] == "extract_values":
                extractor = "field" if step["fields"] else "none"
                for argument in step["arguments"]:
                    if str(argument.get("name", "")).strip() == "extractor":
                        extractor = str(argument.get("value", "")).strip()
                        break
                supported_extractors = {
                    "field",
                    "after_label",
                    "after_phrase",
                    "before_phrase",
                    "between_phrases",
                    "regex",
                    "url",
                    "identifier",
                    "date_time",
                    "number",
                    "event_series",
                }
                if extractor not in supported_extractors:
                    step["tool"] = "model_extract"
                    step["arguments"] = []
            if step["tool"] == "join_records" and len(step["inputs"]) == 1:
                step["tool"] = "expand_source_context"
                step["arguments"] = []
            steps.append(step)
        return {"contract_id": str(program.get("contract_id", "")).strip(), "steps": steps}

    @staticmethod
    def _bind_root_searches_to_contract(
        program: dict[str, Any],
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach model-owned contract surfaces to root searches as ranking hints."""
        semantic_terms: list[str] = []
        for key in (
            "target_phrases",
            "constraint_phrases",
            "scope_phrases",
            "relation_phrases",
        ):
            for value in contract.get(key, []):
                text = str(value).strip()
                if text and text not in semantic_terms:
                    semantic_terms.append(text)
        steps: list[dict[str, Any]] = []
        for raw in program["steps"]:
            step = dict(raw)
            if step["tool"] == "search_records" and not step["inputs"]:
                terms = list(step["terms"])
                hint_terms = list(step.get("_contract_terms", []))
                for term in semantic_terms:
                    if term not in terms and term not in hint_terms:
                        hint_terms.append(term)
                step["_contract_terms"] = hint_terms[:12]
            steps.append(step)
        return {**program, "steps": steps}

    def _validate_program(self, contract: dict[str, Any], program: dict[str, Any]) -> None:
        if program["contract_id"] != contract["contract_id"]:
            raise ProgramValidationError("query program contract mismatch")
        if not program["steps"]:
            raise ProgramValidationError("query program has no steps")
        for index, compact in enumerate(program["steps"]):
            step = expand_step(compact)
            tool = step["tool"]
            if tool not in _TOOL_DESCRIPTIONS:
                raise ProgramValidationError(f"unknown tool at step {index}")
            if any(ref < 0 or ref >= index for ref in step["inputs"]):
                raise ProgramValidationError(f"invalid input reference at step {index}")
            if tool == "sample_records" and not self.catalog.has_collection(step["collection"]):
                raise ProgramValidationError(f"unknown sample collection at step {index}")
            if tool == "search_records" and not step["inputs"]:
                if not step["collection"]:
                    compact["collection"] = "all_records"
                    step["collection"] = "all_records"
                if not self.catalog.has_collection(step["collection"]):
                    raise ProgramValidationError(f"unknown search collection at step {index}")
                if not step["terms"]:
                    raise ProgramValidationError(f"search step {index} has no terms")
            if tool == "join_records" and len(step["inputs"]) != 2:
                raise ProgramValidationError(f"join step {index} requires two inputs")
            if tool == "expand_source_context" and not step["inputs"]:
                raise ProgramValidationError(f"context expansion step {index} requires input")
            if tool in _RECORD_INPUT_TOOLS - {"search_records", "sample_records", "join_records"}:
                if not step["inputs"] and not self.catalog.has_collection(step["collection"]):
                    raise ProgramValidationError(f"record tool step {index} has no input")
            if tool in _VALUE_INPUT_TOOLS and tool != "calculate" and not step["inputs"]:
                raise ProgramValidationError(f"value tool step {index} has no input")

    def _execute_program(
        self,
        contract: dict[str, Any],
        program: dict[str, Any],
    ) -> dict[int, ToolResult]:
        return self.executor.execute(
            program["steps"],
            semantic_extractor=lambda step_id, step, results: self._model_extract(
                contract, step_id, step, results
            ),
        )

    @staticmethod
    def _safe_surface(value: str) -> str:
        clean = str(value).strip()
        if clean.startswith(("http://", "https://")):
            clean = clean.rstrip(".,;:!?)]}")
        return clean

    @classmethod
    def _surface_scalar(cls, value: Any) -> str:
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        clean = cls._safe_surface(str(value).strip())
        if clean.lower() == "true":
            return "Yes"
        if clean.lower() == "false":
            return "No"
        return clean

    @classmethod
    def _calculation_operands_grounded(
        cls,
        step: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> bool:
        if not step["numbers"] or not step["inputs"]:
            return True
        material_parts: list[str] = []
        for index in step["inputs"]:
            result = results.get(index)
            if result is None:
                continue
            material_parts.extend(str(value) for value in result.values)
            if result.scalar is not None:
                material_parts.append(str(result.scalar))
            material_parts.extend(record.text for record in result.records)
        material_words = cls._surface_words(" ".join(material_parts))
        return all(
            cls._surface_scalar(number).casefold() in material_words
            for number in step["numbers"]
        )

    def _normalize_event_verdict(
        self,
        contract: dict[str, Any],
        verdict: dict[str, Any],
        result: ToolResult,
    ) -> dict[str, Any]:
        """Reconcile contradictory model stages only with exact cited grounding."""
        normalized = dict(verdict)
        contradiction_bases = {
            "explicit_denial",
            "authoritative_not_proven",
            "impossibility",
        }

        def apply_structural_invariants(value: dict[str, Any]) -> dict[str, Any]:
            basis = str(value.get("evidence_basis", "mixed_or_other"))
            if value.get("verdict") == "contradicts" and basis not in contradiction_bases:
                value["verdict"] = "insufficient"
                value["correction_clause"] = ""
            if value.get("verdict") == "supports" and basis != "explicit_support":
                value["verdict"] = "insufficient"
                value["correction_clause"] = ""
            if (
                value.get("verdict") in {"supports", "contradicts"}
                and value.get("scope_binding") == "none"
            ):
                value["verdict"] = "insufficient"
                value["correction_clause"] = ""
            return value

        extraction = result.diagnostics.get("extraction", {})
        if not isinstance(extraction, dict):
            return apply_structural_invariants(normalized)
        available = {record.record_id for record in result.records}
        extraction_ids = set(extraction.get("evidence_record_ids", []))
        if not extraction_ids or not extraction_ids.issubset(available):
            return apply_structural_invariants(normalized)
        relation = str(extraction.get("evidence_relation", ""))
        if (
            relation == "direct_contradiction"
            and str(contract.get("authority_mode", "")) == "explicit_official"
        ):
            normalized["verdict"] = "contradicts"
            normalized["scope_binding"] = "direct"
            normalized["evidence_basis"] = "explicit_denial"
            normalized["evidence_record_ids"] = list(extraction.get("evidence_record_ids", []))
        if extraction.get("status") != "extracted":
            return apply_structural_invariants(normalized)
        values = {
            str(value).strip().lower()
            for value in result.values
            if str(value).strip()
        }
        basis = str(normalized.get("evidence_basis", "mixed_or_other"))
        if (
            relation == "direct_contradiction"
            and values.intersection({"false", "no"})
            and basis in contradiction_bases
        ):
            normalized["verdict"] = "contradicts"
            normalized["evidence_record_ids"] = list(extraction.get("evidence_record_ids", []))
            if not str(normalized.get("correction_clause", "")).strip():
                normalized["correction_clause"] = str(extraction.get("reason", "")).strip()
        elif (
            relation in {"direct_support", "structured_field", "derived"}
            and values.intersection({"true", "yes"})
            and basis == "explicit_support"
        ):
            normalized["verdict"] = "supports"
            normalized["evidence_record_ids"] = list(extraction.get("evidence_record_ids", []))
        return apply_structural_invariants(normalized)

    def _render_event_fact_answer(
        self,
        contract: dict[str, Any],
        verdict: dict[str, Any],
        evidence: list[dict[str, Any]],
        available: set[str],
        fallback: str,
    ) -> tuple[str, dict[str, Any] | None]:
        """Render a validated verdict without adding facts or evidence."""
        schema = grounded_answer_schema(contract["contract_id"])
        prompt = (
            "Render the final minimal answer from an already validated yes/no verdict. Do not reconsider the verdict, "
            "add facts, or use outside knowledge. Preserve only cited evidence. For a contradiction, begin with 'No;' "
            "and use the shortest source-authority clause. Prefer the authority label in a source header or path, such "
            "as final judgment, final decision, verified schedule, or inspection, over a participant label. Omit entity "
            "names and event details already present in the question. For support, begin with 'Yes' and add a clause only "
            "when essential. Return the canonical answer only in grounded_answer.answer.\n"
            f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
            f"Validated verdict: {json.dumps(verdict, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(evidence, ensure_ascii=False, default=str)}"
        )
        payload = self.model.complete_json(
            "grounded_answer",
            prompt,
            schema,
            max_tokens=500,
        )
        rendered = payload["grounded_answer"]
        if rendered["contract_id"] != contract["contract_id"]:
            raise ProgramValidationError("grounded answer contract mismatch")
        if rendered["answer_shape"] != contract["answer_shape"]:
            raise ProgramValidationError("grounded answer shape mismatch")
        cited = set(rendered["evidence_record_ids"])
        if not cited or not cited.issubset(available):
            raise ProgramValidationError("grounded answer cites unavailable records")
        answer = str(rendered["answer"]).strip()
        if rendered["status"] != "answered" or not answer:
            return fallback, rendered
        answer = self._clean_rendered_answer(answer)
        expected = "yes" if verdict["verdict"] == "supports" else "no"
        if answer.split(";", 1)[0].strip().lower() != expected:
            raise ProgramValidationError("grounded answer polarity mismatch")
        if verdict["verdict"] == "contradicts":
            answer = self._repair_short_negative_clause(answer, rendered, evidence, fallback)
        return answer, rendered

    @staticmethod
    def _clean_rendered_answer(answer: str) -> str:
        """Remove citation metadata from final answer text; evidence travels separately."""
        clean = str(answer).strip()
        while " (" in clean:
            candidate = clean[:-1].rstrip() if clean.endswith(".") else clean
            if not candidate.endswith(")"):
                break
            prefix, suffix = candidate.rsplit(" (", 1)
            marker = suffix[:-1].casefold()
            if "source" not in marker and "record" not in marker:
                break
            clean = prefix.rstrip()
            if answer.endswith(".") and not clean.endswith((".", "!", "?")):
                clean += "."
        for prefix in ("No; ", "Yes; "):
            if clean.startswith(prefix) and len(clean) > len(prefix):
                first = clean[len(prefix)]
                if first.isupper():
                    clean = prefix + first.lower() + clean[len(prefix) + 1 :]
                break
        return clean

    @staticmethod
    def _surface_words(value: str) -> list[str]:
        words: list[str] = []
        current: list[str] = []
        for char in str(value).casefold():
            if char.isalnum():
                current.append(char)
            elif current:
                words.append("".join(current))
                current = []
        if current:
            words.append("".join(current))
        return words

    @classmethod
    def _normalized_surface(cls, value: str) -> str:
        return " ".join(cls._surface_words(value))

    @classmethod
    def _surface_supported_by_evidence(
        cls,
        surface: str,
        evidence: list[dict[str, Any]],
    ) -> bool:
        normalized = cls._normalized_surface(surface)
        if not normalized:
            return False
        for item in evidence:
            material = (
                str(item.get("excerpt", ""))
                + "\n"
                + json.dumps(item.get("data", {}), ensure_ascii=False, default=str)
            )
            if normalized in cls._normalized_surface(material):
                return True
        return False

    @staticmethod
    def _is_iso_date(value: str) -> bool:
        text = str(value)
        return (
            len(text) == 10
            and text[4] == "-"
            and text[7] == "-"
            and text[:4].isdigit()
            and text[5:7].isdigit()
            and text[8:10].isdigit()
        )

    @classmethod
    def _iso_timestamp_surfaces(cls, material: str) -> list[str]:
        text = str(material)
        surfaces: list[str] = []
        for index in range(max(0, len(text) - 15)):
            date = text[index : index + 10]
            if not cls._is_iso_date(date):
                continue
            if index + 16 > len(text):
                continue
            separator = text[index + 10]
            hour = text[index + 11 : index + 13]
            minute = text[index + 14 : index + 16]
            if (
                separator not in {" ", "T"}
                or text[index + 13] != ":"
                or not hour.isdigit()
                or not minute.isdigit()
            ):
                continue
            end = index + 16
            if (
                end + 3 <= len(text)
                and text[end] == ":"
                and text[end + 1 : end + 3].isdigit()
            ):
                end += 3
            if end < len(text) and text[end] == "Z":
                end += 1
            elif (
                end + 6 <= len(text)
                and text[end] in {"+", "-"}
                and text[end + 1 : end + 3].isdigit()
                and text[end + 3] == ":"
                and text[end + 4 : end + 6].isdigit()
            ):
                end += 6
            surface = text[index:end].strip().rstrip(".,;:!?)]}\"'")
            if surface not in surfaces:
                surfaces.append(surface)
        return surfaces

    @classmethod
    def _repair_temporal_prefix_surface(
        cls,
        value: str,
        review: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> str:
        surface = str(value).strip()
        date = surface.rstrip(".,;:!?")
        if not cls._is_iso_date(date):
            return surface
        material = str(review.get("reason", ""))
        for item in evidence:
            material += "\n" + str(item.get("excerpt", ""))
            material += "\n" + json.dumps(
                item.get("data", {}),
                ensure_ascii=False,
                default=str,
            )
        matches = [
            candidate
            for candidate in cls._iso_timestamp_surfaces(material)
            if candidate.startswith(date + " ") or candidate.startswith(date + "T")
        ]
        return matches[0] if len(matches) == 1 else surface

    @classmethod
    def _negative_clause_word_count(cls, answer: str) -> int:
        if ";" not in answer:
            return 0
        return len(cls._surface_words(answer.split(";", 1)[1]))

    @classmethod
    def _repair_short_negative_clause(
        cls,
        answer: str,
        rendered: dict[str, Any],
        evidence: list[dict[str, Any]],
        fallback: str,
    ) -> str:
        """Prefer a cited predicate over an authority label alone."""
        if cls._negative_clause_word_count(answer) >= 2:
            return answer
        if (
            fallback
            and fallback.split(";", 1)[0].strip().lower() == "no"
            and cls._negative_clause_word_count(fallback)
            > cls._negative_clause_word_count(answer)
            and cls._surface_supported_by_evidence(fallback.split(";", 1)[1], evidence)
        ):
            return cls._clean_rendered_answer(fallback)
        reason = cls._clean_rendered_answer(str(rendered.get("reason", "")).strip())
        for prefix in ("No; ", "Yes; "):
            if reason.startswith(prefix):
                reason = reason[len(prefix) :].strip()
                break
        reason = reason.strip().strip("\"'")
        if not (3 <= len(cls._surface_words(reason)) <= 40):
            return answer
        if not cls._surface_supported_by_evidence(reason, evidence):
            return answer
        if not reason.endswith((".", "!", "?")):
            reason += "."
        return cls._clean_rendered_answer(f"No; {reason}")

    def _event_fact_answer(
        self,
        contract: dict[str, Any],
        program: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> Answer | None:
        """Adjudicate a model-owned yes/no proposition with cited evidence."""
        if contract.get("answer_shape") != "boolean" or not program["steps"]:
            return None
        result = results.get(len(program["steps"]) - 1)
        if result is None:
            return None
        records = self._dedupe_records(result.records)[:60]
        if not records:
            return None
        evidence = self._bounded_record_views(
            records,
            max_records=60,
            per_record_chars=1400,
            total_chars=20000,
        )
        extraction = result.diagnostics.get("extraction", {})
        relation = str(extraction.get("evidence_relation", ""))
        extraction_ids = set(extraction.get("evidence_record_ids", []))
        available_ids = {record.record_id for record in records}
        values = {
            str(value).strip().lower()
            for value in result.values
            if str(value).strip()
        }
        insufficient_relation = relation in {
            "state_only",
            "nonactual_content",
            "absence",
            "unknown",
        }
        inconsistent_polarity = (
            relation == "direct_support" and bool(values.intersection({"false", "no"}))
        ) or (
            relation == "direct_contradiction" and bool(values.intersection({"true", "yes"}))
        )
        if (
            (insufficient_relation or inconsistent_polarity)
            and extraction_ids
            and extraction_ids.issubset(available_ids)
        ):
            evidence_by_id = {record.record_id: record.model_view() for record in records}
            grounded = tuple(
                evidence_by_id[record_id]
                for record_id in extraction.get("evidence_record_ids", [])
                if record_id in evidence_by_id
            )
            return Answer(
                "unknown",
                evidence=grounded,
                diagnostics={
                    "reason": (
                        "inconsistent_event_evidence_relation"
                        if inconsistent_polarity
                        else "insufficient_event_evidence"
                    ),
                    "semantic_contract": contract,
                    "terminal_result": self._result_observation(result),
                    "trace": self.model_query_trace,
                },
            )
        schema = event_fact_verdict_schema(contract["contract_id"])
        prompt = (
            "Adjudicate the yes/no factual proposition in the immutable semantic contract using only the supplied "
            "evidence. Return supports only when the proposition is established, contradicts when cited authoritative "
            "or corrective evidence explicitly denies it or establishes that it was not proven, and insufficient otherwise. "
            "Classify evidence_basis exactly: explicit_support only for a direct actual-world assertion of occurrence; "
            "explicit_denial only for a direct denial of occurrence; authoritative_not_proven only for an authoritative "
            "finding that the proposition was not proven; impossibility only for evidence making occurrence impossible; "
            "state_only for a later or current state; absence_only for missing records or silence; nonactual_only for dreams, "
            "fiction, plans, beliefs, hypotheticals, or quoted allegations without actual-world confirmation; mixed_or_other "
            "otherwise. For authoritative_not_proven, authority_label must be the shortest authority/finality label "
            "explicitly established by the source header, title, or path, and decisive_predicate must be the shortest "
            "subject-free predicate expressing the finding; omit entities and event details already in the question. "
            "Example shapes are authority_label='the final judgment' and decisive_predicate='found no proof'; include the natural determiner in authority_label when grammatical. For every other "
            "evidence basis, use empty strings for authority_label and decisive_predicate. A later or current state that is "
            "compatible with the event having occurred and later been reversed is state-only circumstantial evidence, not "
            "a contradiction. Do not infer non-occurrence from such a state. For a contradiction, correction_clause must "
            "preserve the decisive authority or finality and state the "
            "grounded correction without a leading Yes or No. Use the shortest authority-framed clause: when a source "
            "header or path identifies a final judgment, final decision, verified schedule, inspection, or equivalent, "
            "name that authority directly rather than replacing it with a participant such as a court or person. Omit "
            "entity names and event details already stated in the question, and keep only the decisive predicate. Begin "
            "ordinary clauses in lowercase. For support, correction_clause may be empty unless a qualification is "
            "essential. Source paths and file names may establish document authority and finality. Cite only supplied "
            "record IDs and do not use outside knowledge.\n"
            f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
            f"Terminal result: {json.dumps(self._result_observation(result), ensure_ascii=False, default=str)}\n"
            f"Evidence: {json.dumps(evidence, ensure_ascii=False, default=str)}"
        )
        payload = self.model.complete_json(
            "event_fact_verdict",
            prompt,
            schema,
            max_tokens=700,
        )
        verdict = self._normalize_event_verdict(
            contract,
            payload["event_fact_verdict"],
            result,
        )
        if verdict["contract_id"] != contract["contract_id"]:
            raise ProgramValidationError("event verdict contract mismatch")
        available = {record.record_id for record in records}
        cited = set(verdict["evidence_record_ids"])
        if not cited or not cited.issubset(available):
            raise ProgramValidationError("event verdict cites unavailable records")
        evidence_by_id = {record.record_id: record.model_view() for record in records}
        grounded = tuple(
            evidence_by_id[record_id]
            for record_id in verdict["evidence_record_ids"]
            if record_id in evidence_by_id
        )
        if verdict["verdict"] == "insufficient":
            return Answer(
                "unknown",
                evidence=grounded,
                diagnostics={
                    "reason": "validated_event_fact_insufficient",
                    "semantic_contract": contract,
                    "event_fact_verdict": verdict,
                    "terminal_result": self._result_observation(result),
                    "trace": self.model_query_trace,
                },
            )
        prefix = "Yes" if verdict["verdict"] == "supports" else "No"
        clause = str(verdict["correction_clause"]).strip()
        if clause.lower().startswith(("yes", "no")):
            text = clause
        else:
            text = prefix + (f"; {clause}" if clause else "")
        rendered = None
        if verdict.get("evidence_basis") == "authoritative_not_proven":
            authority = str(verdict.get("authority_label", "")).strip().rstrip(". ")
            predicate = str(verdict.get("decisive_predicate", "")).strip().rstrip(". ")
            if authority and predicate:
                text = f"No; {authority} {predicate}."
            else:
                text, rendered = self._render_event_fact_answer(
                    contract,
                    verdict,
                    list(grounded),
                    available,
                    text,
                )
        else:
            text, rendered = self._render_event_fact_answer(
                contract,
                verdict,
                list(grounded),
                available,
                text,
            )
        return Answer(
            text,
            evidence=grounded,
            diagnostics={
                "reason": "validated_event_fact_verdict",
                "semantic_contract": contract,
                "event_fact_verdict": verdict,
                "grounded_answer": rendered,
                "terminal_result": self._result_observation(result),
                "trace": self.model_query_trace,
            },
        )

    def _terminal_record_answer(
        self,
        contract: dict[str, Any],
        program: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> Answer | None:
        """Resolve a scalar directly from terminal records using a cited model verdict."""
        temporal_mode = str(contract.get("temporal_mode", "none"))
        if (
            contract.get("answer_shape") in {"list", "boolean"}
            or temporal_mode == "none"
            or not program["steps"]
        ):
            return None
        index = len(program["steps"]) - 1
        result = results.get(index)
        if result is None or not result.records:
            return None
        records = self._dedupe_records(result.records)[:40]
        evidence = self._bounded_record_views(
            records,
            max_records=40,
            per_record_chars=1600,
            total_chars=18000,
        )
        schema = grounded_answer_schema(contract["contract_id"])
        prompt = (
            "Answer the immutable semantic contract only when the supplied terminal records directly and completely "
            "support one scalar answer. Bind every value to the target entity and relation in the contract; ignore "
            "records about other entities even if they use the same field label. For temporal_mode current or final, "
            "if the target has an explicit dated or ordered sequence of states, return the state at the greatest "
            "timestamp or last explicit sequence position. Do not require the word current when a dated sequence "
            "establishes recency. An explicit absence statement that says the requested scalar value is not established "
            "is not itself a scalar answer; return unknown instead of copying the denial. When supplied records establish "
            "that absence, cite those records with the unknown response; when evidence is merely incomplete, cite no "
            "records. Use no outside knowledge and cite only supplied record IDs. Return unknown if a further source, "
            "join, or unresolved relation is required.\n"
            f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
            f"Terminal records: {json.dumps(evidence, ensure_ascii=False, default=str)}"
        )
        payload = self.model.complete_json(
            "terminal_record_answer",
            prompt,
            schema,
            max_tokens=700,
        )
        grounded = payload["grounded_answer"]
        if grounded["contract_id"] != contract["contract_id"]:
            raise ProgramValidationError("terminal record answer contract mismatch")
        if grounded["answer_shape"] != contract["answer_shape"]:
            raise ProgramValidationError("terminal record answer shape mismatch")
        available = {record.record_id for record in records}
        cited = set(grounded["evidence_record_ids"])
        if not cited.issubset(available):
            raise ProgramValidationError("terminal record answer cites unavailable records")
        evidence_by_id = {record.record_id: record.model_view() for record in records}
        cited_evidence = tuple(
            evidence_by_id[record_id]
            for record_id in grounded["evidence_record_ids"]
            if record_id in evidence_by_id
        )
        if grounded["status"] != "answered":
            if str(grounded["answer"]).strip() and cited_evidence:
                grounded = {**grounded, "status": "answered"}
            else:
                if grounded["status"] == "unknown" and cited_evidence:
                    return Answer(
                        "unknown",
                        evidence=cited_evidence,
                        diagnostics={
                            "reason": "validated_terminal_record_unknown",
                            "semantic_contract": contract,
                            "grounded_answer": grounded,
                            "terminal_result": self._result_observation(result),
                            "trace": self.model_query_trace,
                        },
                    )
                return None
        answer = self._safe_surface(str(grounded["answer"]).strip())
        if not answer:
            return None
        if not cited:
            raise ProgramValidationError("terminal record answer cites unavailable records")
        return Answer(
            answer,
            evidence=cited_evidence,
            diagnostics={
                "reason": "validated_terminal_record_answer",
                "semantic_contract": contract,
                "grounded_answer": grounded,
                "terminal_result": self._result_observation(result),
                "trace": self.model_query_trace,
            },
        )

    def _deterministic_terminal_answer(
        self,
        contract: dict[str, Any],
        program: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> Answer | None:
        """Surface a model-selected deterministic terminal scalar with provenance."""
        if contract.get("answer_shape") == "boolean":
            return None
        if not program["steps"] or not results:
            return None
        index = len(program["steps"]) - 1
        if index not in results:
            return None
        step = expand_step(program["steps"][index])
        result = results[index]
        if step["tool"] in {"calculate", "aggregate_values"}:
            if step["tool"] == "calculate" and not self._calculation_operands_grounded(
                step,
                results,
            ):
                return None
            value = result.scalar
        elif step["tool"] == "extract_values" and contract["answer_shape"] != "list":
            scalar_values = self._distinct_scalar_values(result.values)
            if len(scalar_values) != 1:
                return None
            mechanically_typed = {
                "url": "url",
                "number": "number",
                "identifier": "identifier",
            }
            if mechanically_typed.get(contract["answer_shape"]) != step["extractor"]:
                return None
            value = scalar_values[0]
            if step["extractor"] == "number" and len(step["inputs"]) == 1:
                upstream = results.get(step["inputs"][0])
                upstream_values = [] if upstream is None else list(upstream.values)
                if upstream is not None and upstream.scalar is not None:
                    upstream_values.append(upstream.scalar)
                upstream_surfaces: list[str] = []
                for upstream_value in upstream_values:
                    surface = str(upstream_value).strip().rstrip(".,;:!?")
                    if surface and surface not in upstream_surfaces:
                        upstream_surfaces.append(surface)
                if len(upstream_surfaces) == 1:
                    upstream_number = self._unit_bearing_numeric_value(upstream_surfaces[0])
                    extracted_number = self._parse_numeric_surface(value)
                    if upstream_number is not None and upstream_number == extracted_number:
                        value = upstream_surfaces[0]
        elif step["tool"] == "project_values" and contract["answer_shape"] != "list":
            scalar_values = self._distinct_scalar_values(result.values)
            if len(scalar_values) != 1:
                return None
            value = scalar_values[0]
        elif step["tool"] == "model_extract" and contract["answer_shape"] != "list":
            diagnostics = result.diagnostics
            extraction = diagnostics.get("numeric_repair") or diagnostics.get("extraction") or {}
            relation = str(extraction.get("evidence_relation", ""))
            if str(contract.get("temporal_mode", "none")) != "none" and relation == "direct_support":
                return None
            if (
                diagnostics.get("status") != "extracted"
                or len(result.values) != 1
                or relation not in {
                    "direct_support",
                    "direct_contradiction",
                    "structured_field",
                    "derived",
                }
            ):
                return None
            value = result.values[0]
        else:
            return None
        if value is None:
            return None
        if contract.get("requires_explicit_evidence") and not result.records:
            return None
        text = self._surface_scalar(value)
        if not text:
            return None
        records = self._dedupe_records(result.records)[:60]
        evidence = tuple(record.model_view() for record in records)
        rendered = None
        render_attempted = False
        original_text = text
        if step["tool"] == "model_extract" and self._should_render_review_answer(contract, text):
            render_attempted = True
            review = {
                "contract_id": contract["contract_id"],
                "status": "answered",
                "answer": text,
                "answer_items": [],
                "answer_shape": contract["answer_shape"],
                "evidence_record_ids": [record.record_id for record in records],
                "searches": [],
                "confidence": 1.0,
                "reason": "Terminal extraction supplied a grounded scalar.",
            }
            try:
                text, rendered = self._render_review_answer(
                    contract,
                    review,
                    list(evidence),
                    {record.record_id for record in records},
                    text,
                )
            except (ModelError, ProgramValidationError, ValueError, KeyError, TypeError):
                rendered = None
        if render_attempted and text == original_text:
            return None
        return Answer(
            text,
            evidence=evidence,
            diagnostics={
                "reason": "validated_terminal_scalar",
                "semantic_contract": contract,
                "terminal_step": step,
                "terminal_result": self._result_observation(result),
                "grounded_answer": rendered,
                "trace": self.model_query_trace,
            },
        )

    @staticmethod
    def _dedupe_records(records: list[SourceRecord]) -> list[SourceRecord]:
        output: list[SourceRecord] = []
        seen: set[str] = set()
        for record in records:
            if record.record_id not in seen:
                seen.add(record.record_id)
                output.append(record)
        return output

    @staticmethod
    def _distinct_scalar_values(values: list[Any]) -> list[Any]:
        output: list[Any] = []
        seen: set[str] = set()
        for value in values:
            surface = str(value).strip()
            if not surface:
                continue
            key = " ".join(surface.casefold().rstrip(".,;:!?").split())
            if key not in seen:
                seen.add(key)
                output.append(value)
        return output

    def _records_for_model_step(
        self,
        step: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> list[SourceRecord]:
        records: list[SourceRecord] = []
        for ref in step["inputs"]:
            records.extend(results[ref].records)
        if not step["inputs"] and step["collection"]:
            records.extend(self.catalog.collection_records(step["collection"]))
        return self._dedupe_records(records)[: max(1, min(step["limit"], 80))]

    @staticmethod
    def _bounded_record_views(
        records: list[SourceRecord],
        *,
        max_records: int = 60,
        per_record_chars: int = 1200,
        total_chars: int = 18000,
    ) -> list[dict[str, Any]]:
        """Pack ordered evidence without exceeding the model context budget."""
        views: list[dict[str, Any]] = []
        used = 0
        for record in records[:max_records]:
            view = record.model_view(per_record_chars)
            rendered = json.dumps(view, ensure_ascii=False, default=str)
            if views and used + len(rendered) > total_chars:
                break
            if not views and len(rendered) > total_chars:
                view = record.model_view(max(200, total_chars // 2))
                rendered = json.dumps(view, ensure_ascii=False, default=str)
            views.append(view)
            used += len(rendered)
        return views

    @staticmethod
    def _parse_numeric_surface(value: Any) -> int | float | None:
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return int(number) if number.is_integer() else number

    @staticmethod
    def _unit_bearing_numeric_value(value: Any) -> int | float | None:
        """Return a leading number only when a nonnumeric written suffix remains."""
        text = str(value).strip()
        if not text:
            return None
        index = 1 if text[0] in "+-" else 0
        start = index
        decimal_seen = False
        while index < len(text):
            character = text[index]
            if character.isdigit() or character == ",":
                index += 1
                continue
            if character == "." and not decimal_seen:
                decimal_seen = True
                index += 1
                continue
            break
        numeric = text[start:index].replace(",", "")
        if not numeric or not any(character.isdigit() for character in numeric):
            return None
        try:
            number = float(numeric)
        except ValueError:
            return None
        suffix = text[index:].strip().rstrip(".,;:!?")
        if not suffix:
            return None
        suffix_is_written = any(character.isalpha() for character in suffix) or all(
            not character.isalnum() and not character.isspace()
            for character in suffix
        )
        if not suffix_is_written:
            return None
        return int(number) if number.is_integer() else number

    @classmethod
    def _unit_bearing_numeric_surface(cls, value: Any) -> bool:
        return cls._unit_bearing_numeric_value(value) is not None

    def _repair_numeric_extraction(
        self,
        contract: dict[str, Any],
        extraction: dict[str, Any],
        evidence: list[dict[str, Any]],
        available: set[str],
    ) -> dict[str, Any]:
        schema = numeric_value_repair_schema(contract["contract_id"])
        prompt = (
            "Repair a claimed numeric extraction under the immutable semantic contract. Return the requested numeric "
            "answer itself, never a row index, path, record id, or position. For a count, count all matching evidence "
            "records. Use no outside knowledge and cite only supplied record IDs.\n"
            f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
            f"Prior extraction: {json.dumps(extraction, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(evidence, ensure_ascii=False, default=str)}"
        )
        payload = self.model.complete_json(
            "numeric_value_repair",
            prompt,
            schema,
            max_tokens=500,
        )
        repair = payload["numeric_value_repair"]
        if repair["contract_id"] != contract["contract_id"]:
            raise ProgramValidationError("numeric repair contract mismatch")
        cited = set(repair["evidence_record_ids"])
        if not cited.issubset(available):
            raise ProgramValidationError("numeric repair cites unavailable records")
        return repair

    def _model_extract(
        self,
        contract: dict[str, Any],
        step_id: str,
        step: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> ToolResult:
        records = self._records_for_model_step(step, results)
        schema = tool_extraction_schema(contract["contract_id"])
        evidence = self._bounded_record_views(records, total_chars=20000)
        prompt = (
            "Extract only values explicitly supported by the bounded evidence and the immutable semantic contract. "
            "Do not use outside knowledge. Preserve list completeness, role binding, temporal scope, polarity, and "
            "source attribution. Return unknown with no values when the requested value is not established. For boolean "
            "event facts, direct_support means the queried event itself is explicitly asserted to have occurred, while "
            "direct_contradiction means the queried event itself is explicitly denied or an authoritative source says it "
            "was not proven. Use state_only for a resulting state, location, or consequence that does not explicitly deny "
            "the event. Use nonactual_content for dreams, fiction, plans, allegations, quotes, or beliefs not established "
            "as actual. State-only, nonactual, absence-only, and unknown evidence must return status unknown with no values; "
            "never infer event truth from a consequence alone. For scalar value requests, evidence that explicitly says the "
            "requested value is absent or not established is absence evidence, not the answer text; return unknown with no "
            "values and evidence_relation absence. Cite only record IDs supplied below. Values must be minimal "
            "surfaces; lists are arrays in this schema.\n"
            f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
            f"Tool call: {json.dumps(step, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(evidence, ensure_ascii=False, default=str)}"
        )
        payload = self.model.complete_json("tool_extraction", prompt, schema, max_tokens=900)
        extraction = payload["tool_extraction"]
        if extraction["contract_id"] != contract["contract_id"]:
            raise ProgramValidationError("tool extraction contract mismatch")
        if extraction["answer_shape"] != contract["answer_shape"]:
            raise ProgramValidationError("tool extraction answer shape mismatch")
        available = {record.record_id for record in records}
        cited = set(extraction["evidence_record_ids"])
        if not cited.issubset(available):
            raise ProgramValidationError("tool extraction cites unavailable records")
        values = [str(value).strip() for value in extraction["values"] if str(value).strip()]
        numeric_repair: dict[str, Any] | None = None
        relation = str(extraction.get("evidence_relation", "")).strip()
        if extraction["status"] != "unknown" and relation in {"absence", "unknown"}:
            extraction = {**extraction, "status": "unknown", "values": []}
            values = []
        if extraction["status"] == "unknown":
            values = []
        elif not values:
            raise ProgramValidationError("extracted status requires values")
        elif contract["answer_shape"] == "number":
            parsed = [self._parse_numeric_surface(value) for value in values]
            parsed = [value for value in parsed if value is not None]
            if len(values) == 1 and self._unit_bearing_numeric_surface(values[0]):
                # Preserve a directly extracted written quantity when the model
                # chose number shape too narrowly; stripping its unit changes it.
                values = [self._safe_surface(values[0])]
            elif len(parsed) == 1 and len(values) == 1:
                values = [self._surface_scalar(parsed[0])]
            else:
                numeric_repair = self._repair_numeric_extraction(
                    contract, extraction, evidence, available
                )
                if numeric_repair["status"] == "extracted":
                    values = [self._surface_scalar(numeric_repair["value"])]
                else:
                    values = []
                    extraction = {**extraction, "status": "unknown"}
        diagnostics = {"status": extraction["status"], "extraction": extraction}
        if numeric_repair is not None:
            diagnostics["numeric_repair"] = numeric_repair
        return ToolResult(
            step_id,
            "values",
            records=records,
            values=values,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _result_observation(result: ToolResult) -> dict[str, Any]:
        diagnostics = dict(result.diagnostics)
        return {
            "kind": result.kind,
            "values": result.values[:60],
            "scalar": result.scalar,
            "diagnostics": diagnostics,
            "record_ids": [record.record_id for record in result.records[:60]],
        }

    def _review_observations(
        self,
        results: dict[int, ToolResult],
    ) -> tuple[list[dict[str, Any]], dict[str, SourceRecord]]:
        record_map: dict[str, SourceRecord] = {}
        steps: list[dict[str, Any]] = []
        for index in sorted(results):
            result = results[index]
            for record in result.records:
                record_map.setdefault(record.record_id, record)
            steps.append({"step_id": index, **self._result_observation(result)})
        records = self._bounded_record_views(
            list(record_map.values()),
            max_records=40,
            per_record_chars=900,
            total_chars=12000,
        )
        return [{"steps": steps, "records": records}], record_map

    def _review(
        self,
        profile: dict[str, Any],
        contract: dict[str, Any],
        program: dict[str, Any],
        results: dict[int, ToolResult],
        round_index: int,
        allow_search: bool,
    ) -> dict[str, Any]:
        observations, _ = self._review_observations(results)
        _, derived = self._available_material(results)
        derived_candidates = sorted(derived)[:80]
        schema = evidence_review_schema(contract["contract_id"])
        search_instruction = (
            "When evidence is incomplete, return status search and propose up to four literal searches. Search terms "
            "may use names, identifiers, titles, or relations discovered in current evidence, enabling general "
            "multi-hop retrieval."
            if allow_search
            else "No more retrieval rounds are available. Return answered only if complete evidence is present; otherwise unknown."
        )
        prompt = (
            "Judge whether the accumulated tool results fully answer the immutable semantic contract. Use only supplied "
            "evidence and derived tool values. Treat explicit paraphrases, inflections, punctuation variants, intervening "
            "descriptive words, and equivalent structured fields as support; do not require the question wording to appear "
            "verbatim. Source paths, file names, collection paths, and structured source metadata are evidence for "
            "document and scope constraints. Collection preference is retrieval guidance only, not an authority rule: do "
            "not reject a directly cited coherent record merely because another representation might exist. A deterministic "
            "scalar or value produced by an executed tool is valid when its operands or inputs are shown and its carried "
            "record provenance supports the requested context. Do not request redundant searches after the answer is already "
            "directly supported. Do not reward plausible partial or outside-knowledge answers. For list requests, include every "
            "explicitly supported member needed by the request. Put list members in answer_items and leave answer empty; "
            "for scalar requests put only the minimal value in answer and leave answer_items empty. Cite only supplied record IDs. "
            "For number answers, answer must be the numeric value using digits with optional sign or decimal point; never "
            "return punctuation, a separator, or explanatory text as the numeric answer. "
            "When the original question asks for a specific labeled item and the evidence has a direct label-value pair for "
            "the target phrase, the labeled value can answer the question even if the semantic contract categorized the "
            "request as a definition; require an abstract glossary definition only when the question explicitly asks for "
            "meaning, definition, or what the term refers to. "
            "For person or named-entity scalar requests, answer with the minimal name/entity span, not a descriptive role "
            "label plus that span; when cited text has a '<role label> <name> <predicate>' shape, omit the role label "
            "unless it is required to identify the requested actor. "
            "If the evidence only states that the requested scalar value is absent or not established, return unknown rather "
            "than using that absence statement as the answer. "
            f"{search_instruction}\n"
            f"Round: {round_index}\n"
            f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
            f"Derived answer candidates: {json.dumps(derived_candidates, ensure_ascii=False, default=str)}\n"
            f"Executed program: {json.dumps(program, ensure_ascii=False)}\n"
            f"Question-focused catalog for optional follow-up retrieval only: "
            f"{self.catalog.summary(3000, query=contract['question'])}\n"
            f"Accumulated observations and evidence: {json.dumps(observations, ensure_ascii=False, default=str)}"
        )
        payload = self.model.complete_json("evidence_review", prompt, schema, max_tokens=1400)
        review = self._normalize_review(contract, payload["evidence_review"], results)
        self._validate_review(contract, review, results, allow_search)
        return review

    @staticmethod
    def _available_material(results: dict[int, ToolResult]) -> tuple[set[str], set[str]]:
        record_ids = {
            record.record_id
            for result in results.values()
            for record in result.records
        }
        derived = {
            str(value)
            for result in results.values()
            for value in ([result.scalar] if result.scalar is not None else result.values)
        }
        return record_ids, derived

    def _normalize_review(
        self,
        contract: dict[str, Any],
        review: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> dict[str, Any]:
        """Repair only schema-valid status/search contradictions with exact grounding."""
        normalized = dict(review)
        answer = self._strip_inline_citation_marker(normalized.get("answer", ""))
        answer_items = [
            self._strip_inline_citation_marker(value)
            for value in normalized.get("answer_items", [])
            if self._strip_inline_citation_marker(value)
        ]
        normalized["answer"] = answer
        normalized["answer_items"] = answer_items
        if contract["answer_shape"] == "list":
            if answer_items:
                normalized["answer"] = ""
                answer = ""
            elif answer:
                normalized["answer_items"] = [answer]
                normalized["answer"] = ""
                answer_items = [answer]
                answer = ""
        elif answer_items:
            if not answer and len(answer_items) == 1:
                normalized["answer"] = answer_items[0]
                answer = answer_items[0]
            normalized["answer_items"] = []
            answer_items = []
        record_ids, derived = self._available_material(results)
        answer_values = answer_items if contract["answer_shape"] == "list" else ([answer] if answer else [])
        cited_values_are_derived = bool(answer_values) and all(value in derived for value in answer_values)
        cited = set(normalized.get("evidence_record_ids", []))
        if cited_values_are_derived and not cited.issubset(record_ids):
            repaired_ids = self._record_ids_for_derived_values(results, answer_values)
            if repaired_ids:
                normalized["evidence_record_ids"] = repaired_ids
                cited = set(repaired_ids)
        if contract["answer_shape"] == "list" and answer_values and not cited.issubset(record_ids):
            repaired_ids = self._record_ids_for_surface_values(results, answer_values)
            if repaired_ids:
                normalized["evidence_record_ids"] = repaired_ids
                cited = set(repaired_ids)
        grounded = cited.issubset(record_ids) and bool(cited)
        cited_surfaces = [
            " ".join(record.search_text.casefold().split())
            for result in results.values()
            for record in result.records
            if record.record_id in cited
        ]
        if contract["answer_shape"] == "list":
            exact_derived = bool(answer_items) and all(item in derived for item in answer_items)
            surface_grounded = bool(answer_items) and all(
                any(" ".join(item.casefold().split()) in surface for surface in cited_surfaces)
                for item in answer_items
            )
            has_answer = bool(answer_items) and not answer
        else:
            exact_derived = bool(answer) and answer in derived
            normalized_answer = " ".join(answer.casefold().split())
            surface_grounded = bool(answer) and any(
                normalized_answer in surface for surface in cited_surfaces
            )
            has_answer = bool(answer) and not answer_items
        if has_answer and grounded and (exact_derived or surface_grounded):
            normalized["status"] = "answered"
            normalized["searches"] = []
        elif normalized.get("status") in {"search", "unknown"}:
            normalized["answer"] = ""
            normalized["answer_items"] = []
            if normalized["status"] == "unknown":
                normalized["searches"] = []
        return normalized

    @staticmethod
    def _record_ids_for_derived_values(
        results: dict[int, ToolResult],
        values: list[str],
    ) -> list[str]:
        repaired: list[str] = []
        for target in values:
            target_text = str(target).strip()
            if not target_text:
                return []
            matched: list[str] = []
            for result in results.values():
                candidates = [result.scalar] if result.scalar is not None else result.values
                if any(str(candidate).strip() == target_text for candidate in candidates):
                    matched.extend(record.record_id for record in result.records)
            if not matched:
                return []
            for record_id in matched:
                if record_id not in repaired:
                    repaired.append(record_id)
        return repaired

    @staticmethod
    def _record_ids_for_surface_values(
        results: dict[int, ToolResult],
        values: list[str],
    ) -> list[str]:
        repaired: list[str] = []
        for target in values:
            target_text = " ".join(str(target).casefold().split())
            if not target_text:
                return []
            matched: list[str] = []
            for result in results.values():
                for record in result.records:
                    surface = " ".join(record.search_text.casefold().split())
                    if target_text in surface:
                        matched.append(record.record_id)
            if not matched:
                return []
            for record_id in matched:
                if record_id not in repaired:
                    repaired.append(record_id)
        return repaired

    @staticmethod
    def _strip_inline_citation_marker(value: Any) -> str:
        """Remove model-added source tags when evidence IDs are transported separately."""
        clean = str(value).strip()
        if clean.endswith("】") and "【" in clean:
            candidate = clean.rsplit("【", 1)[0].strip()
            if candidate:
                clean = candidate
        return clean

    def _validate_review(
        self,
        contract: dict[str, Any],
        review: dict[str, Any],
        results: dict[int, ToolResult],
        allow_search: bool,
    ) -> None:
        if review["contract_id"] != contract["contract_id"]:
            raise ProgramValidationError("evidence review contract mismatch")
        if review["answer_shape"] != contract["answer_shape"]:
            raise ProgramValidationError("evidence review answer shape mismatch")
        status = review["status"]
        answer = str(review["answer"]).strip()
        answer_items = [str(value).strip() for value in review["answer_items"] if str(value).strip()]
        searches = review["searches"]
        record_ids, derived = self._available_material(results)
        if not set(review["evidence_record_ids"]).issubset(record_ids):
            raise ProgramValidationError("evidence review cites unavailable records")
        if status == "answered":
            if searches:
                raise ProgramValidationError("answered review cannot contain searches")
            if contract["answer_shape"] == "list":
                if answer or not answer_items:
                    raise ProgramValidationError("list review requires answer_items only")
                derived_match = all(item in derived for item in answer_items)
            else:
                if not answer or answer_items:
                    raise ProgramValidationError("scalar review requires answer only")
                derived_match = answer in derived
                if contract["answer_shape"] == "boolean":
                    normalized_answer = answer.strip().casefold()
                    normalized_derived = {
                        str(value).strip().casefold()
                        for value in derived
                    }
                    derived_match = derived_match or normalized_answer in normalized_derived
                    if normalized_answer in {"true", "false"} and not derived_match:
                        raise ProgramValidationError("bare boolean review requires derived value")
            if not review["evidence_record_ids"] and not derived_match:
                raise ProgramValidationError("answered review requires evidence or exact derived value")
        elif status == "search":
            if answer or answer_items or not searches or not allow_search:
                raise ProgramValidationError("search review has invalid answer or search state")
        elif status == "unknown":
            if answer or answer_items or searches:
                raise ProgramValidationError("unknown review cannot contain answer or searches")

    def _execute_followup_searches(
        self,
        searches: list[dict[str, Any]],
        results: dict[int, ToolResult],
        seen_searches: set[str],
    ) -> int:
        added = 0
        next_index = max(results, default=-1) + 1
        for search in searches:
            collection = str(search["collection"]).strip() or "all_records"
            if not self.catalog.has_collection(collection):
                collection = "all_records"
            terms = [str(value).strip() for value in search["terms"] if str(value).strip()]
            if not terms:
                continue
            mode = search["mode"]
            fields = [str(value).strip() for value in search["fields"] if str(value).strip()]
            limit = max(1, min(int(search["limit"]), 100))
            signature = json.dumps(
                [collection, terms, mode, fields, limit],
                ensure_ascii=False,
                sort_keys=True,
            )
            if signature in seen_searches:
                continue
            seen_searches.add(signature)
            step = {
                "tool": "search_records",
                "inputs": [],
                "collection": collection,
                "terms": terms,
                "fields": fields,
                "filters": [],
                "arguments": [
                    {"name": "mode", "value": mode, "values": [], "numbers": []}
                ],
                "limit": limit,
            }
            result = self.executor.execute([step])[0]
            results[next_index] = ToolResult(
                str(next_index),
                result.kind,
                records=result.records,
                values=result.values,
                scalar=result.scalar,
                diagnostics={**result.diagnostics, "followup_search": search},
            )
            next_index += 1
            added += 1
        return added

    def _review_until_complete(
        self,
        profile: dict[str, Any],
        contract: dict[str, Any],
        program: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> Answer:
        max_rounds = max(1, min(int(os.environ.get("KMD_MAX_RETRIEVAL_ROUNDS", "3")), 5))
        seen_searches: set[str] = set()
        reviews: list[dict[str, Any]] = []
        for round_index in range(max_rounds):
            allow_search = round_index < max_rounds - 1
            review = self._review(profile, contract, program, results, round_index, allow_search)
            reviews.append(review)
            self.model_query_trace["reviews"] = reviews
            if review["status"] == "answered":
                return self._answer_from_review(contract, review, results)
            if review["status"] == "unknown":
                return Answer(
                    "unknown",
                    diagnostics={"review": review, "trace": self.model_query_trace},
                )
            added = self._execute_followup_searches(
                review["searches"], results, seen_searches
            )
            if added == 0:
                break
        return Answer(
            "unknown",
            diagnostics={
                "reason": "retrieval_exhausted",
                "reviews": reviews,
                "trace": self.model_query_trace,
            },
        )

    def _answer_from_review(
        self,
        contract: dict[str, Any],
        review: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> Answer:
        evidence_by_id = {
            record.record_id: record.model_view()
            for result in results.values()
            for record in result.records
        }
        evidence = tuple(
            evidence_by_id[record_id]
            for record_id in review["evidence_record_ids"]
            if record_id in evidence_by_id
        )
        if review["answer_shape"] == "list":
            items = [
                self._repair_temporal_prefix_surface(
                    self._safe_surface(str(value).strip()),
                    review,
                    list(evidence),
                )
                for value in review["answer_items"]
                if str(value).strip()
            ]
            text = "; ".join(items)
        else:
            text = self._repair_temporal_prefix_surface(
                self._safe_surface(str(review["answer"]).strip()),
                review,
                list(evidence),
            )
        rendered = None
        if self._should_render_review_answer(contract, text):
            rendered_text, rendered = self._render_review_answer(
                contract,
                review,
                list(evidence),
                set(evidence_by_id),
                text,
            )
            text = rendered_text
        return Answer(
            text,
            evidence=evidence,
            diagnostics={
                "review": review,
                "grounded_answer": rendered,
                "trace": self.model_query_trace,
            },
        )

    @staticmethod
    def _should_render_review_answer(contract: dict[str, Any], text: str) -> bool:
        slot = str(contract.get("answer_slot", "")).strip()
        return (
            (slot == "person" or slot == "entity" or slot.endswith("_entity"))
            and str(contract.get("answer_shape", "")).strip() in {"text", "list"}
            and " " in str(text).strip()
        )

    def _render_review_answer(
        self,
        contract: dict[str, Any],
        review: dict[str, Any],
        evidence: list[dict[str, Any]],
        available: set[str],
        fallback: str,
    ) -> tuple[str, dict[str, Any] | None]:
        schema = grounded_answer_schema(contract["contract_id"])
        prompt = (
            "Render the final minimal answer from an already validated evidence review. Keep the same answer content "
            "and cite only supplied record IDs. Do not add source names, record IDs, or citations to the answer text. "
            "For person or named-entity answers, return the minimal requested name and omit descriptive role words before the name "
            "unless the role or honorific is necessary to identify the person. A leading occupation, office, or "
            "function word before a name is descriptive; when the following name tokens uniquely identify "
            "the requested person or entity, return those name tokens only, even if the role label is capitalized. "
            "For an actor phrase shaped like '<role label> <name> <predicate>', answer with the name span, not the "
            "role label. If reducing a current answer by dropping a leading role label, copy the remaining name tokens "
            "exactly as a contiguous substring; never edit spelling or invent a variant. Preserve punctuation-bearing honorifics or initials "
            "when they are part of the cited answer surface. Examples: cited 'Recorder Nalto logged ticket T-100' -> "
            "answer 'Nalto'; cited 'Pavo Lexton signed the form' -> answer 'Pavo Lexton'; cited "
            "'Dr. Omi Varek signed the form' -> answer 'Dr. Omi Varek'; cited 'Claimant Aster Guild filed a note' "
            "-> answer 'Aster Guild'. For list-shaped answers, put the final "
            "display string in grounded_answer.answer using '; ' between items when there is more than one. Use no "
            "outside knowledge.\n"
            f"Semantic contract: {json.dumps(contract, ensure_ascii=False)}\n"
            f"Validated review: {json.dumps(review, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(evidence, ensure_ascii=False, default=str)}"
        )
        payload = self.model.complete_json(
            "grounded_answer",
            prompt,
            schema,
            max_tokens=500,
        )
        grounded = payload["grounded_answer"]
        if grounded["contract_id"] != contract["contract_id"]:
            raise ProgramValidationError("review grounded answer contract mismatch")
        if grounded["answer_shape"] != contract["answer_shape"]:
            raise ProgramValidationError("review grounded answer shape mismatch")
        cited = set(grounded["evidence_record_ids"])
        if not cited or not cited.issubset(available):
            raise ProgramValidationError("review grounded answer cites unavailable records")
        answer = self._clean_rendered_answer(str(grounded["answer"]).strip())
        if grounded["status"] != "answered" or not answer:
            return fallback, grounded
        fallback_text = self._safe_surface(str(fallback).strip())
        if (
            fallback_text
            and len(answer.split()) > len(fallback_text.split())
            and fallback_text.casefold() in answer.casefold()
        ):
            answer = fallback_text
            grounded = {**grounded, "answer": answer}
        cited_surfaces = [
            " ".join(str(item.get("excerpt", "")).casefold().split())
            for item in evidence
            if item.get("record_id") in cited
        ]
        normalized_answer = " ".join(answer.casefold().split())
        normalized_fallback = " ".join(str(fallback).casefold().split())
        if normalized_answer not in normalized_fallback and not any(
            normalized_answer in surface for surface in cited_surfaces
        ):
            return fallback, grounded
        return answer, grounded

    def dspg_counts(self) -> dict[str, int]:
        return {
            "records": len(self.catalog.records),
            "preferred_records": len(self.catalog.preferred_records()),
            "collections": len(self.catalog.collections),
        }

    def dspg_integrity(self) -> str:
        return "ok" if self.catalog.records else "empty"
