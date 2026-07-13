"""Generic deterministic tools selected by the model-owned compiler."""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Callable

from .catalog import SourceCatalog, values_at_path
from .models import SourceRecord, ToolResult


def _compare(values: list[Any], operator: str, value: str, expected: list[str]) -> bool:
    normalized = [str(item).strip().lower() for item in values]
    target = value.strip().lower()
    targets = [item.strip().lower() for item in expected]
    if operator == "exists":
        return bool(values)
    if operator == "equals":
        return any(item == target for item in normalized)
    if operator == "not_equals":
        return all(item != target for item in normalized)
    if operator == "contains":
        return any(target in item for item in normalized)
    if operator == "contains_all":
        return all(any(target_item in item for item in normalized) for target_item in targets)
    if operator == "contains_any":
        return any(any(target_item in item for item in normalized) for target_item in targets)
    if operator == "in":
        return any(item in targets for item in normalized)
    try:
        numbers = [float(item) for item in normalized]
        target_number = float(target)
    except ValueError:
        return False
    if operator == "less_than":
        return any(item < target_number for item in numbers)
    if operator == "less_or_equal":
        return any(item <= target_number for item in numbers)
    if operator == "greater_than":
        return any(item > target_number for item in numbers)
    if operator == "greater_or_equal":
        return any(item >= target_number for item in numbers)
    return False


def _dedupe(values: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _normalize_search(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _stem_token(token: str) -> str:
    irregular = {
        "found": "find", "bought": "buy", "wrote": "write", "written": "write",
        "spoke": "speak", "spoken": "speak", "made": "make", "given": "give",
        "prove": "proof", "proved": "proof", "proven": "proof", "proof": "proof",
    }
    if token in irregular:
        return irregular[token]
    for suffix in ("ations", "ation", "ions", "ion", "ing", "ers", "ors", "ed", "er", "or", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            stem = token[: -len(suffix)]
            if len(stem) >= 4 and stem[-1:] == stem[-2:-1]:
                stem = stem[:-1]
            return stem
    return token


def _one_edit_apart(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    i = j = differences = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
        else:
            differences += 1
            j += 1
            if differences > 1:
                return False
    return True


def _term_match_score(term: str, haystack: str) -> int:
    """Return a generic lexical score using exact phrase or bounded token coverage."""
    if not term:
        return 0
    if term in haystack:
        return 20 + len(term.split())
    ignored_tokens = {
        "a", "an", "the", "about", "of", "for", "in", "on", "by", "to", "from",
        "with", "at", "who", "what", "which", "is", "are", "was", "were", "did",
        "does", "do", "current", "latest", "final", "earliest", "newest", "oldest",
        "most", "recent", "now", "currently", "note", "document", "file", "text",
        "record", "records", "data", "semantic", "really", "actually", "factually", "truly", "real",
        "meaningful", "credible", "reliable", "authoritative", "trustworthy", "trusted",
        "valid", "clean", "after", "before", "during", "since", "until",
        "following", "preceding",
        "regarding", "concerning", "according", "per", "says", "said", "believes",
        "believed", "reported", "wrote", "written", "forwarded", "quoted",
    }
    query_tokens = [token for token in term.split() if token and token not in ignored_tokens]
    if not query_tokens:
        return 1
    haystack_tokens = set(haystack.split())

    def token_matches(query_token: str) -> bool:
        if query_token in haystack_tokens:
            return True
        query_stem = _stem_token(query_token)
        for candidate in haystack_tokens:
            candidate_stem = _stem_token(candidate)
            if query_stem == candidate_stem and len(query_stem) >= 3:
                return True
            if min(len(query_stem), len(candidate_stem)) >= 5 and _one_edit_apart(query_stem, candidate_stem):
                return True
            if min(len(query_token), len(candidate)) >= 5 and (
                query_token.startswith(candidate) or candidate.startswith(query_token)
            ):
                return True
        return False

    hits = sum(1 for token in query_tokens if token_matches(token))
    required = max(1, (2 * len(query_tokens) + 2) // 3)
    return hits if hits >= required else 0


def _parse_time(value: str) -> tuple[int, str]:
    text = value.strip()
    candidates = [text, text.replace("Z", "+00:00")]
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            return (0, parsed.isoformat())
        except ValueError:
            pass
    match = re.search(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b", text)
    if match:
        candidate = match.group(0).replace(" ", "T")
        try:
            parsed = datetime.fromisoformat(candidate)
            return (0, parsed.isoformat())
        except ValueError:
            return (1, candidate)
    return (2, text)


def _clean_value(value: Any, kind: str, strip_chars: str) -> str:
    text = str(value).strip()
    if strip_chars:
        text = text.strip(strip_chars)
    text = text.strip(" \t\r\n:=")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"\"", "'"}:
        text = text[1:-1].strip()
    if kind == "url":
        text = text.rstrip(".,;:)]}")
    elif kind in {"identifier", "date_time", "number"}:
        text = text.rstrip(".,;:)]}")
    return text


def _record_text(record: SourceRecord) -> str:
    return record.text or json.dumps(record.data, ensure_ascii=False, default=str)


def _regex_values(
    text: str,
    pattern: str,
    value_group: str,
    time_group: str,
) -> list[tuple[str, str | None]]:
    try:
        compiled = re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)
    except re.error:
        return []
    output: list[tuple[str, str | None]] = []
    for match in compiled.finditer(text):
        try:
            group_key: str | int = int(value_group) if value_group.isdigit() else value_group
            value = match.group(group_key) if value_group else match.group(0)
        except (IndexError, KeyError):
            value = match.group(0)
        time_value: str | None = None
        if time_group:
            try:
                time_key: str | int = int(time_group) if time_group.isdigit() else time_group
                time_value = match.group(time_key)
            except (IndexError, KeyError):
                time_value = None
        output.append((str(value), time_value))
    return output


def expand_step(step: dict[str, Any]) -> dict[str, Any]:
    """Expand a compact strict tool call into executor arguments."""
    expanded: dict[str, Any] = {
        "tool": step.get("tool", ""),
        "inputs": list(step.get("inputs", [])),
        "collection": step.get("collection", ""),
        "terms": list(step.get("terms", [])),
        "mode": step.get("mode", "none"),
        "filters": list(step.get("filters", [])),
        "fields": list(step.get("fields", [])),
        "left_field": step.get("left_field", ""),
        "right_field": step.get("right_field", ""),
        "sort_field": step.get("sort_field", ""),
        "direction": step.get("direction", "none"),
        "aggregate": step.get("aggregate", "none"),
        "operation": step.get("operation", "none"),
        "numbers": list(step.get("numbers", [])),
        "extractor": step.get("extractor", "none"),
        "label": step.get("label", ""),
        "start_phrase": step.get("start_phrase", ""),
        "end_phrase": step.get("end_phrase", ""),
        "pattern": step.get("pattern", ""),
        "value_group": step.get("value_group", ""),
        "time_group": step.get("time_group", ""),
        "occurrence": step.get("occurrence", "none"),
        "value_kind": step.get("value_kind", "text"),
        "strip_chars": step.get("strip_chars", ""),
        "distinct": bool(step.get("distinct", False)),
        "limit": int(step.get("limit", 20)),
    }
    for argument in step.get("arguments", []):
        name = argument["name"]
        value = argument.get("value", "")
        values = list(argument.get("values", []))
        if name == "distinct":
            expanded[name] = value.strip().lower() in {"true", "1", "yes"}
        else:
            expanded[name] = value
        if name == "extractor" and argument.get("numbers") and value in {"regex", "event_series"}:
            if not expanded["value_group"]:
                expanded["value_group"] = str(int(argument["numbers"][0]))
            if len(argument["numbers"]) > 1 and not expanded["time_group"]:
                expanded["time_group"] = str(int(argument["numbers"][1]))
        if name == "extractor" and values:
            if value == "after_label" and not expanded["label"]:
                expanded["label"] = values[0]
            elif value == "after_phrase" and not expanded["start_phrase"]:
                expanded["start_phrase"] = values[0]
            elif value == "before_phrase" and not expanded["end_phrase"]:
                expanded["end_phrase"] = values[0]
            elif value == "between_phrases":
                if not expanded["start_phrase"]:
                    expanded["start_phrase"] = values[0]
                if len(values) > 1 and not expanded["end_phrase"]:
                    expanded["end_phrase"] = values[1]
            elif value in {"regex", "event_series"} and not expanded["pattern"]:
                expanded["pattern"] = values[0]
        if argument.get("numbers"):
            expanded["numbers"] = list(argument["numbers"])
    if expanded["tool"] == "model_extract":
        expanded["mode"] = "none"
        expanded["extractor"] = "none"
        expanded["direction"] = "none"
        expanded["aggregate"] = "none"
        expanded["operation"] = "none"
    elif expanded["extractor"] in {"extract_values", "model_extract"}:
        expanded["extractor"] = "none"
    if expanded["tool"] == "search_records" and expanded["mode"] == "none" and expanded["terms"]:
        expanded["mode"] = "all"
    if expanded["mode"] == "all_or_phrase":
        expanded["mode"] = "phrase" if len(expanded["terms"]) == 1 else "all"
    if expanded["direction"] == "desc":
        expanded["direction"] = "descending"
    if expanded["direction"] == "asc":
        expanded["direction"] = "ascending"
    if expanded["tool"] == "extract_values" and expanded["extractor"] == "none" and expanded["fields"]:
        expanded["extractor"] = "field"
    if expanded["extractor"] in {"regex", "event_series"} and expanded["pattern"] and not expanded["value_group"]:
        named_groups = re.findall(r"\(\?P<([A-Za-z_][A-Za-z0-9_]*)>", expanded["pattern"])
        if named_groups:
            expanded["value_group"] = named_groups[-1]
    return expanded


class ToolExecutor:
    def __init__(self, catalog: SourceCatalog):
        self.catalog = catalog

    def execute(
        self,
        steps: list[dict[str, Any]],
        semantic_extractor: Callable[[str, dict[str, Any], dict[int, ToolResult]], ToolResult] | None = None,
    ) -> dict[int, ToolResult]:
        results: dict[int, ToolResult] = {}
        expanded_steps = [expand_step(step) for step in steps]
        for index, step in enumerate(expanded_steps):
            if step["tool"] == "model_extract":
                if semantic_extractor is None:
                    raise ValueError("model_extract requires a semantic extractor callback")
                results[index] = semantic_extractor(str(index), step, results)
            else:
                results[index] = self._execute_step(index, step, results)
        return results

    def _input_records(self, step: dict[str, Any], results: dict[int, ToolResult]) -> list[SourceRecord]:
        records: list[SourceRecord] = []
        for ref in step["inputs"]:
            records.extend(results[ref].records)
        if not step["inputs"] and step["collection"]:
            records = self.catalog.collection_records(step["collection"])
        deduped: list[SourceRecord] = []
        seen: set[str] = set()
        for record in records:
            if record.record_id not in seen:
                seen.add(record.record_id)
                deduped.append(record)
        return deduped

    def _input_values(self, step: dict[str, Any], results: dict[int, ToolResult]) -> list[Any]:
        values: list[Any] = []
        for ref in step["inputs"]:
            result = results[ref]
            values.extend(result.values)
            if result.scalar is not None:
                values.append(result.scalar)
        return values

    def _execute_step(
        self,
        index: int,
        step: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> ToolResult:
        tool = step["tool"]
        step_id = str(index)
        limit = max(0, min(int(step["limit"]), 5000))

        if tool == "sample_records":
            records = self.catalog.collection_records(step["collection"])
            return ToolResult(
                step_id,
                "records",
                records=records[: limit or 20],
                diagnostics={"available": len(records)},
            )

        if tool == "search_records":
            records = self._input_records(step, results) if step["inputs"] else self.catalog.collection_records(step["collection"])
            terms = [_normalize_search(item) for item in step["terms"] if _normalize_search(item)]
            selected_fields = [_normalize_search(item) for item in step["fields"] if _normalize_search(item)]
            phrase = terms[0] if step["mode"] == "phrase" and terms else ""
            scored: list[tuple[int, int, SourceRecord]] = []
            for record in records:
                haystack = _normalize_search(record.search_text)
                term_scores = [_term_match_score(term, haystack) for term in terms]
                hits = [term for term, score in zip(terms, term_scores) if score > 0]
                field_scores = [_term_match_score(field, haystack) for field in selected_fields]
                field_hits = [field for field, score in zip(selected_fields, field_scores) if score > 0]
                mode = step["mode"]
                match = (
                    (mode == "phrase" and bool(phrase) and phrase in haystack)
                    or (mode == "all" and bool(terms) and all(score > 0 for score in term_scores))
                    or (mode == "any" and any(score > 0 for score in term_scores))
                )
                if selected_fields and not field_hits:
                    match = False
                if match and step["filters"]:
                    match = all(
                        _compare(
                            values_at_path(record.data, item["field_path"]),
                            item["operator"],
                            item["value"],
                            item["values"],
                        )
                        for item in step["filters"]
                    )
                if match:
                    score = sum(term_scores) + 10 * sum(field_scores) + (20 if phrase and phrase in haystack else 0)
                    scored.append((score, len(haystack), record))
            scored.sort(key=lambda item: (-item[0], item[1], item[2].source_path, item[2].record_index))
            selected = [record for _, _, record in scored[: limit or 50]]
            return ToolResult(
                step_id,
                "records",
                records=selected,
                diagnostics={"matched": len(scored), "terms": terms, "fields": selected_fields},
            )

        if tool == "expand_source_context":
            records = self._input_records(step, results)
            source_paths = {record.source_path for record in records}
            expanded = self.catalog.records_for_sources(source_paths)
            return ToolResult(
                step_id,
                "records",
                records=expanded[: limit or len(expanded)],
                diagnostics={"source_paths": sorted(source_paths), "expanded": len(expanded)},
            )

        if tool == "filter_records":
            records = self._input_records(step, results)
            selected = [
                record
                for record in records
                if all(
                    _compare(
                        values_at_path(record.data, item["field_path"]),
                        item["operator"],
                        item["value"],
                        item["values"],
                    )
                    for item in step["filters"]
                )
            ]
            return ToolResult(
                step_id,
                "records",
                records=selected[: limit or len(selected)],
                diagnostics={"input": len(records), "matched": len(selected)},
            )

        if tool == "project_values":
            records = self._input_records(step, results)
            values: list[Any] = []
            evidence: list[dict[str, Any]] = []
            for record in records:
                for path in step["fields"]:
                    for value in values_at_path(record.data, path):
                        if isinstance(value, (dict, list)):
                            continue
                        values.append(value)
                        evidence.append(
                            {"value": value, "record_id": record.record_id, "field_path": path}
                        )
            if step["distinct"]:
                values = _dedupe(values)
            return ToolResult(
                step_id,
                "values",
                values=values[: limit or len(values)],
                records=records[: limit or len(records)],
                diagnostics={"evidence": evidence[:200]},
            )

        if tool == "extract_values":
            return self._extract_values(step_id, step, results, limit)

        if tool == "join_records":
            left = results[step["inputs"][0]].records
            right = results[step["inputs"][1]].records
            joined: list[SourceRecord] = []
            for left_record in left:
                left_values = values_at_path(left_record.data, step["left_field"])
                for right_record in right:
                    right_values = values_at_path(right_record.data, step["right_field"])
                    overlap = set(map(str, left_values)) & set(map(str, right_values))
                    if overlap:
                        data = {
                            "left": left_record.data,
                            "right": right_record.data,
                            "join_values": sorted(overlap),
                        }
                        joined.append(
                            SourceRecord(
                                f"{left_record.record_id}:{right_record.record_id}",
                                "joined",
                                left_record.source_path,
                                len(joined),
                                data,
                                left_record.text + "\n" + right_record.text,
                            )
                        )
            return ToolResult(
                step_id,
                "records",
                records=joined[: limit or len(joined)],
                diagnostics={"left": len(left), "right": len(right), "matched": len(joined)},
            )

        if tool in {"union_values", "intersect_values"}:
            groups = [self._values_for_ref(results[ref]) for ref in step["inputs"]]
            if tool == "union_values":
                values = _dedupe([value for group in groups for value in group])
            elif not groups:
                values = []
            else:
                common = set(json.dumps(value, sort_keys=True, default=str) for value in groups[0])
                for group in groups[1:]:
                    common &= set(json.dumps(value, sort_keys=True, default=str) for value in group)
                values = [json.loads(value) for value in sorted(common)]
            return ToolResult(step_id, "values", values=values[: limit or len(values)])

        if tool == "sort_records":
            records = self._input_records(step, results)
            reverse = step["direction"] == "descending"

            def key(record: SourceRecord) -> tuple[int, str]:
                values = values_at_path(record.data, step["sort_field"])
                value = values[0] if values else ""
                try:
                    return (0, f"{float(value):030.9f}")
                except (ValueError, TypeError):
                    return _parse_time(str(value))

            records = sorted(records, key=key, reverse=reverse)
            return ToolResult(step_id, "records", records=records[: limit or len(records)])

        if tool == "calculate":
            numbers: list[float] = [float(value) for value in step["numbers"]]
            for value in self._input_values(step, results):
                try:
                    numbers.append(float(value))
                except (TypeError, ValueError):
                    continue
            operation = step["operation"]
            if operation == "add":
                scalar: Any = sum(numbers)
            elif operation == "subtract":
                scalar = numbers[0] - sum(numbers[1:]) if numbers else None
            elif operation == "multiply":
                scalar = 1.0
                for number in numbers:
                    scalar *= number
            elif operation == "divide":
                scalar = numbers[0] if numbers else None
                for number in numbers[1:]:
                    if number == 0:
                        raise ValueError("division by zero")
                    scalar /= number
            else:
                scalar = None
            if isinstance(scalar, float) and scalar.is_integer():
                scalar = int(scalar)
            return ToolResult(step_id, "scalar", scalar=scalar, values=numbers)

        if tool == "aggregate_values":
            values = self._input_values(step, results)
            if not values:
                records = self._input_records(step, results)
                if step["fields"]:
                    for record in records:
                        values.extend(values_at_path(record.data, step["fields"][0]))
                else:
                    values = list(records)
            aggregation = step["aggregate"]
            if aggregation == "count":
                scalar = len(_dedupe(values) if step["distinct"] else values)
            elif aggregation == "distinct":
                return ToolResult(step_id, "values", values=_dedupe(values))
            elif aggregation == "mode":
                counts: dict[str, int] = {}
                originals: dict[str, Any] = {}
                first_indexes: dict[str, int] = {}
                for index, value in enumerate(values):
                    key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                    counts[key] = counts.get(key, 0) + 1
                    originals.setdefault(key, value)
                    first_indexes.setdefault(key, index)
                best = max(
                    counts,
                    key=lambda key: (counts[key], -first_indexes[key]),
                ) if counts else None
                scalar = originals[best] if best is not None else None
            else:
                numeric: list[float] = []
                for value in values:
                    try:
                        numeric.append(float(value))
                    except (ValueError, TypeError):
                        pass
                if aggregation == "sum":
                    scalar = sum(numeric)
                elif aggregation == "average":
                    scalar = sum(numeric) / len(numeric) if numeric else None
                elif aggregation == "min":
                    scalar = min(values) if values else None
                elif aggregation == "max":
                    scalar = max(values) if values else None
                else:
                    scalar = None
            return ToolResult(step_id, "scalar", scalar=scalar, values=values[:50])

        raise ValueError(f"unsupported tool: {tool}")

    def _extract_values(
        self,
        step_id: str,
        step: dict[str, Any],
        results: dict[int, ToolResult],
        limit: int,
    ) -> ToolResult:
        records = self._input_records(step, results)
        candidates: list[tuple[str, str | None, SourceRecord, str]] = []
        extractor = step["extractor"]
        for record in records:
            text = _record_text(record)
            extracted: list[tuple[str, str | None]] = []
            source_label = extractor
            if extractor == "field":
                for field in step["fields"]:
                    extracted.extend((str(value), None) for value in values_at_path(record.data, field))
                source_label = ",".join(step["fields"])
            elif extractor == "after_label":
                label = re.escape(step["label"].strip())
                if label:
                    pattern = rf"(?im)(?:^|[|,;])\s*{label}\s*[:=]\s*([^\n|,;]+)"
                    extracted = _regex_values(text, pattern, "1", "")
            elif extractor == "after_phrase":
                phrase = re.escape(step["start_phrase"])
                if phrase:
                    pattern = rf"(?im){phrase}\s*[:=\-]?\s*([^\n|;,}}]+)"
                    extracted = _regex_values(text, pattern, "1", "")
            elif extractor == "before_phrase":
                phrase = re.escape(step["end_phrase"])
                if phrase:
                    pattern = rf"(?im)([^\n|;]+?)\s*{phrase}"
                    extracted = _regex_values(text, pattern, "1", "")
            elif extractor == "between_phrases":
                start = re.escape(step["start_phrase"])
                end = re.escape(step["end_phrase"])
                if start and end:
                    pattern = rf"(?ims){start}(.*?){end}"
                    extracted = _regex_values(text, pattern, "1", "")
            elif extractor == "regex":
                extracted = _regex_values(
                    text,
                    step["pattern"],
                    step["value_group"],
                    step["time_group"],
                )
            elif extractor == "url":
                extracted = [(value, None) for value in re.findall(r"https?://[^\s<>\"']+", text)]
            elif extractor == "identifier":
                extracted = [
                    (value, None)
                    for value in re.findall(r"\b[A-Za-z][A-Za-z0-9]*[-_][A-Za-z0-9][A-Za-z0-9_-]*\b", text)
                ]
            elif extractor == "date_time":
                extracted = [
                    (value, value)
                    for value in re.findall(
                        r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?\b",
                        text,
                    )
                ]
            elif extractor == "number":
                extracted = [(value, None) for value in re.findall(r"[+-]?\d+(?:\.\d+)?", text)]
            elif extractor == "event_series":
                extracted = _regex_values(
                    text,
                    step["pattern"],
                    step["value_group"],
                    step["time_group"],
                )
            for value, time_value in extracted:
                cleaned = _clean_value(value, step["value_kind"], step["strip_chars"])
                if cleaned:
                    candidates.append((cleaned, time_value, record, source_label))

        occurrence = step["occurrence"]
        if occurrence in {"latest_by_time", "earliest_by_time"}:
            timed = [item for item in candidates if item[1]]
            timed.sort(key=lambda item: _parse_time(item[1] or ""))
            if timed:
                selected_candidates = timed[-1:] if occurrence == "latest_by_time" else timed[:1]
            else:
                selected_candidates = candidates[-1:] if occurrence == "latest_by_time" else candidates[:1]
        elif occurrence == "last":
            selected_candidates = candidates[-1:]
        elif occurrence == "all":
            selected_candidates = candidates
        else:
            selected_candidates = candidates[:1]

        values = [item[0] for item in selected_candidates]
        if step["distinct"]:
            values = _dedupe(values)
        evidence = [
            {
                "value": value,
                "record_id": record.record_id,
                "extractor": source_label,
                "time": time_value,
            }
            for value, time_value, record, source_label in selected_candidates
        ]
        evidence_ids = {item["record_id"] for item in evidence}
        evidence_records = [record for record in records if record.record_id in evidence_ids]
        return ToolResult(
            step_id,
            "values",
            values=values[: limit or len(values)],
            records=evidence_records,
            diagnostics={"evidence": evidence[:200], "candidate_count": len(candidates)},
        )

    @staticmethod
    def _values_for_ref(result: ToolResult) -> list[Any]:
        if result.values:
            return result.values
        if result.scalar is not None:
            return [result.scalar]
        return [record.record_id for record in result.records]
