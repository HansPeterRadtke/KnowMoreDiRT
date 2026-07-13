"""Deterministic source-shape discovery with coherent logical records."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .models import SourceRecord

_IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _stable_id(*parts: object) -> str:
    material = "\0".join(str(part) for part in parts)
    return hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()[:24]


def _primitive(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _flatten_types(value: Any, prefix: str = "") -> dict[str, str]:
    output: dict[str, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, dict):
                output.update(_flatten_types(child, path))
            elif isinstance(child, list):
                output[path] = "array"
                for item in child[:8]:
                    if isinstance(item, dict):
                        output.update(_flatten_types(item, path))
            else:
                output[path] = type(child).__name__
    return output


def values_at_path(value: Any, path: str) -> list[Any]:
    if not path:
        return [value]
    current = [value]
    for part in path.split("."):
        next_values: list[Any] = []
        for item in current:
            if isinstance(item, dict) and part in item:
                child = item[part]
                next_values.extend(child if isinstance(child, list) else [child])
            elif isinstance(item, list):
                for child in item:
                    if isinstance(child, dict) and part in child:
                        found = child[part]
                        next_values.extend(found if isinstance(found, list) else [found])
        current = next_values
    flattened: list[Any] = []
    for item in current:
        flattened.extend(item if isinstance(item, list) else [item])
    return flattened


class SourceCatalog:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"folder does not exist: {self.root}")
        self.collections: dict[str, list[SourceRecord]] = defaultdict(list)
        self.records: dict[str, SourceRecord] = {}
        self._preferred_record_ids: set[str] = set()
        self._scan()
        if not self.records:
            raise ValueError(f"folder contains no readable records: {self.root}")

    def _add(
        self,
        collection: str,
        source: Path,
        index: int,
        data: dict[str, Any],
        text: str = "",
        *,
        preferred: bool = False,
        representation: str = "record",
    ) -> SourceRecord:
        rel = source.relative_to(self.root).as_posix()
        enriched = dict(data)
        enriched.setdefault("source", {})
        if isinstance(enriched["source"], dict):
            enriched["source"] = {
                **enriched["source"],
                "path": rel,
                "file_name": source.name,
                "file_stem": source.stem,
                "record_index": index,
                "representation": representation,
            }
        record_id = _stable_id(rel, collection, index, json.dumps(enriched, sort_keys=True, default=str))
        record = SourceRecord(record_id, collection, rel, index, enriched, text)
        self.collections[collection].append(record)
        self.records[record_id] = record
        if preferred:
            self._preferred_record_ids.add(record_id)
        return record

    def _scan(self) -> None:
        paths = [
            path
            for path in sorted(self.root.rglob("*"))
            if path.is_file()
            and not any(part in _IGNORED_DIRS for part in path.relative_to(self.root).parts)
        ]
        for path in paths:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parsed = self._parse_json(raw)
            if parsed is not None:
                self._index_json(path, parsed)
            else:
                self._index_text(path, raw)

    @staticmethod
    def _parse_json(raw: str) -> Any | None:
        stripped = raw.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            rows = []
            for line in stripped.splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    return None
            return rows if rows else None

    def _index_json(self, source: Path, value: Any) -> None:
        self._walk_json(source, value, "$", set())

    def _walk_json(self, source: Path, value: Any, path: str, seen: set[int]) -> None:
        if isinstance(value, (dict, list)):
            marker = id(value)
            if marker in seen:
                return
            seen.add(marker)
        rel = source.relative_to(self.root).as_posix()
        if isinstance(value, list):
            collection = f"{rel}::{path}[]"
            for index, item in enumerate(value):
                data = item if isinstance(item, dict) else {"value": item}
                self._add(
                    collection,
                    source,
                    index,
                    data,
                    json.dumps(item, ensure_ascii=False, default=str),
                    preferred=True,
                    representation="json_item",
                )
                self._walk_json(source, item, f"{path}[]", seen)
        elif isinstance(value, dict):
            local_data = {
                key: child
                for key, child in value.items()
                if _primitive(child)
                or (isinstance(child, list) and all(_primitive(item) for item in child))
            }
            if local_data:
                self._add(
                    f"{rel}::{path}{{}}",
                    source,
                    0,
                    local_data,
                    json.dumps(local_data, ensure_ascii=False, default=str),
                    preferred=True,
                    representation="json_object",
                )
            if value and all(isinstance(item, dict) for item in value.values()):
                collection = f"{rel}::{path}{{}}"
                for index, (key, item) in enumerate(value.items()):
                    self._add(
                        collection,
                        source,
                        index + 1,
                        {"map_key": key, **item},
                        json.dumps(item, ensure_ascii=False, default=str),
                        preferred=True,
                        representation="json_map_item",
                    )
            for key, child in value.items():
                child_path = f"{path}.{key}" if path != "$" else str(key)
                if isinstance(child, (dict, list)):
                    self._walk_json(source, child, child_path, seen)

    @staticmethod
    def _label_groups(lines: list[str]) -> list[dict[str, str]]:
        groups: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in lines + [""]:
            match = re.match(r"^\s*([A-Za-z][A-Za-z0-9 _./-]{0,79}):\s*(.*?)\s*$", line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                scheme_token = key.rsplit(None, 1)[-1].lower()
                if value.startswith("//") and scheme_token in {
                    "http", "https", "ftp", "file", "s3", "gs"
                }:
                    if current:
                        groups.append(current)
                        current = {}
                    continue
                if key in current:
                    if current:
                        groups.append(current)
                    current = {}
                current[key] = value
            elif current:
                groups.append(current)
                current = {}
        return groups

    @staticmethod
    def _inline_map(raw: str) -> dict[str, str]:
        stripped = raw.strip()
        if not (stripped.startswith("{") and "}" in stripped):
            return {}
        body = stripped[1 : stripped.find("}")]
        output: dict[str, str] = {}
        for match in re.finditer(
            r"(?:^|,)\s*([A-Za-z][A-Za-z0-9 _./-]{0,79})\s*:\s*(?:\"([^\"]*)\"|'([^']*)'|([^,]+))",
            body,
        ):
            key = match.group(1).strip()
            value = next((group for group in match.groups()[1:] if group is not None), "").strip()
            output[key] = value
        return output

    @staticmethod
    def _balanced_loose_objects(raw: str) -> list[str]:
        objects: list[str] = []
        start: int | None = None
        depth = 0
        quote = ""
        escaped = False
        for index, char in enumerate(raw):
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = ""
                continue
            if char in {"\"", "'"}:
                quote = char
                continue
            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(raw[start : index + 1])
                    start = None
        return objects

    @staticmethod
    def _parse_loose_object(candidate: str) -> dict[str, Any] | None:
        transformed = re.sub(
            r'([,{]\s*)([A-Za-z_][A-Za-z0-9 _./-]*?)(\s*:)',
            lambda match: f'{match.group(1)}"{match.group(2).strip()}"{match.group(3)}',
            candidate,
        )
        transformed = re.sub(r",\s*([}])", r"\1", transformed)
        try:
            value = json.loads(transformed)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict) or len(value) < 2:
            return None
        return value

    def _index_loose_objects(self, source: Path, raw: str) -> None:
        rel = source.relative_to(self.root).as_posix()
        parsed = [
            value
            for candidate in self._balanced_loose_objects(raw)
            if (value := self._parse_loose_object(candidate)) is not None
        ]
        repeated = len(parsed) > 1
        for index, value in enumerate(parsed):
            self._add(
                f"{rel}::loose_objects[]",
                source,
                index,
                value,
                json.dumps(value, ensure_ascii=False, default=str),
                preferred=repeated,
                representation="loose_object",
            )

    @staticmethod
    def _key_value_rows(lines: list[str]) -> list[tuple[str, dict[str, str]]]:
        output: list[tuple[str, dict[str, str]]] = []
        for line in lines:
            parts = [part.strip() for part in line.split("|") if part.strip()]
            data: dict[str, str] = {}
            valid = True
            for part in parts:
                if "=" not in part:
                    valid = False
                    break
                key, value = part.split("=", 1)
                key = key.strip()
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 _./-]{0,79}", key):
                    valid = False
                    break
                data[key] = value.strip()
            if valid and len(data) >= 2:
                output.append((line, data))
        return output

    def _index_text(self, source: Path, raw: str) -> None:
        rel = source.relative_to(self.root).as_posix()
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        groups = self._label_groups(lines)
        inline_map = self._inline_map(raw)
        logical: dict[str, Any] = {"text": raw}
        if len(groups) == 1:
            logical.update(groups[0])
        elif groups:
            logical["label_records"] = groups
        logical.update({key: value for key, value in inline_map.items() if key not in logical})
        self._add(
            "logical_documents",
            source,
            0,
            logical,
            raw,
            preferred=True,
            representation="logical_document",
        )

        blocks = [block.strip() for block in re.split(r"\n\s*\n", raw) if block.strip()]
        for index, block in enumerate(blocks):
            self._add(
                f"{rel}::blocks[]",
                source,
                index,
                {"text": block},
                block,
                representation="block",
            )
        for index, line in enumerate(lines):
            self._add(
                f"{rel}::lines[]",
                source,
                index,
                {"line_number": index + 1, "text": line},
                line,
                representation="line",
            )
        for index, group in enumerate(groups):
            self._add(
                f"{rel}::labeled_records[]",
                source,
                index,
                group,
                json.dumps(group, ensure_ascii=False),
                representation="labeled_record",
            )
        for index, (line, data) in enumerate(self._key_value_rows(lines)):
            self._add(
                f"{rel}::key_value_rows[]",
                source,
                index,
                data,
                line,
                preferred=True,
                representation="key_value_row",
            )
        self._index_loose_objects(source, raw)
        self._index_table(source, raw)

    def _index_table(self, source: Path, raw: str) -> None:
        lines = [line for line in raw.splitlines() if line.strip()]
        if len(lines) < 2:
            return
        candidates: list[tuple[int, int, int, str, list[str], list[list[str]]]] = []
        for header_index, header_line in enumerate(lines[:12]):
            for delimiter in ("\t", "|", ","):
                if header_line.count(delimiter) < 1:
                    continue
                try:
                    header_row = next(csv.reader([header_line], delimiter=delimiter))
                except (csv.Error, StopIteration):
                    continue
                width = len(header_row)
                if width < 2:
                    continue
                headers = [
                    item.strip() or f"column_{index}"
                    for index, item in enumerate(header_row)
                ]
                if len(set(headers)) != len(headers):
                    continue
                if not all(
                    re.fullmatch(r"[A-Za-z][A-Za-z0-9 _./-]{0,79}", header)
                    for header in headers
                ):
                    continue
                data_rows: list[list[str]] = []
                for line in lines[header_index + 1 : header_index + 501]:
                    try:
                        row = next(csv.reader([line], delimiter=delimiter))
                    except (csv.Error, StopIteration):
                        break
                    if len(row) != width:
                        break
                    data_rows.append(row)
                if data_rows:
                    candidates.append(
                        (
                            len(data_rows),
                            width,
                            -header_index,
                            delimiter,
                            headers,
                            data_rows,
                        )
                    )
        if not candidates:
            return
        _, _, _, delimiter, headers, rows = max(candidates)
        rel = source.relative_to(self.root).as_posix()
        for index, row in enumerate(rows):
            data = {headers[column]: value.strip() for column, value in enumerate(row)}
            self._add(
                f"{rel}::table_rows[]",
                source,
                index,
                data,
                delimiter.join(row),
                preferred=True,
                representation="table_row",
            )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for record_id in sorted(self.records):
            digest.update(record_id.encode("ascii"))
        return digest.hexdigest()

    def preferred_records(self) -> list[SourceRecord]:
        records = [self.records[record_id] for record_id in self._preferred_record_ids]
        return sorted(records, key=lambda item: (item.source_path, item.record_index, item.record_id))

    def collection_records(self, collection_path: str) -> list[SourceRecord]:
        if collection_path in {"", "all_records"}:
            preferred = self.preferred_records()
            return preferred or list(self.records.values())
        if collection_path == "all_representations":
            return list(self.records.values())
        if collection_path in self.collections:
            return list(self.collections[collection_path])
        source_matches = [
            record
            for record in self.preferred_records()
            if record.source_path == collection_path
        ]
        return source_matches

    def records_for_sources(self, source_paths: set[str]) -> list[SourceRecord]:
        return [record for record in self.preferred_records() if record.source_path in source_paths]

    def field_paths(self, collection_path: str) -> set[str]:
        fields: set[str] = set()
        records = self.collection_records(collection_path)
        for record in records[:1000]:
            fields.update(_flatten_types(record.data))
        return fields

    def summary(self, max_chars: int = 9000) -> str:
        preferred_ids = self._preferred_record_ids
        payload: list[dict[str, Any]] = [
            {
                "collection_path": "all_records",
                "record_count": len(self.preferred_records()),
                "preferred": True,
                "field_paths": {},
                "samples": [record.model_view(350) for record in self.preferred_records()[:3]],
            }
        ]
        for collection, records in sorted(self.collections.items()):
            fields: dict[str, str] = {}
            for record in records[:80]:
                fields.update(_flatten_types(record.data))
            payload.append(
                {
                    "collection_path": collection,
                    "record_count": len(records),
                    "preferred": any(record.record_id in preferred_ids for record in records),
                    "field_paths": fields,
                    "samples": [records[0].model_view(350)] if records else [],
                }
            )
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
        if len(rendered) <= max_chars:
            return rendered
        for item in payload[1:]:
            item["samples"] = []
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
        if len(rendered) <= max_chars:
            return rendered
        compact = [
            {
                "collection_path": item["collection_path"],
                "record_count": item["record_count"],
                "preferred": item["preferred"],
                "field_paths": item["field_paths"],
            }
            for item in payload
        ]
        return json.dumps(compact, ensure_ascii=False, default=str)[:max_chars]
