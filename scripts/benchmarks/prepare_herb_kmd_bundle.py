#!/usr/bin/env python3
"""Create a leakage-free, artifact-preserving HERB bundle for KnowMoreDiRT.

The HERB normalized layer is used only as an intermediate representation. Each
admissible RAG artifact is extracted to its own plain-text source file using the
normalized raw_text field. Benchmark questions and scoring gold remain sibling
files outside the source folder and oracle-only product membership records are
excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

FORMAT_VERSION = "herb-kmd-official-rag-v1"
DEFAULT_NORMALIZED_ROOT = Path("/data/var/herb_benchmark/normalized/herb_normalized")
DEFAULT_OUTPUT_ROOT = Path("/data/var/herb_benchmark/prepared/kmd_official_rag_v1")
RAG_SOURCE_ARTIFACT_TYPES = frozenset(
    {
        "employee",
        "customer",
        "document",
        "meeting_transcript",
        "meeting_chat",
        "url",
        "pull_request",
        "slack",
    }
)
ORACLE_ONLY_ARTIFACT_TYPES = frozenset({"product"})
FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "answerable_questions",
        "unanswerable_questions",
        "ground_truth",
        "gold_answer",
        "citations",
        "question_id",
        "question",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def slug(value: Any, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return normalized or fallback


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield value


def walk_keys_and_strings(value: Any) -> tuple[set[str], list[str]]:
    keys: set[str] = set()
    strings: list[str] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                keys.add(str(key))
                stack.append(child)
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            strings.append(item)
    return keys, strings


def source_tree_hash(source_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source_root).as_posix().encode("utf-8")
        body = path.read_bytes()
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


def sanitized_question(row: dict[str, Any]) -> dict[str, str]:
    return {
        "question_id": str(row["question_id"]),
        "question": str(row["question"]),
    }


def prepare_bundle(
    normalized_root: Path,
    output_root: Path,
    *,
    force: bool = False,
    max_record_chars: int = 36000,
) -> dict[str, Any]:
    normalized_root = normalized_root.resolve()
    output_root = output_root.resolve()
    artifacts_path = normalized_root / "artifacts.jsonl"
    questions_path = normalized_root / "questions.jsonl"
    gold_path = normalized_root / "gold.jsonl"
    for path in (artifacts_path, questions_path, gold_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if max_record_chars < 1:
        raise ValueError("max_record_chars must be positive")
    if output_root.exists() and not force:
        raise FileExistsError(output_root)

    questions = [sanitized_question(row) for row in iter_jsonl(questions_path)]
    question_ids = [row["question_id"] for row in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("normalized question ids are not unique")
    exact_questions = {row["question"].strip() for row in questions if row["question"].strip()}
    gold_rows = list(iter_jsonl(gold_path))
    if len(gold_rows) != len(questions):
        raise ValueError(f"gold/question count mismatch: {len(gold_rows)} != {len(questions)}")

    tmp_parent = output_root.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=tmp_parent))
    source_root = tmp_root / "source"
    source_root.mkdir(parents=True)
    output_counts: Counter[str] = Counter()
    artifact_index: list[dict[str, Any]] = []
    forbidden_hits: Counter[str] = Counter()
    leaked_questions: list[dict[str, str]] = []
    max_seen_chars = 0
    max_seen_artifact = ""
    normalized_artifact_count = 0
    excluded_oracle_artifacts: Counter[str] = Counter()
    seen_artifact_ids: set[str] = set()
    deduplicated_occurrences: Counter[str] = Counter()

    try:
        for line_number, artifact in enumerate(iter_jsonl(artifacts_path), start=1):
            normalized_artifact_count += 1
            artifact_type_value = str(artifact.get("artifact_type") or "")
            if artifact_type_value in ORACLE_ONLY_ARTIFACT_TYPES:
                excluded_oracle_artifacts[artifact_type_value] += 1
                continue
            if artifact_type_value not in RAG_SOURCE_ARTIFACT_TYPES:
                raise ValueError(
                    f"unsupported HERB RAG artifact type {artifact_type_value!r} at normalized line {line_number}"
                )
            keys, strings = walk_keys_and_strings(artifact)
            for key in sorted(keys.intersection(FORBIDDEN_SOURCE_KEYS)):
                forbidden_hits[key] += 1
            string_set = {value.strip() for value in strings if value.strip()}
            overlaps = exact_questions.intersection(string_set)
            for question in sorted(overlaps):
                leaked_questions.append(
                    {
                        "artifact_id": str(artifact.get("artifact_id") or ""),
                        "question": question,
                    }
                )
            artifact_id = str(artifact.get("artifact_id") or "").strip()
            if not artifact_id:
                raise ValueError(f"normalized artifact at line {line_number} has no artifact_id")
            if artifact_id in seen_artifact_ids:
                if artifact_type_value != "url":
                    raise ValueError(f"duplicate non-URL HERB artifact_id: {artifact_id}")
                deduplicated_occurrences[artifact_type_value] += 1
                continue
            seen_artifact_ids.add(artifact_id)
            body = str(artifact.get("raw_text") or "").strip()
            if not body:
                raise ValueError(f"artifact {artifact_id} has empty raw_text")
            if len(body) > max_record_chars:
                raise ValueError(
                    f"artifact {artifact_id} has {len(body)} chars, "
                    f"exceeding max_record_chars={max_record_chars}"
                )
            if len(body) > max_seen_chars:
                max_seen_chars = len(body)
                max_seen_artifact = artifact_id
            product = slug(artifact.get("product_id"), "global")
            artifact_type = slug(artifact.get("artifact_type"), "artifact")
            identity_material = f"{artifact_type}\x1f{artifact_id}"
            identity_hash = hashlib.sha256(identity_material.encode("utf-8")).hexdigest()[:16]
            artifact_slug = slug(artifact_id, "artifact")[:96]
            rel = Path(product) / artifact_type / f"{artifact_slug}--{identity_hash}.txt"
            target = source_root / rel
            if target.exists():
                raise ValueError(f"prepared source path collision for {artifact_id}: {rel}")
            target.parent.mkdir(parents=True, exist_ok=True)
            encoded_body = (body + "\n").encode("utf-8")
            target.write_bytes(encoded_body)
            key = rel.as_posix()
            output_counts[key] = 1
            artifact_index.append(
                {
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type_value,
                    "product_id": str(artifact.get("product_id") or ""),
                    "source_file": key,
                    "normalized_line_number": line_number,
                    "source_text_sha256": sha256_bytes(encoded_body),
                    "source_text_chars": len(body),
                }
            )

        if forbidden_hits:
            raise ValueError(f"forbidden benchmark keys found in source artifacts: {dict(forbidden_hits)}")
        if leaked_questions:
            raise ValueError(f"official question text found in source artifacts: {leaked_questions[:5]}")

        questions_out = tmp_root / "questions.jsonl"
        with questions_out.open("w", encoding="utf-8", newline="\n") as handle:
            for row in questions:
                handle.write(canonical_json(row) + "\n")
        gold_out = tmp_root / "gold.jsonl"
        shutil.copyfile(gold_path, gold_out)
        index_out = tmp_root / "artifact_index.jsonl"
        with index_out.open("w", encoding="utf-8", newline="\n") as handle:
            for row in artifact_index:
                handle.write(canonical_json(row) + "\n")

        validation = validate_source_folder(
            source_root,
            artifact_index=artifact_index,
            expected_questions=exact_questions,
            max_record_chars=max_record_chars,
        )
        manifest = {
            "format": FORMAT_VERSION,
            "normalized_root": str(normalized_root),
            "source_folder": "source",
            "questions_file": "questions.jsonl",
            "gold_file": "gold.jsonl",
            "artifact_index_file": "artifact_index.jsonl",
            "normalized_artifact_count": normalized_artifact_count,
            "artifact_count": len(artifact_index),
            "excluded_oracle_artifact_count": sum(excluded_oracle_artifacts.values()),
            "excluded_oracle_artifacts_by_type": dict(sorted(excluded_oracle_artifacts.items())),
            "deduplicated_occurrence_count": sum(deduplicated_occurrences.values()),
            "deduplicated_occurrences_by_type": dict(sorted(deduplicated_occurrences.items())),
            "rag_source_artifact_types": sorted(RAG_SOURCE_ARTIFACT_TYPES),
            "oracle_only_artifact_types": sorted(ORACLE_ONLY_ARTIFACT_TYPES),
            "question_count": len(questions),
            "gold_count": len(gold_rows),
            "source_representation": "one-artifact-per-plain-text-file",
            "source_text_field": "raw_text",
            "source_file_count": len(output_counts),
            "source_records_by_file": dict(sorted(output_counts.items())),
            "max_record_chars": max_seen_chars,
            "max_record_artifact_id": max_seen_artifact,
            "max_record_chars_contract": max_record_chars,
            "forbidden_source_keys": sorted(FORBIDDEN_SOURCE_KEYS),
            "forbidden_key_hits": {},
            "official_question_text_hits": 0,
            "input_hashes": {
                "artifacts.jsonl": sha256_file(artifacts_path),
                "questions.jsonl": sha256_file(questions_path),
                "gold.jsonl": sha256_file(gold_path),
            },
            "output_hashes": {
                "questions.jsonl": sha256_file(questions_out),
                "gold.jsonl": sha256_file(gold_out),
                "artifact_index.jsonl": sha256_file(index_out),
                "source_tree": source_tree_hash(source_root),
            },
            "validation": validation,
        }
        manifest_path = tmp_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_prepared_bundle(manifest_path)

        if output_root.exists():
            shutil.rmtree(output_root)
        tmp_root.replace(output_root)
        return json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise


def validate_source_folder(
    source_root: Path,
    *,
    artifact_index: list[dict[str, Any]],
    expected_questions: set[str],
    max_record_chars: int,
) -> dict[str, Any]:
    expected_by_path = {str(row["source_file"]): row for row in artifact_index}
    if len(expected_by_path) != len(artifact_index):
        raise ValueError("artifact index contains duplicate source paths")
    actual_paths = {
        path.relative_to(source_root).as_posix(): path
        for path in source_root.rglob("*")
        if path.is_file()
    }
    missing = sorted(set(expected_by_path) - set(actual_paths))
    extra = sorted(set(actual_paths) - set(expected_by_path))
    if missing or extra:
        raise ValueError(f"prepared source file set mismatch: missing={missing[:5]} extra={extra[:5]}")
    leaked_questions: list[dict[str, str]] = []
    max_seen = 0
    for rel, path in sorted(actual_paths.items()):
        if path.suffix.lower() != ".txt":
            raise ValueError(f"prepared HERB source is not plain text: {rel}")
        body = path.read_text(encoding="utf-8")
        stripped = body.strip()
        if not stripped:
            raise ValueError(f"prepared HERB source is empty: {rel}")
        max_seen = max(max_seen, len(stripped))
        if len(stripped) > max_record_chars:
            raise ValueError(f"{rel} exceeds source character contract")
        row = expected_by_path[rel]
        if sha256_file(path) != str(row["source_text_sha256"]):
            raise ValueError(f"prepared HERB source hash mismatch: {rel}")
        if stripped in expected_questions:
            leaked_questions.append({"path": rel, "question": stripped})
    if leaked_questions:
        raise ValueError(f"official question text found in source folder: {leaked_questions[:5]}")
    return {
        "source_file_count": len(actual_paths),
        "source_record_count": len(actual_paths),
        "max_record_chars": max_seen,
        "forbidden_key_hits": 0,
        "official_question_text_hits": 0,
        "json_parse_errors": 0,
        "plain_text_source_files": len(actual_paths),
    }


def validate_prepared_bundle(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported prepared HERB format: {manifest.get('format')!r}")
    root = manifest_path.parent
    source_root = root / str(manifest["source_folder"])
    questions_path = root / str(manifest["questions_file"])
    gold_path = root / str(manifest["gold_file"])
    index_path = root / str(manifest["artifact_index_file"])
    for path in (source_root, questions_path, gold_path, index_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256_file(questions_path) != manifest["output_hashes"]["questions.jsonl"]:
        raise ValueError("prepared questions hash mismatch")
    if sha256_file(gold_path) != manifest["output_hashes"]["gold.jsonl"]:
        raise ValueError("prepared gold hash mismatch")
    if sha256_file(index_path) != manifest["output_hashes"]["artifact_index.jsonl"]:
        raise ValueError("prepared artifact index hash mismatch")
    if source_tree_hash(source_root) != manifest["output_hashes"]["source_tree"]:
        raise ValueError("prepared source tree hash mismatch")
    questions = list(iter_jsonl(questions_path))
    if len(questions) != int(manifest["question_count"]):
        raise ValueError("prepared question count mismatch")
    if any(set(row) != {"question_id", "question"} for row in questions):
        raise ValueError("prepared questions contain fields other than question_id and question")
    gold = list(iter_jsonl(gold_path))
    if len(gold) != int(manifest["gold_count"]):
        raise ValueError("prepared gold count mismatch")
    index_rows = list(iter_jsonl(index_path))
    if len(index_rows) != int(manifest["artifact_count"]):
        raise ValueError("prepared artifact index count mismatch")
    if set(manifest.get("rag_source_artifact_types", [])) != RAG_SOURCE_ARTIFACT_TYPES:
        raise ValueError("prepared RAG artifact-type allowlist mismatch")
    if set(manifest.get("oracle_only_artifact_types", [])) != ORACLE_ONLY_ARTIFACT_TYPES:
        raise ValueError("prepared oracle-only artifact-type list mismatch")
    normalized_count = int(manifest.get("normalized_artifact_count", manifest["artifact_count"]))
    excluded_count = int(manifest.get("excluded_oracle_artifact_count", 0))
    deduplicated_count = int(manifest.get("deduplicated_occurrence_count", 0))
    if normalized_count != int(manifest["artifact_count"]) + excluded_count + deduplicated_count:
        raise ValueError("prepared normalized/included/excluded/deduplicated artifact accounting mismatch")
    if normalized_count == 39280:
        expected_type_counts = {
            "slack": 33632,
            "document": 400,
            "pull_request": 3562,
            "url": 575,
            "meeting_transcript": 321,
            "meeting_chat": 50,
            "customer": 120,
            "employee": 530,
        }
        actual_type_counts = Counter(str(row.get("artifact_type") or "") for row in index_rows)
        if dict(actual_type_counts) != expected_type_counts:
            raise ValueError(f"prepared HERB type counts do not match published pool: {dict(actual_type_counts)} != {expected_type_counts}")
        if int(manifest["artifact_count"]) != 39190:
            raise ValueError(f"prepared HERB artifact count must be 39190, got {manifest['artifact_count']}")
    if manifest.get("source_representation") != "one-artifact-per-plain-text-file":
        raise ValueError("prepared source representation mismatch")
    if manifest.get("source_text_field") != "raw_text":
        raise ValueError("prepared source text field mismatch")
    validation = validate_source_folder(
        source_root,
        artifact_index=index_rows,
        expected_questions={row["question"].strip() for row in questions if row["question"].strip()},
        max_record_chars=int(manifest["max_record_chars_contract"]),
    )
    return {
        "manifest": manifest,
        "root": root,
        "source_root": source_root,
        "questions_path": questions_path,
        "gold_path": gold_path,
        "validation": validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a leakage-free HERB raw-folder bundle for KMD.")
    parser.add_argument("--normalized-root", default=str(DEFAULT_NORMALIZED_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-record-chars", type=int, default=36000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if args.validate_only:
        result = validate_prepared_bundle(output_root / "manifest.json")
        print(json.dumps(result["manifest"], indent=2, sort_keys=True))
        return 0
    manifest = prepare_bundle(
        Path(args.normalized_root),
        output_root,
        force=args.force,
        max_record_chars=args.max_record_chars,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
