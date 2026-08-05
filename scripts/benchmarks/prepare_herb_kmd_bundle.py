#!/usr/bin/env python3
"""Create a leakage-free, record-safe HERB bundle for KnowMoreDiRT.

The HERB repository's canonical normalized artifacts are already separated from
questions and gold labels. This transform groups those artifacts into JSONL
source files while preserving one complete artifact per line. Questions and
scoring gold remain sibling files outside the source folder.
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

FORMAT_VERSION = "herb-kmd-prepared-v1"
DEFAULT_NORMALIZED_ROOT = Path("/data/var/herb_benchmark/normalized/herb_normalized")
DEFAULT_OUTPUT_ROOT = Path("/data/var/herb_benchmark/prepared/kmd_raw_v1")
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
    handles: dict[Path, Any] = {}
    output_counts: Counter[str] = Counter()
    output_records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    artifact_index: list[dict[str, Any]] = []
    forbidden_hits: Counter[str] = Counter()
    leaked_questions: list[dict[str, str]] = []
    max_seen_chars = 0
    max_seen_artifact = ""

    try:
        for line_number, artifact in enumerate(iter_jsonl(artifacts_path), start=1):
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
            encoded = canonical_json(artifact)
            if len(encoded) > max_record_chars:
                raise ValueError(
                    f"artifact {artifact.get('artifact_id')} has {len(encoded)} chars, "
                    f"exceeding max_record_chars={max_record_chars}"
                )
            if len(encoded) > max_seen_chars:
                max_seen_chars = len(encoded)
                max_seen_artifact = str(artifact.get("artifact_id") or "")
            product = slug(artifact.get("product_id"), "global")
            artifact_type = slug(artifact.get("artifact_type"), "artifact")
            rel = Path(product) / f"{artifact_type}.jsonl"
            target = source_root / rel
            handle = handles.get(target)
            if handle is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                handle = target.open("w", encoding="utf-8", newline="\n")
                handles[target] = handle
            handle.write(encoded + "\n")
            key = rel.as_posix()
            output_counts[key] += 1
            artifact_index.append(
                {
                    "artifact_id": str(artifact.get("artifact_id") or ""),
                    "artifact_type": str(artifact.get("artifact_type") or ""),
                    "product_id": str(artifact.get("product_id") or ""),
                    "source_file": key,
                    "source_record_index": output_counts[key] - 1,
                    "normalized_line_number": line_number,
                    "record_sha256": sha256_bytes(encoded.encode("utf-8")),
                    "record_chars": len(encoded),
                }
            )
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()

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
            expected_records=len(artifact_index),
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
            "artifact_count": len(artifact_index),
            "question_count": len(questions),
            "gold_count": len(gold_rows),
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
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise


def validate_source_folder(
    source_root: Path,
    *,
    expected_records: int,
    expected_questions: set[str],
    max_record_chars: int,
) -> dict[str, Any]:
    record_count = 0
    file_count = 0
    forbidden_hits: Counter[str] = Counter()
    leaked_questions: list[dict[str, str]] = []
    max_seen = 0
    for path in sorted(source_root.rglob("*.jsonl")):
        file_count += 1
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record_count += 1
                encoded = line.rstrip("\r\n")
                max_seen = max(max_seen, len(encoded))
                if len(encoded) > max_record_chars:
                    raise ValueError(f"{path}:{line_number} exceeds record character contract")
                value = json.loads(encoded)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                keys, strings = walk_keys_and_strings(value)
                for key in keys.intersection(FORBIDDEN_SOURCE_KEYS):
                    forbidden_hits[key] += 1
                overlaps = expected_questions.intersection({item.strip() for item in strings if item.strip()})
                for question in overlaps:
                    leaked_questions.append({"path": str(path), "question": question})
    if record_count != expected_records:
        raise ValueError(f"source record count mismatch: {record_count} != {expected_records}")
    if forbidden_hits:
        raise ValueError(f"forbidden source keys found: {dict(forbidden_hits)}")
    if leaked_questions:
        raise ValueError(f"official question text found in source folder: {leaked_questions[:5]}")
    return {
        "source_file_count": file_count,
        "source_record_count": record_count,
        "max_record_chars": max_seen,
        "forbidden_key_hits": 0,
        "official_question_text_hits": 0,
        "json_parse_errors": 0,
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
    validation = validate_source_folder(
        source_root,
        expected_records=int(manifest["artifact_count"]),
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
