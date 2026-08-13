#!/usr/bin/env python3
"""Consolidate trusted legacy KMD model-call caches into one canonical root.

Only known model-cache namespaces are imported. Paths marked quarantine,
contaminated, or failed are excluded. Same-key/same-content entries deduplicate;
same-key/different-content entries are removed from the active namespace and
preserved under _conflicts so KMD recomputes that call rather than reusing an
ambiguous historical result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

CANONICAL_ROOT = Path("/data/var/knowmoredirt/model_cache")
SEARCH_ROOTS = (Path("/data/var/knowmoredirt"), Path("/data/var/herb_benchmark"))
EXCLUDED_MARKERS = ("quarantine", "contaminated", "failed")
NAMESPACE_ALIASES = {
    "frame": "frame", "frame_cache": "frame",
    "chunk_frame": "chunk_frame", "chunk_frame_cache": "chunk_frame",
    "chunk_drs": "chunk_drs", "chunk_drs_cache": "chunk_drs",
    "query_plan": "query_plan", "query_plan_cache": "query_plan",
    "query_drs": "query_drs", "query_drs_cache": "query_drs",
    "query_evidence_repair": "query_evidence_repair", "query_evidence_repair_cache": "query_evidence_repair",
    "query_evidence": "query_evidence", "query_evidence_cache": "query_evidence",
    "evidence_answer": "evidence_answer", "evidence_answer_cache": "evidence_answer",
    "verifier": "verifier", "verifier_cache": "verifier",
    "answer_canonicalization": "answer_canonicalization", "answer_canonicalization_cache": "answer_canonicalization",
    "identity": "identity", "identity_cache": "identity",
    "source_resolution": "source_resolution", "source_resolution_cache": "source_resolution",
    "document_context": "document_context", "document_context_cache": "document_context",
    "evaluation_judge": "evaluation_judge", "evaluation_judge_cache": "evaluation_judge",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded(path: Path) -> bool:
    text = str(path).lower()
    return any(marker in text for marker in EXCLUDED_MARKERS)


def namespace_dirs(search_roots: Iterable[Path], canonical_root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    canonical_resolved = canonical_root.resolve()
    for root in search_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, _filenames in os.walk(root, followlinks=False):
            current = Path(dirpath)
            try:
                if current.resolve() == canonical_resolved or canonical_resolved in current.resolve().parents:
                    dirnames[:] = []
                    continue
            except OSError:
                pass
            if excluded(current):
                dirnames[:] = []
                continue
            namespace = NAMESPACE_ALIASES.get(current.name)
            if namespace:
                found.append((namespace, current))
                dirnames[:] = []
    return found


def copy_into_cache(source: Path, target: Path, *, owner_uid: int, owner_gid: int) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    os.chmod(target, 0o644)
    if os.geteuid() == 0:
        os.chown(target, owner_uid, owner_gid)
    return "copy"


def preserve_conflict(root: Path, namespace: str, filename: str, source: Path, digest: str, *, owner_uid: int, owner_gid: int) -> None:
    conflict_dir = root / "_conflicts" / namespace / filename
    conflict_dir.mkdir(parents=True, exist_ok=True)
    target = conflict_dir / f"{digest}.json"
    if not target.exists():
        copy_into_cache(source, target, owner_uid=owner_uid, owner_gid=owner_gid)


def consolidate(canonical_root: Path, search_roots: Iterable[Path]) -> dict[str, object]:
    canonical_root.mkdir(parents=True, exist_ok=True)
    root_stat = canonical_root.stat()
    owner_uid, owner_gid = root_stat.st_uid, root_stat.st_gid
    stats = {"imported": 0, "deduplicated": 0, "conflicted": 0, "ignored_non_json": 0, "sources": 0}
    conflicted_keys: set[tuple[str, str]] = set()
    source_dirs = namespace_dirs(search_roots, canonical_root)
    stats["sources"] = len(source_dirs)
    for namespace, source_dir in source_dirs:
        target_dir = canonical_root / namespace
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(source_dir.iterdir()):
            if not source.is_file() or source.suffix.lower() != ".json":
                if source.is_file():
                    stats["ignored_non_json"] += 1
                continue
            key = (namespace, source.name)
            source_hash = sha256(source)
            target = target_dir / source.name
            if key in conflicted_keys:
                preserve_conflict(canonical_root, namespace, source.name, source, source_hash, owner_uid=owner_uid, owner_gid=owner_gid)
                continue
            if not target.exists():
                copy_into_cache(source, target, owner_uid=owner_uid, owner_gid=owner_gid)
                stats["imported"] += 1
                continue
            target_hash = sha256(target)
            if target_hash == source_hash:
                stats["deduplicated"] += 1
                continue
            preserve_conflict(canonical_root, namespace, source.name, target, target_hash, owner_uid=owner_uid, owner_gid=owner_gid)
            preserve_conflict(canonical_root, namespace, source.name, source, source_hash, owner_uid=owner_uid, owner_gid=owner_gid)
            target.unlink()
            conflicted_keys.add(key)
            stats["conflicted"] += 1
    manifest = {
        "canonical_root": str(canonical_root),
        "search_roots": [str(path) for path in search_roots],
        "excluded_markers": list(EXCLUDED_MARKERS),
        "stats": stats,
        "conflicted_active_keys_removed": len(conflicted_keys),
    }
    manifest_path = canonical_root / "migration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    normalize_canonical_permissions(canonical_root, owner_uid=owner_uid, owner_gid=owner_gid)
    return manifest



def normalize_canonical_permissions(root: Path, *, owner_uid: int, owner_gid: int) -> None:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        os.chmod(directory, 0o755)
        if os.geteuid() == 0:
            os.chown(directory, owner_uid, owner_gid)
        for filename in filenames:
            path = directory / filename
            if path.is_symlink():
                continue
            os.chmod(path, 0o644)
            if os.geteuid() == 0:
                os.chown(path, owner_uid, owner_gid)


def prune_legacy_namespace_dirs(search_roots: Iterable[Path], canonical_root: Path) -> dict[str, int]:
    removed_dirs = 0
    removed_files = 0
    for _namespace, source_dir in namespace_dirs(search_roots, canonical_root):
        if source_dir.is_symlink() or not source_dir.exists():
            continue
        removed_files += sum(1 for path in source_dir.iterdir() if path.is_file())
        shutil.rmtree(source_dir)
        removed_dirs += 1
    return {"removed_dirs": removed_dirs, "removed_files": removed_files}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--search-root", action="append", type=Path)
    parser.add_argument("--prune-sources", action="store_true", help="Remove trusted legacy namespace directories after successful consolidation.")
    args = parser.parse_args()
    roots = tuple(args.search_root) if args.search_root else SEARCH_ROOTS
    result = consolidate(args.canonical_root, roots)
    if args.prune_sources:
        result["prune"] = prune_legacy_namespace_dirs(roots, args.canonical_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
