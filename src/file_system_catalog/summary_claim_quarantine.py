"""Read-only, source-revision-bound exclusions of reviewed false summary claims.

A generated summary is a retrieval aid, not publisher evidence. Once an exact
representation has been independently contradicted by the verified original,
its vector and text must not be offered again as ordinary ranked evidence.
This never edits native FileDB rows, original sources, workers or embeddings.
Only private, original- and summary-SHA-pinned attestations are admitted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat

SCHEMA = "kmd.reviewed_generated_summary_exclusions.v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-z0-9-]{8,80}\Z")
_MAX_REGISTRY_BYTES = 256 * 1024
_MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
_MAX_ENTRIES = 200


def _read_regular(path: Path, maximum: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("reviewed summary attestation is missing or a symlink")
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= maximum:
        raise ValueError("reviewed summary attestation has invalid size or type")
    raw = path.read_bytes()
    if len(raw) != metadata.st_size:
        raise ValueError("reviewed summary attestation changed while reading")
    return raw


def load_reviewed_summary_exclusions(registry: Path) -> frozenset[tuple[str, str, str, str, str]]:
    """Admit only reviewed incorrect *representation* identities, never whole works.

    Registry and evidence are private operator data. Missing registry means
    no reviewed exclusions have been installed; malformed existing data
    fails closed instead of silently resurfacing known bad claims.
    """
    registry = Path(registry)
    if not registry.exists() and not registry.is_symlink():
        return frozenset()
    try:
        doc = json.loads(_read_regular(registry, _MAX_REGISTRY_BYTES))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid reviewed summary exclusion registry") from exc
    if (not isinstance(doc, dict) or doc.get("schema") != SCHEMA or
            not isinstance(doc.get("entries"), list) or
            len(doc["entries"]) > _MAX_ENTRIES):
        raise ValueError("invalid reviewed summary exclusion registry schema")
    found: set[tuple[str, str, str, str, str]] = set()
    base = (registry.parent / "search-audit").resolve(strict=True)
    for row in doc["entries"]:
        if not isinstance(row, dict):
            raise ValueError("reviewed summary exclusion row is invalid")
        path = row.get("original_path")
        orig, derived, text = (row.get(k) for k in (
            "original_sha256", "derived_sha256", "summary_sha256"))
        rep = row.get("native_representation_id")
        node = row.get("summary_node_id")
        proof_path = row.get("original_review_path")
        proof_digest = row.get("original_review_sha256")
        if (not isinstance(path, str) or not 1 <= len(path) <= 1500 or
                path.startswith("/") or "\\" in path or
                any(part in ("", ".", "..") for part in path.split("/")) or
                not all(isinstance(s, str) and _HEX.fullmatch(s) for s in (
                    orig, derived, text, proof_digest)) or
                not isinstance(rep, str) or not _ID.fullmatch(rep) or
                not isinstance(node, str) or not node.startswith("summary:") or
                not isinstance(proof_path, str) or not proof_path.startswith("/")):
            raise ValueError("reviewed summary exclusion identity is invalid")
        proof_file = Path(proof_path)
        if proof_file.is_symlink() or not proof_file.resolve().is_relative_to(base):
            raise ValueError("review evidence must stay within protected source audits")
        raw = _read_regular(proof_file, _MAX_EVIDENCE_BYTES)
        if hashlib.sha256(raw).hexdigest() != proof_digest:
            raise ValueError("reviewed source evidence has changed")
        try:
            evidence = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("reviewed source evidence is not valid JSON") from exc
        if (not isinstance(evidence, dict) or
                evidence.get("publisher_original_sha256") != orig or
                evidence.get("derived_sha256") != derived or
                evidence.get("incorrect_existing_native_summary_sha256") != text or
                evidence.get("incorrect_existing_native_summary_node_id") != node or
                not isinstance(evidence.get("source_specific_corrected_statement"), str) or
                not evidence["source_specific_corrected_statement"].strip() or
                len(evidence["source_specific_corrected_statement"]) > 5000 or
                type(evidence.get("original_physical_pdf_page")) is not int or
                not 1 <= evidence["original_physical_pdf_page"] <= 100000 or
                evidence.get("original_physical_file_bytes_rehashed") is not True or
                not isinstance(evidence.get("source_row_receipts"), list) or
                not 1 <= len(evidence["source_row_receipts"]) <= 200):
            raise ValueError("review proof does not establish the exact excluded claim")
        for attested in evidence["source_row_receipts"]:
            if (not isinstance(attested, dict) or
                    not isinstance(attested.get("exact_source_text"), str) or
                    hashlib.sha256(attested["exact_source_text"].encode()).hexdigest()
                    != attested.get("line_sha256")):
                raise ValueError("review proof contains an invalid source line")
        key = (path, orig, derived, rep, text)
        if key in found:
            raise ValueError("duplicate reviewed summary exclusion identity")
        found.add(key)
    return frozenset(found)


def exclude_reviewed_summary_nodes(
    candidates: list[dict], exclusions: frozenset[tuple[str, str, str, str, str]],
) -> tuple[list[dict], int]:
    """Filter exact reviewed bad representations; never filter entire originals."""
    retained = []
    excluded = 0
    for row in candidates:
        key = (
            row["original_path"], row["original_sha256"], row["derived_sha256"],
            row["native_representation_id"], row["summary_sha256"],
        )
        if key in exclusions:
            excluded += 1
        else:
            retained.append(row)
    return retained, excluded
