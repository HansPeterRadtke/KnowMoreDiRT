"""Match explicit publication identifiers using the archived acquisition metadata."""
from __future__ import annotations

import re
from collections.abc import Mapping

from .rfc_section import matching_rfc, rfc_number

_FLAGS = re.IGNORECASE | re.ASCII
_DOI = re.compile(r"(?:doi:[ \t]*|https://doi\.org/)?(10\.[0-9]{4,9}/[-._;()/:a-z0-9]+)", _FLAGS)
_MODERN_ARXIV = r"([0-9]{2}(?:0[1-9]|1[0-2])\.[0-9]{4,5}(?:v[1-9][0-9]*)?)"
_ARXIV_ID = re.compile(_MODERN_ARXIV, _FLAGS)
_ARXIV_REFERENCE = re.compile(r"(?:arxiv:[ \t]*|https://arxiv\.org/abs/)" + _MODERN_ARXIV, _FLAGS)
_ARXIV_PDF = re.compile(r"https://arxiv\.org/pdf/" + _MODERN_ARXIV + r"(?:\.pdf)?", _FLAGS)


def publication_identifier(text: str) -> tuple[str, str] | None:
    number = rfc_number(text)
    if number:
        return "catalog_rfc", number
    match = _DOI.fullmatch(text)
    if match:
        return "catalog_doi", match[1].lower()
    match = _ARXIV_REFERENCE.fullmatch(text) or _ARXIV_PDF.fullmatch(text)
    if match:
        return "catalog_arxiv", match[1].lower()
    return None


def _matching_doi(record: dict, requested: str) -> str | None:
    value = record.get("doi")
    if not isinstance(value, str):
        return None
    declared = _DOI.fullmatch(value)
    if not declared or declared[1].lower() != requested:
        return None
    for field in ("source_url", "url"):
        url = record.get(field)
        if isinstance(url, str) and url.lower().startswith("https://doi.org/"):
            linked = _DOI.fullmatch(url)
            if not linked or linked[1].lower() != requested:
                return None
    return requested


def _matching_arxiv(record: dict, requested: str) -> str | None:
    # A cited abstract page may refer to a different document. Take the file
    # identity and any requested version from the archive download URL.
    url = record.get("url")
    downloaded = _ARXIV_PDF.fullmatch(url) if isinstance(url, str) else None
    if not downloaded:
        return None
    archive_id = downloaded[1].lower()
    base = archive_id.partition("v")[0]
    if requested not in (base, archive_id):
        return None
    claims = {archive_id}
    declared = record.get("arxiv_id")
    if declared is not None:
        if not isinstance(declared, str) or not _ARXIV_ID.fullmatch(declared):
            return None
        claims.add(declared.lower())
    source = record.get("source_url")
    if isinstance(source, str) and source.lower().startswith("https://arxiv.org/"):
        linked = _ARXIV_REFERENCE.fullmatch(source) or _ARXIV_PDF.fullmatch(source)
        if not linked:
            return None
        claims.add(linked[1].lower())
    if any(claim.partition("v")[0] != base for claim in claims) or len(claims - {base}) > 1:
        return None
    return "arXiv:" + archive_id


def matching_publications(identifier: tuple[str, str], manifest: Mapping[str, dict]) -> dict[str, str]:
    kind, value = identifier
    media, match = {
        "catalog_doi": ("pdf", _matching_doi),
        "catalog_arxiv": ("pdf", _matching_arxiv),
        "catalog_rfc": ("text", matching_rfc),
    }[kind]
    publications = {}
    for path, record in manifest.items():
        if record.get("kind") == media:
            archived_identifier = match(record, value)
            if archived_identifier is not None:
                publications[path] = archived_identifier
    return publications


def publication_receipt_matches(record: dict, receipt: dict) -> bool:
    """Tie the catalog identifier to the URL recorded for the completed download."""
    url = record.get("url")
    return isinstance(url, str) and bool(url) and url == receipt.get("url")
