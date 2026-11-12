"""Resolve an explicitly requested catalog name before native document ranking."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from .publication_identifier import matching_publications, publication_identifier
from .rfc_section import split_rfc_citation


# An unquoted name inside a longer question needs an affirmative source
# reference. A comparison, exclusion, or incidental topic remains a vector query.
_CITATION = re.compile(
    r"^(?:according to (?:the )?|in (?:the )?|what does (?:the )?|how does (?:the )?|"
    r"which revision of (?:the )?|"
    r"laut (?:dem |der )?|was steht (?:im |in der )?|was sagt (?:der |die |das )?)",
    re.IGNORECASE,
)
_SOURCE_WORD = re.compile(
    r"\s+(?:guide|manual|skill|paper|documentation|readme|leitfaden|handbuch|anleitung|dokumentation)\b",
    re.IGNORECASE,
)
_COORDINATED = re.compile(r"^\s*(?:and|or|versus|vs\.?|und|oder)\b", re.IGNORECASE)
_QUOTES = {'"': '"', '`': '`', '“': '”', '„': '“', '«': '»'}
_SKILL_PATH = re.compile(r"skills/([a-z0-9-]+)/([a-z0-9][a-z0-9-]*)-[0-9a-f]{7,40}/SKILL\.md")
_TRANSLATE = str.maketrans({"’": "'", "‘": "'", "‐": "-", "‑": "-"})


def _name(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).translate(_TRANSLATE).casefold().split())


def _quoted(text: str):
    if not text or text[0] not in _QUOTES:
        return None
    end = text.find(_QUOTES[text[0]], 1)
    if end < 2:
        return None
    return text[1:end], text[end + 1:]


def _requested_names(query: str) -> dict[str, str]:
    """Map complete names to any remaining question about their contents."""
    names = {query: ""}
    if rfc := split_rfc_citation(query):
        names[rfc[0]] = rfc[1]
    quoted = _quoted(query)
    if quoted and not quoted[1].strip():
        names[quoted[0]] = quoted[1].lstrip(" ,:;—")
        if rfc := split_rfc_citation(quoted[0]):
            names[rfc[0]] = rfc[1]
    citation = _CITATION.match(query)
    if citation:
        rest = query[citation.end():]
        quoted = _quoted(rest)
        if quoted and not _COORDINATED.match(quoted[1].lstrip(" ,:;—")):
            names[quoted[0]] = quoted[1].lstrip(" ,:;—")
        elif not quoted:
            name, comma, subject = rest.partition(",")
            if comma and publication_identifier(name.strip()) and not _COORDINATED.match(subject):
                names[name.strip()] = subject.strip()
            for word in _SOURCE_WORD.finditer(rest):
                if not _COORDINATED.match(rest[word.end():]):
                    names[rest[:word.start()]] = rest[word.end():].lstrip(" ,:;—")
    return names


def resolve_source_name(query: str, manifest: Mapping[str, dict]) -> dict | None:
    """Resolve only a complete name. Ambiguity includes cataloged, unindexed editions."""
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        return None
    requested = _requested_names(query.strip())
    # File paths are case-sensitive. A catalog name is never interpreted as a filesystem path.
    paths = {value for value in requested if value in manifest}
    if len(paths) == 1:
        path = paths.pop()
        return _unique(path, "original_path", requested[path])
    if paths:
        return _ambiguous(paths)
    for name, subject in requested.items():
        identifier = publication_identifier(name)
        if identifier is None:
            continue
        publications = matching_publications(identifier, manifest)
        if len(publications) == 1:
            path, archived_identifier = next(iter(publications.items()))
            return {**_unique(path, identifier[0], subject), "matched_identifier": archived_identifier}
        if publications:
            return _ambiguous(publications)
        return {"status": "no_catalog_match", "kind": identifier[0], "matching_catalog_paths": 0}
    forms = {_name(value): subject for value, subject in requested.items() if value.strip()}
    matches = {}
    for path, record in manifest.items():
        title = record.get("title")
        if not isinstance(title, str) or not title:
            continue
        title = _name(title)
        if title in forms:
            matches[path] = ("catalog_title", forms[title])
            continue
        if record.get("source_type") != "agent_skill_instructions":
            continue
        archived = _SKILL_PATH.fullmatch(path)
        if not archived:
            continue
        publisher, slug = archived.groups()
        if title != f"{publisher} agent skill: {slug}":
            continue
        qualified = slug if slug.startswith(publisher + "-") else publisher + "-" + slug
        aliases = [qualified, qualified.replace("-", " ")]
        if "-" in slug:
            aliases.append(slug)
        for alias in aliases:
            if alias in forms:
                matches[path] = ("pinned_skill_name", forms[alias])
                break
    if not matches:
        return None
    if len(matches) > 1:
        return _ambiguous(matches)
    path, (kind, subject) = next(iter(matches.items()))
    return _unique(path, kind, subject)


def _ambiguous(paths):
    return {"status": "ambiguous", "matching_catalog_paths": len(paths),
            "candidate_paths": sorted(paths)[:8]}


def _unique(path, kind, subject):
    result = {"status": "unique", "matching_catalog_paths": 1, "original_path": path, "kind": kind}
    if subject:
        result["subject_query"] = subject
    return result
