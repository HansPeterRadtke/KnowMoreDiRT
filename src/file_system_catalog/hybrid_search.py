"""Bounded, read-only hybrid *candidate* retrieval for KMD's native FileDB.

The semantic FileDB remains authoritative. Callers supply independently verified,
source-linked extracted passages; they are indexed in an in-memory FTS5 table
for one bounded search. This does NOT ingest or modify originals, persist a
second database, generate DRS, or assert that a retrieved proposition is true.

The caller must independently verify the original revision, current derivative
revision and page/section mapping before using a candidate as citable evidence.
"""
from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from typing import Mapping, Sequence

_WORD = re.compile(r"(?u)\b[^\W_]+(?:_[^\W_]+)*\b")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# Generic connective words, not GUI- or library-specific subject rules.
_STOP = frozenset(
    "a an and are as at be by can did do does for from had has have how in is "
    "it its of on or our that the their then there these this those to was were "
    "what when where which who why will with would you your".split()
)


def lexical_token_spans(text: str):
    """Yield bounded whole terms alongside their exact original character offsets."""
    for match in _WORD.finditer(text):
        term = match.group().casefold()
        if 2 <= len(term) <= (128 if "_" in term else 48):
            yield term, match.start(), match.end()


def query_terms(query: str, *, max_terms: int = 20) -> tuple[str, ...]:
    """Return safe, distinct terms for an OR-based FTS query, not raw SQL."""
    if not 1 <= max_terms <= 100:
        raise ValueError("invalid term limit")
    if not isinstance(query, str) or len(query) > 1000:
        raise ValueError("query must be a string of at most 1000 characters")
    terms = []
    seen = set()
    for token, _, _ in lexical_token_spans(query):
        if token in _STOP or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return tuple(terms)


class BoundedPassageIndex:
    """Ephemeral FTS5 over caller-provided *verified* derived source passages.

    The index lives in RAM, not in another persistent SQLite database.
    The original filesystem_entry_id is retained for provenance joins.
    """

    def __init__(self, *, max_passages: int = 6000, max_characters: int = 20_000_000):
        if not 1 <= max_passages <= 100_000 or not 1 <= max_characters <= 200_000_000:
            raise ValueError("unbounded passage index")
        self.max_passages = max_passages
        self.max_characters = max_characters
        self.character_count = 0
        self._metadata: dict[int, dict] = {}
        self._closed = False
        self._db = sqlite3.connect(":memory:")
        try:
            self._db.execute(
                "CREATE VIRTUAL TABLE passage_fts USING fts5("
                "title, source_path, section, passage, "
                "tokenize='unicode61 remove_diacritics 2 tokenchars _')"
            )
        except sqlite3.OperationalError as exc:
            self._db.close()
            raise RuntimeError("SQLite FTS5 is required for ephemeral lexical retrieval") from exc

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        if not self._closed:
            self._db.close()
            self._closed = True

    @property
    def passage_count(self) -> int:
        return len(self._metadata)

    def add_passage(
        self,
        *,
        filesystem_entry_id: int,
        original_path: str,
        original_title: str,
        source_section: str,
        passage: str,
        original_sha256: str,
        derived_sha256: str,
    ) -> None:
        if self._closed:
            raise RuntimeError("passage index is closed")
        parts = original_path.split("/") if isinstance(original_path, str) else ()
        if (
            type(filesystem_entry_id) is not int
            or filesystem_entry_id <= 0
            or not isinstance(original_path, str)
            or not original_path
            or original_path.startswith("/")
            or "\\" in original_path
            or "\x00" in original_path
            or "" in parts
            or "." in parts
            or ".." in parts
            or len(original_path) > 1500
        ):
            raise ValueError("unsafe original FileDB identity or path")
        if (
            not isinstance(original_title, str)
            or len(original_title) > 1000
            or not isinstance(source_section, str)
            or len(source_section) > 300
            or not isinstance(passage, str)
            or not passage.strip()
            or len(passage) > 65_536
        ):
            raise ValueError("invalid or excessive source passage")
        if (
            not isinstance(original_sha256, str)
            or not _SHA256.fullmatch(original_sha256)
            or not isinstance(derived_sha256, str)
            or not _SHA256.fullmatch(derived_sha256)
        ):
            raise ValueError("missing verified original/derived revision identifiers")
        if (
            len(self._metadata) >= self.max_passages
            or self.character_count + len(passage) > self.max_characters
        ):
            raise ValueError("ephemeral index capacity exceeded")
        cursor = self._db.execute(
            "INSERT INTO passage_fts(title,source_path,section,passage) VALUES(?,?,?,?)",
            (original_title, original_path, source_section, passage),
        )
        self.character_count += len(passage)
        self._metadata[cursor.lastrowid] = {
            "filesystem_entry_id": filesystem_entry_id,
            "relative_path_display": original_path,
            "original_path": original_path,
            "original_title": original_title,
            "source_section": source_section,
            "passage": passage,
            "passage_sha256": hashlib.sha256(passage.encode("utf-8")).hexdigest(),
            "original_sha256": original_sha256,
            "derived_sha256": derived_sha256,
            "evidence_status": "candidate_requires_source_revision_check",
        }

    def search(self, query: str, *, limit: int = 300, require_identifier_in_passage: bool = False) -> list[dict]:
        """Page-level lexical candidates; lower SQLite BM25 is better.

        Uses parameterized FTS5 MATCH. Require at least two meaningful terms
        for multi-term questions, so one incidental word cannot dominate.
        """
        if self._closed:
            raise RuntimeError("passage index is closed")
        if not 1 <= limit <= 2000:
            raise ValueError("unbounded search results")
        terms = query_terms(query)
        if not terms:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        identifiers = [t for t in terms if "_" in t]
        if require_identifier_in_passage and identifiers:
            identifier_match = " OR ".join('"' + t + '"' for t in identifiers)
            match = f"({match}) AND passage : ({identifier_match})"
        rows = self._db.execute(
            "SELECT rowid,bm25(passage_fts,6.0,0.15,1.2,1.0) "
            "FROM passage_fts WHERE passage_fts MATCH ? "
            "ORDER BY 2 ASC, rowid ASC LIMIT ?",
            (match, limit),
        ).fetchall()
        out = []
        for rowid, bm25 in rows:
            item = self._metadata[rowid]
            searchable = " ".join((item["original_title"], item["source_section"], item["passage"])).casefold()
            found = sum(
                re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", searchable) is not None
                for term in terms
            )
            if found < min(2, len(terms)):
                continue
            if len(terms) >= 5 and found / len(terms) < 0.25:
                continue
            out.append({
                **item,
                "lexical_bm25": float(bm25),
                "matched_query_terms": found,
                "total_query_terms": len(terms),
                "lexical_passage_rank": len(out) + 1,
            })
        return out


def fuse_search_candidates(
    semantic_candidates: Sequence[Mapping],
    lexical_candidates: Sequence[Mapping],
    *,
    top: int = 10,
    reciprocal_rank_constant: int = 60,
    lexical_weight: float = 2.0,
    semantic_weight: float = 1.0,
    max_passages_per_source: int = 3,
) -> list[dict]:
    """Fuse independent evidence candidate lists, with no truth/confidence claim.

    The two rank lists are already ordered best first. Their raw cosine and
    BM25 values have incompatible scales: use reciprocal ranks instead.
    Deduplicate *documents*, while preserving multiple lexical page candidates.
    """
    if not 1 <= top <= 100 or not 1 <= max_passages_per_source <= 10:
        raise ValueError("invalid result or passage limit")
    if not 1 <= reciprocal_rank_constant <= 1000:
        raise ValueError("invalid reciprocal rank constant")
    if (
        not math.isfinite(lexical_weight)
        or not math.isfinite(semantic_weight)
        or lexical_weight < 0
        or semantic_weight < 0
        or lexical_weight + semantic_weight <= 0
    ):
        raise ValueError("invalid rank weights")
    documents: dict[str, dict] = {}

    def get_document(path):
        return documents.setdefault(path, {
            "original_path": path,
            "semantic_rank": None,
            "lexical_rank": None,
            "semantic_similarity": None,
            "lexical_passages": [],
            "semantic_hits": [],
            "hybrid_score": 0.0,
            "evidence_status": "candidates_only_requires_source_and_context_validation",
        })

    seen_semantic = set()
    for rank, candidate in enumerate(semantic_candidates, 1):
        path = candidate.get("relative_path_display")
        if not isinstance(path, str) or not path:
            continue
        score = candidate.get("score")
        if score is not None and (
            not isinstance(score, (int, float)) or not math.isfinite(score)
        ):
            continue
        row = get_document(path)
        if path not in seen_semantic:
            row["semantic_rank"] = rank
            row["semantic_similarity"] = score
            row["hybrid_score"] += semantic_weight / (reciprocal_rank_constant + rank)
            seen_semantic.add(path)
        if len(row["semantic_hits"]) < max_passages_per_source:
            row["semantic_hits"].append(dict(candidate))

    seen_sections = set()
    for rank, candidate in enumerate(lexical_candidates, 1):
        path = candidate.get("original_path")
        section = candidate.get("source_section")
        if not isinstance(path, str) or not path or not isinstance(section, str):
            continue
        row = get_document(path)
        if row["lexical_rank"] is None:
            row["lexical_rank"] = rank
            row["hybrid_score"] += lexical_weight / (reciprocal_rank_constant + rank)
        unique_key = (path, section, candidate.get("passage_sha256"))
        if unique_key not in seen_sections and len(row["lexical_passages"]) < max_passages_per_source:
            row["lexical_passages"].append(dict(candidate))
            seen_sections.add(unique_key)
    return sorted(
        documents.values(),
        key=lambda row: (-row["hybrid_score"], min(
            x for x in (row["lexical_rank"], row["semantic_rank"]) if x is not None
        ), row["original_path"]),
    )[:top]


def search_hybrid_candidates(
    semantic_candidates: Sequence[Mapping],
    passages: BoundedPassageIndex,
    query: str,
    *,
    top: int = 10,
    lexical_limit: int = 300,
) -> list[dict]:
    """Generic, bounded fusion of KMD FileDB cosine results and lexical pages."""
    return fuse_search_candidates(
        semantic_candidates, passages.search(query, limit=lexical_limit), top=top
    )
