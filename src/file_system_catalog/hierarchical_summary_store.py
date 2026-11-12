"""Append source-linked hierarchical summary vectors to the ORIGINAL KMD FileDB.

Opt-in only. Uses the existing native content_chunks/content_representations
tables, not a new database or alternate source of truth. The caller must supply
the exact original completed-source revision and the current derived-text file
already independently verified against the source-map and completion ledger.

This is a building block; the existing Infra result resolver is not yet aware
of the summary collection, so do not enable on the production FileDB before a
verified query-resolution integration is deployed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

from .content_pipeline import (
    stable_file_id, stable_chunk_id, stable_representation_id, count_words,
    flatten_representations, normalize_analysis,
)
from .content_schema import (
    CHUNK_COLUMN_NAMES, REPRESENTATION_COLUMN_NAMES,
    CHUNK_TABLE_NAME, REPRESENTATION_TABLE_NAME,
)
from .publisher_ccs import CCS_EXTRACTOR, first_page_ccs
from .hierarchical_summary import EmbeddingContextExceeded, SummaryTree, _VERSION


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_complete_summary_tree(tree: SummaryTree) -> None:
    """Reject a missing level, leaf, parent link or original source interval.

    Every level, including all recursively produced intermediary summaries,
    must cover the complete source exactly once. This is mandatory before any
    vector generation or native KMD database write.
    """
    if not tree.nodes or not tree.root_id or tree.original_characters < 1:
        raise ValueError("incomplete hierarchical summary tree")
    nodes = {node.node_id: node for node in tree.nodes}
    if len(nodes) != len(tree.nodes) or tree.root_id not in nodes:
        raise ValueError("duplicate or missing summary node identity")
    max_level = max(node.level for node in tree.nodes)
    if max_level < 0 or max_level > 50:
        raise ValueError("invalid summary depth")
    if sorted({node.level for node in tree.nodes}) != list(range(max_level + 1)):
        raise ValueError("missing intermediate summary level")
    groups = {
        level: sorted(
            (n for n in tree.nodes if n.level == level),
            key=lambda n: (n.start_char, n.end_char, n.node_id),
        )
        for level in range(max_level + 1)
    }
    for level, current in groups.items():
        if current[0].start_char != 0 or current[-1].end_char != tree.original_characters:
            raise ValueError(f"summary level {level} fails full source coverage")
        if any(a.end_char != b.start_char for a, b in zip(current, current[1:])):
            raise ValueError(f"summary level {level} contains a gap or overlap")
        if any(n.start_char >= n.end_char or n.source_sha256 != tree.source_sha256 for n in current):
            raise ValueError(f"summary level {level} has invalid source spans or revision")
    if len(groups[max_level]) != 1 or groups[max_level][0].node_id != tree.root_id:
        raise ValueError("hierarchy must have exactly one full-document root")
    if tree.leaf_sections < 1:
        raise ValueError("missing original source section accounting")
    if any(n.child_ids for n in groups[0]):
        raise ValueError("source-level summary should not reference generated children")
    for level in range(1, max_level + 1):
        parents = groups[level]
        children = groups[level - 1]
        seen = []
        for parent in parents:
            if not parent.child_ids:
                raise ValueError("parent summary has no child summaries")
            try:
                linked = [nodes[k] for k in parent.child_ids]
            except KeyError as error:
                raise ValueError("parent references a missing summary") from error
            if any(child.level != level - 1 for child in linked):
                raise ValueError("parent references wrong summary hierarchy level")
            if (
                linked[0].start_char != parent.start_char
                or linked[-1].end_char != parent.end_char
                or any(a.end_char != b.start_char for a, b in zip(linked, linked[1:]))
            ):
                raise ValueError("parent summary does not cover exactly its child spans")
            seen.extend(child.node_id for child in linked)
        if seen != [child.node_id for child in children]:
            raise ValueError("parent summaries omit, duplicate or reorder child nodes")


_IDENTIFIER = re.compile(r"(?<!\w)[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+(?!\w)")
_SOURCE_UNIT_LINE = re.compile(r"^\[\[SOURCE UNIT: [^\]\n]{1,400}\]\][ \t]*\r?$", re.MULTILINE)
_MAX_NODE_FACETS = 256


def _source_identifier_members(item, *, text, start, end, capacity, embedding_client, limit):
    """Resolve a whole overlength identifier enumeration or reject it before embedding."""
    raw = item["text"]
    terms = [term.strip() for term in raw.split(",", capacity)]
    if (item["kind"] not in ("keyword", "keyphrase") or not 2 <= len(terms) <= capacity
            or len(set(terms)) != len(terms)
            or any(not _IDENTIFIER.fullmatch(term) for term in terms)):
        raise EmbeddingContextExceeded("native facet exceeds embedding context and is not a bounded identifier list")
    if any(embedding_client.token_count(term) > limit for term in terms):
        raise EmbeddingContextExceeded("individual source identifier exceeds embedding context")

    # Coordinates are in the entire verified derivative. Skip its administrative
    # header and unit labels, even if a node starts in the middle of either.
    first_unit = _SOURCE_UNIT_LINE.search(text)
    if first_unit is None:
        raise EmbeddingContextExceeded("no marked publisher body for oversized identifier list")
    missing = set(terms)
    locations = {}
    # One character of lookahead prevents treating the end of a node as the
    # end of an identifier when the next source character continues the word.
    for match in _IDENTIFIER.finditer(text, max(start, first_unit.end()), end + 1):
        term = match.group()
        if term not in missing or match.end() > end:
            continue
        line_start = text.rfind("\n", 0, match.start()) + 1
        if text.startswith("[[SOURCE UNIT: ", line_start):
            continue
        locations[term] = match.span()
        missing.remove(term)
        if not missing:
            break
    if missing:
        raise EmbeddingContextExceeded("oversized identifier list contains a term absent from the publisher span")

    list_digest = _sha(raw)
    return [
        {**item, "kind": "keyword", "text": term, "identifier_list_member": {
            "original_kind": item["kind"], "original_global_rank": item["global_rank"],
            "list_sha256": list_digest, "index": index, "count": len(terms),
            "start_char": locations[term][0], "end_char": locations[term][1],
        }}
        for index, term in enumerate(terms, 1)
    ]


def _fit_native_facets(items, *, text, start, end, embedding_client, limit):
    if len(items) > _MAX_NODE_FACETS:
        raise ValueError("unbounded native facet representation count")
    fitted = []
    global_shift = 0
    facet_shift = 0
    previous_facet = None
    for item in items:
        if item["facet_rank"] != previous_facet:
            previous_facet = item["facet_rank"]
            facet_shift = 0
        members = [item]
        if limit is not None and embedding_client.token_count(item["text"]) > limit:
            members = _source_identifier_members(
                item, text=text, start=start, end=end,
                capacity=_MAX_NODE_FACETS - len(items) - global_shift + 1,
                embedding_client=embedding_client, limit=limit,
            )
        for offset, member in enumerate(members):
            fitted.append({**member, "global_rank": item["global_rank"] + global_shift + offset,
                           "item_rank": item["item_rank"] + facet_shift + offset})
        global_shift += len(members) - 1
        facet_shift += len(members) - 1
    return fitted


def prepare_node_facets(summary, analysis, *, text, start, end, embedding_client, limit):
    """Apply identical text budgets before saving a model node and before embedding it."""
    if limit is not None and embedding_client.token_count(summary) > limit:
        raise EmbeddingContextExceeded("one generated summary exceeds embedding context; regenerate more compactly")
    if analysis is None:
        return []
    analysis = normalize_analysis(analysis)
    if analysis["document_summary"] != summary:
        raise ValueError("native facet analysis summary differs from verified node")
    return _fit_native_facets(
        flatten_representations(analysis)[1:], text=text, start=start, end=end,
        embedding_client=embedding_client, limit=limit,
    )


def retain_fitting_native_fields(original, retry, *, text, start, end, embedding_client, limit):
    """Keep the previous model's admissible fields and use a checked retry for replacements."""
    original = normalize_analysis(original)
    retry = normalize_analysis(retry)
    summary = original["document_summary"]
    if embedding_client.token_count(summary) > limit:
        summary = retry["document_summary"]

    def fits(facets):
        analysis = {"document_summary": summary, "facets": facets}
        if len(flatten_representations(analysis)) - 1 > _MAX_NODE_FACETS:
            return False
        try:
            prepare_node_facets(summary, analysis, text=text, start=start, end=end,
                                embedding_client=embedding_client, limit=limit)
        except EmbeddingContextExceeded:
            return False
        return True

    if fits(original["facets"]):
        return {"document_summary": summary, "facets": original["facets"]}

    retained = []
    for facet in original["facets"]:
        if fits([*retained, facet]):
            retained.append(facet)
            continue
        items = []
        for item in facet["representations"]:
            candidate = {**facet, "representations": [*items, item]}
            if fits([*retained, candidate]):
                items.append(item)
        if items:
            retained.append({**facet, "representations": items})

    # Keep the original order, then add genuinely new terms from the bounded response.
    for facet in retry["facets"]:
        merged = list(retained)
        for i, prior in enumerate(merged):
            if prior["label"].casefold() == facet["label"].casefold():
                merged[i] = {**prior, "representations": prior["representations"] + facet["representations"]}
                break
        else:
            merged.append(facet)
        merged = normalize_analysis({"document_summary": summary, "facets": merged})["facets"]
        if fits(merged):
            retained = merged
    return {"document_summary": summary, "facets": retained}


def _publisher_keyphrases(text, existing, start, end, embedding_client, limit):
    """Append deterministic, publisher-attributed ranks after the model's existing rows."""
    facet_rank = max((item["facet_rank"] for item in existing), default=0) + 1
    result = []
    for term, anchor in first_page_ccs(text).items():
        if len(existing) + len(result) >= _MAX_NODE_FACETS:
            break
        if not start <= anchor["start_char"] < anchor["end_char"] <= end:
            continue
        if limit is not None and embedding_client.token_count(term) > limit:
            continue
        result.append({
            "kind": "keyphrase", "text": term, "facet_label": "Publisher CCS Concepts",
            "facet_strength": "strong", "item_strength": "strong", "facet_rank": facet_rank,
            "item_rank": len(result) + 1, "global_rank": len(existing) + len(result) + 1,
            "publisher_classification": anchor,
        })
    return result


def summary_rows(
    connection: sqlite3.Connection,
    *,
    original_path: str,
    original_sha256: str,
    original_bytes: int,
    derived_sha256: str,
    derived_text: str,
    summary_start_char: int,
    tree: SummaryTree,
    embedding_client,
    analysis_model: str,
    collection_id: str,
    summary_generation_seed: int = 42,
    require_native_terms: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Prepare native summary rows without writing anything to FileDB."""
    if (
        not isinstance(original_path, str)
        or not original_path
        or original_path.startswith("/")
        or original_path.startswith("./")
        or ".." in Path(original_path).parts
        or len(original_path) > 1500
        or type(original_bytes) is not int
        or original_bytes < 0
        or not isinstance(collection_id, str)
        or not collection_id.strip()
        or not isinstance(analysis_model, str)
        or not analysis_model.strip()
    ):
        raise ValueError("invalid original identity or model information")
    for val in (original_sha256, derived_sha256):
        if not isinstance(val, str) or len(val) != 64 or not all(c in "0123456789abcdef" for c in val):
            raise ValueError("invalid source revision identifier")
    if _sha(derived_text) != derived_sha256:
        raise ValueError("derived text SHA differs from source map")
    validate_complete_summary_tree(tree)
    root = tree.root
    if not isinstance(derived_text, str) or type(summary_start_char) is not int:
        raise ValueError("invalid source text or original offset")
    if tree.source_sha256 != _sha(derived_text[summary_start_char:summary_start_char + tree.original_characters]):
        raise ValueError("summary tree source SHA differs from original derived span")
    if (
        type(summary_start_char) is not int
        or summary_start_char < 0
        or root.start_char != 0
        or root.end_char != tree.original_characters
        or summary_start_char + tree.original_characters > len(derived_text)
    ):
        raise ValueError("summary root is not a contiguous subset of the current derived text")
    row = connection.execute(
        "SELECT id,relative_path_b64,size_bytes FROM filesystem_entries "
        "WHERE entry_type='file' AND relative_path_display=?",
        (original_path,),
    ).fetchone()
    if row is None:
        raise ValueError("original not admitted to canonical KMD FileDB")
    native_id, path_b64, actual_bytes = row
    if actual_bytes != original_bytes:
        raise ValueError("native source size differs from original completion ledger")
    if base64.b64decode(path_b64) != original_path.encode("utf-8"):
        raise ValueError("original KMD path bytes differ from exact relative path")
    seen = set()
    source_chunks = []
    representations = []
    file_id = stable_file_id(collection_id, path_b64)
    now = time.time_ns()
    ordered = sorted(tree.nodes, key=lambda n: (n.level, n.start_char, n.end_char, n.node_id))
    if not ordered or len(ordered) > 50_000:
        raise ValueError("empty or unbounded summary nodes")
    # Never feed a longer summary than the embedding model can actually read.
    # The E5 model in Raspi's current production deployment has 512 tokens.
    limit = None
    if hasattr(embedding_client, "token_count") and hasattr(embedding_client, "model_context"):
        limit = int(embedding_client.model_context().configured_tokens)
    facets_by_node = []
    for node in ordered:
        extra = prepare_node_facets(
            node.summary, node.native_analysis, text=derived_text,
            start=summary_start_char + node.start_char, end=summary_start_char + node.end_char,
            embedding_client=embedding_client, limit=limit,
        )
        facets_by_node.append(extra)
    extra_texts = [item["text"] for items in facets_by_node for item in items]
    publisher_items = []
    if require_native_terms and not any(
        item["kind"] in ("keyword", "keyphrase") for items in facets_by_node for item in items
    ):
        # Each printed subject belongs to the full-document root and is indexed only once.
        if original_path.lower().endswith(".pdf") and root.native_analysis is not None:
            publisher_items = _publisher_keyphrases(
                derived_text, facets_by_node[-1], summary_start_char, summary_start_char + root.end_char,
                embedding_client, limit,
            )
        if not publisher_items:
            raise RuntimeError("requested native keyword and keyphrase generation produced no indexable terms")
        facets_by_node[-1].extend(publisher_items)
    all_texts=[node.summary for node in ordered]+extra_texts
    all_vectors=embedding_client.embed(all_texts)
    if len(all_vectors)!=len(all_texts):
        raise RuntimeError("missing native facet or summary embedding")
    if publisher_items:
        # Quantized E5 batches can change with padding. Keep the model's original batch intact.
        publisher_vectors = embedding_client.embed([item["text"] for item in publisher_items])
        if len(publisher_vectors) != len(publisher_items):
            raise RuntimeError("missing publisher classification embedding")
        all_vectors = [*all_vectors, *publisher_vectors]
    vectors=all_vectors[:len(ordered)]
    extra_vectors=iter(all_vectors[len(ordered):])
    if getattr(embedding_client, "embedding_dimension", None) != 384:
        raise ValueError("expected verified 384-dimensional native embedding model")
    if not getattr(embedding_client, "model", "") or not getattr(embedding_client, "revision", ""):
        raise ValueError("missing embedding model identity and revision")
    for ordinal, (node, vector) in enumerate(zip(ordered, vectors, strict=True)):
        if (
            node.source_sha256 != tree.source_sha256
            or node.prompt_version != _VERSION
            or node.summary_sha256 != _sha(node.summary)
            or node.start_char < 0
            or node.end_char <= node.start_char
            or node.end_char > tree.original_characters
            or node.node_id in seen
        ):
            raise ValueError("summary node source provenance is missing or inconsistent")
        seen.add(node.node_id)
        vector = np.asarray(vector, dtype="<f4")
        if vector.shape != (384,) or not np.isfinite(vector).all():
            raise ValueError("invalid summary vector")
        norm = float(np.linalg.norm(vector))
        if norm < 0.99 or norm > 1.01:
            raise ValueError("summary embedding should already be normalized")
        source_start = summary_start_char + node.start_char
        source_end = summary_start_char + node.end_char
        direct_source = derived_text[source_start:source_end]
        source_hash = _sha(direct_source)
        chunk_index = 3_000_000 + node.level * 100_000 + ordinal
        chunk_id = stable_chunk_id(
            file_id, original_sha256, "chunk", chunk_index, source_start, source_end, source_hash,
        )
        c = {
            "chunk_id": chunk_id,
            "file_id": file_id,
            "collection_id": collection_id + ":summary",
            "filesystem_entry_id": native_id,
            "content_object_id": "sha256:" + derived_sha256,
            "content_sha256": original_sha256,
            "chunk_kind": "chunk",
            "chunk_index": chunk_index,
            "start_char": source_start,
            "end_char": source_end,
            "character_count": len(direct_source),
            "word_count": count_words(direct_source),
            "token_count": node.prompt_tokens,
            "text_sha256": source_hash,
            "embedding_model": None,
            "embedding_model_revision": None,
            "embedding_dimension": None,
            "embedding_dtype": None,
            "embedding_norm": None,
            "embedding_blob": None,
            "embedding_sha256": None,
            "created_at_ns": now,
            "updated_at_ns": now,
        }
        source_chunks.append(c)
        # Half precision is supported by KMD native vector decoding and
        # reduces stored source-summary representations to 768 vector bytes.
        compact = np.asarray(vector, dtype="<f2").tobytes()
        compact_sha = hashlib.sha256(compact).hexdigest()
        representation_id = stable_representation_id(
            chunk_id, analysis_model, _VERSION, embedding_client.model, 0,
        )
        provenance = {
            "source_start_char": source_start,
            "source_end_char": source_end,
            "original_sha256": original_sha256,
            "derived_sha256": derived_sha256,
            "source_passage_sha256": source_hash,
            "source_tree_sha256": tree.source_sha256,
            "summary_sha256": node.summary_sha256,
            "summary_level": node.level,
            "source_label": node.source_label,
            "node_id": node.node_id,
            "children": list(node.child_ids),
            "prompt_tokens": node.prompt_tokens,
            "configured_model_context": node.model_context_tokens,
            "reserved_output_tokens": node.allocated_output_tokens,
            "summary_prompt_policy": node.prompt_policy,
            "source_scope": "retrieval_summary_only_not_original_evidence",
        }
        representations.append({
            "representation_id": representation_id,
            "chunk_id": chunk_id,
            "representation_kind": "summary_long" if node.node_id == tree.root_id else "summary",
            "facet_label": "hierarchical_source_summary",
            "facet_strength": "moderate",
            "item_strength": "moderate",
            "facet_rank": 0,
            "item_rank": 0,
            "global_rank": 0,
            "representation_text": node.summary,
            "representation_text_sha256": node.summary_sha256,
            "analysis_model": analysis_model,
            "analysis_model_fingerprint": None,
            "prompt_version": _VERSION,
            "generation_seed": summary_generation_seed,
            "pipeline_version": _VERSION,
            "generation_json": json.dumps(provenance, ensure_ascii=False, sort_keys=True),
            "attributes_json": json.dumps({"source_label": node.source_label,
                                            "summary_node_id": node.node_id},
                                           ensure_ascii=False, sort_keys=True),
            "embedding_model": embedding_client.model,
            "embedding_model_revision": embedding_client.revision,
            "embedding_dimension": 384,
            "embedding_dtype": "float16",
            "embedding_norm": float(np.linalg.norm(np.frombuffer(compact, dtype="<f2").astype("<f4"))),
            "embedding_blob": sqlite3.Binary(compact),
            "embedding_sha256": compact_sha,
            "analysis_status": "complete",
            "analysis_error": None,
            "created_at_ns": now,
            "updated_at_ns": now,
        })
        for item in facets_by_node[ordinal]:
            native_kind=item["kind"]
            native_text=item["text"]
            native_vector=np.asarray(next(extra_vectors),dtype="<f4")
            if native_vector.shape!=(384,) or not np.isfinite(native_vector).all():
                raise ValueError("invalid native facet embedding")
            if not 0.99<=float(np.linalg.norm(native_vector))<=1.01:
                raise ValueError("native facet embedding must be normalized")
            native_blob=np.asarray(native_vector,dtype="<f2").tobytes()
            native_sha=hashlib.sha256(native_blob).hexdigest()
            extractor = CCS_EXTRACTOR if "publisher_classification" in item else analysis_model
            native_representation_id=stable_representation_id(
                chunk_id, extractor, _VERSION, embedding_client.model,
                int(item["global_rank"]),
            )
            representations.append({
                "representation_id":native_representation_id,
                "chunk_id":chunk_id,
                "representation_kind":native_kind,
                "facet_label":item["facet_label"],
                "facet_strength":item["facet_strength"],
                "item_strength":item["item_strength"],
                "facet_rank":int(item["facet_rank"]),
                "item_rank":int(item["item_rank"]),
                "global_rank":int(item["global_rank"]),
                "representation_text":native_text,
                "representation_text_sha256":_sha(native_text),
                "analysis_model":extractor,
                "analysis_model_fingerprint":None,
                "prompt_version":_VERSION,
                "generation_seed":summary_generation_seed,
                "pipeline_version":_VERSION,
                "generation_json":json.dumps({
                    **provenance,
                    "native_facet_kind":native_kind,
                    **({"publisher_classification": item["publisher_classification"],
                        "source_scope": "publisher_printed_classification"}
                       if "publisher_classification" in item else {}),
                    "native_facet_label":item["facet_label"],
                    **({"identifier_list_member": item["identifier_list_member"]}
                       if "identifier_list_member" in item else {}),
                    "summary_sha256":_sha(native_text),
                    "owning_summary_sha256":node.summary_sha256,
                },ensure_ascii=False,sort_keys=True),
                "attributes_json":json.dumps({
                    "source_label":node.source_label,
                    "summary_node_id":node.node_id,
                    "native_facet_kind":native_kind,
                },ensure_ascii=False,sort_keys=True),
                "embedding_model":embedding_client.model,
                "embedding_model_revision":embedding_client.revision,
                "embedding_dimension":384,
                "embedding_dtype":"float16",
                "embedding_norm":float(np.linalg.norm(
                    np.frombuffer(native_blob,dtype="<f2").astype("<f4")
                )),
                "embedding_blob":sqlite3.Binary(native_blob),
                "embedding_sha256":native_sha,
                "analysis_status":"complete",
                "analysis_error":None,
                "created_at_ns":now,
                "updated_at_ns":now,
            })
    ids = {n.node_id for n in tree.nodes}
    if any(child not in ids for node in tree.nodes for child in node.child_ids):
        raise ValueError("summary tree has unknown child node")
    return source_chunks, representations


def append_summary_rows(
    connection: sqlite3.Connection,
    *,
    chunk_rows: list[dict],
    representation_rows: list[dict],
    max_database_bytes: int = 640 * 1024 * 1024,
) -> dict:
    """Guarded, idempotent transaction. Production resolver must be upgraded first."""
    if not chunk_rows or len(representation_rows) < len(chunk_rows):
        raise ValueError("native summary rows or facet representations are missing")
    chunk_ids={row["chunk_id"] for row in chunk_rows}
    if len(chunk_ids)!=len(chunk_rows):
        raise ValueError("duplicate native summary source node")
    if any(row["chunk_id"] not in chunk_ids for row in representation_rows):
        raise ValueError("native facet representation has no source-linked summary chunk")
    if len({row["representation_id"] for row in representation_rows})!=len(representation_rows):
        raise ValueError("duplicate native facet or summary representation")
    if len([r for r in representation_rows if r["representation_kind"] in ("summary","summary_long")])!=len(chunk_rows):
        raise ValueError("every original summary node requires one native summary embedding")
    if not 1024 * 1024 <= max_database_bytes <= 16 * 1024 * 1024 * 1024:
        raise ValueError("invalid FileDB capacity limit")
    if not connection.execute("PRAGMA foreign_keys").fetchone()[0]:
        raise RuntimeError("foreign keys must be enabled for native KMD writes")
    ccols = ",".join('"' + col + '"' for col in CHUNK_COLUMN_NAMES)
    rcols = ",".join('"' + col + '"' for col in REPRESENTATION_COLUMN_NAMES)
    cinsert = f"INSERT OR IGNORE INTO {CHUNK_TABLE_NAME} ({ccols}) VALUES (" + ",".join("?" for _ in CHUNK_COLUMN_NAMES) + ")"
    rinsert = f"INSERT OR IGNORE INTO {REPRESENTATION_TABLE_NAME} ({rcols}) VALUES (" + ",".join("?" for _ in REPRESENTATION_COLUMN_NAMES) + ")"
    before = connection.total_changes
    connection.execute("BEGIN IMMEDIATE")
    try:
        for chunk in chunk_rows:
            connection.execute(cinsert, [chunk[c] for c in CHUNK_COLUMN_NAMES])
            native = connection.execute(
                f"SELECT text_sha256,content_sha256,collection_id FROM {CHUNK_TABLE_NAME} WHERE chunk_id=?",
                (chunk["chunk_id"],),
            ).fetchone()
            if native is None or tuple(native) != (chunk["text_sha256"], chunk["content_sha256"], chunk["collection_id"]):
                raise RuntimeError("conflicting native KMD source revision for summary anchor")
        for representation in representation_rows:
            connection.execute(rinsert, [representation[c] for c in REPRESENTATION_COLUMN_NAMES])
            saved = connection.execute(
                f"SELECT representation_text_sha256,embedding_sha256,prompt_version "
                f"FROM {REPRESENTATION_TABLE_NAME} WHERE representation_id=?",
                (representation["representation_id"],),
            ).fetchone()
            if saved is None or tuple(saved) != (representation["representation_text_sha256"],
                         representation["embedding_sha256"], representation["prompt_version"]):
                raise RuntimeError("conflicting native KMD summary/embedding revision")
        estimated_size = connection.execute("PRAGMA page_count").fetchone()[0] * connection.execute("PRAGMA page_size").fetchone()[0]
        if estimated_size > max_database_bytes:
            raise RuntimeError("original KMD FileDB storage checkpoint would be exceeded")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("native summary insertion violated KMD foreign key constraints")
        added = connection.total_changes - before
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return {"new_rows": added, "total_summary_nodes_requested": len(chunk_rows), "filedb_estimated_bytes": estimated_size}
