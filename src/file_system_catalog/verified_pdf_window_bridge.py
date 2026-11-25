"""Optional physically verified-page context after the existing source search.

The input is the already selected source-evidence response, not a new document
search. Page identities and exact layout offsets are independently checked
against current original bytes before any reading-order window is returned.
The caller must perform its normal collective final revision/native-ID guard
AFTER this function completes. No source ranking, original, or index is changed.
"""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Callable

from .reading_order_page_passages import reading_order_page_passages
from .verified_pdf_neighbor_context import PhysicalPdfPageContextUnavailable

_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _path(value):
    return (isinstance(value, str) and 1 <= len(value) <= 1600
            and value.casefold().endswith(".pdf") and not value.startswith("/")
            and "\\" not in value and "\x00" not in value
            and all(p not in ("", ".", "..") for p in value.split("/")))


def validate_expansion_request(page_budget, passage_count, snippet_characters):
    if type(page_budget) is not int or not 0 <= page_budget <= 5:
        raise ValueError("verified PDF expansion page budget must be 0 through 5")
    if page_budget and (type(passage_count) is not int or not 1 <= passage_count <= 5
                        or type(snippet_characters) is not int
                        or not 160 <= snippet_characters <= 2000):
        raise ValueError("verified PDF expansion requires 1-5 source passages and bounded snippets")


def _anchors(documents):
    """Keep the first physical page from each document before second pages.

    Repeated anchors on a page are all validated, but that page consumes just
    one slot. Source ranks are stable and remain local to their document.
    """
    if not isinstance(documents, list) or len(documents) > 5:
        raise ValueError("unbounded original discovery results")
    queues = []
    paths = set(); native_ids=set()
    for position, doc in enumerate(documents, 1):
        if not isinstance(doc, dict) or type(doc.get("rank")) is not int or doc["rank"] != position:
            raise ValueError("invalid or reordered original discovery rank")
        if doc.get("status") != "verified":
            continue
        path = doc.get("original_path")
        original = doc.get("original_sha256")
        derived = doc.get("derived_sha256")
        native = doc.get("filesystem_entry_id")
        passages = doc.get("passages")
        if (not _path(path) or path in paths or not _sha(original) or not _sha(derived)
                or type(native) is not int or native < 1 or native in native_ids
                or not isinstance(passages, list) or len(passages) > 5):
            raise ValueError("verified PDF discovery identity is incomplete")
        paths.add(path); native_ids.add(native)
        pages = OrderedDict()
        for rank, p in enumerate(passages, 1):
            if not isinstance(p, dict):
                raise ValueError("invalid source excerpt")
            proof = p.get("publisher_pdf_page")
            if not isinstance(proof, dict) or proof.get("status") != "verified":
                continue  # Never turn derivative page labels into physical proof.
            text = p.get("source_excerpt")
            sha = p.get("excerpt_sha256")
            page = proof.get("page")
            layout = proof.get("page_text_sha256")
            start, end = proof.get("start_char"), proof.get("end_char")
            if (p.get("original_path") != path
                    or p.get("original_sha256") != original
                    or p.get("derived_sha256") != derived
                    or p.get("filesystem_entry_id") != native
                    or not isinstance(text, str) or not text.strip() or len(text) > 2000
                    or not _sha(sha) or _digest(text) != sha
                    or type(page) is not int or not 1 <= page <= 999999
                    or not _sha(layout)
                    or type(start) is not int or type(end) is not int
                    or not 0 <= start < end or end-start != len(" ".join(text.split()))):
                raise ValueError("verified passage has invalid exact original/page/offset identity")
            g = pages.setdefault(page, {
                "original_path": path, "original_sha256": original,
                "derived_sha256": derived, "native_original_id": native,
                "document_rank": position, "physical_page": page,
                "layout_page_sha256": layout, "anchors": [],
            })
            if g["layout_page_sha256"] != layout:
                raise ValueError("inconsistent physical page digest for repeated anchors")
            g["anchors"].append({"passage_rank": rank, "excerpt_sha256": sha,
                                 "start_char": start, "end_char": end,
                                 "normalized_excerpt": " ".join(text.split())})
        queues.append(list(pages.values()))
    result = []
    for i in range(max(map(len, queues), default=0)):
        result.extend(q[i] for q in queues if i < len(q))
    return result


def _checked_page(group, context):
    if (not isinstance(context, dict)
            or context.get("source_map_revision_reverified_after_physical_page_extraction") is not True
            or not isinstance(context.get("source"), dict)
            or not isinstance(context.get("physical_pages"), dict)):
        raise ValueError("missing current verified original PDF context")
    src, block = context["source"], context["physical_pages"]
    page = group["physical_page"]
    if (src.get("original_path") != group["original_path"]
            or src.get("original_sha256") != group["original_sha256"]
            or src.get("current_derived_sha256") != group["derived_sha256"]
            or src.get("publisher_original_actual_sha256_verified") is not True
            or src.get("source_map_derivative_actual_sha256_verified") is not True
            or block.get("original_pdf_sha256") != group["original_sha256"]
            or block.get("requested_center_physical_page") != page
            or type(block.get("actual_original_pdf_page_count")) is not int
            or not 1 <= page <= block["actual_original_pdf_page_count"]
            or block.get("neighbor_radius") != 0
            or block.get("original_pdf_revision_verified_before_and_after") is not True):
        raise ValueError("original or derivative revision changed during PDF page expansion")
    rows = block.get("pages")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("missing exact physical page")
    center = rows[0]
    body = center.get("extracted_page_text")
    if (center.get("physical_page") != page or center.get("status") != "verified_original_page_text"
            or center.get("physical_page_poppler_sha256") != group["layout_page_sha256"]
            or not isinstance(body, str) or not 1 <= len(body) <= 18000
            or center.get("extracted_text_sha256") != _digest(body)
            or center.get("text_projection_dropped_non_whitespace") is not False):
        raise ValueError("fresh physical page no longer matches prior layout proof")
    policy=center.get("display_layout_policy")
    if (policy=="verbatim_poppler_page_text" and _digest(body)!=group["layout_page_sha256"]
            or policy not in ("verbatim_poppler_page_text",
                               "horizontal_padding_compacted_all_non_whitespace_and_line_boundaries_preserved")):
        raise ValueError("invalid current layout display projection")
    normalized = " ".join(body.split())
    for anchor in group["anchors"]:
        if normalized[anchor["start_char"]:anchor["end_char"]] != anchor["normalized_excerpt"]:
            raise ValueError("verified layout anchor moved or changed at the exact recorded offsets")
    alt = block.get("optional_center_reading_order_projection")
    if (not isinstance(alt, dict) or alt.get("physical_page") != page
            or alt.get("exact_layout_page_poppler_sha256") != group["layout_page_sha256"]
            or alt.get("extraction_mode") != "poppler_default_reading_order_utf8_not_layout"):
        raise ValueError("reading-order projection is not bound to verified layout page")
    if alt.get("status") == "center_reading_order_text_unavailable":
        if alt.get("extracted_page_text") is not None:
            raise ValueError("unavailable reading-order projection has unexpected text")
        return False
    return True  # The page ranker independently validates the complete alternate digest.


def expand_verified_page_evidence(documents, query, *, read_context: Callable,
                                  page_budget: int, snippet_characters: int):
    """Supplement only verified physical pages, with a global page budget.

    A bounded extraction unavailability is distinct from a mismatched proof:
    the former is explicit, the latter refuses the entire combined result.
    """
    validate_expansion_request(page_budget, 5, snippet_characters)
    if page_budget == 0:
        raise ValueError("PDF page expansion must be explicitly requested")
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 600:
        raise ValueError("invalid bounded expansion query")
    candidates = _anchors(documents)
    output = []
    for g in candidates[:page_budget]:
        record = {k: v for k, v in g.items() if k != "anchors"}
        record["source_layout_anchors"] = [
            {k: v for k, v in a.items() if k != "normalized_excerpt"} for a in g["anchors"]]
        try:
            context = read_context(g["original_path"], g["physical_page"])
        except PhysicalPdfPageContextUnavailable:
            output.append({**record, "status": "physical_page_context_unavailable", "windows": []})
            continue
        if not _checked_page(g, context):
            output.append({**record, "status": "reading_order_context_unavailable", "windows": []})
            continue
        ranked = reading_order_page_passages(context, query, top=3,
                     max_characters=snippet_characters, include_inflection_matches=True)
        # The original exact layout anchor can be noncontiguous after
        # column-order normalization. Report that fact, not a fake coordinate.
        alternate=" ".join(context["physical_pages"]["optional_center_reading_order_projection"]["extracted_page_text"].split())
        for a, proof in zip(g["anchors"], record["source_layout_anchors"], strict=True):
            proof["layout_anchor_also_contiguous_in_reading_order_page"]=a["normalized_excerpt"] in alternate
            proof["window_ranks_containing_complete_normalized_layout_anchor"]=[
                w["rank"] for w in ranked["passages"] if a["normalized_excerpt"] in w["text"]]
        output.append({**record, "status": "verified_reading_order_windows",
                       "reading_order_page_sha256": ranked["reading_order_page_sha256"],
                       "normalized_reading_order_page_sha256": ranked["normalized_reading_order_page_sha256"],
                       "offset_basis": ranked["offset_basis"],
                       "sentence_boundaries_are_heuristic_not_guaranteed_complete": True,
                       "windows": ranked["passages"]})
    return {
        "schema": "kmd.optin_verified_selected_pdf_page_windows.v1",
        "distinct_verified_candidate_pages": len(candidates),
        "page_budget": page_budget, "pages_examined": len(output),
        "candidate_pages_not_examined_due_to_budget": max(0, len(candidates)-page_budget),
        "pages_with_verified_reading_order_context": sum(x["status"] == "verified_reading_order_windows" for x in output),
        "candidate_windows": sum(len(x["windows"]) for x in output),
        "allocation": "round_robin_by_distinct_page_within_original_rank",
        "max_windows_per_page": 3,
        "max_characters_per_window": snippet_characters,
        "native_document_ranking_and_original_passages_unchanged": True,
        "no_additional_original_or_physical_page_discovery": True,
        "not_automatic_factual_entailment_or_answer_truth": True,
        "pages": output,
    }


def expansion_audit_summary(result):
    """Record only coordinates/hashes/counts, not publisher text."""
    return {**{k: v for k, v in result.items() if k != "pages"},
            "pages": [{**{k: v for k, v in page.items() if k != "windows"},
                       "windows": [{k: v for k, v in window.items() if k != "text"}
                                   for window in page["windows"]]}
                      for page in result["pages"]]}
