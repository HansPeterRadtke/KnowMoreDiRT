"""Independent current-source attestation for private frozen claim inputs.

This is a source-byte and source-identity check, NOT a verification of the
manual semantic labels. The caller must provide the real original-file page
reader and current native/source-map identity check. Do not treat benchmark
receipts or labels as facts about the world.
"""
from __future__ import annotations
import hashlib
import re
from typing import Callable

from .source_claim_assessment import validate_dataset


def _sha(text:str)->str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def attest_frozen_claim_sources(dataset:dict, *,
        read_current_pdf_context:Callable[[str,int],dict],
        native_identities:Callable[[list[str]],dict],
        final_current_source_check:Callable[[dict],None])->dict:
    """Check source spans on the same current original and actual page.

    Group shared PDF pages so separate spans do not require repeated full
    physical extractions. Recheck selected native IDs and current source
    revisions after all page extraction; unrelated indexing is allowed.
    This is not atomic multi-store generation pinning.
    """
    validate_dataset(dataset)
    publisher={k:v for k,v in dataset["sources"].items()
               if v["kind"]=="publisher_pdf_reading_order_span"}
    originals=sorted({v["source_path"] for v in publisher.values()})
    if len(originals)>30:
        raise ValueError("too many publisher originals for one final source-revision check")
    before=native_identities(originals) if originals else {}
    if (not isinstance(before,dict) or set(before)!=set(originals)
            or not all(type(x) is int and x>0 for x in before.values())
            or len(set(before.values()))!=len(before)):
        raise ValueError("missing or ambiguous current native original identities")
    pages={}; revisions={}; out=[]
    for key,src in publisher.items():
        path=src["source_path"]; page=src["physical_page"]
        if (path,page) not in pages:
            pages[path,page]=read_current_pdf_context(path,page)
        context=pages[path,page]
        if (not isinstance(context,dict)
                or context.get("source_map_revision_reverified_after_physical_page_extraction") is not True
                or not isinstance(context.get("source"),dict)
                or not isinstance(context.get("physical_pages"),dict)):
            raise ValueError("missing independently verified current original page context")
        current=context["source"]; block=context["physical_pages"]
        if (current.get("original_path")!=path
                or current.get("original_sha256")!=src["original_sha256"]
                or current.get("current_derived_sha256")!=src["derived_sha256"]
                or current.get("source_url")!=src["source_url"]
                or current.get("title")!=src["title"]
                or current.get("publisher_original_actual_sha256_verified") is not True
                or current.get("source_map_derivative_actual_sha256_verified") is not True
                or block.get("original_pdf_sha256")!=src["original_sha256"]
                or block.get("requested_center_physical_page")!=page
                or block.get("original_pdf_revision_verified_before_and_after") is not True):
            raise ValueError("frozen source differs from current approved publisher revision or identity")
        rows=block.get("pages")
        if not isinstance(rows,list) or len(rows)!=1 or not isinstance(rows[0],dict):
            raise ValueError("expected one explicitly selected physical page")
        center=rows[0]
        layout_digest=center.get("physical_page_poppler_sha256")
        if not isinstance(layout_digest,str) or re.fullmatch(r"[0-9a-f]{64}",layout_digest) is None:
            raise ValueError("missing original layout page digest")
        alternate=block.get("optional_center_reading_order_projection")
        if (center.get("physical_page")!=page or center.get("status")!="verified_original_page_text"
                or not isinstance(alternate,dict) or alternate.get("physical_page")!=page
                or alternate.get("status")!="same_physical_page_alternate_reading_order_text"
                or alternate.get("extraction_mode")!="poppler_default_reading_order_utf8_not_layout"
                or alternate.get("exact_layout_page_poppler_sha256")!=center.get("physical_page_poppler_sha256")
                or alternate.get("complete_extractor_output_not_truncated") is not True
                or alternate.get("reading_order_poppler_sha256")!=src["page_text_sha256"]
                or alternate.get("extracted_text_sha256")!=src["page_text_sha256"]):
            raise ValueError("frozen source reading-order page proof does not match")
        body=alternate.get("extracted_page_text")
        if not isinstance(body,str) or _sha(body)!=src["page_text_sha256"]:
            raise ValueError("current original page text digest mismatch")
        normalized=" ".join(body.split())
        if normalized[src["start_char"]:src["end_char"]]!=src["text"]:
            raise ValueError("frozen source span differs from exact current physical page")
        old=revisions.get(path)
        if old is not None and old!=current:
            raise ValueError("same publisher original changed during page attestation")
        revisions[path]=current
        out.append({"evidence_id":key,"original_path":path,
                    "native_original_id":before[path],"physical_page":page,
                    "original_sha256":src["original_sha256"],
                    "derived_sha256":src["derived_sha256"],
                    "reading_order_page_sha256":src["page_text_sha256"],
                    "normalized_span_start":src["start_char"],
                    "normalized_span_end":src["end_char"],
                    "normalized_span_sha256":src["sha256"]})
    if revisions:
        final_current_source_check(revisions)
    after=native_identities(originals) if originals else {}
    if before!=after:
        raise ValueError("native original identity changed during frozen claim attestation")
    return {"publisher_originals_reverified":len(originals),
            "physical_original_pages_reverified":len(pages),
            "publisher_spans_reverified":len(out),
            "synthetic_fixture_spans_not_publisher_attested":len(dataset["sources"])-len(out),
            "source_revisions_and_native_ids_rechecked_after_all_pages":True,
            "semantic_gold_labels_not_proven_by_source_hash":True,
            "cross_store_immutable_generation_pinned":False,
            "safe_to_cutover_live":False,
            "publisher_source_text_not_in_attestation":True,"spans":out}
