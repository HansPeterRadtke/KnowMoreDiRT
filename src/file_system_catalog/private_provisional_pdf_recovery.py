"""Privately assemble *unverified* text-recovery previews from publisher PDFs.

The complete original source and the corrected physical-page markup remain
untouched; only physical pages previously skipped by the safe staged reader
may have their *provisional* machine text substituted in a new independent
private preview. A preview is NOT an approved source-map derivative, quote,
native reindex generation, user answer or rollback authorization.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from .staged_physical_page_search import MARKER,source_units_from_corrected_text
from .alternate_pdf_page_search import exact_skipped_staged_physical_pages


def assemble_unverified_original_pdf_glyph_preview(
     approved_physical_stage_text:str,*,physical_page_count:int,
     alternate_pages:list[dict])->dict:
    """Page-by-page exact change audit; never drop pages, alter labels or lie.

    The candidate text is useful for *previewing* retrieval of damaged
    source pages, not production. Unavailable alternate pages retain the
    original, damaged Poppler source text and the omission remains explicit.
    Every unchanged original page's literal source body remains byte-identical
    (at the Unicode text level) to the earlier approved corrected stage.
    """
    if (not isinstance(approved_physical_stage_text,str)
            or not approved_physical_stage_text.startswith('KMD source provenance')
            or type(physical_page_count) is not int
            or not 1<=physical_page_count<=450
            or not isinstance(alternate_pages,list)
            or len(alternate_pages)>32):
        raise ValueError('invalid approved original physical PDF page preview input')
    omitted=exact_skipped_staged_physical_pages(approved_physical_stage_text,
                    physical_page_count,physical_page_count)
    if len(alternate_pages)!=len(omitted) or not omitted:
        raise ValueError('missing or duplicated previously skipped source pages')
    page_ids=[x.get('physical_pdf_page') if isinstance(x,dict) else None for x in alternate_pages]
    if page_ids!=omitted:
        raise ValueError('alternate source pages must exactly match the skipped original page numbers')
    replace={p['physical_pdf_page']:p for p in alternate_pages
             if p.get('status')=='source_bounded_provisional_machine_text'}
    for page,record in replace.items():
        body=record.get('text')
        if (not isinstance(body,str) or not body.strip() or '\x00' in body or '\ufffd' in body
            or MARKER.search(body) is not None or len(body)>18000):
            raise ValueError('unreadable or untrusted replacement of an original physical PDF page')
        if hashlib.sha256(body.encode()).hexdigest()!=record.get('display_text_sha256'):
            raise ValueError('provisional machine text SHA failed')
    for p in alternate_pages:
        if p.get('status') not in ('source_bounded_provisional_machine_text','unavailable'):
            raise ValueError('unknown alternate machine source text provenance status')
        if p.get('status')=='unavailable' and p.get('text') is not None:
            raise ValueError('unavailable source must not supply invented text')
    markers=list(MARKER.finditer(approved_physical_stage_text))
    assert len(markers)==physical_page_count
    pieces=[approved_physical_stage_text[:markers[0].start()]]
    changes=[];unchanged=[]
    for i,marker in enumerate(markers):
        n=int(marker[2]);end=markers[i+1].start() if i+1<len(markers) else len(approved_physical_stage_text)
        prior=approved_physical_stage_text[marker.end():end]
        new=replace.get(n)
        pieces.append(marker[0])
        if new is not None:
            body=new['text']
            if not body.endswith('\n'):body+='\n'
            pieces.append(body)
            changes.append({'original_physical_pdf_page':n,
                'before_text_sha256':hashlib.sha256(prior.encode()).hexdigest(),
                'after_provisional_text_sha256':hashlib.sha256(body.encode()).hexdigest(),
                'machine_extractor_raw_text_sha256':new['raw_machine_text_sha256'],
                'source_page_text_is_not_visually_or_semantically_verified':True})
        else:
            pieces.append(prior)
            unchanged.append({'original_physical_pdf_page':n,
                'text_sha256':hashlib.sha256(prior.encode()).hexdigest(),
                'prior_exact_source_unit_preserved':True})
    preview=''.join(pieces)
    markers_after=list(MARKER.finditer(preview))
    if ([int(x[2]) for x in markers_after]!=list(range(1,physical_page_count+1))
          or len(markers_after)!=len(markers)):
        raise ValueError('provisional assembly changed original physical page identities')
    new_omitted=exact_skipped_staged_physical_pages(preview,physical_page_count,physical_page_count)
    if not set(new_omitted).issubset(set(omitted)):
        raise ValueError('provisional source generation damaged previously readable PDF source pages')
    if len(omitted)-len(new_omitted)!=len(changes):
        raise ValueError('provisional candidate did not recover the intended bounded original pages')
    # Compare each unchanged byte-exact source unit, independently of total
    # assembled text SHA. The wrapper and all physical page labels are equal.
    for i,marker in enumerate(markers):
        n=int(marker[2]);if_new=markers_after[i]
        old_end=markers[i+1].start() if i+1<len(markers) else len(approved_physical_stage_text)
        new_end=markers_after[i+1].start() if i+1<len(markers_after) else len(preview)
        if (n not in replace and approved_physical_stage_text[marker.end():old_end]
             !=preview[if_new.end():new_end]):
            raise ValueError('provisional candidate mutated an unselected publisher physical page')
    return {'schema':'kmd.unreleased_private_original_pdf_glyph_repair_preview.v1',
      'preview_text':preview,
      'approved_prior_stage_sha256':hashlib.sha256(approved_physical_stage_text.encode()).hexdigest(),
      'unreleased_private_preview_sha256':hashlib.sha256(preview.encode()).hexdigest(),
      'original_physical_page_count':physical_page_count,
      'previously_unreadable_page_units':len(omitted),
      'provisional_alt_recovery_count':len(changes),
      'unrecovered_original_pages':new_omitted,
      'new_temporary_searchable_original_physical_pages':len(list(source_units_from_corrected_text(
        preview,physical_page_count,physical_page_count))),
      'prior_temporary_searchable_original_physical_pages':len(list(source_units_from_corrected_text(
        approved_physical_stage_text,physical_page_count,physical_page_count))),
      'replaced_page_proof_without_book_text':changes,
      'all_original_page_count_and_label_order_preserved':True,
      'unchanged_original_physical_page_bodies_exactly_preserved':True,
      'not_a_publisher_visual_validation':True,
      'not_approved_for_live_source_map_filedb_vector_or_summary_cutover':True,
      'private_review_only_no_automatic_promotion':True}
