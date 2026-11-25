"""Evidence first, exact original physical page second; never classify truth.

The existing KMD source-evidence reader independently verifies an exact
selected original, source SHA, native FileDB identity, derivative offsets,
excerpt text/hash and Poppler publisher-page quote match. Only a physically
verified excerpt is allowed to become the anchor for adjacent page context.
"""
from __future__ import annotations
import hashlib
import re

HEX=re.compile(r'[0-9a-f]{64}\Z')


def validate_passage_selection(passage_rank:int|None,expected_identity:tuple[str,str,str,int,str]|None)->None:
    # Explicit rank selection is meaningful only when it still identifies the
    # very same original revision, derivative revision, excerpt and physical page.
    if passage_rank is None:
        if expected_identity is not None:
            raise ValueError('expected passage identity requires explicit passage rank')
        return
    if (type(passage_rank) is not int or not 1<=passage_rank<=3
            or not isinstance(expected_identity,tuple) or len(expected_identity)!=5
            or not all(isinstance(h,str) and HEX.fullmatch(h) for h in (*expected_identity[:3],expected_identity[4]))
            or type(expected_identity[3]) is not int or not 1<=expected_identity[3]<=999999):
        raise ValueError('explicit passage rank requires all four SHA digests and exact physical page')


def verified_physical_page_candidate(payload:dict,original_path:str,*,
                                     passage_rank:int|None=None,
                                     expected_identity:tuple[str,str,str,int,str]|None=None)->dict:
    validate_passage_selection(passage_rank,expected_identity)
    if (not isinstance(payload,dict) or not isinstance(original_path,str)
            or payload.get('original_path')!=original_path
            or not isinstance(payload.get('passages'),list)
            or not 1<=len(payload['passages'])<=3
            or not isinstance(payload.get('original_sha256'),str)
            or not isinstance(payload.get('derived_sha256'),str)
            or not HEX.fullmatch(payload['original_sha256'])
            or not HEX.fullmatch(payload['derived_sha256'])):
        raise ValueError('unverifiable exact-original KMD evidence request')
    if passage_rank is not None:
        if (payload['original_sha256'],payload['derived_sha256'])!=expected_identity[:2]:
            raise ValueError('explicit ranked passage source revision no longer matches')
        if passage_rank>len(payload['passages']):
            raise ValueError('explicit ranked passage no longer exists')
    for rank,passage in enumerate(payload['passages'],1):
        if not isinstance(passage,dict):raise ValueError('invalid native publisher source passage')
        if (passage.get('original_path')!=original_path
                or passage.get('original_sha256')!=payload['original_sha256']
                or passage.get('derived_sha256')!=payload['derived_sha256']):
            raise ValueError('unrecognized or stale native original source passage identity')
        excerpt=passage.get('source_excerpt');sha=passage.get('excerpt_sha256')
        if (not isinstance(excerpt,str) or not 3<=len(excerpt)<=600
                or not isinstance(sha,str) or not HEX.fullmatch(sha)
                or hashlib.sha256(excerpt.encode('utf-8')).hexdigest()!=sha):
            raise ValueError('native excerpt differs from exact current source SHA')
        if passage_rank is not None:
            if rank!=passage_rank:
                continue
            if sha!=expected_identity[2]:
                raise ValueError('explicit ranked passage excerpt no longer matches')
        proof=passage.get('publisher_pdf_page')
        if not isinstance(proof,dict) or proof.get('status')!='verified':
            if passage_rank is not None:
                raise ValueError('explicit ranked passage lacks verified physical PDF page; no fallback')
            continue
        page=proof.get('page')
        if (type(page) is not int or not 1<=page<=999999
                or not isinstance(proof.get('page_text_sha256'),str)
                or not HEX.fullmatch(proof['page_text_sha256'])):
            raise ValueError('source reader claimed a page without a valid physical PDF proof')
        if passage_rank is not None and (page,proof['page_text_sha256'])!=expected_identity[3:]:
            raise ValueError('explicit ranked passage physical page no longer matches')
        label=passage.get('source_section')
        derived_label_page=None
        if isinstance(label,str):
            old_page=re.match(r'^PDF page ([1-9][0-9]*)(?=$|,| \|| \[)',label)
            if old_page:
                derived_label_page=int(old_page.group(1))
        mismatch=derived_label_page is not None and derived_label_page!=page
        return {'first_verified_source_excerpt_rank':rank if passage_rank is None else None,
            'selected_source_excerpt_rank':rank,
            'source_excerpt_selection':'first_verified' if passage_rank is None else 'explicit_rank_exact_prior_identity',
            'explicit_prior_passage_identity_matched':passage_rank is not None,
            'original_path':original_path,
            'original_sha256':payload['original_sha256'],
            'derived_sha256':payload['derived_sha256'],
            'original_derived_page_label_number':derived_label_page,
            'original_derived_page_label_disagrees_with_physical_page':mismatch,
            'corrected_physical_page_citation_required':mismatch,
            'source_excerpt_sha256':sha,
            'source_excerpt':excerpt,
            'physical_pdf_page':page,
            'physical_pdf_page_text_sha256':proof['page_text_sha256'],
            'page_proof_basis':proof.get('basis'),
            'source_section_derivative_label':passage.get('source_section'),
            'citation_scope':'verified original physical PDF page, not a factual answer or visual-glyph proof'}
    raise ValueError('no verified original physical PDF source excerpt: cannot infer another page')


def bind_original_pdf_context_to_source_evidence(anchor:dict,page_context:dict)->dict:
    if (not isinstance(anchor,dict) or not isinstance(page_context,dict)
            or not isinstance(page_context.get('source'),dict)
            or not isinstance(page_context.get('physical_pages'),dict)):
        raise ValueError('missing exact source and original-page context')
    src=page_context['source'];block=page_context['physical_pages']
    pages=block.get('pages')
    # The same page bytes can persist across distinct editions, or after a
    # changed derivative. Bind the entire source revision, not just the page.
    for key in ('original_sha256','derived_sha256'):
        if not isinstance(anchor.get(key),str) or not HEX.fullmatch(anchor[key]):
            raise ValueError('missing exact source revision in original PDF handoff anchor')
    if (src.get('original_sha256')!=anchor['original_sha256']
            or src.get('current_derived_sha256')!=anchor['derived_sha256']):
        raise ValueError('original or derivative revision changed between source evidence and physical page context')
    if (src.get('original_path')!=anchor.get('original_path')
            or block.get('requested_center_physical_page')!=anchor.get('physical_pdf_page')
            or src.get('original_sha256')!=block.get('original_pdf_sha256')
            or block.get('required_center_excerpt_verified') is not True
            or page_context.get('source_map_revision_reverified_after_physical_page_extraction') is not True
            or not isinstance(pages,list)):
        raise ValueError('original PDF context changed since publisher-page source search')
    center=[v for v in pages if v.get('physical_page')==anchor['physical_pdf_page']]
    if (len(center)!=1 or center[0].get('physical_page_poppler_sha256')!=anchor['physical_pdf_page_text_sha256']):
        raise ValueError('physical PDF page text digest diverges across source/evidence operators')
    if 'optional_center_reading_order_projection' in block:
        alternate=block['optional_center_reading_order_projection']
        if (not isinstance(alternate,dict)
                or alternate.get('physical_page')!=anchor['physical_pdf_page']
                or alternate.get('exact_layout_page_poppler_sha256')!=anchor['physical_pdf_page_text_sha256']
                or alternate.get('extraction_mode')!='poppler_default_reading_order_utf8_not_layout'
                or alternate.get('does_not_replace_verified_layout_excerpt_or_offsets') is not True
                or alternate.get('visual_reading_order_or_factual_claim_truth_verified') is not False):
            raise ValueError('alternate reading-order projection is not bound to the verified center page')
        status=alternate.get('status')
        body=alternate.get('extracted_page_text')
        if status=='same_physical_page_alternate_reading_order_text':
            if not isinstance(body,str) or not body.strip() or len(body)>18000:
                raise ValueError('alternate reading-order text is invalid or oversized')
            digest=hashlib.sha256(body.encode('utf8')).hexdigest()
            contiguous=' '.join(anchor['source_excerpt'].split()) in ' '.join(body.split())
            if (alternate.get('reading_order_poppler_sha256')!=digest
                    or alternate.get('extracted_text_sha256')!=digest
                    or alternate.get('text_characters')!=len(body)
                    or alternate.get('complete_extractor_output_not_truncated') is not True
                    or alternate.get('required_layout_excerpt_also_contiguous_in_reading_order') is not contiguous):
                raise ValueError('alternate reading-order text proof differs from the returned bytes or anchor')
        elif (status!='center_reading_order_text_unavailable' or body is not None
              or alternate.get('required_layout_excerpt_also_contiguous_in_reading_order') is not None):
            raise ValueError('unrecognized alternate reading-order projection status')
    return {'schema':'kmd.source_page_to_neighbor_original_context_no_answer_claim.v1',
            'status':'verified_original_pdf_page_context_not_factual_answer',
            'original_path':anchor['original_path'],
            'first_verified_source_excerpt_rank':anchor['first_verified_source_excerpt_rank'],
            'selected_source_excerpt_rank':anchor['selected_source_excerpt_rank'],
            'source_excerpt_selection':anchor['source_excerpt_selection'],
            'explicit_prior_passage_identity_matched':anchor['explicit_prior_passage_identity_matched'],
            'original_source_excerpt_sha256':anchor['source_excerpt_sha256'],
            'original_derived_page_label_number':anchor['original_derived_page_label_number'],
            'correct_physical_original_pdf_page':anchor['physical_pdf_page'],
            'original_derived_page_label_disagrees_with_physical_page':
                anchor['original_derived_page_label_disagrees_with_physical_page'],
            'physical_page_citation_replaces_old_derivative_label_only_if_disagreement':
                anchor['corrected_physical_page_citation_required'],
            'physical_source_page_sha256_cross_verified_twice':True,
            'source_revision_and_original_pdf_sha_reverified':True,
            'source_original_and_derived_revision_match_across_operators':True,
            'cross_store_immutable_generation_pinned':False,
            'source_context_factual_answer_not_established':True,
            'source':src,'physical_pages':block,
            'warning':'Publisher benchmark questions, quotes, dreams, assumptions and obsolete claims remain scoped examples. Inspect the complete governing text and actual document genre before answering real-world questions.'}
