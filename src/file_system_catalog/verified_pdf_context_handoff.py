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


def verified_physical_page_candidate(payload:dict,original_path:str)->dict:
    if (not isinstance(payload,dict) or not isinstance(original_path,str)
            or payload.get('original_path')!=original_path
            or not isinstance(payload.get('passages'),list)
            or not 1<=len(payload['passages'])<=3
            or not HEX.fullmatch(payload.get('original_sha256',''))
            or not HEX.fullmatch(payload.get('derived_sha256',''))):
        raise ValueError('unverifiable exact-original KMD evidence request')
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
        proof=passage.get('publisher_pdf_page')
        if not isinstance(proof,dict) or proof.get('status')!='verified':
            continue
        page=proof.get('page')
        if (type(page) is not int or not 1<=page<=999999
                or not isinstance(proof.get('page_text_sha256'),str)
                or not HEX.fullmatch(proof['page_text_sha256'])):
            raise ValueError('source reader claimed a page without a valid physical PDF proof')
        return {'first_verified_source_excerpt_rank':rank,
            'original_path':original_path,
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
    if (src.get('original_path')!=anchor.get('original_path')
            or block.get('requested_center_physical_page')!=anchor.get('physical_pdf_page')
            or src.get('original_sha256')!=block.get('original_pdf_sha256')
            or not block.get('required_center_excerpt_verified')
            or not page_context.get('source_map_revision_reverified_after_physical_page_extraction')
            or not isinstance(pages,list)):
        raise ValueError('original PDF context changed since publisher-page source search')
    center=[v for v in pages if v.get('physical_page')==anchor['physical_pdf_page']]
    if (len(center)!=1 or center[0].get('physical_page_poppler_sha256')!=anchor['physical_pdf_page_text_sha256']):
        raise ValueError('physical PDF page text digest diverges across source/evidence operators')
    return {'schema':'kmd.source_page_to_neighbor_original_context_no_answer_claim.v1',
            'status':'verified_original_pdf_page_context_not_factual_answer',
            'original_path':anchor['original_path'],
            'first_verified_source_excerpt_rank':anchor['first_verified_source_excerpt_rank'],
            'original_source_excerpt_sha256':anchor['source_excerpt_sha256'],
            'physical_source_page_sha256_cross_verified_twice':True,
            'source_revision_and_original_pdf_sha_reverified':True,
            'source_context_factual_answer_not_established':True,
            'source':src,'physical_pages':block,
            'warning':'Publisher benchmark questions, quotes, dreams, assumptions and obsolete claims remain scoped examples. Inspect the complete governing text and actual document genre before answering real-world questions.'}
