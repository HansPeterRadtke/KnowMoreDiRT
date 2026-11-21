"""Source-byte verified, independent physical-PDF-page retrieval acceptance.

A frozen expected original PAGE is separate from the source selected by a
semantic search. Scoring physical page membership does NOT prove generated
answer factuality, complete passages, or the position of other valid pages.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sqlite3

from .source_review_audit import safe_file, sha_file
from .publisher_pdf_page import verify_pdf_page_excerpts

SHA=re.compile(r'[0-9a-f]{64}\Z')
MARKER=re.compile(r'\[\[SOURCE UNIT: PDF page ([1-9][0-9]{0,5})\]\]')


def validate_publisher_page_gold(*, frozen:dict, source_map:sqlite3.Connection,
                                 original_root:Path,derivative_root:Path,
                                 completed_receipts:dict[str,str], verifier=None,
                                 max_bytes:int=64*1024*1024) -> list[dict]:
    """Rehash full original PDF and page-anchored excerpt before retrieval."""
    if (not isinstance(frozen,dict)
            or frozen.get('schema')!='kmd.publisher_original_physical_page_gold.v1'
            or frozen.get('frozen_before_query_replay') is not True):
        raise ValueError('not a previously frozen independent publisher page holdout')
    cases=frozen.get('cases')
    if not isinstance(cases,list) or not 1<=len(cases)<=20:
        raise ValueError('invalid bounded physical page gold population')
    if type(max_bytes) is not int or not 128<=max_bytes<=64*1024*1024:
        raise ValueError('invalid bounded full-source verification byte size')
    verify=verifier or verify_pdf_page_excerpts
    visited=set();verified=[]
    for case in cases:
        if not isinstance(case,dict):
            raise ValueError('invalid physical page gold case')
        id=case.get('id');path=case.get('source_original_path')
        query=case.get('query');page=case.get('gold_page')
        quote=case.get('publisher_source_excerpt')
        quoted_sha=case.get('publisher_excerpt_sha256')
        orig_sha=case.get('original_sha256');der_sha=case.get('derived_sha256')
        page_sha=case.get('original_pdf_physical_page_text_sha256')
        if (not isinstance(id,str) or not 1<=len(id)<=128 or id in visited
                or not isinstance(query,str) or not 8<=len(query)<=750
                or type(page) is not int or not 1<=page<=5000
                or not isinstance(quote,str) or not 35<=len(quote)<=400
                or not isinstance(quoted_sha,str) or not SHA.fullmatch(quoted_sha)
                or hashlib.sha256(quote.encode()).hexdigest()!=quoted_sha
                or not isinstance(orig_sha,str) or not SHA.fullmatch(orig_sha)
                or not isinstance(der_sha,str) or not SHA.fullmatch(der_sha)
                or not isinstance(page_sha,str) or not SHA.fullmatch(page_sha)
                or case.get('physical_page_verified') is not True):
            raise ValueError('physical page gold lacks independent query, source, or excerpt evidence')
        visited.add(id)
        if path is None or not isinstance(path,str) or not path.endswith('.pdf'):
            raise ValueError('physical-page gold must cite one real publisher PDF')
        original=safe_file(original_root,path)
        rows=source_map.execute('''SELECT source_sha256,derived_sha256,derived_path,status
                                   FROM originals WHERE source_path=?''',(path,)).fetchall()
        if (len(rows)!=1 or rows[0][3]!='complete'
                or rows[0][0]!=orig_sha or rows[0][1]!=der_sha
                or completed_receipts.get(path)!=orig_sha):
            raise ValueError('gold publisher original differs from complete acquisition/source-map identity')
        derivative=safe_file(derivative_root,rows[0][2])
        if (original.stat().st_size>max_bytes or derivative.stat().st_size>max_bytes
                or sha_file(original)!=orig_sha or sha_file(derivative)!=der_sha):
            raise ValueError('publisher original or extracted original bytes differ from frozen gold')
        extracted=derivative.read_text('utf8')
        offset=extracted.find(quote)
        if offset<0 or '[[SOURCE UNIT:' not in extracted:
            raise ValueError('gold excerpt absent from current extracted publisher source')
        previous=[(m.start(),int(m.group(1))) for m in MARKER.finditer(extracted) if m.start()<offset]
        if not previous or previous[-1][1]!=page:
            raise ValueError('predeclared quote not assigned to expected extracted physical page')
        # This independent source PDF check is the *gold evidence source*,
        # not a result of the native retrieval program being benchmarked.
        proof=verify(original,orig_sha,[{'source_section':f'PDF page {page}',
                                          'source_excerpt':quote}])
        if (not isinstance(proof,list) or len(proof)!=1 or not isinstance(proof[0],dict)
                or proof[0].get('status')!='verified' or proof[0].get('page')!=page
                or proof[0].get('page_text_sha256')!=page_sha):
            raise ValueError('original PDF physical-page gold cannot be independently reverified')
        verified.append({
            'id':id,'query':query,'query_sha256':hashlib.sha256(query.encode()).hexdigest(),
            'source_original_path':path,'original_sha256':orig_sha,'derived_sha256':der_sha,
            'gold_physical_pdf_page':page,'frozen_original_pdf_page_text_sha256':page_sha,
            'original_publisher_and_derived_bytes_rehashed':True,
        })
    return verified


def score_source_evidence_pages(*, gold:dict, result:dict,top:int) -> dict:
    """Count verified original physical page hits, never raw derivative labels."""
    if type(top) is not int or not 1<=top<=10:
        raise ValueError('invalid physical-page retrieval cutoff')
    if (not isinstance(gold,dict) or not isinstance(result,dict)
            or result.get('original_path')!=gold.get('source_original_path')
            or result.get('original_sha256')!=gold.get('original_sha256')
            or result.get('derived_sha256')!=gold.get('derived_sha256')
            or result.get('query')!=gold.get('query')
            or result.get('retrieval_engine')!='verified_original_ephemeral_fts5_v1'):
        raise ValueError('native source-page query differs from pinned publisher original')
    hits=result.get('passages')
    if not isinstance(hits,list) or len(hits)>top:
        raise ValueError('native source evidence passages missing or unbounded')
    found=[];rank=None
    for i,row in enumerate(hits,1):
        if (not isinstance(row,dict) or row.get('original_path')!=gold['source_original_path']
                or row.get('original_sha256')!=gold['original_sha256']
                or row.get('derived_sha256')!=gold['derived_sha256']
                or row.get('status')!='source_lexical_candidate_not_verified_answer'):
            raise ValueError('native source passage lacks current publisher revision')
        quote=row.get('source_excerpt');digest=row.get('excerpt_sha256')
        if (not isinstance(quote,str) or not quote
                or not isinstance(digest,str) or not SHA.fullmatch(digest)
                or hashlib.sha256(quote.encode()).hexdigest()!=digest):
            raise ValueError('native displayed source passage is not exact digest checked')
        page_record=row.get('publisher_pdf_page')
        if not isinstance(page_record,dict):
            raise ValueError('native physical page check missing')
        page=page_record.get('page') if page_record.get('status')=='verified' else None
        if page is not None:
            if (type(page) is not int or not 1<=page<=5000
                    or not isinstance(page_record.get('page_text_sha256'),str)
                    or not SHA.fullmatch(page_record['page_text_sha256'])):
                raise ValueError('native original page identity malformed')
        found.append({'native_passage_rank':i,'verified_original_physical_page':page,
                      'original_page_proof_status':page_record.get('status'),
                      'candidate_excerpt_sha256':digest})
        if page==gold['gold_physical_pdf_page'] and rank is None:
            rank=i
    return {
        'id':gold['id'],'query_sha256':gold['query_sha256'],
        'source_original_sha256_reverified':gold['original_sha256'],
        'expected_independently_verified_physical_pdf_page':gold['gold_physical_pdf_page'],
        'first_verified_gold_page_rank_at_top':rank,
        'retrieved_original_page_candidates':found,
        'gold_page_hit':rank is not None,
        'page_recall_not_answer_truth':True,
    }
