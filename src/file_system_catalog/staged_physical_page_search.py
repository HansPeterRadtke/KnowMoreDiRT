"""RAM-only publisher-page lookup across privately staged, corrected PDFs.

This supplements—not replaces—the canonical native FileDB and original-folder
search. Only independently rehashed original publisher PDFs may substantiate a
physical page citation. All staged text and generated FTS rows are transient.
The live source-map/derivative/vectors/summary trees are never modified.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .hybrid_search import query_terms
from .publisher_pdf_page import verify_pdf_page_excerpts
from .verified_pdf_neighbor_context import verified_page_neighbors
from .source_review_audit import safe_file, sha_file
from .source_evidence import _best_excerpt

SHA=re.compile(r'[a-f0-9]{64}\Z')
MARKER=re.compile(r'(?m)^\[\[SOURCE UNIT: (PDF page ([1-9][0-9]{0,5})(?:, continued section ([1-9][0-9]*))?)\]\][ \t]*(?:\r?\n|$)')
MAX_RECORDS=327
MAX_STAGED_BYTES=32*1024*1024
MAX_TOTAL_BYTES=128*1024*1024
MAX_ROWS=30000


def source_units_from_corrected_text(text:str, physical_pages:int, expected_max:int):
    """Read only the publisher body, rejecting malformed page identities."""
    if not isinstance(text,str) or type(physical_pages) is not int or not 1<=physical_pages<=450:
        raise ValueError('untrusted PDF physical page count')
    labels=list(MARKER.finditer(text))
    if not labels or not text.startswith('KMD source provenance'):
        raise ValueError('staged publisher text lacks known source-unit provenance wrapper')
    if len(labels)>MAX_ROWS or type(expected_max) is not int or int(labels[-1][2])!=expected_max:
        raise ValueError('wrong staged source-map physical page bound')
    previous=0;previous_section=0
    for i,marker in enumerate(labels):
        page=int(marker[2]);continued=int(marker[3] or 0)
        if page>physical_pages or page<previous:
            raise ValueError('out-of-bounds or backward corrected source page labels')
        if page==previous and (continued==0 or continued!=previous_section+1):
            raise ValueError('repeated physical PDF page without correct continued-section sequence')
        if page>previous and continued not in (0,1):
            raise ValueError('source-page continuation does not begin at its first section')
        previous=page;previous_section=continued
        end=labels[i+1].start() if i+1<len(labels) else len(text)
        body=text[marker.end():end]
        if MARKER.search(body) or '\x00' in body and body.count('\x00')>max(20,len(body)//40):
            # Damaged PDF glyph codes are not quoteable page evidence.
            continue
        if body.strip():
            yield (marker[1],page,body)


class StagedPhysicalPageRAMIndex:
    def __init__(self,max_records:int=MAX_RECORDS):
        if type(max_records) is not int or not 1<=max_records<=MAX_RECORDS:
            raise ValueError('invalid bounded original source cohort')
        self.db=sqlite3.connect(':memory:')
        self.db.execute("CREATE VIRTUAL TABLE passages USING fts5(body, path UNINDEXED, tokenize='porter unicode61 remove_diacritics 2')")
        self.max_records=max_records
        self.loaded_originals=set();self.rows={}
        self.bytes_indexed=0;self.closed=False

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def close(self):
        if not self.closed:self.db.close();self.closed=True

    def add(self, item:dict, *, stage_root:Path):
        if self.closed:raise ValueError('RAM-only index already closed')
        path=item.get('original_path')
        o=item.get('original_sha256');new=item.get('new_staged_derived_sha256')
        stage=item.get('staged_private_file')
        if (not isinstance(path,str) or path in self.loaded_originals
                or len(self.loaded_originals)>=self.max_records
                or not path.endswith('.pdf') or path.startswith('/') or '\\' in path
                or any(x in ('','.','..') for x in path.split('/'))
                or not all(isinstance(v,str) and SHA.fullmatch(v) for v in (o,new))
                or not isinstance(stage,str) or not Path(stage).is_absolute()
                or not Path(stage).is_relative_to(stage_root)):
            raise ValueError('duplicate or unsafe private original source')
        candidate=safe_file(stage_root,str(Path(stage).relative_to(stage_root)))
        if candidate.stat().st_size>MAX_STAGED_BYTES or self.bytes_indexed+candidate.stat().st_size>MAX_TOTAL_BYTES:
            raise ValueError('exceeds bounded private page evidence corpus size')
        if sha_file(candidate)!=new:
            raise ValueError('private staged publisher physical page candidate SHA changed')
        content=candidate.read_text('utf8','strict')
        original_pages=item.get('original_pdf_physical_pages');maximum=item.get('new_max_staged_page_label')
        added=0
        for label,pg,body in source_units_from_corrected_text(content,original_pages,maximum):
            rowid=len(self.rows)+1
            if rowid>MAX_ROWS:raise ValueError('too many physical publisher page units')
            self.db.execute('INSERT INTO passages (rowid,body,path) VALUES (?,?,?)',(rowid,body,path))
            self.rows[rowid]=(path,label,pg,body,item)
            added+=1
        if not added:raise ValueError('no verifiable physical publisher page text in staged source')
        self.loaded_originals.add(path)
        self.bytes_indexed+=candidate.stat().st_size

    def commit(self):
        if self.closed:raise ValueError('RAM index already closed')
        self.db.commit()

    def ranked(self,query:str,*,limit:int=60,original_path:str|None=None):
        if self.closed or type(limit) is not int or not 1<=limit<=100:
            raise ValueError('invalid bounded evidence search')
        if not isinstance(query,str) or len(query)>400:
            raise ValueError('invalid bounded query')
        terms=query_terms(query)
        if not terms:raise ValueError('no useful publisher-source search terms')
        clause=' OR '.join('"'+token.replace('"','""')+'"' for token in terms)
        # A named publisher source must not lose its relevant page merely
        # because another original has more repeated generic query words.
        if original_path is not None:
            if original_path not in self.loaded_originals:
                raise ValueError('original source not present in this verified RAM index')
            sql=('SELECT rowid,bm25(passages,1.0) AS score FROM passages '
                 'WHERE passages MATCH ? AND path=? ORDER BY score,rowid LIMIT ?')
            args=(clause,original_path,limit)
        else:
            sql=('SELECT rowid,bm25(passages,1.0) AS score FROM passages '
                 'WHERE passages MATCH ? ORDER BY score,rowid LIMIT ?')
            args=(clause,limit)
        ranked=self.db.execute(sql,args).fetchall()
        return [(self.rows[int(i)],float(score)) for i,score in ranked]


def verify_staged_page_result(hit,*,query:str,original_root:Path,derived_root:Path,
                              stage_root:Path,completed_receipts:dict[str,str],
                              source_map:sqlite3.Connection,native_db:sqlite3.Connection,
                              verifier=verify_pdf_page_excerpts)->dict:
    """Independently verify ALL three revisions and source-specific PDF page."""
    (path,label,page,body,item),score=hit
    o=item['original_sha256'];old=item['old_derived_sha256'];new=item['new_staged_derived_sha256']
    if completed_receipts.get(path)!=o:
        raise ValueError('original acquisition receipt changed')
    row=source_map.execute('SELECT source_sha256,derived_sha256,derived_path,status FROM originals WHERE source_path=?',(path,)).fetchall()
    if len(row)!=1 or row[0]!=(o,old,path+'.txt','complete'):
        raise ValueError('native source-map original or old derivative revision changed')
    native=native_db.execute("SELECT id FROM filesystem_entries WHERE entry_type='file' AND relative_path_display=?",(path,)).fetchall()
    if len(native)!=1 or native[0][0]!=item.get('native_original_filedb_id'):
        raise ValueError('current native FileDB original identity differs from private stage')
    original=safe_file(original_root,path)
    oldtext=safe_file(derived_root,path+'.txt')
    candidate=safe_file(stage_root,str(Path(item['staged_private_file']).relative_to(stage_root)))
    if sha_file(original)!=o or sha_file(oldtext)!=old or sha_file(candidate)!=new:
        raise ValueError('original, old live derivative, or staged physical PDF page revision differs')
    terms=query_terms(query)
    start,end,covered=_best_excerpt(body,terms,length=520)
    # A correct physical PDF page is not automatically a relevant answer.
    # Metadata-only matches and excerpts matching only a generic word in a
    # multi-term question must not be presented as strong source evidence.
    required_terms=min(2,len(terms))
    if len(covered)<required_terms:
        return {'status':'insufficient_source_query_term_coverage',
                'reason':'not_enough_distinct_query_terms_in_original_passage',
                'original_path':path,'covered_terms':list(covered),'required_terms':required_terms}
    excerpt=body[start:end].strip()
    # Deliberately refuse low-information and unreadable publisher glyph spans.
    if len(excerpt)<35 or ('\x00' in excerpt and excerpt.count('\x00')>3):
        return {'status':'not_independently_verified','reason':'unusable_source_excerpt','original_path':path}
    proof=verifier(original,o,[{'source_section':label,'source_excerpt':excerpt}])[0]
    if proof.get('status')!='verified' or proof.get('page')!=page or proof.get('basis')!='derivative_label_confirmed':
        return {'status':'not_independently_verified','reason':str(proof.get('reason','original_pdf_page_mismatch'))[:100],
                'original_path':path,'source_page_label':label}
    # A verified physical page is only meaningful for the same *current*
    # corpus revision used when the RAM index was built. An extraction or
    # native writer may have advanced while Poppler inspected the original.
    # Verify all source identities a SECOND time after the page proof.
    # A normal SQLite read-only connection provides fresh committed rows
    # between statements; the caller must not hold a read transaction.
    if (not isinstance(proof.get('page_text_sha256'),str)
            or not SHA.fullmatch(proof['page_text_sha256'])):
        raise ValueError('physical publisher page proof lacks a valid text digest')
    after_map=source_map.execute(
        'SELECT source_sha256,derived_sha256,derived_path,status FROM originals WHERE source_path=?',
        (path,)).fetchall()
    if len(after_map)!=1 or after_map[0]!=(o,old,path+'.txt','complete'):
        raise ValueError('native source-map revision changed during physical PDF page verification')
    after_native=native_db.execute(
        "SELECT id FROM filesystem_entries WHERE entry_type='file' AND relative_path_display=?",
        (path,)).fetchall()
    if len(after_native)!=1 or after_native[0][0]!=item.get('native_original_filedb_id'):
        raise ValueError('native original FileDB identity changed during physical PDF page verification')
    if (sha_file(original)!=o or sha_file(oldtext)!=old
            or sha_file(candidate)!=new):
        raise ValueError('original or derivative revision changed during physical PDF page verification')
    return {'status':'publisher_original_physical_pdf_page_verified',
            'original_path':path,'publisher_original_sha256':o,
            'current_old_derived_sha256':old,'corrected_private_derived_sha256':new,
            'native_original_filedb_entry_id':native[0][0],
            'physical_pdf_page':page,'source_section':label,
            'publisher_source_excerpt':excerpt,'source_excerpt_sha256':hashlib.sha256(excerpt.encode()).hexdigest(),
            'page_text_sha256':proof['page_text_sha256'],
            'page_verification_basis':proof['basis'],
            'covered_query_terms':list(covered),
            'read_only_live_source_and_all_three_complete_sha256_rechecked':True,
            'source_map_native_id_and_three_file_sha_rechecked_after_original_pdf_page_proof':True,
            'temporary_staged_evidence_only_not_native_embedding_or_summary_replacement':True,
            'factual_answer_verified':False,'score':score}


def attach_staged_physical_page_neighbors(
    hit, verified:dict, *, original_root:Path, derived_root:Path,
    stage_root:Path, completed_receipts:dict[str,str],
    source_map:sqlite3.Connection, native_db:sqlite3.Connection,
    neighbor_radius:int=1, reader=verified_page_neighbors,
) -> dict:
    """Optional verified original physical page neighbors around staged source evidence.

    The original staged passage must already have passed the separate full
    revision and publisher physical page verification. Independently recheck
    the full original PDF and matching center page SHA, and then recheck every
    current source revision *again* after the neighbor extraction. A changed
    staged or live source during either page read must never be cited.
    """
    if type(neighbor_radius) is not int or neighbor_radius not in (0,1):
        raise ValueError('invalid bounded physical PDF neighbor radius')
    if not isinstance(verified,dict) or verified.get('status')!='publisher_original_physical_pdf_page_verified':
        raise ValueError('neighbor context requires an independently verified staged publisher page')
    (path,label,page,_,item),_ = hit
    original_sha=item.get('original_sha256')
    old=item.get('old_derived_sha256');stage_sha=item.get('new_staged_derived_sha256')
    excerpt=verified.get('publisher_source_excerpt')
    excerpt_sha=verified.get('source_excerpt_sha256')
    page_sha=verified.get('page_text_sha256')
    if (verified.get('original_path')!=path or verified.get('source_section')!=label
            or verified.get('physical_pdf_page')!=page
            or verified.get('publisher_original_sha256')!=original_sha
            or verified.get('current_old_derived_sha256')!=old
            or verified.get('corrected_private_derived_sha256')!=stage_sha
            or verified.get('native_original_filedb_entry_id')!=item.get('native_original_filedb_id')
            or verified.get('page_verification_basis')!='derivative_label_confirmed'
            or not all(isinstance(v,str) and SHA.fullmatch(v) for v in
                       (original_sha,old,stage_sha,page_sha,excerpt_sha))
            or not isinstance(excerpt,str) or not 3<=len(excerpt)<=600
            or hashlib.sha256(excerpt.encode('utf8')).hexdigest()!=excerpt_sha):
        raise ValueError('staged original citation and publisher page anchor identity disagree')
    if completed_receipts.get(path)!=original_sha:
        raise ValueError('completed acquisition original approval changed before context read')
    original=safe_file(original_root,path)
    old_derived=safe_file(derived_root,path+'.txt')
    stage_path=Path(item['staged_private_file'])
    if not stage_path.is_relative_to(stage_root):
        raise ValueError('private corrected candidate is outside approved stage')
    staged=safe_file(stage_root,str(stage_path.relative_to(stage_root)))
    context=reader(original,original_sha,requested_page=page,
                   neighbor_radius=neighbor_radius,required_center_excerpt=excerpt)
    if (not isinstance(context,dict) or not context.get('original_pdf_revision_verified_before_and_after')
            or not context.get('required_center_excerpt_verified')
            or not context.get('separate_original_pages_not_joined_into_one_quotation')
            or context.get('original_pdf_sha256')!=original_sha
            or context.get('requested_center_physical_page')!=page
            or context.get('neighbor_radius')!=neighbor_radius):
        raise ValueError('adjacent original PDF source context proof is incomplete or from another document')
    pages=context.get('pages')
    if not isinstance(pages,list) or not 1<=len(pages)<=3:
        raise ValueError('adjacent publisher page context exceeds the physical page limit')
    physical=[p.get('physical_page') for p in pages if isinstance(p,dict)]
    if len(physical)!=len(pages) or len(set(physical))!=len(physical) or physical!=sorted(physical):
        raise ValueError('adjacent physical pages are not distinct and ordered')
    center=[p for p in pages if p['physical_page']==page]
    if (len(center)!=1 or center[0].get('status')!='verified_original_page_text'
            or not isinstance(center[0].get('extracted_page_text'),str)
            or not center[0]['extracted_page_text'].strip()
            or center[0].get('physical_page_poppler_sha256')!=page_sha):
        raise ValueError('publisher center PDF page digest changed or source text unavailable across two independent readers')
    for neighbor in pages:
        state=neighbor.get('status')
        if state=='adjacent_original_pdf_page_text_unavailable':
            if neighbor.get('extracted_page_text') is not None:
                raise ValueError('unverified original PDF neighbor has invented text')
        elif (state!='verified_original_page_text'
              or not isinstance(neighbor.get('extracted_page_text'),str)
              or not neighbor['extracted_page_text'].strip()):
            raise ValueError('neighboring original PDF page lacks a valid bounded text status')
    if completed_receipts.get(path)!=original_sha:
        raise ValueError('completed acquisition approval changed after context read')
    latest_map=source_map.execute(
        'SELECT source_sha256,derived_sha256,derived_path,status FROM originals WHERE source_path=?',
        (path,)).fetchall()
    if len(latest_map)!=1 or latest_map[0]!=(original_sha,old,path+'.txt','complete'):
        raise ValueError('native source-map revision changed during neighbor extraction')
    latest_native=native_db.execute(
        "SELECT id FROM filesystem_entries WHERE entry_type='file' AND relative_path_display=?",
        (path,)).fetchall()
    if len(latest_native)!=1 or latest_native[0][0]!=item.get('native_original_filedb_id'):
        raise ValueError('native original FileDB identity changed during neighbor extraction')
    if sha_file(original)!=original_sha or sha_file(old_derived)!=old or sha_file(staged)!=stage_sha:
        raise ValueError('original or corrected/old derivative changed during neighbor extraction')
    return {**verified,
        'publisher_original_physical_page_neighbors':context,
        'center_physical_pdf_page_sha_independently_correlated_twice':True,
        'source_map_native_id_and_all_three_source_sha_rechecked_after_neighbor_extraction':True,
        'neighbor_context_factual_answer_verified':False,
        'neighbor_context_does_not_replace_native_filedb_or_index':True}
