"""RAM-only publisher-page lookup across privately staged, corrected PDFs.

This supplements—not replaces—the canonical native FileDB and original-folder
search. Only independently rehashed original publisher PDFs may substantiate a
physical page citation. All staged text and generated FTS rows are transient.
The live source-map/derivative/vectors/summary trees are never modified.
"""
from __future__ import annotations
from collections import Counter
import hashlib
from pathlib import Path
import re
import sqlite3

from .hybrid_search import query_terms
from .publisher_pdf_page import verify_pdf_page_excerpts
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

    def ranked(self,query:str,*,limit:int=60):
        if self.closed or type(limit) is not int or not 1<=limit<=100:
            raise ValueError('invalid bounded evidence search')
        if not isinstance(query,str) or len(query)>400:
            raise ValueError('invalid bounded query')
        terms=query_terms(query)
        if not terms:raise ValueError('no useful publisher-source search terms')
        clause=' OR '.join('"'+token.replace('"','""')+'"' for token in terms)
        ranked=self.db.execute('SELECT rowid,bm25(passages,1.0) AS score '
                     'FROM passages WHERE passages MATCH ? ORDER BY score,rowid LIMIT ?',
                     (clause,limit)).fetchall()
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
            'temporary_staged_evidence_only_not_native_embedding_or_summary_replacement':True,
            'factual_answer_verified':False,'score':score}
