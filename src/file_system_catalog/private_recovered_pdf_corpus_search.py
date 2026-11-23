"""Ephemeral, strictly provisional original-PDF recovered physical-page search.

Index ONLY the 411 previously unreadable pages in owner-private, SHA-checked
and not-released corrected text previews. Never turn a machine-generated page
into an authoritative quotation; independent publisher-physical-page proof
is necessary even for a short excerpt. Neither original FileDB nor the normal
approved staged page index is modified or replaced by this temporary search.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import re
import sqlite3
from .staged_physical_page_search import MARKER
from .hybrid_search import query_terms
from .source_evidence import _best_excerpt

SHA=re.compile('[0-9a-f]{64}\\Z')
MAX_ORIGINALS=170
MAX_REPLACED_PAGES=423
MAX_TEXT_BYTES=64*1024*1024


def candidate_source_page_bodies(approved_private_preview_text:str,
                                  page_specific_change_receipts:list[dict],
                                  physical_page_count:int)->list[tuple[int,str,str]]:
    """Extract only explicitly replaced physical pages, with exact body SHA.

    Never assume that a passed-in unpublished PDF body is accurate, or that a
    source heading placed by an untrusted model constitutes an original page.
    The original page count and candidate file SHA must be attested by caller.
    """
    if (not isinstance(approved_private_preview_text,str)
          or not approved_private_preview_text.startswith('KMD source provenance')
          or type(physical_page_count) is not int or not 1<=physical_page_count<=450
          or not isinstance(page_specific_change_receipts,list)
          or not 0<=len(page_specific_change_receipts)<=32):
        raise ValueError('invalid bounded private original PDF recovery preview')
    markers=list(MARKER.finditer(approved_private_preview_text))
    if len(markers)!=physical_page_count or [int(x[2]) for x in markers]!=list(range(1,physical_page_count+1)):
        raise ValueError('private original PDF page marker boundaries are wrong or incomplete')
    # An approved original may have no replacement text that passes the
    # strict zero-NUL quality guard. Retain its source identity with zero
    # indexed pages instead of pretending the missing text was recovered.
    changes={}
    for record in page_specific_change_receipts:
        if not isinstance(record,dict):raise ValueError('malformed machine text page change receipt')
        page=record.get('original_physical_pdf_page')
        sha=record.get('after_provisional_text_sha256')
        if (type(page) is not int or not 1<=page<=physical_page_count
              or page in changes or not isinstance(sha,str) or SHA.fullmatch(sha) is None
              or record.get('source_page_text_is_not_visually_or_semantically_verified') is not True):
            raise ValueError('malformed or duplicate provisional physical PDF page revision')
        changes[page]=sha
    result=[]
    for k,m in enumerate(markers):
        page=int(m[2]);if_sha=changes.get(page)
        if if_sha is None:continue
        stop=markers[k+1].start() if k+1<len(markers) else len(approved_private_preview_text)
        body=approved_private_preview_text[m.end():stop]
        if (hashlib.sha256(body.encode()).hexdigest()!=if_sha or not body.strip()
             or '\x00' in body or '\ufffd' in body or len(body)>18001
             or MARKER.search(body)):
            raise ValueError('changed/replaced/corrupted private PDF recovery page body')
        result.append((page,body,if_sha))
    if len(result)!=len(changes):raise ValueError('not all approved physical-page change receipts found')
    return result


class PrivateRecoveredOriginalPageRAMIndex:
    """One-shot in-memory search; never persist an unauthorized text index."""
    def __init__(self,*,max_originals=MAX_ORIGINALS):
        if type(max_originals) is not int or not 1<=max_originals<=MAX_ORIGINALS:
            raise ValueError('invalid bounded provisional original source cohort')
        self.db=sqlite3.connect(':memory:')
        self.db.execute('CREATE VIRTUAL TABLE physical_recovered USING fts5(body,path UNINDEXED,tokenize=\'porter unicode61 remove_diacritics 2\')')
        self.max_originals=max_originals;self.seen=set();self.entries={};self.bytes=0;self.closed=False

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
    def close(self):
        if not self.closed:self.db.close();self.closed=True

    def add_approved_source_preview(self,*,original_path:str,original_sha256:str,
              private_preview_text:str,physical_page_count:int,
              source_specific_page_change_receipts:list[dict],
              current_old_live_derived_sha256:str,
              approved_first_corrected_page_stage_sha256:str,
              unapproved_preview_sha256:str,
              native_original_file_id:int):
        if (self.closed or not isinstance(original_path,str)
           or not original_path.endswith('.pdf') or original_path.startswith('/')
           or '\\' in original_path or any(p in ('','.','..') for p in original_path.split('/'))
           or original_path in self.seen or len(self.seen)>=self.max_originals
           or not all(isinstance(v,str) and SHA.fullmatch(v) for v in
            (original_sha256,current_old_live_derived_sha256,
             approved_first_corrected_page_stage_sha256,unapproved_preview_sha256))
           or type(native_original_file_id) is not int or native_original_file_id<1):
            raise ValueError('invalid, duplicate, or oversized original publisher PDF source cohort')
        if (not isinstance(private_preview_text,str) or
             hashlib.sha256(private_preview_text.encode()).hexdigest()!=unapproved_preview_sha256):
            raise ValueError('private provisional original PDF review preview changed since approval snapshot')
        rows=candidate_source_page_bodies(private_preview_text,
              source_specific_page_change_receipts,physical_page_count)
        for page,body,sha in rows:
            rowid=len(self.entries)+1
            if rowid>MAX_REPLACED_PAGES:raise ValueError('private temporary PDF page cohort exceeds guarded limit')
            self.bytes+=len(body.encode('utf8'))
            if self.bytes>MAX_TEXT_BYTES:raise ValueError('temporary recovered PDF text exceeds RAM safety guard')
            self.db.execute('INSERT INTO physical_recovered(rowid,body,path) VALUES (?,?,?)',(rowid,body,original_path))
            self.entries[rowid]={'original_path':original_path,'original_sha256':original_sha256,
              'physical_pdf_page':page,'body':body,
              'body_sha256':sha,'native_original_file_id':native_original_file_id,
              'approved_first_corrected_page_stage_sha256':approved_first_corrected_page_stage_sha256,
              'current_old_live_derived_sha256':current_old_live_derived_sha256,
              'unapproved_preview_sha256':unapproved_preview_sha256,
              'full_page_text_not_approved_as_publisher_quote':True}
        self.seen.add(original_path)

    def commit(self):
        if self.closed:raise ValueError('closed private index')
        self.db.commit()

    def provisional_page_for_independent_quotation_check(self,*,original_path:str,
             original_sha256:str,physical_pdf_page:int,source_body_sha256:str)->str:
        """Return only a pinned private recovered page for second-extractor QA.

        This is **not** a verified original citation. Its sole purpose is
        selecting a specific short verbatim fragment for an independent
        original-file, same-physical-page corroboration. Public query output
        must never include the unverified full page body.
        """
        if (self.closed or not isinstance(original_path,str) or original_path not in self.seen
             or type(physical_pdf_page) is not int or physical_pdf_page<1
             or not isinstance(original_sha256,str) or SHA.fullmatch(original_sha256) is None
             or not isinstance(source_body_sha256,str) or SHA.fullmatch(source_body_sha256) is None):
            raise ValueError('invalid private recovered original page selection')
        rows=[entry for entry in self.entries.values()
              if entry['original_path']==original_path and
              entry['physical_pdf_page']==physical_pdf_page and
              entry['original_sha256']==original_sha256 and
              entry['body_sha256']==source_body_sha256]
        if len(rows)!=1:
            raise ValueError('original physical page, current revision or private candidate not present')
        text=rows[0]['body']
        if hashlib.sha256(text.encode()).hexdigest()!=source_body_sha256:
            raise ValueError('private source text changed after approved page indexing')
        return text

    def ranked(self,query:str,*,limit:int=30,original_path:str|None=None)->list[dict]:
        if (self.closed or type(limit) is not int or not 1<=limit<=100
              or not isinstance(query,str) or not 1<=len(query)<=400):
            raise ValueError('invalid bounded provisional recovered PDF search query')
        tokens=query_terms(query)
        if not tokens:raise ValueError('query does not contain searchable source terms')
        clause=' OR '.join('"'+t.replace('"','""')+'"' for t in tokens)
        if original_path is not None:
            if original_path not in self.seen:
                raise ValueError('requested original PDF source not present in bounded private retrieval cohort')
            rows=self.db.execute('SELECT rowid,bm25(physical_recovered,1.0) '
                'FROM physical_recovered WHERE physical_recovered MATCH ? AND path=? ORDER BY 2,1 LIMIT ?',
                (clause,original_path,limit)).fetchall()
        else:
            rows=self.db.execute('SELECT rowid,bm25(physical_recovered,1.0) '
                'FROM physical_recovered WHERE physical_recovered MATCH ? ORDER BY 2,1 LIMIT ?',
                (clause,limit)).fetchall()
        accepted=[]
        for rowid,score in rows:
            base=self.entries[int(rowid)]
            start,stop,covered=_best_excerpt(base['body'],tokens,length=520)
            if len(covered)<min(2,len(tokens)):continue
            excerpt=base['body'][start:stop].strip()
            if len(excerpt)<35:continue
            accepted.append({'original_path':base['original_path'],
               'original_sha256':base['original_sha256'],
               'physical_pdf_page':base['physical_pdf_page'],
               'source_body_sha256':base['body_sha256'],
               'native_original_file_id':base['native_original_file_id'],
               'current_old_live_derived_sha256':base['current_old_live_derived_sha256'],
               'approved_first_corrected_page_stage_sha256':base['approved_first_corrected_page_stage_sha256'],
               'unapproved_preview_sha256':base['unapproved_preview_sha256'],
               'provisional_machine_excerpt':excerpt,
               'covered_distinct_query_terms':sorted(covered),
               'source_page_full_text_not_approved_as_publisher_quote':True,
               'lexical_score':float(score)})
        return accepted
