"""Ephemeral original-PDF clean-island search for STRICTLY skipped glyph pages.

Every candidate is a private machine-text locator only. Split on invalid glyphs
BEFORE tokenization; no quote is sourced from it without independent matching
Poppler text on the same hash-pinned publisher-original physical PDF page.
Nothing is added to native FileDB/source-map/derivative caches or indexed persistently.
"""
from __future__ import annotations
import hashlib
import re
import sqlite3
from .hybrid_search import query_terms
from .source_evidence import _best_excerpt

SHA = re.compile(r'[0-9a-f]{64}\Z')
BROKEN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]')
MAX_PAGES=12
MAX_ISLANDS=512
MAX_RAW_BYTES=4*1024*1024
MAX_ISLAND_BYTES=18000
MAX_RAM_TEXT_BYTES=2*1024*1024


def candidate_original_glyph_islands(raw:bytes,*,max_islands:int=MAX_ISLANDS)->list[str]:
    """Keep separate clean text segments; never repair or join across a bad glyph.

    These are search candidates, NOT authoritative original publisher quotes.
    They may have incorrect column order even when they contain clean words.
    """
    if (not isinstance(raw,bytes) or not raw or len(raw)>=MAX_RAW_BYTES
            or type(max_islands) is not int or not 1<=max_islands<=MAX_ISLANDS):
        raise ValueError('unbounded or absent physical original PDF machine-text candidate')
    try:text=raw.decode('utf8','strict')
    except UnicodeDecodeError as exc:raise ValueError('invalid UTF-8 original PDF machine text') from exc
    chunks=[]
    for segment in BROKEN.split(text):
        if not segment.strip():continue
        words=segment.split()
        # Text position windows in the same *unbroken* source segment only.
        # Never clip words or create an artificial line across corrupt glyphs.
        while words:
            selected=[];chars=0
            for w in words:
                if len(w.encode('utf8'))>MAX_ISLAND_BYTES:
                    raise ValueError('machine page contains oversized untrusted token')
                size=len(w.encode('utf8'))+1
                if selected and chars+size>MAX_ISLAND_BYTES:break
                selected.append(w);chars+=size
            if not selected:raise ValueError('invalid source text island selection')
            if len(selected)>=8:
                body=' '.join(selected)
                if ('[[SOURCE UNIT:' in body or '\ufffd' in body or '\x00' in body
                        or len(body.encode('utf8'))>MAX_ISLAND_BYTES):
                    raise ValueError('private source page marker or invalid glyph in clean island')
                chunks.append(body)
                if len(chunks)>max_islands:raise ValueError('too many unverified source fragments')
            # Long segments overlap by at most eight intact words to avoid
            # losing a query at a limit, while never crossing a bad glyph.
            count=len(selected)
            if count==len(words):break
            words=words[max(1,count-8):]
    return chunks


class OriginalGlyphIslandRAMIndex:
    """Bounded, one-shot RAM-only lexical discovery, never publisher verification."""
    def __init__(self):
        self.db=sqlite3.connect(':memory:')
        self.db.execute('CREATE VIRTUAL TABLE text_fragments USING fts5(body,path UNINDEXED,tokenize="porter unicode61 remove_diacritics 2")')
        self.entries={};self.keys=set();self.bytes=0;self.closed=False
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        if not self.closed:self.db.close();self.closed=True
    def add_pinned_page(self,*,path:str,original_sha256:str,physical_page:int,
                        raw_machine_text:bytes,original_page_count:int,
                        native_original_file_id:int,current_old_derived_sha256:str,
                        corrected_staged_derived_sha256:str)->int:
        if (self.closed or not isinstance(path,str) or not path.endswith('.pdf')
                or not 1<=len(path)<=1600 or path.startswith('/') or '\\' in path
                or any(p in ('','.','..') for p in path.split('/'))
                or any(not isinstance(x,str) or SHA.fullmatch(x) is None for x in
                       (original_sha256,current_old_derived_sha256,corrected_staged_derived_sha256))
                or type(physical_page) is not int or type(original_page_count) is not int
                or not 1<=physical_page<=original_page_count<=450
                or type(native_original_file_id) is not int or native_original_file_id<1
                or (path,physical_page) in self.keys or len(self.keys)>=MAX_PAGES):
            raise ValueError('invalid or duplicated publisher original PDF source/glyph page')
        islands=candidate_original_glyph_islands(raw_machine_text)
        if len(self.entries)+len(islands)>MAX_ISLANDS:
            raise ValueError('temporary isolated glyph retrieval index exceeds page/fragment safety limits')
        for body in islands:
            n=len(body.encode('utf8'))
            self.bytes+=n
            if self.bytes>MAX_RAM_TEXT_BYTES:raise ValueError('temporary original glyph snippets exceed bounded RAM budget')
            rowid=len(self.entries)+1
            self.db.execute('INSERT INTO text_fragments(rowid,body,path) VALUES(?,?,?)',(rowid,body,path))
            self.entries[rowid]={'path':path,'original_sha256':original_sha256,'physical_pdf_page':physical_page,
                                 'body':body,'body_sha256':hashlib.sha256(body.encode('utf8')).hexdigest(),
                                 'native_original_file_id':native_original_file_id,
                                 'current_old_derived_sha256':current_old_derived_sha256,
                                 'corrected_staged_derived_sha256':corrected_staged_derived_sha256,
                                 'not_original_PDF_visual_or_full_page_fidelity':True}
        self.keys.add((path,physical_page))
        return len(islands)
    def commit(self):
        if self.closed:raise ValueError('closed original-page ephemeral search')
        self.db.commit()
    def ranked(self,query:str,*,limit:int=30,original_path:str|None=None)->list[dict]:
        if (self.closed or not isinstance(query,str) or not 1<=len(query)<=400
                or type(limit) is not int or not 1<=limit<=60):
            raise ValueError('invalid original glyph island query')
        terms=query_terms(query)
        if not terms:raise ValueError('question lacks original-source search terms')
        expr=' OR '.join('"'+x.replace('"','""')+'"' for x in terms)
        # FTS5 Porter stemming can rank single-term morphological matches ahead
        # of the exact multi-term fragments required by _best_excerpt. Scan the
        # entire strictly bounded in-memory index before applying the caller's
        # result limit, otherwise the top N raw FTS hits can all be rejected
        # and hide a valid lower-ranked original physical-page locator.
        if original_path is None:
            rows=self.db.execute('SELECT rowid,bm25(text_fragments,1.0) FROM text_fragments WHERE text_fragments MATCH ? ORDER BY 2,1 LIMIT ?',
                                (expr,MAX_ISLANDS)).fetchall()
        else:
            if original_path not in {path for path,page in self.keys}:
                raise ValueError('requested original source is outside strict unavailable-page cohort')
            rows=self.db.execute('SELECT rowid,bm25(text_fragments,1.0) FROM text_fragments WHERE text_fragments MATCH ? AND path=? ORDER BY 2,1 LIMIT ?',
                                 (expr,original_path,MAX_ISLANDS)).fetchall()
        accepted=[]
        for rowid,score in rows:
            e=self.entries[int(rowid)]
            start,stop,covered=_best_excerpt(e['body'],terms,length=520)
            if len(covered)<min(2,len(terms)):continue
            accepted.append({'row_id':int(rowid),'path':e['path'],'original_sha256':e['original_sha256'],
                'physical_pdf_page':e['physical_pdf_page'],'body_sha256':e['body_sha256'],
                'native_original_file_id':e['native_original_file_id'],
                'current_old_derived_sha256':e['current_old_derived_sha256'],
                'corrected_staged_derived_sha256':e['corrected_staged_derived_sha256'],
                'query_terms_in_machine_only_locator':sorted(covered),'lexical_score':float(score),
                'source_excerpt_not_corroborated_or_quoted':True})
            if len(accepted)>=limit:break
        return accepted
    def private_fragment_for_second_extractor(self,*,row_id:int,path:str,original_sha256:str,
                                               physical_page:int,body_sha256:str)->str:
        if (self.closed or type(row_id) is not int or row_id not in self.entries
                or not isinstance(body_sha256,str) or SHA.fullmatch(body_sha256) is None):
            raise ValueError('missing private original page fragment')
        e=self.entries[row_id]
        if (e['path']!=path or e['original_sha256']!=original_sha256 or
            e['physical_pdf_page']!=physical_page or e['body_sha256']!=body_sha256 or
            hashlib.sha256(e['body'].encode()).hexdigest()!=body_sha256):
            raise ValueError('private original source identity, physical page or text changed')
        return e['body']
