"""Read-only RAM-only publisher first-source-unit FTS5 for unsummarized originals.

Never a replacement for native semantic search or a second persistent DB.
Every returned source is rechecked against approved complete acquisition,
current source-map original/derivative SHA and unique native FileDB identity.
A token match or original-page excerpt is a retrieval candidate, not an answer.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sqlite3
from typing import NamedTuple

from .hybrid_search import query_terms
from .source_review_audit import safe_file,sha_file
from .source_evidence import _best_excerpt
from .publisher_pdf_page import verify_pdf_page_excerpts

UNIT=re.compile(r'(?m)^\[\[SOURCE UNIT: ([^\]\r\n]{1,500})\]\][ \t]*(?:\r?\n|$)')
SOURCE_SHA=re.compile('[0-9a-f]{64}\\Z')
FIRST_UNIT_PREFIX_BYTES=16*1024
FIRST_UNIT_INDEX_CHARS=6500
MAX_ROWS=25000


class Entry(NamedTuple):
    original_path: str
    original_sha256: str
    derived_sha256: str
    derived_path: str
    title: str
    first_unit_label: str
    excerpt_start_char: int
    excerpt_end_char: int
    excerpt: str


def source_firstunit_from_prefix(prefix:bytes) -> tuple[str,int,int,str] | None:
    """Exact first publisher unit in an original-derived UTF8 prefix.

    Never mistake KMD's generated catalog header for publisher text.
    Partial trailing UTF8 at byte 16KiB is excluded; no claim of whole-file
    extraction, page or text accuracy follows from a sampling prefix.
    """
    if not isinstance(prefix,bytes) or len(prefix)>FIRST_UNIT_PREFIX_BYTES:
        raise ValueError('oversized or nonbyte source prefix')
    text=prefix.decode('utf8','ignore')
    first=UNIT.search(text)
    if first is None:
        return None
    next_unit=UNIT.search(text,first.end())
    ending=min(len(text),first.end()+FIRST_UNIT_INDEX_CHARS,
               next_unit.start() if next_unit else len(text))
    start=first.end()
    segment=text[start:ending]
    if not segment.strip() or len(segment.strip())<25:
        return None
    # A PDF whose text layer decodes as NUL-interleaved glyph codes must
    # remain an explicit source-coverage exclusion, not a readable quote.
    nul=segment.count('\x00')
    if nul>=12 and nul*50>=len(segment):
        return None
    return first.group(1),start,ending,segment


class FirstUnitRAMIndex:
    """Temporary original-path-keyed dual FTS5 index, never a disk schema."""
    def __init__(self,max_records:int=MAX_ROWS):
        if type(max_records) is not int or not 1<=max_records<=MAX_ROWS:
            raise ValueError('source preview cardinality must be bounded')
        self.max_records=max_records
        self.db=sqlite3.connect(':memory:')
        self.db.execute("CREATE VIRTUAL TABLE literal USING fts5(title,source_topic,source_body,tokenize='unicode61 remove_diacritics 2 tokenchars _')")
        self.db.execute("CREATE VIRTUAL TABLE porter USING fts5(title,source_topic,source_body,tokenize='porter unicode61 remove_diacritics 2 tokenchars _')")
        self.entries:dict[int,Entry]={}
        self.failed_samples:list[dict]=[]
        self.bytes_sampled=0
        self.closed=False

    def __enter__(self):return self
    def __exit__(self,*_args):self.close()

    def close(self):
        if not self.closed:
            self.db.close()
            self.closed=True

    def add(self,*,path:str,original_sha:str,derived_sha:str,derived_path:str,
            title:str,original_text_prefix:bytes):
        if self.closed:raise ValueError('first-unit index closed')
        if len(self.entries)>=self.max_records:raise ValueError('source catalog exceeds explicit bound')
        parts=path.split('/') if isinstance(path,str) else ()
        # This quoted catalog text is untrusted until full source proof.
        # Still reject unsafe original identities before they enter FTS.
        if (not isinstance(path,str) or not 1<=len(path)<=1600 or path.startswith('/')
                or '\\' in path or '\x00' in path
                or not parts or any(part in ('','.','..') for part in parts)
                or not isinstance(derived_path,str)
                or not isinstance(title,str) or len(title)>1000
                or not isinstance(original_sha,str) or not SOURCE_SHA.fullmatch(original_sha)
                or not isinstance(derived_sha,str) or not SOURCE_SHA.fullmatch(derived_sha)):
            raise ValueError('publisher first-unit original identity invalid')
        item=source_firstunit_from_prefix(original_text_prefix)
        self.bytes_sampled+=len(original_text_prefix)
        if item is None:
            self.failed_samples.append({'original_path':path,'reason':'publisher_source_unit_not_found_in_bounded_prefix'})
            return
        label,start,end,segment=item
        nextid=len(self.entries)+1
        source_topic=path.split('/',1)[0]
        for name in ('literal','porter'):
            self.db.execute(f'INSERT INTO {name}(rowid,title,source_topic,source_body) VALUES(?,?,?,?)',
                            (nextid,title,source_topic,segment))
        self.entries[nextid]=Entry(path,original_sha,derived_sha,derived_path,title,label,start,end,segment)

    def commit(self):
        if self.closed:raise ValueError('first-unit index closed')
        self.db.commit()

    def ranked(self,query:str,*,channel:str,limit:int=50) -> list[tuple[Entry,int,float]]:
        if self.closed or channel not in ('literal','porter'):
            raise ValueError('unsupported publisher lexical source discovery channel')
        if type(limit) is not int or not 1<=limit<=100:
            raise ValueError('unbounded source candidate count')
        terms=query_terms(query)
        if not terms:
            raise ValueError('no informative first-source-unit search terms')
        clause=' OR '.join('"'+term.replace('"','""')+'"' for term in terms)
        rows=self.db.execute(f'SELECT rowid,bm25({channel},3.0,0.4,1.0) FROM {channel} '
                             f'WHERE {channel} MATCH ? ORDER BY 2,rowid LIMIT ?',
                             (clause,limit)).fetchall()
        return [(self.entries[ident],ident,float(score)) for ident,score in rows]


def load_complete_source_previews(*,source_map:sqlite3.Connection,derived_root:Path,
                                  index:FirstUnitRAMIndex,max_source_records:int=MAX_ROWS) -> dict:
    """Read only sixteen kilobytes per approved extracted original for discovery.

    Index entries are *unverified candidates* until verify_index_candidate
    rehashes their complete original and derived publisher source bytes.
    """
    if type(max_source_records) is not int or not 1<=max_source_records<=MAX_ROWS:
        raise ValueError('untrusted source-map import volume')
    rowset=source_map.execute("SELECT source_path,source_sha256,derived_sha256,derived_path,original_title "
                              "FROM originals WHERE status='complete' ORDER BY source_path").fetchall()
    if len(rowset)>max_source_records:raise ValueError('source-map exceeds bounded ephemeral discovery population')
    for path,original_sha,derivative_sha,derived_path,title in rowset:
        try:
            source=safe_file(derived_root,derived_path)
            with source.open('rb') as stream:
                prefix=stream.read(FIRST_UNIT_PREFIX_BYTES)
            index.add(path=path,original_sha=original_sha,derived_sha=derivative_sha,
                      derived_path=derived_path,title=title or '',original_text_prefix=prefix)
        except (ValueError,OSError,TypeError) as error:
            # No unsafe candidate is indexed. The skip must remain visible
            # to callers, never interpreted as a verified source omission.
            index.failed_samples.append({'original_path':str(path)[:1600],
                                          'reason':f'{type(error).__name__}: {str(error)[:180]}'})
    index.commit()
    return {'eligible_complete_source_map_records':len(rowset),
            'publisher_firstunit_rows_indexed':len(index.entries),
            'bounded_prefix_bytes_read':index.bytes_sampled,
            'skipped_sources':len(index.failed_samples)}


def verify_index_candidate(*,entry:Entry,original_root:Path,derived_root:Path,
                           completed_receipts:dict[str,str],native_db:sqlite3.Connection,
                           source_map:sqlite3.Connection,query:str,verify_pdf_page:bool=False,
                           pdf_verifier=None,max_original_bytes:int=512*1024*1024) -> dict:
    """Read-only source/provenance attestation before displaying a lexical hit."""
    if type(verify_pdf_page) is not bool:
        raise ValueError('invalid opt-in PDF page verification')
    if type(max_original_bytes) is not int or not 1<=max_original_bytes<=512*1024*1024:
        raise ValueError('unbounded publisher source evidence original')
    if completed_receipts.get(entry.original_path)!=entry.original_sha256:
        raise ValueError('original no longer has an exact completed acquisition receipt')
    current=source_map.execute(
        'SELECT source_sha256,derived_sha256,derived_path,status,original_title '
        'FROM originals WHERE source_path=?',(entry.original_path,)
    ).fetchall()
    if (len(current)!=1 or current[0][3]!='complete'
            or current[0][0]!=entry.original_sha256
            or current[0][1]!=entry.derived_sha256
            or current[0][2]!=entry.derived_path
            or (current[0][4] or '')!=entry.title):
        raise ValueError('first-page lexical candidate is not the current source-map revision')
    original=safe_file(original_root,entry.original_path)
    derived=safe_file(derived_root,entry.derived_path)
    if original.stat().st_size>max_original_bytes or derived.stat().st_size>max_original_bytes:
        raise ValueError('publisher original/derived source exceeds bounded full byte check')
    if sha_file(original)!=entry.original_sha256 or sha_file(derived)!=entry.derived_sha256:
        raise ValueError('publisher original or derivative full SHA differs from current source-map revision')
    native=native_db.execute("SELECT id FROM filesystem_entries WHERE relative_path_display=? "
                             "AND entry_type='file'",(entry.original_path,)).fetchall()
    if len(native)!=1 or type(native[0][0]) is not int:
        raise ValueError('source has no unique native KnowMoreDiRT FileDB original entry')
    raw=derived.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry.derived_sha256:
        raise ValueError('source derivative changed between full-source verification and evidence extraction')
    text=raw.decode('utf8','strict')
    if (entry.excerpt_end_char>len(text)
            or text[entry.excerpt_start_char:entry.excerpt_end_char]!=entry.excerpt):
        raise ValueError('native extracted original page/section span changed after first page sample')
    start,end,_covered=_best_excerpt(entry.excerpt,query_terms(query),length=520)
    excerpt=entry.excerpt[start:end]
    passage_start=entry.excerpt_start_char+start
    passage_end=entry.excerpt_start_char+end
    if text[passage_start:passage_end]!=excerpt:
        raise ValueError('source-body preview location differs from current original derivative')
    pdf_status={'status':'not_requested_or_not_pdf','physical_pdf_page_verified':False}
    if verify_pdf_page and entry.original_path.casefold().endswith('.pdf'):
        match=re.fullmatch(r'PDF page ([1-9][0-9]{0,5})',entry.first_unit_label)
        if match:
            verifier=pdf_verifier or verify_pdf_page_excerpts
            proof=verifier(original,entry.original_sha256,[{
                'source_section':entry.first_unit_label,
                'source_excerpt':excerpt,
            }])[0]
            if proof.get('status')=='verified' and proof.get('page')==int(match[1]):
                pdf_status={**proof,'physical_pdf_page_verified':True}
            else:
                pdf_status={'status':'not_independently_verified','physical_pdf_page_verified':False,
                            'reason':str(proof.get('reason','publisher_page_mismatch'))[:100]}
        else:
            pdf_status={'status':'first_source_unit_lacks_a_physical_pdf_page','physical_pdf_page_verified':False}
    return {
        'original_path':entry.original_path,
        'original_title':entry.title,
        'native_original_filedb_entry_id':native[0][0],
        'publisher_original_sha256':entry.original_sha256,
        'current_derivative_sha256':entry.derived_sha256,
        'current_approved_publisher_and_extracted_bytes_rehashed':True,
        'source_unit_label':entry.first_unit_label,
        'source_start_char':entry.excerpt_start_char,
        'source_end_char':entry.excerpt_end_char,
        'publisher_first_unit_preview':excerpt,
        'source_preview_start_char':passage_start,
        'source_preview_end_char':passage_end,
        'source_preview_sha256':hashlib.sha256(excerpt.encode()).hexdigest(),
        'physical_pdf_page':pdf_status,
        'quoted_text_is_original_publisher_bytes':False,
        'factual_answer_verified':False,
        'evidence_warning':'This is an FTS-ranked current derived-original preview, NOT a verified answer, semantic confidence or publisher-byte quotation. Physical PDF page proof applies only when explicitly verified.',
    }
