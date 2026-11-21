"""Read-only RAM-only publisher first-source-unit FTS5 for unsummarized originals.

Never a replacement for native semantic search or a second persistent DB.
Every returned source is rechecked against approved complete acquisition,
current source-map original/derivative SHA and unique native FileDB identity.
A token match or original-page excerpt is a retrieval candidate, not an answer.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import sqlite3
import subprocess
from typing import NamedTuple

from .hybrid_search import query_terms
from .source_review_audit import safe_file,sha_file
from .source_evidence import _best_excerpt
from .publisher_pdf_page import verify_pdf_page_excerpts

UNIT=re.compile(r'(?m)^\[\[SOURCE UNIT: ([^\]\r\n]{1,500})\]\][ \t]*(?:\r?\n|$)')
SOURCE_SHA=re.compile('[0-9a-f]{64}\Z')
PDF_LABEL=re.compile(r'(?m)^\[\[SOURCE UNIT: PDF page ([1-9][0-9]{0,5})(?:, continued section [1-9][0-9]*)?\]\]')
PDFINFO_PAGES=re.compile(r'(?m)^Pages:[ \t]+([1-9][0-9]{0,5})[ \t]*$')
FIRST_UNIT_PREFIX_BYTES=16*1024
MAX_FALLBACK_PREFIX_BYTES=64*1024
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


def source_firstunit_from_prefix(prefix:bytes,*,max_prefix_bytes:int=FIRST_UNIT_PREFIX_BYTES) -> tuple[str,int,int,str] | None:
    """Earliest *usable* publisher unit in an original-derived UTF8 prefix.

    A reference book may begin with image-only title/copyright pages. Some
    PDFs have NUL-interleaved text-layer glyph codes only on the front page
    but fully readable publisher text on the next original physical page.
    Search bounded consecutive *actual* source-unit markers; never stitch
    separate pages together or treat generated catalog text as a source.
    The returned coordinates always refer to the unmodified current text.
    No claim of exhaustive text/physical-page truth follows from sampling.
    """
    if (type(max_prefix_bytes) is not int
            or max_prefix_bytes not in (FIRST_UNIT_PREFIX_BYTES,MAX_FALLBACK_PREFIX_BYTES)
            or not isinstance(prefix,bytes) or len(prefix)>max_prefix_bytes):
        raise ValueError('oversized or nonbyte source prefix')
    text=prefix.decode('utf8','ignore')
    markers=list(UNIT.finditer(text))
    for i,marker in enumerate(markers[:64]):
        start=marker.end()
        ending=min(len(text),start+FIRST_UNIT_INDEX_CHARS,
                   markers[i+1].start() if i+1<len(markers) else len(text))
        segment=text[start:ending]
        if len(segment.strip())<25:
            continue
        # A physical page whose extraction is unreadable must be excluded,
        # even if another earlier/shorter unit carries the query title.
        nul=segment.count('\x00')
        if nul>=12 and nul*50>=len(segment):
            continue
        return marker.group(1),start,ending,segment
    return None


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
        self.extended_source_preview_reads=0
        self.closed=False

    def __enter__(self):return self
    def __exit__(self,*_args):self.close()

    def close(self):
        if not self.closed:
            self.db.close()
            self.closed=True

    def add(self,*,path:str,original_sha:str,derived_sha:str,derived_path:str,
            title:str,original_text_prefix:bytes,
            max_prefix_bytes:int=FIRST_UNIT_PREFIX_BYTES):
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
        item=source_firstunit_from_prefix(original_text_prefix,max_prefix_bytes=max_prefix_bytes)
        self.bytes_sampled+=len(original_text_prefix)
        if max_prefix_bytes==MAX_FALLBACK_PREFIX_BYTES and len(original_text_prefix)>FIRST_UNIT_PREFIX_BYTES:
            self.extended_source_preview_reads+=1
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
                # Only exceptional sources lacking any readable first-page
                # evidence may use the larger bounded prefix; the normal
                # seventeen-thousand-source scan remains unchanged.
                if source_firstunit_from_prefix(prefix) is None:
                    prefix+=stream.read(MAX_FALLBACK_PREFIX_BYTES-FIRST_UNIT_PREFIX_BYTES)
            bound=(MAX_FALLBACK_PREFIX_BYTES if len(prefix)>FIRST_UNIT_PREFIX_BYTES
                   else FIRST_UNIT_PREFIX_BYTES)
            index.add(path=path,original_sha=original_sha,derived_sha=derivative_sha,
                      derived_path=derived_path,title=title or '',original_text_prefix=prefix,
                      max_prefix_bytes=bound)
        except (ValueError,OSError,TypeError) as error:
            # No unsafe candidate is indexed. The skip must remain visible
            # to callers, never interpreted as a verified source omission.
            index.failed_samples.append({'original_path':str(path)[:1600],
                                          'reason':f'{type(error).__name__}: {str(error)[:180]}'})
    index.commit()
    # Display only source identities and bounded skip reasons, never copied
    # publisher text or generated catalog metadata. A partial sample is not
    # proof that the omitted original is missing from the library.
    skipped=index.failed_samples[:150]
    return {'eligible_complete_source_map_records':len(rowset),
            'publisher_firstunit_rows_indexed':len(index.entries),
            'bounded_prefix_bytes_read':index.bytes_sampled,
            'extended_source_preview_reads':index.extended_source_preview_reads,
            'ordinary_preview_bytes':FIRST_UNIT_PREFIX_BYTES,
            'exceptional_extended_preview_max_bytes':MAX_FALLBACK_PREFIX_BYTES,
            'skipped_sources':len(index.failed_samples),
            'skipped_source_identities':skipped,
            'skipped_source_details_truncated':len(index.failed_samples)>len(skipped)}


def publisher_pdf_page_map_bounds(original:Path, derived_text:str) -> dict:
    """Read-only ORIGINAL-PDF page count compared with every DERIVED page label.

    The check is deliberately a *negative page-provenance gate*: exceeding
    the publisher's actual page count conclusively invalidates some cached
    page labels, but matching counts do NOT prove correct page positions.
    Excerpts still require independent physical original-page verification.
    """
    if not isinstance(derived_text,str):
        raise ValueError('PDF page-map check requires a current complete derivative')
    labels=[int(z) for z in PDF_LABEL.findall(derived_text)]
    if not labels:
        return {'status':'no_derived_pdf_page_labels',
                'publisher_page_position_verified':False}
    before=original.stat(follow_symlinks=False)
    try:
        output=subprocess.run(
            ['/usr/bin/pdfinfo',str(original)],capture_output=True,
            text=True,timeout=8,check=False,
            env={**os.environ,'LC_ALL':'C'},
        )
    except (OSError,subprocess.TimeoutExpired):
        after=original.stat(follow_symlinks=False)
        if (not os.path.samestat(before,after) or before.st_size!=after.st_size
                or before.st_mtime_ns!=after.st_mtime_ns
                or before.st_ctime_ns!=after.st_ctime_ns):
            raise ValueError('publisher original changed during physical PDF page-count inspection')
        return {'status':'physical_page_count_unavailable',
                'publisher_page_position_verified':False,
                'max_derived_page_label':max(labels)}
    after=original.stat(follow_symlinks=False)
    if (not os.path.samestat(before,after) or before.st_size!=after.st_size
            or before.st_mtime_ns!=after.st_mtime_ns
            or before.st_ctime_ns!=after.st_ctime_ns):
        raise ValueError('publisher original changed during physical PDF page-count inspection')
    if output.returncode or len(output.stdout)>32768:
        return {'status':'physical_page_count_unavailable',
                'publisher_page_position_verified':False,
                'max_derived_page_label':max(labels)}
    matches=PDFINFO_PAGES.findall(output.stdout)
    if len(matches)!=1:
        return {'status':'physical_page_count_unavailable',
                'publisher_page_position_verified':False,
                'max_derived_page_label':max(labels)}
    count=int(matches[0]);max_label=max(labels)
    unsafe=max_label>count
    return {
        'status':('derived_page_labels_exceed_original_physical_pages' if unsafe
                  else 'page_label_bounds_consistent_only_not_verification'),
        'original_physical_pdf_pages':count,
        'derived_source_unit_labels':len(labels),
        'max_derived_page_label':max_label,
        'derived_pdf_page_map_suspect':unsafe,
        'publisher_page_position_verified':False,
        'warning':('Some cached extracted PDF page labels are invalid; verify actual publisher page independently before any page citation.'
                   if unsafe else 'Bounds agree, but no individual physical PDF page or answer truth was verified.'),
    }


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
    is_pdf=entry.original_path.casefold().endswith('.pdf')
    page_scope=(publisher_pdf_page_map_bounds(original,text) if is_pdf
                else {'status':'not_pdf','publisher_page_position_verified':False})
    pdf_status={'status':'not_requested_or_not_pdf','physical_pdf_page_verified':False}
    if page_scope.get('derived_pdf_page_map_suspect'):
        pdf_status={'status':'unverified_derived_page_map_exceeds_physical_pdf',
                    'physical_pdf_page_verified':False,
                    'reason':'The cached derivative has page labels past the original publisher PDF physical page count.'}
    if verify_pdf_page and is_pdf:
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
        'source_pdf_page_map_integrity':page_scope,
        'source_unit_label_scope':'derived_source_text_only_not_physical_page_proof',
        'quoted_text_is_original_publisher_bytes':False,
        'factual_answer_verified':False,
        'evidence_warning':'This is an FTS-ranked current derived-original preview, NOT a verified answer, semantic confidence or publisher-byte quotation. Physical PDF page proof applies only when explicitly verified.',
    }
