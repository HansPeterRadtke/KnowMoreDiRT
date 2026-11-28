"""Source-scoped read-only ephemeral FTS5 over COMPLETE extracted PDF sections.

This supplements first-unit FTS and recursive native semantic search where
most original source pages have not yet been summarized. Never a persistent
second database or an excuse to trust PDF page labels, model text or metadata
as a literal publisher-byte quote. Callers verify all selected original files
and current source-map/native identities before showing candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import sqlite3
from pathlib import Path

from .hybrid_search import query_terms
from .publisher_firstunit_discovery import UNIT,validate_literal_source_directory_prefix
from .source_review_audit import safe_file

MAX_DOCUMENTS=64
MAX_DERIVED_BYTES_PER_SOURCE=1_500_000
MAX_DERIVED_BYTES_TOTAL=16_000_000
MAX_FULLSOURCE_WINDOWS=6000
WINDOW_CHARS=3600
WINDOW_OVERLAP_CHARS=200
_MAX_SOURCE_ORIGINAL_BYTES=32*1024*1024
_SHA=re.compile(r'[0-9a-f]{64}\Z')


@dataclass(frozen=True)
class FullSection:
    original_path:str
    original_sha256:str
    derived_sha256:str
    derived_path:str
    title:str
    label:str
    source_start:int
    source_end:int
    original_body:str


def bounded_original_pdf_sections(path:str,original_sha:str,derived_sha:str,
                                  derived_path:str,title:str,text:str,
                                  *,report_corrupt_units:bool=False):
    """Index every usable original PDF-derived text unit with exact coordinates.

    Units with corrupt text must be explicitly rejected, not stitched to
    another source/page. Each usable source unit is fully spanned (overlap
    is strictly a retrieval aid) rather than silently truncated at page one.
    """
    if (not isinstance(path,str) or not path.endswith('.pdf') or path.startswith('/')
            or '\\' in path or any(p in ('','.','..') for p in path.split('/'))
            or not isinstance(title,str) or not 1<=len(title)<=1000
            or not isinstance(derived_path,str) or not derived_path==path+'.txt'
            or not isinstance(original_sha,str) or not _SHA.fullmatch(original_sha)
            or not isinstance(derived_sha,str) or not _SHA.fullmatch(derived_sha)
            or not isinstance(text,str) or not 1<=len(text)<=MAX_DERIVED_BYTES_PER_SOURCE):
        raise ValueError('invalid bounded original PDF/derived identity')
    markers=list(UNIT.finditer(text))
    if not markers or len(markers)>5000:
        raise ValueError('complete original lacks bounded source-unit markers')
    if type(report_corrupt_units) is not bool:
        raise ValueError('invalid explicitly logged PDF source-unit omission option')
    spans=[]
    omitted=[]
    for index,m in enumerate(markers):
        label=m.group(1)
        a=m.end()
        b=markers[index+1].start() if index+1<len(markers) else len(text)
        if not a<b or len(text[a:b].strip())<25:
            if report_corrupt_units:
                omitted.append({'original_path':path,'source_unit_label':label,
                                'source_start_char':a,'source_end_char':b,
                                'reason':'empty_or_too_short_extracted_original_PDF_source_unit',
                                'original_section_omitted_from_FTS':True,
                                'physical_pdf_page_not_verified':True})
            continue
        content=text[a:b]
        nul=content.count('\x00')
        if nul>=12 and nul*50>=len(content):
            if not report_corrupt_units:
                raise ValueError('original source unit has unusable NUL-corrupted PDF glyph text')
            omitted.append({'original_path':path,'source_unit_label':label,
                            'source_start_char':a,'source_end_char':b,
                            'nul_characters':nul,
                            'reason':'unreadable_NUL_interleaved_original_PDF_text_layer',
                            'original_section_omitted_from_FTS':True,
                            'physical_pdf_page_not_verified':True})
            continue
        start=a
        while start<b:
            end=min(start+WINDOW_CHARS,b)
            snippet=text[start:end]
            if snippet.strip():
                spans.append(FullSection(path,original_sha,derived_sha,derived_path,title,
                                         label,start,end,snippet))
                if len(spans)>MAX_FULLSOURCE_WINDOWS:
                    raise ValueError('original source page exceeds total bounded windows')
            if end==b:break
            start=end-WINDOW_OVERLAP_CHARS
    if not spans:
        raise ValueError('no readable original PDF source units in complete derivative')
    return (spans,omitted) if report_corrupt_units else spans


class OriginalSectionRAMIndex:
    """Independent literal/Porter FTS5 ranks of complete scoped source units."""

    def __init__(self):
        self.db=sqlite3.connect(':memory:')
        for channel,tokenizer in (
            ('literal','unicode61 remove_diacritics 2 tokenchars _'),
            ('porter','porter unicode61 remove_diacritics 2 tokenchars _'),
        ):
            self.db.execute(f'CREATE VIRTUAL TABLE {channel} USING fts5('
                            'metadata_title,original_section_text,'
                            f"tokenize='{tokenizer}')")
        self.entries={}
        self.documents={}
        self.derivative_bytes_read=0
        self.derived_sha_verified=0
        self.originals_whose_content_is_covered=0
        self.fullsection_windows=0
        self.unreadable_source_units_explicitly_omitted=[]
        self.closed=False

    def __enter__(self):return self
    def __exit__(self,*_args):self.close()

    def close(self):
        if not self.closed:self.db.close();self.closed=True

    def add_source(self,*,path,original_sha,derived_sha,derived_path,
                   title,derived_blob):
        if self.closed:raise ValueError('closed original-source index')
        if not isinstance(derived_blob,bytes):
            raise ValueError('source extraction must be exact bytes')
        if len(self.documents)>=MAX_DOCUMENTS or self.derivative_bytes_read+len(derived_blob)>MAX_DERIVED_BYTES_TOTAL:
            raise ValueError('publisher PDF source scope exceeds bounded in-memory byte budget')
        if path in self.documents:
            raise ValueError('duplicate original PDF source identity')
        if len(derived_blob)>MAX_DERIVED_BYTES_PER_SOURCE:
            raise ValueError('single extracted original source exceeds explicit byte budget')
        if hashlib.sha256(derived_blob).hexdigest()!=derived_sha:
            raise ValueError('original extracted PDF bytes differ from current source map')
        text=derived_blob.decode('utf8','strict')
        windows,omitted=bounded_original_pdf_sections(
            path,original_sha,derived_sha,derived_path,title,text,
            report_corrupt_units=True)
        if len(self.unreadable_source_units_explicitly_omitted)+len(omitted)>10000:
            raise ValueError('original publisher PDF source unit omission list exceeds budget')
        if len(self.entries)+len(windows)>MAX_FULLSOURCE_WINDOWS:
            raise ValueError('publisher source scope exceeds bounded in-memory source-page window count')
        for window in windows:
            rowid=len(self.entries)+1
            for channel in ('literal','porter'):
                self.db.execute(
                    f'INSERT INTO {channel}(rowid,metadata_title,original_section_text) VALUES(?,?,?)',
                    (rowid,window.title,window.original_body))
            self.entries[rowid]=window
        self.documents[path]=(original_sha,derived_sha,derived_path,title,len(text))
        self.unreadable_source_units_explicitly_omitted.extend(omitted)
        self.derivative_bytes_read+=len(derived_blob)
        self.derived_sha_verified+=1
        self.originals_whose_content_is_covered+=1
        self.fullsection_windows+=len(windows)

    def commit(self):
        if self.closed:raise ValueError('source-section index closed')
        self.db.commit()

    def ranked(self,query:str,*,channel:str,limit:int=200):
        if (self.closed or channel not in ('literal','porter')
                or type(limit) is not int or not 1<=limit<=500):
            raise ValueError('invalid original source window retrieval budget')
        if not isinstance(query,str) or not 3<=len(query)<=600:
            raise ValueError('unbounded original document question')
        terms=query_terms(query)
        if not terms:raise ValueError('question has no searchable lexical terms')
        clause=' OR '.join('"'+term.replace('"','""')+'"' for term in terms)
        rows=self.db.execute(f'SELECT rowid,bm25({channel},1.0,1.0) FROM {channel} '
                             f'WHERE {channel} MATCH ? ORDER BY 2,rowid LIMIT ?',
                             (clause,limit)).fetchall()
        return [(self.entries[row],row,float(score)) for row,score in rows]


def add_current_scope(*,source_map:sqlite3.Connection,derived_root:Path,
                      prefix:str,index:OriginalSectionRAMIndex)->dict:
    """No full original/PDF hash until candidate selection, but every input
    *derived* text revision is already exact SHA-checked before indexing.
    """
    validate_literal_source_directory_prefix(prefix)
    if not isinstance(index,OriginalSectionRAMIndex):
        raise ValueError('invalid bounded PDF original source index')
    rows=source_map.execute(
        "SELECT source_path,source_sha256,derived_sha256,derived_path,original_title,original_bytes "
        "FROM originals WHERE status='complete' AND original_kind='pdf' "
        "AND substr(source_path,1,?)=? ORDER BY source_path",
        (len(prefix),prefix)).fetchall()
    if not 1<=len(rows)<=MAX_DOCUMENTS:
        raise ValueError('invalid, missing or oversized curated PDF original subfolder')
    for path,osha,dsha,derived_path,title,original_bytes in rows:
        if type(original_bytes) is not int or not 1<=original_bytes<=_MAX_SOURCE_ORIGINAL_BYTES:
            raise ValueError('original source PDF exceeds whole-source attestation bound')
        file=safe_file(derived_root,derived_path)
        if file.stat().st_size>MAX_DERIVED_BYTES_PER_SOURCE:
            raise ValueError('derived original exceeds bounded full-PDF source text scope')
        before=file.stat(follow_symlinks=False)
        blob=file.read_bytes()
        after=file.stat(follow_symlinks=False)
        if (before.st_ino!=after.st_ino or before.st_dev!=after.st_dev
                or before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns
                or before.st_ctime_ns!=after.st_ctime_ns):
            raise ValueError('original source extraction changed during full-section index')
        index.add_source(path=path,original_sha=osha,derived_sha=dsha,derived_path=derived_path,
                         title=title or '(untitled original)',derived_blob=blob)
    index.commit()
    return {'curated_original_subfolder_literal_prefix':prefix,
            'source_map_eligible_pdf_originals':len(rows),
            'complete_derived_sha_checked_source_documents':index.derived_sha_verified,
            'all_usable_original_source_units_windowed_not_just_first_unit':True,
            'original_pdf_source_units_explicitly_omitted':len(index.unreadable_source_units_explicitly_omitted),
            'unreadable_extracted_pdf_source_units_omitted':sum(
                z['reason']=='unreadable_NUL_interleaved_original_PDF_text_layer'
                for z in index.unreadable_source_units_explicitly_omitted),
            'empty_or_short_pdf_source_units_omitted':sum(
                z['reason']=='empty_or_too_short_extracted_original_PDF_source_unit'
                for z in index.unreadable_source_units_explicitly_omitted),
            'source_unit_omission_diagnostics':index.unreadable_source_units_explicitly_omitted[:75],
            'source_unit_omission_diagnostics_truncated':len(index.unreadable_source_units_explicitly_omitted)>75,
            'completely_original_page_covered_not_claimed':True,
            'original_derived_source_bytes_consumed':index.derivative_bytes_read,
            'original_source_windows_indexed':index.fullsection_windows,
            'all_scoped_eligible_pdf_sources_admitted':len(rows)==index.derived_sha_verified,
            'all_publisher_original_bytes_rechecked_during_index_build':False,
            'page_labels_are_extracted_metadata_not_verified_physical_pdf_pages':True,
            'no_persistent_database_or_model_created':True}


def verify_current_source_window(*,window:FullSection,query:str,
                                 original_root:Path,derived_root:Path,
                                 source_map:sqlite3.Connection,native_db:sqlite3.Connection,
                                 completed_receipts:dict[str,str],phrase:str|None=None)->dict:
    """Rehash full approved original and derivative, then exact later-page text.

    The source-map and FileDB are opened by the caller in read-only mode.
    An extracted page label is *not* a publisher PDF physical page proof.
    """
    from .source_evidence import _best_excerpt
    from .publisher_firstunit_discovery import compiled_required_original_source_phrase
    if not isinstance(window,FullSection) or not 3<=len(query)<=600:
        raise ValueError('untrusted original PDF window proof request')
    if completed_receipts.get(window.original_path)!=window.original_sha256:
        raise ValueError('no matching approved completed publisher original acquisition')
    rows=source_map.execute(
        'SELECT source_sha256,derived_sha256,derived_path,original_title,status '
        'FROM originals WHERE source_path=?',(window.original_path,)).fetchall()
    if (len(rows)!=1 or rows[0][4]!='complete'
            or rows[0][:4]!=(window.original_sha256,window.derived_sha256,
                            window.derived_path,window.title)):
        raise ValueError('publisher source-map revision changed after complete-source scan')
    original=safe_file(original_root,window.original_path)
    derived=safe_file(derived_root,window.derived_path)
    if original.stat().st_size>_MAX_SOURCE_ORIGINAL_BYTES or derived.stat().st_size>MAX_DERIVED_BYTES_PER_SOURCE:
        raise ValueError('selected original exceeded approved bounded full-SHA proof size')
    from .source_review_audit import sha_file
    if sha_file(original)!=window.original_sha256 or sha_file(derived)!=window.derived_sha256:
        raise ValueError('publisher original/derived bytes differ from scanned current approved revision')
    native=native_db.execute(
        "SELECT id FROM filesystem_entries WHERE relative_path_display=? AND entry_type='file'",
        (window.original_path,)).fetchall()
    if len(native)!=1 or type(native[0][0]) is not int or native[0][0]<=0:
        raise ValueError('native FileDB does not contain one unique original publisher document identity')
    raw=derived.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=window.derived_sha256:
        raise ValueError('original source derivative replaced after full SHA check')
    text=raw.decode('utf8','strict')
    if (type(window.source_start) is not int or type(window.source_end) is not int
            or not 0<=window.source_start<window.source_end<=len(text)
            or text[window.source_start:window.source_end]!=window.original_body):
        raise ValueError('current original-derived source section no longer has exact text offsets')
    # A complete original PDF can contain thirty-eight physical pages while
    # an older cached extracted text labels pages as high as fifty-two.
    # Existing KMD physical page-count verification is intentionally
    # independent of semantic relevance and of real per-page proof. Surface
    # its negative evidence for every selected original, never silently
    # reinterpret a derived page label as a publisher physical page.
    from .publisher_firstunit_discovery import publisher_pdf_page_map_bounds
    original_page_bounds=publisher_pdf_page_map_bounds(original,text)
    if not isinstance(original_page_bounds,dict) or not isinstance(original_page_bounds.get('status'),str):
        raise ValueError('invalid original PDF physical page count diagnostic')
    if phrase is None:
        start,end,_overlap=_best_excerpt(window.original_body,query_terms(query),length=520)
    else:
        pattern=compiled_required_original_source_phrase(phrase)
        matches=list(pattern.finditer(window.original_body))
        if not matches:
            raise ValueError('selected current publisher source section lacks required exact phrase')
        mark=matches[0]
        start=max(0,mark.start()-210)
        end=min(len(window.original_body),max(mark.end()+210,start+350))
        if end-start>520:
            raise ValueError('requested literal phrase context exceeds bounded source excerpt')
    excerpt=window.original_body[start:end]
    abs_start=window.source_start+start
    abs_end=window.source_start+end
    if (not excerpt.strip() or len(excerpt)>520 or text[abs_start:abs_end]!=excerpt):
        raise ValueError('extracted later-page evidence leaves original source section')
    return {
        'original_path':window.original_path,
        'original_source_title_metadata_not_literal_publisher_quote':window.title,
        'publisher_original_sha256':window.original_sha256,
        'current_derivative_sha256':window.derived_sha256,
        'current_native_original_filedb_entry_id':native[0][0],
        'extracted_source_unit_label_not_physically_pdf_verified':window.label,
        'current_original_section_window_start_char':window.source_start,
        'current_original_section_window_end_char':window.source_end,
        'original_section_window_sha256':hashlib.sha256(window.original_body.encode()).hexdigest(),
        'extracted_source_excerpt':excerpt,
        'extracted_source_excerpt_start_char':abs_start,
        'extracted_source_excerpt_end_char':abs_end,
        'extracted_source_excerpt_sha256':hashlib.sha256(excerpt.encode()).hexdigest(),
        'extracted_source_excerpt_not_literal_publisher_pdf_bytes':True,
        'complete_original_and_derived_bytes_full_sha_reverified':True,
        'current_unique_native_filedb_identity_verified':True,
        'original_pdf_physical_page_verified':False,
        'physical_publisher_page_count_bounds_review':original_page_bounds,
        'derived_page_labels_exceed_original_physical_page_count':bool(
            original_page_bounds.get('derived_pdf_page_map_suspect')),
        'even_consistent_page_counts_do_not_verify_specific_physical_page':True,
        'factual_answer_verified':False,
        'entire_source_relevant_or_complete':False,
    }
