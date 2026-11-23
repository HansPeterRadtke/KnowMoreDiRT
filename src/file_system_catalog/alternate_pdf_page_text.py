"""Bounded original-PDF Ghostscript text candidates, never a verified source quotation.

This is a separate opt-in read-only candidate discovery path for damaged
physical pages, not an OCR replacement or admission to permanent native
FileDB. In particular the original publisher PDF *page number* may be
attested, but Ghostscript output is NOT proof that all glyphs, visual
columns, figures, negation and source discourse were faithfully extracted.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time

from .publisher_pdf_page import _poppler

PAGE_COUNT=re.compile(rb'(?m)^Pages:[ \t]+([1-9][0-9]{0,5})[ \t]*$')
WORD=re.compile(r"(?u)\b[^\W\d_][\w'-]{2,}\b")
MAX_PAGES_PER_QUERY=32
MAX_OUTPUT_BYTES=4*1024*1024
MAX_DISPLAY_CHARS=18000
MAX_DEADLINE_SECONDS=120


def original_pdf_gs_page(fd:int,page:int,deadline:float) -> bytes|None:
    """Restrict Ghostscript resources and text output; use the same open PDF fd."""
    remaining=deadline-time.monotonic()
    if remaining<=0:return None
    args=['/usr/bin/prlimit',f'--fsize={MAX_OUTPUT_BYTES}','--as=1073741824','--',
          '/usr/bin/gs','-q','-dSAFER','-dNOPAUSE','-dBATCH','-sDEVICE=txtwrite',
          f'-dFirstPage={page}',f'-dLastPage={page}','-sOutputFile=-',
          f'/proc/self/fd/{fd}']
    try:
        with tempfile.TemporaryFile() as output:
            proc=subprocess.run(args,pass_fds=(fd,),stdout=output,stderr=subprocess.DEVNULL,
                timeout=min(12,max(.2,remaining)),check=False,
                env={**os.environ,'LC_ALL':'C'})
            if proc.returncode or output.tell()>=MAX_OUTPUT_BYTES:
                return None
            output.seek(0)
            return output.read()
    except (OSError,subprocess.TimeoutExpired):return None


def quality_of_alternate_page(raw:bytes|None, *, max_chars:int=MAX_DISPLAY_CHARS)->dict:
    """Accept a *provisional text candidate*, not publisher-visual correctness.

    Preserve all original non-whitespace characters and line boundaries.
    Fail closed if output is empty, oversized after lossless horizontal
    compaction, contains even one NUL/replacement glyph or fewer than thirty
    word tokens. Rejecting some useful pages is safer than silently asserting
    a corrupted scientific quote.
    """
    if type(max_chars) is not int or not 1000<=max_chars<=MAX_DISPLAY_CHARS:
        raise ValueError('invalid bounded alternate original-page text limit')
    if raw is None or not isinstance(raw,bytes) or not raw or len(raw)>=MAX_OUTPUT_BYTES:
        return {'status':'unavailable','reason':'alternate_original_page_text_unavailable_or_excessive'}
    try:text=raw.decode('utf8','strict')
    except UnicodeDecodeError:
        return {'status':'unavailable','reason':'alternate_original_page_text_invalid_UTF8'}
    if ('\x00' in text or '\ufffd' in text or not text.strip()):
        return {'status':'unavailable','reason':'alternate_original_page_text_has_unreadable_glyphs'}
    tokens=WORD.findall(text)
    if len(tokens)<30:
        return {'status':'unavailable','reason':'alternate_original_page_text_has_insufficient_words'}
    display=text;layout='raw_Ghostscript_txtwrite_text'
    if len(display)>max_chars:
        display=re.sub(r'[ \t]{2,}',' ',text)
        if ''.join(display.split())!=''.join(text.split()):
            raise ValueError('provisional source display changed original non-whitespace characters')
        if len(display)>max_chars:
            return {'status':'unavailable','reason':'alternate_original_page_text_still_oversized_without_truncation'}
        layout='horizontal_padding_compacted_line_boundaries_and_non_whitespace_preserved'
    return {'status':'source_bounded_provisional_machine_text',
            'text':display,
            'raw_machine_text_sha256':hashlib.sha256(raw).hexdigest(),
            'display_text_sha256':hashlib.sha256(display.encode('utf8')).hexdigest(),
            'raw_character_count':len(text),'display_character_count':len(display),
            'word_tokens':len(tokens),'unique_casefold_word_tokens':len(set(w.casefold() for w in tokens)),
            'layout_policy':layout,'publisher_visual_or_semantic_text_fidelity_verified':False,
            'source_page_content_not_an_authoritative_quote':True}


def extract_alternate_original_pdf_pages(path:Path,expected_sha256:str,
                *,pages:list[int],expected_physical_page_count:int,
                page_reader=original_pdf_gs_page,pdfinfo_reader=_poppler,
                max_pages:int=MAX_PAGES_PER_QUERY)->dict:
    """Full publisher PDF SHA, exact physical pages, safe per-page GS candidate.

    The full original PDF remains open on one read-only file descriptor for
    pdfinfo and all page-reader calls. Hash before and after and refuse
    symlinks, replacement, mutation, incomplete page identity or invalid
    output. This does NOT validate visual reading order or real-world claims.
    """
    if (not isinstance(path,Path) or path.suffix.lower()!='.pdf'
            or not isinstance(expected_sha256,str)
            or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None
            or type(expected_physical_page_count) is not int
            or not 1<=expected_physical_page_count<=999999
            or type(max_pages) is not int or not 1<=max_pages<=MAX_PAGES_PER_QUERY
            or not isinstance(pages,list) or not 1<=len(pages)<=max_pages
            or any(type(p) is not int or not 1<=p<=expected_physical_page_count for p in pages)
            or len(set(pages))!=len(pages) or pages!=sorted(pages)):
        raise ValueError('invalid source revision, exact physical PDF pages or bounded batch')
    flags=os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW
    with os.fdopen(os.open(path,flags),'rb') as publisher:
        before=os.fstat(publisher.fileno())
        if not stat.S_ISREG(before.st_mode) or hashlib.file_digest(publisher,'sha256').hexdigest()!=expected_sha256:
            raise ValueError('original publisher PDF does not match the approved SHA-256 revision')
        fd=publisher.fileno();pdf_name=f'/proc/self/fd/{fd}'
        deadline=time.monotonic()+MAX_DEADLINE_SECONDS
        raw_info=pdfinfo_reader(['/usr/bin/pdfinfo',pdf_name],fd,deadline)
        matches=PAGE_COUNT.findall(raw_info or b'')
        if len(matches)!=1 or int(matches[0])!=expected_physical_page_count:
            raise ValueError('publisher original physical PDF page count differs from staged source')
        extracted=[]
        for page in pages:
            result=quality_of_alternate_page(page_reader(fd,page,deadline))
            extracted.append({'physical_pdf_page':page,'status':result['status'],
                **{k:v for k,v in result.items() if k!='status'}})
        stat_after=path.stat(follow_symlinks=False)
        if (not os.path.samestat(before,stat_after) or before.st_size!=stat_after.st_size
                or before.st_mtime_ns!=stat_after.st_mtime_ns
                or before.st_ctime_ns!=stat_after.st_ctime_ns):
            raise ValueError('publisher original PDF changed during alternate extraction')
        publisher.seek(0)
        if hashlib.file_digest(publisher,'sha256').hexdigest()!=expected_sha256:
            raise ValueError('publisher original PDF bytes changed during alternate extraction')
    return {'schema':'kmd.readonly_original_pdf_bounded_machine_text_candidates.v1',
            'original_pdf_sha256':expected_sha256,
            'physical_pdf_pages':expected_physical_page_count,
            'original_pdf_sha_verified_before_and_after_same_file_description':True,
            'all_page_numbers_from_original_pdfinfo_not_fake_derivative_labels':True,
            'alternate_extractor':'local_Ghostscript_txtwrite_dSAFER_resource_bounded',
            'original_pdf_untouched':True,
            'machine_extracted_text_is_provisional_not_visually_verified':True,
            'not_a_native_semantic_index_or_factual_answer':True,
            'pages':extracted}
