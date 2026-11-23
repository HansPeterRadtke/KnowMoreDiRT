"""Bounded same-file physical PDF page context; no semantic truth inference.

A source excerpt's extracted page label is NOT sufficient evidence of its
physical page. The requested anchor page is selected explicitly, confirmed
by the PDF's real page count, and optionally matched against an exact source
phrase in a Poppler extraction of that original file. Neighbor pages remain
separate independently SHA-tracked page bodies; no fabricated joined quote.
The source PDF SHA is checked using the same open original file descriptor
used by the bounded Poppler worker, with file-replacement detection.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import re
import stat
import time

from .publisher_pdf_page import _poppler

PAGE_COUNT = re.compile(rb"(?m)^Pages:[ \t]+([1-9][0-9]{0,5})[ \t]*$")
_MAX_CHARACTERS=18000
_MAX_SECONDS=23


class PhysicalPdfPageContextUnavailable(ValueError):
    """Text for a physical page cannot be supplied; not a source-revision mismatch."""


def verified_page_neighbors(pdf:Path,expected_sha256:str,*,
        requested_page:int,neighbor_radius:int=1,
        required_center_excerpt:str|None=None,
        max_page_characters:int=_MAX_CHARACTERS)->dict:
    """Extract exact original PDF page text with publisher-file revision proof.

    The whole Poppler text page may be imperfect for image-only, ligature or
    unusual font sources. Output is authoritative *physical page location*
    and extracted text, NOT visual/semantic claim truth or permission to copy.
    Any required center excerpt must be contained on that original page.
    """
    if (not isinstance(pdf,Path) or pdf.suffix.lower()!='.pdf'
            or not isinstance(expected_sha256,str) or not re.fullmatch('[0-9a-f]{64}',expected_sha256)
            or type(requested_page) is not int or not 1<=requested_page<=999999
            or type(neighbor_radius) is not int or neighbor_radius not in (0,1)
            or type(max_page_characters) is not int or not 1000<=max_page_characters<=_MAX_CHARACTERS):
        raise ValueError('invalid bounded original PDF physical page request')
    if required_center_excerpt is not None and (not isinstance(required_center_excerpt,str)
            or not 3<=len(required_center_excerpt)<=600
            or not required_center_excerpt.strip()):
        raise ValueError('invalid required publisher PDF page anchor phrase')
    flags=os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW
    with os.fdopen(os.open(pdf,flags),'rb') as source:
        before=os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('publisher PDF is not a regular original file')
        if hashlib.file_digest(source,'sha256').hexdigest()!=expected_sha256:
            raise ValueError('publisher original SHA differs from completed acquisition')
        fd=source.fileno();name=f'/proc/self/fd/{fd}'
        deadline=time.monotonic()+_MAX_SECONDS
        raw_info=_poppler(['/usr/bin/pdfinfo',name],fd,deadline)
        counts=PAGE_COUNT.findall(raw_info or b'')
        if len(counts)!=1:raise ValueError('original PDF physical page count could not be proven')
        actual_page_count=int(counts[0])
        if requested_page>actual_page_count:
            raise ValueError('requested PDF page outside actual original physical page count')
        selected=range(max(1,requested_page-neighbor_radius),
                       min(actual_page_count,requested_page+neighbor_radius)+1)
        pages=[]
        for page in selected:
            raw=_poppler(['/usr/bin/pdftotext','-f',str(page),'-l',str(page),
                          '-layout','-enc','UTF-8',name,'-'],fd,deadline)
            unavailable=None
            if raw is None:
                unavailable='physical publisher page text extraction unavailable'
                body=''
            else:
                try:body=raw.decode('utf-8',errors='strict')
                except UnicodeDecodeError:
                    unavailable='original PDF page is not valid extracted UTF-8'
                    body=''
            if unavailable is None and (not body.strip() or body.count('\x00')>=12):
                unavailable='original PDF physical page lacks usable text (OCR/visual inspection separately required)'
            if unavailable is not None:
                if page==requested_page:
                    raise PhysicalPdfPageContextUnavailable(unavailable)
                # The named physical page exists in this original, but there
                # is NO valid textual context. Do not invent OCR, copy NUL
                # glyph codes, or silently omit this neighboring page.
                pages.append({'physical_page':page,
                              'status':'adjacent_original_pdf_page_text_unavailable',
                              'reason':unavailable,
                              'physical_page_poppler_sha256':hashlib.sha256(raw).hexdigest() if raw is not None else None,
                              'extracted_page_text':None,
                              'requires_original_pdf_visual_or_authorized_ocr_review':True,
                              'factual_claim_truth_verified':False})
                continue
            # Never silently crop the beginning or end where a condition or
            # negation may govern the retrieved snippet.
            # Poppler -layout can insert extraordinary amounts of horizontal
            # padding while the actual scholarly content is quite short. A
            # real source page with 21K layout characters and 5K words must
            # not become uncitable just because its columns are aligned.
            # Preserve EVERY non-whitespace token, original physical-page
            # SHA, and ALL line boundaries. Compact only duplicate horizontal
            # spaces/tabs, declaring the display projection explicitly.
            display=body
            layout_policy='verbatim_poppler_page_text'
            if len(body)>max_page_characters:
                display=re.sub(r'[ \t]{2,}', ' ', body)
                if ''.join(display.split())!=''.join(body.split()):
                    raise ValueError('source projection changed non-whitespace original PDF page text')
                if len(display)>max_page_characters:
                    if page==requested_page:
                        raise PhysicalPdfPageContextUnavailable('physical source page exceeds bounded context: no silent truncation')
                    pages.append({'physical_page':page,
                                  'status':'adjacent_original_pdf_page_text_unavailable',
                                  'reason':'adjacent original PDF physical page exceeds bounded context even after lossless horizontal spacing compaction',
                                  'physical_page_poppler_sha256':hashlib.sha256(raw).hexdigest(),
                                  'extracted_page_text':None,
                                  'requires_original_pdf_visual_or_authorized_ocr_review':True,
                                  'factual_claim_truth_verified':False})
                    continue
                layout_policy='horizontal_padding_compacted_all_non_whitespace_and_line_boundaries_preserved'
            pages.append({'physical_page':page,'physical_page_poppler_sha256':hashlib.sha256(raw).hexdigest(),
                          'status':'verified_original_page_text',
                          'extracted_text_sha256':hashlib.sha256(display.encode()).hexdigest(),
                          'raw_poppler_page_text_characters':len(body),
                          'text_characters':len(display),'extracted_page_text':display,
                          'display_layout_policy':layout_policy,
                          'text_projection_dropped_non_whitespace':False,
                          'factual_claim_truth_verified':False,
                          'provenance':'SHA-verified original PDF physical page via Poppler. When horizontal padding was compacted, line boundaries and every non-whitespace character were preserved but column alignment was lost. Not publisher visual glyph proof.'})
        normalized=' '.join(next(row['extracted_page_text'] for row in pages if row['physical_page']==requested_page).split())
        if required_center_excerpt is not None and ' '.join(required_center_excerpt.split()) not in normalized:
            raise ValueError('required source excerpt absent from the selected physical original PDF page')
        after=os.stat(pdf,follow_symlinks=False)
        if (not os.path.samestat(before,after) or before.st_size!=after.st_size
                or before.st_mtime_ns!=after.st_mtime_ns or before.st_ctime_ns!=after.st_ctime_ns):
            raise ValueError('publisher PDF changed during page extraction')
    return {'source_kind':'original_physical_pdf_pages_with_page_boundary_provenance',
            'original_pdf_sha256':expected_sha256,'actual_original_pdf_page_count':actual_page_count,
            'requested_center_physical_page':requested_page,'neighbor_radius':neighbor_radius,
            'required_center_excerpt_verified':required_center_excerpt is not None,
            'original_pdf_revision_verified_before_and_after':True,
            'separate_original_pages_not_joined_into_one_quotation':True,
            'unavailable_neighbor_text_pages':sum(p.get('status')=='adjacent_original_pdf_page_text_unavailable' for p in pages),
            'all_returned_neighbors_have_usable_extracted_text':all(p.get('status')=='verified_original_page_text' for p in pages),
            'source_original_publisher_bytes_are_untouched':True,
            'context_cannot_by_itself_prove_a_factual_claim':True,
            'pages':pages,
            'warning':('These are separate Poppler-extracted text pages from the SHA-verified original PDF. '
                       'Oversized layout-only spacing may be compacted horizontally without dropping '
                       'non-whitespace text or line boundaries; original visual column alignment then requires direct PDF inspection. '
                       'A source may quote, negate, hypothesize, retract or supersede claims on another page. '
                       'A separately listed neighboring physical page may have no usable text; its contents remain unknown unless visually inspected. '
                       'Read their governing context, publication status and visuals; do not assert a real-world fact solely from matched text.')}
