#!/usr/bin/env python3
"""One approved publisher original physical PDF page, provisional glyph rescue.

Unlike the 327-PDF private stage-only reader, no existing staged derivative
is required. Exact original/full derived/native source identity is checked
before and after one bounded original PDF Ghostscript + Poppler read.
No OCR, source download, native FileDB/vector write, model, or live cutover.

Confirmed is a SAME-PHYSICAL-PDF-PAGE SHORT QUOTE status only. An uncorroborated
machine snippet is a candidate and may be wrong about words, layout or fact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from file_system_catalog.current_original_glyph_page import candidate_current_original_physical_glyph_page
from file_system_catalog.pdf_visual_link import safe_private_result_path

ORIGINAL=Path('/data/var/Knowledge Library')
DER=Path('/data/var/knowmoredirt/knowledge-library/semantic/text')
SM=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3')
NATIVE=Path('/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3')
CATALOG=Path('/data/infra/etc/knowledge-library-sources.json')
LEDGER=ORIGINAL/'_management/status.json'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('query',help='Search terms for ONE chosen original publisher physical PDF page')
    ap.add_argument('--original-path',required=True,help='Exact already approved original PDF relative path')
    ap.add_argument('--physical-page',required=True,type=int,help='ONE physical publisher PDF page number, not a cached derived source unit')
    ap.add_argument('--output',required=True,type=Path,help='NEW owner-private proof within the existing KMD search-audit tree')
    ap.add_argument('--original-root',type=Path,default=ORIGINAL)
    ap.add_argument('--derived-root',type=Path,default=DER)
    ap.add_argument('--source-map',type=Path,default=SM)
    ap.add_argument('--native-db',type=Path,default=NATIVE)
    ap.add_argument('--approved-catalog',type=Path,default=CATALOG)
    ap.add_argument('--completed-ledger',type=Path,default=LEDGER)
    args=ap.parse_args()
    output=safe_private_result_path(args.output,source_root=args.original_root,
                                    derived_root=args.derived_root,native_filedb=args.native_db,
                                    source_map_db=args.source_map)
    # The original PDF may be commercially licensed or confidential to the
    # owner's offline library. Restrict every new report to the source
    # FileDB's dedicated private audit subtree, even when a caller supplies
    # an otherwise writable destination outside the protected native DB.
    owner_private=(args.native_db.parent/'search-audit')
    if (not owner_private.is_dir() or owner_private.is_symlink()
            or not output.parent.is_relative_to(owner_private.resolve(strict=True))):
        raise ValueError('original publisher private page search must stay inside the native FileDB owner-private search-audit directory')
    candidate=candidate_current_original_physical_glyph_page(
        original_path=args.original_path,page=args.physical_page,query=args.query,
        original_root=args.original_root,derived_root=args.derived_root,
        source_map=args.source_map,native_db=args.native_db,
        catalog=args.approved_catalog,completed_ledger=args.completed_ledger)
    payload=(json.dumps(candidate,indent=2,ensure_ascii=False)+'\n').encode()
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o600)
    with os.fdopen(fd,'wb') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    hits=candidate['ranked_same_page_source_candidates']['accepted_provisional_ranked_pages']
    confirmed=next((x for x in hits if x['independent_poppler_same_physical_page_passage_corroborated']),None)
    short=next((x for x in hits if x['corroborated_short_fragment_only'] is not None),None)
    print('EXACT_ORIGINAL',args.original_path,'PHYSICAL_PAGE',args.physical_page)
    print('CURRENT_ORIGINAL_AND_NATIVE_REVISION_RECHECKED',True)
    print('ONE_EXACT_SAME_PAGE_PUBLISHER_EXCERPT_CONFIRMED',confirmed is not None)
    print('SHORT_ORIGINAL_SAME_PAGE_FRAGMENT_CONFIRMED',short is not None)
    print('UNCORROBORATED_MACHINE_TEXT_REMAINS_PROVISIONAL',confirmed is None)
    print('PRIVATE_SOURCE_RESULT',output,'SHA256',hashlib.sha256(payload).hexdigest())


if __name__=='__main__':
    main()
