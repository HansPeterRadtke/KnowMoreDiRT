#!/usr/bin/env python3
"""Opt-in read-only source-pinned lexical evidence -> true PDF neighbor pages.

Uses original KMD source-evidence (which verifies a real physical page match),
then rereads and SHA-attests the original PDF physical page and neighbors.
The existing default search, native FileDB and all source bytes are unchanged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from file_system_catalog.verified_pdf_context_handoff import (
    verified_physical_page_candidate,bind_original_pdf_context_to_source_evidence,validate_passage_selection,
)
from raspi_original_pdf_page_neighbors import original_pdf_context


def context_from_current_source_evidence(query:str,original_path:str,*,neighbor_radius:int=1,
                                         include_center_reading_order:bool=False,
                                         passage_rank:int|None=None,
                                         expected_identity:tuple[str,str,str,int,str]|None=None)->dict:
    validate_passage_selection(passage_rank,expected_identity)
    if (not isinstance(query,str) or not 3<=len(query.strip())<=500
            or not isinstance(original_path,str) or not 1<=len(original_path)<=1600
            or not original_path.lower().endswith('.pdf') or original_path.startswith('/')
            or '\\' in original_path or any(p in ('','.','..') for p in original_path.split('/'))
            or type(neighbor_radius) is not int or neighbor_radius not in (0,1)
            or type(include_center_reading_order) is not bool):
        raise ValueError('invalid bounded exact-original contextual source question')
    reader=ROOT/'scripts/raspi_hierarchical_library.py'
    if not reader.is_file() or reader.is_symlink():
        raise ValueError('native KMD source-evidence operator unavailable')
    proc=subprocess.run([
        sys.executable,str(reader),'source-evidence',query,
        '--original-path',original_path,'--top','3','--snippet-characters','500',
    ],capture_output=True,text=True,timeout=100,check=False)
    if proc.returncode or len(proc.stdout)>2_000_000:
        raise ValueError('native current-source citation lookup failed; no unverified output')
    try:source=json.loads(proc.stdout)
    except (ValueError,TypeError) as exc:raise ValueError('native source-evidence payload is not valid JSON') from exc
    anchor=verified_physical_page_candidate(source,original_path,passage_rank=passage_rank,
                                                expected_identity=expected_identity)
    exact=original_pdf_context(original_path,anchor['physical_pdf_page'],
                neighbor_radius=neighbor_radius,required_center_excerpt=anchor['source_excerpt'],
                include_center_reading_order=include_center_reading_order)
    return bind_original_pdf_context_to_source_evidence(anchor,exact)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('query')
    ap.add_argument('--original-path',required=True)
    ap.add_argument('--neighbor-radius',type=int,default=1,choices=(0,1))
    ap.add_argument('--include-center-reading-order',action='store_true',help='Separate SHA-tracked default Poppler text of the same center physical page; not a replacement for layout proof')
    ap.add_argument('--passage-rank',type=int,choices=(1,2,3),default=None,
                    help='Exact reviewed source-evidence rank; requires all four expected digests and physical page from a matching top-3, 500-character source-evidence query')
    ap.add_argument('--expected-original-sha256',default=None)
    ap.add_argument('--expected-derived-sha256',default=None)
    ap.add_argument('--expected-excerpt-sha256',default=None)
    ap.add_argument('--expected-physical-page',type=int,default=None)
    ap.add_argument('--expected-page-text-sha256',default=None)
    args=ap.parse_args()
    identity=(args.expected_original_sha256,args.expected_derived_sha256,args.expected_excerpt_sha256,
              args.expected_physical_page,args.expected_page_text_sha256)
    expected_identity=identity if any(v is not None for v in identity) else None
    print(json.dumps(context_from_current_source_evidence(args.query,args.original_path,
                         neighbor_radius=args.neighbor_radius,
                         include_center_reading_order=args.include_center_reading_order,
                         passage_rank=args.passage_rank,expected_identity=expected_identity),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
