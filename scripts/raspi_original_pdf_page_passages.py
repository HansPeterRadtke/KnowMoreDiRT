#!/usr/bin/env python3
"""Read-only lexical passage candidates within an explicit approved PDF page."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"src"));sys.path.insert(0,str(HERE))
from file_system_catalog.reading_order_page_passages import reading_order_page_passages
from raspi_original_pdf_page_neighbors import original_pdf_context
from raspi_attest_source_claim_dataset import current_native_ids, final_sources

_SHA=re.compile(r"[0-9a-f]{64}\Z")

def current_page_passages(original_path:str,page:int,query:str,*,top:int=3,max_characters:int=900,
                          expected_identity:tuple[str,str,str]|None=None,
                          include_inflection_matches:bool=False)->dict:
    if expected_identity is not None and (not isinstance(expected_identity,tuple)
            or len(expected_identity)!=3 or not all(isinstance(v,str) and _SHA.fullmatch(v)
                                                      for v in expected_identity)):
        raise ValueError("supply all three exact prior original/derived/reading-order page digests")
    before=current_native_ids([original_path])
    context=original_pdf_context(original_path,page,neighbor_radius=0,include_center_reading_order=True)
    result=reading_order_page_passages(context,query,top=top,max_characters=max_characters,
                                       include_inflection_matches=include_inflection_matches)
    if expected_identity is not None and (
            result["original_sha256"],result["derived_sha256"],result["reading_order_page_sha256"])!=expected_identity:
        raise ValueError("requested exact prior page/source revision changed")
    final_sources({original_path:context["source"]})
    after=current_native_ids([original_path])
    if before!=after:
        raise ValueError("native source identity changed during page-local passage search")
    result.update(native_original_id=before[original_path],
                  selected_source_revision_and_native_id_rechecked=True,
                  cross_store_immutable_generation_pinned=False,
                  safe_to_cutover_live=False)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query");ap.add_argument("--original-path",required=True)
    ap.add_argument("--page",type=int,required=True)
    ap.add_argument("--top",type=int,default=3);ap.add_argument("--max-characters",type=int,default=900)
    ap.add_argument("--expected-original-sha256");ap.add_argument("--expected-derived-sha256")
    ap.add_argument("--expected-reading-order-page-sha256")
    ap.add_argument("--include-inflection-matches",action="store_true",help="Opt-in lower-weight Porter stemming for ordinary English words; exact text unchanged")
    a=ap.parse_args()
    identity=(a.expected_original_sha256,a.expected_derived_sha256,a.expected_reading_order_page_sha256)
    print(json.dumps(current_page_passages(a.original_path,a.page,a.query,top=a.top,
                         max_characters=a.max_characters,
                         expected_identity=identity if any(x is not None for x in identity) else None,
                         include_inflection_matches=a.include_inflection_matches),
                     ensure_ascii=False,indent=2))

if __name__=="__main__":main()
