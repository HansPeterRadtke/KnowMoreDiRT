#!/usr/bin/env python3
"""Read-only complete physical-page discovery inside one approved short PDF."""
from __future__ import annotations
import argparse
import json
from contextlib import closing
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/"src"),str(HERE)]
from file_system_catalog.reading_order_document_pages import (
    read_complete_original_pdf_pages,rank_document_page_candidates,
    verify_selected_document_pages,BoundedPdfDiscoveryUnavailable,
)
from file_system_catalog.source_triage_provenance import verify_selected_source
from file_system_catalog.source_review_audit import safe_file
from raspi_original_pdf_page_neighbors import original_pdf_context,ROOT,DERIVED,SOURCE_MAP,CATALOG,LEDGER,unique_rows
from raspi_attest_source_claim_dataset import readonly,current_native_ids,final_sources


def current_document_identity(path: str) -> tuple[dict,int]:
    """Verify the approved current original and native identity independently."""
    if not isinstance(path,str) or not path.lower().endswith(".pdf"):
        raise ValueError("one exact approved PDF path is required")
    originals=current_native_ids([path])
    catalog=unique_rows(json.loads(CATALOG.read_bytes())["sources"])
    ledger=unique_rows(json.loads(LEDGER.read_bytes())["items"])
    if path not in catalog or catalog[path].get("kind")!="pdf":
        raise ValueError("one completed approved PDF source is required")
    with closing(readonly(SOURCE_MAP)) as sm:
        revisions=sm.execute("SELECT source_sha256,derived_sha256,text_characters FROM originals WHERE source_path=? AND status='complete'",(path,)).fetchall()
        if len(revisions)!=1:
            raise ValueError("missing unique current original source revision")
        source=verify_selected_source(path,source_versions={path:revisions[0]},
             source_map=sm,approval_ledger=ledger,catalog=catalog,original_root=ROOT,derived_root=DERIVED)
    return source,originals[path]


def current_document_passages(path:str,query:str,*,top_pages:int=5,snippet_characters:int=900)->dict:
    if (not isinstance(path,str) or not path.lower().endswith(".pdf")
            or not isinstance(query,str) or not 1<=len(query.strip())<=600
            or type(top_pages) is not int or not 1<=top_pages<=5
            or type(snippet_characters) is not int or not 160<=snippet_characters<=2000):
        raise ValueError("one exact approved PDF path is required")
    source,native_id=current_document_identity(path)
    originals={path:native_id}
    # Both the complete-source extraction and independent selected-page
    # verification use the exact approved publisher bytes and revision.
    scan=read_complete_original_pdf_pages(safe_file(ROOT,path),source["original_sha256"])
    ranked=rank_document_page_candidates(scan,query,top_pages=top_pages,snippet_characters=snippet_characters)
    result=verify_selected_document_pages(ranked,source=source,
             read_context=lambda p,n:original_pdf_context(p,n,neighbor_radius=0,include_center_reading_order=True))
    final_sources({path:source})
    if current_native_ids([path])!=originals:
        raise ValueError("native original changed during independent document page discovery")
    return {"schema":"kmd.readonly_independent_original_pdf_document_passages.v1",
            "original_path":path,"title":source["title"],"source_url":source["source_url"],
            "original_sha256":source["original_sha256"],"derived_sha256":source["current_derived_sha256"],
            "native_original_id":originals[path],
            "extraction_method":scan["extraction_method"],
            "complete_reading_order_extraction_sha256":scan["complete_extraction_sha256"],
            "complete_source_reading_order_scan_with_bounded_unavailable_pages":True,
            "selected_source_revision_and_native_id_rechecked":True,
            "cross_store_immutable_generation_pinned":False,"safe_to_cutover_live":False,
            **result}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query");ap.add_argument("--original-path",required=True)
    ap.add_argument("--top-pages",type=int,default=5);ap.add_argument("--snippet-characters",type=int,default=900)
    a=ap.parse_args()
    try:r=current_document_passages(a.original_path,a.query,top_pages=a.top_pages,snippet_characters=a.snippet_characters)
    except BoundedPdfDiscoveryUnavailable as e:
        # Explicitly unavailable, not an empty successfully searched source.
        print(json.dumps({"status":"complete_document_discovery_unavailable","reason":str(e),
              "no_full_document_retrieval_claim":True}))
        raise SystemExit(2)
    print(json.dumps(r,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
