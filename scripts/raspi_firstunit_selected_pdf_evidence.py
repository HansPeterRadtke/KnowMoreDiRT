#!/usr/bin/env python3
"""Expand exact private independent lexical results into verified PDF evidence."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/"src"),str(HERE)]
from file_system_catalog.firstunit_pdf_evidence import (
    select_lexical_pdf_candidates,require_lexical_current_identity,
)
from file_system_catalog.reading_order_document_pages import BoundedPdfDiscoveryUnavailable
from file_system_catalog.source_review_audit import safe_file
from raspi_original_pdf_document_passages import current_document_identity,current_document_passages
from raspi_attest_source_claim_dataset import final_sources,current_native_ids

AUDIT=Path("/data/var/knowmoredirt/knowledge-library/search-audit")
MAX_RECEIPT=16*1024*1024


def verified_lexical_pdf_evidence(receipt,*,originals_per_query=1,top_pages=5,
                                   snippet_characters=900):
    if (type(top_pages) is not int or not 1<=top_pages<=5
            or type(snippet_characters) is not int or not 160<=snippet_characters<=2000):
        raise ValueError("bounded PDF evidence window required")
    selection=select_lexical_pdf_candidates(receipt,originals_per_query=originals_per_query)
    revisions={};native_ids={};rows=[]
    for q in selection:
        results=[]
        for candidate in q["selected_originals"]:
            path=candidate["original_path"]
            source,native_id=current_document_identity(path)
            require_lexical_current_identity(candidate,source,native_id)
            if path in revisions and (source!=revisions[path] or native_id!=native_ids[path]):
                raise ValueError("source changed across lexical queries")
            revisions[path]=source;native_ids[path]=native_id
            try:
                evidence=current_document_passages(path,q["query"],top_pages=top_pages,
                                                    snippet_characters=snippet_characters)
            except BoundedPdfDiscoveryUnavailable as e:
                results.append({**candidate,"status":"complete_document_discovery_unavailable",
                                "reason":str(e),"verified_page_evidence":None})
                continue
            if ((evidence.get("original_sha256"),evidence.get("derived_sha256"),
                 evidence.get("native_original_id"),evidence.get("original_path"))
                    !=(candidate["original_sha256"],candidate["derived_sha256"],
                       candidate["native_original_id"],path)
                    or evidence.get("selected_source_revision_and_native_id_rechecked") is not True):
                raise ValueError("expanded document differs from exact lexical result")
            results.append({**candidate,"status":"verified_document_discovery",
                            "verified_page_evidence":evidence})
        rows.append({"id":q["id"],"query":q["query"],"query_sha256":q["query_sha256"],
                     "available_distinct_pdf_candidates":q["available_distinct_pdf_candidates"],
                     "selected_originals":results})
    if revisions:
        final_sources(revisions)
        if current_native_ids(sorted(revisions))!=native_ids:
            raise ValueError("native identity changed before final lexical evidence output")
    return {"schema":"kmd.readonly_lexical_selected_verified_pdf_evidence.v1",
            "checked_utc":datetime.now(timezone.utc).isoformat(),
            "query_count":len(rows),"originals_per_query":originals_per_query,
            "top_pages_per_original":top_pages,"snippet_characters":snippet_characters,
            "unique_selected_originals_rechecked":len(revisions),
            "independent_lexical_ranks_preserved":True,"native_document_ranking_unchanged":True,
            "no_factual_answer_or_visual_reading_order_verified":True,
            "cross_store_immutable_generation_pinned":False,"safe_to_cutover_live":False,
            "queries":rows}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lexical-receipt",required=True,type=Path)
    ap.add_argument("--expected-sha256",required=True)
    ap.add_argument("--originals-per-query",type=int,default=1)
    ap.add_argument("--top-pages",type=int,default=5)
    ap.add_argument("--snippet-characters",type=int,default=900)
    ap.add_argument("--output",required=True,type=Path)
    a=ap.parse_args()
    root=AUDIT.resolve(strict=True)
    if not a.lexical_receipt.is_absolute() or not a.lexical_receipt.is_relative_to(root):
        raise ValueError("input must be in private search-audit directory")
    path=safe_file(root,str(a.lexical_receipt.relative_to(root)))
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),"rb") as f:
        raw=f.read(MAX_RECEIPT+1)
    digest=hashlib.sha256(raw).hexdigest()
    if len(raw)>MAX_RECEIPT or digest!=a.expected_sha256:
        raise ValueError("lexical input changed or exceeds receipt bound")
    if (not a.output.is_absolute() or not a.output.is_relative_to(root)
            or a.output.exists() or a.output.is_symlink()
            or not a.output.parent.is_dir() or a.output.parent.is_symlink()
            or a.output.parent.resolve()!=a.output.parent
            or not a.output.resolve().is_relative_to(root)):
        raise ValueError("new private evidence output required")
    result=verified_lexical_pdf_evidence(json.loads(raw),originals_per_query=a.originals_per_query,
                          top_pages=a.top_pages,snippet_characters=a.snippet_characters)
    result["input_lexical_receipt_sha256"]=digest
    body=(json.dumps(result,ensure_ascii=False,indent=2)+"\n").encode()
    fd=os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o600)
    with os.fdopen(fd,"wb") as f:
        f.write(body);f.flush();os.fsync(f.fileno())
    print(json.dumps({"query_count":result["query_count"],
         "unique_selected_originals_rechecked":result["unique_selected_originals_rechecked"],
         "output_sha256":hashlib.sha256(body).hexdigest()},indent=2))

if __name__=="__main__":main()
