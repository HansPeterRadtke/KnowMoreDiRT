#!/usr/bin/env python3
"""Read-only source-byte verification of post-hoc retrieval review claims."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from file_system_catalog.source_review_audit import validate_reviews


def _input(path: Path, expected: str | None, *, limit_bytes: int = 5*1024*1024) -> tuple[dict,str]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit_bytes:
        raise ValueError("missing, symlinked, or oversized private review input")
    content = path.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError("frozen input SHA mismatch")
    return json.loads(content),sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("shadow","review","source-map","original-root","derived-root","completed-ledger","output"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--shadow-sha256",required=True)
    p.add_argument("--review-sha256",required=True)
    p.add_argument("--verify-physical-pdf-pages",action="store_true",
                   help="Independently rehash cited PDFs and check quoted excerpts on original physical pages")
    a=p.parse_args()
    if a.output.exists() or a.output.is_symlink() or not a.output.parent.is_dir() or a.output.parent.is_symlink():
        raise ValueError("private report output already exists or directory unavailable")
    shadow,digest = _input(a.shadow,a.shadow_sha256)
    review,review_digest = _input(a.review,a.review_sha256)
    ledger,_ = _input(a.completed_ledger,None,limit_bytes=32*1024*1024)
    if not isinstance(ledger,dict) or not isinstance(ledger.get("items"),list):
        raise ValueError("invalid completed acquisition ledger")
    receipts={}
    for row in ledger["items"]:
        if row.get("status") != "complete":
            continue
        path=row.get("path")
        if path in receipts:
            raise ValueError("duplicated completed original in acquisition ledger")
        receipts[path]=row.get("sha256")
    if not a.source_map.is_file() or a.source_map.is_symlink():
        raise ValueError("canonical original source-map missing or symlinked")
    with sqlite3.connect(a.source_map.resolve().as_uri()+"?mode=ro",uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        report=validate_reviews(shadow=shadow,review=review,source_map=con,
                                original_root=a.original_root,derived_root=a.derived_root,
                                receipts=receipts, verify_physical_pdf_pages=a.verify_physical_pdf_pages)
    report["frozen_shadow_sha256"]=digest
    report["manual_review_file_sha256"]=review_digest
    report["source_map_read_only"]=True
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2,ensure_ascii=False)
        f.write("\n")
    a.output.chmod(0o600)
    print(json.dumps({"private_report":str(a.output),"reviewed":report["reviewed"],
                      "grade_counts":report["grade_counts"],
                      "source_byte_provenance_verified":True},ensure_ascii=False))


if __name__=="__main__":
    main()
