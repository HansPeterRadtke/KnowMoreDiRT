#!/usr/bin/env python3
"""Read-only current publisher/source-map/native identity proof for frozen claims."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"src"));sys.path.insert(0,str(HERE))
from file_system_catalog.source_claim_dataset_provenance import attest_frozen_claim_sources
from file_system_catalog.source_triage_provenance import verify_selected_source,require_unchanged_final_source_map
from raspi_original_pdf_page_neighbors import (
    original_pdf_context, ROOT, DERIVED, SOURCE_MAP, CATALOG, LEDGER, unique_rows,
)

NATIVE=Path("/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3")
MAX_INPUT=2*1024*1024

def readonly(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError("canonical database missing or symlinked")
    c=sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True,timeout=20)
    c.execute("PRAGMA query_only=ON")
    return c

def current_native_ids(paths):
    with readonly(NATIVE) as db:
        out={}
        for path in paths:
            rows=db.execute("SELECT id FROM filesystem_entries WHERE entry_type='file' AND relative_path_display=?",(path,)).fetchall()
            if len(rows)!=1:
                raise ValueError("no unique native FileDB original for frozen claim source")
            out[path]=rows[0][0]
        return out

def final_sources(revisions):
    # Re-read current approval ledger and manifest, then rehash selected
    # original and derivative files. Use a fresh source-map read afterwards.
    catalog=unique_rows(json.loads(CATALOG.read_bytes())["sources"])
    ledger=unique_rows(json.loads(LEDGER.read_bytes())["items"])
    with readonly(SOURCE_MAP) as sm:
        for path,old in revisions.items():
            rows=sm.execute("SELECT source_sha256,derived_sha256,text_characters FROM originals WHERE source_path=? AND status='complete'",(path,)).fetchall()
            if len(rows)!=1:
                raise ValueError("missing current source revision in final claim check")
            again=verify_selected_source(path,source_versions={path:rows[0]},
                    source_map=sm,approval_ledger=ledger,catalog=catalog,
                    original_root=ROOT,derived_root=DERIVED)
            if again!=old:
                raise ValueError("selected original changed during frozen claim attestation")
    with readonly(SOURCE_MAP) as sm:
        require_unchanged_final_source_map(revisions,newest_source_map=sm)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",required=True,type=Path);ap.add_argument("--output",required=True,type=Path)
    a=ap.parse_args()
    if not a.dataset.is_file() or a.dataset.is_symlink() or a.dataset.stat().st_size>MAX_INPUT:
        raise ValueError("invalid private frozen claim dataset")
    if a.output.exists() or a.output.is_symlink() or not a.output.parent.is_dir() or a.output.parent.is_symlink():
        raise ValueError("new private attestation path required")
    # Never write to the original library, native DB directory's data files,
    # or a source-controlled directory. Only the existing private audit root.
    audit=NATIVE.parent/"search-audit"
    if not a.output.resolve().is_relative_to(audit.resolve()):
        raise ValueError("attestation output must remain in private search-audit directory")
    raw=a.dataset.read_bytes()
    if len(raw)>MAX_INPUT:raise ValueError("dataset exceeds bounded input")
    data=json.loads(raw)
    result=attest_frozen_claim_sources(data,
        read_current_pdf_context=lambda path,page:original_pdf_context(path,page,
                       neighbor_radius=0,include_center_reading_order=True),
        native_identities=current_native_ids,final_current_source_check=final_sources)
    result.update(schema="kmd.private_frozen_claim_current_source_attestation.v1",
                  checked_utc=datetime.now(timezone.utc).isoformat(),
                  dataset_sha256=hashlib.sha256(raw).hexdigest())
    fd=os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o600)
    with os.fdopen(fd,"w",encoding="utf8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:v for k,v in result.items() if k!="spans"},indent=2))

if __name__=="__main__":main()
