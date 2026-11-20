#!/usr/bin/env python3
"""Optional dual KMD source lookup: separate native summary and original results."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys


def _command(argv:list[str],*,seconds:int=120)->dict:
    result=subprocess.run(argv,text=True,capture_output=True,timeout=seconds,check=False)
    if result.returncode:
        raise ValueError("native KMD read-only lookup failed; no unverified partial output emitted")
    return json.loads(result.stdout)


def _receipts(path:Path)->dict[str,str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>32*1024**2:
        raise ValueError("completed original receipt absent or untrusted")
    raw=json.loads(path.read_text(encoding="utf-8"))
    rows=raw.get("items",[])
    if not isinstance(rows,list):
        raise ValueError("invalid completed original receipt")
    out={}
    for row in rows:
        if not isinstance(row,dict):continue
        path=row.get("path")
        digest=row.get("sha256")
        if row.get("status")=="complete" and isinstance(path,str) and isinstance(digest,str):
            if path in out and out[path]!=digest:raise ValueError("ambiguous publisher download receipt")
            out[path]=digest
    return out


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("query")
    p.add_argument("--top",type=int,default=8)
    p.add_argument("--summary-cli",type=Path,default=Path("/data/src/github/KnowMoreDiRT/scripts/raspi_hierarchical_library.py"))
    p.add_argument("--original-passage-cli",type=Path,default=Path("/data/bin/knowledge-library-semantic-search"))
    p.add_argument("--source-map",type=Path,default=Path("/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3"))
    p.add_argument("--original-root",type=Path,default=Path("/data/var/Knowledge Library"))
    p.add_argument("--derivative-root",type=Path,default=Path("/data/var/knowmoredirt/knowledge-library/semantic/text"))
    p.add_argument("--download-ledger",type=Path,default=Path("/data/var/Knowledge Library/_management/status.json"))
    args=p.parse_args()
    if (not isinstance(args.query,str) or not 3<=len(args.query)<=600
            or type(args.top) is not int or not 1<=args.top<=20):
        raise ValueError("invalid bounded query or top")
    for name in (args.summary_cli,args.original_passage_cli,args.source_map):
        if not name.is_file() or name.is_symlink():
            raise ValueError("source verification CLI or native source-map unavailable")
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    from file_system_catalog.dual_source_discovery import combined_dual_discovery
    # Both established native programs already apply source-provenance
    # rejection. The combined helper additionally rehashes the publisher
    # original, whole derivative and exact raw passage before reporting.
    summary=_command([sys.executable,str(args.summary_cli),"search",
                      args.query,"--top",str(args.top)])
    raw=_command([str(args.original_passage_cli),args.query,
                  "--mode","overview","--top",str(min(30,args.top*3))])
    with sqlite3.connect(args.source_map.resolve().as_uri()+"?mode=ro",uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        result=combined_dual_discovery(
            summary_payload=summary,original_payload=raw,
            source_map=con,original_root=args.original_root.resolve(strict=True),
            derivative_root=args.derivative_root.resolve(strict=True),
            original_receipts=_receipts(args.download_ledger),
            max_originals=args.top,
        )
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
