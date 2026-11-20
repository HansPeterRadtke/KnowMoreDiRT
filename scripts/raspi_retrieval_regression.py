#!/usr/bin/env python3
"""Read-only, reproducible exact-source retrieval drift benchmark.

Gold queries are predeclared; partial results stay private if a test fails.
Do not use this to automatically choose embedding/ranking parameters.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold",type=Path,required=True)
    p.add_argument("--baseline",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--database",type=Path,required=True)
    p.add_argument("--source-map",type=Path,required=True)
    p.add_argument("--search-cli",type=Path,required=True)
    p.add_argument("--top",type=int,default=10)
    p.add_argument("--case-id",action="append",
                   help="Optional explicitly identified case to replay; subtotal is NOT comparable with full historical totals")
    args=p.parse_args()
    if type(args.top) is not int or not 1<=args.top<=100:
        raise ValueError("invalid retrieval top limit")
    dest=args.output.expanduser()
    if dest.exists() or dest.is_symlink() or not dest.parent.is_dir() or dest.parent.is_symlink():
        raise ValueError("private benchmark output must be a new file in an existing real directory")
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    from file_system_catalog.retrieval_regression import (
        validated_gold,source_record,evaluate_case,score,
    )
    gold=args.gold.read_bytes()
    if len(gold)>100000:
        raise ValueError("gold file too large")
    full_cases=validated_gold(json.loads(gold))
    cases=full_cases
    if args.case_id:
        selected=set(args.case_id)
        if len(selected)!=len(args.case_id):
            raise ValueError("duplicate selected case id")
        available={case["id"] for case in full_cases}
        if not selected.issubset(available):
            raise ValueError("unknown selected case id")
        cases=[case for case in full_cases if case["id"] in selected]
    baseline=None
    if args.baseline:
        raw=json.loads(args.baseline.read_text())
        baseline=raw["summary"] if len(cases)==len(full_cases) else None
    if not all(p.is_file() and not p.is_symlink() for p in
               (args.database,args.source_map,args.search_cli)):
        raise ValueError("canonical source, map or search executable missing/symlinked")
    # Persistent connection is query-only; the tested CLI independently
    # performs its own current original+derivative byte provenance checks.
    with sqlite3.connect(args.database.resolve().as_uri()+"?mode=ro",uri=True) as db, \
         sqlite3.connect(args.source_map.resolve().as_uri()+"?mode=ro",uri=True) as source_map:
        db.execute("PRAGMA query_only=ON")
        source_map.execute("PRAGMA query_only=ON")
        rows=[]
        def search(q):
            result=subprocess.run(
                [sys.executable,str(args.search_cli),"search",q,
                 "--top",str(args.top)],capture_output=True,
                text=True,timeout=120,check=True,
            )
            payload=json.loads(result.stdout)
            if payload.get("provenance_policy")!="publisher_and_derived_sha256_checked;summary_is_not_primary_evidence":
                raise ValueError("search did not report expected publisher revision policy")
            return [x["original_path"] for x in payload["documents"]]
        for n,case in enumerate(cases,1):
            started=time.monotonic()
            row=evaluate_case(case,search,lambda path:source_record(db,source_map,path))
            row["elapsed_seconds"]=round(time.monotonic()-started,2)
            rows.append(row)
            print(f"CHECK {n}/{len(cases)} {row['id']} rank={row['gold_rank']}",flush=True)
    report={
        "schema":"kmd.predeclared_original_retrieval_drift.v1",
        "checked_at_utc":datetime.now(timezone.utc).isoformat(),
        "fixed_gold_sha256":hashlib.sha256(gold).hexdigest(),
        "tested_search_cli":str(args.search_cli),
        "method":"current native KMD original-linked search, results are discovery candidates not source facts",
        "selected_case_ids":[x["id"] for x in cases],
        "complete_fixed_population":len(cases)==len(full_cases),
        "scope":"only exact path predeclared gold; historical negatives are not labeled unanswerable",
        "summary":score(rows,baseline),
        "cases":rows,
    }
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    dest.chmod(0o600)
    print(json.dumps(report["summary"],indent=2))


if __name__=="__main__":
    main()
