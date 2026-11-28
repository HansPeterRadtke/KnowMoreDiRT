#!/usr/bin/env python3
"""Read-only, reproducible exact-source retrieval drift benchmark.

Gold queries are predeclared; partial results stay private if a test fails.
Do not use this to automatically choose embedding/ranking parameters.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
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
    p.add_argument("--attest-html-gold",action="store_true",
                   help="Independently hash current publisher HTML/text and derivative bytes and check frozen supporting source excerpts before search")
    p.add_argument("--original-root",type=Path)
    p.add_argument("--derived-root",type=Path)
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
        validated_gold,source_record,evaluate_case,score,assert_equivalent_gold,
        open_live_readonly_db,
        attest_html_original_gold,
    )
    gold=args.gold.read_bytes()
    if len(gold)>100000:
        raise ValueError("gold file too large")
    document=json.loads(gold)
    full_cases=validated_gold(document["cases"] if isinstance(document,dict) else document)
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
        if len(cases)==len(full_cases):
            assert_equivalent_gold(
                cases,raw,fixed_gold_sha256=hashlib.sha256(gold).hexdigest())
            baseline=raw["summary"]
    if args.attest_html_gold and (args.original_root is None or args.derived_root is None):
        raise ValueError("strict original-source attestation requires --original-root and --derived-root")
    if not all(p.is_file() and not p.is_symlink() for p in
               (args.database,args.source_map,args.search_cli)):
        raise ValueError("canonical source, map or search executable missing/symlinked")
    # Persistent connection is query-only; the tested CLI independently
    # performs its own current original+derivative byte provenance checks.
    # The live native FileDB uses rollback journaling and two original-index
    # writers. A briefly held EXCLUSIVE write lock is not missing gold: wait
    # boundedly for its release, then still fail closed on persistent locks.
    # Both connections remain mode=ro and query_only throughout the benchmark.
    with closing(open_live_readonly_db(args.database)) as db, \
         closing(open_live_readonly_db(args.source_map)) as source_map:
        publisher_attestations=None
        if args.attest_html_gold:
            if not isinstance(document,dict) or not isinstance(document.get("cases"),list):
                raise ValueError("strict attestation requires predeclared case provenance in a gold wrapper")
            publisher_attestations=attest_html_original_gold(
                document["cases"],{case["id"] for case in cases},
                original_root=args.original_root,derivative_root=args.derived_root,
                source_map=source_map,
            )
            print(f"PUBLISHER_GOLD_ATTESTED {len(publisher_attestations)} queries",flush=True)
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
            if publisher_attestations is not None:
                row["publisher_gold_attestation"]=publisher_attestations[case["id"]]
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
        "all_publisher_gold_bytes_rechecked":bool(args.attest_html_gold),
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
