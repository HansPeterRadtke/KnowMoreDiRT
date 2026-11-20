#!/usr/bin/env python3
"""Reconcile one native KMD FileDB and selectively rehash original publisher files."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--derived-root",type=Path,required=True)
    p.add_argument("--approved-catalog",type=Path,required=True)
    p.add_argument("--completed-ledger",type=Path,required=True)
    p.add_argument("--source-map-db",type=Path,required=True)
    p.add_argument("--native-filedb",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",default="KnowMoreDiRT-source-integrity-2026-11-20")
    p.add_argument("--original-byte-limit-mib",type=int,default=20)
    p.add_argument("--total-sample-budget-mib",type=int,default=400)
    args=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    from file_system_catalog.source_revision_audit import (
        audit,private_audit_destination,
    )
    output=private_audit_destination(
        args.output,source_root=args.source_root,derived_root=args.derived_root,
        native_filedb=args.native_filedb,source_map_db=args.source_map_db,
    )
    data=audit(
        source_root=args.source_root,derived_root=args.derived_root,
        approved_catalog=args.approved_catalog,completed_ledger=args.completed_ledger,
        source_map_db=args.source_map_db,native_filedb=args.native_filedb,
        seed=args.seed,max_original_bytes=args.original_byte_limit_mib*1024**2,
        max_total_bytes=args.total_sample_budget_mib*1024**2,
    )
    out=output
    out.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    out.chmod(0o600)
    s=data["summary"]
    print(json.dumps({
        "report_path":str(out),"metadata_originals":s["curated_source_map_originals"],
        "metadata_problem_categories":s["metadata_problem_categories"],
        "sample_originals_verified":s["originals_independently_hashed"],
        "sample_derivatives_verified":s["current_derivatives_independently_hashed"],
        "sample_with_failures":s["selected_originals_with_failures"],
        "sample_skips":s["skipped_due_budgets"],
        "native_summarized":s["native_summary_originals"],
        "native_overview":s["native_overview_originals"],
    },indent=2))


if __name__=="__main__":
    main()
