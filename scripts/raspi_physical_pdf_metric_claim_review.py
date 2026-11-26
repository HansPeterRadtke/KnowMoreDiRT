#!/usr/bin/env python3
"""Review named scientific model percentages on one current publisher PDF page.

Never edits a native summary or claims that geometric proximity is fact.
The exact source, derived text, complete native summary graph and acquisition
receipt are independently verified before and after the source-page scan.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

ROOT_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_REPO / "src"))
from file_system_catalog.physical_pdf_metric_review import (
    read_hashed_original_pdf_page_tsv,
    review_original_table_numeric_cells,
    validate_model_labels,
    validate_numeric_values,
)
from file_system_catalog.recent_source_tree_audit import (
    MAX_SOURCE_BYTES, MAX_DERIVED_BYTES, audit_current_source_tree,
)
from file_system_catalog.source_review_audit import safe_file
from raspi_summary_claim_evidence import (
    reviewed_native_root, read_approved_revision,
    ROOT, DERIVED, SOURCE_MAP, FILEDB, LEDGER,
)


def _named_claims(native: dict, focus: str, *, maximum: int = 16) -> list[dict]:
    if (not isinstance(native, dict)
            or native.get("schema") != "kmd.current_original_root_claim_evidence_review.v1"
            or native.get("original_and_native_tree_checked_again_after_review") is not True
            or not isinstance(native.get("all_native_node_claim_reviews"), list)
            or not isinstance(focus, str) or not isinstance(maximum, int)
            or isinstance(maximum, bool) or not 1 <= maximum <= 16):
        raise ValueError("requires verified current native summary with all child nodes")
    matcher = re.compile(r"(?<!\w)" + re.escape(focus) + r"(?!\w)", re.I)
    found = []
    for node in native["all_native_node_claim_reviews"]:
        for claim in node["claim_evidence"]["claims"]:
            if not matcher.search(claim["claim_text"]):
                continue
            val = claim.get("matched_value", "")
            if not isinstance(val, str) or not re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{1,3})?%", val):
                continue
            found.append({
                "native_node_id": node["node_id"],
                "native_node_level": node["level"],
                "native_node_original_source_start": node["original_source_start_char"],
                "native_node_original_source_end": node["original_source_end_char"],
                "native_summary_sha256": node["native_summary_sha256"],
                "generated_claim_text": claim["claim_text"],
                "generated_claim_sha256": claim["claim_sha256"],
                "claimed_numeric_percent": val,
                "source_number_match_status": claim["source_numeric_text_match_kind"],
                "source_text_anchor_locations": [
                    {"derived_original_source_unit_label": x["source_unit_label"],
                     "source_numeric_start_char": x["matched_anchor_char"]}
                    for x in claim["source_evidence"][:3]
                ],
                "claim_is_untrusted_generated_text_not_a_source_fact": True,
            })
            if len(found) > maximum:
                raise ValueError("too many current native summary metric review claims")
    return found


def review_current_pdf_summary_metrics(
    *, original_path: str, physical_page: int, focus_model: str,
    other_models: list[str], comparison_numbers: list[str],
    metric_headings:list[str],
    original_root: Path, derived_root: Path, source_map: Path,
    native_db: Path, completed_ledger: Path,
    max_native_nodes: int = 128,
) -> dict:
    if not isinstance(original_path, str) or not original_path.lower().endswith(".pdf"):
        raise ValueError("one approved original PDF relative path required")
    if type(physical_page) is not int or not 1 <= physical_page <= 500:
        raise ValueError("physical PDF table page must be a bounded positive integer")
    labels = validate_model_labels([focus_model, *other_models])
    if not isinstance(metric_headings,list):
        raise ValueError("explicit physical source metric headers must be a list")
    if metric_headings:
        validate_model_labels(metric_headings)
    if not isinstance(comparison_numbers,list):
        raise ValueError("comparison cells must be an explicit bounded list")
    if comparison_numbers:
        validate_numeric_values(comparison_numbers)
    if any(p.is_symlink() or not p.is_file() for p in (source_map,native_db,completed_ledger)):
        raise ValueError("current source map/native DB/approved acquisition ledger missing")
    publisher=safe_file(original_root,original_path)
    approved=read_approved_revision(completed_ledger,original_path)
    with sqlite3.connect(source_map.resolve().as_uri()+"?mode=ro",uri=True,timeout=15) as sm,\
         sqlite3.connect(native_db.resolve().as_uri()+"?mode=ro",uri=True,timeout=15) as db:
        sm.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA query_only=ON")
        native=reviewed_native_root(
            original_path=original_path,source_map=sm,native_db=db,
            original_root=original_root,derived_root=derived_root,
            completed_receipts={original_path:approved},
            context_chars=1200, all_nodes=True,
            maximum_reviewed_nodes=max_native_nodes,
            focus_subject_label=focus_model,
        )
        if native["original_sha256"] != approved:
            raise ValueError("current native original differs from unique completed acquisition receipt")
        ids=db.execute("SELECT id FROM filesystem_entries "
                       "WHERE relative_path_display=? AND entry_type='file'",
                       (original_path,)).fetchall()
        if len(ids)!=1 or type(ids[0][0]) is not int:
            raise ValueError("unique original native FileDB entry absent")
        native_original_id=ids[0][0]
        names=_named_claims(native,focus_model)
        numeric_values=list(dict.fromkeys(
            [x["claimed_numeric_percent"].rstrip("%") for x in names]
            +comparison_numbers))
        if numeric_values:
            validate_numeric_values(numeric_values)
        words,phys=read_hashed_original_pdf_page_tsv(
            publisher,source_sha256=approved,page=physical_page)
        geometry=review_original_table_numeric_cells(
            words=words,page=physical_page,
            model_labels=labels,numeric_values=numeric_values,
            explicit_metric_headings=metric_headings,
        )
        # The source can change between native graph check and physical-PDF
        # page extraction. Fail closed, rather than publish stale table cells.
        final=audit_current_source_tree(
            original_path=original_path,source_map=sm,native_db=db,
            original_root=original_root,derivative_root=derived_root,
            completed_receipts={original_path:approved},
            max_original_bytes=MAX_SOURCE_BYTES,
            max_derived_bytes=MAX_DERIVED_BYTES,
        )
        if final != native["verified_complete_native_source_tree"]:
            raise ValueError("source/native tree changed while reading physical publisher page")
    comparisons=[]
    for claim in names:
        val=claim["claimed_numeric_percent"].rstrip("%")
        candidates=[item for item in geometry["numeric_cell_candidates"]
                    if item["number"]==val]
        associations={x["unique_explicit_named_model_in_same_row_if_any"]
                      for x in candidates}
        # This is a candidate selection hint for a human review of the table
        # heading, numeric units and experimental protocol, NEVER a verdict.
        focus_visible=focus_model in associations
        other_observed={x for x in associations if x and x!=focus_model}
        remote_hints=[item['unique_distant_named_model_if_no_direct_match']
                      for item in candidates if item.get('unique_distant_named_model_if_no_direct_match')]
        if not candidates:
            cue="numeric_value_not_found_on_requested_original_physical_page"
        elif focus_visible:
            cue="focus_model_appears_near_at_least_one_matching_number_review_metric_column"
        elif other_observed:
            cue="numeric_value_near_other_named_model_but_not_focus_model_review_original_table"
        elif remote_hints:
            cue="named_model_only_far_left_same_horizontal_row_review_full_wide_table_and_metric_column"
        else:
            cue="numeric_value_present_without_unique_matching_named_model_review_table"
        comparisons.append({**claim,
            "physical_page_numeric_cell_candidates":candidates[:12],
            "source_geometry_review_cue":cue,
            "could_be_other_metric_or_parallel_column":True,
            "model_and_metric_attribution_factual_truth":"NOT_DETERMINED",
            "source_revision_and_native_tree_unchanged_after_review":True})
    return {
        "schema":"kmd.source_pinned_current_native_summary_original_pdf_metric_review.v1",
        "reviewed_at_utc":datetime.now(timezone.utc).isoformat(),
        "current_original_path":original_path,
        "approved_publisher_original_sha256":approved,
        "current_extracted_source_sha256":native["derived_sha256"],
        "current_native_summary_root_sha256":native["current_native_root_summary_sha256"],
        "current_native_original_document_id":native_original_id,
        "all_current_native_summary_nodes_reverified":native["original_and_native_tree_checked_again_after_review"],
        "source_and_entire_native_summary_graph_checked_after_physical_page_read":True,
        "explicit_focus_model":focus_model,
        "explicit_alternative_model_labels":other_models,
        "explicit_metric_column_labels_to_review":metric_headings,
        "original_physical_pdf_page_metadata":phys,
        "original_pdf_numeric_table_word_candidates":geometry,
        "current_generated_claims_containing_focus_model":comparisons,
        "full_current_native_tree_source_revision_sha_again_after_pdf_scan":hashlib.sha256(
            json.dumps(final,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest(),
        "num_current_native_nodes_reverified":final["current_native_summary_nodes_fully_verified"],
        "comparison_model_and_metric_truth_not_automatically_verified":True,
        "original_pdf_text_layer_not_complete_visual_table_figure_verification":True,
        "native_summary_vectors_and_original_source_files_unchanged":True,
        "no_source_acquisition_or_model_invocation":True,
        "publication_or_customer_external_effects":0,
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original-path",required=True)
    ap.add_argument("--physical-page",required=True,type=int)
    ap.add_argument("--focus-model",required=True)
    ap.add_argument("--other-model",action="append",default=[],
                    help="Explicit other model name visible in the physical table. May repeat.")
    ap.add_argument("--comparison-number",action="append",default=[],
                    help="Optional printed numerical table cell, e.g. 77.0; not a model result assertion.")
    ap.add_argument("--metric-heading",action="append",default=[],
                    help="Explicit metric column heading printed above table cells, e.g. LoCoMo or LongMemEval-S; geometric review only.")
    ap.add_argument("--maximum-reviewed-nodes",type=int,default=128)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--original-root",type=Path,default=ROOT)
    ap.add_argument("--derived-root",type=Path,default=DERIVED)
    ap.add_argument("--source-map",type=Path,default=SOURCE_MAP)
    ap.add_argument("--native-db",type=Path,default=FILEDB)
    ap.add_argument("--completed-ledger",type=Path,default=LEDGER)
    args=ap.parse_args()
    out=args.output
    if (out.suffix!=".json" or out.exists() or out.is_symlink()
            or not out.parent.is_dir() or out.parent.is_symlink()):
        raise ValueError("source-reviewed original-PDF table evidence requires a fresh private JSON path")
    result=review_current_pdf_summary_metrics(
        original_path=args.original_path,physical_page=args.physical_page,
        focus_model=args.focus_model,other_models=args.other_model,
        comparison_numbers=args.comparison_number,metric_headings=args.metric_heading,
        original_root=args.original_root,derived_root=args.derived_root,
        source_map=args.source_map,native_db=args.native_db,
        completed_ledger=args.completed_ledger,max_native_nodes=args.maximum_reviewed_nodes,
    )
    old_umask=os.umask(0o077)
    try:
        with out.open("x",encoding="utf8") as stream:
            os.chmod(out,0o600)
            json.dump(result,stream,indent=2,ensure_ascii=False)
            stream.write("\n")
            stream.flush();os.fsync(stream.fileno())
    finally:
        os.umask(old_umask)
    print(json.dumps({"status":"source_pinned_physical_pdf_table_review_saved",
                      "output":str(out),"current_generated_claim_review_candidates":len(result["current_generated_claims_containing_focus_model"]),
                      "physical_pdf_cells_examined":result["original_pdf_numeric_table_word_candidates"]["numeric_cells_found"],
                      "factual_truth_automatically_decided":False,
                      "native_database_or_original_changed":False}))


if __name__=="__main__":
    main()
