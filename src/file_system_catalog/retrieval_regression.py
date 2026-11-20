"""Predeclared exact-original ranking regression audit, not a retrieval solver.

A ranking result is a discovery candidate, never source truth. Dataset labels
are fixed outside the tested index. Historical negative labels mean "no
predeclared gold", NOT objectively unanswerable or irrelevant.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Callable


def validated_gold(raw:list[dict]) -> list[dict]:
    if not isinstance(raw,list) or not 1<=len(raw)<=200:
        raise ValueError("gold cases must be a bounded nonempty list")
    ids=set()
    cases=[]
    for row in raw:
        if not isinstance(row,dict):
            raise ValueError("invalid gold case")
        label=row.get("id")
        query=row.get("query")
        expected=row.get("expected_path")
        if (not isinstance(label,str) or not 1<=len(label)<=128 or label in ids
                or not isinstance(query,str) or not 3<=len(query)<=1000
                or (expected is not None and
                    (not isinstance(expected,str) or not expected or
                     expected.startswith("/") or "\\" in expected or
                     any(part in ("",".","..") for part in expected.split("/"))))):
            raise ValueError("invalid, duplicate, or unsafe retrieval gold case")
        ids.add(label)
        cases.append({"id":label,"query":query,"expected_path":expected,
                      "category":row.get("category")})
    return cases


def source_record(db:sqlite3.Connection,source_map:sqlite3.Connection,
                  path:str|None)->dict:
    if path is None:
        return {"gold_status":"no_gold_predeclared_not_unanswerable",
                "original_present":None,"native_summary_rows":None,
                "current_source_revision_verified":None}
    rows=db.execute(
        "SELECT id FROM filesystem_entries WHERE relative_path_display=? AND entry_type='file'",
        (path,),
    ).fetchall()
    if len(rows)!=1:
        return {"gold_status":"canonical_native_original_absent_or_ambiguous",
                "original_present":False,"native_summary_rows":0,
                "current_source_revision_verified":False}
    (entry_id,)=rows[0]
    summary=db.execute(
        "SELECT count(*) FROM content_chunks WHERE filesystem_entry_id=? "
        "AND collection_id LIKE '%:summary'",
        (entry_id,),
    ).fetchone()[0]
    versions=source_map.execute(
        "SELECT status,source_sha256,derived_sha256 FROM originals WHERE source_path=?",
        (path,),
    ).fetchall()
    if len(versions)!=1 or versions[0][0]!="complete":
        return {"gold_status":"source_map_missing_or_not_complete",
                "original_present":True,"native_summary_rows":int(summary),
                "current_source_revision_verified":False}
    # Source-map digest identity alone is not a full byte recheck. This is
    # intentionally metadata/revision corroboration, not a literal citation.
    status,orig,deriv=versions[0]
    if not all(isinstance(v,str) and len(v)==64 for v in (orig,deriv)):
        return {"gold_status":"invalid_source_map_hash",
                "original_present":True,"native_summary_rows":int(summary),
                "current_source_revision_verified":False}
    # Both source revisions must agree. A matching original SHA is NOT
    # enough: the derived text may have changed without the original PDF
    # changing, and a historical summary might still refer to its old text.
    # This is still metadata corroboration, not a fresh publisher-file
    # byte hash; the normal KMD search validates returned original passages.
    valid=0
    for (generation,) in db.execute(
        "SELECT r.generation_json FROM content_chunks c "
        "JOIN content_representations r ON r.chunk_id=c.chunk_id "
        "WHERE c.filesystem_entry_id=? AND c.collection_id LIKE '%:summary' "
        "AND c.content_sha256=? AND r.prompt_version='kmd-hierarchical-source-summary-v1' "
        "AND r.analysis_status='complete' AND r.representation_kind IN ('summary','summary_long')",
        (entry_id,orig),
    ):
        try:
            provenance=json.loads(generation)
        except (TypeError,ValueError):
            continue
        if (isinstance(provenance,dict)
                and provenance.get('original_sha256')==orig
                and provenance.get('derived_sha256')==deriv):
            valid+=1
    return {"gold_status":"native_summaries_match_source_map_revision" if valid else
            "native_summary_missing_current_source_revision",
            "original_present":True,"native_summary_rows":int(summary),
            "current_source_revision_verified":bool(valid),
            "matching_revision_summary_representations":valid,
            "source_hash_refers_to_map_not_rehashed_bytes":True}


def evaluate_case(case:dict, search:Callable[[str],list[str]],
                  source_info:Callable[[str|None],dict])->dict:
    paths=search(case["query"])
    if not isinstance(paths,list) or len(paths)>100 or any(not isinstance(x,str) for x in paths):
        raise ValueError("search results not a bounded list of original paths")
    if len(set(paths))!=len(paths):
        raise ValueError("search returned duplicate original identity")
    expected=case["expected_path"]
    rank=paths.index(expected)+1 if expected in paths else None
    return {
        "id":case["id"],
        "category":case.get("category"),
        "query_sha256":hashlib.sha256(case["query"].encode("utf-8")).hexdigest(),
        "gold_path":expected,
        "gold_rank":rank,
        "top_three_original_paths":paths[:3],
        "source_identity_evidence":source_info(expected),
        "historical_negative_means_absent_gold_not_no_answer":expected is None,
    }


def score(rows:list[dict],baseline:dict|None=None)->dict:
    positives=[r for r in rows if r["gold_path"] is not None]
    negatives=[r for r in rows if r["gold_path"] is None]
    total=len(positives)
    values=[r["gold_rank"] for r in positives]
    summary={
        "cases":len(rows),
        "positive_gold_cases":total,
        "historical_negative_unlabeled_cases":len(negatives),
        "correct_top1":sum(v==1 for v in values),
        "correct_top3":sum(v is not None and v<=3 for v in values),
        "correct_top5":sum(v is not None and v<=5 for v in values),
        "correct_top10":sum(v is not None and v<=10 for v in values),
        "mean_reciprocal_rank_at10":round(sum(1/v for v in values if v)/max(1,total),6),
        "gold_with_missing_or_stale_native_source":sum(
            r["source_identity_evidence"].get("gold_status")!="native_summaries_match_source_map_revision"
            for r in positives),
    }
    if baseline is not None:
        summary["historical_top1"]=baseline.get("top1_count")
        summary["historical_top10"]=baseline.get("top10_count")
        if isinstance(summary["historical_top10"],int):
            summary["top10_difference_vs_historical"]=summary["correct_top10"]-summary["historical_top10"]
    return summary
