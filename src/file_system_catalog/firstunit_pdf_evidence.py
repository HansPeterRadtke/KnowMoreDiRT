"""Preserve independent lexical ranks while selecting distinct original PDFs.

Receipt metadata is navigation only. The caller must independently attest
current original/derivative/native identity before and after the selected
PDF evidence reads.
"""
from __future__ import annotations

import hashlib
import re

SCHEMA="kmd.readonly_publisher_first_source_unit_dual_lexical_discovery.v1"
_CHANNELS=("literal","porter")
_SHA=re.compile(r"[0-9a-f]{64}\Z")


def _path(value):
    return (isinstance(value,str) and 1<=len(value)<=1600
            and not value.startswith("/") and "\\" not in value and "\x00" not in value
            and all(p not in ("",".","..") for p in value.split("/")))


def _identity(row):
    original=row.get("publisher_original_sha256");derived=row.get("current_derivative_sha256")
    native=row.get("native_original_filedb_entry_id")
    if (not isinstance(original,str) or not _SHA.fullmatch(original)
            or not isinstance(derived,str) or not _SHA.fullmatch(derived)
            or type(native) is not int or native<1
            or row.get("current_approved_publisher_and_extracted_bytes_rehashed") is not True):
        raise ValueError("lexical receipt lacks exact original/derivative/native identity")
    return original,derived,native


def select_lexical_pdf_candidates(receipt:dict,*,originals_per_query:int=1)->list[dict]:
    """Bounded round-robin selection, no ranking fusion or score comparison."""
    if (not isinstance(receipt,dict) or receipt.get("schema")!=SCHEMA
            or receipt.get("independent_rank_channels_no_fusion") is not True
            or receipt.get("native_filedb_and_source_map_read_only") is not True
            or receipt.get("all_displayed_candidates_original_and_full_derivative_sha_reverified") is not True
            or type(originals_per_query) is not int or not 1<=originals_per_query<=3
            or not isinstance(receipt.get("queries"),list)
            or not 1<=len(receipt["queries"])<=8
            or type(receipt.get("query_count")) is not int
            or receipt["query_count"]!=len(receipt["queries"])):
        raise ValueError("bounded independent lexical receipt required")
    selected=[];seen_queries=set();global_ids={};global_paths={}
    for q in receipt["queries"]:
        if not isinstance(q,dict):raise ValueError("invalid query")
        label=q.get("id");query=q.get("query")
        if (not isinstance(label,str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}",label)
                or label in seen_queries or not isinstance(query,str)
                or not 3<=len(query.strip())<=600 or len(query)>600
                or q.get("query_sha256")!=hashlib.sha256(query.encode()).hexdigest()
                or q.get("rank_fusion_applied") is not False
                or not isinstance(q.get("independent_rank_channels"),dict)
                or set(q["independent_rank_channels"])!=set(_CHANNELS)):
            raise ValueError("changed or invalid lexical query identity")
        seen_queries.add(label)
        per_path={};orders={}
        for channel in _CHANNELS:
            item=q["independent_rank_channels"][channel]
            rows=item.get("source_originals") if isinstance(item,dict) else None
            if (not isinstance(rows,list) or len(rows)>20
                    or type(item.get("returned_current_originals")) is not int
                    or item["returned_current_originals"]!=len(rows)):
                raise ValueError("unbounded or inconsistent lexical ranking")
            orders[channel]=[];seen=set()
            for rank,row in enumerate(rows,1):
                if (not isinstance(row,dict) or type(row.get("source_rank")) is not int
                        or row["source_rank"]!=rank
                        or row.get("rank_channel")!=channel+"_title_firstpage_bm25"
                        or not _path(row.get("original_path"))
                        or row["original_path"] in seen):
                    raise ValueError("invalid or duplicate independent lexical rank")
                path=row["original_path"];seen.add(path)
                identity=_identity(row)
                if ((path in global_paths and global_paths[path]!=identity)
                        or (identity[2] in global_ids and global_ids[identity[2]]!=path)):
                    raise ValueError("lexical channels disagree about source revision or native ID")
                global_paths[path]=identity;global_ids[identity[2]]=path
                if not path.casefold().endswith(".pdf"):continue
                entry=per_path.setdefault(path,{"original_path":path,
                    "original_sha256":identity[0],"derived_sha256":identity[1],
                    "native_original_id":identity[2],"independent_lexical_ranks":{}})
                entry["independent_lexical_ranks"][channel]=rank
                orders[channel].append((rank,path))
        # Round-robin the original rank numbers, not the compressed PDF-only lists.
        # Non-PDF results remain in the input receipt and are not re-ranked.
        positions={c:dict(orders[c]) for c in _CHANNELS};out=[];chosen=set()
        for rank in range(1,21):
            for channel in _CHANNELS:
                path=positions[channel].get(rank)
                if path is not None and path not in chosen:
                    out.append(per_path[path]);chosen.add(path)
                if len(out)>=originals_per_query:break
            if len(out)>=originals_per_query:break
        selected.append({"id":label,"query":query,"query_sha256":q["query_sha256"],
                         "selected_originals":out,"available_distinct_pdf_candidates":len(per_path)})
    return selected


def require_lexical_current_identity(candidate,source,native_id):
    if (not isinstance(source,dict)
            or source.get("original_path")!=candidate["original_path"]
            or source.get("original_sha256")!=candidate["original_sha256"]
            or source.get("current_derived_sha256")!=candidate["derived_sha256"]
            or type(native_id) is not int or native_id!=candidate["native_original_id"]
            or source.get("publisher_original_actual_sha256_verified") is not True
            or source.get("source_map_derivative_actual_sha256_verified") is not True):
        raise ValueError("lexical receipt source identity is not the current approved revision")
