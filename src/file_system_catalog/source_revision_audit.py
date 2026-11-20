"""Read-only curated Knowledge Library source revision + index coverage audit.

Reconciles ALL curated source metadata but only hashes a deterministic
bounded, cross-type sample of actual original and derived bytes.
No source/model downloads, mutations, DB rebuild, external writes or
pretence that vector presence proves passage semantics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import stat
from typing import Any

MAX_MANIFEST_BYTES=64*1024**2
MAX_ORIGINAL_SAMPLE=20*1024**2
MAX_DERIVED_SAMPLE=24*1024**2
MAX_ERROR_ORIGINAL=48*1024**2
MAX_SAMPLE_BYTES=400*1024**2
_QUOTAS={"pdf":48, "html":16, "skill":12, "text":12, "zip":5, "xlsx":5}


def _path(relative: str, root: Path) -> Path:
    if not isinstance(relative,str) or not 1<=len(relative)<=1600:
        raise ValueError("invalid relative original or derivative path")
    parts=relative.split("/")
    if (PurePosixPath(relative).is_absolute() or
        "\\" in relative or "\x00" in relative or
        any(p in ("", ".", "..") for p in parts)):
        raise ValueError("path traversal or symlink-unsafe original")
    root=root.resolve(strict=True)
    out=root/relative
    if (not out.is_file() or out.is_symlink()
            or not out.resolve(strict=True).is_relative_to(root)):
        raise ValueError("source missing, escaped, or symlinked")
    return out


def private_audit_destination(
    output:Path, *, source_root:Path, derived_root:Path,
    native_filedb:Path, source_map_db:Path,
) -> Path:
    """New report only; never under original or canonical native data paths."""
    dest=Path(output).expanduser()
    if dest.exists() or dest.is_symlink():
        raise ValueError("refuse to overwrite existing provenance audit")
    if not dest.parent.is_dir():
        raise ValueError("private audit output parent is missing")
    parent=dest.parent.resolve(strict=True)
    original=Path(source_root).resolve(strict=True)
    derivative=Path(derived_root).resolve(strict=True)
    native=Path(native_filedb).resolve(strict=True)
    smap=Path(source_map_db).resolve(strict=True)
    if (parent.is_relative_to(original) or parent.is_relative_to(derivative)
            or parent==native.parent or parent==smap.parent
            or dest.resolve(strict=False) in (native,smap)):
        raise ValueError("refuse audit output inside protected originals or native data")
    if parent.is_relative_to(native.parent):
        if not parent.is_relative_to(native.parent/"search-audit"):
            raise ValueError("native KMD private audit must use the search-audit subtree")
    return parent/dest.name


def _read_catalog(path: Path, field:str) -> tuple[dict[str,dict],list[str]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_MANIFEST_BYTES:
        raise ValueError("catalog unavailable, symlink or excessive")
    data=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data,dict) or not isinstance(data.get(field),list):
        raise ValueError("invalid source/receipt array")
    items={}
    duplicates=[]
    for row in data[field]:
        if not isinstance(row,dict) or not isinstance(row.get("path"),str):
            raise ValueError("invalid catalog source row")
        name=row["path"]
        if name in items:
            duplicates.append(name)
        else:
            items[name]=row
    return items,duplicates


def _db(path:Path)->sqlite3.Connection:
    if path.is_symlink() or not path.is_file():
        raise ValueError("canonical native/source-map DB absent or symlink")
    uri=path.resolve(strict=True).as_uri()+"?mode=ro"
    db=sqlite3.connect(uri,uri=True,timeout=12)
    db.execute("pragma query_only=ON")
    return db


def _digest_stable(path:Path, maximum:int)->tuple[str,int]:
    before=path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_size>maximum
        or before.st_size<0):
        raise ValueError("source outside sampled byte limit or is not regular")
    h=hashlib.sha256()
    total=0
    with path.open("rb") as stream:
        while piece:=stream.read(1024*1024):
            h.update(piece);total+=len(piece)
    after=path.lstat()
    if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
        !=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
        or total!=before.st_size):
        raise ValueError("source revision changed while calculating its bytes")
    return h.hexdigest(),total


def _verified_derived_body_count(
    text:str, *, original_path:str, original_sha:str, source_kind:str,
) -> int:
    """Match Infra's extraction accounting, excluding KMD's generated header.

    `text_characters` in source-map counts the literal *source-unit blocks*,
    not the synthetic publisher provenance preamble. The blocks start with
    two newlines before the first `[[SOURCE UNIT: ...]]` marker.
    """
    title="KMD source provenance (the original PDF/HTML/ZIP/XLSX, not a new publisher book):"
    first,sep,following=text.partition("\n")
    if first!=title or not sep:
        raise ValueError("missing canonical KMD extracted-text provenance header")
    jsonline,sep2,_=following.partition("\n")
    if not sep2:
        raise ValueError("missing embedded original provenance JSON")
    try:
        meta=json.loads(jsonline)
    except (ValueError,TypeError) as exc:
        raise ValueError("invalid embedded original revision JSON") from exc
    if (not isinstance(meta,dict) or meta.get("source_path")!=original_path
            or meta.get("source_sha256")!=original_sha
            or meta.get("kind")!=source_kind):
        raise ValueError("derived wrapper points to a different source revision")
    header=title+"\n"+jsonline+"\n\n"
    if not text.startswith(header) or not text[len(header):].startswith("\n\n[[SOURCE UNIT: "):
        raise ValueError("missing exact source-unit boundary after generated header")
    return len(text)-len(header)


def _selection_score(seed:str,name:str)->str:
    return hashlib.sha256((seed+"\x00"+name).encode("utf-8")).hexdigest()


def deterministic_sample(
    records:list[dict], *,
    seed:str="KnowMoreDiRT-source-integrity-2026-11-20",
    quotas:dict[str,int]|None=None,
    max_original_bytes:int=MAX_ORIGINAL_SAMPLE,
)->list[dict]:
    if not isinstance(seed,str) or not 1<=len(seed)<=100:
        raise ValueError("invalid deterministic sample seed")
    quota=_QUOTAS if quotas is None else quotas
    if (not isinstance(quota,dict) or
            any(not isinstance(k,str) or type(v) is not int or v<0 or v>100
                for k,v in quota.items())
            or type(max_original_bytes) is not int or max_original_bytes<1):
        raise ValueError("invalid bounded sample quotas")
    selected={}
    by_kind=defaultdict(list)
    for row in records:
        kind=row["kind"]
        if row["status"]=="error":
            selected[row["path"]]=row
        elif kind in quota and row["source_bytes"]<=max_original_bytes:
            by_kind[kind].append(row)
    for kind,items in by_kind.items():
        wanted=quota.get(kind,0)
        ranks=lambda r:(_selection_score(seed,r["path"]),r["path"])
        covered=sorted((r for r in items if r["summarized"]),key=ranks)
        pending=sorted((r for r in items if not r["summarized"]),key=ranks)
        # Preserves both summarized and unsummarized originals, if present.
        chosen=covered[:wanted//2]+pending[:wanted//2]
        taken={r["path"] for r in chosen}
        leftovers=sorted((r for r in items if r["path"] not in taken),key=ranks)
        chosen+=leftovers[:max(0,wanted-len(chosen))]
        for row in chosen:
            selected[row["path"]]=row
    return [selected[k] for k in sorted(selected)]


def audit(
    *, source_root:Path,derived_root:Path,
    approved_catalog:Path,completed_ledger:Path,
    source_map_db:Path,native_filedb:Path,
    seed:str="KnowMoreDiRT-source-integrity-2026-11-20",
    quotas:dict[str,int]|None=None,
    max_original_bytes:int=MAX_ORIGINAL_SAMPLE,
    max_derived_bytes:int=MAX_DERIVED_SAMPLE,
    max_total_bytes:int=MAX_SAMPLE_BYTES,
) -> dict[str,Any]:
    if (not 1<=max_original_bytes<=50*1024**2 or
        not 1<=max_derived_bytes<=50*1024**2 or
        not 1<=max_total_bytes<=1024**3):
        raise ValueError("audit byte budgets exceed expected safety policy")
    source_root=Path(source_root).resolve(strict=True)
    derived_root=Path(derived_root).resolve(strict=True)
    if derived_root.is_relative_to(source_root):
        raise ValueError("derived text must be outside immutable originals")
    catalog,cat_dupes=_read_catalog(Path(approved_catalog),"sources")
    ledger,ledger_dupes=_read_catalog(Path(completed_ledger),"items")
    problems=defaultdict(list)
    if cat_dupes:problems["catalog_duplicate_paths"].extend(cat_dupes)
    if ledger_dupes:problems["ledger_duplicate_paths"].extend(ledger_dupes)
    with _db(Path(source_map_db)) as src, _db(Path(native_filedb)) as kmd:
        original_rows=src.execute(
            "select source_path,original_kind,status,source_sha256,"
            "original_bytes,derived_path,derived_sha256,text_characters,"
            "original_title,original_url from originals"
        ).fetchall()
        native=defaultdict(list)
        for entry_id,path,size,stored_sha,hash_status in kmd.execute(
            "SELECT id,relative_path_display,size_bytes,content_sha256,hash_status "
            "FROM filesystem_entries WHERE entry_type='file'"
        ):
            native[path].append((entry_id,size,stored_sha,hash_status))
        indexed=defaultdict(Counter)
        for entry_id,kind,count in kmd.execute(
            "SELECT filesystem_entry_id,collection_id,count(*) "
            "FROM content_chunks GROUP BY filesystem_entry_id,collection_id"
        ):
            indexed[entry_id][kind]=count
        integrity=kmd.execute("pragma quick_check").fetchone()[0]
        foreign_errors=kmd.execute("pragma foreign_key_check").fetchall()
    original_paths=set()
    rows=[]
    for (path,kind,status,sha,size,derivative,dsha,characters,title,url) in original_rows:
        if path in original_paths:
            problems["source_map_duplicate_paths"].append(path)
        original_paths.add(path)
        sourcecat=catalog.get(path)
        receipt=ledger.get(path)
        matches=native.get(path,[])
        if sourcecat is None:
            problems["catalog_missing"].append(path)
        elif sourcecat.get("kind")!=kind:
            problems["kind_mismatch"].append(path)
        if sourcecat is not None:
            if sourcecat.get("title") != title:
                problems["catalog_title_differs_from_source_map"].append(path)
            if sourcecat.get("source_url",sourcecat.get("url")) != url:
                problems["catalog_publisher_url_differs_from_source_map"].append(path)
        if receipt is None:
            problems["receipt_missing"].append(path)
        elif (receipt.get("status")!="complete"
              or receipt.get("sha256")!=sha
              or receipt.get("bytes_received")!=size):
            problems["receipt_revision_or_size_mismatch"].append(path)
        if len(matches)!=1:
            problems["native_file_row_not_unique"].append(path)
        elif matches[0][1]!=size:
            problems["native_file_row_size_mismatch"].append(path)
        if status not in ("complete","error"):
            problems["unrecognized_source_map_status"].append(path)
        file_id=matches[0][0] if len(matches)==1 else None
        counts=indexed.get(file_id,Counter()) if file_id is not None else Counter()
        summary_chunks=sum(v for k,v in counts.items() if str(k).endswith(":summary"))
        overview_chunks=sum(v for k,v in counts.items() if str(k).endswith(":overview"))
        full_chunks=sum(v for k,v in counts.items() if str(k).endswith(":full"))
        if status=="complete" and overview_chunks==0:
            problems["extracted_without_overview"].append(path)
        if status=="error" and (summary_chunks or overview_chunks or full_chunks):
            problems["error_source_with_unverified_vectors"].append(path)
        rows.append({
            "path":path,"kind":kind,"status":status,"source_sha":sha,
            "source_bytes":size,"derived_path":derivative,"derived_sha":dsha,
            "text_characters":characters,
            "native_id":file_id,"native_stored_sha":matches[0][2] if len(matches)==1 else None,
            "native_hash_status":matches[0][3] if len(matches)==1 else None,
            "summarized":summary_chunks>0,"summary_chunks":summary_chunks,
            "overview_chunks":overview_chunks,"full_chunks":full_chunks,
        })
    for key in set(catalog)-original_paths:
        problems["catalog_not_in_source_map"].append(key)
    for key in set(ledger)-original_paths:
        problems["ledger_not_in_source_map"].append(key)
    if integrity!="ok":problems["filedb_quick_check_failed"].append(str(integrity))
    if foreign_errors:
        problems["filedb_foreign_key_faults"].extend([str(x)[:80] for x in foreign_errors[:30]])
    selected=deterministic_sample(rows,seed=seed,quotas=quotas,
                                  max_original_bytes=max_original_bytes)
    sample=[]
    bytes_hashed=0
    for row in selected:
        path=row["path"]
        receipt=ledger.get(path)
        report={"path":path,"kind":row["kind"],"status":row["status"],
                "native_id":row["native_id"],
                "summary_chunks":row["summary_chunks"],
                "overview_chunks":row["overview_chunks"],
                "full_chunks":row["full_chunks"],
                "original_file_revision_verified":False,
                "extracted_derivative_revision_verified":False,
                "source_receipt_sha256":row["source_sha"],
                "source_derived_sha256":row["derived_sha"],
                "failures":[],"skips":[]}
        if bytes_hashed+row["source_bytes"]>max_total_bytes:
            report["skips"].append("whole_audit_byte_budget")
        else:
            try:
                source=_path(path,source_root)
                cap=(MAX_ERROR_ORIGINAL if row["status"]=="error"
                     else max_original_bytes)
                actual,size=_digest_stable(source,cap)
                bytes_hashed+=size
                if (actual!=row["source_sha"] or
                    not receipt or receipt.get("sha256")!=actual or
                    receipt.get("bytes_received")!=size):
                    report["failures"].append("publisher_sha_or_completed_receipt_mismatch")
                else:
                    report["original_file_revision_verified"]=True
                    report["actual_original_bytes"]=size
            except (OSError,ValueError,UnicodeError) as exc:
                report["failures"].append("original_revision_unverifiable:"+type(exc).__name__)
        if row["status"]=="complete":
            if not isinstance(row["derived_path"],str) or not isinstance(row["derived_sha"],str):
                report["failures"].append("source_map_completed_missing_derivative")
            elif bytes_hashed+min(int(row["text_characters"] or 0),max_derived_bytes)>max_total_bytes:
                report["skips"].append("derived_audit_byte_budget")
            else:
                try:
                    derivative=_path(row["derived_path"],derived_root)
                    actual,size=_digest_stable(derivative,max_derived_bytes)
                    bytes_hashed+=size
                    if actual!=row["derived_sha"]:
                        report["failures"].append("derived_sha_differs_from_source_map")
                    else:
                        raw=derivative.read_bytes()
                        # An additional read guards size and verifies original
                        # char accounting, never normalizing CRLF.
                        txt=raw.decode("utf-8")
                        body_chars=_verified_derived_body_count(
                            txt,original_path=path,
                            original_sha=row["source_sha"],source_kind=row["kind"],
                        )
                        if body_chars!=row["text_characters"]:
                            report["failures"].append("source_map_body_character_count_mismatch")
                        else:
                            report["extracted_derivative_revision_verified"]=True
                            report["actual_derived_bytes"]=size
                            report["actual_derived_characters_with_generated_header"]=len(txt)
                            report["verified_current_source_unit_characters"]=body_chars
                except (OSError,ValueError,UnicodeError) as exc:
                    report["failures"].append("derived_revision_unverifiable:"+type(exc).__name__)
        sample.append(report)
    metadata_problems={k:{"count":len(v),"example_paths":v[:3]}
                       for k,v in sorted(problems.items())}
    summary={
        "curated_source_map_originals":len(rows),
        "catalog_entries":len(catalog),"ledger_entries":len(ledger),
        "filedb_all_file_entries":sum(map(len,native.values())),
        "source_states":dict(Counter(r["status"] for r in rows)),
        "source_types":dict(Counter(r["kind"] for r in rows)),
        "native_overview_originals":sum(r["overview_chunks"]>0 for r in rows),
        "native_summary_originals":sum(r["summary_chunks"]>0 for r in rows),
        "native_full_originals":sum(r["full_chunks"]>0 for r in rows),
        "metadata_problems":metadata_problems,
        "metadata_problem_categories":len(metadata_problems),
        "filedb_quick_check":integrity,
        "filedb_foreign_key_fault_count":len(foreign_errors),
        "deterministic_selected_originals":len(selected),
        "originals_independently_hashed":sum(r["original_file_revision_verified"] for r in sample),
        "current_derivatives_independently_hashed":sum(r["extracted_derivative_revision_verified"] for r in sample),
        "selected_originals_with_failures":sum(bool(r["failures"]) for r in sample),
        "sampled_bytes_read_at_least":bytes_hashed,
        "skipped_due_budgets":sum(bool(r["skips"]) for r in sample),
    }
    return {
        "schema":"kmd.source-revision-stratified-qa.v1",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "read_only":True,
        "full_metadata_reconciled_not_full_file_content_hashed":True,
        "scope":"One native FileDB and its full curated catalog/source-map/ledger; bounded deterministic publisher+derived byte checks",
        "seed":seed,
        "quotas":_QUOTAS if quotas is None else quotas,
        "max_original_sample_bytes":max_original_bytes,
        "max_derived_sample_bytes":max_derived_bytes,
        "total_sample_byte_budget":max_total_bytes,
        "summary":summary,
        "samples":sample,
        "limitations":[
            "Full metadata joins do not authenticate bytes of every one of the 17717 original files.",
            "Only the selected, non-skipped originals and converted derivatives have been rehashed byte for byte.",
            "Presence of native vectors does not establish factual truth, semantic search quality or correct publisher page mapping.",
            "The native database is live; concurrent new summary rows can change counts after the snapshot.",
            "An extracted source can be lossy; a verified derivative SHA proves the current extraction revision, not completeness of OCR or charts.",
        ],
    }
