"""Independent read-only linkage of All2Text PDF crops to canonical KMD.

Verifies source receipts via full original bytes, native FileDB identity,
source-map original SHA, current derived SHA and exact PDF page-unit caption
prefix. Never writes the original documents or KMD databases. A caption
prefix is source text evidence, not a statement of real-world truth.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import subprocess
import tempfile
from typing import Any

_REGION_SCHEMA = "all2text.verified_pdf_visual_regions.v1"
_CONTEXT_SCHEMA = "all2text.visual_context.v1"
_PAGE_RE = re.compile(r"\[\[SOURCE UNIT:\s*PDF page\s+([0-9]+)\]\]",re.I)
_HEX_SHA = re.compile(r"[a-f0-9]{64}")
_MAX_BYTES = 3 * 1024 * 1024
_MAX_PDF = 256 * 1024 * 1024


def _sha(path:Path)->str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def _safe_rel(path:str, suffix:str|None=None)->str:
    if not isinstance(path,str) or not 1<=len(path)<=1600:
        raise ValueError("invalid input relative path")
    p=PurePosixPath(path)
    if (p.is_absolute() or "\\" in path or
            any(seg in ("",".","..") for seg in path.split("/"))):
        raise ValueError("unsafe relative path")
    if suffix is not None and p.suffix.casefold()!=suffix:
        raise ValueError("wrong source extension")
    return p.as_posix()


def _read_limited_json(p:Path)->dict:
    if p.is_symlink() or not p.is_file() or p.stat().st_size>_MAX_BYTES:
        raise ValueError("untrusted missing/oversized/symlinked provenance JSON")
    x=json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(x,dict):
        raise ValueError("provenance JSON must be an object")
    return x


def safe_private_result_path(
    output: Path, *, source_root: Path, derived_root: Path,
    native_filedb: Path, source_map_db: Path,
) -> Path:
    """A new report MUST NOT enter protected original or native-data trees.

    KMD's canonical database parent may contain a separate approved
    search-audit child; writing a report directly into its database root or
    semantic/source-map tree is forbidden. Ancestor symlinks are resolved.
    """
    target=Path(output).expanduser()
    if target.exists() or target.is_symlink():
        raise ValueError("refuse to overwrite existing original-linked proof")
    if not target.parent.is_dir():
        raise ValueError("private output parent directory missing")
    parent=target.parent.resolve(strict=True)
    candidate=parent/target.name
    orig=Path(source_root).resolve(strict=True)
    derived=Path(derived_root).resolve(strict=True)
    filedb=Path(native_filedb).resolve(strict=True)
    source_map=Path(source_map_db).resolve(strict=True)
    if (parent.is_relative_to(orig)
            or parent.is_relative_to(derived)
            or candidate in (filedb,source_map)
            or parent==filedb.parent
            or parent==source_map.parent):
        raise ValueError("refuse writing source-linked proof inside protected original or native-data paths")
    # An operator may keep private evidence below the FileDB working root,
    # but only under its purpose-built search-audit subtree.
    if parent.is_relative_to(filedb.parent):
        audit=filedb.parent/"search-audit"
        if not parent.is_relative_to(audit):
            raise ValueError("proof inside native FileDB root requires private search-audit path")
    return candidate


def _inside(root:Path,relative:str)->Path:
    p=root/_safe_rel(relative)
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError("symlinked, missing or escaped source/derived file")
    return p


def _fold_with_positions(text:str)->tuple[str,list[int]]:
    """Collapsed whitespace/casefold text plus *exact original* char offsets."""
    folded=[]
    positions=[]
    for pos,ch in enumerate(text):
        if ch.isspace():
            if folded and folded[-1]!=" ":
                folded.append(" ")
                positions.append(pos)
        else:
            for part in ch.casefold():
                folded.append(part)
                positions.append(pos)
    return "".join(folded),positions


def _caption_anchor(derived:str,physical_page:int,caption:str)->dict:
    prefix=" ".join(caption.split()).casefold()[:96]
    if len(prefix)<24:
        return {"status":"insufficient_caption_text","evidence_scope":"none"}
    markers=list(_PAGE_RE.finditer(derived))
    pages=[(i,match) for i,match in enumerate(markers)
           if int(match.group(1))==physical_page]
    if len(pages)!=1:
        return {"status":"physical_page_marker_absent_or_ambiguous",
                "evidence_scope":"none"}
    i,marker=pages[0]
    begin=marker.end()
    end=markers[i+1].start() if i+1<len(markers) else len(derived)
    section=derived[begin:end]
    text,offsets=_fold_with_positions(section)
    occurrences=[match.start() for match in re.finditer(re.escape(prefix),text)]
    if len(occurrences)!=1:
        return {"status":"caption_prefix_missing_or_ambiguous_in_page",
                "occurrence_count":len(occurrences),"evidence_scope":"none"}
    start_index=occurrences[0]
    left=begin+offsets[start_index]
    right=begin+offsets[start_index+len(prefix)-1]+1
    span=derived[left:right]
    assert _fold_with_positions(span)[0]==prefix
    return {
        "status":"verified_original_derived_page_caption_prefix",
        "evidence_scope":"exact_current_extracted_text_span_not_original_publisher_bytes",
        "physical_page_source_unit":physical_page,
        "derived_start_char":left,
        "derived_end_char":right,
        "derived_prefix_sha256":hashlib.sha256(span.encode("utf-8")).hexdigest(),
        "normalized_prefix":prefix,
        "matched_characters":len(prefix),
        "warning":"This matches text in the original-linked derived PDF page unit; it does not prove the caption explains the entire figure or establish literal PDF publisher bytes.",
    }


def _rerender_original_page(
    original:Path,number:int,expected_raster_sha:str,max_edge:int,
)->bool:
    """Independent Poppler image SHA check; no original/DB writes."""
    if not 500<=max_edge<=1600:
        raise ValueError("unbounded original page pixel edge")
    with tempfile.TemporaryDirectory(prefix="kmd-pdf-page-proof-") as tmp:
        prefix=Path(tmp)/"render"
        args=["pdftoppm","-f",str(number),"-l",str(number),
              "-singlefile","-scale-to",str(max_edge),"-png",
              str(original),str(prefix)]
        try:
            run=subprocess.run(args,timeout=45,capture_output=True)
        except (FileNotFoundError,subprocess.TimeoutExpired) as exc:
            raise ValueError("independent physical-page renderer unavailable") from exc
        out=prefix.with_suffix(".png")
        if run.returncode!=0 or not out.is_file():
            raise ValueError("independent physical PDF page renderer failed")
        return _sha(out)==expected_raster_sha


def link_pdf_regions(
    *,region_manifest:Path,context_manifest:Path,
    source_root:Path,source_map_db:Path,filedb:Path,
    derived_root:Path,independently_rerender:bool=True,
)->dict[str,Any]:
    from PIL import Image
    region_manifest=Path(region_manifest)
    context_manifest=Path(context_manifest)
    region_dir=region_manifest.parent
    report=_read_limited_json(region_manifest)
    context=_read_limited_json(context_manifest)
    if report.get("schema")!=_REGION_SCHEMA or context.get("schema")!=_CONTEXT_SCHEMA:
        raise ValueError("unknown original region/context schema")
    info=report.get("original")
    regions=report.get("regions")
    pages=report.get("pages")
    if (not isinstance(info,dict) or not isinstance(regions,list)
            or not isinstance(pages,list) or len(regions)>100
            or not 1<=len(pages)<=4):
        raise ValueError("invalid bounded source region payload")
    relative=_safe_rel(info.get("relative_path"),".pdf")
    original=_inside(Path(source_root).resolve(),relative)
    original_sha=info.get("source_sha256")
    if (not isinstance(original_sha,str) or not _HEX_SHA.fullmatch(original_sha)
            or original.stat().st_size>_MAX_PDF or _sha(original)!=original_sha
            or original_sha!=info.get("download_receipt_sha256")):
        raise ValueError("publisher original revision does not match bounded provenance receipt")
    if not isinstance(context.get("entries"),dict):
        raise ValueError("invalid source context entries")
    root=Path(derived_root).resolve()
    src=sqlite3.connect(Path(source_map_db).resolve().as_uri()+"?mode=ro",uri=True)
    native=sqlite3.connect(Path(filedb).resolve().as_uri()+"?mode=ro",uri=True)
    try:
        src.execute("PRAGMA query_only=ON")
        native.execute("PRAGMA query_only=ON")
        mapped=src.execute(
            "SELECT status,source_sha256,derived_sha256,derived_path,error "
            "FROM originals WHERE source_path=?",(relative,),
        ).fetchall()
        ids=native.execute(
            "SELECT id FROM filesystem_entries "
            "WHERE relative_path_display=? AND entry_type='file'",
            (relative,),
        ).fetchall()
        if len(mapped)!=1 or len(ids)!=1:
            raise ValueError("canonical source-map or FileDB identity not unique")
        indexed_collections=[
            {"collection_id":str(name),"chunk_count":int(count),
             "embedded_chunk_count":int(embedded or 0)}
            for name,count,embedded in native.execute(
                "SELECT collection_id,count(*),sum(embedding_blob is not null) "
                "FROM content_chunks WHERE filesystem_entry_id=? "
                "GROUP BY collection_id ORDER BY collection_id",
                (ids[0][0],),
            ).fetchall()
        ]
        status,map_sha,derived_sha,derived_path,_error=mapped[0]
        if map_sha!=original_sha:
            raise ValueError("source-map original revision differs from publisher original")
        derived=None
        if status=="complete":
            if (not isinstance(derived_sha,str) or not _HEX_SHA.fullmatch(derived_sha)
                    or not isinstance(derived_path,str)):
                raise ValueError("completed source-map lacks a verified derivative")
            derived_file=_inside(root,derived_path)
            if _sha(derived_file)!=derived_sha:
                raise ValueError("derived source SHA changed since source-map completion")
            derived=derived_file.read_text(encoding="utf-8")
        elif status=="error":
            derived_sha=None
        else:
            raise ValueError("unrecognized source-map status for original")
    finally:
        src.close()
        native.close()

    pages_by_num={}
    for page in pages:
        n=page.get("physical_pdf_page")
        if type(n) is not int or n<1 or n>20000 or n in pages_by_num:
            raise ValueError("invalid physical page inventory")
        pages_by_num[n]=page
    page_render_cache={}
    results=[]
    seen=set()
    for item in regions:
        if not isinstance(item,dict):
            raise ValueError("invalid region row")
        region_rel=_safe_rel(item.get("region_path"),".png")
        number=item.get("physical_pdf_page")
        if (type(number) is not int or number not in pages_by_num
                or item.get("source_relative_path")!=relative
                or item.get("original_pdf_sha256")!=original_sha
                or region_rel in seen):
            raise ValueError("source/page/crop identity mismatch")
        seen.add(region_rel)
        cropped=_inside(region_dir,region_rel)
        crop_sha=item.get("region_png_sha256")
        if not isinstance(crop_sha,str) or not _HEX_SHA.fullmatch(crop_sha) or _sha(cropped)!=crop_sha:
            raise ValueError("region crop hash invalid")
        ctx=context["entries"].get(cropped.name)
        if (not isinstance(ctx,dict) or ctx.get("image_sha256")!=crop_sha
                or ctx.get("parent_pdf_sha256")!=original_sha
                or ctx.get("physical_page")!=number
                or ctx.get("region_role")!=item.get("region_role")):
            raise ValueError("image context refers to another original/page/region")
        pagefile=_inside(region_dir,item.get("rendered_page_path"))
        ps=item.get("rendered_page_sha256")
        if not isinstance(ps,str) or _sha(pagefile)!=ps:
            raise ValueError("stored page raster SHA does not match recorded page")
        from PIL import Image
        with Image.open(pagefile) as pageimg,Image.open(cropped) as cropimg:
            pageimg.load();cropimg.load()
            bbox=item.get("raster_region_pixel_bbox_top_left")
            if (not isinstance(bbox,list) or len(bbox)!=4 or
                    any(type(v) is not int for v in bbox) or
                    not (0<=bbox[0]<bbox[2]<=pageimg.width and
                         0<=bbox[1]<bbox[3]<=pageimg.height)):
                raise ValueError("invalid physical-page raster pixel rectangle")
            if pageimg.crop(tuple(bbox)).tobytes()!=cropimg.tobytes():
                raise ValueError("crop pixels differ from recorded parent page rectangle")
            raster_edge=max(pageimg.size)
        if independently_rerender:
            if number not in page_render_cache:
                page_render_cache[number]=_rerender_original_page(
                    original,number,ps,raster_edge)
            if not page_render_cache[number]:
                raise ValueError("independent physical PDF page raster SHA did not reproduce")
        caption=item.get("caption_text_for_prompt") or ""
        if not isinstance(caption,str):
            raise ValueError("invalid caption text")
        if caption:
            expected_preview=item.get("caption_preview_sha256")
            if hashlib.sha256(caption.encode("utf-8")).hexdigest()!=expected_preview:
                raise ValueError("caption preview SHA mismatch")
        anchor=(
            _caption_anchor(derived,number,caption)
            if derived is not None and caption else
            {"status":"no_completed_text_extraction" if derived is None else "no_caption_in_region",
             "evidence_scope":"none"}
        )
        results.append({
            "native_filesystem_entry_id":ids[0][0],
            "source_map_status":status,
            "source_original_sha256":original_sha,
            "source_derived_sha256":derived_sha,
            "region_path":region_rel,
            "physical_pdf_page":number,
            "raster_sha256":crop_sha,
            "independent_physical_pdf_page_raster_verified":independently_rerender,
            "region_role":item["region_role"],
            "caption_anchor":anchor,
        })
    if len(seen)!=len(context["entries"]):
        raise ValueError("context count differs from audited crop region set")
    if _sha(original)!=original_sha:
        raise ValueError("publisher PDF changed during linkage proof")
    return {
        "schema":"kmd.original_linked_visual_regions.v1",
        "source_relative_path":relative,
        "source_original_sha256":original_sha,
        "native_filesystem_entry_id":ids[0][0],
        "source_map_status":status,
        "native_indexed_collections":indexed_collections,
        "current_derived_sha256":derived_sha,
        "region_manifest_sha256":_sha(region_manifest),
        "visual_context_manifest_sha256":_sha(context_manifest),
        "caption_proof_limitation":"Only exact text from current derived PDF page-unit, not publisher-byte/caption semantics.",
        "physical_page_proof":"Independent rerender plus source original SHA, pixel rectangle/crop SHA" if independently_rerender else
                              "Saved page crop only; independent original-page rerender NOT performed",
        "no_originals_or_databases_modified":True,
        "regions":results,
    }
