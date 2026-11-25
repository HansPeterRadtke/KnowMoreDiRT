"""Independent bounded discovery across a short PDF or an explicit physical-page range.

Whole-document Poppler text is a candidate locator only. A selected page is
read again via the independently verified current-original physical-page
operator before any candidate text is returned. No derivative or native index
is changed and no physical page is inferred from a derivative label.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import time

from .hybrid_search import lexical_token_spans, query_terms
from .publisher_pdf_page import _poppler
from .reading_order_page_passages import (
    _numeric_keys, _page_context, _porter_tokens, _rank_normalized_reading_order_text,
)
from .verified_pdf_neighbor_context import PhysicalPdfPageContextUnavailable, PAGE_COUNT

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_BAD_GLYPH = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\ufffd]")
MAX_ORIGINAL_BYTES = 32 * 1024 * 1024
MAX_PAGES = 64
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_PAGE_CHARS = 18000
MAX_PHYSICAL_PAGE_NUMBER = 999999


class BoundedPdfDiscoveryUnavailable(ValueError):
    """The requested source scope cannot be examined within the explicit bounds."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_digest(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("missing expected original revision digest")


def validate_explicit_page_range(first_page, last_page):
    """A caller-provided physical range is explicit, bounded and never clamped."""
    if (type(first_page) is not int or type(last_page) is not int
            or not 1<=first_page<=last_page<=MAX_PHYSICAL_PAGE_NUMBER
            or last_page-first_page+1>MAX_PAGES):
        raise ValueError("both exact physical page bounds required; maximum 64 pages per range")


def _scope_bounds(record):
    """Return actual total, scanned first/last, and explicit-range mode."""
    if not isinstance(record,dict):
        raise ValueError("missing physical-page scan metadata")
    total=record.get("physical_page_count")
    if type(total) is not int or not 1<=total<=MAX_PHYSICAL_PAGE_NUMBER:
        raise ValueError("invalid actual physical PDF page count")
    fields=("scan_scope","scanned_first_physical_page","scanned_last_physical_page")
    present=[k in record for k in fields]
    if not any(present):
        if total>MAX_PAGES:
            raise ValueError("full scan exceeds 64 physical pages")
        return total,1,total,False
    if not all(present) or record["scan_scope"]!="explicit_physical_page_range":
        raise ValueError("incomplete explicit physical-page scope")
    first=record["scanned_first_physical_page"];last=record["scanned_last_physical_page"]
    validate_explicit_page_range(first,last)
    if last>total:
        raise ValueError("requested physical page range exceeds actual PDF page count")
    return total,first,last,True


def _range_scope(first,last):
    return {"scan_scope":"explicit_physical_page_range",
            "scanned_first_physical_page":first,"scanned_last_physical_page":last}


def split_complete_reading_order_pages(raw: bytes, page_count: int) -> list[dict]:
    """Accept exactly one form-feed terminator per physical page.

    The candidate page byte hashes include that terminator; do not silently
    guess page boundaries or omit a missing physical page.
    """
    if (not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_TEXT_BYTES
            or type(page_count) is not int or not 1 <= page_count <= MAX_PAGES):
        raise BoundedPdfDiscoveryUnavailable("complete PDF text exceeds explicit source bounds")
    parts = raw.split(b"\f")
    if len(parts) != page_count+1 or parts[-1] != b"":
        raise BoundedPdfDiscoveryUnavailable("complete PDF extraction did not preserve exact physical-page separators")
    return [_page_record(part+b"\f",page) for page,part in enumerate(parts[:-1],1)]


def _page_record(data: bytes, page: int) -> dict:
    entry={"physical_page":page,"reading_order_page_sha256":_sha(data)}
    try:
        body=data.decode("utf-8","strict")
    except UnicodeDecodeError:
        return {**entry,"status":"invalid_utf8","text":None}
    if _BAD_GLYPH.search(body):
        return {**entry,"status":"invalid_glyphs","text":None}
    if len(body)>MAX_PAGE_CHARS:
        return {**entry,"status":"page_text_exceeds_bound","text":None}
    if not body.strip():
        return {**entry,"status":"page_text_empty","text":None}
    return {**entry,"status":"usable_candidate","text":body}


def _read_explicit_physical_pages(name: str, fd: int, first: int, last: int, deadline: float):
    """Recover embedded form feeds using actual single-page requests, never splitting them."""
    pages=[];total=0;digest=hashlib.sha256()
    for page in range(first,last+1):
        raw=_poppler(["/usr/bin/pdftotext","-f",str(page),"-l",str(page),
                      "-enc","UTF-8",name,"-"],fd,deadline)
        if raw is None or not raw.endswith(b"\f"):
            raise BoundedPdfDiscoveryUnavailable("explicit physical-page extraction unavailable or missing terminator")
        total+=len(raw)
        if total>MAX_TEXT_BYTES:
            raise BoundedPdfDiscoveryUnavailable("explicit physical-page text exceeds complete-source bound")
        digest.update(raw)
        pages.append(_page_record(raw,page))
    return pages,digest.hexdigest()


def _read_original_pdf_pages(pdf: Path, expected_sha256: str, *, first_page=None, last_page=None) -> dict:
    """Open/hash the exact same original file description read by Poppler."""
    explicit=first_page is not None or last_page is not None
    if explicit:
        validate_explicit_page_range(first_page,last_page)
    _require_digest(expected_sha256)
    if not isinstance(pdf, Path) or pdf.suffix.casefold() != ".pdf":
        raise ValueError("one approved PDF file is required")
    with os.fdopen(os.open(pdf, os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("original PDF is not a regular file")
        if not 0 < before.st_size <= MAX_ORIGINAL_BYTES:
            raise BoundedPdfDiscoveryUnavailable("original PDF exceeds 32 MiB whole-document discovery bound")
        if hashlib.file_digest(stream,"sha256").hexdigest() != expected_sha256:
            raise ValueError("publisher original changed before complete page discovery")
        fd=stream.fileno();name=f"/proc/self/fd/{fd}"
        deadline=time.monotonic()+40
        info=_poppler(["/usr/bin/pdfinfo", name],fd,deadline)
        counts=PAGE_COUNT.findall(info or b"")
        if len(counts)!=1 or not 1<=int(counts[0])<=MAX_PHYSICAL_PAGE_NUMBER:
            raise BoundedPdfDiscoveryUnavailable("actual physical PDF page count unavailable")
        count=int(counts[0])
        if explicit:
            if last_page>count:
                raise ValueError("requested physical page range exceeds actual PDF page count")
            first,last=first_page,last_page
        else:
            if count>MAX_PAGES:
                raise BoundedPdfDiscoveryUnavailable("physical PDF page count above the 64-page whole-document bound")
            first,last=1,count
        scanned_count=last-first+1
        raw=_poppler(["/usr/bin/pdftotext","-f",str(first),"-l",str(last),
                      "-enc","UTF-8",name,"-"],fd,deadline)
        if raw is None:
            raise BoundedPdfDiscoveryUnavailable("complete bounded PDF text extraction unavailable")
        if not 0<len(raw)<=MAX_TEXT_BYTES:
            raise BoundedPdfDiscoveryUnavailable("complete PDF text exceeds explicit source bounds")
        if raw.count(b"\f")==scanned_count and raw.endswith(b"\f"):
            pages=split_complete_reading_order_pages(raw,scanned_count)
            if explicit:
                for page in pages:
                    page["physical_page"]+=first-1
            digest=_sha(raw)
            extraction_method=("explicit_range_exact_form_feed_boundaries" if explicit
                               else "complete_pdf_exact_form_feed_boundaries")
        else:
            # An embedded form feed can be ordinary extracted whitespace,
            # not a physical page boundary. Re-read each numbered page.
            pages,digest=_read_explicit_physical_pages(name,fd,first,last,deadline)
            extraction_method="explicit_numbered_physical_page_reads_after_separator_mismatch"
        after=os.stat(pdf,follow_symlinks=False)
        if (not os.path.samestat(before,after) or before.st_size!=after.st_size
                or before.st_mtime_ns!=after.st_mtime_ns or before.st_ctime_ns!=after.st_ctime_ns):
            raise ValueError("publisher PDF changed during complete page discovery")
    return {"original_sha256":expected_sha256,"physical_page_count":count,
            **({**_range_scope(first,last),"range_extraction_sha256":digest} if explicit
               else {"complete_extraction_sha256":digest}),
            "extraction_method":extraction_method,"pages":pages}


def read_complete_original_pdf_pages(pdf: Path, expected_sha256: str) -> dict:
    """Full scan retains the existing strict short-document bound."""
    return _read_original_pdf_pages(pdf,expected_sha256)


def read_original_pdf_page_range(pdf: Path, expected_sha256: str, *, first_page:int,last_page:int) -> dict:
    """Scan every page of exactly the requested range, not the rest of the book."""
    validate_explicit_page_range(first_page,last_page)
    return _read_original_pdf_pages(pdf,expected_sha256,first_page=first_page,last_page=last_page)


def rank_document_page_candidates(scan: dict, query: str, *, top_pages: int=5,
                                  snippet_characters: int=900) -> dict:
    """Compare local sentence-window term coverage using document-wide weights.

    The candidates remain private machine text until the separate current-page
    verification below. Page-local rank scores are not compared as if their
    differently estimated term frequencies were calibrated across pages.
    """
    if (not isinstance(query,str) or not 1<=len(query.strip())<=600
            or type(top_pages) is not int or not 1<=top_pages<=5
            or type(snippet_characters) is not int or not 160<=snippet_characters<=2000):
        raise ValueError("invalid bounded original document passage query")
    total_pages,first,last,explicit=_scope_bounds(scan)
    if not isinstance(scan.get("pages"),list) or len(scan["pages"])!=last-first+1:
        raise ValueError("complete requested physical-page scan required")
    _require_digest(scan.get("original_sha256"))
    _require_digest(scan.get("range_extraction_sha256" if explicit else "complete_extraction_sha256"))
    if explicit and "complete_extraction_sha256" in scan:
        raise ValueError("explicit range must not claim a full-source extraction digest")
    terms={t for t in query_terms(query) if not t.isdecimal()}
    numbers=_numeric_keys(query)
    if not terms and not numbers:
        raise ValueError("no informative document search terms")
    good=[];unavailable=[];total=0;all_words=set(terms)
    for page,entry in enumerate(scan["pages"],first):
        if not isinstance(entry,dict) or type(entry.get("physical_page")) is not int or entry["physical_page"]!=page:
            raise ValueError("physical-page sequence changed")
        _require_digest(entry.get("reading_order_page_sha256"))
        text=entry.get("text")
        if entry.get("status")!="usable_candidate":
            if text is not None or entry.get("status") not in (
                    "invalid_utf8","invalid_glyphs","page_text_exceeds_bound","page_text_empty"):
                raise ValueError("invalid unavailable physical-page record")
            unavailable.append({"physical_page":page,"status":entry["status"]})
            continue
        if (not isinstance(text,str) or not text.strip() or len(text)>MAX_PAGE_CHARS
                or _BAD_GLYPH.search(text)
                or _sha(text.encode("utf-8"))!=entry["reading_order_page_sha256"]):
            raise ValueError("machine page text differs from exact scan identity")
        total+=len(text.encode("utf-8"))
        if total>MAX_TEXT_BYTES:
            raise ValueError("candidate document text exceeds bounds")
        normalized=" ".join(text.split())
        words={t for t,_,_ in lexical_token_spans(normalized)}
        all_words.update(words)
        good.append((entry,normalized,words))
    stems=_porter_tokens(all_words)
    qstems={}
    for t in terms:
        if t in stems:qstems.setdefault(stems[t],set()).add(t)
    freq=Counter()
    for _,_,words in good:
        freq.update((terms&words)|{t for w in words if w in stems for t in qstems.get(stems[w],())})
    weights={t:1+math.log((1+len(good))/(1+freq[t])) for t in terms}
    candidates=[]
    for entry,body,_ in good:
        result=_rank_normalized_reading_order_text(body,query,top=3,
                max_characters=snippet_characters,include_inflection_matches=True,term_weights=weights)
        if not result["passages"]:continue
        w=result["passages"][0]
        exact=w["core_matched_query_terms"];morph=w["core_matched_query_inflection_terms"]
        nums=w["core_matched_query_number_literals"]
        score=(round(sum(weights[t] for t in exact)+.55*sum(weights[t] for t in morph)+2*len(nums),9),
               len(nums),len(exact),-(w["core_end_char"]-w["core_start_char"]),-entry["physical_page"])
        candidates.append((score,entry,body,result["passages"]))
    candidates.sort(key=lambda x:x[0],reverse=True)
    return {"physical_page_count":total_pages,
            **({**_range_scope(first,last),"scanned_physical_page_count":last-first+1,
                "physical_pages_outside_requested_range":total_pages-(last-first+1)} if explicit else {}),
            "usable_machine_text_pages":len(good),"unavailable_pages":unavailable,
            "matching_candidate_pages":len(candidates),"top_pages":top_pages,
            "max_characters_per_window":snippet_characters,
            "term_weights":weights,
            "private_candidates":[{"page_candidate_rank":i,"physical_page":e["physical_page"],
                 "reading_order_page_sha256":e["reading_order_page_sha256"],
                 "normalized_reading_order_page_sha256":_sha(body.encode("utf-8")),
                 "page_candidate_score":score[0],"windows":windows}
                 for i,(score,e,body,windows) in enumerate(candidates[:top_pages],1)]}


def verify_selected_document_pages(ranked:dict,*,source:dict,read_context) -> dict:
    """Independently reread the selected exact physical pages; never substitute."""
    if (not isinstance(ranked,dict) or not isinstance(ranked.get("private_candidates"),list)
            or not 0<=len(ranked["private_candidates"])<=5
            or not isinstance(source,dict)):
        raise ValueError("missing bounded original document candidates")
    out=[];seen=set()
    count,first,last,explicit=_scope_bounds(ranked)
    limit=ranked.get("max_characters_per_window")
    if type(limit) is not int or not 160<=limit<=2000:
        raise ValueError("invalid bounded physical page window size")
    if explicit and (type(ranked.get("scanned_physical_page_count")) is not int
            or ranked["scanned_physical_page_count"]!=last-first+1
            or type(ranked.get("physical_pages_outside_requested_range")) is not int
            or ranked["physical_pages_outside_requested_range"]!=count-(last-first+1)):
        raise ValueError("inconsistent explicit physical-page coverage")
    for rank,candidate in enumerate(ranked["private_candidates"],1):
        if (not isinstance(candidate,dict) or type(candidate.get("page_candidate_rank")) is not int
                or candidate["page_candidate_rank"]!=rank
                or type(candidate.get("physical_page")) is not int
                or not first<=candidate["physical_page"]<=last
                or candidate["physical_page"] in seen):
            raise ValueError("invalid duplicate or reordered document page candidate")
        _require_digest(candidate.get("reading_order_page_sha256"))
        _require_digest(candidate.get("normalized_reading_order_page_sha256"))
        page=candidate["physical_page"];seen.add(page)
        public={k:v for k,v in candidate.items() if k!="windows"}
        try:
            ctx=read_context(source["original_path"],page)
        except PhysicalPdfPageContextUnavailable:
            out.append({**public,"status":"physical_page_context_unavailable","windows":[]})
            continue
        # An explicit bounded alternate extraction failure is not an
        # affirmative absence of evidence. Preserve the slot with no text.
        block=ctx.get("physical_pages",{}) if isinstance(ctx,dict) else {}
        alt=block.get("optional_center_reading_order_projection",{})
        current=ctx.get("source",{}) if isinstance(ctx,dict) else {}
        if (current.get("original_path")!=source["original_path"]
                or current.get("original_sha256")!=source["original_sha256"]
                or current.get("current_derived_sha256")!=source["current_derived_sha256"]
                or block.get("requested_center_physical_page")!=page
                or block.get("actual_original_pdf_page_count")!=count
                or ctx.get("source_map_revision_reverified_after_physical_page_extraction") is not True
                or current.get("publisher_original_actual_sha256_verified") is not True
                or current.get("source_map_derivative_actual_sha256_verified") is not True
                or block.get("original_pdf_revision_verified_before_and_after") is not True):
            raise ValueError("original source revision or physical-page identity changed")
        if (isinstance(alt,dict) and alt.get("status")=="center_reading_order_text_unavailable"
                and alt.get("extracted_page_text") is None):
            out.append({**public,"status":"reading_order_context_unavailable","windows":[]})
            continue
        _,actual_page,body,layout_sha,reading_sha=_page_context(ctx)
        if (actual_page!=page or reading_sha!=candidate["reading_order_page_sha256"]
                or _sha(body.encode("utf-8"))!=candidate["normalized_reading_order_page_sha256"]):
            raise ValueError("whole-document machine page differs from independent original page read")
        windows=candidate["windows"]
        if not isinstance(windows,list) or not 1<=len(windows)<=3:
            raise ValueError("invalid bounded document window count")
        for rank,w in enumerate(windows,1):
            s,e=w.get("start_char"),w.get("end_char")
            if (type(w.get("rank")) is not int or w["rank"]!=rank or type(s) is not int or type(e) is not int
                    or not 0<=s<e<=len(body) or e-s>limit or body[s:e]!=w.get("text")
                    or _sha(body[s:e].encode("utf-8"))!=w.get("span_sha256")):
                raise ValueError("whole-document candidate window failed exact independent page offset proof")
        out.append({**public,"status":"verified_reading_order_windows","layout_page_sha256":layout_sha,
                    "offset_basis":"whitespace-normalized Poppler default reading order on one physical page",
                    "windows":windows})
    return {**{k:v for k,v in ranked.items() if k not in ("private_candidates","term_weights")},
            "pages":out,"pages_with_verified_reading_order_windows":sum(
                p["status"]=="verified_reading_order_windows" for p in out),
            "independent_original_physical_page_reread_required":True,
            "visual_reading_order_or_factual_claim_truth_verified":False,
            "native_document_ranking_unchanged":True,
            "no_cross_page_joined_quotation":True,
            "physical_pages_outside_complete_scan_not_inferred":True}
