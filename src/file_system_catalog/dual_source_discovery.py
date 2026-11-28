"""Two independently ranked, source-revision-checked native KMD search channels.

The generated-summary ranking and original extracted-passage ranking are
reported separately. No fusion or confidence threshold is trained/applied.
This is a read-only source discovery helper; verified current *derivative*
text is not automatically the correct original physical PDF page or a
complete factual answer. No second corpus/index/database is created.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import sqlite3
from typing import Any

from .publisher_pdf_page import verify_pdf_page_excerpts


MAX_PASSAGE_BYTES = 131072
READ_BLOCK = 1024 * 1024


def safe_corpus_relative(path: str) -> str:
    if (not isinstance(path, str) or not 1 <= len(path) <= 1600
            or path.startswith("/") or "\\" in path
            or any(part in ("", ".", "..") for part in path.split("/"))):
        raise ValueError("unsafe relative publisher source path")
    return PurePosixPath(path).as_posix()


def existing_file(root: Path, relative: str) -> Path:
    path = root / safe_corpus_relative(relative)
    if (path.is_symlink() or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError("publisher/derivative path is missing, symlinked or escapes root")
    return path


def _hash_file_with_passage(path: Path, needle: bytes | None) -> tuple[str, bool | None]:
    """Rehash complete derivative and match only substantive source-unit bytes.

    Extracted text begins with generated catalog metadata and a source unit.
    Prior code began the searchable region at the opening marker, so it
    incorrectly accepted an extractor-written PDF page label as original
    publisher text. The same problem recurred on subsequent page labels.
    Search only text between complete markers, preserving bounded overlap
    across read boundaries, never across separate source units. A malformed
    or oversized marker is not publisher evidence.
    """
    if needle is not None and (not isinstance(needle, bytes) or not needle):
        raise ValueError("source passage proof requires nonempty bytes")
    digest = hashlib.sha256()
    if needle is None:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(READ_BLOCK), b""):
                digest.update(chunk)
        return digest.hexdigest(), True

    source_marker = b"[[SOURCE UNIT:"
    marker_end = b"]]"
    max_marker_label_bytes = 512
    state = "before_first_source"
    pending = b""
    publisher_overlap = b""
    found = False
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(READ_BLOCK), b""):
            digest.update(chunk)
            if found:
                continue  # Even after a match, verify the FULL derivative SHA.
            block = pending + chunk
            pending = b""
            while block:
                if state == "before_first_source":
                    at = block.find(source_marker)
                    if at < 0:
                        pending = block[-(len(source_marker) - 1):]
                        break
                    block = block[at + len(source_marker):]
                    state = "source_unit_label"
                    continue
                if state == "source_unit_label":
                    close = block.find(marker_end)
                    if close < 0:
                        if len(block) > max_marker_label_bytes:
                            raise ValueError("oversized or unclosed source-unit label")
                        pending = block
                        break
                    if close > max_marker_label_bytes:
                        raise ValueError("oversized source-unit label")
                    block = block[close + len(marker_end):]
                    state = "publisher_source"
                    publisher_overlap = b""  # Never join two source units.
                    continue

                # Only original source body bytes may prove a passage.
                at = block.find(source_marker)
                if at >= 0:
                    body = block[:at]
                    if needle in publisher_overlap + body:
                        found = True
                        break
                    block = block[at + len(source_marker):]
                    state = "source_unit_label"
                    publisher_overlap = b""
                    continue
                # Withhold the possible prefix of a marker at a read boundary.
                keep = len(source_marker) - 1
                if len(block) <= keep:
                    pending = block
                    break
                body = block[:-keep]
                combined = publisher_overlap + body
                if needle in combined:
                    found = True
                    break
                publisher_overlap = combined[-(len(needle) - 1):] if len(needle) > 1 else b""
                pending = block[-keep:]
                break
    if not found and state == "publisher_source" and pending:
        # Omit an unfinished marker prefix even at EOF.
        excluded = 0
        for size in range(min(len(pending), len(source_marker) - 1), 0, -1):
            if pending.endswith(source_marker[:size]):
                excluded = size
                break
        valid_tail = pending[:-excluded] if excluded else pending
        found = needle in publisher_overlap + valid_tail
    return digest.hexdigest(), found


def verify_current_source(
    *, path: str, expected_sha256: str | None,
    raw_passage: str | None,
    raw_passage_sha256: str | None,
    source_map: sqlite3.Connection,
    original_root: Path, derivative_root: Path,
    original_receipts: dict[str, str],
) -> dict[str, Any]:
    """Rehash publisher original and current derivative; no data modification.

    Raw passages must have already passed the native KMD reader's exact
    chunk-position SHA check. We additionally require their text to occur
    in the current full-derived-text bytes. The dual check is insufficient
    to prove a physical PDF page or that context outside this excerpt
    does not qualify or contradict its meaning.
    """
    path = safe_corpus_relative(path)
    rows = source_map.execute(
        "SELECT status,source_sha256,derived_sha256,derived_path "
        "FROM originals WHERE source_path=?", (path,),
    ).fetchall()
    if len(rows) != 1 or rows[0][0] != "complete":
        raise ValueError("source-map original absent, ambiguous or incomplete")
    _, original_sha, derived_sha, derived_rel = rows[0]
    if (not isinstance(original_sha,str) or
            not re.fullmatch("[0-9a-f]{64}",original_sha) or
            not isinstance(derived_sha,str) or
            not re.fullmatch("[0-9a-f]{64}",derived_sha) or
            original_receipts.get(path) != original_sha or
            (expected_sha256 is not None and expected_sha256 != original_sha)):
        raise ValueError("publisher original revision differs from completed receipt or search result")
    original = existing_file(original_root,path)
    derived = existing_file(derivative_root,derived_rel)
    if raw_passage is not None:
        if (not isinstance(raw_passage,str) or not raw_passage
                or not isinstance(raw_passage_sha256,str)
                or not re.fullmatch("[0-9a-f]{64}",raw_passage_sha256)):
            raise ValueError("invalid exact raw passage proof")
        passage = raw_passage.encode("utf-8")
        if (len(passage) > MAX_PASSAGE_BYTES or
                hashlib.sha256(passage).hexdigest() != raw_passage_sha256):
            raise ValueError("raw passage differs from verified chunk digest")
    else:
        passage=None
    actual_original, _ = _hash_file_with_passage(original,None)
    actual_derivative, present = _hash_file_with_passage(derived,passage)
    if actual_original != original_sha or actual_derivative != derived_sha:
        raise ValueError("publisher original or extracted source bytes changed")
    if not present:
        raise ValueError("raw passage absent from current publisher-source section of derivative")
    return {
        "source_original_sha256":original_sha,
        "current_derivative_sha256":derived_sha,
        "original_full_file_sha256_rechecked":True,
        "derivative_full_file_sha256_rechecked":True,
        "raw_passage_sha256_rechecked":raw_passage is not None,
        "raw_passage_present_in_current_derivative":present if raw_passage is not None else None,
        "scope":"Both source revisions verified; PDF physical-page geometry/complete discourse must be checked separately.",
    }



_DERIVED_SOURCE_UNIT = re.compile(r"\[\[SOURCE UNIT: ([^\]\r\n]{1,120})\]\]")
_PHYSICAL_PDF_LABEL = re.compile(r"PDF page ([1-9][0-9]{0,5})$")


def publisher_text_preview(full_passage: str, source_section: str,
                           *, max_chars: int = 350) -> tuple[str,str]:
    """Quote source-unit text, not KMD metadata; avoid spanning PDF pages.

    Native extracted chunks may begin with a generated `[[SOURCE UNIT:]]`
    delimiter or straddle two independently extracted PDF pages. Those
    delimiters are NOT printed publisher text. Prefer the first substantive
    segment, preserving an explicit physical page label if its marker is
    encountered, and never pretend the displayed preview crosses pages.
    The full passage's native position and SHA are verified separately.
    """
    if (not isinstance(full_passage,str) or not full_passage
            or not isinstance(source_section,str)
            or type(max_chars) is not int or not 30<=max_chars<=350):
        raise ValueError('invalid source-derived passage preview')
    delimiters=list(_DERIVED_SOURCE_UNIT.finditer(full_passage))
    segments=[]
    previous_end=0
    label=source_section
    for match in delimiters:
        part=full_passage[previous_end:match.start()].strip()
        if part:
            segments.append((part,label))
        next_label=match.group(1)
        if _PHYSICAL_PDF_LABEL.fullmatch(next_label):
            label=next_label
        else:
            label=source_section  # Non-PDF source-unit labels are still context.
        previous_end=match.end()
    final=full_passage[previous_end:].strip()
    if final:
        segments.append((final,label))
    if not segments:
        raise ValueError('source passage contains only extractor metadata')
    # A page-crossing window may start with only an ending phrase from the
    # preceding page; require a modest substantive excerpt when possible.
    selected=next(((text,section) for text,section in segments if len(text)>=80),segments[0])
    preview=selected[0][:max_chars].rstrip()
    if not preview or preview not in full_passage:
        raise ValueError('source preview is not a literal verified derivative span')
    return preview,selected[1]


def scope_explicit_publisher_query(summary_payload: dict, query: str) -> tuple[dict,str | None,bool,dict | None]:
    """Respect exact publisher identity in BOTH independent search channels.

    Native KMD summary search may resolve an explicit catalog title while the
    independently called original-passage program would otherwise execute
    a global vector lookup with misleading different originals. Enforce the
    known publisher path in both channels; never reinterpret an ambiguous
    or absent catalog source as a different paper. Ordinary topic searches
    continue unchanged. This routes candidates, not factual answers.
    """
    if (not isinstance(summary_payload,dict) or summary_payload.get('query')!=query
            or not isinstance(query,str) or not query):
        raise ValueError('source selection must use identical native query')
    match=summary_payload.get('source_name_match')
    if match is None:
        return summary_payload,None,True,None
    if not isinstance(match,dict):
        raise ValueError('invalid native source-name identity contract')
    status=match.get('status')
    if status not in ('verified','no_verified_index_match','ambiguous','no_catalog_match'):
        raise ValueError('unrecognized explicit original selection state')
    selection={'status':status,'kind':match.get('kind'),
               'policy':'Explicit original identity must not be replaced by unrelated ranked sources.'}
    docs=summary_payload.get('documents')
    if not isinstance(docs,list):
        raise ValueError('native summary channel missing source documents')
    if status in ('ambiguous','no_catalog_match'):
        if status=='ambiguous':
            choices=match.get('candidate_paths')
            if (not isinstance(choices,list) or not choices or len(choices)>30
                    or len(choices)!=len(set(choices))):
                raise ValueError('invalid explicit ambiguous catalog candidate paths')
            selection['candidate_paths']=[safe_corpus_relative(p) for p in choices]
            actual_count=match.get('matching_catalog_paths')
            if (type(actual_count) is not int or actual_count < len(choices)
                    or actual_count <= 1):
                raise ValueError('catalog ambiguity count differs from displayed choices')
            selection['matching_catalog_paths']=actual_count
            selection['candidate_choices_truncated']=actual_count>len(choices)
        return {**summary_payload,'documents':[]},None,False,selection
    path=safe_corpus_relative(match.get('original_path'))
    selection['original_path']=path
    selection['matching_catalog_paths']=1
    if match.get('matching_catalog_paths')!=1:
        raise ValueError('explicit unique original has ambiguous catalog count')
    if status=='verified':
        if not docs or docs[0].get('original_path')!=path:
            raise ValueError('named summary verification lacks the selected publisher original')
        filtered=[record for record in docs if record.get('original_path')==path]
    else:
        # This native summary is missing/stale: a fallback cosine document
        # in the summary result is not a substitute for that original.
        filtered=[]
    return {**summary_payload,'documents':filtered},path,True,selection


def attach_verified_pdf_previews(
    result: dict[str,Any], *, original_root: Path,
    original_receipts: dict[str,str], max_pdf_previews: int,
    verifier=None,
) -> dict[str,Any]:
    """Opt-in physical PDF page proof of only the displayed original preview.

    This operation never reranks either discovery channel. Exact full-publisher
    and derivative SHA checks are performed upstream. The original PDF is
    *again* rehashed inside the physical-page verifier against the completed
    publisher receipt, so a revision change fails closed before a quote can
    become a page citation. A successful proof covers only the displayed
    preview, not a whole derivative passage or factual answer.
    """
    if type(max_pdf_previews) is not int or not 1 <= max_pdf_previews <= 5:
        raise ValueError("PDF preview verification requires 1–5 physical originals")
    if result.get("schema") != "kmd.two_channel_original_discovery.v1":
        raise ValueError("unknown dual source discovery contract")
    rows = result.get("original_passage_channel")
    if not isinstance(rows,list) or len(rows)>20:
        raise ValueError("invalid bounded source passage channel")
    if verifier is None:
        verifier = verify_pdf_page_excerpts
    checked=0
    verified=0
    updated=[]
    for row in rows:
        if not isinstance(row,dict) or row.get("publisher_revision_verified") is not True:
            raise ValueError("unverified publisher revision in dual source channel")
        item=dict(row)
        rel=safe_corpus_relative(item.get("original_path"))
        if Path(rel).suffix.casefold() != ".pdf":
            item["physical_pdf_preview_check"]={"status":"not_pdf"}
        elif checked>=max_pdf_previews:
            item["physical_pdf_preview_check"]={"status":"not_checked_limit"}
        else:
            checked+=1
            expected=original_receipts.get(rel)
            if not isinstance(expected,str) or not re.fullmatch("[0-9a-f]{64}",expected):
                raise ValueError("missing completed original PDF receipt")
            original=existing_file(original_root,rel)
            excerpt=item.get("passage_preview")
            section=item.get("source_section")
            if not isinstance(excerpt,str) or not excerpt or len(excerpt)>350 or not isinstance(section,str):
                raise ValueError("unsafe original preview for PDF page proof")
            proofs=verifier(original,expected,[{"source_section":section,"source_excerpt":excerpt}])
            if not isinstance(proofs,list) or len(proofs)!=1 or not isinstance(proofs[0],dict):
                raise ValueError("invalid PDF physical-page verifier response")
            page=proofs[0]
            if page.get("status")=="verified":
                number=page.get("page")
                digest=page.get("page_text_sha256")
                if type(number) is not int or number<1 or not isinstance(digest,str) or not re.fullmatch("[0-9a-f]{64}",digest):
                    raise ValueError("invalid verified physical page location")
                item["physical_pdf_preview_check"]={
                    "status":"verified", "physical_pdf_page":number,
                    "basis":page.get("basis"), "page_text_sha256":digest,
                    "scope":"Only the displayed derivative excerpt is matched to original PDF page text; not full passage or answer truth.",
                }
                verified+=1
            elif page.get("status")=="unverified":
                item["physical_pdf_preview_check"]={
                    "status":"unverified", "reason":str(page.get("reason","unknown"))[:120],
                }
            else:
                raise ValueError("unknown physical-page verifier status")
        updated.append(item)
    return {
        **result, "original_passage_channel":updated,
        "physical_pdf_preview_audit":{
            "original_pdf_previews_checked":checked,
            "original_pdf_previews_physically_verified":verified,
            "max_original_pdf_previews":max_pdf_previews,
            "evidence_scope":"Display preview only; full passage, page-spanning context and factual answer not certified.",
        },
    }

def combined_dual_discovery(
    *,
    summary_payload: dict, original_payload: dict,
    source_map: sqlite3.Connection,
    original_root: Path, derivative_root: Path,
    original_receipts: dict[str, str],
    max_originals: int,
    omit_verified_nonpublisher_passages: bool = False,
) -> dict[str,Any]:
    """Keep each independently source-verified rank; never replace with RRF."""
    if not 1<=max_originals<=20 or type(omit_verified_nonpublisher_passages) is not bool:
        raise ValueError("invalid bounded original count or untrusted source verification mode")
    summaries=summary_payload.get("documents")
    passages=original_payload.get("results")
    if (not isinstance(summaries,list) or not isinstance(passages,list) or
            len(summaries)>100 or len(passages)>30 or
            summary_payload.get("query") != original_payload.get("query") or
            summary_payload.get("provenance_policy") !=
                "publisher_and_derived_sha256_checked;summary_is_not_primary_evidence" or
            original_payload.get("retrieval_mode") != "overview"):
        raise ValueError("unexpected independent KMD search schemas or query/revision contract")

    verified:dict[str,dict]={}
    passage_seen=set()
    summary_out=[]
    for i,item in enumerate(summaries,1):
        path=safe_corpus_relative(item["original_path"])
        if path in verified:
            raise ValueError("summary search returned duplicate canonical source")
        nodes=item.get("matched_summary_nodes")
        if not isinstance(nodes,list) or not nodes:
            raise ValueError("summary result lacks source-linked native nodes")
        native_revisions={node.get("original_sha256") for node in nodes
                          if isinstance(node,dict)}
        if len(native_revisions)!=1:
            raise ValueError("summary nodes refer to different or missing original revisions")
        proof=verify_current_source(
            path=path,expected_sha256=native_revisions.pop(),
            raw_passage=None,raw_passage_sha256=None,
            source_map=source_map,original_root=original_root,
            derivative_root=derivative_root,original_receipts=original_receipts,
        )
        verified[path]=proof
        if len(summary_out)<max_originals:
            summary_out.append({
                "original_path":path,"summary_rank":i,
                "model_generated_summary_preview":str(item["best_summary"])[:260],
                "summary_similarity_not_probability":item["similarity"],
                "publisher_revision_verified":True,
                "source_scope":"Generated summary is a navigation hint, NOT source text.",
            })
    raw_out=[]
    excluded_nonpublisher=[]
    for i,item in enumerate(passages,1):
        if len(raw_out)>=max_originals:
            break
        path=safe_corpus_relative(item["original_path"])
        # A source can have multiple exact, independent excerpts. We count
        # only the first qualifying appearance in the separately ranked view.
        if path in passage_seen:
            continue
        if item.get("exact_passage_hash_verified") is not True:
            raise ValueError("original passage search returned unverified source text")
        full_passage=item["source_passage"]
        passage_sha=item["source_passage_sha256"]
        expected=item["original_source_sha256"]
        try:
            proof=verify_current_source(
                path=path,expected_sha256=expected,
                raw_passage=full_passage,raw_passage_sha256=passage_sha,
                source_map=source_map,original_root=original_root,
                derivative_root=derivative_root,original_receipts=original_receipts,
            )
        except ValueError as error:
            # Optional *explicitly reported* abstention: the native original
            # scanner can occasionally return a valid chunk SHA whose text
            # occurs only in extractor-written catalog metadata. Publisher
            # and derivative bytes were both rehashed inside the verifier.
            # Do not allow an unquoted metadata header to destroy otherwise
            # verified, useful source excerpts in this one result set.
            # ALL OTHER integrity failures (changed original/derivative,
            # invalid span hash, missing source) still abort the full result.
            if (omit_verified_nonpublisher_passages and
                    str(error)=="raw passage absent from current publisher-source section of derivative"):
                excluded_nonpublisher.append({"original_path":path,
                    "native_candidate_rank":i,
                    "reason":"source_excerpt_not_present_after_publisher_source_unit_marker"})
                continue
            raise
        passage_seen.add(path)
        if path in verified and verified[path]["source_original_sha256"]!=proof["source_original_sha256"]:
            raise ValueError("source revisions differ between the two retrieval channels")
        verified[path]=proof
        if len(raw_out)<max_originals:
            preview,preview_section=publisher_text_preview(full_passage,str(item.get("source_section") or ""))
            raw_out.append({
                "original_path":path,
                "original_source_rank":len(raw_out)+1,
                "native_raw_candidate_rank_before_verification":i,
                "original_source_similarity_not_probability":item["semantic_similarity"],
                "exact_derived_source_passage_sha256":passage_sha,
                "source_section":preview_section,
                "native_chunk_source_section":item.get("source_section"),
                "passage_preview":preview,
                "passage_preview_sha256":hashlib.sha256(preview.encode("utf-8")).hexdigest(),
                "publisher_url":item.get("publisher_url"),
                "source_page_citation_warning":item.get("source_page_citation_warning"),
                "source_scope":"Exact current derivative passage, NOT guaranteed physically correct PDF page or complete governing context.",
                "publisher_revision_verified":True,
            })
    summary_paths={item["original_path"] for item in summary_out}
    source_paths={item["original_path"] for item in raw_out}
    records=[]
    for path in sorted(summary_paths|source_paths):
        s=next((item["summary_rank"] for item in summary_out if item["original_path"]==path),None)
        r=next((item["original_source_rank"] for item in raw_out if item["original_path"]==path),None)
        records.append({
            "original_path":path,
            "summary_rank":s,
            "original_passage_rank":r,
            "has_independently_rehashed_publisher_and_derivative":True,
        })
    return {
        "schema":"kmd.two_channel_original_discovery.v1",
        "query":summary_payload["query"],
        "canonical_filedb_source":"same existing original FileDB for both channels",
        "summary_channel":summary_out,
        "original_passage_channel":raw_out,
        "unique_originals_alphabetically_for_navigation_not_relevance":records,
        "summary_original_count":len(summary_out),
        "raw_original_count":len(raw_out),
        "union_original_count":len(records),
        "originals_found_only_via_original_passages":len(source_paths-summary_paths),
        "originals_found_only_via_generated_summaries":len(summary_paths-source_paths),
        "no_ranking_fusion_or_score_threshold_applied":True,
        "no_database_or_original_mutation":True,
        "uncitable_native_passage_candidates_omitted":len(excluded_nonpublisher),
        "excluded_uncitable_native_passage_sources":excluded_nonpublisher,
        "limitations":[
            "Two independent rank lists. Neither includes an independently established relevance/answerability probability.",
            "SHA-valid source excerpts require complete parent/surrounding discourse and physical-PDF-page verification before factual use.",
            "A raw passage may be a chunk of an original-extracted text derivative, not a literal byte quotation from PDF or HTML source.",
            "No claims that summary and passage rank lists have equal coverage or scores on different tasks.",
            "When explicitly enabled, native passage candidates seen only in extractor metadata are omitted with visible source/reason receipts; all changed-revision failures still abort.",
        ],
    }
