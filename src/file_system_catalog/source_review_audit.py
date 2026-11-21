"""Source-byte-checked post-hoc qualitative review of native retrieval rivals.

This diagnostic never trains/changes ranking, acquires new publications, or
promotes a model assertion into a publisher fact. A human/agent may label a
previously observed candidate provisionally, but the program independently
checks that its cited derivative text and current original publisher bytes
are real and still match the canonical acquisition and source-map digests.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path
import re
import sqlite3

from .publisher_pdf_page import verify_pdf_page_excerpts

SHA = re.compile(r"[0-9a-f]{64}\Z")
GRADES = {"partial_support", "direct_support", "different_topic_observed", "evidence_unclear"}


def safe_file(root: Path, relative: str) -> Path:
    if (not isinstance(relative,str) or not 1 <= len(relative) <= 1600
            or relative.startswith("/") or "\\" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError("unsafe original/derivative review path")
    home = root.resolve(strict=True)
    path = home
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("review path is symlinked")
    if not path.is_file() or not path.resolve(strict=True).is_relative_to(home):
        raise ValueError("review path does not exist inside approved root")
    return path


def sha_file(path: Path) -> str:
    before=path.stat(follow_symlinks=False)
    with path.open("rb") as stream:
        digest=hashlib.file_digest(stream,"sha256").hexdigest()
    after=path.stat(follow_symlinks=False)
    if (not os.path.samestat(before,after) or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns or before.st_ctime_ns != after.st_ctime_ns):
        raise ValueError("review original changed during source hash")
    return digest


def validate_reviews(*, shadow: dict, review: dict, source_map: sqlite3.Connection,
                     original_root: Path, derived_root: Path, receipts: dict[str,str],
                     verify_physical_pdf_pages: bool = False, pdf_verifier=None) -> dict:
    if shadow.get("schema") != "kmd.read_only_raw_overview_vs_hierarchical_summary_shadow.v1":
        raise ValueError("unexpected fixed retrieval result schema")
    if review.get("schema") != "kmd.posthoc_top_three_source_review.v1":
        raise ValueError("unrecognized source review schema")
    if type(verify_physical_pdf_pages) is not bool:
        raise ValueError("physical page verification requires a boolean flag")
    if pdf_verifier is None:
        pdf_verifier=verify_pdf_page_excerpts
    cases = shadow.get("cases")
    reviews = review.get("cases")
    if not isinstance(cases,list) or not isinstance(reviews,list) or not 1 <= len(reviews) <= 100:
        raise ValueError("missing bounded frozen cases or source reviews")
    ids = {(case["family"],case["id"]): case for case in cases}
    if len(ids) != len(cases):
        raise ValueError("ambiguous frozen query identities")
    seen = set()
    completed = []
    for row in reviews:
        if not isinstance(row,dict):
            raise ValueError("invalid review entry")
        key = (row.get("family"),row.get("case_id"))
        if key not in ids:
            raise ValueError("review references unknown frozen question")
        source = row.get("candidate_path")
        rel = safe_file(original_root,source)
        if (key,source) in seen:
            raise ValueError("duplicate candidate source review")
        seen.add((key,source))
        case = ids[key]
        if source in case.get("known_originals",[]):
            raise ValueError("post-hoc review cannot duplicate predeclared original gold")
        raw = case.get("raw_top_three")
        summary = case.get("summary_top_three")
        if (not isinstance(raw,list) or not isinstance(summary,list)
                or source not in raw and source not in summary):
            raise ValueError("review candidate was not in frozen top-three results")
        grade = row.get("review_grade")
        if grade not in GRADES:
            raise ValueError("unsupported manual review grade")
        notes = row.get("review_notes")
        if not isinstance(notes,str) or not 15 <= len(notes) <= 1000:
            raise ValueError("missing bounded reason for manual grade")
        original_sha = row.get("original_sha256")
        derived_sha = row.get("derived_sha256")
        if (not isinstance(original_sha,str) or not SHA.fullmatch(original_sha)
                or not isinstance(derived_sha,str) or not SHA.fullmatch(derived_sha)):
            raise ValueError("source review lacks current verified revision digests")
        matches = source_map.execute(
            "SELECT source_sha256,derived_sha256,derived_path,status "
            "FROM originals WHERE source_path=?",(source,)
        ).fetchall()
        if (len(matches) != 1 or matches[0][3] != "complete"
                or matches[0][0] != original_sha or matches[0][1] != derived_sha
                or receipts.get(source) != original_sha):
            raise ValueError("review source disagrees with current approved source-map/receipt")
        if sha_file(rel) != original_sha:
            raise ValueError("original publisher source bytes changed")
        derivative = safe_file(derived_root,matches[0][2])
        if derivative.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("extracted text exceeds bounded review evidence size")
        before_derivative=derivative.stat(follow_symlinks=False)
        blob = derivative.read_bytes()
        after_derivative=derivative.stat(follow_symlinks=False)
        if (not os.path.samestat(before_derivative,after_derivative)
                or before_derivative.st_size != after_derivative.st_size
                or before_derivative.st_mtime_ns != after_derivative.st_mtime_ns
                or before_derivative.st_ctime_ns != after_derivative.st_ctime_ns):
            raise ValueError("current derivative was replaced during source review")
        if hashlib.sha256(blob).hexdigest() != derived_sha:
            raise ValueError("current derivative source bytes changed")
        try:
            text = blob.decode("utf-8",errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("derived text is not strict UTF-8") from error
        marker = text.find("[[SOURCE UNIT:")
        if marker < 0:
            raise ValueError("derivative lacks publisher text boundary")
        marker_end = text.find("]]",marker + len("[[SOURCE UNIT:"))
        if marker_end < 0:
            raise ValueError("invalid publisher text marker")
        publisher_start = marker_end + 2
        anchors = row.get("evidence_anchors")
        if not isinstance(anchors,list) or not 1 <= len(anchors) <= 6:
            raise ValueError("no bounded source evidence anchors")
        verified_spans = []
        for anchor in anchors:
            if not isinstance(anchor,dict):
                raise ValueError("invalid evidence anchor")
            offset = anchor.get("source_start_char")
            excerpt = anchor.get("exact_derived_excerpt")
            excerpt_sha = anchor.get("excerpt_sha256")
            if (type(offset) is not int or offset < publisher_start
                    or not isinstance(excerpt,str) or not 20 <= len(excerpt) <= 400
                    or not isinstance(excerpt_sha,str) or not SHA.fullmatch(excerpt_sha)
                    or hashlib.sha256(excerpt.encode()).hexdigest() != excerpt_sha
                    or text[offset:offset+len(excerpt)] != excerpt):
                raise ValueError("quoted review span absent from current source publisher text")
            verified_spans.append({"start_char":offset,"end_char":offset+len(excerpt),
                                   "excerpt_sha256":excerpt_sha})
        page_proofs = []
        if verify_physical_pdf_pages and rel.suffix.casefold() == ".pdf":
            markers = [(m.start(),int(m.group(1))) for m in re.finditer(
                r"(?m)^\[\[SOURCE UNIT: PDF page ([1-9][0-9]*)\]\][ \t]*(?:\r?$)", text)]
            checks = []
            mapped=[]
            for i,anchor in enumerate(anchors):
                start=anchor["source_start_char"]
                end=start+len(anchor["exact_derived_excerpt"])
                before=[(pos,page) for pos,page in markers if pos <= start]
                after=[pos for pos,_page in markers if start < pos < end]
                if not before or after:
                    page_proofs.append({"status":"unverified","reason":"missing_or_crossing_source_page_marker"})
                else:
                    label=before[-1][1]
                    page_proofs.append(None)
                    mapped.append(i)
                    checks.append({"source_section":f"PDF page {label}",
                                   "source_excerpt":anchor["exact_derived_excerpt"]})
            if checks:
                proofs=pdf_verifier(rel,original_sha,checks)
                if not isinstance(proofs,list) or len(proofs)!=len(checks):
                    raise ValueError("incorrect physical-PDF proof cardinality")
                for idx,proof in zip(mapped,proofs):
                    if not isinstance(proof,dict) or proof.get("status") not in ("verified","unverified"):
                        raise ValueError("invalid original physical-PDF page proof")
                    if proof["status"]=="verified":
                        num=proof.get("page")
                        digest=proof.get("page_text_sha256")
                        if type(num) is not int or num<1 or not isinstance(digest,str) or not SHA.fullmatch(digest):
                            raise ValueError("malformed verified PDF physical-page identity")
                        page_proofs[idx]={"status":"verified","page":num,
                                          "page_text_sha256":digest,"basis":proof.get("basis")}
                    else:
                        page_proofs[idx]={"status":"unverified",
                                          "reason":str(proof.get("reason","unknown"))[:120]}
        elif verify_physical_pdf_pages:
            page_proofs=[{"status":"not_pdf"} for _ in anchors]
        completed.append({
            "family":key[0], "case_id":key[1], "original_path":source,
            "review_grade_unvalidated_semantic_judgement":grade,
            "source_original_sha256_reverified":original_sha,
            "source_derivative_sha256_reverified":derived_sha,
            "publisher_original_bytes":rel.stat().st_size,
            "matched_original_overview_rank":raw.index(source)+1 if source in raw else None,
            "matched_summary_rank":summary.index(source)+1 if source in summary else None,
            "source_text_anchors_sha_verified":verified_spans,
            "physical_original_pdf_pages":page_proofs if verify_physical_pdf_pages else None,
            "publisher_and_derivative_complete_bytes_sha_verified":True,
        })
    summary = Counter(x["review_grade_unvalidated_semantic_judgement"] for x in completed)
    return {
        "schema":"kmd.read_only_posthoc_original_source_adjudication.v1",
        "review_status":"POSTHOC, manually interpreted, NOT independent or exhaustive relevance gold",
        "warning":"Passage and source SHA validation proves identities and exact derivative text, not semantic relevance or answer correctness. Do not tune production ranks on this small inspected set.",
        "reviewed":len(completed),"grade_counts":dict(summary),"cases":completed,
        "native_corpus_or_ranker_changed":False,
        "physical_pdf_page_verification_requested":verify_physical_pdf_pages,
    }
