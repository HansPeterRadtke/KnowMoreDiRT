"""Bounded read-only original-source passage locator, not answer verification.

This is deliberately a second stage *after* document discovery or an
explicit original-file choice. It searches the verified current extracted
source through a temporary RAM-only FTS5 index and returns traceable
source spans/page labels, not semantic answer confidence or invented quotes.

No new persistent vector/metadata DB, no source file modification.
"""
from __future__ import annotations

import hashlib
import re

from .hybrid_search import BoundedPassageIndex, query_terms
from .hierarchical_summary import split_sections


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _best_excerpt(
    passage: str, terms: tuple[str, ...], *,
    length: int,
) -> tuple[int, int, tuple[str, ...]]:
    """Prefer dense exact query-term coverage within a bounded source span."""
    if not 160 <= length <= 2000:
        raise ValueError("invalid excerpt length")
    if len(passage) <= length:
        found = tuple(sorted(set(terms) & {
            t.casefold() for t in re.findall(r"(?u)\b[^\W_]{2,48}\b",passage)
        }))
        return 0,len(passage),found
    matches=[]
    for term in terms:
        expr=r"(?<!\w)"+re.escape(term)+r"(?!\w)"
        for match in list(re.finditer(expr,passage,flags=re.IGNORECASE))[:50]:
            matches.append(match.start())
    if not matches:
        return 0,min(len(passage),length),()
    options={0}
    for pos in matches:
        options.add(max(0,min(len(passage)-length,pos-length//3)))
        options.add(max(0,min(len(passage)-length,pos-length//2)))
    best=(0,0,())
    bestscore=(-1,-1)
    for start in sorted(options):
        end=min(len(passage),start+length)
        words={
            t.casefold() for t in re.findall(r"(?u)\b[^\W_]{2,48}\b",passage[start:end])
        }
        covered=tuple(sorted(set(terms)&words))
        # All else equal, prefer a later passage containing more distinct
        # matches; never invent a citation for terms outside the source.
        score=(len(covered),sum(t in words for t in terms))
        if score>bestscore:
            bestscore=score
            best=(start,end,covered)
    return best


def source_evidence_candidates(
    *,
    original_path: str,
    original_title: str,
    filesystem_entry_id: int,
    original_sha256: str,
    derived_sha256: str,
    derived_text: str,
    query: str,
    top: int = 5,
    snippet_characters: int = 900,
    max_source_characters: int = 20_000_000,
    max_passages: int = 30_000,
) -> dict:
    """Return exact original-source lexical candidates with stable char spans.

    Preconditions:
      caller has already checked the publisher original file SHA and byte
      size against the accepted source ledger and checked this derivative
      against source_map for the exact original path.

    The derivative text itself is *also* hashed here so direct callers
    cannot accidentally bypass the revision precondition.
    """
    if not 1<=top<=50:
        raise ValueError("invalid bounded requested source passage count")
    if not 160<=snippet_characters<=2000:
        raise ValueError("invalid source snippet size")
    if not 1<=max_source_characters<=100_000_000 or not 1<=max_passages<=100_000:
        raise ValueError("unbounded source evidence indexing")
    if not isinstance(derived_text,str) or not derived_text.strip():
        raise ValueError("missing extracted original source text")
    if len(derived_text)>max_source_characters:
        raise ValueError("source exceeds explicit RAM-only FTS bound; no silent sampling")
    if _sha(derived_text)!=derived_sha256:
        raise ValueError("source text digest differs from required derivative revision")
    terms=query_terms(query)
    if not terms:
        raise ValueError("no informative source search terms")
    sections=split_sections(derived_text)
    offsets={}
    chunks=0
    with BoundedPassageIndex(
        max_passages=max_passages,max_characters=max_source_characters
    ) as idx:
        for section in sections:
            # The extraction ledger prepends a KMD-owned metadata JSON
            # header containing titles and search keywords. It is NOT
            # publisher evidence. Never index this prologue as an
            # answer-bearing source passage, although its bytes remain
            # included in the verified full derivative hash and offsets.
            if (len(sections)>1 and section.start_char==0
                    and section.label=="document opening"
                    and section.text.startswith("KMD source provenance ")):
                continue
            # >65k-sized sections are divided without changing source
            # character indices; the model's source unit labels stay intact.
            part=0
            for rel in range(0,len(section.text),60_000):
                text=section.text[rel:rel+60_000]
                if not text.strip():
                    continue
                part+=1
                label=section.label if len(section.text)<=60_000 else (
                    f"{section.label} [part {part}]"
                )
                start=section.start_char+rel
                end=start+len(text)
                digest=_sha(text)
                key=(label,digest)
                if key in offsets:
                    # Identical source units can occur; preserve distinct
                    # offset identities in the label instead of mixing them.
                    label=f"{label} [source offset {start}]"
                    key=(label,digest)
                offsets[key]=(start,end)
                idx.add_passage(
                    filesystem_entry_id=filesystem_entry_id,
                    original_path=original_path,
                    original_title=original_title,
                    source_section=label,
                    passage=text,
                    original_sha256=original_sha256,
                    derived_sha256=derived_sha256,
                )
                chunks+=1
        # Oversample a small bounded group then rank by literal evidence
        # found in the source *body*. A matched document title alone is
        # not a supporting source passage.
        found=idx.search(query,limit=min(2000,max(100,top*12)))
    out=[]
    for item in found:
        passage=item["passage"]
        matches={
            x.casefold() for x in re.findall(r"(?u)\b[^\W_]{2,48}\b",passage)
        } & set(terms)
        if len(matches)<min(2,len(terms)):
            continue
        start,end=offsets[(item["source_section"],item["passage_sha256"])]
        small_start,small_end,covered=_best_excerpt(
            passage,terms,length=snippet_characters
        )
        excerpt=passage[small_start:small_end]
        out.append({
            "original_path":original_path,
            "filesystem_entry_id":filesystem_entry_id,
            "original_sha256":original_sha256,
            "derived_sha256":derived_sha256,
            "source_section":item["source_section"],
            "source_start_char":start,
            "source_end_char":end,
            "source_span_sha256":item["passage_sha256"],
            "excerpt_start_char":start+small_start,
            "excerpt_end_char":start+small_end,
            "excerpt_sha256":_sha(excerpt),
            "source_excerpt":excerpt,
            "lexical_bm25":item["lexical_bm25"],
            "matched_source_terms":sorted(matches),
            "matched_excerpt_terms":list(covered),
            "total_informative_query_terms":len(terms),
            "status":"source_lexical_candidate_not_verified_answer",
        })
        if len(out)>=top:
            break
    return {
        "original_path":original_path,
        "original_sha256":original_sha256,
        "derived_sha256":derived_sha256,
        "source_characters":len(derived_text),
        "source_units":len(sections),
        "ephemeral_fts_passages":chunks,
        "returned_passages":len(out),
        "passages":out,
        "evidence_warning":(
            "These are literal matches in the verified original source, "
            "not proof the passage answers the question. Source scope, "
            "negation, date and contradictions require separate validation."
        ),
    }
