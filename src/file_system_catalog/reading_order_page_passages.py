"""Opt-in sentence-window candidates inside one verified physical PDF page.

This is a page-local lexical locator, not a document reranker or an entailment
classifier. Whitespace-normalized exact slices are bound to the independently
read Poppler reading-order projection. Sentence boundaries are heuristic and
the alternate projection does not establish visual reading order. Neither the
publisher original, derived text, FileDB nor default retrieval is changed.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
import hashlib
import math
import re
import sqlite3

from .hybrid_search import lexical_token_spans, query_terms

_SHA=re.compile(r"[0-9a-f]{64}\Z")
_END=re.compile(r"[.!?]+[\"'”’)\]]{0,3}(?=\s|$)")
_ABBREV=re.compile(r"(?i)(?:\b(?:e\.g|i\.e|et al|fig|eq|ref|sec|dr|mr|mrs|vs|etc|vol|no|pp)|\b[A-Z])\.$")
_NUMBER=re.compile(
    r"(?i)(?<![\w.+\-−])(?P<sign>[+\-−]?)"
    r"(?P<value>(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)"
    r"(?:\s*(?P<unit>percentage points?|percent points?|%p|%|percent|×|times?)(?!\w))?"
    r"(?!\w|\.[0-9])"
)
_WORD=re.compile(r"\S+")
_MAX_PAGE=18000


def _sha(text:str)->str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha(x)->bool:
    return isinstance(x,str) and _SHA.fullmatch(x) is not None


def _page_context(context:dict)->tuple[dict,int,str,str,str]:
    if (not isinstance(context,dict)
            or context.get("source_map_revision_reverified_after_physical_page_extraction") is not True
            or not isinstance(context.get("source"),dict)
            or not isinstance(context.get("physical_pages"),dict)):
        raise ValueError("current original physical-page context required")
    src=context["source"];block=context["physical_pages"]
    path=src.get("original_path")
    page=block.get("requested_center_physical_page")
    count=block.get("actual_original_pdf_page_count")
    if (not isinstance(path,str) or not 1<=len(path)<=1600
            or path.startswith("/") or "\\" in path or not path.lower().endswith(".pdf")
            or any(s in ("",".","..") for s in path.split("/"))
            or not _is_sha(src.get("original_sha256"))
            or not _is_sha(src.get("current_derived_sha256"))
            or src.get("publisher_original_actual_sha256_verified") is not True
            or src.get("source_map_derivative_actual_sha256_verified") is not True
            or block.get("original_pdf_sha256")!=src["original_sha256"]
            or block.get("original_pdf_revision_verified_before_and_after") is not True
            or type(page) is not int or type(count) is not int or not 1<=page<=count):
        raise ValueError("unverified original or current derivative revision/page")
    rows=block.get("pages")
    if (not isinstance(rows,list) or len(rows)!=1 or not isinstance(rows[0],dict)
            or block.get("neighbor_radius")!=0 or rows[0].get("physical_page")!=page
            or rows[0].get("status")!="verified_original_page_text"):
        raise ValueError("one explicit verified physical page required")
    layout_sha=rows[0].get("physical_page_poppler_sha256")
    alt=block.get("optional_center_reading_order_projection")
    if (not _is_sha(layout_sha) or not isinstance(alt,dict)
            or alt.get("physical_page")!=page
            or alt.get("status")!="same_physical_page_alternate_reading_order_text"
            or alt.get("extraction_mode")!="poppler_default_reading_order_utf8_not_layout"
            or alt.get("exact_layout_page_poppler_sha256")!=layout_sha
            or alt.get("complete_extractor_output_not_truncated") is not True
            or not _is_sha(alt.get("reading_order_poppler_sha256"))
            or alt.get("extracted_text_sha256")!=alt["reading_order_poppler_sha256"]):
        raise ValueError("verified complete reading-order projection required")
    body=alt.get("extracted_page_text")
    if (not isinstance(body,str) or not 1<=len(body)<=_MAX_PAGE or not body.strip()
            or _sha(body)!=alt["reading_order_poppler_sha256"]):
        raise ValueError("current physical page reading-order text digest mismatch")
    return src,page," ".join(body.split()),layout_sha,alt["reading_order_poppler_sha256"]


def _sentence_units(text:str)->list[tuple[int,int]]:
    """Heuristic punctuation boundaries; preserve every non-whitespace byte of slices."""
    start=0;out=[]
    for m in _END.finditer(text):
        end=m.end()
        # Common abbreviations and initial letters are not reliable endings.
        if m.group().startswith(".") and _ABBREV.search(text[max(start,end-24):end]):
            continue
        nxt=end
        while nxt<len(text) and text[nxt].isspace():
            nxt+=1
        # Avoid splitting common abbreviations and continuing lowercase prose.
        if nxt<len(text) and text[nxt].islower() and m.group().startswith("."):
            continue
        if start<end:
            out.append((start,end))
        start=nxt
    if start<len(text):
        out.append((start,len(text)))
    return out


def _numeric_keys(text:str)->set[str]:
    """Exact numeric value/unit matching for ranking only, never a factual comparison."""
    out=set()
    for m in _NUMBER.finditer(text):
        # Canonicalize only explicitly written decimal literals and common
        # unit spellings. String arithmetic avoids Decimal context rounding.
        token=m["value"].replace(",","")
        if len(token)>60:continue
        whole,_,fraction=token.partition(".")
        value=(whole.lstrip("0") or "0")
        if fraction.rstrip("0"):value+="."+fraction.rstrip("0")
        sign="-" if m["sign"] in ("-","−") and value!="0" else ""
        written=(m["unit"] or "").casefold()
        unit=("%p" if written in ("%p","percentage point","percentage points","percent point","percent points")
              else "%" if written in ("%","percent") else "×" if written in ("×","time","times") else "")
        out.add(sign+value+unit)
    return out


def _long_unit_windows(text:str,start:int,end:int,limit:int)->list[tuple[int,int]]:
    """Word-aligned explicitly partial windows; no silent character clipping."""
    words=list(_WORD.finditer(text,start,end));out=[];i=0
    while i<len(words):
        j=i
        while j<len(words) and words[j].end()-words[i].start()<=limit:
            j+=1
        if j==i:
            # An unbroken token longer than the entire bound is unavailable.
            i+=1;continue
        out.append((words[i].start(),words[j-1].end()))
        if j==len(words):
            break
        i=max(i+1,i+(j-i)//2)
    return out



def _porter_tokens(terms:set[str])->dict[str,str]:
    """Use SQLite's built-in Porter tokenizer, not an invented suffix list.

    Only ordinary bounded ASCII words participate. Numeric literals,
    identifiers, non-ASCII words and publisher text remain unchanged.
    """
    words=sorted(t for t in terms if t.isascii() and t.isalpha() and 4<=len(t)<=48)
    if not words:return {}
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE VIRTUAL TABLE stem_input USING fts5(word, tokenize='porter unicode61')")
        db.executemany("INSERT INTO stem_input(rowid,word) VALUES (?,?)",enumerate(words,1))
        db.execute("CREATE VIRTUAL TABLE stem_vocab USING fts5vocab(stem_input,'instance')")
        rows=db.execute("SELECT doc,term FROM stem_vocab ORDER BY doc").fetchall()
    if len(rows)!=len(words) or [doc for doc,_ in rows]!=list(range(1,len(words)+1)):
        raise ValueError("unexpected Porter tokenization for single-word inputs")
    return {words[doc-1]:stem for doc,stem in rows}


def _rank_normalized_reading_order_text(text:str,query:str,*,top:int=3,max_characters:int=900,
                                       include_inflection_matches:bool=False,
                                       term_weights:dict|None=None)->dict:
    """Pure bounded machine-text locator; no original or page proof by itself."""
    if (not isinstance(text,str) or not text.strip() or len(text)>_MAX_PAGE
            or text!=" ".join(text.split())
            or type(top) is not int or not 1<=top<=5 or type(max_characters) is not int
            or not 160<=max_characters<=2000 or type(include_inflection_matches) is not bool):
        raise ValueError("invalid bounded normalized reading-order text")
    terms={t for t in query_terms(query) if not t.isdecimal()}
    numbers=_numeric_keys(query)
    if not terms and not numbers:
        raise ValueError("no informative page-local query terms")
    stems={};query_stems={}
    if include_inflection_matches:
        stems=_porter_tokens(terms|{t for t,_,_ in lexical_token_spans(text)})
        for term in terms:
            if term in stems:query_stems.setdefault(stems[term],set()).add(term)
    units=_sentence_units(text)
    unit_features=[]
    frequencies=Counter()
    def features(s,e):
        words={t for t,_,_ in lexical_token_spans(text[s:e])}
        exact=terms & words
        inflections={t for word in words if word in stems
                     for t in query_stems.get(stems[word],())}-exact
        return exact,inflections,numbers & _numeric_keys(text[s:e])
    for s,e in units:
        f=features(s,e);unit_features.append(f);frequencies.update(f[0]|f[1])
    if term_weights is None:
        weights={t:1+math.log((1+len(units))/(1+frequencies[t])) for t in terms}
    else:
        if (not isinstance(term_weights,dict) or set(term_weights)!=terms
                or any(type(v) not in (int,float) or not math.isfinite(v) or not 1<=v<=20
                       for v in term_weights.values())):
            raise ValueError("invalid shared page-discovery term weights")
        weights=term_weights
    candidates=[]
    def add(s,e,words,morph,nums,first,last,partial=False):
        morph=morph-words
        if not words and not morph and not nums:return
        # Cover exact distinctive terms, with additional full numeric literals.
        # Shorter core breaks ties; surrounding context is expanded separately.
        score=(round(sum(weights[t] for t in sorted(words))
                     +0.55*sum(weights[t] for t in sorted(morph))+2*len(nums),9),
               len(nums),len(words),-(e-s),-s)
        candidates.append((score,s,e,first,last,partial,words,morph,nums))
    long_count=0
    for i,(s,e) in enumerate(units):
        if e-s>max_characters:
            long_count+=1
            for left,right in _long_unit_windows(text,s,e,max_characters):
                w,m,n=features(left,right);add(left,right,w,m,n,i,i,True)
            continue
        w=set();m=set();n=set()
        for j in range(i,min(len(units),i+6)):
            end=units[j][1]
            if end-s>max_characters:break
            w.update(unit_features[j][0]);m.update(unit_features[j][1]);n.update(unit_features[j][2])
            add(s,end,set(w),set(m),set(n),i,j)
    candidates.sort(key=lambda c:c[0],reverse=True)
    chosen=[]
    for score,s,e,i,j,partial,words,morph,nums in candidates:
        core_s,core_e=s,e
        # Add immediately governing context only as a contiguous exact slice,
        # without cropping a neighboring heuristic sentence to fit the limit.
        if not partial:
            for k in (i-1,j+1):
                if 0<=k<len(units):
                    left=min(s,units[k][0]);right=max(e,units[k][1])
                    if right-left<=max_characters:s,e=left,right
        if any(max(0,min(e,x["end_char"])-max(s,x["start_char"]))/
               min(e-s,x["end_char"]-x["start_char"])>=0.8 for x in chosen):
            continue
        excerpt=text[s:e]
        actual_words,actual_morph,actual_nums=features(s,e)
        chosen.append({"rank":len(chosen)+1,"start_char":s,"end_char":e,
            "core_start_char":core_s,"core_end_char":core_e,
            "text":excerpt,"span_sha256":_sha(excerpt),
            "matched_query_terms":sorted(actual_words),
            "matched_query_inflection_terms":sorted(actual_morph),
            "matched_query_number_literals":sorted(actual_nums),
            "core_matched_query_terms":sorted(words),"core_matched_query_inflection_terms":sorted(morph),
            "core_matched_query_number_literals":sorted(nums),
            "heuristic_sentence_window":not partial,
            "partial_overlong_heuristic_sentence":partial,
            "touches_physical_page_boundary":s==0 or e==len(text)})
        if len(chosen)>=top:break
    return {"heuristic_sentence_units":len(units),"overlong_heuristic_units":long_count,
            "candidate_windows_considered":len(candidates),"passages":chosen}


def reading_order_page_passages(context:dict,query:str,*,top:int=3,max_characters:int=900,
                                include_inflection_matches:bool=False)->dict:
    """Rank bounded, exact same-page candidate windows; keep provenance and uncertainty."""
    if (type(top) is not int or not 1<=top<=5 or type(max_characters) is not int
            or not 160<=max_characters<=2000 or type(include_inflection_matches) is not bool):
        raise ValueError("invalid bounded page-local search parameters")
    src,page,text,layout_sha,reading_sha=_page_context(context)
    ranked=_rank_normalized_reading_order_text(text,query,top=top,max_characters=max_characters,
                    include_inflection_matches=include_inflection_matches)
    return {"schema":"kmd.verified_original_page_reading_order_lexical_candidates.v1",
            "original_path":src["original_path"],"title":src.get("title"),
            "source_url":src.get("source_url"),"physical_page":page,
            "original_sha256":src["original_sha256"],"derived_sha256":src["current_derived_sha256"],
            "layout_page_sha256":layout_sha,"reading_order_page_sha256":reading_sha,
            "normalized_reading_order_page_sha256":_sha(text),"normalized_page_characters":len(text),
            "offset_basis":"whitespace-normalized Poppler default reading order on one physical page",
            **ranked,"max_characters_per_candidate":max_characters,"native_document_ranking_unchanged":True,
            "visual_reading_order_or_factual_claim_truth_verified":False,
            "context_cannot_by_itself_prove_a_factual_claim":True,
            "sentence_boundaries_are_heuristic_not_guaranteed_complete":True,
            "query_number_matching_is_retrieval_only":True,
            "inflection_matching_enabled":include_inflection_matches,
            "query_inflection_matching_is_retrieval_only":True}
