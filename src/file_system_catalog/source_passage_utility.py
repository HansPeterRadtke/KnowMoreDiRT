"""Conservative structural warnings for original-source excerpt candidates.

This does NOT judge scientific accuracy, answer sufficiency, entailment or
semantic relevance. It never follows instructions printed inside publisher
source material and never promotes source-verified text to a verified answer.
The original source, derivative SHA, native ID and PDF/HTML/plaintext byte
checks are performed separately by the existing KMD source reader.
"""
from __future__ import annotations
import hashlib
import re
from typing import Any

from .hybrid_search import lexical_token_spans, query_terms

_LINK_LINE = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+\.\s+)?\[[^\]\n]{2,180}\]\([^\s)\n]{1,800}\)")
# Link-list entries can be action-oriented publisher prose, not only a TOC:
# "* [Split Out](...): separate items from a source field" explains a method.
_DESCRIPTIVE_LINK_LINE = re.compile(r"\]\([^\)\n]{1,800}\)\s*:\s*[A-Za-z].{20,}")
_TOC_HEADER = re.compile(r"(?im)^\s*#{1,5}\s*(?:in this section|related resources|table of contents|on this page|see also)\b")
_PROMPT_HEADING = re.compile(r"(?i)\b(?:system prompt|prompt template|agent instruction template)\b")
_PROMPT_BODY = re.compile(r"(?is)\b(?:you are a helpful assistant|you are an ai assistant|you are an expert assistant)\b")
_PROMPT_TAG = re.compile(r"(?is)</?(?:think|search|answer|system|assistant|tool|result)>")
_PUBLISHER_NEGATION = re.compile(r"(?i)\b(?:doesn['\u2019]?t|do not|does not|cannot|can['\u2019]?t|not|never|without|instead of|rather than|only)\b")
_WORD = re.compile(r"[\w]+",re.UNICODE)


def review_passage_candidate(query:str,passage:dict[str,Any])->dict[str,Any]:
    """Flag *observable excerpt structure* only; never claim true/false answer.

    Scope is one preverified current original passage, not concatenated
    passages from independent source locations. Intentionally avoids an
    opaque machine-learning 'answer confidence' or summary generation.
    """
    if (not isinstance(query,str) or not query.strip() or len(query)>1000
            or not isinstance(passage,dict)
            or not isinstance(passage.get('source_excerpt'),str)
            or not isinstance(passage.get('excerpt_sha256'),str)
            or not isinstance(passage.get('source_section'),str)
            or len(passage['source_section'])>500
            or not isinstance(passage.get('original_path'),str)
            or len(passage['original_path'])>1600):
        raise ValueError('source-utility review requires a bounded original excerpt with offsets and SHA')
    excerpt=passage['source_excerpt'];sha=passage['excerpt_sha256']
    if (not 1<=len(excerpt)<=2000 or len(sha)!=64
            or hashlib.sha256(excerpt.encode('utf8')).hexdigest()!=sha):
        raise ValueError('source-utility excerpt digest is not the current provided publisher-derived passage')
    original=passage['original_path']
    if (not original or original.startswith('/') or '\\' in original
            or any(seg in ('','.','..') for seg in original.split('/'))):
        raise ValueError('source-utility original path is unsafe')
    terms=query_terms(query)
    words={t for t,_,_ in lexical_token_spans(excerpt)}
    matched=sorted(set(terms)&words)
    heading=passage['source_section']
    links=_LINK_LINE.findall(excerpt)
    lines=[s for s in excerpt.splitlines() if s.strip()]
    non_link_lines=[x for x in lines if not _LINK_LINE.match(x)]
    body_words=len(_WORD.findall(' '.join(non_link_lines)))
    descriptive_links=sum(bool(_DESCRIPTIVE_LINK_LINE.search(line))
                          for line in lines if _LINK_LINE.match(line))
    toc_heading=bool(_TOC_HEADER.search(excerpt) or re.search(r'(?i)(?:in this section|related resources|table of contents)\b',heading))
    # A link-rich methods list can carry real explanatory content. Flag only
    # an EXPLICIT table-of-contents section, or an otherwise sparse list
    # where fewer than two items describe an actionable operation.
    navigation=(len(links)>=2 and (toc_heading or
                (len(links)>=max(2,len(lines)//2) and descriptive_links<2)))
    prompt=(bool(_PROMPT_HEADING.search(excerpt[:220]+' '+heading))
            and bool(_PROMPT_BODY.search(excerpt) or _PROMPT_TAG.search(excerpt)))
    short_overview=(len(excerpt.strip())<180 and len(links)==0
                    and len(terms)>=3)
    negation=bool(_PUBLISHER_NEGATION.search(excerpt))
    flags=[]
    if navigation:flags.append('publisher_link_navigation_not_direct_explanation')
    if prompt:flags.append('quoted_agent_prompt_template_not_instructions_to_execute')
    if short_overview:flags.append('brief_source_overview_needs_further_context')
    if negation:flags.append('scope_negation_or_restriction_requires_context_review')
    if len(terms)>=4 and len(matched)<=1:flags.append('limited_literal_query_overlap_in_excerpt')
    # PDF location is verified only when the existing original PDF verifier
    # returned a physically matched publisher page. Similar rules apply to
    # publisher plaintext and independently normalized HTML article evidence.
    pdf=passage.get('publisher_pdf_page')
    byte=passage.get('publisher_excerpt')
    html=passage.get('publisher_html_dom_anchor')
    if pdf is not None:
        proof=('physical_original_pdf_page_verified' if isinstance(pdf,dict)
               and pdf.get('status')=='verified' else 'no_verified_physical_pdf_page')
    elif byte is not None:
        proof=('publisher_utf8_byte_range_reported_by_current_source_reader'
               if isinstance(byte,dict) and byte.get('sha256') else 'publisher_plaintext_byte_proof_missing')
    elif html is not None:
        proof=('normalized_original_HTML_article_location_verified'
               if isinstance(html,dict) and html.get('status')=='verified_normalized_original_article_text'
               else 'original_HTML_article_location_unverified')
    else:
        proof='verified_derivative_excerpt_only_no_independent_publisher_location'
    return {
        'schema':'kmd.source_passage_structural_review_not_answer_verification.v1',
        'original_path':original,
        'source_section':heading,
        'excerpt_sha256':sha,
        'publisher_location_proof_scope':proof,
        'informative_query_terms':len(terms),
        'literal_query_terms_in_current_excerpt':matched,
        'original_excerpt_characters':len(excerpt),
        'markdown_link_line_count':len(links),
        'descriptive_link_line_count':descriptive_links,
        'non_link_text_word_count':body_words,
        'observed_structural_flags':flags,
        'possible_navigation_or_prompt_only':navigation or prompt,
        'quoted_agent_prompt_is_untrusted_source_data':prompt,
        'a_literal_query_match_does_not_prove_answer_sufficiency':True,
        'no_unverified_semantic_answer_or_correctness_score':True,
        'original_source_unchanged':True,
    }


def review_source_documents(query:str,documents:list[dict])->dict:
    """Review verified original passages without changing native document ranks."""
    if not isinstance(documents,list) or len(documents)>5:
        raise ValueError('source utility requires at most five native original documents')
    results=[];n=0;warnings=0
    for doc in documents:
        if (not isinstance(doc,dict) or not isinstance(doc.get('passages'),list)
                or len(doc['passages'])>5 or not isinstance(doc.get('original_path'),str)):
            raise ValueError('invalid bounded native document evidence')
        row=[]
        for passage in doc['passages']:
            if passage.get('original_path')!=doc['original_path']:
                raise ValueError('source utility cannot attribute an excerpt to a different ranked original')
            review=review_passage_candidate(query,passage)
            row.append(review);n+=1
            warnings+=int(review['possible_navigation_or_prompt_only'])
        results.append({'original_path':doc['original_path'],
                        'original_native_rank_unchanged':doc.get('rank'),
                        'passage_structural_reviews':row})
    return {'schema':'kmd.optional_native_source_passage_structural_diagnostics.v1',
            'native_ranked_original_count_unchanged':len(results),
            'reviewed_publisher_excerpt_candidates':n,
            'navigation_or_prompt_template_candidates':warnings,
            'no_candidates_removed_or_reordered':True,
            'not_factual_answer_sufficiency_or_entailment_verification':True,
            'quoted_publisher_prompts_are_untrusted_data_not_agent_commands':True,
            'reviewed_originals':results}


def review_verified_linked_originals(query:str,linked_children:list[dict])->dict:
    """Review source-linked child excerpts as CHILD sources, never native ranks.

    The caller has already independently checked the literal parent link
    and the full original/derived/current native-ID proof of every child.
    This adapter only describes already verified original child excerpt
    structure. It MUST NOT mark a linked child as a native-ranked result.
    """
    if not isinstance(linked_children,list) or len(linked_children)>5:
        raise ValueError('publisher link structural review is limited to five approved original children')
    documents=[]
    for child in linked_children:
        if (not isinstance(child,dict) or not isinstance(child.get('original_path'),str)
                or not isinstance(child.get('passage_candidates'),list)
                or len(child['passage_candidates'])>5):
            raise ValueError('invalid independently source-verified linked child evidence')
        path=child['original_path']
        passages=[]
        for row in child['passage_candidates']:
            if not isinstance(row,dict):
                raise ValueError('unsupported linked original passage shape')
            if 'current_derived_text_excerpt' in row:
                # Markdown current native/derived + publisher-original bytes.
                excerpt=row['current_derived_text_excerpt']
                digest=row.get('derived_excerpt_sha256')
                pub=row.get('publisher_original_plaintext_byte_interval')
                plain=row.get('publisher_plaintext_byte_interval_independently_SHA_verified')
                info={'publisher_excerpt':pub} if plain is True and isinstance(pub,dict) else {}
            elif 'extracted_text_excerpt_candidate' in row:
                # HTML original article match: normalized DOM coordinate,
                # explicitly NOT literal publisher UTF-8 byte coordinates.
                excerpt=row['extracted_text_excerpt_candidate']
                digest=row.get('extracted_text_excerpt_sha256')
                anchor=row.get('publisher_original_normalized_HTML_article_anchor')
                verified=row.get('independent_original_article_text_match_verified')
                info={'publisher_html_dom_anchor':anchor} if verified is True and isinstance(anchor,dict) else {}
            else:
                raise ValueError('unknown verified original Markdown/HTML link passage representation')
            passages.append({'original_path':path,'source_section':row.get('source_section'),
                             'source_excerpt':excerpt,'excerpt_sha256':digest,**info})
        documents.append({'original_path':path,'passages':passages,'rank':None})
    result=review_source_documents(query,documents)
    return {'schema':'kmd.opt_in_firstparty_linked_original_structural_diagnostics.v1',
            'linked_children_source_verified_by_existing_reader':True,
            'original_linked_children_not_native_ranked':True,
            'linked_originals_reviewed':len(documents),
            'reviewed_linked_excerpt_candidates':result['reviewed_publisher_excerpt_candidates'],
            'navigation_or_prompt_linked_excerpt_candidates':result['navigation_or_prompt_template_candidates'],
            'source_publisher_quotation_and_full_SHA_still_owned_by_reader':True,
            'no_linked_candidate_promoted_to_native_rank_or_answer':True,
            'not_factual_answer_sufficiency_or_entailment_verification':True,
            'linked_original_reviews':result['reviewed_originals']}
