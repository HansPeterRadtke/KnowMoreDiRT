"""Opt-in, source-bound candidate discovery among otherwise unreadable PDF pages.

The source-page number comes from the physical publisher PDF, but the words
extracted by a *single* alternate machine may be wrong. Ranking is ephemeral;
no candidate may be called an independently verified original quotation
unless a separate Poppler pass matches the exact source phrase on that same
physical page. Even corroboration does not assert factual truth or a correct
interpretation of figures, layout, negation, quotations or conditions.
"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
import sqlite3

from .hybrid_search import query_terms
from .staged_physical_page_search import MARKER,source_units_from_corrected_text
from .source_review_audit import safe_file,sha_file
from .source_evidence import _best_excerpt
from .publisher_pdf_page import verify_pdf_page_excerpts

SHA=re.compile(r'[a-f0-9]{64}\Z')


def exact_skipped_staged_physical_pages(staged_source:str, physical_page_count:int,
                                        expected_max_page_label:int)->list[int]:
    """Only pages skipped by the current approved temporary Poppler page search."""
    if (type(physical_page_count) is not int or not 1<=physical_page_count<=450
            or type(expected_max_page_label) is not int
            or expected_max_page_label!=physical_page_count):
        raise ValueError('untrusted physical publisher PDF page count')
    markers=list(MARKER.finditer(staged_source))
    pages=[int(x[2]) for x in markers]
    if pages!=list(range(1,physical_page_count+1)):
        raise ValueError('staged original PDF source does not contain exactly one ordered physical page unit')
    readable={page for _,page,_ in source_units_from_corrected_text(
        staged_source,physical_page_count,expected_max_page_label)}
    return sorted(set(pages)-readable)


def require_current_approved_staged_pdf_source(item:dict,*,
        original_root:Path,derived_root:Path,stage_root:Path,
        source_map:sqlite3.Connection,native:sqlite3.Connection,
        completed_acquisitions:dict[str,str])->dict:
    """Fail closed if the original, live old, private new or native ID drifted."""
    if (not isinstance(item,dict) or not isinstance(item.get('original_path'),str)):
        raise ValueError('malformed exact original source receipt')
    path=item['original_path']
    if (len(path)>1600 or not path.endswith('.pdf') or path.startswith('/')
            or '\\' in path or any(p in ('','.','..') for p in path.split('/'))):
        raise ValueError('unsafe original source path')
    o=item.get('original_sha256');old=item.get('old_derived_sha256');new=item.get('new_staged_derived_sha256')
    native_id=item.get('native_original_filedb_id')
    if (not all(isinstance(v,str) and SHA.fullmatch(v) for v in (o,old,new))
            or type(native_id) is not int or native_id<1
            or completed_acquisitions.get(path)!=o):
        raise ValueError('original publisher source is not a completed approved revision')
    if (not isinstance(item.get('staged_private_file'),str) or not
            Path(item['staged_private_file']).is_absolute()):
        raise ValueError('staged source lacks a private absolute file identity')
    if (source_map.execute('PRAGMA query_only').fetchone()[0]!=1
            or native.execute('PRAGMA query_only').fetchone()[0]!=1):
        raise ValueError('source and native database connections must be read-only')
    versions=source_map.execute('SELECT source_sha256,derived_sha256,derived_path,status '
                  'FROM originals WHERE source_path=?',(path,)).fetchall()
    if versions!=[(o,old,path+'.txt','complete')]:
        raise ValueError('current source-map no longer contains the expected old revision')
    entries=native.execute('SELECT id FROM filesystem_entries WHERE entry_type=? AND relative_path_display=?',
                ('file',path)).fetchall()
    if entries!=[(native_id,)]:
        raise ValueError('native original FileDB identity differs from staged approval')
    original=safe_file(original_root,path)
    oldtext=safe_file(derived_root,path+'.txt')
    candidate=Path(item['staged_private_file'])
    if not candidate.is_relative_to(stage_root):
        raise ValueError('private stage source is outside approved temporary source tree')
    candidate=safe_file(stage_root,str(candidate.relative_to(stage_root)))
    if sha_file(original)!=o or sha_file(oldtext)!=old or sha_file(candidate)!=new:
        raise ValueError('original source, old live derivative or private stage SHA changed')
    return {'original_path':path,'original_sha256':o,'old_derived_sha256':old,
        'staged_derived_sha256':new,'native_id':native_id,'original_pdf':original,
        'old_source':oldtext,'new_staged_source':candidate}


def corroborate_specific_alternate_source_fragments(*,physical_page:int,
      provisional_text:str,query:str,independent_verifier,
      original_pdf:Path,original_sha256:str,max_fragments:int=18)->dict:
    """Corroborate ONLY verbatim same-page phrases, not the whole GS text.

    Search for *bounded* contiguous fragments with multiple relevant query
    terms. Poppler must independently locate the exact phrase on the exact
    physical page of the same fully SHA-verified original PDF. A verified
    subset is never used to confer verified status on neighboring lines or
    the full provisional page preview.
    """
    if (type(physical_page) is not int or physical_page<1
            or not isinstance(provisional_text,str) or len(provisional_text)>18000
            or not isinstance(query,str) or not 1<=len(query)<=400
            or type(max_fragments) is not int or not 1<=max_fragments<=24
            or not isinstance(original_pdf,Path)
            or not isinstance(original_sha256,str)
            or SHA.fullmatch(original_sha256) is None):
        raise ValueError('invalid bounded publisher original physical-page fragment evidence')
    terms=set(query_terms(query))
    if not terms:raise ValueError('no useful original-source query terms')
    tokens=provisional_text.split()
    if len(tokens)<10:return {'original_physical_pdf_page':physical_page,
        'exact_same_page_corroborated_fragment':None,
        'whole_provisional_text_verified_by_excerpt':False,
        'factual_answer_verified':False}
    # Long whole-sentence windows keep most of the previously confirmed
    # physical PDF passages. A reserved small fraction of the same bounded
    # verification batch handles technical PDF pages where glyph/column
    # damage leaves only shorter exact same-page machine-text fragments.
    # Never elevate the whole provisional machine page or detached context.
    window_sizes=(36,28,20,14,12,10,8)
    candidates={size:[] for size in window_sizes}
    for size in window_sizes:
        if len(tokens)<size:continue
        step=(max(3,size//6) if size>=20 else (2 if size>=14 else 1))
        for start in range(0,len(tokens)-size+1,step):
            fragment=' '.join(tokens[start:start+size])
            # A source-attested eight-word technical passage may be only
            # seventy characters long. The separate six-unique-word and
            # same-publisher-page independent Poppler checks remain mandatory;
            # source passage context is NOT automatically verified.
            if (not 70<=len(fragment)<=480
                    or chr(0) in fragment or chr(0xfffd) in fragment):
                continue
            covered=set(query_terms(fragment)) & terms
            if len(covered)<min(2,len(terms)):continue
            if size<=12:
                # Eight-word source excerpts need at least six distinct
                # substantive-looking words, not repeated boilerplate.
                unique_words={w.casefold() for w in re.findall(
                    r"(?u)\b[^\W\d_][\w'-]{2,}\b",fragment)}
                if len(unique_words)<6:continue
            candidates[size].append((len(covered),len(fragment),start,fragment,size))
    for size in window_sizes:
        candidates[size].sort(key=lambda x:(-x[0],-x[1],x[2]))
    shorter=(12,10,8)
    longer=(36,28,20,14)
    short_quota=max(1,max_fragments//3)
    long_quota=max_fragments-short_quota
    selected=[];seen=set();positions={size:[] for size in window_sizes}
    def admit(sizes,limit):
        options=sorted((row for size in sizes for row in candidates[size]),
                       key=lambda x:(-x[0],-x[1],x[2]))
        added=0
        for n,l,at,fragment,size in options:
            # On short eight-to-twelve-word windows, one shifted token can
            # make an otherwise exact source phrase unprovable. Allow two
            # overlapping independent short candidates within the small,
            # fixed-size page-verification budget. Longer windows still need
            # spatial diversity to avoid spending all probes in one column.
            if fragment in seen or (size>=14 and any(abs(at-old)<max(4,size//3)
                                       for old in positions[size])):continue
            seen.add(fragment);positions[size].append(at)
            selected.append((n,l,fragment));added+=1
            if added>=limit:return
    admit(longer,long_quota)
    # Source-original phrase agreement may exist for twelve, ten or eight
    # words even when every longer provisional sentence differs in columns
    # or glyph order. Explicit short slots prevent a long-only ranking from
    # starving independently verifiable compact original-source quotes.
    # Reserve independently attempted windows at *each* shorter length.
    # Otherwise dozens of 12-word candidates could crowd out the only
    # exact eight-word text surviving a corrupt two-column PDF page.
    for rank,size in enumerate(shorter):
        share=short_quota//len(shorter)+(1 if rank<short_quota%len(shorter) else 0)
        if share:admit((size,),share)
    if len(selected)<max_fragments:
        admit(window_sizes,max_fragments-len(selected))
    selected=selected[:max_fragments]
    if not selected:return {'original_physical_pdf_page':physical_page,
        'exact_same_page_corroborated_fragment':None,
        'whole_provisional_text_verified_by_excerpt':False,
        'factual_answer_verified':False}
    submitted=[{'source_section':f'PDF page {physical_page}','source_excerpt':f} for _,_,f in selected]
    verified=independent_verifier(original_pdf,original_sha256,submitted)
    if not isinstance(verified,list) or len(verified)!=len(submitted):
        raise ValueError('independent publisher PDF phrase corroboration returned wrong cardinality')
    accepted=[]
    for (term_count,length,fragment),proof in zip(selected,verified):
        if (isinstance(proof,dict) and proof.get('status')=='verified'
                and proof.get('page')==physical_page
                and proof.get('basis')=='derivative_label_confirmed'
                and isinstance(proof.get('page_text_sha256'),str)
                and SHA.fullmatch(proof['page_text_sha256'])):
            accepted.append((term_count,length,fragment,proof))
    best=max(accepted,key=lambda x:(x[0],x[1])) if accepted else None
    return {'original_physical_pdf_page':physical_page,
            'exact_same_page_corroborated_fragment':(
                {'text':best[2],'fragment_sha256':hashlib.sha256(best[2].encode()).hexdigest(),
                 'original_pdf_poppler_physical_page_sha256':best[3]['page_text_sha256'],
                 'physical_pdf_page':physical_page,
                 'matched_query_terms_count':best[0],
                 'basis':'two_independent_machine_extractors_exact_same_original_physical_page_phrase',
                 'full_machine_page_and_visual_layout_not_verified':True}
                if best else None),
            'tested_candidate_fragments':len(selected),
            'whole_provisional_text_verified_by_excerpt':False,
            'factual_answer_verified':False}


def provisional_alternate_query_hits(query:str, pages:list[dict],*,top:int=4,
       corroborate_with_poppler=True, original_pdf:Path|None=None,
       expected_original_sha256:str|None=None,
       independent_verifier=verify_pdf_page_excerpts)->dict:
    """Rank only source-bound provisional machine page bodies in RAM.

    An exact same-page Poppler phrase match may corroborate a passage. It
    cannot elevate an uncorroborated Ghostscript-only excerpt to a source
    verified quotation, and it never establishes a real-world factual answer.
    """
    if (not isinstance(query,str) or not 1<=len(query)<=400
            or type(top) is not int or not 1<=top<=8
            or not isinstance(pages,list) or len(pages)>32):
        raise ValueError('invalid bounded provisional original-page candidate search')
    terms=query_terms(query)
    if not terms:raise ValueError('no useful search terms in original source question')
    if corroborate_with_poppler and (not isinstance(original_pdf,Path)
               or not isinstance(expected_original_sha256,str)
               or SHA.fullmatch(expected_original_sha256) is None):
        raise ValueError('independent physical-PDF phrase verification requires pinned original SHA')
    db=sqlite3.connect(':memory:')
    accepted={};excluded=0
    try:
        db.execute("CREATE VIRTUAL TABLE original_passages USING fts5(text,tokenize='porter unicode61 remove_diacritics 2')")
        for i,row in enumerate(pages,1):
            if (not isinstance(row,dict)
                    or row.get('status')!='source_bounded_provisional_machine_text'):
                excluded+=1;continue
            if (type(row.get('physical_pdf_page')) is not int
                  or not isinstance(row.get('text'),str) or not row['text'].strip()
                  or not isinstance(row.get('display_text_sha256'),str)
                  or hashlib.sha256(row['text'].encode()).hexdigest()!=row['display_text_sha256']):
                raise ValueError('altered or untrusted extracted original physical PDF page text')
            db.execute('INSERT INTO original_passages(rowid,text) VALUES(?,?)',(i,row['text']))
            accepted[i]=row
        db.commit()
        clause=' OR '.join('"'+t.replace('"','""')+'"' for t in terms)
        ranked=db.execute('SELECT rowid,bm25(original_passages,1.0) FROM original_passages '
                          'WHERE original_passages MATCH ? ORDER BY 2,1 LIMIT 32',(clause,)).fetchall()
    finally:db.close()
    found=[];uncorroborated=0;weak=0
    for rowid,score in ranked:
        row=accepted[int(rowid)]
        begin,end,covered=_best_excerpt(row['text'],terms,length=500)
        if len(covered)<min(2,len(terms)):
            weak+=1;continue
        excerpt=row['text'][begin:end].strip()
        if len(excerpt)<35 or '\x00' in excerpt or '\ufffd' in excerpt:
            weak+=1;continue
        physical=int(row['physical_pdf_page'])
        proof=None
        if corroborate_with_poppler:
            proof=independent_verifier(original_pdf,expected_original_sha256,
                [{'source_section':f'PDF page {physical}','source_excerpt':excerpt}])[0]
        corroborated=(isinstance(proof,dict)
                and proof.get('status')=='verified'
                and proof.get('page')==physical
                and proof.get('basis')=='derivative_label_confirmed'
                and isinstance(proof.get('page_text_sha256'),str)
                and SHA.fullmatch(proof['page_text_sha256']) is not None)
        # Always expose a JSON-serializable *boolean*: re.fullmatch returns
        # a Match object when a valid physical-page SHA was corroborated.
        # In the successful path that Match previously leaked into the
        # evidence report and prevented saving the source provenance result.
        assert type(corroborated) is bool
        fragment={'exact_same_page_corroborated_fragment':None,
                  'whole_provisional_text_verified_by_excerpt':False,
                  'factual_answer_verified':False}
        if corroborated:
            fragment['whole_provisional_text_verified_by_excerpt']=False
        elif corroborate_with_poppler:
            fragment=corroborate_specific_alternate_source_fragments(
                physical_page=physical,provisional_text=row['text'],query=query,
                independent_verifier=independent_verifier,original_pdf=original_pdf,
                original_sha256=expected_original_sha256)
        if not corroborated:uncorroborated+=1
        found.append({'physical_pdf_page':physical,
            'status':('same_page_second_extractor_exact_phrase_corroborated'
                      if corroborated else 'provisional_alternate_machine_text_only_not_a_verified_quote'),
            'corroborated_short_fragment_only':fragment['exact_same_page_corroborated_fragment'],
            'short_fragment_does_not_verify_unmatched_surrounding_text':True,
            'provisional_original_page_excerpt':excerpt,
            'provisional_excerpt_sha256':hashlib.sha256(excerpt.encode()).hexdigest(),
            'alternate_extraction_sha256':row['raw_machine_text_sha256'],
            'original_pdf_page_text_cross_extractor_sha256':proof.get('page_text_sha256') if corroborated else None,
            'independent_poppler_same_physical_page_passage_corroborated':corroborated,
            'publisher_original_visual_layout_correctness_verified':False,
            'factual_answer_verified':False,'score':float(score),
            'covered_distinct_query_terms':sorted(covered)})
        if len(found)>=top:break
    return {'schema':'kmd.ephemeral_provisional_alternate_pdf_physical_page_lexical_search.v1',
            'temporary_alternate_page_candidates_indexed':len(accepted),
            'excluded_unusable_alternate_pages':excluded,
            'weak_word_coverage_candidates_excluded':weak,
            'accepted_provisional_ranked_pages':found,
            'returned_uncorroborated_machine_only_page_excerpts':uncorroborated,
            'source_page_number_proved_by_original_pdf_not_a_visual_quote_proof':True,
            'no_native_database_vectors_summaries_or_source_overwrite':True,
            'no_factual_answers_asserted':True}
