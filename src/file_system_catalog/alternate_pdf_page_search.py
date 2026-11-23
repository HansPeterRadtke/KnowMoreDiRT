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
                and proof.get('basis')=='derivative_label_confirmed')
        if not corroborated:uncorroborated+=1
        found.append({'physical_pdf_page':physical,
            'status':('same_page_second_extractor_exact_phrase_corroborated'
                      if corroborated else 'provisional_alternate_machine_text_only_not_a_verified_quote'),
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
