"""Optional publisher-verified corrected-PDF page evidence alongside native retrieval.

The main KnowMoreDiRT native search can identify relevant original PDFs even
when their existing cached page offsets are obsolete. With explicit private
staging opt-in, inspect only the native-selected originals' independently
verified corrected page text in RAM. Never change or mislabel native rows,
source-map revisions, publisher originals, summaries, or running workers.
"""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
from .source_review_audit import safe_file
from .staged_physical_page_search import StagedPhysicalPageRAMIndex, verify_staged_page_result

EXPECTED_SCHEMA='kmd.all_327_approved_original_pdf_private_stage_revision_and_physical_page_preflight.v1'


def pinned_private_corrected_sources(*,stage_preflight:Path,private_root:Path):
    if (not stage_preflight.is_absolute() or not private_root.is_absolute()
            or not private_root.is_dir() or private_root.is_symlink()
            or not stage_preflight.is_relative_to(private_root)):
        raise ValueError('private publisher staging receipt must stay within the approved audit root')
    file=safe_file(private_root,str(stage_preflight.relative_to(private_root)))
    if file.stat().st_size>8*1024*1024:
        raise ValueError('oversized private original PDF stage receipt')
    data=json.loads(file.read_bytes())
    if (data.get('schema')!=EXPECTED_SCHEMA
            or data.get('source_pdf_originals_verified_and_staged')!=327
            or data.get('all_original_old_derived_and_new_staged_full_sha_reverified_after_staging') is not True
            or data.get('current_source_map_old_derivative_revisions_and_native_ids_reverified') is not True
            or data.get('status')!='ALL_PRIVATE_STAGE_CANDIDATES_VERIFIED_NOT_PUBLISHED_TO_LIVE_SEARCH'):
        raise ValueError('private publisher PDF source cohort is unapproved or stale')
    records=data.get('original_sources_and_corrected_private_derivative_candidates')
    if not isinstance(records,list) or len(records)!=327:
        raise ValueError('private publisher repair cohort incomplete')
    out={}
    for rec in records:
        if not isinstance(rec,dict):raise ValueError('malformed original publisher stage record')
        path=rec.get('original_path')
        if (not isinstance(path,str) or not path.endswith('.pdf') or path.startswith('/')
                or '\\' in path or any(part in ('','.','..') for part in path.split('/'))
                or path in out):
            raise ValueError('ambiguous original publisher identity in private stage')
        out[path]=rec
    return out


def verified_native_selected_private_page_overlay(*,documents:list[dict],query:str,
                   staged_records:dict[str,dict],stage_root:Path,
                   original_root:Path,derived_root:Path,
                   completed_receipts:dict[str,str],source_map:sqlite3.Connection,
                   native:sqlite3.Connection,top_per_original:int=2,
                   candidates_per_original:int=25,verifier=None):
    """Return only private corrected pages proven against exact current PDF bytes.

    Selection is the normal independent native search. This is not a replacement
    for its stale vector content or an assertion of semantic/factual correctness.
    Source-version/identity mismatches fail closed instead of mixing revisions.
    """
    if (not isinstance(documents,list) or len(documents)>5
            or not 1<=top_per_original<=5 or not top_per_original<=candidates_per_original<=100
            or not isinstance(query,str) or not query.strip() or len(query)>400
            or source_map.execute('PRAGMA query_only').fetchone()[0]!=1
            or native.execute('PRAGMA query_only').fetchone()[0]!=1):
        raise ValueError('invalid bounded read-only private PDF page query')
    results=[];seen=set()
    for doc in documents:
        if not isinstance(doc,dict):raise ValueError('invalid native original search document')
        path=doc.get('original_path')
        if not isinstance(path,str) or path in seen:
            raise ValueError('duplicate or unknown native original in candidate set')
        seen.add(path)
        staged=staged_records.get(path)
        if not staged:
            continue  # Not an affected PDF; native results remain authoritative.
        if doc.get('status') not in ('verified','no_verified_excerpt'):
            results.append({'original_path':path,'status':'not_checked_native_revision_unavailable',
                            'corrected_private_pdf_excerpts':[],
                            'no_claim_of_native_reindexing':True})
            continue
        if (doc.get('original_sha256')!=staged.get('original_sha256')
                or doc.get('derived_sha256')!=staged.get('old_derived_sha256')
                or doc.get('filesystem_entry_id')!=staged.get('native_original_filedb_id')):
            raise ValueError('native original and independently staged revision identity diverge')
        with StagedPhysicalPageRAMIndex(max_records=1) as index:
            index.add(staged,stage_root=stage_root)
            index.commit()
            ranked=index.ranked(query,limit=candidates_per_original,original_path=path)
            verified=[];rejected=0
            for hit in ranked:
                extra={'verifier':verifier} if verifier is not None else {}
                proof=verify_staged_page_result(hit,query=query,
                     original_root=original_root,derived_root=derived_root,
                     stage_root=stage_root,completed_receipts=completed_receipts,
                     source_map=source_map,native_db=native,**extra)
                if proof.get('status')=='publisher_original_physical_pdf_page_verified':
                    verified.append(proof)
                    if len(verified)>=top_per_original:break
                else:rejected+=1
        results.append({'original_path':path,'status':
                            'publisher_original_physical_pages_verified' if verified
                            else 'no_independently_verified_corrected_pdf_page',
                        'corrected_private_pdf_excerpts':verified,
                        'weak_or_unverified_candidates_rejected':rejected,
                        'ranked_source_candidates_considered':min(len(ranked),
                                             rejected+len(verified)),
                        'corrected_pages_are_temporary_original_pdf_verified_not_native':True,
                        'existing_native_filedb_embeddings_and_summary_revisions_unchanged':True,
                        'no_claim_of_native_reindexing':True})
    return {'enabled_by_explicit_private_stage_preflight':True,
            'native_documents_received':len(documents),
            'staged_originals_in_native_selection':len(results),
            'publisher_verified_corrected_passages':sum(len(z['corrected_private_pdf_excerpts']) for z in results),
            'source_originals_with_private_page_evidence':results,
            'source_pages_are_lexical_candidates_not_verified_answers':True,
            'not_live_source_map_cache_or_semantic_index_migration':True,
            'read_only_live_corpus_and_private_stage':True}
