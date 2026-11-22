"""Strict source identity for optional read-only third-channel document discovery.

Document navigation only: never mistake embeddings, titles, summaries or a
filename match for original factual proof. Publisher page locations remain
unknown until separate original physical-page checks succeed.
"""
from __future__ import annotations
from pathlib import Path
import sqlite3

from .source_review_audit import safe_file,sha_file


def verify_selected_source(path:str,*,source_versions:dict,source_map:sqlite3.Connection,
                           approval_ledger:dict,catalog:dict,
                           original_root:Path,derived_root:Path)->dict:
    """Fail closed for exactly one current, uniquely completed original.

    The source-map SQLite connection must be read-only. Caller must have
    verified unique catalogue and approved-receipt path identities.
    """
    if source_map.execute('PRAGMA query_only').fetchone()!=(1,):
        raise ValueError('source citation verification requires query-only source map')
    if (not isinstance(path,str) or not path or path.startswith('/')
            or '\\' in path or any(seg in ('','.','..') for seg in path.split('/'))
            or path not in source_versions or path not in approval_ledger
            or path not in catalog):
        raise ValueError('source not uniquely present in current original catalog')
    source_sha,der_sha,chars=source_versions[path]
    if (not isinstance(source_sha,str) or len(source_sha)!=64
            or not isinstance(der_sha,str) or len(der_sha)!=64
            or type(chars) is not int or chars<1):
        raise ValueError('invalid current publisher source-map revision identity')
    ledger=approval_ledger[path]
    rows=source_map.execute('''SELECT source_sha256,derived_sha256,derived_path,
                    status,original_bytes FROM originals WHERE source_path=?''',(path,)).fetchall()
    if (len(rows)!=1 or rows[0][0]!=source_sha or rows[0][1]!=der_sha
            or rows[0][2]!=path+'.txt' or rows[0][3]!='complete'
            or rows[0][4]!=ledger.get('bytes_received')
            or ledger.get('sha256')!=source_sha or ledger.get('status')!='complete'):
        raise ValueError('source-map/approved original revision changed during document discovery')
    original=safe_file(original_root,path)
    derived=safe_file(derived_root,path+'.txt')
    if (sha_file(original)!=source_sha or sha_file(derived)!=der_sha
            or original.stat().st_size!=ledger['bytes_received']):
        raise ValueError('actual publisher original or derived text SHA differs from source ledger')
    source=catalog[path]
    return {
        'original_path':path,'title':source.get('title') or Path(path).name,
        'source_url':source.get('source_url') or source.get('url'),
        'original_kind':source.get('kind'),
        'publication_year':source.get('publication_year'),
        'license_or_rights_note':source.get('rights_note') or source.get('license'),
        'original_sha256':source_sha,'current_derived_sha256':der_sha,
        'publisher_original_actual_sha256_verified':True,
        'source_map_derivative_actual_sha256_verified':True,
        'whole_source_size_matches_completed_acquisition':True,
        'individual_original_physical_pdf_pages_verified':False,
        'factual_answer_or_claim_supported_by_retrieval_alone':False,
        'source_scope':'original navigation only; inspect governing source section, original page and discourse before asserting facts',
    }
