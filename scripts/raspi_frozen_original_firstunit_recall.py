#!/usr/bin/env python3
"""Compare exact earlier-frozen original-source labels with current first-unit FTS.

No new gold questions, original download, model calls, native index writes or
source text publication. Exact returned original source identities and frozen
positive publisher files are verified against current native FileDB,
completed acquisition receipts and full derived/publisher SHA. Output is
a new owner-private receipt, not truth, score fusion or rank tuning.
"""
from __future__ import annotations
import argparse
from contextlib import closing
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.frozen_original_firstunit_recall import (
    _gold_rows,evaluate_frozen_firstunit_original_recall,
    complementary_multi_original_source_presence,
)
from file_system_catalog.source_review_audit import safe_file,sha_file

BASE=Path('/data/var/knowmoredirt/knowledge-library')
ORIG=Path('/data/var/Knowledge Library')
DERIVED=BASE/'semantic/text'
NATIVE=BASE/'filesystem.sqlite3'
SOURCE_MAP=BASE/'semantic/source-map.sqlite3'
LEDGER=ORIG/'_management/status.json'
CATALOG=Path('/data/infra/etc/knowledge-library-sources.json')
MAX_GOLD_BYTES=128*1024
MAX_QUERY_BYTES=128*1024
MAX_REPORT_BYTES=12*1024*1024
MAX_METADATA_BYTES=64*1024*1024
MAX_ORIGINAL_BYTES=128*1024*1024
MAX_DERIVED_BYTES=64*1024*1024


def _document(path:Path,max_bytes:int)->dict|list:
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=max_bytes:
        raise ValueError('frozen original-source evaluation input unavailable, symlinked or oversized')
    return json.loads(path.read_bytes())


def _unique(items:list[dict],key:str)->dict[str,dict]:
    if not isinstance(items,list) or len(items)>25000:
        raise ValueError('unbounded original source catalog')
    result={}
    for row in items:
        if not isinstance(row,dict) or not isinstance(row.get(key),str) or row[key] in result:
            raise ValueError('missing or duplicate original source metadata')
        result[row[key]]=row
    return result


def source_attestation(
    *,
    independent:list[dict],
    multi:list[dict],
    original_root:Path,
    derived_root:Path,
    source_map:Path,
    native_db:Path,
    catalog:Path,
    ledger:Path,
) -> dict[str,tuple[str,str,int]]:
    """Full SHA re-hash all known gold sources, even absent ranked positives."""
    gold=_gold_rows(independent,multi)
    positives=sorted({p for item in gold for p in item['known_originals']})
    if not 1<=len(positives)<=38:
        raise ValueError('wrong frozen source relevance population cardinality')
    catalogs=_unique(_document(catalog,MAX_METADATA_BYTES)['sources'],'path')
    receipts=_unique(_document(ledger,MAX_METADATA_BYTES)['items'],'path')
    for path in (source_map,native_db):
        if path.is_symlink() or not path.is_file():
            raise ValueError('current native or source-map database missing or symlinked')
    evidence={}
    with closing(sqlite3.connect(source_map.resolve().as_uri()+'?mode=ro',uri=True,timeout=35)) as source,         closing(sqlite3.connect(native_db.resolve().as_uri()+'?mode=ro',uri=True,timeout=35)) as native:
        source.execute('PRAGMA query_only=ON')
        native.execute('PRAGMA query_only=ON')
        for name in positives:
            source_rows=source.execute(
                'SELECT source_sha256,derived_sha256,derived_path,original_bytes,original_url,original_kind,status '
                'FROM originals WHERE source_path=?',(name,)).fetchall()
            if len(source_rows)!=1 or source_rows[0][6]!='complete':
                raise ValueError('frozen original missing current completed source-map revision')
            sha,der,drel,nbytes,url,kind,_=source_rows[0]
            catalog_row=catalogs.get(name)
            receipt=receipts.get(name)
            if (not isinstance(catalog_row,dict) or not isinstance(receipt,dict)
                    or catalog_row.get('kind')!=kind or catalog_row.get('source_url')!=url
                    or receipt.get('status')!='complete' or receipt.get('sha256')!=sha
                    or receipt.get('bytes_received')!=nbytes or drel!=name+'.txt'):
                raise ValueError('publisher catalog/source map/acquisition receipt revision mismatch')
            original=safe_file(original_root,name)
            derived=safe_file(derived_root,drel)
            if (type(nbytes) is not int or nbytes!=original.stat().st_size
                    or not 0<nbytes<=MAX_ORIGINAL_BYTES
                    or not 0<derived.stat().st_size<=MAX_DERIVED_BYTES):
                raise ValueError('frozen current original exceeds full-byte source proof bounds')
            if sha_file(original)!=sha or sha_file(derived)!=der:
                raise ValueError('frozen publisher original or derived SHA no longer matches current source')
            rows=native.execute("SELECT id,size_bytes FROM filesystem_entries "
                                "WHERE relative_path_display=? AND entry_type='file'",(name,)).fetchall()
            if len(rows)!=1 or type(rows[0][0]) is not int or rows[0][1]!=nbytes:
                raise ValueError('missing/ambiguous native original file ID or source byte count')
            evidence[name]=(sha,der,rows[0][0])
            if source.execute('SELECT source_sha256,derived_sha256,derived_path,original_bytes,original_url,original_kind,status '
                              'FROM originals WHERE source_path=?',(name,)).fetchall()!=source_rows:
                raise ValueError('frozen original source revision changed during physical SHA check')
        # Strict re-read of original/derived and selected native IDs after
        # the initial group to avoid claiming the current native index still
        # matched an earlier observation. Not atomic across independent DBs.
        for name in positives:
            sha,der,ident=evidence[name]
            got=source.execute('SELECT source_sha256,derived_sha256,derived_path,status '
                               'FROM originals WHERE source_path=?',(name,)).fetchall()
            if (len(got)!=1 or got[0][0]!=sha or got[0][1]!=der or got[0][3]!='complete'):
                raise ValueError('frozen original changed before final native verification')
            if sha_file(safe_file(original_root,name))!=sha or sha_file(
                    safe_file(derived_root,got[0][2]))!=der:
                raise ValueError('original publisher or derivative changed during final frozen-source recheck')
            if native.execute("SELECT id FROM filesystem_entries WHERE relative_path_display=? AND entry_type='file'",
                              (name,)).fetchall()!=[(ident,)]:
                raise ValueError('native original FileDB ID changed during frozen-source readback')
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--independent-frozen',type=Path,required=True)
    p.add_argument('--multi-frozen',type=Path,required=True)
    p.add_argument('--expected-independent-gold-sha256',required=True,
                   help='Previously frozen full original independent source gold bytes SHA; never infer immutability from question names')
    p.add_argument('--expected-multi-gold-sha256',required=True,
                   help='Previously frozen full multi-positive original source gold bytes SHA')
    p.add_argument('--predeclared-query-file',type=Path,required=True)
    p.add_argument('--firstunit-source-report',type=Path,required=True)
    p.add_argument('--current-native-multi-report',type=Path,
                   help='Optional separate original-SHA-attested current KMD five-multi case report; presence-union only, NEVER a fused top ten ranking')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--original-root',type=Path,default=ORIG)
    p.add_argument('--derived-root',type=Path,default=DERIVED)
    p.add_argument('--source-map',type=Path,default=SOURCE_MAP)
    p.add_argument('--native-db',type=Path,default=NATIVE)
    p.add_argument('--catalog',type=Path,default=CATALOG)
    p.add_argument('--completed-ledger',type=Path,default=LEDGER)
    args=p.parse_args()
    dest=args.output
    # The original source reports may contain private research materials.
    # Never write the score/excerpt proof outside KMD's private audit tree.
    audit=args.native_db.parent/'search-audit'
    if (not dest.is_absolute() or dest.is_symlink() or dest.exists()
            or dest.suffix!='.json' or not dest.parent.is_dir()
            or dest.parent.is_symlink() or audit.is_symlink() or not audit.is_dir()
            or not dest.resolve().is_relative_to(audit.resolve(strict=True))):
        raise ValueError('frozen original source report must be new under owner-private KMD search-audit')
    for label,value in (
            ('independent',args.expected_independent_gold_sha256),
            ('multi',args.expected_multi_gold_sha256)):
        if not isinstance(value,str) or len(value)!=64 or any(ch not in '0123456789abcdef' for ch in value):
            raise ValueError('expected frozen '+label+' source SHA must be a literal prior 64-character hex digest')
    independent=_document(args.independent_frozen,MAX_GOLD_BYTES)
    multi=_document(args.multi_frozen,MAX_GOLD_BYTES)
    original_sha_inputs={
        'independent':hashlib.sha256(args.independent_frozen.read_bytes()).hexdigest(),
        'multi':hashlib.sha256(args.multi_frozen.read_bytes()).hexdigest()}
    if (original_sha_inputs['independent']!=args.expected_independent_gold_sha256
            or original_sha_inputs['multi']!=args.expected_multi_gold_sha256):
        raise ValueError('frozen gold source file bytes changed since externally predeclared SHA')
    if not isinstance(multi,dict) or not isinstance(multi.get('cases'),list):
        raise ValueError('missing frozen multi-source positive labels')
    query_data=_document(args.predeclared_query_file,MAX_QUERY_BYTES)
    source_data=_document(args.firstunit_source_report,MAX_REPORT_BYTES)
    proofs=source_attestation(independent=independent,multi=multi['cases'],
              original_root=args.original_root,derived_root=args.derived_root,
              source_map=args.source_map,native_db=args.native_db,
              catalog=args.catalog,ledger=args.completed_ledger)
    summary=evaluate_frozen_firstunit_original_recall(
        independent=independent,multi=multi['cases'],
        queries_file=query_data,source_report=source_data,
        expected_current_native_sha_by_original=proofs)
    if (hashlib.sha256(args.independent_frozen.read_bytes()).hexdigest()!=original_sha_inputs['independent']
            or hashlib.sha256(args.multi_frozen.read_bytes()).hexdigest()!=original_sha_inputs['multi']):
        raise ValueError('frozen predeclared source gold bytes changed during current original validation')
    complementary=None
    if args.current_native_multi_report is not None:
        native=_document(args.current_native_multi_report,MAX_REPORT_BYTES)
        complementary=complementary_multi_original_source_presence(
            frozen_multi=multi['cases'],
            frozen_multi_full_file_sha256=original_sha_inputs['multi'],
            verified_firstunit_summary=summary,
            current_native_report=native)
    payload={'schema':'kmd.current_frozen_original_firstunit_source_recall_private.v1',
       'checked_utc':datetime.now(timezone.utc).isoformat(),
       'source_experiment_name':'frozen_original_source_firstunit_two_independent_lexical_ranks',
       'independent_27_gold_sha256':original_sha_inputs['independent'],
       'multi_5_gold_sha256':original_sha_inputs['multi'],
       'frozen_source_gold_sha256_matched_explicit_prior_predeclared_hashes':True,
       'source_query_population_sha256':hashlib.sha256(args.predeclared_query_file.read_bytes()).hexdigest(),
       'original_source_report_sha256':hashlib.sha256(args.firstunit_source_report.read_bytes()).hexdigest(),
       'positive_gold_publisher_originals_full_SHA_native_ID_rechecked_twice':len(proofs),
       'no_original_bytes_or_untrusted_source_excerpt_copied_into_benchmark':True,
       'same_source_original_identity_checked_for_firstunit_hits':True,
       'cannot_infer_native_rank_improvement_from_historical_earlier_73_snapshot':True,
       'candidate_source_recall_is_not_factual_answer_accuracy':True,
       'native_summary_and_original_filedb_unmodified':True,
       'source_recall':summary,
       **({'optional_native_and_original_source_presence_not_rank_fusion':complementary,
           'current_native_multi_original_report_sha256':
                hashlib.sha256(args.current_native_multi_report.read_bytes()).hexdigest()}
          if complementary is not None else {}),
       'original_frozen_multi_and_native_are_independently_issued_snapshots_not_atomic_generation':True}
    raw=(json.dumps(payload,indent=2,ensure_ascii=False)+'\n').encode()
    fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o600)
    with os.fdopen(fd,'wb') as file:
        file.write(raw);file.flush();os.fsync(file.fileno())
    print('FROZEN_SOURCE_ORIGINAL_REFERENCE_RANKS_VERIFIED',
          summary['known_gold_questions'],
          summary['known_positive_labels_including_repeated_paths'],
          summary['separate_source_channel_metrics'],
          flush=True)
    print('OWNER_PRIVATE_FROZEN_SOURCE_RESULT',dest,hashlib.sha256(raw).hexdigest())


if __name__=='__main__':main()
