#!/usr/bin/env python3
"""Reproduce fixed ten-original source recall vs optional official links.

Reads THREE exact prior private SHA-attested 73-case reports. Independently
rehashes all current source-known originals and every candidate link actually
consumed by the most expansive fixed ten-slot policy; validates acquisition
catalog, derivative SHA and unique native FileDB original identity, with
separate read-only SQLite data_version checks. No model inference, retrieval
re-ranking, new source acquisition, index write or owner-facing report.
"""
from __future__ import annotations
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.frozen_reference_budget import compare_frozen_reference_budgets
from file_system_catalog.source_review_audit import safe_file,sha_file

PRIVATE=Path('/data/var/knowmoredirt/knowledge-library/search-audit')
CORPUS=Path('/data/var/Knowledge Library')
DERIVED=Path('/data/var/knowmoredirt/knowledge-library/semantic/text')
SOURCE_MAP=Path('/data/var/knowmoredirt/knowledge-library/semantic/source-map.sqlite3')
NATIVE=Path('/data/var/knowmoredirt/knowledge-library/filesystem.sqlite3')
CATALOG=Path('/data/infra/etc/knowledge-library-sources.json')
LEDGER=CORPUS/'_management/status.json'
DEFAULT_NATIVE=PRIVATE/'20261124-wave74-frozen-73-same-snapshot-current-native-original-label-recall.json'
DEFAULT_HTML=PRIVATE/'20261124-wave81-73-frozen-source-labels-original-HTML-link-recovery-results.json'
DEFAULT_MD=PRIVATE/'20261124-wave81-original-73-frozen-gold-Markdown-links-after-code-span-hardening.json'


def _load_input(path:Path):
    if (not isinstance(path,Path) or not path.is_absolute() or path.is_symlink()
            or not path.is_file() or not path.resolve().is_relative_to(PRIVATE.resolve())
            or not 100<=path.stat().st_size<=5_000_000):
        raise ValueError('reference quality evaluation requires an existing bounded private frozen source receipt')
    raw=path.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def _unique_items(values,key):
    rows={}
    for x in values:
        if not isinstance(x,dict) or not isinstance(x.get(key),str) or not x[key] or x[key] in rows:
            raise ValueError('source manifest/ledger contains ambiguous or unbounded approved entry')
        rows[x[key]]=x
    return rows


def _remember(expect:dict,path:str,sha:str):
    prior=expect.get(path)
    if prior is not None and prior!=sha:
        raise ValueError('frozen official original SHA differs across parent, target or known-source label')
    expect[path]=sha


def revalidate_current_source_paths(native:dict,html:dict,markdown:dict,comparison:dict)->dict:
    """Private source QA; does NOT create a cross-store atomic reader lease."""
    catalog_blob=CATALOG.read_bytes();ledger_blob=LEDGER.read_bytes()
    catalog=_unique_items(json.loads(catalog_blob)['sources'],'path')
    ledger=_unique_items(json.loads(ledger_blob)['items'],'path')
    casesN=_unique_items(native['cases'],'id')
    casesH=_unique_items(html['cases'],'id')
    casesM=_unique_items(markdown['cases'],'id')
    expected={};selected_links=[]
    for id,case in casesN.items():
        for path,revision in case['source_revisions'].items():
            _remember(expected,path,revision['original_sha256'])
        chosen=set(next(x for x in comparison['cases'] if x['id']==id)['methods']['5']['added_original_link_candidates'])
        for link in casesH[id]['approved_source_link_proofs']:
            if link['target_original_path'] in chosen:
                _remember(expected,link['parent_original_path'],link['source_link_original_SHA'])
                _remember(expected,link['target_original_path'],link['target_original_receipt_SHA'])
                selected_links.append(link['target_original_path'])
        for link in casesM[id]['firstparty_approved_link_candidates']:
            if link['target_original'] in chosen:
                _remember(expected,link['parent_original'],link['publisher_original_SHA256'])
                _remember(expected,link['target_original'],link['target_receipt_SHA256'])
                selected_links.append(link['target_original'])
    if not 79<=len(expected)<=650:
        raise ValueError('source known gold or selected first-party link candidates lost scope')
    if any(path not in catalog or path not in ledger for path in expected):
        raise ValueError('original source candidate missing from approved catalog or completed receipt')
    totals={'original_bytes_sha_read':0,'extracted_bytes_sha_read':0,'native_original_IDs_verified':0}
    revision_map={}
    with sqlite3.connect(SOURCE_MAP.resolve().as_uri()+'?mode=ro',uri=True,timeout=30) as source,\
         sqlite3.connect(NATIVE.resolve().as_uri()+'?mode=ro',uri=True,timeout=30) as native_db:
        source.execute('PRAGMA query_only=ON');native_db.execute('PRAGMA query_only=ON')
        source_start=source.execute('PRAGMA data_version').fetchone()[0]
        native_start=native_db.execute('PRAGMA data_version').fetchone()[0]
        # Read SELECTED source-map/native identities before beginning costly
        # publisher byte hashing. Resident indexers are allowed to add
        # unrelated rows while a long audit is in progress; a global
        # `PRAGMA data_version` increase is NOT itself a stale cited source.
        # Reject if the identity or revision of ANY selected path changes,
        # regardless of when in the hashing loop that path was checked.
        selected_before={}
        for path,expected_sha in sorted(expected.items()):
            rows=source.execute('SELECT source_sha256,derived_sha256,derived_path,status FROM originals WHERE source_path=?',(path,)).fetchall()
            ids=native_db.execute("SELECT id FROM filesystem_entries WHERE entry_type='file' AND relative_path_display=?",(path,)).fetchall()
            if (len(rows)!=1 or rows[0][0]!=expected_sha or rows[0][3]!='complete'
                    or len(ids)!=1 or type(ids[0][0]) is not int or ids[0][0]<1):
                raise ValueError('selected source-map or native FileDB identity not current before original hash evaluation')
            selected_before[path]=(rows[0],ids[0][0])
        for path,expected_sha in sorted(expected.items()):
            item=catalog[path];receipt=ledger[path]
            if (receipt.get('status')!='complete' or receipt.get('sha256')!=expected_sha
                    or receipt.get('url')!=item.get('url')
                    or type(receipt.get('bytes_received')) is not int):
                raise ValueError('approved acquisition receipt or publisher identity is not current')
            versions=source.execute('SELECT source_sha256,derived_sha256,derived_path,status FROM originals WHERE source_path=?',(path,)).fetchall()
            if (len(versions)!=1 or versions[0]!=selected_before[path][0]
                    or versions[0][0]!=expected_sha or versions[0][3]!='complete'):
                raise ValueError('selected source-map original or derived revision changed during file verification')
            orig=safe_file(CORPUS,path)
            if orig.stat().st_size!=receipt['bytes_received'] or orig.stat().st_size>256*1024*1024:
                raise ValueError('unbounded or changed publisher original size')
            if sha_file(orig)!=expected_sha:
                raise ValueError('approved publisher original contents no longer match original SHA')
            derivative=safe_file(DERIVED,versions[0][2])
            if derivative.stat().st_size>256*1024*1024:
                raise ValueError('unbounded current derived original text size')
            if sha_file(derivative)!=versions[0][1]:
                raise ValueError('approved derived original contents no longer match current SHA')
            ids=native_db.execute("SELECT id FROM filesystem_entries WHERE entry_type='file' AND relative_path_display=?",(path,)).fetchall()
            if len(ids)!=1 or type(ids[0][0]) is not int or ids[0][0]<1:
                raise ValueError('source original missing unique current native FileDB identity')
            if ids[0][0]!=selected_before[path][1]:
                raise ValueError('selected original native FileDB identity changed during file verification')
            revision_map[path]=(versions[0][0],versions[0][1],versions[0][2],ids[0][0])
            totals['original_bytes_sha_read']+=orig.stat().st_size
            totals['extracted_bytes_sha_read']+=derivative.stat().st_size
            totals['native_original_IDs_verified']+=1
        source_unrelated_writes_seen=(source.execute('PRAGMA data_version').fetchone()[0]!=source_start)
        native_unrelated_writes_seen=(native_db.execute('PRAGMA data_version').fetchone()[0]!=native_start)
        # These global version changes may include harmless unrelated new
        # documents. Each SELECTED path's pre/post source-map SHA and native
        # identity are checked again in fresh connections below. The audit
        # still does NOT claim an atomic reader lease across stores.
    if CATALOG.read_bytes()!=catalog_blob or LEDGER.read_bytes()!=ledger_blob:
        raise ValueError('approved original publisher acquisition/catalog revision changed during evaluation')
    # A fresh read-only source-map/ID handle catches committed updates after
    # the first one was released. Full physical bytes were each rehashed
    # earlier, not kept under a cross-SQLite immutable admission protocol.
    with sqlite3.connect(SOURCE_MAP.resolve().as_uri()+'?mode=ro',uri=True,timeout=30) as sm,\
         sqlite3.connect(NATIVE.resolve().as_uri()+'?mode=ro',uri=True,timeout=30) as nd:
        sm.execute('PRAGMA query_only=ON');nd.execute('PRAGMA query_only=ON')
        for path,old in revision_map.items():
            current=sm.execute('SELECT source_sha256,derived_sha256,derived_path,status FROM originals WHERE source_path=?',(path,)).fetchall()
            id_rows=nd.execute("SELECT id FROM filesystem_entries WHERE entry_type='file' AND relative_path_display=?",(path,)).fetchall()
            if len(current)!=1 or current[0][:3]!=old[:3] or current[0][3]!='complete' or id_rows!=[(old[3],)]:
                raise ValueError('publisher source-map or native FileDB identity no longer current after benchmark')
    return {'global_source_map_commits_observed_during_audit':source_unrelated_writes_seen,
            'global_native_FileDB_commits_observed_during_audit':native_unrelated_writes_seen,
            'selected_source_map_revision_and_native_ID_verified_before_during_and_after_all_hashes':True,
            'unrelated_global_writer_commits_are_not_proof_of_selected_original_staleness':True,
            'current_full_source_publisher_and_derived_sha_verified':len(expected),
            'unique_source_IDs_verified':totals['native_original_IDs_verified'],
            'input_selected_link_occurrences_verifiable':len(selected_links),
            'publisher_original_bytes_stream_SHA_read':totals['original_bytes_sha_read'],
            'derived_original_bytes_stream_SHA_read':totals['extracted_bytes_sha_read'],
            'current_catalog_ledger_source_map_and_native_IDs_rechecked_after_full_hashes':True,
            'cross_store_atomic_generation_reader_admission_not_claimed':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native',type=Path,default=DEFAULT_NATIVE)
    ap.add_argument('--html',type=Path,default=DEFAULT_HTML)
    ap.add_argument('--markdown',type=Path,default=DEFAULT_MD)
    ap.add_argument('--private-output',type=Path,required=True)
    args=ap.parse_args()
    output=args.private_output
    if (not output.is_absolute() or output.is_symlink() or output.exists()
            or not output.parent.is_dir() or output.parent.is_symlink()
            or not output.resolve().is_relative_to(PRIVATE.resolve())):
        raise ValueError('refusing unsafe or already occupied private frozen budget result file')
    native,nsha=_load_input(args.native)
    html,hsha=_load_input(args.html)
    md,msha=_load_input(args.markdown)
    report=compare_frozen_reference_budgets(native,html,md,native_input_sha256=nsha)
    result=revalidate_current_source_paths(native,html,md,report)
    report['checked_utc']=datetime.now(timezone.utc).isoformat()
    report['input_HTML_source_study_sha256']=hsha
    report['input_Markdown_source_study_sha256']=msha
    report['current_source_and_link_candidate_reverification']=result
    report['live_KMD_native_ranking_changed']=False
    report['original_publisher_or_derived_source_files_written']=False
    report['no_source_download_or_model_inference']=True
    blob=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    fd=os.open(output,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as file:
        file.write(blob);file.flush();os.fsync(file.fileno())
    print('FROZEN_FIXED_SOURCE_QUESTIONS',report['frozen_original_questions'])
    print('KNOWN_ORIGINALS_IN_ALL_QUESTIONS',report['source_known_original_labels'])
    print('CURRENT_ORIGINAL_AND_DERIVED_SHA_STREAM_VERIFIED',result['current_full_source_publisher_and_derived_sha_verified'])
    for group in ('previous_46','independent_27','all'):
        for k in ('10','9','8','7','5'):
            v=report['group_scores'][group][k]
            print('FIXED_TEN_SLOT_SOURCE_LABEL_RECALL',group,k,v['source_known_label_hits'],v['net_delta_hits'])
    print('NO_DEFAULT_NATIVE_KMD_RANK_MODIFICATION',True)
    print('PRIVATE_EVIDENCE',output)

if __name__=='__main__':main()
