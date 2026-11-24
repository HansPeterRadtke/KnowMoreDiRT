#!/usr/bin/env python3
"""Attest and compare two predeclared same-snapshot retrieval evaluations.

Does not mutate native indexes, models, source maps or original documents.
This evaluates only known source navigation and never answer correctness.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from file_system_catalog.frozen_source_retrieval_drift import report_from_files

ALLOWED=Path('/data/var/knowmoredirt/knowledge-library/search-audit')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',type=Path,required=True)
    parser.add_argument('--new',type=Path,required=True)
    parser.add_argument('--private-output',type=Path,required=True)
    args=parser.parse_args()
    output=args.private_output
    if (not output.is_absolute() or output.is_symlink() or output.exists()
            or not output.parent.is_dir() or output.parent.is_symlink()
            or not output.resolve().is_relative_to(ALLOWED.resolve())):
        raise ValueError('output must be new file under private KMD search audit root')
    report=report_from_files(args.old,args.new)
    raw=(json.dumps(report,indent=2,ensure_ascii=False)+'\n').encode()
    descriptor=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    ten=report['channel_scores']['rrf_three']['10']
    print('FROZEN_COMPARABLE_SOURCE_LABELS',report['comparable_source_labels'])
    print('CURRENT_NATIVE_ORIGINAL_ROOTS',report['newer_complete_original_roots'])
    print('RRF_THREE_TOP10',ten)
    print('CHANGED_SOURCE_SHA_NOT_SCORED',report['incomparable_source_labels'])
    print('PRIVATE_REPORT',output)

if __name__=='__main__':main()
