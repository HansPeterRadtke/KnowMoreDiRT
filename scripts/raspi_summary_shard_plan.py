#!/usr/bin/env python3
"""Generate a read-only KMD model-source shard plan from actual native state.

Does not touch systemd, restart workers/models, change source ownership, or
modify the native database. Requires a SHA-pinned private observed-rate report.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.summary_shard_planner import compare_assignments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-map',type=Path,required=True)
    p.add_argument('--filedb',type=Path,required=True)
    p.add_argument('--observed-rates',type=Path,required=True)
    p.add_argument('--rate-report-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    for f in (args.source_map,args.filedb,args.observed_rates):
        if not f.is_file() or f.is_symlink():
            raise ValueError('canonical KMD source or rates file missing or symlinked')
    if args.observed_rates.stat().st_size>2*1024*1024:
        raise ValueError('oversized private observed rate report')
    if (args.output.exists() or args.output.is_symlink()
            or not args.output.parent.is_dir() or args.output.parent.is_symlink()):
        raise ValueError('review report output exists or has an unsafe parent')
    blob=args.observed_rates.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=args.rate_report_sha256:
        raise ValueError('recent measured-rate evidence hash changed')
    old=json.loads(blob)
    units=old['units']
    rates={
        'coder':units['knowledge-library-hierarchical-summary.service']['character_throughput']['publisher_chars_per_hour'],
        'q4':units['knowledge-library-hierarchical-summary-q4.service']['character_throughput']['publisher_chars_per_hour'],
    }
    spec=importlib.util.spec_from_file_location('raspi_readonly_source_worker',ROOT/'scripts/raspi_hierarchical_worker.py')
    worker=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    # Read-only on the exact supplied native source map and FileDB.
    worker.SOURCE_MAP=args.source_map
    worker.FILEDB=args.filedb
    available,completed=worker.scope()
    buckets={n:{'pending':0,'pending_characters':0} for n in range(4)}
    for path,_sha,_deriv,characters in available:
        if path not in completed:
            b=buckets[worker.shard_for_original(path,4)]
            b['pending']+=1
            b['pending_characters']+=int(characters)
    result=compare_assignments(buckets,chars_per_hour=rates,
                               current={'coder':[1,2,3],'q4':[0]})
    result['observed_rates_report_sha256']=args.rate_report_sha256
    result['source_map_and_filedb_read_only']=True
    with args.output.open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    args.output.chmod(0o600)
    print(json.dumps({'private_report':str(args.output),
                      'current_makespan_sensitivity_hours':result['current']['sensitivity_hours_at_observed_rates']['makespan'],
                      'minimal_churn_candidate':result['best_with_at_most_one_shard_reassigned'],
                      'source_data_modified':False}))


if __name__=='__main__':
    main()
