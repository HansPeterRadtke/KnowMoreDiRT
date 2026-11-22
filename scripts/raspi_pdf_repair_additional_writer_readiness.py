#!/usr/bin/env python3
"""Read-only known writer/timer inventory for original-PDF source-map migration.

A pause of the two recursive-summary workers is not an all-writer barrier:
extractor and direct native semantic writer own distinct installed timers.
This never stops a timer, restarts a worker, changes native source records,
creates a source holding manifest or grants native migration authority.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.source_page_writer_inventory import (
    WRITERS,CATALOG_ONLY,assess_other_writers,
)

PRIVATE=Path('/data/var/knowmoredirt/knowledge-library/search-audit')
SUMMARY_SCRIPT=ROOT/'scripts/raspi_publisher_pdf_quiesce_readiness.py'


def show_unit(name:str)->dict:
    approved={u for service,timer,_ in WRITERS.values() for u in (service,timer)}
    approved.update(CATALOG_ONLY)
    if name not in approved:
        raise ValueError('refusing unrecognized systemd service inventory')
    result=subprocess.run(['systemctl','show',name,
                  '-p','LoadState','-p','ActiveState','-p','MainPID','--no-pager'],
                  capture_output=True,text=True,timeout=6,check=False)
    if result.returncode:
        raise RuntimeError('systemd writer/timer state unreadable')
    pairs={}
    for line in result.stdout.splitlines():
        if '=' not in line:continue
        key,value=line.split('=',1)
        if key in pairs:raise ValueError('duplicated systemd state field')
        pairs[key]=value
    # systemd timer units have no MainPID property at all. An absent
    # MainPID for a TIMER is the documented unit type, not a fabricated
    # running process. For actual services MainPID must still be present.
    if name.endswith('.timer') and set(pairs)=={'LoadState','ActiveState'}:
        pairs['MainPID']='0'
    if set(pairs)!={'LoadState','ActiveState','MainPID'} or not pairs['MainPID'].isdigit():
        raise ValueError('incomplete actual systemd writer/timer state')
    return {'load_state':pairs['LoadState'],'active_state':pairs['ActiveState'],
            'main_pid':int(pairs['MainPID'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-private-output',required=True,type=Path)
    args=p.parse_args()
    output=args.new_private_output
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or not output.parent.is_dir() or output.parent.is_symlink()
            or not output.parent.resolve(strict=True).is_relative_to(PRIVATE.resolve(strict=True))):
        raise ValueError('writer inventory output must be NEW and inside private Raspi search audit')
    summary=subprocess.run([sys.executable,str(SUMMARY_SCRIPT)],
                     capture_output=True,text=True,timeout=20,check=False)
    if summary.returncode or not summary.stdout.strip():
        raise RuntimeError('independent summary-worker source-boundary readiness unavailable')
    proof=json.loads(summary.stdout)
    units={u for service,timer,_ in WRITERS.values() for u in (service,timer)}
    units.update(CATALOG_ONLY)
    states={name:show_unit(name) for name in sorted(units)}
    result=assess_other_writers(unit_states=states,summary_pause=proof)
    blob=(json.dumps(result,indent=2,ensure_ascii=False)+'\n').encode('utf8')
    fd=os.open(output,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as out:
        out.write(blob);out.flush();os.fsync(out.fileno())
    print(json.dumps({'known_native_writer_services_checked':len(WRITERS),
         'extra_service_and_timer_blocks':len(result['explicit_release_blockers']),
         'reported_safe_to_cutover_live':False,
         'output':str(output)},sort_keys=True))

if __name__=='__main__':main()
