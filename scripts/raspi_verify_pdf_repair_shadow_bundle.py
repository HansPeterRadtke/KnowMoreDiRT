#!/usr/bin/env python3
"""Independent READ-ONLY verification of the complete protected recovery bundle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_system_catalog.source_page_shadow_cohort import verify_private_shadow_bundle
AUDITS=Path('/data/var/knowmoredirt/knowledge-library/search-audit')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--private-bundle',required=True,type=Path)
    args=ap.parse_args()
    if not args.private_bundle.is_absolute() or not args.private_bundle.resolve().is_relative_to(AUDITS.resolve()):
        raise ValueError('read-only verification is scoped to private source audit recovery only')
    print(json.dumps(verify_private_shadow_bundle(bundle_root=args.private_bundle),sort_keys=True))

if __name__=='__main__':main()
