#!/usr/bin/env python3
"""Read-only verification of All2Text PDF figures against native KnowMoreDiRT."""
import argparse
import json
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regions",type=Path,required=True)
    p.add_argument("--visual-context",type=Path,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--source-map-db",type=Path,required=True)
    p.add_argument("--filedb",type=Path,required=True)
    p.add_argument("--derived-text-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    from file_system_catalog.pdf_visual_link import link_pdf_regions
    data=link_pdf_regions(
        region_manifest=args.regions,context_manifest=args.visual_context,
        source_root=args.source_root,source_map_db=args.source_map_db,
        filedb=args.filedb,derived_root=args.derived_text_root,
        independently_rerender=True,
    )
    out=args.output.expanduser()
    if out.exists() or out.is_symlink():
        raise ValueError("refuse to overwrite existing original-linked proof")
    if not out.parent.is_dir():
        raise ValueError("private output parent directory missing")
    out.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    out.chmod(0o600)
    print(json.dumps({
        "original":data["source_relative_path"],
        "native_entry_id":data["native_filesystem_entry_id"],
        "source_map_status":data["source_map_status"],
        "regions":len(data["regions"]),
        "caption_anchors":[r["caption_anchor"]["status"] for r in data["regions"]],
        "output":str(out),
    },indent=2))


if __name__=="__main__": main()
