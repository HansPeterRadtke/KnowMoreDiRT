#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from knowmoredirt.evaluation import evaluate_fixture, evaluation_to_dict

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", required=True)
parser.add_argument("--qa", required=True)
parser.add_argument("--json-out", default="")
args = parser.parse_args()
result = evaluate_fixture(args.corpus, args.qa)
print(f"score={result.correct}/{result.total} ({result.score:.3f})")
if args.json_out:
    Path(args.json_out).write_text(json.dumps(evaluation_to_dict(result), indent=2) + "\n")
