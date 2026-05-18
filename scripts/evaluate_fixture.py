#!/usr/bin/env python3
"""Evaluate the current KMD engine on the messy raw-text fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowmoredirt.evaluation import evaluate_fixture, evaluation_to_dict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="tests/fixtures/messy_raw_corpus")
    parser.add_argument("--qa", default="tests/fixtures/messy_raw_corpus_qa.json")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    result = evaluate_fixture(args.corpus, args.qa)
    print(f"score={result.correct}/{result.total} ({result.score:.3f})")
    for category, values in result.by_category.items():
        print(f"{category}: {values['correct']}/{values['total']} ({values['score']:.3f})")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(evaluation_to_dict(result), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

