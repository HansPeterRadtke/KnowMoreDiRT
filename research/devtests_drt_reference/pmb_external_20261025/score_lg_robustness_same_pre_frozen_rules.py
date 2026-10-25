"""Use exactly the previous gold scorer on separately frozen spaCy LG parse.

No new source retrieval, no change to scorer logic, no model inference reruns.
This is post-hoc model-capacity robustness, not tuned/heldout model selection.
"""
from pathlib import Path
import hashlib
import score_dependency_vs_drs_negation as evaluator
HERE=Path(__file__).parent
FROZEN=HERE/'frozen_spacy_lg_dependency_predictions_20261025.json'
OUT=HERE/'spacy_lg_dependency_vs_published_drs_gold_results_20261025.json'
if __name__=='__main__':
 evaluator.F=FROZEN
 evaluator.OUT=OUT
 evaluator.F_SHA='f529470c7e49d04ab42c0e14604e7d0f4e80800e9feeb63eebc3fd03b790c6a5'
 evaluator.main()
