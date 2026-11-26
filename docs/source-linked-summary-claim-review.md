# Read-only source-linked claim review for recursive summaries

The offline Knowledge Library uses recursive summaries to **find originals**.
Those generated summaries are not guaranteed factual evidence. A review of the
SciNet research paper found a real failure mode: a task-qualified low percentage
was generalized to all tasks, an author-reported percentage lost its experiment
conditions, and a *predicted* GPU benefit was phrased as a measured improvement.

This diagnostic adds source-linked **review candidates**, not automatically
corrected science or a replacement summarizer. It makes the original passages
quick to inspect while preserving all native summary vectors and worker state.
It does **not** need an Internet connection, LLM request, new publication,
new source ingestion, database copy, new indexing job or service restart.

## Run on the existing Raspi files

On Raspi, from /data/src/github/KnowMoreDiRT:

~~~bash
umask 077
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. nice -n 10 python3 \
  scripts/raspi_summary_claim_evidence.py \
  --all-nodes \
  --original-path 'ai-agents/agent-reliability-and-automation/2026-scinet-evaluating-ai-agents-in-relation-aware-scientific-literature-retrieval-0acf27acce.pdf' \
  --output '/data/var/knowmoredirt/knowledge-library/search-audit/SCINET-CLAIM-REVIEW-UNIQUE.json'
~~~

Use a **fresh** private output path each time; an existing file is rejected.
The output file has restrictive owner-only permissions. The default review excerpts include up to two thousand characters around each matched original-source anchor so a figure and its nearby experimental setup can be read together. Use --context-chars to request a shorter review between one hundred and sixty and two thousand characters; this changes context shown, not the original source or summary. The CLI prints only
where it saved the review and how many candidates need inspection. The full
source excerpts and source identity hashes are in the private JSON file.
For independently constructed fixtures, the original-root, derived-root,
source-map, native-db, and completed-ledger arguments are explicit overrides.

By default, only the **document-root** summary is screened. The optional
--all-nodes flag also screens each current child and parent summary against
**only that node's exact original-text interval**, retaining global source
character positions and SHA digests. This exposes where a dropped qualifier
first entered the recursive summary chain. At most sixty-four nodes are
admitted by default; if the current tree has more, the command rejects
rather than silently reporting a partial review. The explicit
--maximum-reviewed-nodes parameter may be set up to one hundred and twenty-eight.

The program checks (and then checks again) the approved publisher-original
SHA, current extracted-text SHA, native FileDB document identity, every
summary node and embedding, complete recursive parent-child spans, and the
exact current root summary revision. It uses read-only SQLite connections
and refuses symlink/path-escape source paths. This is a point-in-time check:
another worker may update an index or original **after** the review finishes.

## Interpret review candidates, not scores of correctness

The numeric locator has three explicitly different kinds. An
**exact_percent_text** match includes the percent sign with the same
spelling as the generated claim. A **numerically_equivalent_percent_text**
match has a percentage with equivalent numeric value but different decimal
precision, such as zero percent versus zero-point-zero percent. A
**bare_number_no_unit** match is only a number without a percent symbol,
often because a PDF table prints the percentage unit in its column heading.
The last two are **candidate locations only**: compare the original table
heading, row, control, units and experiment before accepting any claim.
Neither numerical equality nor matching a bare cell proves a percentage,
let alone that the same metric or test is being described. Values without a
text-layer match remain unresolved, not automatically false.

The quantified_summary_claim label means a generated summary contains a
percentage. The reviewer locates identical numeric text in the verified
extracted original and shows nearby words and page/source-unit labels.
A matching number **does not** establish that the task, metric, model,
control, sample size or attribution is the same. In particular, an
abstract's number can require qualifications from another section or
from a rendered figure.

### Optional original-table row attribution check

For a particular paper with an explicitly confirmed model or subject label,
add `--focus-subject-label 'AgentMemRec'` to the read-only command above.
Use the `--all-nodes` option to inspect child summaries as well as the root.
This **does not infer a label automatically** from a model output or a paper
title. It uses the reviewer-supplied string as literal text and only
screens quantified claims that contain that same focus label.

For a PDF table numeric cell without a percent sign, the extra
`original_extracted_table_row_attribution_candidates` records the closest
actual extracted-source line around the matched number, its original
character offsets, exact content SHA and physical page *label*. It also
indicates whether the focus label appeared immediately **before** that
number on the same extracted line. If not, the result is a prompt for
checking the source table's row and column labels—not proof of an error.
A table can have several side-by-side columns, wrapped rows, or a missing
text layer. Even seeing the label nearby does not establish the right
metric, unit or experiment; a complete original PDF layout review is
required for a consequential scientific statement.

This check exposed a real case in the already archived research paper
*Agent Memory as Recommendation*: a generated child summary associates
**80.3% LoCoMo accuracy** with AgentMemRec, but the original paper's
Table 2 (physical page eight) places the copied 80.3 result on **T-Mem's**
row. The paper reports **77.0** for AgentMemRec with optional write-time
records and **77.4** for its T-Mem reproduction, while **83.2** is its
LongMemEval-S result on all five hundred questions. These numbers are
paper-reported results, not tests on the owner's agent. The review now
locates the numerical values and distinct row text without ever replacing
the live native summary. The verified source-specific correction and exact
publisher-document SHA are retained in the private evidence index.

The new hints are **not** false-claim classifications: a row whose subject
is missing from the local line can be valid because of table formatting,
and a matching string can be unrelated in a parallel column. The output
always retains `automatic_truth_verdict: NONE`. The program leaves the
canonical FileDB, source-map, generated summaries and embedding vectors
unchanged and never accesses a remote language model.

The possible_forecast_reported_as_result label is narrower: an unqualified
GPU performance sentence has a nearby original mention of improving
throughput **conditionally**. It flags an important comparison for review,
but never asserts that the paper has no other measured GPU result. A source
that merely uses a GPU for training or embeddings is not necessarily
evidence of faster relational retrieval.

If a percentage is not found in the extracted text, the status is
not_found_in_extracted_text_check_original_figures, **not** "false".
When the original text near a matched number uses an explicit qualifier,
the output now supplies a short, separately hashed and offset-linked
**possible_missing_qualification_review_cues** array. These are lexical
review hints, not assertions that the summary lost a meaningful condition.
The cue must be within two hundred and forty source characters of the
specific match, even when the displayed evidence has a larger context.
This distinction prevents an unrelated nearby claim's caveat from being
automatically associated with a different percentage. An already repeated
literal qualifier is not listed as missing; equivalent phrasing still needs
human or model interpretation against the original.

The original PDF may present the number only in a figure, table or
illustration. Source-unit labels are taken from the marker containing the
actual matched anchor, not merely from the beginning of a cross-page
excerpt. PDF page markers are not automatically verified against rendered
page images. For a consequential claim, inspect the actual original,
including figures, and retain the human or agent review separately.

The library never changes native summaries, original files, source-map
rows, embeddings, rankings, source permissions or acquisition rights in
this command. Flags do not silently suppress search results or make claims
of certainty. The still-unfinished work is a separately validated,
provenance-preserving way to *improve generation itself* without
invalidating an already active prompt-versioned summary index.

## Tests and repeatable verification

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m unittest -q \
  tests.filesystem_database.test_summary_claim_evidence \
  tests.filesystem_database.test_recent_source_tree_audit
~~~

The synthetic tests include exact source-row attribution candidates, literal
reviewer-supplied subject labels, child-span isolation, correct quantified
observations, conditional GPU claims, missing figure-only numbers, original-byte changes, untrusted
source instructions, decimal percentages, bounded output, exact excerpt
SHAs and correct page labels. The real SciNet run may identify multiple
**candidates**; the independent PDF semantic audit remains necessary to
judge actual drift. The program is a source-linked triage tool, not a
full-corpus accuracy estimate.

Recent private comparative evidence is kept in the existing Raspi
/data/var/knowmoredirt/knowledge-library/search-audit tree. Do not commit
publisher PDF bytes, private owner recordings, raw private research text
or the review JSON files to the code repository.
