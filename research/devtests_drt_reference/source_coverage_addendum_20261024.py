"""Record only semantically reviewed additions; preserve preceding 827-file ledger.

A checksum is an integrity aid, NOT proof that an agent understood the text.
"""
import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path('/data/infra')
HERE=Path(__file__).parent
ORIGINAL=HERE/'infra_source_coverage_20261024.json'
base=json.loads(ORIGINAL.read_text()); old={x['path']:x for x in base['entries']}
ADDITIONS=[
 ('docs/guidlines/programming/README.md','semantic_read_complete','All 144 lines in bounded ranges: purpose, correct before fast, state/concurrency, branch assumptions, search and task-visible failure, reusable helpers',[]),
 ('docs/guidlines/testing.md','semantic_read_complete','All 118 lines in bounded ranges: independent oracle, heldout selection, nested optimization leakage, real/failed environment checks and KMD scope validation',[]),
 ('docs/guidlines/agents/context_quality_and_relevance.md','semantic_read_complete','All 37 lines: three relevance purposes, task context, structured model calls, measured numeric/enums, uncertain reviewer agreement',[]),
 ('docs/guidlines/agents/large_sources.md','semantic_read_complete','All 28 lines: read means task-dependent semantic coverage; preserve original and exact source links; inventory is not semantic review',[]),
 ('docs/guidlines/appendix/recording-integration-2026-10-24.md','semantic_read_complete','All source-to-guideline routing and ten source distinctions: research questions vs answer evidence, model experiments pending, preserved originals',[]),
 ('docs/guidlines/original-recordings/agents/Recording 2026-10-20 11-48-38-894.txt','semantic_read_complete','All 13,179 characters in ranges 0–6,000 and 6,000–end: one fast model step, complete structured history, event ordering, external payloads, scoped deletion',[]),
 ('docs/guidlines/original-recordings/agents/Recording 2026-08-05 12-00-31.txt','semantic_read_complete','All 2,801 characters: context-size versus complete agent history, manuals/tools read as needed, DRT plus file database as possible knowledge layer',[]),
 ('docs/guidlines/original-recordings/agents/Recording 2026-09-17 11-49-21-406.txt','semantic_read_partial','Selected four bounded windows of 19,980 characters: agent-owned generic Python and file/db workspace, optional KMD vs SQLite, topic documents to DRT',[[0,591],[9296,11149],[13126,14145],[18751,19724]]),
 ('docs/guidlines/original-recordings/agents/Recording 2026-08-29 16-51-40-798.txt','semantic_read_partial','Nine searched source windows of 122,466 characters: alternative text transformations, embedding/DRT, summarize versus preserve, cost, semantic research; rest is not claimed semantically read',[[89520,90727],[91566,92539],[94431,95404],[101654,102627],[105400,106440],[110892,111871],[116176,117155],[119382,121498],[121896,122466]]),
]
add=[]
for path,status,scope,ranges in ADDITIONS:
 p=ROOT/path
 assert p.exists() and path in old
 previous=old[path]['status']
 assert previous!='semantic_read_complete',path
 b=p.read_bytes()
 add.append({'path':path,'previous_status':previous,'new_status':status,'semantic_scope':scope,'reviewed_character_windows':ranges if ranges else 'entire source','source_size_bytes':len(b),'source_sha256':hashlib.sha256(b).hexdigest()})
counts=Counter(x['status'] for x in base['entries'])
for x in add:
 counts[x['previous_status']]-=1;counts[x['new_status']]+=1
assert sum(counts.values())==base['tracked_file_count']
report={'type':'additive_source_coverage_review_not_full_repo_completion','prior_ledger_sha256':hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),'prior_tracked_file_count':base['tracked_file_count'],'updated_status_counts':dict(counts),'added_semantic_reviews':add,'remaining_limitation':'Other tracked human-readable Infra sources are inventoried or keyword screened only. Neither hashes nor broad inventory establish complete semantic coverage. All original text remains in Infra; none is edited.'}
(HERE/'infra_source_coverage_addendum_20261024.json').write_text(json.dumps(report,indent=2)+'\n')
print('UPDATED_SOURCE_COVERAGE',report['updated_status_counts'])
