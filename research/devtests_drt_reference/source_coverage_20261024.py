"""Auditable content-versus-inventory ledger, never scan or print secrets."""
import json,subprocess,datetime
from pathlib import Path
ROOT=Path('/data/infra'); OUT=Path(__file__).parent/'infra_source_coverage_20261024.json'
paths=[x.decode() for x in subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).split(b'\0') if x]
fully={
'AGENTS.md':'Repo ownership, permission and Git-concurrency rule',
'docs/guidlines/README.md':'Recording authority, semantics, user priority and audio-style instructions',
'docs/guidlines/doc/audio_style.md':'Natural spoken reporting and output formatting',
'docs/guidlines/source_traceability.md':'Proposition/source traceability and preserved negatives',
'docs/guidlines/security_privacy.md':'No credentials or unauthorised mutation, least privilege',
'docs/guidlines/knowledge_retrieval.md':'Dream/world/time/default factual scope, neighbor/DRS alternatives and corpus-only answering',
'docs/guidlines/kmd/knowledge_and_memory_design.md':'KMD file metadata, chunk spans, vectors and DRS scope; name source',
'docs/guidlines/agents/research_and_evidence.md':'Premise and fact separation, conditional derivation, open world, model agreement not proof',
'docs/guidlines/agents/history_storage_and_deletion.md':'History retention and explicit user deletion across all derivatives',
'docs/guidlines/appendix/recording-research-2026-10-24.md':'DRS and OWL/PROV-O/solver alternatives, local validation caveats',
'docs/guidlines/original-recordings/kmd/Recording 2026-08-05 17-22-26-192.txt':'Source file metadata, chunks/offsets, embeddings, iterative and cross-chunk DRS, dream-law and date scopes',
'docs/guidlines/original-recordings/kmd/Recording 2026-08-06 12-09-51-499.txt':'Timmy dream not an official law, real-world unknown, scoped related evidence, semantic evaluation',
'docs/guidlines/original-recordings/kmd/Recording 2026-08-06 17-30-53-705.txt':'Context budget, vectors/query expansion, earlier/later/distant scope, retroactive dream disclosure',
'docs/guidlines/original-recordings/agents/Recording 2026-10-17 23-00-29-387.txt':'Research loops, dense material retained, open questions, comparing texts, formal logic/axiom caveats',
'docs/guidlines/original-recordings/agents/Recording 2026-10-18 18-38-21-960.txt':'Formal knowledge and provenance, hypotheses and falsification, conditional assumptions, model consensus calibration'
}
partial={
'docs/guidlines/programming/README.md':'Programming/test/guideline material read only in a truncated combined output; full semantic coverage not confirmed',
'docs/guidlines/testing.md':'Relevant testing requirements reviewed; full contiguous read not confirmed',
'docs/guidlines/agents/context_quality_and_relevance.md':'Section keyword/line excerpts reviewed; not full text',
'docs/guidlines/agents/large_sources.md':'Selected complete substantive lines and source-handling principles reviewed',
'docs/guidlines/agents/architecture.md':'Selected orchestration/resource sections and lines reviewed',
'docs/guidlines/appendix/recording-integration-2026-10-24.md':'File inventory and selected references, not full text',
'docs/guidlines/appendix/recording-coverage.md':'Recording routing/ledger lines, not semantic completeness',
'docs/guidlines/original-recordings/agents/Recording 2026-10-20 11-48-38-894.txt':'First six thousand of about thirteen thousand characters read, history guideline read in full'
}
rows=[]
for name in paths:
 if name in fully: status='semantic_read_complete';scope=fully[name]
 elif name in partial:status='semantic_read_partial';scope=partial[name]
 elif name.startswith('docs/guidlines/original-recordings/agents/') and name.endswith('.txt'):status='lexical_screen_only';scope='Searched for KMD, DRT, DRS, inference/knowledge/contradiction terms; not read completely'
 else: status='inventory_only';scope='Metadata path listed; no substantive content review claimed'
 rows.append({'path':name,'status':status,'semantic_scope':scope})
from collections import Counter
payload={'snapshot_host':'raspi','repo_path':'/data/infra','snapshot_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'tracked_file_count':len(paths),'counts_by_status':dict(Counter(x['status'] for x in rows)),'method':'git ls-files path inventory. Detailed semantic source readings belong only to status semantic_read_complete. Other statuses are not proof of coverage. No raw credentials or source file contents are stored here. Prior Nitro status and snapshot can differ.','entries':rows}
OUT.write_text(json.dumps(payload,indent=2)+'\n');print('SOURCE_LEDGER',len(rows),payload['counts_by_status'])
