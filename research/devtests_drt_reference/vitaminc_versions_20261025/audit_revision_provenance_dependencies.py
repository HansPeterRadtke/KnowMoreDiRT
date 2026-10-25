"""Full original source-family dependence audit and clustered uncertainty.

The same Wikipedia revision transition may generate several case IDs and
multiple human claim labels; they are NOT independent scientific observations.
No new NLI calls or changes to original gold or frozen source predictions.
"""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,random
from score_real_revision_contrast import fullgroups,pairs_by_claim
HERE=Path(__file__).parent
SOURCE=Path('/data/tmp/kmd-vitaminc-original-test-20261025.jsonl')
FROZEN=HERE/'frozen_real_revision_source_only_20261025.json'
GOLD=HERE/'human_gold_revision_contrast_results_20261025.json'
OUT=HERE/'real_revision_provenance_cluster_audit_20261025.json'

def bootstrap_family(families,iterations=1200,seed=20261025):
 vals=list(families.values());rng=random.Random(seed);samples=[]
 for i in range(iterations):
  selected=[vals[rng.randrange(len(vals))] for j in vals]
  den=sum(k[1] for k in selected)
  samples.append(sum(k[0] for k in selected)/den if den else 0.0)
 samples.sort()
 return {'revision_families':len(vals),'source_claims':sum(d for n,d in vals),'label_flips':sum(n for n,d in vals),
    'source_family_cluster_bootstrap95':[round(samples[30],6),round(samples[1170],6)],
    'cluster_unit':'same original Wikipedia wiki_revision_id across source cases, not independent claim instances'}

def main():
 if OUT.exists():raise RuntimeError('Do not overwrite already recorded original source-dependency audit')
 raw=SOURCE.read_bytes();original=[json.loads(x) for x in raw.decode().splitlines() if x];groups=fullgroups(original)
 assert len(groups)==5994
 frozen=json.loads(FROZEN.read_text());correct=json.loads(GOLD.read_text())
 assert correct['original_test_sha256']==hashlib.sha256(raw).hexdigest()
 selected={x['case_id'] for x in frozen['case_ids_and_input_hashes_only']}
 full_id=defaultdict(list);sample_id=defaultdict(list);page=defaultdict(set)
 for cid,rows in groups.items():
  rev=rows[0]['wiki_revision_id'];labelpairs=pairs_by_claim(rows)
  n=sum(x['source_label_changed_across_versions'] for x in labelpairs)
  full_id[rev].append((n,len(labelpairs),cid));page[rev].add(rows[0]['page'])
  if cid in selected:sample_id[rev].append((n,len(labelpairs),cid))
 assert all(len(z)==1 for z in page.values())
 aggregate=lambda by:{key:(sum(v[0] for v in vals),sum(v[1] for v in vals)) for key,vals in by.items()}
 lens=Counter(len(v) for v in full_id.values());slens=Counter(len(v) for v in sample_id.values())
 rec={'original_source_sha256':hashlib.sha256(raw).hexdigest(),
  'human_gold_original_score_sha256':hashlib.sha256(GOLD.read_bytes()).hexdigest(),
  'original_real_four_way_groups':len(groups),
  'independent_wikipedia_revision_transition_families_in_original_subset':len(full_id),
  'groups_per_revision_transition_distribution':{str(k):v for k,v in sorted(lens.items())},
  'frozen_sample_groups':len(selected),'frozen_sample_distinct_revision_families':len(sample_id),
  'frozen_groups_per_revision_distribution':{str(k):v for k,v in sorted(slens.items())},
  'original_human_label_flip_frequency_clustered_by_revision_id':bootstrap_family(aggregate(full_id)),
  'sample_label_flip_frequency_clustered_by_revision_id':bootstrap_family(aggregate(sample_id)),
  'examples_of_many_claims_derived_from_one_revision':[{'revision_id':k,'page':next(iter(page[k])),'n_distinct_case_ids':len(v),'case_ids':[x[2] for x in v][:8]} for k,v in sorted(full_id.items(),key=lambda x:-len(x[1]))[:10]],
  'scientific_limitations':['A distinct Wikipedia revision transition is an AUTHOR/PAGE dependency grouping, NOT an independent new experiment or factual truth.',
    'Different Wikipedia article revisions may themselves copy the same source study, so revision clustering still overstates world-level evidentiary independence.',
    'Clustering by revision ID mitigates within-edit generated claim dependence; no independent valid-time timestamp is supplied in test file.',
    'Bootstrap confidence intervals only reflect variation across these intentionally contrastive publisher-curated revisions, not natural news claims or all Wikipedia pages.',
    'Model agreement or majority citation remains non-independent without underlying experiment/dataset trace and validation.']}
 OUT.write_text(json.dumps(rec,indent=2)+'\n')
 print('SOURCE_DEPENDENCE',len(groups),'caseids',len(full_id),'distinct_revision_families','group_multiplicity',lens)
 print('LABEL_FLIP_CLUSTERED_ORIGINAL',rec['original_human_label_flip_frequency_clustered_by_revision_id'])
 print('SAMPLE',len(selected),'caseIDs',len(sample_id),'families',rec['sample_label_flip_frequency_clustered_by_revision_id'])
if __name__=='__main__':main()
