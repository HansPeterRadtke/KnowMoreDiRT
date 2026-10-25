"""Post-hoc source-cluster robustness for already-frozen HotpotQA retrieval.

Does not tune retrieval or repeat original experiment; source cluster defined
by original first-ranked article rather than gold, preserving related questions.
"""
import hashlib,json,random
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).parent
PRED=HERE/'frozen_unlabelled_retrieval_20261025.json'
GOLD=HERE/'hotpot_external_multidoc_gold_results_20261025.json'
OUT=HERE/'source_clustered_paired_uncertainty_20261025.json'

def bootstrap(cases):
 clusters=defaultdict(list)
 for x in cases: clusters[x['first_article']].append(x['effect'])
 keys=list(clusters)
 rng=random.Random(20261025)
 boots=[]
 for _ in range(1750):
  chosen=[clusters[keys[rng.randrange(len(keys))]] for j in keys]
  denom=sum(len(z) for z in chosen);boots.append(sum(sum(z) for z in chosen)/denom)
 boots.sort()
 return {'n_questions':len(cases),'first_source_clusters':len(keys),'largest_cluster':max(len(x) for x in clusters.values()),
    'paired_difference':round(sum(x['effect'] for x in cases)/len(cases),6),
    'cluster_bootstrap95':[round(boots[int(0.025*len(boots))],6),round(boots[int(0.975*len(boots))],6)],
    'cluster_method':'Resample distinct first-retrieved article title as cluster, carrying all questions associated with that title; random seed 20261025, n=1750'}

def main():
 if OUT.exists():raise RuntimeError('Already analyzed, preserve record')
 f=json.loads(PRED.read_text())['predictions'];g=json.loads(GOLD.read_text())['cases'];assert len(f)==len(g)==7405
 rows=[]
 for a,b in zip(f,g):
  assert a['uid']==b['uid']
  rows.append({'type':b['type'],'first_article':a['ranks']['fulltext_bm25'][0],
     'effect':int(b['methods']['linked_secondhop']['2']['both_docs'])-int(b['methods']['fulltext_bm25']['2']['both_docs'])})
 z={'method':'Post-hoc clustered uncertainty descriptive control, no new retrieval or model inference','original_frozen_pred_sha256':hashlib.sha256(PRED.read_bytes()).hexdigest(),
     'original_external_gold_score_sha256':hashlib.sha256(GOLD.read_bytes()).hexdigest(),
     'groups':{'overall':bootstrap(rows),'bridge':bootstrap([x for x in rows if x['type']=='bridge']),
        'comparison':bootstrap([x for x in rows if x['type']=='comparison'])},
     'limitations':['Clusters by first retrieved Wikipedia article, not original underlying author/research experiment','Bootstrap assumes conditional exchangeability of source clusters and does not establish generalization outside HotpotQA','This controls one dependence type; no adjustment for repeated gold second sources or question paraphrases, and the source title link methods remain fixed']}
 OUT.write_text(json.dumps(z,indent=2)+'\n')
 for group,val in z['groups'].items():print(group,val)
if __name__=='__main__':main()
