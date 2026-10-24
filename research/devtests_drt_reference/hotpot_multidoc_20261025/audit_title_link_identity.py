"""Post-hoc identity-link source offset audit, without rerunning retrieval.

This DOES use previously consumed human gold to label wins/losses, so samples
are descriptive failure forensics, not a new heldout comparison or tuning set.
"""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json
HERE=Path(__file__).parent
SRC=Path('/data/tmp/kmd-hotpot-original-distractor-20261025.json')
PRED=HERE/'frozen_unlabelled_retrieval_20261025.json'
GOLD=HERE/'hotpot_external_multidoc_gold_results_20261025.json'
OUT=HERE/'title_identity_source_link_audit_20261025.json'

def main():
 if OUT.exists():raise RuntimeError('Completed audit exists; do not overwrite')
 orig=json.loads(SRC.read_text());f=json.loads(PRED.read_text());gold=json.loads(GOLD.read_text())
 assert f['source_sha256']==gold['source_sha256']==hashlib.sha256(SRC.read_bytes()).hexdigest()
 assert len(orig)==len(f['predictions'])==len(gold['cases'])==7405
 rows=[];count=Counter()
 for d,p,g in zip(orig,f['predictions'],gold['cases']):
  assert d['_id']==p['uid']==g['uid']
  old=p['ranks']['fulltext_bm25'][:2];new=p['ranks']['linked_secondhop'][:2]
  if old==new:continue
  source={title:sentences for title,sentences in d['context']}
  matched_title=new[1];match=[]
  for i,sentence in enumerate(source[new[0]]):
   lower=' '+sentence.lower()+' ';needle=' '+matched_title.lower()+' ';at=lower.find(needle)
   if at>=0:match.append({'source_title':new[0],'source_sentence_index':i,'source_sentence_sha256':hashlib.sha256(sentence.encode()).hexdigest(),
     'match_start_in_sentence':at-1,'matched_title':matched_title,'short_source_substring':sentence[max(0,at-55):min(len(sentence),at+len(needle)+65)]})
  gdocs=set(g['gold_doc_titles']);win=(gdocs<=set(new)) and not (gdocs<=set(old));loss=(gdocs<=set(old)) and not(gdocs<=set(new))
  status='win' if win else 'loss' if loss else 'neither'
  count[(d['type'],status,'exact_source_title_mention' if match else 'no_exact_mention')]+=1
  rows.append({'uid':d['_id'],'question':d['question'],'question_type':d['type'],'old_title_pair':old,'new_title_pair':new,
     'human_gold_source_titles':sorted(gdocs),'paired_outcome':status,
     'new_second_article_exactly_mentioned_by_first':bool(match),'source_mention_spans':match[:2]})
 records=[]
 for typ in ('bridge','comparison'):
  for status in ('win','loss','neither'):
   sub=sorted((x for x in rows if x['question_type']==typ and x['paired_outcome']==status),key=lambda x:hashlib.sha256(x['uid'].encode()).hexdigest())
   records.extend(sub[:4])
 output={'original_data_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'original_frozen_predictions_sha256':hashlib.sha256(PRED.read_bytes()).hexdigest(),
    'human_gold_score_sha256':hashlib.sha256(GOLD.read_bytes()).hexdigest(),
    'comparison':'Post-hoc audit of changed second-hop article: exact in-text title mention and whether two human gold supporting articles were recovered. Counts are descriptive, no new tuning.',
    'total_differing_top2_doc_choices':len(rows),
    'source_mention_outcome_counts':[{'type':k[0],'outcome':k[1],'evidence_type':k[2],'count':v} for k,v in sorted(count.items())],
    'sample_link_cases':records,
    'limitations':['Title occurrence in an article does not establish subject coreference or a true relationship; it only supplies an identifiable candidate relation','This audit was run after human labels were examined, so representative examples cannot serve as independent blinded performance evidence','The exact source title text may be a distractor or incidental mention; the graph is lexical and not production KMD DRS','Pre-selected ten article candidate set is an oracle dataset wrapper; global web retrieval quality is unmeasured']}
 OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
 print('CHANGED',len(rows),'COUNT')
 for k,v in sorted(count.items()):print(k,v)
 print('EXAMPLES')
 for x in records[:9]:
  m=x['source_mention_spans']
  print(x['question_type'],x['paired_outcome'],'Q',x['question'][:105],'OLD',x['old_title_pair'],'NEW',x['new_title_pair'],'LINK_SNIPPET',m[0]['short_source_substring'][:120] if m else '-')
if __name__=='__main__':main()
