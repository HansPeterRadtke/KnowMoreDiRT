"""Cross-literature discovery with genuine dated clinical source data.

Untrained, narrow source-aware lexical-bridge experiment: candidate from two
separate pre-1986 literatures and a real 1989 independent clinical test.
This is neither LLM discovery, general automatic DRS, nor causal proof.
Original PubMed/EuropePMC records remain outside Git, with full source hashes.
Never silently fill unavailable 1986 original text from the model.
"""
import hashlib, html, json, re
from pathlib import Path

ROOT=Path('/data/tmp/kmd-raynaud-originals-20261024')
HERE=Path(__file__).parent
IDS={'1975_viscosity_heterogeneous':'53042','1976_raynaud_cold_viscosity':'58309',
     '1977_primary_negative':'72898','1980_raynaud_viscosity':'6775675',
     '1984_fish_oil_peripheral':'6320945','1986_swanson_hypothesis':'3797213',
     '1989_randomized_trial':'2536517'}
EXPECTED_SHA={
 '53042':'81d1cd48d708c808d804c7e3313f71a2ab2e9f4720eb60a577d11a0137ea0a4e',
 '58309':'7ed23e6cadc0e1ca3d387c9e25d181dd63a045552175a5f00a40a752e6270eff',
 '72898':'a945f811c67130fe38082dc5e1b40cd1fed92d63326243183005117f88491532',
 '6775675':'2a36297ca7d97ac0b3ba507d76830171de0b045afa244e288d870025eaaaed9e',
 '6320945':'afaf88a9095965fdbcbacd6175c29a4892441759954d3ca1f7d3558f7bf3297c',
 '3797213':'8f802aca16f929c013e723eb75b3d88de89456c98e1029653df88fd68a1f11a3',
 '2536517':'3f7190388771361ddd4968fb0ca9c0dbf75e3686beb1df99b626a383e7290238',
}
OUT=HERE/'raynaud_archival_discovery_results_20261024.json'

def normalize_abstract(t):
 return re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',t or ''))).strip()

def source_record(pmid):
 p=ROOT/(pmid+'.json');raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED_SHA[pmid],pmid
 doc=json.loads(raw)
 assert str(doc.get('pmid'))==pmid
 a=normalize_abstract(doc.get('abstractText'))
 return {'id':pmid,'pub_year':int(doc['pubYear']),'title':doc['title'],
         'authors':doc['authorString'],'doi':doc.get('doi'),
         'source_uri':'https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/',
         'source_record_sha256':EXPECTED_SHA[pmid], 'abstract_sha256':hashlib.sha256(a.encode()).hexdigest(),
         'abstract':a,'abstract_available':bool(a)}

def exact_evidence(doc,pattern):
 m=re.search(pattern,doc['abstract'],re.I)
 if not m:return None
 a,b=m.span();return {'source_id':doc['id'],'exact_char_offsets':[a,b],
                     'exact_evidence':doc['abstract'][a:b],
                     'parent_abstract_sha256':doc['abstract_sha256']}

def full_claim(doc,pattern,category,target,polarity,scope,limitations):
 evidence=exact_evidence(doc,pattern)
 assert evidence is not None,(doc['id'],pattern)
 return {'claim_category':category,'relation_target':target,'observational_polarity':polarity,
         'population_scope':scope,'publication_year':doc['pub_year'],
         'status':'observed_in_study_not_universally_established',
         'source_author':doc['authors'],'source_doi':doc['doi'],'source_uri':doc['source_uri'],
         'source_provenance':evidence,'known_limits':limitations}

def extract_published_studies(doc):
 id=doc['id']
 if id=='6320945':
  return full_claim(doc,r'a statistically significant reduction in whole blood viscosity was observed at seven weeks',
    'fish_oil_reduces_viscosity','whole blood viscosity','decreased','peripheral_arterial_disease_epa_rich_fish_oil',
    ['Independent PAD group, not Raynaud patients','No transfer to clinical disease outcome without an additional causal premise'])
 if id=='53042':
  assert exact_evidence(doc,r'Sixteen patients had scleroderma')
  return full_claim(doc,r'mean plasma viscosity was significantly .*? elevated over established normal controls',
    'raynaud_secondary_dominated_viscosity','blood viscosity','elevated','Raynaud_cohort_16_of_20_scleroderma',
    ['Study primarily involves secondary disease, not primary','Association does not prove viscosity causes disease'])
 if id=='58309':
  return full_claim(doc,r'viscosity of the patients\x27 blood was significantly higher .*? low shear-rate',
    'raynaud_observed_viscosity','blood viscosity','elevated','Raynaud_10_patients_phenotype_not_resolved',
    ['10 Raynaud and 10 control participants','Cold/shear-dependent, disease subtype not established'])
 if id=='72898':
  claim=full_claim(doc,r'no significant difference, however, in whole-blood and plasma viscosity between ten young women with primary Raynaud\x27s phenomenon and ten age and sex matched controls',
    'primary_raynaud_no_detected_viscosity_difference','blood viscosity','no_detected_difference','primary_Raynaud_10_vs_10',
    ['Failure to find a difference is not proof all primary Raynaud has normal viscosity',
     'Sample ten patients and ten controls'])
  claim['additional_intervention_negative']=exact_evidence(doc,r'Venesection, which reduces plasma viscosity, did not change the ease with which Raynaud\x27s phenomenon was provoked in five patients')
  assert claim['additional_intervention_negative']
  return claim
 if id=='6775675':
  return full_claim(doc,r'Whole-blood viscosity .*? was significantly increased in 29 females with Raynaud\x27s syndrome',
    'raynaud_observed_viscosity','blood viscosity','elevated','Raynaud_women_29_subtype_not_determined',
    ['Subtype unspecified','Disease mechanism may not be single-factor viscosity'])
 return None

def parse_trial(doc):
 assert doc['id']=='2536517' and doc['pub_year']==1989
 t=doc['abstract']
 assert 'primary, but not secondary' in t
 n=re.search(r'Thirty-two patients with primary or secondary Raynaud\x27s phenomenon were randomly assigned',t)
 assert n
 d=re.search(r'increased from\s*([\d.]+)\s*\+/-\s*([\d.]+)\s*minutes baseline to\s*([\d.]+)\s*\+/-\s*([\d.]+)\s*minutes at six weeks\s*\(p\s*=\s*([\d.]+)\)',t,re.I)
 if not d:
  # Keep a clear diagnostic rather than rewriting the original report's gold.
  idx=t.find('increased from');raise ValueError('trial timing source text not parsed: '+repr(t[idx:idx+195]))
 a,b,c,e,p=map(float,d.groups())
 n1=re.search(r'Five of 11 patients .*? compared with one of nine patients .*? \(p\s*=\s*([\d.]+)\)',t,re.I)
 assert n1
 clin=exact_evidence(doc,r'conclude that the ingestion of fish oil improves tolerance to cold exposure and delays the onset of vasospasm in patients with primary, but not secondary, Raynaud\x27s phenomenon')
 assert clin
 return {'trial_authors':doc['authors'],'doi':doc['doi'],'source_id':doc['id'],'publication_year':doc['pub_year'],
   'design_text':doc['abstract'][n.start():n.end()], 'total_randomized':32,
   'treatment':'fish oil vs olive oil placebo; subgroup primary versus secondary',
   'measured_time_to_onset_baseline_minutes':a,'reported_plus_minus_at_baseline':b,
   'measured_time_to_onset_six_weeks_minutes':c,'reported_plus_minus_at_six_weeks':e,
   'descriptive_change_minutes':round(c-a,3),
   'reported_p_value':p,'reported_numbers_source':{'offsets':[d.start(),d.end()],'text':t[d.start():d.end()]},
   'reported_primary_vasospasm_absence_fish_oil':'5 of 11', 'reported_primary_vasospasm_absence_placebo':'1 of 9',
   'descriptive_risk_difference':round(5/11-1/9,5),
   'reported_subgroup_result':'benefit in primary, not established in secondary',
   'subgroup_conclusion_source':clin,
   'no_causal_mechanism_proven':True,
   'critical_limitations':['Small n=32 full trial, subgroup numbers small; no patient-level data to independently recompute p values',
      'Primary and secondary effects cannot be silently pooled as identical',
      'Original trial did not prove effect mediated only by whole-blood viscosity',
      'Published plus-minus statistics have no provided covariance; cannot compute paired uncertainty from aggregate text alone']}

def run():
 if OUT.exists():raise RuntimeError('Completed output exists; do not redo or overwrite')
 docs={key:source_record(id) for key,id in IDS.items()}
 early={k:d for k,d in docs.items() if d['pub_year']<1986}
 assert len(early)==5
 # This is an intentionally simple, auditable lexical discovery operation: a
 # biologically typed mediation link is a HYPOTHESIS, not entailment.
 docs_fish=[d for d in early.values() if re.search(r'fish oil|eicosapentaenoic acid',d['title']+' '+d['abstract'],re.I) and re.search('viscosity',d['abstract'],re.I)]
 docs_raynaud=[d for d in early.values() if re.search('raynaud',d['title'],re.I) and re.search('viscosity',d['abstract'],re.I)]
 assert docs_fish and docs_raynaud
 # Same vocabulary without a shared patient population is not proof.
 join={'shared_literal_concept':'blood viscosity','fish_oil_source_pmids':[x['id'] for x in docs_fish],
       'raynaud_source_pmids':[x['id'] for x in docs_raynaud],
       'candidate':'Could fish-oil supplementation affect Raynaud symptoms through a change in blood viscosity?',
       'inference_type':'plausible_mediated_hypothesis_not_logical_entailment',
       'not_entailed_reasons':['Vascular-disease treatment data are not automatically transportable to Raynaud primary or secondary cohorts',
            'Association of Raynaud with viscosity is not causal proof','Pre-1986 primary-Raynaud null study and venesection intervention challenge a simplistic mediator',
            'Trial and prior-art novelty were not established using the early source subset alone'],
       'no_direct_claim_in_curated_pre1986_abstracts':all(not(re.search('fish oil',d['abstract'],re.I) and re.search('raynaud',d['abstract'],re.I)) for d in early.values()),
       'not_exhaustive_world_prior_art_search':True}
 claims=[x for d in early.values() if (x:=extract_published_studies(d))]
 assert len(claims)==len(early)
 sw=docs['1986_swanson_hypothesis'];trial=docs['1989_randomized_trial']
 assert not sw['abstract_available']
 trial_extracted=parse_trial(trial)
 assert 'swanson' not in trial['authors'].lower()
 results={
 'purpose':'Source-grounded discovery potential and counterexamples on published independent studies; not production model DRS, not automated novel scientific discovery',
 'sources':[{k:d[k] for k in ('id','pub_year','title','authors','doi','source_uri','source_record_sha256','abstract_sha256','abstract_available')} for d in docs.values()],
 'pre1986_automatic_lexical_literature_bridge':join,
 'pre1986_automatic_pattern_extractions':claims,
 'original_swanson_paper':{'id':sw['id'],'year':1986,'author':sw['authors'],'title':sw['title'],'abstract_in_original_pubmed':'UNAVAILABLE',
    'cannot_extract_1986_full_text_from_metadata_alone':True,'observational_status':'historical literature-based hypothesis, not an empirical fish-oil Raynaud trial'},
 'independent_followup_randomized_study':trial_extracted,
 'as_of_date_controls':{'as_of_1985_12_31_source_pmids':sorted(d['id'] for d in early.values()),
     'excluded_future_pmids':[sw['id'],trial['id']],
     '1986_hypothesis_available_by_1986_not_1985':True,
     '1989_trial_unavailable_for_1986_hypothesis_generation':True},
 'negative_and_mismatch_controls':{
     '1977_primary_null_venesection_negative_retained':any(x['claim_category']=='primary_raynaud_no_detected_viscosity_difference' and x.get('additional_intervention_negative') for x in claims),
     '1975_cohort_dominated_by_secondary_not_primary':any(x['population_scope']=='Raynaud_cohort_16_of_20_scleroderma' for x in claims),
     'a_positive_1989_primary_trial_does_not_validate_viscosity_only_mechanism':True,
     '1989_no_detected_secondary_benefit_is_not_proof_of_harm':True,
     'publication_dates_cannot_be_swapped_without_leakage':True,
     'quoted_hypothesis_not_equivalent_to_clinical_observation':True},
 'limitations':['Curated small set of actual papers and narrow hand-specified text patterns; the system did NOT discover all relevant PubMed articles by itself',
 'A lexical shared-concept join produces a candidate question, not logically valid causal inference or demonstrated novel discovery',
 'Author identity and study group are read from metadata and bounded source spans, not model-extracted DRS',
 'An original 1986 abstract is unavailable; no pretending its title is full original text',
 'The later RCT interpretation relies on reported paper authors and text; there was no independent clinical replication or patient-level reanalysis',
 'Other Raynaud studies were not exhaustively retrieved and current treatment recommendations are outside this historical discovery check',
 'Some original 1975/1976/1977 differences are scope, cohort and measurement condition differences, not strict logical contradiction']}
 OUT.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
 print('EARLY_DOCS',len(early),'LEXICAL_JOIN',len(docs_fish),'fish',len(docs_raynaud),'Raynaud','TARGET HYPOTHESIS',join['candidate'])
 print('CONFLICTING_MEDIATOR',results['negative_and_mismatch_controls'])
 print('FUTURE_RCT',trial_extracted['descriptive_change_minutes'],'minutes','primary+','secondary not established','total N',trial_extracted['total_randomized'])
 print('ASOF',results['as_of_date_controls'])
if __name__=='__main__':run()
