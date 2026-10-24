"""Automatically extract real paper measurements with source/time provenance.

Bounded empirical scientific-evidence comparison (OPERA vs independent ICARUS).
No KMD DRS model, no human labels passed into parser; rule-based patterns
perform numeric extraction, provenance grouping and conditional prediction.
HTML snapshots are externally retrieved original web pages kept in /data/tmp;
only small source-linked extracted excerpts and their SHA-256 enter this report.
"""
import hashlib,json,math,re
from pathlib import Path
from bs4 import BeautifulSoup

ROOT=Path('/data/tmp/kmd-scientific-realworld-sources-20261024')
HERE=Path(__file__).parent
SOURCES={
'cern':'cern_opera_historical_updates.html',
'opera_2012':'opera_2012_arxiv_abstract.html',
'icarus_2012':'icarus_precision_2012_arxiv_abstract.html',
}
URLS={
'cern':'https://home.cern/opera-experiment-reports-anomaly-in-flight-time-of-neutrinos-from-cern-to-gran-sasso/',
'opera_2012':'https://arxiv.org/abs/1212.1276',
'icarus_2012':'https://arxiv.org/abs/1208.2629',
}
SHA={
'cern':'b05c859f657a6daa0fbf39cdbf5d7152d41d40a0eef4b85da2489e1e546a758a',
'opera_2012':'83e4b30b63f3829985cf1b0175eb9eb9be3a46a15f6527a6801e2d3f789fac61',
'icarus_2012':'3ce12182de2d3a96db96dedf0bdd5f29d29738ec62141e3cb96184a4c34dfb17',
}

def normalize_math(t):
 t=t.replace('\\pm',' ± ').replace('\\sim',' ~ ').replace('\\',' ')
 for key in ('stat.','syst.'):
  t=t.replace('('+key+')','').replace(key,'')
 return re.sub(r'\s+',' ',t)

def parse_numeric(t):
 norm=normalize_math(t)
 m=re.search(r'(?<!\w)\(\s*([+-]?\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)',norm)
 if not m: return None
 vals=[float(x) for x in m.groups()]
 orig_t=t
 anchor=t.find(m.group(1))
 return {'value_ns':vals[0],'statistical_ns':vals[1],'systematic_ns':vals[2],
         'combined_ns_assuming_independent_errors':round(math.hypot(vals[1],vals[2]),6),
         'evidence_extract':orig_t[max(0,anchor-95):min(len(t),anchor+115)],'offset_in_source':anchor}

def source(name):
 p=ROOT/SOURCES[name];b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==SHA[name]
 return BeautifulSoup(b.decode(errors='replace'),'html.parser')

def dated_cern_claims():
 s=source('cern')
 nodes=s.find_all(['h4','h3','p'])
 date='2011-09-23';section='initial_2011_press'
 out=[]
 for e in nodes:
  t=e.get_text(' ',strip=True)
  if e.name=='h4':
   if t.lower().startswith('update'):
    m=re.search(r'(\d+)\s+(June|March|February|November)\s+(20\d\d)',t)
    if m:
     month={'February':'02','March':'03','June':'06','November':'11'}[m.group(2)]
     date=f'{m.group(3)}-{month}-{int(m.group(1)):02d}';section='update_'+date
   if t.lower().startswith('original press release'):date='2011-09-23';section='original_press_2011'
  if e.name!='p':continue
  low=t.lower()
  patterns=[('initial_early_arrival',r'20 parts per million above the speed of light'),
            ('four_luminal_measurements',r'all measure a neutrino time of flight consistent with the speed of light'),
            ('icarus_independent_measurement',r'icarus experiment has independent timing from opera'),
            ('possible_fault',r'possible effects that could have an influence'),
            ('hypothetical_future',r'if this measurement is confirmed')]
  for label,pattern in patterns:
   match=re.search(pattern,low)
   if match:
    quote_or_hypo= bool(re.search(r'\bif\s+(?:this|confirmed|the)\b',t.lower()))
    out.append({'claim_type':label,'date':date,'section':section,'source':'cern','source_url':URLS['cern'],
      'publisher':'CERN','source_paragraph_sha256':hashlib.sha256(t.encode()).hexdigest(),
      'trigger_text':t[match.start():match.end()],'trigger_char_offsets':[match.start(),match.end()],
      'source_context_snippet':t[max(0,match.start()-125):min(len(t),match.end()+165)],
      'is_quote_or_conditional':quote_or_hypo,'cannot_promote_to_universal_truth':True})
 return out

def primary_measurement(name):
 s=source(name);node=s.find('blockquote',class_='abstract') or s.find('blockquote')
 assert node is not None
 full=node.get_text(' ',strip=True)
 extracted=parse_numeric(full);assert extracted,name
 title=s.find('h1');name_label=title.get_text(' ',strip=True) if title else ''
 return {'source':name,'url':URLS[name],'document_title':name_label,'paper_reference':'arxiv:'+URLS[name].split('/')[-1],
         'source_sha256':SHA[name], 'abstract_sha256':hashlib.sha256(full.encode()).hexdigest(),
         'year_of_measurement':2012,'measured_experiment':('OPERA' if 'OPERA' in name_label else 'ICARUS'),
         **extracted}

def compare(opera,icarus):
 # Explicit hypothesis inferred from reported 2011 headline and 730 km beam length.
 length_km=730.0;c_m_s=299792458.0;delta_speed=20/1e6
 baseline_ns=length_km*1e3/c_m_s*1e9
 predicted_ns=baseline_ns-(baseline_ns/(1+delta_speed))
 res={'hypothesis':'Twenty parts per million speed excess over c, on a 730 km path, if it had been a real propagation speed effect',
  'derived_expected_early_arrival_ns':round(predicted_ns,4), 'arithmetic_assumptions':['2011 CERN headline approximates measurement as 20 ppm','Same fixed baseline of 730 km','Speed is constant on the baseline','Emission and detection timings calibrated correctly'],
  'measurements':[opera,icarus],
  'difference_icarus_minus_opera_ns':round(icarus['value_ns']-opera['value_ns'],4),
  'combined_2012_independent_error_ns':round(math.hypot(icarus['combined_ns_assuming_independent_errors'],opera['combined_ns_assuming_independent_errors']),5),
  'not_scientific_truth_inferred':'A contradictory measurement is a challenge to the early claim, not a proof about universal velocities or all neutrinos; shared systematic correlations and model assumptions need scrutiny',
  'counterfactuals':[
   {'intervention':'Swap 2011 OPERA source date with 2012 ICARUS date','invalidates':'historical attribution and knowledge availability; cannot report independent ICARUS finding as already known in 2011'},
   {'intervention':'Wrap early OPERA result in a dream or fictional quotation','invalidates':'real-world observation evidence; quoting/dreaming does not validate measurement'},
   {'intervention':'Replace 730 km baseline with a different distance','invalidates':'computed nanosecond prediction without recalculating distance'},
   {'intervention':'Replace neutrino measurement with anti-neutrino value','invalidates':'experiment/predicate identity match'},
   {'intervention':'Interpret the 2012 conditional faulty-connector hypothesis as a proven explanation in February','invalidates':'hypothesis/confirmed scope; subsequent tests required'},
  ]}
 return res

def run():
 claims=dated_cern_claims()
 needed=['initial_early_arrival','four_luminal_measurements','icarus_independent_measurement','possible_fault']
 counts={k:len([c for c in claims if c['claim_type']==k]) for k in needed}
 assert all(counts.values()),counts
 initial=[c for c in claims if c['claim_type']=='initial_early_arrival'][0]
 latest=[c for c in claims if c['claim_type']=='four_luminal_measurements'][0]
 assert initial['date']=='2011-09-23',initial
 assert latest['date']=='2012-06-08',latest
 op=primary_measurement('opera_2012');ic=primary_measurement('icarus_2012')
 assert all(x['combined_ns_assuming_independent_errors']>0 for x in [op,ic])
 result={'dataset_type':'real separately published scientific observations plus dated institutional announcements','sources':{k:{'url':URLS[k],'sha256':SHA[k]} for k in SOURCES},'cern_claims':claims,
 'measurements':{'opera_2012':op,'icarus_2012':ic},'derived_hypothesis_check':compare(op,ic),
 'source_separation_checks':{'preliminary_2011_is_not_june2012_confirmation':initial['date']!=latest['date'],'different_2012_experiment_authors':op['measured_experiment']!=ic['measured_experiment']},
 'scope_warnings':['Source articles contain statements/reporting, not universal axioms','CERN historical page includes both 2011 and 2012 sections; treating page publication as one time is false','Quoted conditional statements are not independently measured facts','The uncertainty quadrature assumes independent errors; shared clock systematics may correlate','Counterfactual interventions are prespecified conceptual controls, not external expert-labeled gold','This experiment extracts headline numerics with a narrow parser, not arbitrary scientific prose or full DRT']}
 out=HERE/'real_neutrino_source_analysis_results_20261024.json'
 if out.exists():raise RuntimeError('refusing to overwrite earlier experiment')
 out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print('CERN_DATED',[(x['claim_type'],x['date'],x['section']) for x in claims]);print('OPERA_EXTRACTED',op['value_ns'],op['statistical_ns'],op['systematic_ns'])
 print('ICARUS_EXTRACTED',ic['value_ns'],ic['statistical_ns'],ic['systematic_ns']);print('EARLY_20PPM_PREDICTED_NS',result['derived_hypothesis_check']['derived_expected_early_arrival_ns'])
if __name__=='__main__':run()
