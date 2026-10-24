"""Versioned identity-corrected followup, preserves V1 negative result unchanged.

V1 mistakenly used the category H1 heading to label both arxiv sources
ICARUS. V2 resolves publisher-supplied `citation_title`, `citation_author`,
`citation_date`, and asserts experiment and source match. Measurements are
reextracted from unchanged independently hosted article abstract HTML.
"""
from datetime import datetime
from pathlib import Path
import json,hashlib,math
from real_neutrino_source_reasoning import (source,dated_cern_claims,parse_numeric,compare,URLS,SHA,HERE)

def primary_measurement_verified(name):
 s=source(name)
 md={key:(s.find('meta',attrs={'name':key}) or {}).get('content','').strip() for key in ['citation_title','citation_author','citation_date']}
 assert all(md.values()), ('missing arxiv source identity metadata',name)
 expected='OPERA' if name=='opera_2012' else 'ICARUS' if name=='icarus_2012' else None
 assert expected in md['citation_title'], ('wrong experiment identity',name,md['citation_title'])
 assert md['citation_date'][:4]=='2012',('wrong paper publication year',name)
 b=s.find('blockquote',class_='abstract') or s.find('blockquote')
 assert b is not None
 abstract=b.get_text(' ',strip=True)
 measurement=parse_numeric(abstract);assert measurement,('missing numeric measurement',name)
 return {'source':name,'url':URLS[name], 'document_title':md['citation_title'],
     'paper_author':md['citation_author'],'paper_date':md['citation_date'],
     'paper_reference':'arxiv:'+URLS[name].split('/')[-1],
     'source_sha256':SHA[name],'abstract_sha256':hashlib.sha256(abstract.encode()).hexdigest(),
     'measured_experiment':expected,'year_of_measurement':2012,**measurement}

def corrected_result():
 old=HERE/'real_neutrino_source_analysis_results_20261024.json'
 assert old.exists(),'preserve V1 negative before correcting'
 old_content=json.loads(old.read_text())
 assert old_content['measurements']['opera_2012']['measured_experiment']=='ICARUS','V1 failure changed; stop'
 claims=dated_cern_claims()
 op=primary_measurement_verified('opera_2012');ic=primary_measurement_verified('icarus_2012')
 assert op['paper_author']!=ic['paper_author']
 result={'correction_of':'real_neutrino_source_analysis_results_20261024.json',
   'original_wrong_identity':'The earlier extractor treated the generic arXiv category h1 as paper title, marking OPERA as ICARUS. V1 JSON is immutable negative evidence, not overwritten.',
   'source_identity_procedure':'Use publisher citation_title, citation_author, citation_date metadata, not first h1; source name/URL independently checked against experiment name and publication year.',
   'source_files_sha256':SHA,'original_sources_url':URLS,
   'old_result_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
   'measurements':{'opera_2012':op,'icarus_2012':ic},
   'historical_cern_claims':claims,
   'derived_hypothesis_check':compare(op,ic),
   'verification':{'experiment_distinct':op['measured_experiment']!=ic['measured_experiment'],
                   'publication_year_match':op['paper_date'].startswith('2012') and ic['paper_date'].startswith('2012'),
                   'source_metadata_not_generic_heading':all('Measurement of the neutrino velocity' in x['document_title'] or 'measurement of the neutrino velocity' in x['document_title'] for x in [op,ic]),
                   'old_source_numeric_values_retained':op['value_ns']==old_content['measurements']['opera_2012']['value_ns'] and ic['value_ns']==old_content['measurements']['icarus_2012']['value_ns']}}
 assert all(result['verification'].values())
 return result

if __name__=='__main__':
 dest=HERE/'real_neutrino_source_analysis_corrected_20261024.json'
 if dest.exists():raise SystemExit('Do not overwrite corrected output')
 payload=corrected_result()
 dest.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
 print('PRESERVED_PREVIOUS_BAD_RESULT',payload['old_result_sha256'])
 for key,entry in payload['measurements'].items(): print('SOURCE',key,'AUTHOR',entry['paper_author'],'EXPERIMENT',entry['measured_experiment'],'PUBLICATION',entry['paper_date'],'DELTA_NS',entry['value_ns'])
 print('HYPOTHESIS_EXPECTED_EARLY_NS',payload['derived_hypothesis_check']['derived_expected_early_arrival_ns'])
