"""Recover Swanson's 1989 first-person retrospective as an attributed claim.

Historical author's retrospective ≠ original 1986 full text ≠ independent
1989 clinical trial. Source remains in /data/tmp and is never rewritten.
"""
from bs4 import BeautifulSoup
from pathlib import Path
import hashlib,json,re
HERE=Path(__file__).parent
SRC=Path('/data/tmp/kmd-raynaud-originals-20261024/swanson_1989_first_person_1989_letter.html')
OUT=HERE/'raynaud_author_provenance_results_20261024.json'
SHA='b8528fe0ce99aafc4b52d5751716f5aadbba1ef0ff736d44d0a328721cfa4bed'
if OUT.exists():raise RuntimeError('Previously completed source extraction exists')
b=SRC.read_bytes();assert hashlib.sha256(b).hexdigest()==SHA
text=BeautifulSoup(b,'html.parser').get_text(' ',strip=True)
compact=re.sub(r'\s+',' ',text)
phrases={
 'author_describes_cross_literature_hypothesis':r'I showed how two noninteracting sets of articles together implied that dietary fish oil might benefit Raynaud patients, a hypothesis not previously proposed',
 'author_places_original_medical_argument_in_1986':r'The complete medical argument was published in the biomedical literature in 1986',
 'author_reports_independent_clinical_test_later':r'A controlled clinical trial of dietary fish oil in 32 Raynaud patients was recently reported by researchers at the Albany Medical College',
 'signature':r'Don R\. Swanson Graduate Library School University of Chicago',
}
spans={}
for key,p in phrases.items():
 m=re.search(p,compact,re.I)
 assert m is not None,key
 spans[key]={'source_document_sha256':SHA,'source_char_offsets_in_whitespace_normalized_archive':[m.start(),m.end()],
             'exact_short_snippet':compact[m.start():m.end()]}
result={'source_url':'https://garfield.library.upenn.edu/papers/jasis40%283%29p152y1989%281%29.html','year':1989,
 'source_sha256':SHA,'document_type':'1989 first-person letter by original hypothesis author',
 'historical_hypothesis_author':'Don R. Swanson',
 'not_the_original_1986_paper':True,'not_independent_clinical_replication':True,
 'source_spans':spans,
 'attribution_rule':'Do not treat Swanson retrospective assertion of 1986 novelty as an independently comprehensive prior-art search. Separate from clinical trial authors and clinical results.',
 'time_rule':'The retrospective 1989 letter was published after the 1986 hypothesis and 1988 conference trial report; it cannot be included in an as-of-1985 hypothesis generator.'}
OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('SWANSON_SOURCE_REFERENCES_OK',len(spans),'YEAR',result['year'],'AUTHOR',result['historical_hypothesis_author'])
