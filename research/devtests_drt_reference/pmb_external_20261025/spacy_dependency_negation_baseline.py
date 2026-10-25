"""Off-the-shelf source-only dependency syntax vs published model-extracted DRS.

One fixed, untuned dependency algorithm over original PMB 3.0 test+dev source
utterances. Human-corrected DRS gold is used ONLY after source-only predictions
have been frozen; model-generated DRS also compared to gold in later scoring.
Target metric: lemma multiset of VERB/ADJECTIVE lexical concepts inside negated
scope (not mere negation operator presence). Does not cover nominal predicates,
world/time nesting, event roles, cross-document coreference or science facts.

Unlike the earlier lexical token cue baseline, this attempts to resolve the
negation dependency's HEAD to a content predicate in natural sentences.
"""
from pathlib import Path
from collections import Counter
import hashlib,json
import spacy
from operator_scope_predicate_audit import GROOT,SHA
HERE=Path(__file__).parent
DEST=HERE/'frozen_spacy_dependency_predictions_20261025.json'
SOURCE_MODEL='en_core_web_sm'
VAPOS={'VERB','ADJ'}
NEG_LEX={'no','not','never','none','nobody','nothing','nowhere','neither','without','hardly','rarely','barely'}

def preferred(target):
    # Syntactic target only: choose lexical content under copula be if adjective
    # complement is explicitly attached, otherwise nearest lexical main verb.
    if target.pos_=='AUX' and target.head is not target:
        target=target.head
    if target.pos_=='AUX' and target.lemma_.lower()=='be':
        complement=next((z for z in target.children if z.dep_ in ('acomp','attr','oprd') and z.pos_ in VAPOS),None)
        if complement is not None:target=complement
    if target.pos_ in VAPOS:return target.lemma_.lower().strip()
    # nearest governing predicate within 5 ancestors. Avoid guessing a named
    # noun's role from text: it is not a proposition by itself.
    for ancestor in list(target.ancestors)[:5]:
        if ancestor.pos_ in VAPOS:return ancestor.lemma_.lower().strip()
        if ancestor.pos_=='AUX' and ancestor.lemma_.lower()=='be':
            comp=next((z for z in ancestor.children if z.dep_ in ('acomp','attr') and z.pos_ in VAPOS),None)
            if comp is not None:return comp.lemma_.lower().strip()
    return None

def source_negation_candidates(doc):
    lemmas=Counter();rules=[]
    for t in doc:
        tok=t.text.casefold();lemma=t.lemma_.casefold()
        if t.dep_=='neg' or (lemma in NEG_LEX and (t.dep_ in ('det','advmod','prep','nsubj','dobj','pobj','cc','ROOT','mark') or tok=='neither')):
            if tok in ('hardly','rarely','barely'):
                # weak negative polarity in natural language but not always
                # mapped to formal DRS negation; account as candidate, never gold.
                pass
            if lemma in ('no','none','nobody','nothing','neither') or tok in ('neither','nor'):
                # source quantifiers commonly scope entire predicate of clause.
                target=t.head
                for a in [target]+list(target.ancestors):
                    if a.pos_ in VAPOS:
                        target=a;break
            else:target=t.head
            ans=preferred(target)
            if ans:lemmas[ans]+=1;rules.append({'cue':t.text,'cue_index':t.i,'extracted_lemma':ans,'rule':'negation_dependency_head_or_ancestor'})
        # No labeled source examples were used to select these source-language
        # patterns: they are general English negative constructions.
        if tok=='but' and t.i>0 and doc[t.i-1].lemma_.lower() in ('anything','everything','nothing'):
            comp=next((z for z in list(t.subtree) if z.pos_ in VAPOS),None)
            if comp is not None:
                ans=comp.lemma_.lower().strip();lemmas[ans]+=1;rules.append({'cue':'anything-but','cue_index':t.i,'extracted_lemma':ans,'rule':'postpositive_exception'})
        if lemma=='too' and t.head.pos_=='ADJ':
            comp=next((x for x in t.head.children if x.pos_=='VERB' and x.dep_ in ('xcomp','advcl')),None)
            if comp is not None:
                ans=comp.lemma_.lower().strip();lemmas[ans]+=1;rules.append({'cue':'too+adjective+infinitive','cue_index':t.i,'extracted_lemma':ans,'rule':'prevented_modal_infinitive'})
    return lemmas,rules

def main():
    if DEST.exists():raise RuntimeError('Original baseline frozen; do not repeat or overwrite')
    nlp=spacy.load(SOURCE_MODEL,disable=['ner'])
    model_version=nlp.meta.get('version')
    output={'schema':'source-only dependency negation target lexical VERB/ADJ head, no human DRS consulted',
      'model':SOURCE_MODEL,'spacy_version':spacy.__version__,'spacy_model_version':model_version,
      'source_git_version':'RikVN/DRS_parsing@190838f9adea2af616eb76156d2e66c8565a6c6e',
      'fixed_rules':'neg dependency head; general negative polarity quantifiers and modifiers; anything-but postpositive adjective; too adjective plus verb xcomp',
      'source_sha256':{},'splits':{},
      'limitations':['Source-only dependency parse is not full provenance+neighbor retrieval; PMB texts mostly single sentences','No training, hyperparameter tuning, or annotated DRS provided to spaCy or source-only rule logic','Target metric restricted to negated verb/adjective lemma concepts: noun negation and modal quantifier scopes are explicitly outside it','Source negation cues can be mapped wrongly; output retains each cue, token index and source sentence','Gold already seen in previous research, so a historical pretraining/prompt exposure guarantee is NOT made']}
    for split in ('test','dev'):
        source=GROOT/(split+'.txt.raw');digest=hashlib.sha256(source.read_bytes()).hexdigest();assert digest==SHA[split]['raw']
        sents=source.read_text().splitlines()
        docs=list(nlp.pipe(sents,batch_size=64,n_process=1))
        rows=[]
        for i,doc in enumerate(docs):
            lemma,trace=source_negation_candidates(doc)
            rows.append({'sentence_index':i,'source_sha256':hashlib.sha256(sents[i].encode()).hexdigest(),
                'predicted_negated_va_lemmas':list(lemma.elements()),'source_cues_and_dependency_trace':trace})
        output['source_sha256'][split]=digest;output['splits'][split]={'n':len(rows),'predictions':rows}
        print('FROZEN',split,'n',len(rows),'with_source_negative_candidate',sum(bool(x['predicted_negated_va_lemmas']) for x in rows),flush=True)
    DEST.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print('FROZEN_PRED_SHA256',hashlib.sha256(DEST.read_bytes()).hexdigest())
if __name__=='__main__':main()
