"""Isolated research-only forward/backward implication solver with proof links.

Adds justified contrapositive propagation to the earlier test chaser;
production KMD is unchanged. Exact source labels are retained at each step.
"""
import hashlib,json,random
from collections import Counter
from pathlib import Path
from scoped_reasoning_probe_20261024 import decode,chase
from scoped_reasoning_property_audit_20261024 import independent_sat_oracle,CLAIMS,SCOPES,ATOMS
HERE=Path(__file__).parent
SEED=88123
HOLDOUT_COUNT=1024

def bidirectional(facts,ask):
    positive={};negative={};implications=[]
    for i,statement in enumerate(facts):
        operator,world,period=decode(statement)
        if (world,period)!=('real','2026'):continue
        source=f'source:{i}'
        if operator.startswith('+'):positive.setdefault(operator[1:],(source,))
        elif operator.startswith('-'):negative.setdefault(operator[1:],(source,))
        else:
            antecedent,consequent=operator.split('>')
            implications.append((antecedent,consequent,source))
    for _ in range(1+2*len(implications)+len(facts)):
        clashes=positive.keys()&negative.keys()
        if clashes:
            atom=sorted(clashes)[0]
            return 'contradictory',{'atom':atom,'positive_sources':positive[atom],'negative_sources':negative[atom]}
        changed=False
        for a,b,src in implications:
            if a in positive and b not in positive:
                positive[b]=positive[a]+(src,);changed=True
            if b in negative and a not in negative:
                negative[a]=negative[b]+(src,);changed=True
        if not changed:break
    else:raise RuntimeError('bounded_horn_propagation_nonterminating')
    clashes=positive.keys()&negative.keys()
    if clashes:
        atom=sorted(clashes)[0];return 'contradictory',{'atom':atom,'positive_sources':positive[atom],'negative_sources':negative[atom]}
    if ask in positive:return 'entailed',{'positive_sources':positive[ask]}
    if ask in negative:return 'refuted',{'negative_sources':negative[ask]}
    return 'unknown',{}

def audit(cases,name,seed=None):
    results=[]
    for row in cases:
        expected=independent_sat_oracle(row['facts'],row['ask'])
        predicted,trace=bidirectional(row['facts'],row['ask'])
        baseline,_=chase(row['facts'],row['ask'])
        for item in (trace.get('positive_sources',()),trace.get('negative_sources',())):
            assert all(s.startswith('source:') and int(s[7:])<len(row['facts']) for s in item)
        results.append({'id':row['id'],'source_facts':row['facts'],'asked_atom':row['ask'],'expected_independent_sympy':expected,'candidate':predicted,'old_forward_only':baseline,'correct':predicted==expected,'premise_trace':trace})
    output={'name':name,'seed':seed,'total':len(cases),'correct':sum(x['correct'] for x in results),'errors':[x for x in results if not x['correct']],'old_forward_only_correct':sum(x['old_forward_only']==x['expected_independent_sympy'] for x in results),'all_results':results,'limitations':['Quantifier-free Horn implication fragment over four atoms only','Premises are stipulated and real-world truth is not validated','Source links are synthetic proposition indices, not original text offsets','Results do not establish production KMD inference or science discovery']}
    (HERE/f'scoped_bidirectional_{name}_20261024.json').write_text(json.dumps(output,indent=2)+'\n')
    print(name,'solver',output['correct'],'/',output['total'],'previous',output['old_forward_only_correct'],'remaining_errors',len(output['errors']))

def main():
    original=json.loads((HERE/'scoped_reasoning_property_cases_20261024.json').read_text())['cases']
    audit(original,'previously_consumed')
    rng=random.Random(SEED)
    heldout=[{'id':f'bidir-heldout-{i:04d}','facts':[rng.choice(CLAIMS)+rng.choice(SCOPES) for _ in range(rng.randint(0,9))],'ask':rng.choice(ATOMS)} for i in range(HOLDOUT_COUNT)]
    frozen={'seed':SEED,'rule_pool':CLAIMS,'scope_pool':SCOPES,'cases':heldout}
    encoded=json.dumps(frozen,sort_keys=True,indent=2)+'\n'
    path=HERE/'scoped_bidirectional_unseen_cases_20261024.json'
    if path.exists():assert path.read_text()==encoded
    else:path.write_text(encoded)
    audit(heldout,'unseen_seed',seed=SEED)
if __name__=='__main__':main()
