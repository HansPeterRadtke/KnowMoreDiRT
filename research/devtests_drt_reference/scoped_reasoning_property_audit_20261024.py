"""Property audit of prior bounded chaser, with independent SymPy SAT oracle.

No gold fixture changes. Seed and grammar fixed before first run; results are
NOT KMD production behavior and do NOT prove natural language extraction.
"""
import hashlib,json,random,sys
from collections import Counter
from pathlib import Path
from sympy import symbols,And,Not,Implies
from sympy.logic.inference import satisfiable
from scoped_reasoning_probe_20261024 import chase,decode
HERE=Path(__file__).parent
SEED=20261024
COUNT=512
ATOMS='abcd'
SCOPES=['','@real:2025','@dream','@quote','@real:2026']
CLAIMS=(['+'+a for a in ATOMS]+['-'+a for a in ATOMS]+[a+'>'+b for a in ATOMS for b in ATOMS if a!=b])

def independent_sat_oracle(facts,ask):
    assert ask in ATOMS
    bs={name:var for name,var in zip(ATOMS,symbols('a b c d'))}
    formulas=[]
    for raw in facts:
        clause,world,date=decode(raw)
        if (world,date)!=('real','2026'):continue
        if clause.startswith('+'):term=bs[clause[1:]]
        elif clause.startswith('-'):term=Not(bs[clause[1:]])
        else:
            a,b=clause.split('>');term=Implies(bs[a],bs[b])
        formulas.append(term)
    conjunction=And(*formulas)
    # SAT works with abstract source assumptions; inconsistency takes priority.
    if satisfiable(conjunction) is False:return 'contradictory'
    if satisfiable(And(conjunction,Not(bs[ask]))) is False:return 'entailed'
    if satisfiable(And(conjunction,bs[ask])) is False:return 'refuted'
    return 'unknown'

def make_cases():
    rng=random.Random(SEED)
    return [{'id':f'scoped-seed-{i:03d}','facts':[rng.choice(CLAIMS)+rng.choice(SCOPES) for _ in range(rng.randint(0,7))],'ask':rng.choice(ATOMS)} for i in range(COUNT)]

def main():
    cases=make_cases()
    cases_bytes=(json.dumps({'seed':SEED,'grammar':{'atoms':ATOMS,'scopes':SCOPES,'claim_count':len(CLAIMS),'n':COUNT},'cases':cases},sort_keys=True,indent=2)+'\n').encode()
    path=HERE/'scoped_reasoning_property_cases_20261024.json'
    if path.exists():assert path.read_bytes()==cases_bytes,'CASES_FROZEN_NO_REWRITE'
    else:path.write_bytes(cases_bytes)
    rows=[]
    for case in cases:
        expected=independent_sat_oracle(case['facts'],case['ask'])
        actual,trace=chase(case['facts'],case['ask'])
        rows.append({'id':case['id'],'expected_sympy_sat':expected,'observed_existing_chaser':actual,'correct':actual==expected,'proof':trace,'source_facts':case['facts'],'asked_atom':case['ask']})
    mismatches=[x for x in rows if not x['correct']]
    result={'case_sha256':hashlib.sha256(cases_bytes).hexdigest(),'total':COUNT,'correct':COUNT-len(mismatches),'incorrect':len(mismatches),'confusion':[{'expected':a,'actual':b,'count':count} for (a,b),count in sorted(Counter((x['expected_sympy_sat'],x['observed_existing_chaser']) for x in rows).items())],'label_totals':dict(Counter(x['expected_sympy_sat'] for x in rows)),'failures':mismatches,'limit':['Four Boolean atoms only, explicit premises/implication clauses, not full DRT/first-order logic','SymPy independent SAT oracle for logic labels; source assumptions themselves not verified','No natural language interpretation, real-world experiments or independent scientific discovery','The earlier 13/13 small fixtures and 7/7 observational labels are not retroactively overwritten']}
    (HERE/'scoped_reasoning_property_results_20261024.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SCOPED_SAT_AUDIT',result['correct'],'/',COUNT,'incorrect',result['incorrect'],'labels',result['label_totals'],'confusion',result['confusion'])
    print('COUNTEREXAMPLE',mismatches[0] if mismatches else 'none')
if __name__=='__main__':main()
