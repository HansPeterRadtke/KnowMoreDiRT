"""Independent finite truth-table oracle versus scoped inference.

The oracle enumerates every Boolean assignment independently from the forward
chainer. Neither accepts absent facts as false. Inconsistent premise sets are
explicitly flagged rather than treated as proof of everything. This is *not*
a natural-language parser or confirmation that document premises are true.
"""
import json
from pathlib import Path
from itertools import product
from collections import defaultdict

HERE=Path(__file__).parent
DATA=json.loads((HERE/'scoped_reasoning_cases_20261024.json').read_text())

def decode(text):
    if '@' in text: op,scope=text.split('@',1)
    else: op,scope=text,'real:2026'
    if ':' in scope: world,period=scope.split(':',1)
    else: world,period=scope,'2026'
    return op,world,period

def oracle(facts,ask):
    # Query always explicitly targets asserted real-world conditions in 2026.
    scope=('real','2026'); clauses=[decode(s)[0] for s in facts if decode(s)[1:]==scope]
    atoms=sorted({c.replace('>','').strip('+-') for c in clauses+['+'+ask] for c in c if c.isalpha()})
    worlds=[]
    for values in product([False,True],repeat=len(atoms)):
        d=dict(zip(atoms,values));valid=True
        for c in clauses:
            if c.startswith('+'):valid &= d[c[1:]]
            elif c.startswith('-'):valid &= not d[c[1:]]
            else:
                a,b=c.split('>'); valid &= (not d[a]) or d[b]
        if valid:worlds.append(d)
    if not worlds:return 'contradictory'
    if all(w[ask] for w in worlds):return 'entailed'
    if all(not w[ask] for w in worlds):return 'refuted'
    return 'unknown'

def chase(facts,ask):
    scoped=defaultdict(list)
    for idx,t in enumerate(facts):
        op,world,date=decode(t);scoped[(world,date)].append((op,f'source:{idx}'))
    terms=scoped[('real','2026')];known=set();negative=set();proof={};rules=[]
    for op,source in terms:
        if op.startswith('+'):
            known.add(op[1:]);proof.setdefault(op[1:],[source])
        elif op.startswith('-'):negative.add(op[1:])
        else:rules.append((op.split('>'),source))
    if known&negative:return 'contradictory',{}
    for _ in range(len(rules)+len(terms)+1):
        newly=[]
        for (a,b),src in rules:
            if a in known and b not in known:
                if b in negative:return 'contradictory',{}
                newly.append((b,proof[a]+[src]))
        if not newly:break
        for b,chain in newly:known.add(b);proof.setdefault(b,chain)
    if ask in known:return 'entailed',{'premise_sources':proof[ask]}
    if ask in negative:return 'refuted',{}
    return 'unknown',{}

def hypothesis(observations,rule):
    a,b=rule.split('>'); facts=defaultdict(set)
    for o in observations:
        claim,world,date=decode(o)
        if world=='real' and date=='2026':facts[claim[1:]].add(claim[0])
    if any(len(s)>1 for s in facts.values()):return 'inconsistent_observations'
    if '+' in facts[a] and '-' in facts[b]:return 'falsified'
    if '+' in facts[a] and '+' in facts[b]:return 'not_falsified_by_this_case'
    return 'unresolved'

def run():
    rows=[]
    for c in DATA['cases']:
        truth=oracle(c['facts'],c['ask']); observed,proof=chase(c['facts'],c['ask']);rows.append({'id':c['id'],'preset_expected':c['want'],'independent_truth_table':truth,'observed':observed,'proof':proof,'correct':observed==truth==c['want']})
    hypot=[{'id':x['id'],'preset_expected':x['want'],'observed':hypothesis(x['observations'],x['rule']),'correct':hypothesis(x['observations'],x['rule'])==x['want']} for x in DATA['hypotheses']]
    result={'kind':'finite stipulated source-scope logic, not natural language truth','cases':rows,'hypothesis_cases':hypot,'proof_tests_correct':sum(x['correct'] for x in rows),'proof_tests_total':len(rows),'falsification_tests_correct':sum(x['correct'] for x in hypot),'falsification_tests_total':len(hypot)}
    print(json.dumps(result,indent=2));assert all(x['correct'] for x in rows+hypot)
if __name__=='__main__':run()
