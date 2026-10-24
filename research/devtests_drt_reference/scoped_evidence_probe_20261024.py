"""Deterministic scoped-evidence ablation, NOT an evaluation of KMD's live DRS model.

Each question asks whether proposition P is asserted in reality, under a chosen
speaker/authority and validity time. We compare an exact-match chunk-only baseline,
a window-expansion provenance baseline, and explicit scope+identity filtering.
Synthetic fixture is fixed independently of the implementation; no model calls.
"""
from dataclasses import dataclass
import json

@dataclass(frozen=True)
class Case:
    name: str
    chunks: tuple[str, ...]
    fact: str
    expected: bool
    scope: str
    source: str
    valid_at: int

CASES = (
 Case('plain',('Agency states:', 'Bridge A is open.'),'Bridge A is open.',True,'real','agency',2026),
 Case('dream_cross_chunk',('Mia dreamed last night:', 'Bridge A is open.'),'Bridge A is open.',False,'dream','Mia',2026),
 Case('fiction_cross_chunk',('A story says:', 'Bridge A is open.'),'Bridge A is open.',False,'fiction','narrator',2026),
 Case('quotation_cross_chunk',('Mia quotes a rumor:', 'Bridge A is open.'),'Bridge A is open.',False,'quote','Mia',2026),
 Case('negated',('Agency correction: NOT', 'Bridge A is open.'),'Bridge A is open.',False,'negated','agency',2026),
 Case('conditional',('Agency says only IF approved:', 'Bridge A is open.'),'Bridge A is open.',False,'conditional','agency',2026),
 Case('expired',('Agency notice dated 2024, expires 2025:', 'Bridge A is open.'),'Bridge A is open.',False,'real','agency',2024),
 Case('wrong_authority',('Unverified visitor says:', 'Bridge A is open.'),'Bridge A is open.',False,'real','visitor',2026),
 Case('conflict',('Agency says: Bridge A is open.','Agency correction: Bridge A is not open.'),'Bridge A is open.',False,'conflict','agency',2026),
 Case('different_place',('Agency says: Bridge B is open.',),'Bridge A is open.',False,'real','agency',2026),
 Case('agency_recent',('Agency notice 2026:', 'Bridge A is open.'),'Bridge A is open.',True,'real','agency',2026),
 Case('scoped_quoted_fact',('Agency confirms current road status:', 'Bridge A is open.'),'Bridge A is open.',True,'real','agency',2026),
)

def chunk_only(c):
    return any(c.fact in chunk for chunk in c.chunks)

def neighbor_provenance(c):
    combined = ' '.join(c.chunks).lower()
    if c.fact.lower() not in combined: return False
    if any(x in combined for x in ('dreamed','story says','quotes a rumor','not bridge','not open','only if','expires')): return False
    return True

def scoped(c):
    # Metadata is an independently supplied gold scope; perfect parsing is NOT assumed.
    return c.fact in ' '.join(c.chunks) and c.scope == 'real' and c.source == 'agency' and c.valid_at == 2026

def run():
    methods={'chunk_only':chunk_only,'neighbor_provenance':neighbor_provenance,'scoped_oracle_metadata':scoped}
    results={k:{'correct':sum(fn(c)==c.expected for c in CASES),'total':len(CASES),'errors':[c.name for c in CASES if fn(c)!=c.expected]} for k,fn in methods.items()}
    # Independent logical derivation requires explicit premises and retains them.
    premises={'A':'If approved, the bridge may open','B':'Approval issued'}
    derivation={'conclusion':'Bridge may open','rule':'modus_ponens','premises':['A','B'],'valid_given_premises':True,'empirically_verified':False}
    contradiction={'claims':['Bridge open','Bridge not open'],'same_entity':'Bridge A','same_time':2026,'same_scope':'agency current','syntactic_incompatibility':True,'truth_resolved':False}
    result={'cases':[c.__dict__ for c in CASES],'methods':results,'demonstrations':{'derivation':derivation,'contradiction':contradiction},'limitations':['Synthetic, not heldout real documents','Scoped mode is an oracle metadata upper bound, not DRS parsing','Rules cannot establish extracted premises are true','Conflict resolution needs source ordering and authority']}
    print(json.dumps(result,indent=2))
if __name__=='__main__':run()
