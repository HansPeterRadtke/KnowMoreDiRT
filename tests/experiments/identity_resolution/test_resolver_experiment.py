from resolver_experiment import Candidate, Mention, resolve

def c(rid,*,score=.8,aliases=(),attrs=None,typ='person',recency=0.0,start=None,end=None):
    return Candidate(rid,typ,frozenset(aliases),attrs or {},start,end,score,recency)

def test_exact_unique_identifier_resolves_despite_weaker_vector():
    m=Mention('the customer','person',{'email':'john@example.test','role':'customer'})
    result=resolve(m,[c('wrong',score=.96,attrs={'email':'other@example.test','role':'customer'}),c('john',score=.60,attrs={'email':'john@example.test','role':'customer'})])
    assert result.state=='resolved' and result.referent_id=='john'

def test_same_role_without_distinguishing_evidence_stays_ambiguous():
    m=Mention('the customer','person',{'role':'customer'})
    result=resolve(m,[c('a',score=.83,attrs={'role':'customer'}),c('b',score=.82,attrs={'role':'customer'})])
    assert result.state=='ambiguous'

def test_pronoun_can_resolve_from_relations_and_recency():
    m=Mention('they','person',{'organization':'Aster Labs','product':'Orchid'})
    result=resolve(m,[c('john',score=.76,recency=.9,attrs={'organization':'Aster Labs','product':'Orchid'}),c('maria',score=.79,recency=.1,attrs={'organization':'Blue Harbor','product':'Nimbus'})])
    assert result.state=='resolved' and result.referent_id=='john'

def test_entity_type_conflict_forbids_merge():
    m=Mention('Aster Labs','organization')
    result=resolve(m,[c('person',score=.99,aliases={'Aster Labs'},typ='person')])
    assert result.state=='new'

def test_temporal_incompatibility_forbids_merge():
    m=Mention('the manager','person',{'role':'manager'},year=2025)
    result=resolve(m,[c('old-manager',score=.95,attrs={'role':'manager'},start=2015,end=2020)])
    assert result.state=='new'

def test_exact_alias_resolves():
    m=Mention('J. Adler','person')
    result=resolve(m,[c('john',score=.55,aliases={'John Adler','J. Adler'}),c('jane',score=.80,aliases={'Jane Adler'})])
    assert result.state=='resolved' and result.referent_id=='john'

def test_new_entity_is_not_forced_to_existing_referent():
    m=Mention('the new auditor','person',{'organization':'Novel Org'},explicit_new=True)
    result=resolve(m,[c('old',score=.99,attrs={'organization':'Other Org'})])
    assert result.state=='new'

def test_close_candidates_preserve_ambiguity_even_above_threshold():
    m=Mention('Alex','person')
    result=resolve(m,[c('alex-a',score=.99,aliases={'Alex'}),c('alex-b',score=.98,aliases={'Alex'})])
    assert result.state=='ambiguous'
