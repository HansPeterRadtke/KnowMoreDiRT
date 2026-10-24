"""Narrow research regression tests, not the production KMD test suite."""
import random,unittest
from scoped_bidirectional_reasoner_20261024 import bidirectional
from scoped_reasoning_property_audit_20261024 import independent_sat_oracle,CLAIMS,SCOPES,ATOMS

class ScopedReasoningResearchTests(unittest.TestCase):
    def test_modus_tollens_exposes_old_missing_negative_consequence(self):
        facts=['-d@real:2026','b>d','b>c@real:2026','c>d@real:2026']
        observed,trace=bidirectional(facts,'c')
        self.assertEqual(observed,'refuted')
        self.assertEqual(observed,independent_sat_oracle(facts,'c'))
        self.assertIn('source:0',trace['negative_sources'])
        self.assertIn('source:3',trace['negative_sources'])

    def test_dream_statement_not_promoted_to_reality(self):
        facts=['+a@dream','a>b@real:2026']
        self.assertEqual(bidirectional(facts,'b')[0],'unknown')

    def test_mixed_time_cannot_complete_a_chain(self):
        facts=['+a@real:2026','a>b@real:2025']
        self.assertEqual(bidirectional(facts,'b')[0],'unknown')

    def test_quoted_negation_does_not_disprove_reality(self):
        facts=['+a@real:2026','-a@quote']
        self.assertEqual(bidirectional(facts,'a')[0],'entailed')

    def test_same_scope_contradiction_cites_both_sources(self):
        status,trace=bidirectional(['+b','-b@real:2026'],'b')
        self.assertEqual(status,'contradictory')
        self.assertEqual(set(trace['positive_sources']+trace['negative_sources']),{'source:0','source:1'})

    def test_unproved_is_not_false(self):
        self.assertEqual(bidirectional(['a>b'],'b')[0],'unknown')

    def test_distinct_atoms_are_not_guessed_identical(self):
        self.assertEqual(bidirectional(['+a'],'b')[0],'unknown')

    def test_stipulated_labeled_random_cases_vs_independent_sympy(self):
        rng=random.Random(1316)
        for i in range(64):
            facts=[rng.choice(CLAIMS)+rng.choice(SCOPES) for _ in range(rng.randrange(0,10))]
            asked=rng.choice(ATOMS)
            with self.subTest(i=i,facts=facts,asked=asked):
                self.assertEqual(bidirectional(facts,asked)[0],independent_sat_oracle(facts,asked))

if __name__=='__main__':unittest.main()
