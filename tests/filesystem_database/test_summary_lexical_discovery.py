"""Only candidate ranking from ephemeral generated-summary text; no source claims."""
from __future__ import annotations

import unittest

from scripts.raspi_source_verified_summary_discovery import lexical_ranked_summary_candidates


SOURCE_SHA = "a"*64
DERIVED_SHA = "b"*64
ROOT_SHA = "c"*64
ROWS = [
    ("ordinary/docs/onboarding.pdf", "The system uses an onboarding checklist for new contributors.", SOURCE_SHA, DERIVED_SHA, ROOT_SHA),
    ("ai-agents/recover-agent.pdf", "Agent repairs malicious instructions inserted into tool responses and restores legitimate user intent.", SOURCE_SHA, DERIVED_SHA, ROOT_SHA),
    ("ai-agents/t1-bench.pdf", "Benchmark of dining, bar, hotel and flight booking workflows performed by agents.", SOURCE_SHA, DERIVED_SHA, ROOT_SHA),
]


class EphemeralGeneratedSummaryLexicalTest(unittest.TestCase):
    def test_domain_question_ranks_matched_root_without_claiming_publisher_evidence(self):
        question="An assistant sees malicious instructions inserted into a tool response. How can it recover legitimate user task safely?"
        for porter in (False,True):
            with self.subTest(porter=porter):
                rows=lexical_ranked_summary_candidates(ROWS,question,porter=porter,limit=3)
                self.assertEqual(ROWS[1][0],rows[0]["path"])
                self.assertEqual(1,rows[0]["lexical_rank_among_current_summary_roots"])
                self.assertEqual(SOURCE_SHA,rows[0]["original_sha256_from_metadata"])
                self.assertNotIn("publisher_original_bytes_verified",rows[0])
                self.assertNotIn("source_passage",rows[0])
                self.assertNotIn("factual_confidence",rows[0])

    def test_safe_query_tokenization_does_not_execute_sql_payload(self):
        for text in ('" UNION SELECT * FROM sqlite_master --',
                     'tool; DROP TABLE summaries; --',
                     '%%%% " \\ ( ) <>'):
            with self.subTest(query=text):
                rows=lexical_ranked_summary_candidates(ROWS,text,porter=True,limit=3)
                self.assertIsInstance(rows,list)
        rows=lexical_ranked_summary_candidates(ROWS,"dining hotel flight booking",porter=True,limit=3)
        self.assertEqual(ROWS[2][0],rows[0]["path"])

    def test_input_limits_and_no_query_terms(self):
        for query in ("", " "*601, 77):
            with self.subTest(query=query),self.assertRaises(ValueError):
                lexical_ranked_summary_candidates(ROWS,query)
        for limit in (0,101,True):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                lexical_ranked_summary_candidates(ROWS,"dining",limit=limit)
        self.assertEqual([],lexical_ranked_summary_candidates(ROWS,"and the to for",limit=3))


    def test_ephemeral_summary_index_rejects_excessive_source_rows(self):
        with self.assertRaisesRegex(ValueError,'safety bound'):
            lexical_ranked_summary_candidates(ROWS*9000,'tool response',limit=3)
        with self.assertRaisesRegex(ValueError,'safety bound'):
            lexical_ranked_summary_candidates(None,'tool response',limit=3)


    def test_archive_publisher_directory_scope_validation(self):
        from scripts.raspi_source_verified_summary_discovery import validate_publisher_url_prefix
        valid='https://docs.blender.org/manual/en/4.5/'
        self.assertIsNone(validate_publisher_url_prefix(None))
        self.assertEqual(valid,validate_publisher_url_prefix(valid))
        self.assertEqual(valid,validate_publisher_url_prefix('https://DOCS.BLENDER.ORG/manual/en/4.5/'))
        for bad in ('', 'https://docs.blender.org/manual/en/4.5',
                    'http://docs.blender.org/manual/en/4.5/',
                    'https://docs.blender.org/manual/en/4.5/../5.2/',
                    'https://user:password@docs.blender.org/manual/en/4.5/',
                    'https://docs.blender.org/manual/en/4.5/?change=5.2',
                    'https://docs.blender.org/manual/en/4.5/#fragment',
                    'https://docs.blender.org/manual/en/4.5/\n',
                    'https://docs.blender.org:443/manual/en/4.5/',
                    'https://docs.blender.org/manual/en/4.5/%2e%2e/5.2/',
                    True,object()):
            with self.subTest(value=repr(bad)),self.assertRaises(ValueError):
                validate_publisher_url_prefix(bad)


if __name__=="__main__":
    unittest.main()
