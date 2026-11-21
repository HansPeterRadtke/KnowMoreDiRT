"""CLI contract: exactly scoped queries, completed receipts, no guessed fallback."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def module():
    path=Path(__file__).resolve().parents[2]/'scripts/raspi_publisher_firstunit_discovery.py'
    spec=importlib.util.spec_from_file_location('bounded_firstunit_cli_under_test',path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class PublisherFirstUnitCLIContractTests(unittest.TestCase):
    def setUp(self):
        self.c=module()
        folder=TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.root=Path(folder.name)
        self.receipts=self.root/'completed-sources.json'
        self.queries=self.root/'batch.json'
        self.receipts.write_text(json.dumps({'items':[
            {'path':'papers/one.pdf','status':'complete','sha256':'a'*64},
            {'path':'papers/unindexed.pdf','status':'error','sha256':'b'*64}]},sort_keys=True))

    def test_single_query_and_batch_are_distinct_authorized_surfaces(self):
        self.assertEqual([{'id':'single','query':'find 3d materials'}],
                         self.c.input_queries(query='find 3d materials',query_file=None))
        cases=[{'id':'first','query':'answer from the same original PDF'},
               {'id':'second','query':'tool-use reward from a research document'}]
        self.queries.write_text(json.dumps({'queries':cases}))
        self.assertEqual(cases,self.c.input_queries(query=None,query_file=self.queries))
        for arguments in ((None,None),('test',self.queries)):
            with self.subTest(args=arguments),self.assertRaisesRegex(ValueError,'exactly one'):
                self.c.input_queries(query=arguments[0],query_file=arguments[1])

    def test_batches_must_have_unambiguous_bounded_and_explicit_original_queries(self):
        bads=[{'queries':[]}, {'queries':[{'id':'a','query':'foo'}]*2},
              {'queries':[{'id':'a/b','query':'hello'}]},
              {'queries':[{'id':'a','query':'yes','source':'secret'}]},
              {'queries':[{'id':'a','query':'a'*601}]},
              {'queries':[{'id':True,'query':'test'}]},
              {'queries':[{'id':'a','query':'   '}]},
              {'queries':[{'id':str(i),'query':'test'} for i in range(33)]},
              {'questions':[{'id':'a','query':'test'}]},
              []]
        for bad in bads:
            with self.subTest(bad=str(bad)[:100]):
                self.queries.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    self.c.input_queries(query=None,query_file=self.queries)
        self.queries.unlink()
        self.queries.symlink_to(self.receipts)
        with self.assertRaises(ValueError):
            self.c.input_queries(query=None,query_file=self.queries)

    def test_completed_ledger_requires_exact_hash_and_no_duplicate_or_untrusted_status(self):
        good=self.c.approved_original_receipts(self.receipts)
        self.assertEqual({'papers/one.pdf':'a'*64},good)
        for items in (
            [{'path':'papers/one.pdf','status':'complete','sha256':'z'*64}],
            [{'path':'papers/one.pdf','status':'complete','sha256':'a'*64}]*2,
            [{'path':'papers/one.pdf','status':'complete','sha256':None}],
            ['malformed'],
        ):
            with self.subTest(items=items):
                self.receipts.write_text(json.dumps({'items':items}))
                with self.assertRaises(ValueError):
                    self.c.approved_original_receipts(self.receipts)
        self.receipts.unlink()
        self.receipts.symlink_to(self.root/'absent-ledger.json')
        with self.assertRaises(ValueError):
            self.c.approved_original_receipts(self.receipts)


if __name__=='__main__':
    unittest.main()
