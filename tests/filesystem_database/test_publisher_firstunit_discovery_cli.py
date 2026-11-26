"""CLI contract: exactly scoped queries, completed receipts, no guessed fallback."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
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
        mixed=[*cases,{'id':'third','query':'which agent paper reports model results',
                      'require_source_phrase':'AgentMemRec'}]
        self.queries.write_text(json.dumps({'queries':mixed}))
        self.assertEqual(mixed,self.c.input_queries(query=None,query_file=self.queries))
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
              {'queries':[{'id':'a','query':'test','require_source_phrase':None}]},
              {'queries':[{'id':'a','query':'test','require_source_phrase':'A\nB'}]},
              {'queries':[{'id':'a','query':'test','require_source_phrase':'x'*161}]},
              {'queries':[{'id':'a','query':'test','require_source_phrase':'match','source':'other'}]},
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


class PublisherFirstUnitRequiredSourcePhraseEndToEndTests(unittest.TestCase):
    def setUp(self):
        folder=TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root=Path(folder.name)
        self.orig=self.root/'approved-originals';self.orig.mkdir()
        self.derived=self.root/'current-derivatives';self.derived.mkdir()
        self.source_map=self.root/'source-map.sqlite3'
        self.native=self.root/'native-fileDB.sqlite3'
        self.ledger=self.root/'completed.json'
        self.originals={}
        info=[
            ('documents/first.md','A study of wedding album proof source',
             '[[SOURCE UNIT: Markdown page 1]]\n'
             'Verified original research about wedding album proof source from trips. '
             'The evaluation protocol compares coherent photo collections rather '
             'than isolated random images.\n'
             '[[SOURCE UNIT: Markdown page 2]]\n'
             'Unrelated future work and possible additional ideas.'),
            ('documents/late.md','A catalog title naming wedding album proof source',
             'Catalog metadata heading: includes wedding album proof source, '
             'but this heading is not the publisher original.\n'
             '[[SOURCE UNIT: Markdown page 1]]\n'
             'This earlier section concerns GPU memory scheduling and batching. '
             'No statement about photo collections is established here.\n'
             '[[SOURCE UNIT: Markdown page 2]]\n'
             'A later publisher page discusses wedding album proof source '
             'and visual reasoning, not part of the first source unit.'),
            ('documents/header.md','A generic source manual',
             'Catalog title wrongly claiming wedding album proof source.\n'
             '[[SOURCE UNIT: Markdown page 1]]\n'
             'An ordinary source note about databases, operations, and logs '
             'which does not discuss a photo collection at all.'),
        ]
        receipts=[]
        with sqlite3.connect(self.source_map) as sm,sqlite3.connect(self.native) as n:
            sm.execute('CREATE TABLE originals(source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,original_title TEXT,status TEXT)')
            n.execute('CREATE TABLE filesystem_entries(id INTEGER,relative_path_display TEXT,entry_type TEXT)')
            for index,(path,title,body) in enumerate(info,1):
                original=self.orig/path;original.parent.mkdir(exist_ok=True)
                original.write_text('Original publisher document identity: '+path)
                derivative=self.derived/(path+'.txt');derivative.parent.mkdir(exist_ok=True)
                derivative.write_text(body)
                osh=hashlib.sha256(original.read_bytes()).hexdigest()
                dsh=hashlib.sha256(derivative.read_bytes()).hexdigest()
                receipts.append({'path':path,'sha256':osh,'status':'complete'})
                sm.execute('INSERT INTO originals VALUES(?,?,?,?,?,?)',
                           (path,osh,dsh,path+'.txt',title,'complete'))
                n.execute('INSERT INTO filesystem_entries VALUES(?,?,?)',
                          (index,path,'file'))
                self.originals[path]={'title':title,'sha':osh,'derived_sha':dsh,
                                      'original':original,'derived':derivative}
        self.ledger.write_text(json.dumps({'items':receipts}))
        self.cli=(Path(__file__).resolve().parents[2]/
                  'scripts/raspi_publisher_firstunit_discovery.py')
        self.repo=Path(__file__).resolve().parents[2]
        self.base_args=[
            sys.executable,str(self.cli),
            '--query','wedding album proof source documents',
            '--top','3',
            '--source-map',str(self.source_map),
            '--native-db',str(self.native),
            '--original-root',str(self.orig),
            '--derivative-root',str(self.derived),
            '--completed-ledger',str(self.ledger),
        ]

    def run_cli(self,outfile:str,extra=()):
        out=self.root/outfile
        proc=subprocess.run(
            [*self.base_args,*extra,'--output',str(out)],
            cwd=self.repo,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',
                               'PYTHONPATH':'src:.'},
            capture_output=True,text=True,timeout=25,
        )
        return proc,out

    def test_real_sqlite_and_original_sha_end_to_end_strict_firstunit(self):
        before={path:(i['original'].read_bytes(),i['derived'].read_bytes())
                for path,i in self.originals.items()}
        prior,file1=self.run_cli('legacy-default.json')
        self.assertEqual(0,prior.returncode,prior.stderr[-900:])
        old=json.loads(file1.read_text())
        self.assertEqual('kmd.readonly_publisher_first_source_unit_dual_lexical_discovery.v1',
                         old['schema'])
        self.assertNotIn('explicit_required_first_source_unit_phrase',old)
        self.assertEqual(3,old['source_scan']['publisher_firstunit_rows_indexed'])
        self.assertGreaterEqual(old['queries'][0]['independent_rank_channels']['porter']['returned_current_originals'],2)
        required,file2=self.run_cli(
            'strict-verified-unit.json',
            extra=('--require-source-phrase','wedding album proof source'),
        )
        self.assertEqual(0,required.returncode,required.stderr[-900:])
        fresh=json.loads(file2.read_text())
        self.assertEqual('kmd.optional_attested_first_source_unit_phrase_gated_lexical.v1',
                         fresh['schema'])
        self.assertEqual('wedding album proof source',fresh['explicit_required_first_source_unit_phrase'])
        self.assertTrue(fresh['ranked_candidate_scan_bounded_and_not_exhaustive_full_corpus_phrase_search'])
        for channel in ('literal','porter'):
            r=fresh['queries'][0]['independent_rank_channels'][channel]
            self.assertEqual(1,r['returned_current_originals'])
            self.assertEqual([],r['stale_or_unverified_candidates_omitted'])
            self.assertGreaterEqual(len(r['verified_source_unit_candidates_missing_requested_phrase']),1)
            row=r['source_originals'][0]
            self.assertEqual('documents/first.md',row['original_path'])
            proof=row['explicit_required_source_phrase_proof']
            self.assertTrue(proof['title_and_generated_metadata_not_used_for_phrase_acceptance'])
            self.assertTrue(proof['entire_approved_original_and_derived_bytes_verified_before_matching'])
            self.assertFalse(row['factual_answer_verified'])
            self.assertEqual(1,proof['matches_in_verified_first_original_unit'])
            original=self.originals[row['original_path']]
            self.assertEqual(original['sha'],row['publisher_original_sha256'])
            self.assertEqual(original['derived_sha'],row['current_derivative_sha256'])
            att=proof['source_attestations'][0]
            derived_text=original['derived'].read_text()
            self.assertEqual('wedding album proof source',
                             derived_text[att['source_start_char']:att['source_end_char']])
            self.assertEqual(att['literal_phrase_sha256'],hashlib.sha256(
                derived_text[att['source_start_char']:att['source_end_char']].encode()).hexdigest())
            self.assertEqual(att['source_context_excerpt'],derived_text[
                att['source_context_start_char']:att['source_context_end_char']])
            self.assertEqual(att['source_context_sha256'],hashlib.sha256(
                att['source_context_excerpt'].encode()).hexdigest())
            self.assertTrue(att['not_a_factual_answer_or_entailment_proof'])
            self.assertEqual('Markdown page 1',att['source_unit_label'])
            self.assertTrue(r['required_phrase_checks_are_only_on_bounded_ranked_candidates'])
            for excluded in r['verified_source_unit_candidates_missing_requested_phrase']:
                self.assertTrue(excluded['current_original_and_derived_SHA_verified'])
                self.assertTrue(excluded['not_a_claim_about_other_document_pages_or_figure_content'])
        for path,original in self.originals.items():
            self.assertEqual(before[path],(
                original['original'].read_bytes(),original['derived'].read_bytes()
            ))

    def test_real_cli_refuses_to_write_partial_result_when_verified_source_read_is_locked(self):
        c=module()
        output=self.root/'should-not-write-stale-source-as-empty.json'
        exception=sqlite3.OperationalError('database is locked')
        exception.sqlite_errorcode=sqlite3.SQLITE_BUSY
        argv=[str(self.cli),*self.base_args[2:],
              '--require-source-phrase','wedding album proof source',
              '--output',str(output)]
        with patch.object(sys,'argv',argv),\
             patch.object(c,'verify_index_candidate',
                          side_effect=[exception,exception]) as check,\
             patch.object(c.time,'sleep') as delay:
            with self.assertRaisesRegex(RuntimeError,'query aborted'):
                c.main()
            self.assertEqual(2,check.call_count)
            delay.assert_called_once_with(0.4)
        self.assertFalse(output.exists(),
                         'Persistent SQLITE_BUSY must not become a false no-match JSON receipt')

    def test_real_cli_unexpected_database_error_is_not_reported_as_unverified_source(self):
        c=module()
        output=self.root/'should-not-write-schema-failure.json'
        argv=[str(self.cli),*self.base_args[2:],
              '--require-source-phrase','wedding album proof source',
              '--output',str(output)]
        with patch.object(sys,'argv',argv),\
             patch.object(c,'verify_index_candidate',
                          side_effect=sqlite3.OperationalError('no such table: important')) as verify,\
             patch.object(c.time,'sleep') as delay:
            with self.assertRaisesRegex(sqlite3.OperationalError,'no such table'):
                c.main()
            self.assertEqual(1,verify.call_count)
            delay.assert_not_called()
        self.assertFalse(output.exists())

    def test_multiple_distinct_per_question_required_phrases_share_one_source_index(self):
        batch=[
            {'id':'known','query':'wedding album proof source documents',
             'require_source_phrase':'wedding album proof source'},
            {'id':'negative','query':'wedding album proof source documents',
             'require_source_phrase':'ZJXQ-NO-ORIGINAL-ENTITY-2026'},
            {'id':'ordinary','query':'wedding album proof source documents'},
        ]
        source=self.root/'individual-phrase-batch.json'
        source.write_text(json.dumps({'queries':batch}))
        output=self.root/'per-question-proof.json'
        args=[sys.executable,str(self.cli),
            '--queries-file',str(source),*self.base_args[4:],
            '--output',str(output)]
        proc=subprocess.run(args,cwd=self.repo,capture_output=True,text=True,
            timeout=25,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'src:.'})
        self.assertEqual(0,proc.returncode,proc.stderr[-900:])
        response=json.loads(output.read_text())
        self.assertEqual(3,response['query_count'])
        self.assertEqual(3,response['source_scan']['publisher_firstunit_rows_indexed'])
        self.assertEqual(2,response['per_query_source_phrase_count'])
        self.assertIsNone(response['global_required_first_source_unit_phrase'])
        self.assertEqual(
            ['known','negative','ordinary'],[row['id'] for row in response['queries']])
        for row in response['queries']:
            channel=row['independent_rank_channels']['porter']
            if row['id']=='known':
                self.assertEqual(1,channel['returned_current_originals'])
                self.assertEqual('documents/first.md',channel['source_originals'][0]['original_path'])
                self.assertTrue(row['source_phrase_rule_applied_to_this_question'])
            elif row['id']=='negative':
                self.assertEqual(0,channel['returned_current_originals'])
                self.assertTrue(row['source_phrase_rule_applied_to_this_question'])
                self.assertGreaterEqual(
                    len(channel['verified_source_unit_candidates_missing_requested_phrase']),1)
            else:
                self.assertNotIn('source_phrase_rule_applied_to_this_question',row)
                self.assertGreaterEqual(channel['returned_current_originals'],2)
                self.assertNotIn('verified_source_unit_candidates_missing_requested_phrase',channel)
        collision=self.root/'reject-global-and-perquery.json'
        args=[sys.executable,str(self.cli),'--queries-file',str(source),
              *self.base_args[4:],
              '--require-source-phrase','wedding album proof source',
              '--output',str(collision)]
        bad=subprocess.run(args,cwd=self.repo,capture_output=True,text=True,timeout=25,
                           env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'src:.'})
        self.assertNotEqual(0,bad.returncode)
        self.assertIn('conflict',bad.stderr)
        self.assertFalse(collision.exists())

    def test_nonexistent_phrase_proves_only_absence_in_ranked_firstunits(self):
        proc,out=self.run_cli('strict-nonexistent.json',extra=(
            '--require-source-phrase','NONEXISTENT-PROTECTED-TOKEN-4321',
        ))
        self.assertEqual(0,proc.returncode,proc.stderr[-700:])
        data=json.loads(out.read_text())
        self.assertTrue(data['negative_or_empty_results_do_not_prove_absence_in_other_units'])
        for channel in ('literal','porter'):
            item=data['queries'][0]['independent_rank_channels'][channel]
            self.assertEqual(0,item['returned_current_originals'])
            self.assertEqual([],item['source_originals'])
            self.assertGreaterEqual(len(item['verified_source_unit_candidates_missing_requested_phrase']),1)
        wrong,out=self.run_cli('invalid-phrase.json',extra=(
            '--require-source-phrase','\ninvalid',
        ))
        self.assertNotEqual(0,wrong.returncode)
        self.assertFalse(out.exists())




class OnlyKnownReadonlySQLiteBusyMayRetryTests(unittest.TestCase):
    def setUp(self):
        self.c=module()

    @staticmethod
    def sql_error(code,detail='database is locked'):
        err=sqlite3.OperationalError(detail)
        err.sqlite_errorcode=code
        return err

    def test_valid_sqlite_busy_and_locked_have_only_one_readonly_retry(self):
        for code in (sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED):
            with self.subTest(code=code):
                expected={'approved_original_sha256':'a'*64}
                with patch.object(self.c,'verify_index_candidate',
                                  side_effect=[self.sql_error(code),expected]) as run,\
                     patch.object(self.c.time,'sleep') as delay:
                    actual=self.c.verify_original_candidate_with_bounded_busy_retry(
                        entry='synthetic-approved-candidate',
                    )
                    self.assertEqual(expected,actual)
                    self.assertEqual(2,run.call_count)
                    delay.assert_called_once_with(0.4)

    def test_second_busy_fails_complete_search_not_empty_misclassification(self):
        for code in (sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED):
            with self.subTest(code=code),\
                 patch.object(self.c,'verify_index_candidate',
                              side_effect=[self.sql_error(code),self.sql_error(code)]) as run,\
                 patch.object(self.c.time,'sleep') as delay:
                with self.assertRaisesRegex(RuntimeError,'query aborted'):
                    self.c.verify_original_candidate_with_bounded_busy_retry(
                        entry='synthetic-approved-candidate')
                self.assertEqual(2,run.call_count)
                delay.assert_called_once_with(0.4)

    def test_unknown_sqlite_failure_never_retried_or_misclassified(self):
        for err in (sqlite3.OperationalError('no such table: originals'),
                    self.sql_error(sqlite3.SQLITE_CORRUPT,'database is corrupt')):
            with self.subTest(error=str(err)),\
                 patch.object(self.c,'verify_index_candidate',side_effect=err) as run,\
                 patch.object(self.c.time,'sleep') as delay:
                with self.assertRaises(sqlite3.OperationalError):
                    self.c.verify_original_candidate_with_bounded_busy_retry(
                        entry='synthetic-approved-candidate')
                self.assertEqual(1,run.call_count)
                delay.assert_not_called()

    def test_real_private_sqlite_exclusive_lock_carries_busy_code(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'exclusive-lock-test.sqlite3'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE important_source(val INTEGER)')
                db.execute('INSERT INTO important_source VALUES (1)')
            owner=sqlite3.connect(path)
            other=sqlite3.connect(path,timeout=0)
            try:
                owner.execute('BEGIN EXCLUSIVE')
                with self.assertRaises(sqlite3.OperationalError) as raised:
                    other.execute('SELECT * FROM important_source').fetchall()
                self.assertTrue(self.c._sqlite_busy_or_locked(raised.exception))
                self.assertEqual(sqlite3.SQLITE_BUSY,
                                 raised.exception.sqlite_errorcode & 255)
            finally:
                owner.rollback();other.close();owner.close()



if __name__=='__main__':
    unittest.main()
