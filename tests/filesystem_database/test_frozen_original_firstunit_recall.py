"""Synthetic immutable independent-27 / multi-five source QA and safe CLI."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from file_system_catalog.frozen_original_firstunit_recall import (
    evaluate_frozen_firstunit_original_recall,
    complementary_multi_original_source_presence,
)
from scripts import raspi_frozen_original_firstunit_recall as cli


def sha(raw:bytes|str)->str:
    if isinstance(raw,str):raw=raw.encode('utf8')
    return hashlib.sha256(raw).hexdigest()


class FrozenOriginalFirstUnitFixture(unittest.TestCase):
    def make(self):
        independent=[
            {'id':f'ind_{i}','query':f'Original source reference question {i}',
             'expected_path':f'papers/ind_{i}.html'} for i in range(27)]
        multi=[]
        for i,n in enumerate((2,2,2,2,3)):
            multi.append({'id':f'multi_{i}','query':f'Multiple original source question {i}',
                          'relevant_paths':[f'papers/multi_{i}_{j}.html' for j in range(n)]})
        cases=[{'id':z['id'],'query':z['query']} for z in independent+multi]
        gold={p for z in independent for p in (z['expected_path'],)}
        gold.update(p for z in multi for p in z['relevant_paths'])
        self.assertEqual(38,len(gold))
        attested={p:('a'*64,'b'*64,i+1) for i,p in enumerate(sorted(gold))}
        queries=[]
        for z in independent+multi:
            originals=[z['expected_path']] if 'expected_path' in z else z['relevant_paths']
            chans={}
            for ch in ('literal','porter'):
                hits=[]
                for k,p in enumerate(originals,1):
                    preview=f'This approved source preview refers to query {z["id"]}.'
                    hits.append({
                        'original_path':p,'source_rank':k,
                        'current_approved_publisher_and_extracted_bytes_rehashed':True,
                        'quoted_text_is_original_publisher_bytes':False,
                        'factual_answer_verified':False,
                        'source_unit_label_scope':'derived_source_text_only_not_physical_page_proof',
                        'native_original_filedb_entry_id':attested[p][2],
                        'publisher_first_unit_preview':preview,
                        'source_preview_sha256':sha(preview),
                        'source_preview_start_char':10,
                        'source_preview_end_char':10+len(preview),
                        'publisher_original_sha256':attested[p][0],
                        'current_derivative_sha256':attested[p][1],
                    })
                chans[ch]={'returned_current_originals':len(hits),'source_originals':hits}
            queries.append({'id':z['id'],'query':z['query'],
                            'query_sha256':sha(z['query']),
                            'rank_fusion_applied':False,
                            'independent_rank_channels':chans})
        data={'schema':'kmd.readonly_publisher_first_source_unit_dual_lexical_discovery.v1',
              'query_count':len(cases),'independent_rank_channels_no_fusion':True,
              'native_filedb_and_source_map_read_only':True,
              'no_persistent_second_database_created':True,
              'all_displayed_candidates_original_and_full_derivative_sha_reverified':True,
              'originals_downloaded_or_modified':0,
              'source_scan':{'eligible_complete_source_map_records':100,
                             'publisher_firstunit_rows_indexed':100,
                             'skipped_sources':0},
              'queries':queries}
        return independent,multi,{'queries':cases},data,attested

    def score(self,ind,multi,input,actual,attested):
        return evaluate_frozen_firstunit_original_recall(
            independent=ind,multi=multi,queries_file=input,source_report=actual,
            expected_current_native_sha_by_original=attested)


class FrozenFirstUnitEvaluationTests(FrozenOriginalFirstUnitFixture):
    def test_exact_frozen_independent_and_multi_source_references(self):
        ind,multi,inp,actual,attested=self.make()
        data=self.score(ind,multi,inp,actual,attested)
        self.assertEqual(32,data['known_gold_questions'])
        self.assertEqual(38,data['known_positive_labels_including_repeated_paths'])
        self.assertEqual(38,data['known_positive_originals'])
        self.assertEqual(100,data['source_preview_indexed_original_count'])
        self.assertTrue(data['no_score_fusion_original_raw_sources_or_native_ranking_change'])
        self.assertTrue(data['publisher_visual_PDF_page_or_factual_answer_verification_not_claimed'])
        self.assertEqual(76,data['ranked_positive_source_occurrences_cross_checked_against_current_native_ID'])
        self.assertTrue(data['no_union_confidence_or_fused_rank_defined'])
        union=data['distinct_channel_candidate_union_presence_only_not_ranked_top10']
        self.assertEqual(27,union['independent_27']['known_positive_labels_found_in_either_source_channel_at10'])
        self.assertEqual(27,union['independent_27']['questions_with_all_positives_in_either_source_channel_at10'])
        self.assertEqual(11,union['multiple_5']['known_positive_labels_found_in_either_source_channel_at10'])
        i=data['separate_source_channel_metrics']['independent_27']
        m=data['separate_source_channel_metrics']['multiple_5']
        for channel in ('literal','porter'):
            self.assertEqual(27,i[channel]['known_originals_at1'])
            self.assertEqual(27,i[channel]['questions_with_every_known_source_at10'])
            self.assertEqual(11,m[channel]['known_originals_at10'])
            self.assertEqual(5,m[channel]['known_originals_at1'])
            self.assertEqual(5,m[channel]['questions_with_every_known_source_at10'])
        json.dumps(data)

    def test_frozen_questions_and_original_label_mutation_fail_closed(self):
        ind,multi,inp,actual,attested=self.make()
        changed=deepcopy(actual)
        changed['queries'][0]['query']='a new question that changes the source label'
        with self.assertRaisesRegex(ValueError,'changed previously frozen'):
            self.score(ind,multi,inp,changed,attested)
        changed=deepcopy(inp)
        changed['queries'][0]['query']='different query'
        with self.assertRaises(ValueError):self.score(ind,multi,changed,actual,attested)
        changed=deepcopy(multi)
        changed[0]['relevant_paths'][0]='papers/other.html'
        with self.assertRaisesRegex(ValueError,'not all predeclared'):
            self.score(ind,changed,inp,actual,attested)

    def test_original_and_derivative_sha_or_native_ID_mismatch_refused(self):
        ind,multi,inp,actual,attested=self.make()
        a=deepcopy(actual)
        first=a['queries'][0]['independent_rank_channels']['literal']['source_originals'][0]
        first['current_derivative_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'ranked golden original no longer'):
            self.score(ind,multi,inp,a,attested)
        a=deepcopy(actual)
        first=a['queries'][0]['independent_rank_channels']['literal']['source_originals'][0]
        first['native_original_filedb_entry_id']+=1
        with self.assertRaisesRegex(ValueError,'ranked golden original no longer'):
            self.score(ind,multi,inp,a,attested)
        wrong=attested.copy();wrong.pop('papers/ind_0.html')
        with self.assertRaisesRegex(ValueError,'not all predeclared'):
            self.score(ind,multi,inp,actual,wrong)

    def test_no_forged_excerpt_current_original_or_factual_claim(self):
        ind,multi,inp,actual,attested=self.make()
        for k,value in [
            ('source_preview_sha256','f'*64),
            ('factual_answer_verified',True),
            ('quoted_text_is_original_publisher_bytes',True),
            ('current_approved_publisher_and_extracted_bytes_rehashed',False),
            ('source_unit_label_scope','literal_original_PDF_page'),
            ('source_preview_start_char',-1),
            ('source_rank',3),
        ]:
            with self.subTest(field=k):
                changed=deepcopy(actual)
                changed['queries'][0]['independent_rank_channels']['literal']['source_originals'][0][k]=value
                with self.assertRaisesRegex(ValueError,'missing or forged'):
                    self.score(ind,multi,inp,changed,attested)

    def test_incomplete_scoped_or_fused_original_source_results_refused(self):
        ind,multi,inp,actual,attested=self.make()
        for field,value in [
            ('query_count',31),
            ('independent_rank_channels_no_fusion',False),
            ('no_persistent_second_database_created',False),
            ('native_filedb_and_source_map_read_only',False),
            ('originals_downloaded_or_modified',1),
            ('literal_source_directory_prefix','papers/'),
        ]:
            with self.subTest(field=field):
                changed=deepcopy(actual);changed[field]=value
                with self.assertRaises(ValueError):self.score(ind,multi,inp,changed,attested)
        changed=deepcopy(actual);changed['source_scan']['skipped_sources']=1
        with self.assertRaisesRegex(ValueError,'incomplete'):
            self.score(ind,multi,inp,changed,attested)

    def test_duplicate_original_or_rank_and_multi_label_without_source_fail(self):
        ind,multi,inp,actual,attested=self.make()
        changed=deepcopy(actual)
        row=changed['queries'][27]['independent_rank_channels']['literal']
        row['source_originals'][1]=dict(row['source_originals'][0],source_rank=2)
        with self.assertRaisesRegex(ValueError,'duplicate source original'):
            self.score(ind,multi,inp,changed,attested)
        changed=deepcopy(multi);changed[0]['relevant_paths'][1]=changed[0]['relevant_paths'][0]
        with self.assertRaisesRegex(ValueError,'duplicated multi-original'):
            self.score(ind,changed,inp,actual,attested)


class ComplementaryNativeAndPublisherSourceTests(FrozenOriginalFirstUnitFixture):
    def sample(self):
        ind,multi,inp,original,attested=self.make()
        # One known source is missed by both first-unit lexical variants.
        for row in original['queries']:
            if row['id']!='multi_0':
                continue
            for channel in row['independent_rank_channels'].values():
                channel['source_originals'].pop()
                channel['returned_current_originals']=len(channel['source_originals'])
        scored=self.score(ind,multi,inp,original,attested)
        sha_gold=sha(json.dumps({'cases':multi},sort_keys=True))
        native={'schema':'kmd.predeclared_original_retrieval_drift.v1',
                'fixed_gold_sha256':sha_gold,
                'all_publisher_gold_bytes_rechecked':True,
                'complete_fixed_population':True,
                'scope':'only exact path predeclared gold; historical negatives are not labeled unanswerable',
                'cases':[]}
        for case in multi:
            data={z:(i+1) for i,z in enumerate(case['relevant_paths'])}
            if case['id']=='multi_2':
                data={z:None for z in case['relevant_paths']}
            native['cases'].append({
                'id':case['id'],'query_sha256':sha(case['query']),
                'known_relevant_original_paths':case['relevant_paths'],
                'known_relevant_ranks':data})
        return multi,sha_gold,scored,native

    def test_complementary_separate_readers_rescue_all_known_source_labels_without_fusion(self):
        multi,sha_gold,scored,native=self.sample()
        report=complementary_multi_original_source_presence(
            frozen_multi=multi,frozen_multi_full_file_sha256=sha_gold,
            verified_firstunit_summary=scored,current_native_report=native)
        s=report['scores']
        self.assertEqual(11,s['positive_originals'])
        self.assertEqual(9,s['native_top10_source_positives'])
        self.assertEqual(10,s['original_firstunit_either_channel_top10_source_positives'])
        self.assertEqual(11,s['unranked_source_presence_union_positives'])
        self.assertEqual(2,s['source_labels_found_exclusively_in_original_firstunit'])
        self.assertEqual(1,s['source_labels_found_exclusively_in_native'])
        self.assertEqual(5,s['questions_with_all_known_positive_originals_in_unranked_union'])
        self.assertTrue(report['no_single_top10_source_union_rank_defined'])
        self.assertTrue(report['not_a_global_rank_quality_or_answer_accuracy_result'])
        json.dumps(report)

    def test_native_frozen_sha_query_source_order_and_trust_flags_must_agree(self):
        multi,sha_gold,scored,native=self.sample()
        for field,value in (
            ('fixed_gold_sha256','b'*64),
            ('all_publisher_gold_bytes_rechecked',False),
            ('complete_fixed_population',False),
            ('scope','general answer classifier'),
        ):
            with self.subTest(field=field),self.assertRaises(ValueError):
                complementary_multi_original_source_presence(
                    frozen_multi=multi,frozen_multi_full_file_sha256=sha_gold,
                    verified_firstunit_summary=scored,
                    current_native_report={**native,field:value})
        altered=deepcopy(native)
        altered['cases'][2]['query_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'frozen question'):
            complementary_multi_original_source_presence(
                frozen_multi=multi,frozen_multi_full_file_sha256=sha_gold,
                verified_firstunit_summary=scored,current_native_report=altered)
        altered=deepcopy(native)
        altered['cases'][1]['known_relevant_original_paths']=list(
            reversed(altered['cases'][1]['known_relevant_original_paths']))
        with self.assertRaisesRegex(ValueError,'frozen question'):
            complementary_multi_original_source_presence(
                frozen_multi=multi,frozen_multi_full_file_sha256=sha_gold,
                verified_firstunit_summary=scored,current_native_report=altered)

class FrozenFirstUnitCurrentSourceAuditTests(FrozenOriginalFirstUnitFixture):
    def setUp(self):
        self.dir=TemporaryDirectory();self.addCleanup(self.dir.cleanup)
        self.root=Path(self.dir.name)
        self.originals=self.root/'originals'
        self.derived=self.root/'derived'
        self.originals.mkdir();self.derived.mkdir()
        self.native=self.root/'filedb.sqlite3'
        self.sm=self.root/'source-map.sqlite3'
        self.catalog=self.root/'approved.json'
        self.ledger=self.root/'completed.json'
        self.ind,self.multi,self.inp,self.source,self.gold=self.make()
        self.orig_bytes={}
        self.derived_bytes={}
        with sqlite3.connect(self.sm) as sm,sqlite3.connect(self.native) as native:
            sm.execute('create table originals(source_path text,source_sha256 text,derived_sha256 text,derived_path text,original_bytes int,original_url text,original_kind text,status text)')
            native.execute('create table filesystem_entries(id int,relative_path_display text,entry_type text,size_bytes int)')
            catalog=[];ledger=[]
            for path,(_sha,_derived,ident) in self.gold.items():
                p=self.originals/path; p.parent.mkdir(parents=True,exist_ok=True)
                d=self.derived/(path+'.txt');d.parent.mkdir(parents=True,exist_ok=True)
                raw=f'Original publisher source number {ident} with archived evidence.'.encode()
                text=f'KMD extracted source from original document {ident}.'.encode()
                p.write_bytes(raw);d.write_bytes(text)
                self.orig_bytes[path]=raw;self.derived_bytes[path]=text
                ssha,dsha=sha(raw),sha(text)
                self.gold[path]=(ssha,dsha,ident)
                url=f'https://publisher.example.test/original/{ident}.html'
                catalog.append({'path':path,'kind':'html','source_url':url,'url':url})
                ledger.append({'path':path,'status':'complete','sha256':ssha,'bytes_received':len(raw)})
                sm.execute('insert into originals values(?,?,?,?,?,?,?,?)',
                           (path,ssha,dsha,path+'.txt',len(raw),url,'html','complete'))
                native.execute('insert into filesystem_entries values(?,?,?,?)',(ident,path,'file',len(raw)))
        self.catalog.write_text(json.dumps({'sources':catalog}))
        self.ledger.write_text(json.dumps({'items':ledger}))
        for row in self.source['queries']:
            for c in row['independent_rank_channels'].values():
                for hit in c['source_originals']:
                    path=hit['original_path']
                    hit['publisher_original_sha256']=self.gold[path][0]
                    hit['current_derivative_sha256']=self.gold[path][1]

    def kwargs(self):
        return {'independent':self.ind,'multi':self.multi,
                'original_root':self.originals,'derived_root':self.derived,
                'source_map':self.sm,'native_db':self.native,
                'catalog':self.catalog,'ledger':self.ledger}

    def test_current_original_full_sha_and_unique_native_id_are_validated(self):
        proof=cli.source_attestation(**self.kwargs())
        self.assertEqual(self.gold,proof)
        self.assertEqual(38,len(proof))

    def test_original_and_derived_and_native_changes_fail_before_positive_result(self):
        first='papers/ind_0.html'
        original=self.originals/first
        original.write_bytes(b'x'*len(self.orig_bytes[first]))
        with self.assertRaisesRegex(ValueError,'original or derived SHA'):
            cli.source_attestation(**self.kwargs())
        original.write_bytes(self.orig_bytes[first])
        derived=self.derived/(first+'.txt')
        derived.write_bytes(b'x'*len(self.derived_bytes[first]))
        with self.assertRaisesRegex(ValueError,'original or derived SHA'):
            cli.source_attestation(**self.kwargs())
        derived.write_bytes(self.derived_bytes[first])
        with sqlite3.connect(self.native) as d:
            d.execute("update filesystem_entries set id=99999 where relative_path_display=?",(first,))
        proof=cli.source_attestation(**self.kwargs())
        with self.assertRaisesRegex(ValueError,'ranked golden original no longer'):
            evaluate_frozen_firstunit_original_recall(independent=self.ind,multi=self.multi,
                queries_file=self.inp,source_report=self.source,
                expected_current_native_sha_by_original=proof)

    def test_current_real_cli_saves_only_frozen_source_counts_under_private_native_audit(self):
        audit=self.root/'search-audit'
        audit.mkdir(mode=0o700)
        ind=self.root/'ind.json'
        multi=self.root/'multi.json'
        qs=self.root/'queries.json'
        source=self.root/'source.json'
        target=audit/'frozen_original_sha_source_rank.json'
        ind.write_text(json.dumps(self.ind))
        multi.write_text(json.dumps({'cases':self.multi}))
        qs.write_text(json.dumps(self.inp))
        source.write_text(json.dumps(self.source))
        argv=['raspi_frozen_original_firstunit_recall.py',
              '--independent-frozen',str(ind),
              '--multi-frozen',str(multi),
              '--expected-independent-gold-sha256',sha(ind.read_bytes()),
              '--expected-multi-gold-sha256',sha(multi.read_bytes()),
              '--predeclared-query-file',str(qs),
              '--firstunit-source-report',str(source),
              '--output',str(target),
              '--original-root',str(self.originals),
              '--derived-root',str(self.derived),
              '--source-map',str(self.sm),
              '--native-db',str(self.native),
              '--catalog',str(self.catalog),
              '--completed-ledger',str(self.ledger)]
        with mock.patch.object(sys,'argv',argv):
            cli.main()
        data=json.loads(target.read_bytes())
        self.assertEqual('kmd.current_frozen_original_firstunit_source_recall_private.v1',data['schema'])
        self.assertEqual(38,data['positive_gold_publisher_originals_full_SHA_native_ID_rechecked_twice'])
        self.assertEqual(27,data['source_recall']['separate_source_channel_metrics']['independent_27']['porter']['known_originals_at10'])
        self.assertTrue(data['no_original_bytes_or_untrusted_source_excerpt_copied_into_benchmark'])
        self.assertTrue(data['frozen_source_gold_sha256_matched_explicit_prior_predeclared_hashes'])
        self.assertEqual(0o600,target.stat().st_mode&0o777)
        self.assertNotIn('Original publisher source number',target.read_text())
        # Even an existing owner-private audit cannot be overwritten. Output
        # refusal occurs before any live source/read work is attempted.
        with mock.patch.object(sys,'argv',argv),mock.patch.object(cli,'source_attestation') as attestor:
            with self.assertRaisesRegex(ValueError,'must be new'):
                cli.main()
            attestor.assert_not_called()
        external=self.root/'untrusted_report_outside_original_source_private_tree.json'
        argv[argv.index(str(target))]=str(external)
        with mock.patch.object(sys,'argv',argv),mock.patch.object(cli,'source_attestation') as attestor:
            with self.assertRaisesRegex(ValueError,'owner-private'):
                cli.main()
            attestor.assert_not_called()
        self.assertFalse(external.exists())

    def test_wrong_predeclared_frozen_gold_full_byte_shas_fail_before_original_scan(self):
        audit=self.root/'search-audit'
        audit.mkdir()
        ind=self.root/'independent-gold.json'
        multi=self.root/'multiple-gold.json'
        q=self.root/'frozen-query.json'
        report=self.root/'current-source-report.json'
        ind.write_text(json.dumps(self.ind))
        multi.write_text(json.dumps({'cases':self.multi}))
        q.write_text(json.dumps(self.inp))
        report.write_text(json.dumps(self.source))
        argv=['frozen-source-eval.py','--independent-frozen',str(ind),
              '--multi-frozen',str(multi),
              '--expected-independent-gold-sha256','f'*64,
              '--expected-multi-gold-sha256',sha(multi.read_bytes()),
              '--predeclared-query-file',str(q),
              '--firstunit-source-report',str(report),
              '--output',str(audit/'should_not_exist.json'),
              '--original-root',str(self.originals),
              '--derived-root',str(self.derived),
              '--source-map',str(self.sm),
              '--native-db',str(self.native),
              '--catalog',str(self.catalog),
              '--completed-ledger',str(self.ledger)]
        with mock.patch.object(sys,'argv',argv),mock.patch.object(cli,'source_attestation') as validator:
            with self.assertRaisesRegex(ValueError,'changed since externally predeclared SHA'):
                cli.main()
            validator.assert_not_called()
        self.assertFalse((audit/'should_not_exist.json').exists())

    def test_duplicate_native_original_ID_is_rejected_by_current_gold_identity(self):
        first='papers/ind_0.html'
        fid=self.gold[first][2]
        with sqlite3.connect(self.native) as d:
            d.execute("insert into filesystem_entries values(?,?,?,?)",
                      (fid,first,'file',len(self.orig_bytes[first])))
        with self.assertRaisesRegex(ValueError,'missing/ambiguous native'):
            cli.source_attestation(**self.kwargs())

    def test_original_symlink_cannot_serve_as_replacement_approved_document(self):
        first='papers/ind_0.html'
        original=self.originals/first
        alternate=self.root/'external.pdf'
        alternate.write_bytes(self.orig_bytes[first])
        original.unlink()
        original.symlink_to(alternate)
        with self.assertRaises(ValueError):
            cli.source_attestation(**self.kwargs())

    def test_forged_or_duplicate_acquisition_approval_refused(self):
        row=json.loads(self.ledger.read_bytes())
        row['items'][0]['status']='pending'
        self.ledger.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError,'catalog/source map/acquisition'):
            cli.source_attestation(**self.kwargs())


if __name__=='__main__': unittest.main()
