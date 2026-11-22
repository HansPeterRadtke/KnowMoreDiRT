"""Real private-file SHA proof and strict current-source revision admission."""
from __future__ import annotations
import hashlib
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from file_system_catalog.source_triage_provenance import verify_selected_source


class SourceTriageProvenanceTests(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        self.root=root/'originals';self.derived=root/'derived'
        self.path='research/a/actual-primary-reference.html'
        for directory in (self.root,self.derived):
            (directory/'research/a').mkdir(parents=True)
        self.original=self.root/self.path
        self.text=self.derived/(self.path+'.txt')
        self.original.write_bytes(b'<html><body>Primary publisher source.</body></html>')
        self.text.write_bytes(b'[[SOURCE UNIT: html body]]\nPrimary publisher source.')
        self.osh=hashlib.sha256(self.original.read_bytes()).hexdigest()
        self.dsh=hashlib.sha256(self.text.read_bytes()).hexdigest()
        self.database=root/'source-map.sqlite3'
        with sqlite3.connect(self.database) as db:
            db.execute('CREATE TABLE originals (source_path TEXT,source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,status TEXT,original_bytes INTEGER)')
            db.execute('INSERT INTO originals VALUES (?,?,?,?,?,?)',
                 (self.path,self.osh,self.dsh,self.path+'.txt','complete',self.original.stat().st_size))
        self.version={self.path:(self.osh,self.dsh,70)}
        self.ledger={self.path:{'status':'complete','sha256':self.osh,
                     'bytes_received':self.original.stat().st_size}}
        self.catalog={self.path:{'kind':'html','title':'Actual original reference',
                       'source_url':'https://original.example/reference',
                       'publication_year':2026,'rights_note':'Publisher copyright remains with author.'}}
        self.ro=sqlite3.connect(self.database.resolve().as_uri()+'?mode=ro',uri=True)
        self.ro.execute('PRAGMA query_only=ON')
        self.addCleanup(self.ro.close)

    def run_proof(self,**kwargs):
        args={'path':self.path,'source_versions':self.version,'source_map':self.ro,
              'approval_ledger':self.ledger,'catalog':self.catalog,
              'original_root':self.root,'derived_root':self.derived}
        args.update(kwargs)
        return verify_selected_source(**args)

    def test_true_original_and_current_derivative_full_sha_and_factual_scope(self):
        result=self.run_proof()
        self.assertEqual(self.path,result['original_path'])
        self.assertEqual(self.osh,result['original_sha256'])
        self.assertEqual(self.dsh,result['current_derived_sha256'])
        self.assertTrue(result['publisher_original_actual_sha256_verified'])
        self.assertTrue(result['source_map_derivative_actual_sha256_verified'])
        self.assertFalse(result['individual_original_physical_pdf_pages_verified'])
        self.assertFalse(result['factual_answer_or_claim_supported_by_retrieval_alone'])
        self.assertEqual('Publisher copyright remains with author.',result['license_or_rights_note'])

    def test_tampered_original_byte_refused(self):
        self.original.write_bytes(self.original.read_bytes()+b'untrusted tail')
        with self.assertRaisesRegex(ValueError,'actual publisher original'):
            self.run_proof()

    def test_tampered_current_derived_byte_refused(self):
        self.text.write_bytes(b'fabricated source')
        with self.assertRaisesRegex(ValueError,'actual publisher original'):
            self.run_proof()

    def test_stale_revision_and_ambiguous_map_refused(self):
        with self.assertRaisesRegex(ValueError,'source-map/approved original'):
            self.run_proof(source_versions={self.path:(self.osh,'0'*64,70)})
        self.ledger[self.path]['bytes_received']=123
        with self.assertRaisesRegex(ValueError,'source-map/approved original'):
            self.run_proof()

    def test_read_only_map_required_not_write_connection(self):
        with sqlite3.connect(self.database) as writable:
            with self.assertRaisesRegex(ValueError,'query-only'):
                self.run_proof(source_map=writable)

    def test_symlinked_source_and_directory_traversal_rejected(self):
        shadow=self.root/'research/a/shadow.html'
        shadow.symlink_to(self.original)
        values=dict(self.version);values['research/a/shadow.html']=(self.osh,self.dsh,70)
        approved=dict(self.ledger);approved['research/a/shadow.html']=dict(self.ledger[self.path])
        catalog=dict(self.catalog);catalog['research/a/shadow.html']=dict(self.catalog[self.path])
        with self.assertRaises(ValueError):self.run_proof(path='../invalid')
        with self.assertRaises(ValueError):self.run_proof(path='research/a/shadow.html',source_versions=values,approval_ledger=approved,catalog=catalog)

if __name__=='__main__':unittest.main()
