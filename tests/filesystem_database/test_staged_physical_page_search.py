"""No-write, original physical PDF page verified staged text retrieval."""
from __future__ import annotations
import hashlib
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.staged_physical_page_search import (
    StagedPhysicalPageRAMIndex,source_units_from_corrected_text,verify_staged_page_result,
)


def digest(data:bytes):return hashlib.sha256(data).hexdigest()


class StagedPhysicalPageSearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.orig=self.root/'original';self.live=self.root/'live';self.stage=self.root/'private'
        for p in (self.orig,self.live,self.stage):p.mkdir()
        self.relative='ai-agents/known-source.pdf'
        self.original=b'%PDF-private-original-fixture'
        self.old=b'old cached inaccurate page labels'
        self.source=("KMD source provenance original fixture\n{}\n\n"
             "[[SOURCE UNIT: PDF page 1]]\nSpeech synthesis requires punctuation boundaries and model evaluation.\n"
             "[[SOURCE UNIT: PDF page 2]]\nVector retrieval must cite original publisher physical pages and preserve page provenance.\n")
        self.staged=self.source.encode()
        for base,content,suffix in ((self.orig,self.original,''),(self.live,self.old,'.txt'),(self.stage,self.staged,'.txt')):
            path=base/(self.relative+suffix);path.parent.mkdir();path.write_bytes(content)
        self.item={'original_path':self.relative,'original_sha256':digest(self.original),
                   'old_derived_sha256':digest(self.old),'new_staged_derived_sha256':digest(self.staged),
                   'staged_private_file':str(self.stage/(self.relative+'.txt')),
                   'native_original_filedb_id':33,'original_pdf_physical_pages':2,'new_max_staged_page_label':2}
        self.sm=sqlite3.connect(':memory:');self.addCleanup(self.sm.close)
        self.sm.execute('CREATE TABLE originals (source_sha256 TEXT,derived_sha256 TEXT,derived_path TEXT,status TEXT,source_path TEXT)')
        self.sm.execute('INSERT INTO originals VALUES (?,?,?,?,?)',(digest(self.original),digest(self.old),self.relative+'.txt','complete',self.relative))
        self.native=sqlite3.connect(':memory:');self.addCleanup(self.native.close)
        self.native.execute('CREATE TABLE filesystem_entries (id INTEGER,entry_type TEXT,relative_path_display TEXT)')
        self.native.execute('INSERT INTO filesystem_entries VALUES (?,?,?)',(33,'file',self.relative))
        self.sm.commit();self.native.commit()

    def test_both_page_units_without_catalog_header(self):
        units=list(source_units_from_corrected_text(self.source,2,2))
        self.assertEqual([1,2],[row[1] for row in units])
        self.assertNotIn('provenance original fixture',units[0][2])
        self.assertIn('physical pages',units[1][2])

    def test_out_of_bounds_or_repeated_pages_rejected(self):
        with self.assertRaisesRegex(ValueError,'out-of-bounds'):
            list(source_units_from_corrected_text(self.source,1,2))
        broken=self.source.replace('PDF page 2','PDF page 1')
        with self.assertRaisesRegex(ValueError,'repeated'):
            list(source_units_from_corrected_text(broken,2,1))

    def test_exact_verified_publisher_page_result(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            ranked=index.ranked('publisher physical page provenance')
            self.assertTrue(ranked)
            self.assertEqual(2,ranked[0][0][2])
            def verified(_source,_sha,passages):
                self.assertEqual('PDF page 2',passages[0]['source_section'])
                self.assertIn('preserve page provenance',passages[0]['source_excerpt'])
                return [{'status':'verified','page':2,'basis':'derivative_label_confirmed','page_text_sha256':'x'*64}]
            r=verify_staged_page_result(ranked[0],query='publisher physical page provenance',
                original_root=self.orig,derived_root=self.live,stage_root=self.stage,
                completed_receipts={self.relative:digest(self.original)},
                source_map=self.sm,native_db=self.native,verifier=verified)
            self.assertEqual('publisher_original_physical_pdf_page_verified',r['status'])
            self.assertEqual(2,r['physical_pdf_page'])
            self.assertFalse(r['factual_answer_verified'])

    def test_wrong_original_page_never_verified(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            r=verify_staged_page_result(index.ranked('retrieval original publisher')[0],
                query='retrieval original publisher',original_root=self.orig,derived_root=self.live,
                stage_root=self.stage,completed_receipts={self.relative:digest(self.original)},
                source_map=self.sm,native_db=self.native,
                verifier=lambda *_:[{'status':'verified','page':1,'basis':'unique_across_pdf'}])
            self.assertEqual('not_independently_verified',r['status'])

    def test_replaced_derivative_or_staged_source_fails_closed(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            hit=index.ranked('publisher physical page')[0]
            (self.live/(self.relative+'.txt')).write_bytes(b'changed original live derivative')
            with self.assertRaisesRegex(ValueError,'revision differs'):
                verify_staged_page_result(hit,query='publisher physical page',original_root=self.orig,
                    derived_root=self.live,stage_root=self.stage,
                    completed_receipts={self.relative:digest(self.original)},source_map=self.sm,native_db=self.native)

    def test_duplicate_source_entry_rejected(self):
        with StagedPhysicalPageRAMIndex(2) as index:
            index.add(self.item,stage_root=self.stage)
            with self.assertRaisesRegex(ValueError,'duplicate'):index.add(self.item,stage_root=self.stage)

    def test_filename_does_not_rank_without_publisher_body_word(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            self.assertEqual([],index.ranked('known'))

    def test_exact_publisher_page_but_irrelevant_excerpt_is_excluded(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            # The page contains 'publisher' but not multiple relevant terms.
            hit=index.ranked('publisher extraterrestrial ontology')[0]
            result=verify_staged_page_result(hit,query='publisher extraterrestrial ontology',
                original_root=self.orig,derived_root=self.live,stage_root=self.stage,
                completed_receipts={self.relative:digest(self.original)},
                source_map=self.sm,native_db=self.native,
                verifier=lambda *_:[{'status':'verified','page':2,'basis':'derivative_label_confirmed'}])
            self.assertEqual('insufficient_source_query_term_coverage',result['status'])

    def test_no_verified_original_page_without_proof(self):
        with StagedPhysicalPageRAMIndex(1) as index:
            index.add(self.item,stage_root=self.stage);index.commit()
            result=verify_staged_page_result(index.ranked('publisher page')[0],query='publisher page',
                original_root=self.orig,derived_root=self.live,stage_root=self.stage,
                completed_receipts={self.relative:digest(self.original)},
                source_map=self.sm,native_db=self.native,
                verifier=lambda *_:[{'status':'unverified','reason':'excerpt_absent'}])
            self.assertNotEqual('publisher_original_physical_pdf_page_verified',result['status'])


if __name__=='__main__':unittest.main()
