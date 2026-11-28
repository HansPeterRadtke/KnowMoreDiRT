"""Source-verified false claim exclusion, no mutation of original index data."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from file_system_catalog.summary_claim_quarantine import (
    SCHEMA, load_reviewed_summary_exclusions, exclude_reviewed_summary_nodes,
)


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


class ReviewedSourceClaimExclusionsTest(unittest.TestCase):
    def setUp(self):
        self.work = TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        base = Path(self.work.name)
        self.proof = base / 'search-audit' / 'original-table' / 'source-review.json'
        self.proof.parent.mkdir(parents=True)
        self.registry = base / 'summary-claim-exclusions.json'
        self.entry = {
            'original_path': 'agent-papers/approved.pdf',
            'original_sha256': 'a' * 64, 'derived_sha256': 'b' * 64,
            'summary_sha256': 'c' * 64,
            'summary_node_id': 'summary:reviewed',
            'native_representation_id': '12345678-1234-4321-abcd-0123456789ab',
            'original_review_path': str(self.proof),
        }
        self.review = {
            'publisher_original_sha256': self.entry['original_sha256'],
            'derived_sha256': self.entry['derived_sha256'],
            'incorrect_existing_native_summary_sha256': self.entry['summary_sha256'],
            'incorrect_existing_native_summary_node_id': self.entry['summary_node_id'],
            'source_specific_corrected_statement': 'The publisher source table assigns this figure to a different system.',
            'original_physical_pdf_page': 8,
            'original_physical_file_bytes_rehashed': True,
            'source_row_receipts': [
                {'exact_source_text': 'different system 80.3%',
                 'line_sha256': sha('different system 80.3%')},
            ],
        }
        self.save()

    def save(self):
        self.proof.write_text(json.dumps(self.review))
        self.entry['original_review_sha256'] = sha(self.proof.read_bytes())
        self.registry.write_text(json.dumps({'schema': SCHEMA, 'entries': [self.entry]}))

    def hit(self, **updates):
        row = {
            'original_path': self.entry['original_path'],
            'original_sha256': self.entry['original_sha256'],
            'derived_sha256': self.entry['derived_sha256'],
            'native_representation_id': self.entry['native_representation_id'],
            'summary_sha256': self.entry['summary_sha256'],
        }
        row.update(updates)
        return row

    def test_exact_original_and_revision_and_representation_are_excluded(self):
        keys = load_reviewed_summary_exclusions(self.registry)
        self.assertEqual(1, len(keys))
        rejected = self.hit()
        current_other_revision = self.hit(derived_sha256='d'*64)
        other_representation = self.hit(native_representation_id='other-valid-representation')
        different_original = self.hit(original_path='agent-papers/another.pdf')
        same_document_different_summary = self.hit(summary_sha256='f'*64)
        hits = [rejected,current_other_revision,other_representation,different_original,same_document_different_summary]
        saved = [dict(r) for r in hits]
        found, dropped = exclude_reviewed_summary_nodes(hits, keys)
        self.assertEqual(1, dropped)
        self.assertEqual(hits[1:], found)
        self.assertEqual(saved, hits)
        self.assertEqual((hits,0), exclude_reviewed_summary_nodes(hits, frozenset()))

    def test_missing_file_is_not_an_implicit_global_block(self):
        self.assertEqual(frozenset(),load_reviewed_summary_exclusions(self.registry.with_name('not-installed.json')))

    def test_changed_source_review_fails_closed(self):
        self.proof.write_text('{}')
        with self.assertRaisesRegex(ValueError,'evidence has changed'):
            load_reviewed_summary_exclusions(self.registry)

    def test_inaccurate_attestation_field_fails_closed(self):
        for key,value in [
            ('publisher_original_sha256','f'*64),
            ('incorrect_existing_native_summary_sha256','f'*64),
            ('original_physical_file_bytes_rehashed',False),
            ('source_row_receipts',[{'exact_source_text':'different system 80.3%','line_sha256':'a'*64}]),
        ]:
            with self.subTest(field=key):
                prior=self.review[key]
                self.review[key]=value;self.save()
                with self.assertRaises(ValueError):
                    load_reviewed_summary_exclusions(self.registry)
                self.review[key]=prior;self.save()

    def test_registry_invalid_or_duplicate_fails_closed(self):
        saved=self.registry.read_bytes()
        self.registry.write_bytes(b'not JSON')
        with self.assertRaises(ValueError): load_reviewed_summary_exclusions(self.registry)
        self.registry.write_bytes(saved)
        self.registry.write_text(json.dumps({'schema': SCHEMA,'entries':[self.entry,self.entry]}))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            load_reviewed_summary_exclusions(self.registry)
        self.registry.write_bytes(saved)
        self.entry['original_path']='../escape.pdf';self.save()
        with self.assertRaises(ValueError):load_reviewed_summary_exclusions(self.registry)

    def test_registry_or_proof_symlink_refused(self):
        saved=self.registry.with_name('separate.json')
        saved.write_bytes(self.registry.read_bytes())
        self.registry.unlink();self.registry.symlink_to(saved)
        with self.assertRaises(ValueError):load_reviewed_summary_exclusions(self.registry)
        self.registry.unlink();self.registry.write_bytes(saved.read_bytes())
        self.proof.rename(self.proof.with_name('actual.json'))
        self.proof.symlink_to(self.proof.with_name('actual.json'))
        with self.assertRaises(ValueError):load_reviewed_summary_exclusions(self.registry)


class ExistingNativeSummaryRankingExclusionsTest(unittest.TestCase):
    def test_best_generated_summary_changes_without_native_mutation(self):
        from test_hierarchical_summary_store import SummaryStorageTest, FakeEmbeddings
        from file_system_catalog.hierarchical_summary_store import summary_rows, append_summary_rows
        from file_system_catalog.summary_search_batched import search_summary_entries_batched

        db, kwargs = SummaryStorageTest().prepare()
        try:
            chunks, representations = summary_rows(db, **kwargs)
            append_summary_rows(db, chunk_rows=chunks, representation_rows=representations)
            query = FakeEmbeddings().embed([kwargs['tree'].root.summary])[0]
            before = search_summary_entries_batched(
                db, query, limit=3, nodes_per_document=5,
                expected_model=FakeEmbeddings.model,
            )
            self.assertEqual(1,len(before))
            self.assertGreater(len(before[0]['matched_summary_nodes']),1)
            top = before[0]['matched_summary_nodes'][0]
            excluded = frozenset({(
                top['original_path'], top['original_sha256'], top['derived_sha256'],
                top['native_representation_id'], top['summary_sha256'],
            )})
            deleted = [0]
            def filter_rows(rows):
                current, count = exclude_reviewed_summary_nodes(rows, excluded)
                deleted[0] += count
                return current
            state = db.total_changes
            after = search_summary_entries_batched(
                db, query, limit=3, nodes_per_document=5,
                expected_model=FakeEmbeddings.model,
                filter_source_nodes=filter_rows,
            )
            self.assertEqual(1,deleted[0])
            self.assertEqual(1,len(after))
            self.assertEqual(before[0]['original_path'], after[0]['original_path'])
            self.assertNotEqual(top['native_representation_id'],
                                after[0]['matched_summary_nodes'][0]['native_representation_id'])
            self.assertEqual(len(before[0]['matched_summary_nodes'])-1,
                             len(after[0]['matched_summary_nodes']))
            self.assertEqual(state,db.total_changes)
            stale_exclusion = frozenset({(
                top['original_path'],top['original_sha256'],'f'*64,
                top['native_representation_id'],top['summary_sha256'],
            )})
            untouched = search_summary_entries_batched(
                db,query,expected_model=FakeEmbeddings.model,
                filter_source_nodes=lambda rows:
                    exclude_reviewed_summary_nodes(rows, stale_exclusion)[0],
            )
            self.assertEqual(before[0]['best_summary'],untouched[0]['best_summary'])
        finally:
            db.close()


if __name__ == '__main__':unittest.main()
