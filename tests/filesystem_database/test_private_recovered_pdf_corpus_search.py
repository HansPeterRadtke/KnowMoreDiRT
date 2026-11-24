"""RAM-only original physical PDF fragment corpus; never a source quotation by itself."""
from __future__ import annotations
import hashlib
import unittest
from file_system_catalog.private_recovered_pdf_corpus_search import (
 candidate_source_page_bodies,PrivateRecoveredOriginalPageRAMIndex,
)

def sha(text):return hashlib.sha256(text.encode()).hexdigest()

class RecoveredOriginalPageRAMIndexTests(unittest.TestCase):
 def setUp(self):
  self.path='ai-agents/author-approved-paper.pdf'
  self.body=('Scientists studied document-level retrieval and source page identity '
             'but experimental claims require source verification. Native vector '
             'search and contextual evidence must account for original physical '
             'pages, scholarly citations and edition-specific source versions. '*3+'\n')
  self.preview=('KMD source provenance approved private publisher PDF\n'
             '[[SOURCE UNIT: PDF page 1]]\n'
             'An unmodified page about an entirely unrelated source topic.\n'
             '[[SOURCE UNIT: PDF page 2]]\n'+self.body+
             '[[SOURCE UNIT: PDF page 3]]\n'
             'Another untouched page with a caution that examples are not real facts.\n')
  self.changes=[{'original_physical_pdf_page':2,'after_provisional_text_sha256':sha(self.body),
            'source_page_text_is_not_visually_or_semantically_verified':True}]
  self.old='a'*64;self.new='b'*64;self.orig='c'*64

 def add(self,index,**kw):
  return index.add_approved_source_preview(original_path=kw.get('path',self.path),
     original_sha256=self.orig,private_preview_text=kw.get('preview',self.preview),
     physical_page_count=kw.get('pages',3),
     source_specific_page_change_receipts=kw.get('changes',self.changes),
     current_old_live_derived_sha256=self.old,
     approved_first_corrected_page_stage_sha256=self.new,
     unapproved_preview_sha256=kw.get('preview_sha',sha(kw.get('preview',self.preview))),
     native_original_file_id=kw.get('id',55))

 def test_source_page_boundaries_require_exact_hashes(self):
  original=candidate_source_page_bodies(self.preview,self.changes,3)
  self.assertEqual([(2,self.body,sha(self.body))],original)
  broken=[dict(self.changes[0],after_provisional_text_sha256='0'*64)]
  with self.assertRaisesRegex(ValueError,'changed/replaced/corrupted'):
   candidate_source_page_bodies(self.preview,broken,3)

 def test_only_private_recovered_page_is_indexed_no_header_or_unchanged_text(self):
  with PrivateRecoveredOriginalPageRAMIndex(max_originals=2) as i:
   self.add(i);i.commit()
   self.assertEqual(1,len(i.entries))
   self.assertEqual(1,len(i.seen))
   found=i.ranked('document level retrieval source page identity')
   self.assertEqual(1,len(found))
   row=found[0]
   self.assertEqual(self.path,row['original_path'])
   self.assertEqual(2,row['physical_pdf_page'])
   self.assertEqual(sha(self.body),row['source_body_sha256'])
   self.assertTrue(row['source_page_full_text_not_approved_as_publisher_quote'])
   self.assertEqual([],i.ranked('private publisher approved'))
   self.assertEqual([],i.ranked('untouched caution'))

 def test_no_glyph_page_heading_or_duplicate_original_can_be_injected(self):
  with PrivateRecoveredOriginalPageRAMIndex() as idx:
   self.add(idx)
   with self.assertRaisesRegex(ValueError,'duplicate'):
    self.add(idx)
   with self.assertRaises(ValueError):
    self.add(idx,path='../outside.pdf')

 def test_zero_eligible_machine_pages_remain_unavailable_not_fabricated(self):
  self.assertEqual([],candidate_source_page_bodies(self.preview,[],3))
  with PrivateRecoveredOriginalPageRAMIndex(max_originals=1) as idx:
   self.add(idx,changes=[]);idx.commit()
   self.assertIn(self.path,idx.seen)
   self.assertEqual(0,len(idx.entries))
   self.assertEqual([],idx.ranked('document level retrieval source page identity'))

 def test_pinned_private_page_text_for_independent_source_quote_validation(self):
  with PrivateRecoveredOriginalPageRAMIndex() as idx:
   self.add(idx);idx.commit()
   recovered=idx.provisional_page_for_independent_quotation_check(
      original_path=self.path,original_sha256=self.orig,physical_pdf_page=2,
      source_body_sha256=sha(self.body))
   self.assertEqual(self.body,recovered)
   self.assertEqual(1,len(idx.entries))
   for path,source_sha,page,body_sha in (
       (self.path,self.orig,1,sha(self.body)),
       ('science/another.pdf',self.orig,2,sha(self.body)),
       (self.path,'f'*64,2,sha(self.body)),
       (self.path,self.orig,2,'e'*64)):
    with self.subTest(path=path,page=page,source_sha=source_sha),self.assertRaises(ValueError):
     idx.provisional_page_for_independent_quotation_check(
        original_path=path,original_sha256=source_sha,physical_pdf_page=page,
        source_body_sha256=body_sha)
   with self.assertRaises(ValueError):
    idx.provisional_page_for_independent_quotation_check(
        original_path=self.path,original_sha256=self.orig,
        physical_pdf_page=True,source_body_sha256=sha(self.body))
  with self.assertRaises(ValueError):
   idx.provisional_page_for_independent_quotation_check(
      original_path=self.path,original_sha256=self.orig,
      physical_pdf_page=2,source_body_sha256=sha(self.body))

 def test_failed_machine_extraction_source_is_represented_with_zero_pages(self):
  # Some approved original PDFs have no permitted recovered machine pages.
  # They must remain in the 170-source inventory, never acquire invented
  # full text, and not cause the whole corpus research query to abort.
  clean=('KMD source provenance approved private publisher PDF\n'
         '[[SOURCE UNIT: PDF page 1]]\nSource that was never repaired.\n')
  with PrivateRecoveredOriginalPageRAMIndex(max_originals=1) as idx:
   self.add(idx,preview=clean,pages=1,changes=[])
   idx.commit()
   self.assertEqual(1,len(idx.seen))
   self.assertEqual(0,len(idx.entries))
   self.assertEqual([],idx.ranked('Source never repaired'))

 def test_original_source_scope_is_applied_before_limit_not_postfiltered(self):
  second='science/independently-approved.pdf'
  with PrivateRecoveredOriginalPageRAMIndex(max_originals=2) as idx:
   self.add(idx,path=self.path)
   self.add(idx,path=second)
   idx.commit()
   ranked=idx.ranked('document level retrieval source page identity',limit=1,original_path=second)
   self.assertEqual(1,len(ranked))
   self.assertEqual(second,ranked[0]['original_path'])
   with self.assertRaisesRegex(ValueError,'not present'):
    idx.ranked('document level retrieval',original_path='not-in-this-corpus.pdf')

 def test_wrong_private_full_review_preview_sha_refused(self):
  with PrivateRecoveredOriginalPageRAMIndex() as i:
   with self.assertRaisesRegex(ValueError,'preview changed'):
    self.add(i,preview_sha='d'*64)

 def test_recovered_page_body_cannot_contain_a_second_pdf_page_marker(self):
  damaged_body=self.body+'[[SOURCE UNIT: PDF page 77]]\nFake page\n'
  damaged=self.preview.replace(self.body,damaged_body)
  changes=[dict(self.changes[0],after_provisional_text_sha256=sha(damaged_body))]
  with PrivateRecoveredOriginalPageRAMIndex() as i:
   with self.assertRaises(ValueError):self.add(i,preview=damaged,changes=changes)

 def test_raw_unreadable_glyph_and_malformed_original_count_refused(self):
  damaged=self.preview.replace(self.body,'irrelevant\x00 source\n')
  changes=[dict(self.changes[0],after_provisional_text_sha256=sha('irrelevant\x00 source\n'))]
  with PrivateRecoveredOriginalPageRAMIndex() as i:
   with self.assertRaisesRegex(ValueError,'corrupted'):self.add(i,preview=damaged,changes=changes)
  with self.assertRaisesRegex(ValueError,'incomplete'):
   candidate_source_page_bodies(self.preview,self.changes,4)

 def test_porter_ranked_single_term_pages_do_not_hide_late_verified_two_term_page(self):
  # FTS5 Porter matches "run" against "running" but our exact excerpt
  # verifier rejects that spelling as an original query-term confirmation.
  # Twenty-seven short, falsely high-ranked pages must not consume limit=1.
  from file_system_catalog.hybrid_search import query_terms
  short=('run pipeline stable method publisher evidence exact same source '
         'physical page revision details\n')
  valid=('running pipeline '+('unrelated contextual paragraph about original '
         'source provenance and private PDF pages without empirical claims. '*130)+'\n')
  def source(page_bodies):
   preview='KMD source provenance approved private publisher PDF\n'
   changes=[]
   for page,body in enumerate(page_bodies,1):
    preview+=f'[[SOURCE UNIT: PDF page {page}]]\n'+body
    changes.append({'original_physical_pdf_page':page,
      'after_provisional_text_sha256':sha(body),
      'source_page_text_is_not_visually_or_semantically_verified':True})
   return preview,changes
  with PrivateRecoveredOriginalPageRAMIndex(max_originals=2) as idx:
   first,first_changes=source([short]*27)
   self.add(idx,path='ai-agents/many-weak-pages.pdf',preview=first,pages=27,changes=first_changes)
   second,second_changes=source([valid])
   self.add(idx,path='science/one-strong-page.pdf',preview=second,pages=1,changes=second_changes)
   idx.commit()
   self.assertEqual(28,len(idx.entries))
   accepted=idx.ranked('running pipeline',limit=1)
   self.assertEqual(1,len(accepted))
   self.assertEqual('science/one-strong-page.pdf',accepted[0]['original_path'])
   self.assertEqual(['pipeline','running'],accepted[0]['covered_distinct_query_terms'])
   self.assertTrue(accepted[0]['source_page_full_text_not_approved_as_publisher_quote'])
   self.assertEqual(1,len(idx.ranked('running pipeline',limit=1,original_path='science/one-strong-page.pdf')))
   self.assertEqual([],idx.ranked('running pipeline',limit=1,original_path='ai-agents/many-weak-pages.pdf'))
   self.assertEqual([],idx.ranked('fabricated nebula orthogonal',limit=1))

 def test_bad_query_and_closed_index_are_rejected(self):
  with PrivateRecoveredOriginalPageRAMIndex() as i:
   self.add(i);i.commit()
   for val in ('',10,'x'*500):
    with self.subTest(query=val),self.assertRaises(ValueError):i.ranked(val)
   with self.assertRaises(ValueError):i.ranked('document retrieval',limit=0)
  with self.assertRaises(ValueError):i.ranked('document retrieval')

 def test_one_generic_word_not_a_relevance_hit(self):
  with PrivateRecoveredOriginalPageRAMIndex() as i:
   self.add(i);i.commit()
   self.assertEqual([],i.ranked('document extraterrestrial nebula'))

if __name__=='__main__':unittest.main()
