"""No arbitrary or symlinked PDF corpus receipts can enter private retrieval."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest


def cli():
 p=Path(__file__).resolve().parents[2]/'scripts/raspi_private_strict_original_pdf_island_search.py'
 spec=importlib.util.spec_from_file_location('kmd_strict_private_original_island_cli',p)
 tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
 return tool

class StrictPrivateOriginalGlyphCliTests(unittest.TestCase):
 def setUp(self):
  t=TemporaryDirectory();self.addCleanup(t.cleanup)
  self.root=Path(t.name);self.private=self.root/'search-audit';self.private.mkdir()
  self.outside=self.root/'outside.json';self.outside.write_text('{"no":true}')
  self.valid=self.private/'approved-private.json';self.valid.write_text(json.dumps({'accepted':True}))
  self.mod=cli()
  patcher=patch.object(self.mod,'AUDITS',self.private)
  patcher.start();self.addCleanup(patcher.stop)
 def test_valid_bounded_private_read_and_no_rejection_of_ordinary_json(self):
  self.assertEqual({'accepted':True},self.mod.bounded_rollup(self.valid))
 def test_foreign_and_symlinked_private_rollups_fail_closed(self):
  link=self.private/'symlink.json';link.symlink_to(self.outside)
  for path in (self.outside,link,self.private/'missing.json',Path('relative.json')):
   with self.subTest(path=str(path)),self.assertRaisesRegex(ValueError,'untrusted'):
    self.mod.bounded_rollup(path)
 def test_oversized_private_unapproved_receipt_refused_before_loading(self):
  large=self.private/'oversized.json';large.write_bytes(b'x'*(512*1024+1))
  with self.assertRaisesRegex(ValueError,'oversized'):
   self.mod.bounded_rollup(large)

if __name__=='__main__':unittest.main()
