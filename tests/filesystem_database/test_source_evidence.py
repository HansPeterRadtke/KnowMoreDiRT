"""Read-only source-evidence location, provenance and CLI audit regression."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase,main,mock

from file_system_catalog.source_evidence import source_evidence_candidates
from file_system_catalog.summary_search_audit import recent_search_audits, append_query_audit


def cli():
    path=Path(__file__).resolve().parents[2]/"scripts"/"raspi_hierarchical_library.py"
    spec=importlib.util.spec_from_file_location("kmd_source_evidence_cli_test",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OriginalEvidenceTests(TestCase):
    def setUp(self):
        self.path="reference/book.pdf"
        self.source=(
            "KMD provenance header\n"
            "[[SOURCE UNIT: PDF page 1]]\n"
            "A general introduction to studying interfaces. \n"
            "[[SOURCE UNIT: PDF page 2]]\n"
            "Graphical user interfaces contain windows, icons, menus and pointers. "
            "Their positioning and graphical design can change navigability.\n"
            "[[SOURCE UNIT: PDF page 3]]\n"
            "The inverse square law is used in some science demonstrations.\n"
        )
        self.derived=hashlib.sha256(self.source.encode()).hexdigest()
        self.original="a"*64
        self.args={
            "original_path":self.path,"original_title":"Human Computer Interaction",
            "filesystem_entry_id":9,"original_sha256":self.original,
            "derived_sha256":self.derived,"derived_text":self.source,
        }

    def test_retrieves_relevant_page_and_exact_original_spans(self):
        result=source_evidence_candidates(
            **self.args,query="graphical user interface windows icons menus",top=3
        )
        self.assertEqual(1,result["returned_passages"])
        record=result["passages"][0]
        self.assertEqual("PDF page 2",record["source_section"])
        self.assertEqual(9,record["filesystem_entry_id"])
        self.assertEqual(self.path,record["original_path"])
        self.assertEqual(self.derived,record["derived_sha256"])
        self.assertEqual(self.source[record["source_start_char"]:record["source_end_char"]],
                         self.source[self.source.index("[[SOURCE UNIT: PDF page 2]]"):
                                     self.source.index("[[SOURCE UNIT: PDF page 3]]")])
        self.assertEqual(hashlib.sha256(
            self.source[record["source_start_char"]:record["source_end_char"]].encode()
        ).hexdigest(),record["source_span_sha256"])
        self.assertEqual(hashlib.sha256(
            self.source[record["excerpt_start_char"]:record["excerpt_end_char"]].encode()
        ).hexdigest(),record["excerpt_sha256"])
        self.assertEqual(
            self.source[record["excerpt_start_char"]:record["excerpt_end_char"]],
            record["source_excerpt"]
        )
        self.assertIn("not_verified_answer",record["status"])

    def test_generated_metadata_header_cannot_be_returned_as_publisher_evidence(self):
        text=(
            "KMD source provenance (the original PDF/HTML/ZIP/XLSX, not a new publisher book):\n"
            '{"title":"Galactic comet zebras neutronium", "source_sha256":"test"}'
            "\n\n[[SOURCE UNIT: PDF page 1]]\n"
            "The original paper discusses graphical design of windows and icons.\n"
        )
        source=dict(self.args,
                    derived_text=text,
                    derived_sha256=hashlib.sha256(text.encode()).hexdigest())
        result=source_evidence_candidates(
            **source,query="galactic comet zebras neutronium",top=5
        )
        self.assertEqual(0,result["returned_passages"])
        visual=source_evidence_candidates(
            **source,query="graphical design windows icons",top=5
        )
        self.assertEqual(1,visual["returned_passages"])
        self.assertEqual("PDF page 1",visual["passages"][0]["source_section"])

    def test_exact_identifier_requires_the_entire_literal_in_the_source_body(self):
        identifier = "APP_SERVICE_URL"
        text = ("KMD source provenance example\n" + identifier + "\n"
                + "[[SOURCE UNIT: APP_SERVICE_URL]]\nOnly the section label contains it. The installation settings configure the service.\n"
                + "[[SOURCE UNIT: PDF page 2]]\nAPP SERVICE URL and APP_SERVICE_URL_BACKUP.\n"
                + "[[SOURCE UNIT: PDF page 3]]\n" + "Background. " * 75
                + "Set APP_SERVICE_URL before installation.\n")
        args = dict(self.args, original_title=identifier, derived_text=text,
                    derived_sha256=hashlib.sha256(text.encode()).hexdigest())
        for query in (identifier.lower(), "APP_SERVICE_URL installation",
                      "How do I configure APP_SERVICE_URL in the installation settings?"):
            result = source_evidence_candidates(**args, query=query, snippet_characters=160)
            self.assertEqual(result["returned_passages"], 1)
            row = result["passages"][0]
            self.assertEqual(row["source_section"], "PDF page 3")
            self.assertIn(identifier.lower(), row["matched_excerpt_terms"])
            self.assertIn(identifier, row["source_excerpt"])
            self.assertEqual(text[row["excerpt_start_char"]:row["excerpt_end_char"]], row["source_excerpt"])
        result = source_evidence_candidates(**args, query="APP_SERVICE_URL_EXTRA")
        self.assertEqual(result["returned_passages"], 0)

    def test_cropped_prefixed_identifier_is_never_credited_as_a_literal(self):
        from file_system_catalog.source_evidence import _best_excerpt
        text = "prefix_APP_SERVICE_URL " + ("installation " * 30) + "Suffix APP_SERVICE_URL."
        start, end, covered = _best_excerpt(text, ("app_service_url", "installation"), length=160)
        self.assertIn("app_service_url", covered)
        self.assertIn("Suffix APP_SERVICE_URL.", text[start:end])
        self.assertGreater(start, text.index("installation"))

    def test_identifier_survives_many_separated_word_decoys_before_the_match(self):
        text = "".join(f"[[SOURCE UNIT: page {i}]]\nAPP SERVICE URL.\n" for i in range(125))
        text += "[[SOURCE UNIT: actual page]]\nAPP_SERVICE_URL config.\n"
        args = dict(self.args, derived_text=text, derived_sha256=hashlib.sha256(text.encode()).hexdigest())
        result = source_evidence_candidates(**args, query="APP_SERVICE_URL", top=1)
        self.assertEqual([r["source_section"] for r in result["passages"]], ["actual page"])

    def test_no_lexical_hit_is_not_mislabeled_as_sufficient(self):
        result=source_evidence_candidates(
            **self.args,query="Galactic interstellar zebras neutronium bananas",
        )
        self.assertEqual(0,result["returned_passages"])
        self.assertIn("not proof",result["evidence_warning"])

    def test_corruption_and_excessive_text_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError,"digest"):
            source_evidence_candidates(
                **dict(self.args,derived_sha256="b"*64),
                query="graphical windows menus"
            )
        with self.assertRaisesRegex(ValueError,"bound"):
            source_evidence_candidates(
                **self.args,query="graphical windows menus",
                max_source_characters=10
            )
        with self.assertRaises(ValueError):
            source_evidence_candidates(
                **self.args,query="graphical windows menus",top=0
            )
        with self.assertRaises(ValueError):
            source_evidence_candidates(
                **self.args,query="and or the"
            )

    def test_long_original_section_keeps_exact_offsets_without_gaps(self):
        text="[[SOURCE UNIT: PDF page 7]]\n"+("Graphical windows menus icons. "*5000)
        args=dict(self.args,derived_text=text,
                  derived_sha256=hashlib.sha256(text.encode()).hexdigest())
        rows=source_evidence_candidates(
            **args,query="graphical windows icons menus",top=8,
            snippet_characters=450
        )
        self.assertGreater(rows["ephemeral_fts_passages"],1)
        self.assertTrue(rows["passages"])
        for row in rows["passages"]:
            self.assertTrue(row["source_start_char"]>=0)
            self.assertTrue(row["source_end_char"]<=len(text))
            self.assertEqual(
                hashlib.sha256(text[row["source_start_char"]:row["source_end_char"]].encode()).hexdigest(),
                row["source_span_sha256"],
            )

    def test_explicit_cli_query_and_evidence_are_automatically_logged(self):
        module=cli()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dbfile=root/"original-kmd.sqlite3"
            c=sqlite3.connect(dbfile)
            c.execute("CREATE TABLE filesystem_entries("
                      "id INTEGER PRIMARY KEY,entry_type TEXT,relative_path_display TEXT)")
            c.execute("INSERT INTO filesystem_entries VALUES(9,'file',?)",(self.path,))
            c.commit();c.close()
            logroot=root/"audits"
            source=dict(self.args, title=self.args["original_title"],
                        original_bytes=130)
            pdf = root / "book.pdf"
            pdf.write_bytes(b"fixture without readable physical pages")
            source["original_sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
            def audit(item):
                return append_query_audit(item,directory=logroot)
            with mock.patch.object(module,"DB",dbfile), \
                 mock.patch.object(module,"checked_original",return_value=source), \
                 mock.patch.object(module,"publisher_path",return_value=pdf), \
                 mock.patch.object(module,"append_query_audit",side_effect=audit), \
                 mock.patch("builtins.print") as printed:
                module.original_source_evidence(SimpleNamespace(
                    original_path=self.path,
                    query="graphical user interface windows icons menus",
                    top=5,snippet_characters=450,
                ))
            self.assertTrue(printed.called)
            saved=recent_search_audits(directory=logroot,limit=1)
            self.assertEqual(1,len(saved))
            self.assertEqual("verified_original_source_evidence",saved[0]["operation"])
            self.assertEqual(self.path,saved[0]["original_path"])
            self.assertEqual(1,saved[0]["count"])
            self.assertEqual(self.derived,saved[0]["derived_sha256"])
            self.assertEqual("PDF page 2",saved[0]["ranked_source_spans"][0]["source_section"])
            self.assertEqual(1, saved[0]["pdf_page_verification"]["unverified"])
            self.assertEqual("page_count_unavailable", saved[0]["ranked_source_spans"][0]["publisher_pdf_page"]["reason"])
            self.assertEqual(0o600,stat.S_IMODE(next(logroot.glob("*.jsonl")).stat().st_mode))


if __name__=="__main__":
    main()
