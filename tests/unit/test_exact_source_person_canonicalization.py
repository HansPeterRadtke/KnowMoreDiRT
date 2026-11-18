"""An exact source-backed person name need not invoke a formatting LLM.

Independent model verification stays mandatory in the caller.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import knowmoredirt.engine as module
from knowmoredirt.answer_types import ExpectedAnswer
from knowmoredirt.engine import KnowMoreDiRTEngine
from knowmoredirt.model_planner import ModelQueryTrace
from knowmoredirt.models import Evidence


def fake_engine():
    engine = object.__new__(KnowMoreDiRTEngine)
    engine._model_client = object()
    engine.model_query_trace = ModelQueryTrace(enabled=True, prompt_hashes=[], response_hashes=[])
    engine._evidence_payload = lambda evidence, limit: [
        {"rel_path":item.rel_path, "text":item.text}
        for item in evidence[:limit]
    ]
    return engine


class SourceExactPersonTests(unittest.TestCase):
    def test_complete_person_name_already_in_source_needs_no_formatting_request(self):
        engine = fake_engine()
        evidence = [Evidence("field_report.txt", "Mira Chen reviewed the Falcon Test Plan.")]
        with patch.object(module, "call_model_answer_canonicalization", side_effect=AssertionError("redundant LLM")):
            result = engine._canonicalize_model_answer_with_local_model(
                "Who reviewed the Falcon Test Plan?", "Mira Chen", ExpectedAnswer("person"), evidence
            )
        self.assertEqual(result, "Mira Chen")
        self.assertEqual(engine.model_query_trace.canonicalization_call_count, 0)

    def test_name_fragment_inside_longer_word_does_not_skip_formatting(self):
        engine = fake_engine()
        evidence = [Evidence("field_report.txt", "Mira Chenery reviewed it.")]
        with patch.object(module, "call_model_answer_canonicalization", return_value={"accepted":True,"answer":"Mira Chen"}) as m:
            engine._canonicalize_model_answer_with_local_model(
                "Who reviewed it?", "Mira Chen", ExpectedAnswer("person"), evidence
            )
        self.assertEqual(m.call_count, 1)

    def test_nonperson_still_calls_canonicalization(self):
        engine = fake_engine()
        evidence = [Evidence("field_report.txt", "Mira Chen reviewed the record.")]
        with patch.object(module, "call_model_answer_canonicalization", return_value={"accepted":True,"answer":"Mira Chen"}) as m:
            engine._canonicalize_model_answer_with_local_model(
                "Who reviewed it?", "Mira Chen", ExpectedAnswer("actor"), evidence
            )
        self.assertEqual(m.call_count, 1)

    def test_person_not_in_evidence_still_calls_canonicalization(self):
        engine = fake_engine()
        evidence = [Evidence("field_report.txt", "A different engineer reviewed the record.")]
        with patch.object(module, "call_model_answer_canonicalization", return_value={"accepted":True,"answer":"Mira Chen"}) as m:
            engine._canonicalize_model_answer_with_local_model(
                "Who reviewed it?", "Mira Chen", ExpectedAnswer("person"), evidence
            )
        self.assertEqual(m.call_count, 1)


if __name__ == "__main__":
    unittest.main()
