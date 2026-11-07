"""Single-slot voice-gated native KMD model selection, without network access."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from unittest import TestCase, mock
from types import SimpleNamespace

from file_system_catalog.hierarchical_summary_model import LocalSummaryModel
from file_system_catalog import hierarchical_summary as hierarchy


class Clock:
    def __init__(self):
        self.time=0.0
        self.nwaits=0
    def monotonic(self):
        return self.time
    def sleep(self,seconds):
        self.time+=seconds
        self.nwaits+=1


class Response:
    def __init__(self, processing):
        self.content=json.dumps([{"id":0,"is_processing":processing}]).encode()
    def __enter__(self):
        return self
    def __exit__(self,*args):
        return False
    def read(self,n):
        return self.content[:n]


class SharedVoiceSlotAdmissionTests(TestCase):
    def test_when_disabled_no_slot_probes(self):
        client=LocalSummaryModel(
            "http://10.8.0.7:14833",model="coder",shared_slot_idle_seconds=0
        )
        with mock.patch(
            "file_system_catalog.hierarchical_summary_model.urllib.request.urlopen",
            side_effect=AssertionError("disabled admission must not access slots"),
        ):
            client._wait_for_shared_slot()

    def test_stays_away_from_busy_voice_calls_and_yields_idle_interval(self):
        client=LocalSummaryModel(
            "http://10.8.0.7:14833",model="coder",
            shared_slot_idle_seconds=8,admission_max_wait_seconds=45,
        )
        clock=Clock()
        requests=[]
        def get(url,timeout):
            requests.append(url)
            return Response(processing=len(requests)<=3)
        with mock.patch(
            "file_system_catalog.hierarchical_summary_model.urllib.request.urlopen",
            side_effect=get,
        ), mock.patch(
            "file_system_catalog.hierarchical_summary_model.time.monotonic",
            side_effect=clock.monotonic,
        ), mock.patch(
            "file_system_catalog.hierarchical_summary_model.time.sleep",
            side_effect=clock.sleep,
        ):
            client._wait_for_shared_slot()
        self.assertGreaterEqual(clock.time,14)
        self.assertTrue(all(p=="http://10.8.0.7:14833/slots" for p in requests))
        self.assertGreaterEqual(len(requests),8)

    def test_invalid_slot_status_fails_closed_after_bounded_delay(self):
        client=LocalSummaryModel(
            "http://10.8.0.7:14833",model="coder",
            shared_slot_idle_seconds=2,admission_max_wait_seconds=4,
        )
        clock=Clock()
        class BadResponse(Response):
            def __init__(self):
                self.content=b'{"untrusted":"not a slot list"}'
        with mock.patch(
            "file_system_catalog.hierarchical_summary_model.urllib.request.urlopen",
            side_effect=lambda *args,**kwargs:BadResponse(),
        ), mock.patch(
            "file_system_catalog.hierarchical_summary_model.time.monotonic",
            side_effect=clock.monotonic,
        ), mock.patch(
            "file_system_catalog.hierarchical_summary_model.time.sleep",
            side_effect=clock.sleep,
        ):
            with self.assertRaisesRegex(RuntimeError,"shared-slot status is unavailable"):
                client._wait_for_shared_slot()

    def test_model_context_is_detected_and_cap_is_a_separate_quality_choice(self):
        client=LocalSummaryModel(
            "http://10.8.0.7:14833",model="coder",
            context_cap_tokens=16384
        )
        client.client.model_context=lambda:SimpleNamespace(configured_tokens=262144)
        self.assertEqual(16384,client.context_tokens)
        client._context_cap_tokens=None
        self.assertEqual(262144,client.context_tokens)

    def test_compact_source_prompt_remains_original_kmd_facets(self):
        from test_hierarchical_native_facets import FakeNativeFacetModel
        model=FakeNativeFacetModel(3000)
        engine=hierarchy.HierarchicalSummarizer(
            model,output_tokens=700,safety_tokens=100,
            native_facets=True,native_facets_compact=True,
        )
        tree=engine.summarize(
            title="A small source",
            text="[[SOURCE UNIT: PDF page 1]]\nSource methods include named approach A and B.",
        )
        self.assertEqual("native_facets_compact",tree.root.prompt_policy)
        self.assertIn("named methods",model.calls[0][0])
        self.assertIn("native KMD",model.calls[0][0])

    def test_native_retrieval_prompt_keeps_concrete_details_and_numeric_values(self):
        self.assertIn("named method",hierarchy.SUMMARY_FACETS_SYSTEM)
        self.assertIn("actual value",hierarchy.SUMMARY_FACETS_SYSTEM)
        self.assertIn("native keyword",hierarchy.SUMMARY_FACETS_SYSTEM)


if __name__=="__main__":
    import unittest
    unittest.main()
