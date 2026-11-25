"""Source-scoped assessment is a candidate model label, never truth proof."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import urllib.error

from file_system_catalog.source_claim_assessment import (
    validate_dataset, model_input, validate_model_result, score_results,
    pairwise_model_input, validate_pairwise_result, aggregate_pairwise,
)


def source(text,**extra):
    return {"kind":"synthetic_fixture","title":"Test evidence",
            "text":text,"sha256":hashlib.sha256(text.encode()).hexdigest(),**extra}


def dataset():
    return {"schema":"kmd.source_claim_assessment_frozen_dataset.v1",
            "frozen_before_model_inference":True,"source_text_is_private":True,
            "sources":{"a":source("Revision 2 does not support protocol X."),
                       "b":source("Revision 1 supports protocol X.")},
            "cases":[
                {"id":"current","claim":"Revision 2 supports protocol X.",
                 "evidence_ids":["b","a"],"expected_label":"contradicts",
                 "expected_evidence_ids":["a"],"group":"synthetic_scope",
                 "reason":"PRIVATE GOLD REASON"},
                {"id":"old","claim":"Revision 1 supports protocol X.",
                 "evidence_ids":["a","b"],"expected_label":"supports",
                 "expected_evidence_ids":["b"],"group":"synthetic_scope"},
                {"id":"unknown","claim":"Revision 2 supports protocol Y.",
                 "evidence_ids":["b","a"],"expected_label":"insufficient",
                 "expected_evidence_ids":[],"group":"synthetic_scope"},
            ]}


def load_runner():
    path=Path(__file__).resolve().parents[2]/"scripts/benchmark_source_claim_assessment.py"
    spec=importlib.util.spec_from_file_location("test_only_claim_benchmark",path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


class FrozenClaimEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.data=dataset(); self.case=self.data["cases"][0]

    def test_dataset_does_not_leak_gold_in_model_messages(self):
        validate_dataset(self.data)
        system,user,schema=model_input(self.case,self.data["sources"],"scope")
        self.assertIn("superseded",system)
        self.assertNotIn("expected_label",user)
        self.assertNotIn("PRIVATE GOLD REASON",user)
        self.assertNotIn("expected_evidence_ids",user)
        body=json.loads(user)
        self.assertEqual(["E1","E2"],[x["id"] for x in body["evidence"]])
        self.assertEqual(self.data["sources"]["a"]["text"],body["evidence"][1]["text"])
        self.assertEqual(["E1","E2"],schema["properties"]["evidence_ids"]["items"]["enum"])

    def test_original_provenance_metadata_is_required_but_not_new_byte_attestation(self):
        text="Published method excludes the untested configuration."
        self.data["sources"]["a"]=source(text,kind="publisher_pdf_reading_order_span",
             source_path="approved/paper.pdf",source_url="https://example.invalid/original",
             original_sha256="a"*64,derived_sha256="b"*64,page_text_sha256="c"*64,
             physical_page=2,start_char=10,end_char=10+len(text))
        validate_dataset(self.data)
        for key,bad in (("original_sha256","x"),("derived_sha256",None),
                        ("source_path","../a.pdf"),("physical_page",True),
                        ("end_char",10+len(text)+1),("page_text_sha256","x")):
            with self.subTest(key=key):
                x=copy.deepcopy(self.data);x["sources"]["a"][key]=bad
                with self.assertRaises(ValueError):validate_dataset(x)

    def test_changed_source_bytes_and_duplicate_cases_fail(self):
        self.data["sources"]["a"]["text"]="changed"
        with self.assertRaisesRegex(ValueError,"digest"):validate_dataset(self.data)
        d=dataset();d["cases"].append(copy.deepcopy(d["cases"][0]))
        with self.assertRaisesRegex(ValueError,"duplicate"):validate_dataset(d)

    def test_gold_must_be_consistent_and_exact_source_bound(self):
        for patch in ({"expected_evidence_ids":["unknown"]},
                      {"expected_evidence_ids":[]},
                      {"expected_evidence_ids":["a","a"]},
                      {"expected_label":"insufficient"},
                      {"expected_label":"conflict"},
                      {"evidence_ids":["a","a"]}):
            with self.subTest(patch=patch):
                d=dataset();d["cases"][0].update(patch)
                with self.assertRaises(ValueError):validate_dataset(d)

    def test_exact_case_local_ids_map_back_to_supplied_source(self):
        got=validate_model_result(self.case,'{"label":"contradicts","evidence_ids":["E2"]}')
        self.assertEqual({"label":"contradicts","evidence_ids":["a"]},got)
        self.assertEqual({"label":"insufficient","evidence_ids":[]},
                         validate_model_result(self.case,'{"label":"insufficient","evidence_ids":[]}'))

    def test_model_cannot_fabricate_evidence_or_promote_unbound_answer(self):
        for raw in ('{"label":"supports","evidence_ids":["E3"]}',
                    '{"label":"supports","evidence_ids":[]}',
                    '{"label":"insufficient","evidence_ids":["E1"]}',
                    '{"label":"conflict","evidence_ids":["E1"]}',
                    '{"label":"conflict","evidence_ids":["E1","E1"]}',
                    '{"label":"supports","evidence_ids":["a"]}',
                    '{"label":"supports","label":"contradicts","evidence_ids":["E2"]}',
                    '{"label":"true","evidence_ids":["E1"]}',
                    '{"label":"supports","evidence_ids":["E1"],"answer":"yes"}',
                    '{"label":"supports","evidence_ids":[1]}',
                    'not JSON'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):validate_model_result(self.case,raw)

    def test_conflict_requires_two_distinct_bound_sources(self):
        got=validate_model_result(self.case,'{"label":"conflict","evidence_ids":["E2","E1"]}')
        self.assertEqual(["a","b"],got["evidence_ids"])

    def test_pairwise_request_contains_only_one_permitted_passage_and_no_gold(self):
        system,user,schema=pairwise_model_input(self.case,self.data["sources"],"a")
        self.assertIn("unknown, not no",system)
        d=json.loads(user)
        self.assertEqual(d["source"]["text"],self.data["sources"]["a"]["text"])
        self.assertNotIn(self.data["sources"]["b"]["text"],user)
        self.assertNotIn("PRIVATE GOLD REASON",user)
        self.assertNotIn("expected_label",user)
        self.assertEqual(schema["properties"]["relation"]["enum"],["yes","no","unknown"])
        with self.assertRaises(ValueError):
            pairwise_model_input(self.case,self.data["sources"],"outside")

    def test_pairwise_results_are_strict(self):
        for label in ("yes","no","unknown"):
            self.assertEqual(label,validate_pairwise_result(json.dumps({"relation":label})))
        for raw in ('{"relation":"no","relation":"yes"}','{"relation":true}',
                    '{"relation":"conflict"}','{"relation":"unknown","extra":1}',
                    'unknown','{}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):validate_pairwise_result(raw)

    def test_pairwise_aggregation_cannot_turn_missing_source_into_unknown(self):
        for rel,expected in [
            ({"a":"yes","b":"unknown"},{"label":"supports","evidence_ids":["a"]}),
            ({"a":"unknown","b":"no"},{"label":"contradicts","evidence_ids":["b"]}),
            ({"a":"yes","b":"no"},{"label":"conflict","evidence_ids":["b","a"]}),
            ({"a":"unknown","b":"unknown"},{"label":"insufficient","evidence_ids":[]})]:
            self.assertEqual(expected,aggregate_pairwise(self.case,rel))
        for rel in ({"a":"unknown"},{"a":"yes","b":None},
                    {"a":"yes","b":"unknown","c":"unknown"}):
            with self.subTest(rel=rel):
                with self.assertRaises(ValueError):aggregate_pairwise(self.case,rel)

    def test_accuracy_distinguishes_wrong_source_invalid_and_unsupported_decision(self):
        rows=[{"id":"current","result":{"label":"contradicts","evidence_ids":["b"]}},
              {"id":"old","result":None},
              {"id":"unknown","result":{"label":"supports","evidence_ids":["a"]}}]
        s=score_results(self.data["cases"],rows)["overall"]
        self.assertEqual(3,s["cases"]);self.assertEqual(1,s["correct_labels"])
        self.assertEqual(0,s["correct_label_and_exact_decisive_evidence_ids"])
        self.assertEqual(1,s["invalid_model_outputs"])
        self.assertEqual(1,s["unsupported_decisive_labels_on_insufficient_gold"])
        self.assertEqual(0,s["opposite_polarity_errors"])
        self.assertEqual(1,s["confusion"]["supports->invalid"])

    def test_incomplete_or_duplicate_scores_are_not_complete_evaluation(self):
        with self.assertRaises(ValueError):score_results(self.data["cases"],[])
        with self.assertRaises(ValueError):score_results(self.data["cases"],[{"id":"current"}]*3)


class BenchmarkBoundaryTests(unittest.TestCase):
    def setUp(self):self.runner=load_runner()

    def test_model_endpoint_only_loopback_and_exact_model_id(self):
        for invalid in (True,0,80,65536,"8080"):
            with self.subTest(port=invalid):
                with self.assertRaises(ValueError):self.runner.LocalModel(invalid,"local")
        c=self.runner.LocalModel(8080,"local")
        self.assertEqual("http://127.0.0.1:8080",c.base)
        with mock.patch.object(c,"_json",return_value={"data":[{"id":"different"}]}):
            with self.assertRaisesRegex(ValueError,"exactly"):c.verify_model()
        with mock.patch.object(c,"_json",return_value={"data":[{"id":"local"}]}):
            c.verify_model()

    def test_structured_output_requires_completed_stop_and_actual_source_ids(self):
        c=self.runner.LocalModel(8080,"local");d=dataset()
        payload={"choices":[{"finish_reason":"stop","message":{"content":'{"label":"contradicts","evidence_ids":["E2"]}'}}]}
        with mock.patch.object(c,"_json",return_value=payload) as request:
            r=c.judge(d["cases"][0],d["sources"],"scope",42)
        self.assertEqual("contradicts",r["result"]["label"])
        sent=request.call_args.args[1];self.assertFalse(sent["stream"])
        self.assertEqual(80,sent["max_tokens"])
        self.assertTrue(sent["response_format"]["json_schema"]["strict"])
        for reason in ("length","error"):
            changed=copy.deepcopy(payload);changed["choices"][0]["finish_reason"]=reason
            with mock.patch.object(c,"_json",return_value=changed):
                self.assertIsNone(c.judge(d["cases"][0],d["sources"],"scope",42)["result"])

    def test_pairwise_local_calls_aggregate_only_complete_valid_results(self):
        c=self.runner.LocalModel(8080,"local");d=dataset();case=d["cases"][0]
        def reply(label):
            return {"choices":[{"finish_reason":"stop","message":{
                "content":json.dumps({"relation":label})}}]}
        with mock.patch.object(c,"_json",side_effect=[reply("unknown"),reply("no")]) as req:
            r=c.judge(case,d["sources"],"pairwise",42)
        self.assertEqual(2,req.call_count)
        self.assertEqual({"label":"contradicts","evidence_ids":["a"]},r["result"])
        self.assertEqual(["b","a"],[x["evidence_id"] for x in r["source_local_model_calls"]])
        self.assertEqual("no",r["source_local_model_calls"][1]["relation"])
        with mock.patch.object(c,"_json",side_effect=[reply("unknown"),reply("unsupported")]):
            self.assertIsNone(c.judge(case,d["sources"],"pairwise",42)["result"])

    def test_partial_transport_refusal_not_replayed_or_promoted(self):
        d=dataset();client=mock.Mock();gold={"label":"contradicts","evidence_ids":["a"]}
        client.judge.side_effect=[{"result":gold,"elapsed_seconds":0.1},
                                  {"result":gold,"elapsed_seconds":0.1},
                                  urllib.error.URLError("unavailable")]
        with TemporaryDirectory() as directory:
            with (Path(directory)/"private.jsonl").open("w") as f:
                r=self.runner.run_benchmark(d,client,variants=["baseline","scope"],
                    seed=42,checkpoint=f)
            self.assertEqual(2,len((Path(directory)/"private.jsonl").read_text().splitlines()))
        self.assertFalse(r["complete"]);self.assertEqual(1,r["paired_cases_completed"])
        self.assertEqual(3,client.judge.call_count)
        self.assertIn("interruption",r)
        self.assertEqual(1,r["scores_on_same_paired_cases"]["scope"]["overall"]["correct_labels"])

    def test_private_output_exclusive_creation(self):
        with TemporaryDirectory() as directory:
            out=Path(directory)/"receipt.json"
            with self.runner._new_private(out) as f:f.write("{}")
            self.assertEqual(0o600,out.stat().st_mode&0o777)
            with self.assertRaises(ValueError):self.runner._new_private(out)
            link=Path(directory)/"linked.json";link.symlink_to(out)
            with self.assertRaises(ValueError):self.runner._read_private(link)


if __name__=="__main__":unittest.main()
