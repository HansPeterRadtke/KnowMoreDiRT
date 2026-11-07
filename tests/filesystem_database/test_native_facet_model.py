"""Use original KMD file-facet schema with a bounded Qwen generation call."""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from file_system_catalog.hierarchical_summary_model import LocalSummaryModel
from file_system_catalog.content_pipeline import normalize_analysis, flatten_representations


class ControlledNativeFacetTest(unittest.TestCase):
    def fake_model(self):
        m=LocalSummaryModel("http://127.0.0.1:9999",model="fake-llm")
        class FakeClient:
            seed=42
            base_url="http://127.0.0.1:9999"
            per_token_timeout_seconds=5
            def effective_model(self):return "fake-llm"
            def model_context(self):return SimpleNamespace(configured_tokens=8192)
            def _render_prompt(self,*,system,user):return system+" "+user
            def prompt_token_count(self,*,system,user):return 900
            def token_count(self,text):return max(1,(len(text)+3)//4)
        m.client=FakeClient()
        return m

    def test_one_json_model_call_normalized_to_original_native_representations(self):
        m=self.fake_model()
        response={
            "document_summary":"A book about graphical interfaces and accessible buttons.",
            "facets":[{
                "facet_name":"interface layout",
                "facet_strength":"essential",
                "representations":[{
                    "kind":"keyword","item_strength":"strong","text":"buttons"
                },{
                    "kind":"keyphrase","item_strength":"strong","text":"graphical interface layout"
                }],
            }],
        }
        cache=[]
        with mock.patch(
            "file_system_catalog.hierarchical_summary_model.read_model_call",
            return_value=None,
        ), mock.patch(
            "file_system_catalog.hierarchical_summary_model.write_model_call",
            side_effect=lambda key,value:cache.append((key,value)),
        ), mock.patch(
            "file_system_catalog.hierarchical_summary_model.stream_chat_completion_json",
            return_value={"content":json.dumps(response),"finish_reason":"stop",
                          "reasoning_content":None},
        ) as generated:
            normalized=m.generate_retrieval_analysis(
                system="Summarize source and extract concepts",
                user="SOURCE_BEGIN example SOURCE_END",
                output_tokens=1500,
            )
        kinds=[v["kind"] for v in flatten_representations(normalized)]
        self.assertEqual(["summary","topic","keyword","keyphrase"],kinds)
        self.assertEqual("interface layout",normalized["facets"][0]["label"])
        self.assertEqual(1,generated.call_count)
        self.assertEqual(1500,generated.call_args.args[1]["max_tokens"])
        self.assertEqual("file_facet_analysis",
                         generated.call_args.args[1]["response_format"]["json_schema"]["name"])
        self.assertEqual(1,len(cache))
        self.assertIn("native_analysis",cache[0][1])

    def test_overlong_generation_rejected_and_not_cached(self):
        m=self.fake_model()
        with mock.patch(
            "file_system_catalog.hierarchical_summary_model.read_model_call",return_value=None,
        ), mock.patch(
            "file_system_catalog.hierarchical_summary_model.write_model_call",
        ) as cache, mock.patch(
            "file_system_catalog.hierarchical_summary_model.stream_chat_completion_json",
            return_value={"content":"{}","finish_reason":"length",
                          "reasoning_content":None}
        ):
            with self.assertRaisesRegex(RuntimeError,"did not finish"):
                m.generate_retrieval_analysis(
                    system="source",user="SOURCE_BEGIN A SOURCE_END",output_tokens=1500,
                )
        cache.assert_not_called()

    def test_context_cap_reserves_source_and_output_space(self):
        m=self.fake_model()
        m._context_cap_tokens=1200
        self.assertEqual(1200,m.context_tokens)
        with self.assertRaisesRegex(RuntimeError,"reserved context"):
            m.generate_retrieval_analysis(
                system="source",user="source",output_tokens=400,
            )


if __name__=="__main__":
    unittest.main()
