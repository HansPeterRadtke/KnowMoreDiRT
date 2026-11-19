"""Native summary character bound must be enforced by grammar and tokenizer.

The entire original source still enters the model prompt. We never shorten
original publisher text or silently truncate generated analyses after use.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from file_system_catalog.hierarchical_summary_model import LocalSummaryModel
from file_system_catalog.hierarchical_summary import (
    EmbeddingContextExceeded, SUMMARY_FACETS_COMPACT_SYSTEM,
    SUMMARY_FACETS_COMPACT_RETRY_SYSTEM,
    SUMMARY_FACETS_STRICT_SYSTEM,
)
from file_system_catalog.content_pipeline import FILE_ANALYSIS_SCHEMA


class LocalFacetLengthSchemaTests(unittest.TestCase):
    def model(self):
        engine=LocalSummaryModel("http://127.0.0.1:9999",model="test-local-model")
        class Client:
            seed=42
            base_url="http://127.0.0.1:9999"
            per_token_timeout_seconds=5
            def effective_model(self):return "test-local-model"
            def model_context(self):return SimpleNamespace(configured_tokens=8192)
            def _render_prompt(self,*,system,user):return system+"\n"+user
            def prompt_token_count(self,*,system,user):return 300
            def token_count(self,text):return max(1,(len(text)+3)//4)
        engine.client=Client()
        return engine

    def response(self,text):
        return {"content":json.dumps({"document_summary":text,"facets":[]}),
                "finish_reason":"stop","reasoning_content":None}

    def invoke(self,system,text,*,cached=None):
        output=[]
        engine=self.model()
        with patch("file_system_catalog.hierarchical_summary_model.read_model_call",
                   return_value=cached), patch(
            "file_system_catalog.hierarchical_summary_model.write_model_call",
            side_effect=lambda key,result:output.append(result),
        ), patch(
            "file_system_catalog.hierarchical_summary_model.stream_chat_completion_json",
            return_value=self.response(text),
        ) as generated:
            result=engine.generate_retrieval_analysis(
                system=system,user="SOURCE_BEGIN full original source SOURCE_END",
                output_tokens=900,
            )
        return result,output,generated.call_args.args[1]

    def test_original_native_schema_unchanged_and_decoder_enforces_compact_summary(self):
        baseline=copy.deepcopy(FILE_ANALYSIS_SCHEMA)
        for system,cap in (
            (SUMMARY_FACETS_COMPACT_SYSTEM,1200),
            (SUMMARY_FACETS_COMPACT_RETRY_SYSTEM,480),
            (SUMMARY_FACETS_STRICT_SYSTEM,480),
        ):
            with self.subTest(cap=cap,system=system[:45]):
                result,saved,payload=self.invoke(system,"Source-grounded concise summary.")
                summary=payload["response_format"]["json_schema"]["schema"]["properties"]["document_summary"]
                self.assertEqual(1,summary["minLength"])
                self.assertEqual(cap,summary["maxLength"])
                self.assertNotIn("pattern",summary)
                self.assertEqual("Source-grounded concise summary.",result["document_summary"])
                self.assertEqual(1,len(saved))
                self.assertEqual(baseline,FILE_ANALYSIS_SCHEMA)
                self.assertIn("SOURCE_BEGIN full original source SOURCE_END",
                              payload["messages"][1]["content"])

    def test_provider_ignoring_maxlength_fails_without_caching_or_truncating(self):
        e=self.model()
        with patch("file_system_catalog.hierarchical_summary_model.read_model_call",
                   return_value=None), patch(
            "file_system_catalog.hierarchical_summary_model.write_model_call",
        ) as save, patch(
            "file_system_catalog.hierarchical_summary_model.stream_chat_completion_json",
            return_value=self.response("L"*1500),
        ) as generated:
            with self.assertRaisesRegex(EmbeddingContextExceeded,"ignored"):
                e.generate_retrieval_analysis(
                    system=SUMMARY_FACETS_COMPACT_SYSTEM,user="SOURCE_BEGIN source data",
                    output_tokens=900,
                )
        self.assertEqual(1,generated.call_count)
        save.assert_not_called()

    def test_480_character_strict_limit_guard_works_even_if_tokens_under_400(self):
        e=self.model()
        with patch("file_system_catalog.hierarchical_summary_model.read_model_call",
                   return_value=None), patch(
            "file_system_catalog.hierarchical_summary_model.write_model_call",
        ) as save, patch(
            "file_system_catalog.hierarchical_summary_model.stream_chat_completion_json",
            return_value=self.response("S"*481),
        ):
            with self.assertRaisesRegex(EmbeddingContextExceeded,"ignored"):
                e.generate_retrieval_analysis(
                    system=SUMMARY_FACETS_COMPACT_RETRY_SYSTEM,user="SOURCE_BEGIN original",
                    output_tokens=900,
                )
        save.assert_not_called()

    def test_unsafe_old_cache_cannot_bypass_embedding_check(self):
        old={"native_analysis":{"document_summary":"A"*1800,"facets":[]}}
        with patch("file_system_catalog.hierarchical_summary_model.read_model_call",
                   return_value=old), patch(
            "file_system_catalog.hierarchical_summary_model.stream_chat_completion_json",
            side_effect=AssertionError("should reject without LLM call"),
        ) as generate:
            with self.assertRaisesRegex(EmbeddingContextExceeded,"cached"):
                self.model().generate_retrieval_analysis(
                    system=SUMMARY_FACETS_COMPACT_RETRY_SYSTEM,
                    user="SOURCE_BEGIN original",output_tokens=900,
                )
        generate.assert_not_called()

    def test_short_genuine_native_analysis_still_round_trips(self):
        text="Precise paper about typed video tokens and image sequences."
        result,records,payload=self.invoke(SUMMARY_FACETS_COMPACT_RETRY_SYSTEM,text)
        self.assertEqual(text,result["document_summary"])
        self.assertEqual(text,records[0]["native_analysis"]["document_summary"])
        self.assertEqual(480,payload["response_format"]["json_schema"]["schema"]["properties"]["document_summary"]["maxLength"])


if __name__=="__main__":
    unittest.main()
