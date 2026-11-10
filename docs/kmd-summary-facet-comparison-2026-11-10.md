# Summary and facet retrieval comparison on Raspi

On 10 November 2026, the default native summary search found the expected original in the first eight on 33 of 37 fixed questions. Allowing any indexed keyword, topic or keyphrase to supply the winning score reduced this to 30. Requiring support from a summary recovered the three lost targets but offered no measured gain over the default. Production ranking was left unchanged.

## Method

The private case file and scoring protocol were fixed before scoring. The set included three previously reported interface-design questions, 27 previously recorded Unity, Blender and n8n controls, six newly worded interface-design questions and a new NVIDIA domain control. The latter asks about the actual subject of the source that had displaced the menu-design article. All 36 distinct expected originals had native summary representations dated before the run. This set measures exact document discovery. A differently ranked original can also contain a useful answer.

All three methods used the same embeddings from Raspi's local multilingual E5 model and a single temporary SQLite backup of the canonical FileDB. The live connection was read-only and closed before the ranking comparisons. The snapshot included 4,137 summary representations and 98,956 topic, keyword or keyphrase representations. The native search validated stored vector bytes, representation text and provenance and applied a current-revision callback. After ranking, the benchmark verified the physical originals, extracted text and exact character spans for all 309 returned originals and 350 distinct contributing representations, including any facet used for the weighted score.

The candidate reranked the best 24 summary documents. For a matching original and extracted-text revision also present among the best 24 all-representation documents, it added one quarter of any positive difference between the all-representation cosine and the summary cosine. The remaining documents retained their summary score. Matching was at the document revision level. Two supporting hits can come from different sections. No model generation or parameter sweep was involved.

## Results

| Method | Expected original first | Expected original in first eight | Mean reciprocal rank at eight |
| --- | ---: | ---: | ---: |
| Native summaries | 21/37 | 33/37 | 0.6714 |
| Native summaries and unrestricted facets | 18/37 | 30/37 | 0.6047 |
| Facets with summary support | 21/37 | 33/37 | 0.6699 |

For the menu decision-time question, unrestricted facets ranked a PyTorch conversion skill first and lost Hick's Law from the first eight. The default and supported method both ranked Choice Overload first and Hick's Law seventh. All three ranked the same PyTorch skill first when the question actually asked how to convert a PyTorch training script for federated learning.

Relative to summaries alone, unrestricted facets reduced the expected-source rank on seven questions and improved none within the first eight. The supported method matched the default's observed target ranks on 36 questions. It moved the article about controls in a shared visual boundary from fifth to seventh. Four targets missed the first eight in all modes. They concerned Unity reflection probes, n8n field transformation, the Blender Bump node and Jakob's Law. An absent target at the cutoff does not establish its rank below that cutoff. These figures do not evaluate answer correctness or establish whole-library relevance.

## Reproduction

Run the following from this repository on Raspi with an available CPU and sufficient temporary disk space for a consistent native snapshot. Supply a private JSON case file and a new private results path. The tool refuses an existing results path. Use the same indexed revisions for exact replication. Production workers continue to add documents.

```sh
/data/var/knowmoredirt/semantic-venv/bin/python scripts/benchmarks/compare_summary_facets.py \
  --cases "$cases_path" --output "$new_result_path"
```

Each case needs a unique ID in `id`, a `group`, a `query` and an `expected_path` under the library root. Results retain the case-file digest, model revision, snapshot representation counts, ranks, contributing native identifiers and verified source-span hashes. The historical cases, protocol, full results and separate smoke evidence are in the private Raspi search-audit directory named `summary-facet-comparison-20261110`. No source text or embeddings accompany this benchmark in Git.

The test run passed 54 distinct KnowMoreDiRT tests, including six new comparison tests. An end-to-end check with the final CLI independently returned the intended technical source first in all three modes and verified all twelve returned originals on a later snapshot.
