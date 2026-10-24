# External FraCaS semantic inference corpus (not authored for KMD)

These **unaltered** CSV files are the fixed train/development split from the public GitHub repository `felipessalvatore/NLI_datasets`, Git revision `9551b739a830fafb2ec4b98d49a9ecd4bf3dcbd0`. Its root LICENSE is included verbatim under the MIT license. That repository adapts the FraCaS Consortium 1996 textual entailment corpus through Stanford's Bill MacCartney XML conversion. Do not call the project-generated twelve synthetic examples independent FraCaS ground truth. The `dev.csv` split (70 rows, 1 empty hypothesis) is external, pre-existing, and the **first baseline experiment must be frozen before inspecting its expected labels**. Train is used for model-parser integration smoke testing only; dev must not be used to tune prompts, thresholds, rules or selection.

Source: https://github.com/felipessalvatore/NLI_datasets/tree/9551b739a830fafb2ec4b98d49a9ecd4bf3dcbd0/nli_datasets/fracas
Original FraCaS: https://www-nlp.stanford.edu/~wcmac/downloads/

Upstream transform maps `yes` to entailment, `no` to contradiction, and `undef`/`unknown` to neutral. This is *natural-language inference* ground truth, not empirical truth and not an end-to-end KMD retrieval result. Distinct answer-label definitions and four missing original hypotheses must be retained.
