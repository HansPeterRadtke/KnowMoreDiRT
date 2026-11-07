# Per-node progress receipts for recursive KnowMoreDiRT summaries

The Raspberry Pi indexing worker processes one source at a time using the local Qwen model. Longer sources can require many sequential leaf summaries followed by summaries of summaries. Previously the worker progress file only updated when a full source tree was complete, so a large book could appear stuck for minutes or hours while the model was genuinely working.

The new HierarchicalSummarizer on_node callback is invoked after each generated summary node has passed token-limit, source span, node-ID and summary-SHA validation. The CLI writes an atomic private progress receipt under the existing KnowMoreDiRT summaries/progress directory, without including publisher source text or model-generated summary bodies. Each receipt includes the current original and derived SHA-256 identities, generated-node count, leaf-group count, leaf-group percentage, last node hierarchy level, source offsets, summary digest and elapsed time.

Only the first-level source groups have a known denominator. The later parent summarization count depends on the generated lower-level summaries, so the receipt deliberately does not invent a whole-book percentage. On success the source is marked complete only after the complete tree file is durably saved. An interrupted generation is explicitly marked incomplete and cannot be counted as an indexed original. A conflicting pre-existing complete source tree is left unchanged.

The existing Raspi summary status command now checks the active worker original and source-map revision, and can expose its current per-node receipt if it matches the active source and run timestamp. The original native KMD FileDB remains authoritative for actual completed documents and all vectors.

A disposable synthetic 35-section source produced five first-level summaries and one parent. Tests verified leaf progress at 20, 40, 60, 80 and 100 percent, final root linkage and digest, 0600 private file permissions, 0700 private directory permissions, atomic file replacement and explicit incomplete status after an injected model failure. No extra model server or SQLite database is created.

The balanced live worker already completed one 60,842-character research document as a single summary. It subsequently started a substantially larger research PDF to test multi-level summarization. The current in-flight model call is deliberately not interrupted to install this observational feature; new builds pick it up after code integration.

## Live two-document multilevel verification

After the balanced queue was enabled, the continuous Raspi worker completed
two original documents requiring genuine hierarchical reduction.

The original PACER causal-discovery paper contained 200,796 verified extracted
characters. Its canonical KMD rows contain three contiguous source-level
summaries covering ranges 0–71,872, 71,872–144,364, and 144,364–200,796,
plus one full-source parent summary linked to exactly those three children.
All four SHA-linked E5 vectors passed independent original binary, derived
text, node source-span, summary and serialized vector checksum checks.
A live natural-language search for PACER and large-scale interventional
causal discovery returned that original in first place with cosine 0.9236.

The original RFC 9866 root-node failure detector text contained 60,749
verified derived characters. Despite its smaller character count, the
tokenizer required three first-level source summaries plus a root summary.
The source spans were 0–738, 738–59,552 and 59,552–60,749. All four
native summary vectors, source hashes and root child references were
independently verified. A search for Root Node Failure Detection and RPL
border-router crashes returned the RFC original first at cosine 0.9211.

The audit receipt is privately preserved at the existing Raspi
KnowMoreDiRT search-audit/live-multilevel-retrieval-acceptance-20261107.json.
Neither original document's copyrighted content nor the vector database
is committed to Git. SQLite quick check returned OK and foreign keys
had no errors.

After the per-node progress code was integrated, live builds produced
owner-only progress files under summaries/progress. The KMD status
command exposed the current original's source revision, leaf-group
count, generation status and last node checkpoint. A successfully
completed source was marked complete_tree_saved; the following
source was shown as summarizing, not incorrectly reported finished.
The active worker remained up without an unexpected restart.
