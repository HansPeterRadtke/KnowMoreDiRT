# Per-node progress receipts for recursive KnowMoreDiRT summaries

The Raspberry Pi indexing worker processes one source at a time using the local Qwen model. Longer sources can require many sequential leaf summaries followed by summaries of summaries. Previously the worker progress file only updated when a full source tree was complete, so a large book could appear stuck for minutes or hours while the model was genuinely working.

The new HierarchicalSummarizer on_node callback is invoked after each generated summary node has passed token-limit, source span, node-ID and summary-SHA validation. The CLI writes an atomic private progress receipt under the existing KnowMoreDiRT summaries/progress directory, without including publisher source text or model-generated summary bodies. Each receipt includes the current original and derived SHA-256 identities, generated-node count, leaf-group count, leaf-group percentage, last node hierarchy level, source offsets, summary digest and elapsed time.

Only the first-level source groups have a known denominator. The later parent summarization count depends on the generated lower-level summaries, so the receipt deliberately does not invent a whole-book percentage. On success the source is marked complete only after the complete tree file is durably saved. An interrupted generation is explicitly marked incomplete and cannot be counted as an indexed original. A conflicting pre-existing complete source tree is left unchanged.

The existing Raspi summary status command now checks the active worker original and source-map revision, and can expose its current per-node receipt if it matches the active source and run timestamp. The original native KMD FileDB remains authoritative for actual completed documents and all vectors.

A disposable synthetic 35-section source produced five first-level summaries and one parent. Tests verified leaf progress at 20, 40, 60, 80 and 100 percent, final root linkage and digest, 0600 private file permissions, 0700 private directory permissions, atomic file replacement and explicit incomplete status after an injected model failure. No extra model server or SQLite database is created.

The balanced live worker already completed one 60,842-character research document as a single summary. It subsequently started a substantially larger research PDF to test multi-level summarization. The current in-flight model call is deliberately not interrupted to install this observational feature; new builds pick it up after code integration.
