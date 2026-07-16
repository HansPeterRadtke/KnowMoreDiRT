# Storage architecture

KnowMoreDiRT treats the database as the persistent representation of discourse, not as a cache. Documents, chunks, source spans, DRS boxes, discourse referents, conditions, arguments, contexts, identity hypotheses, temporal records, and provenance remain normalized and queryable.

SQLite remains the supported local and reference backend because it is simple, transactional, inspectable, and sufficient for development and moderate corpora. File-backed stores use write-ahead logging, foreign-key checks, a busy timeout, and durable synchronization by default. In-memory stores remain optimized for tests.

A deployment containing very large document collections must not change the DRT model to accommodate storage scale. The physical backend should instead implement the same logical store contract on a server database with partitioning, concurrent ingestion, online backups, and operational monitoring. The first scaling boundary is now explicit through `StoreConfig`; unsupported backends fail clearly rather than silently falling back to SQLite.

The production migration sequence is to stabilize the logical schema and acceptance tests, define backend-neutral store operations, add a transactional server adapter, validate identical DRT accessibility and provenance behavior against both backends, and only then add partitioning and distributed ingestion. Raw text or model output must never become an alternate source of truth merely because the corpus grows.
