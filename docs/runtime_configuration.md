# Runtime configuration and logging

KnowMoreDiRT runtime settings are centralized in `knowmoredirt/default_config.xml` and validated by `kmd_runtime_config`. The default configuration is packaged with the Python distribution; production code no longer reads normal `KMD_*` settings directly from scattered `os.environ` calls.

## Precedence

Effective settings use this precedence:

1. An environment variable with the same setting name.
2. An optional user XML file selected by `KMD_CONFIG_FILE`.
3. The packaged `knowmoredirt/default_config.xml` value.

The configuration loader does not copy packaged defaults into `os.environ`. This matters for cache compatibility: settings that historically affected cache identity only when explicitly overridden still distinguish an explicit environment/user-XML override from the packaged default.

A user XML file contains only settings that differ from the packaged defaults. Unknown settings, duplicates, invalid types, and enforced range violations fail closed during configuration validation.

```xml
<?xml version="1.0" encoding="utf-8"?>
<knowmoredirt-config version="1">
  <settings>
    <setting name="KMD_VECTOR_MIN_SIMILARITY" value="0.61" />
    <setting name="KMD_LOG_LEVEL" value="DEBUG" />
  </settings>
</knowmoredirt-config>
```

Use it with `KMD_CONFIG_FILE=/path/to/kmd.xml`.

## Configuration metadata

Every deployable production setting in the packaged XML records a semantic group, risk level, expected change frequency, and description. Numeric settings also record units and, where a hard safety or semantic constraint exists, enforced minimum/maximum bounds. Enumerated settings record their allowed choices.

The configuration covers model identity and sampling, exact context budgeting, constrained decoding, transport timeouts/retries/backoff, scanner sizing, semantic ingestion/query/verification controls, vector retrieval and embeddings, filesystem memory limits, cache paths, evaluation, logging, and schema-capacity profiles. Pytest-only escape hatches are intentionally not deployable settings.

## Persistent logging

Persistent logging is enabled by default. Runtime state belongs below `/data/var`, so the default log path is `/data/var/knowmoredirt/logs/knowmoredirt.log`. If that path is not writable, KMD falls back to the current user's XDG state directory.

`KMD_LOG_LEVEL` controls detail. `KMD_LOG_MAX_BYTES` and `KMD_LOG_BACKUP_COUNT` control rotation. `KMD_LOG_STDERR=1` mirrors the same records to stderr. `KMD_LOG_ENABLED=0` disables KMD-managed persistent handlers. Initialization/ingestion progress, answer/evaluation phases, benchmark suite boundaries, and transient model retries are logged; tight per-token/per-record loops are not logged by default.

`KMD_PROGRESS` and `KMD_EVAL_PROGRESS` independently control interactive console progress. Turning console progress off does not disable the persistent log.

## Network retry policy

All local-model control calls have a configured timeout and transient retry policy. The default retryable statuses are 408, 425, 429, 500, 502, 503, and 504. Retry attempts, initial backoff, and exponential backoff multiplier are configurable.

Planner-owned semantic generation keeps its higher-level structured retry policy. Direct semantic callers that bypass the planner, such as document-context classification and semantic evaluation, use a separate transport-only retry wrapper. This avoids duplicate semantic retries while still recovering from temporary localhost disconnects.

Embedding and filesystem-catalog HTTP calls use configured request timeouts/retries as well. Retry/backoff safety settings are intentionally excluded from semantic cache fingerprints when they do not alter successful model output.

## Benchmark continuation and compatibility

The internal benchmark continues after incorrect answers by default. An incorrect answer is recorded in results/failure artifacts and subsequent questions continue. `--stop-on-failure` is an explicit debugging opt-in only.

Resume compatibility hashes answer-affecting source files, the packaged configuration XML/config/logging implementation, effective model settings plus their source, any user XML file/hash, suite inputs, and suite-specific environment overrides. A configuration change that can alter semantics therefore cannot silently reuse an incompatible completed benchmark result.

The model-cache fingerprint itself remains output-semantic: moving an unchanged default from Python into XML does not invalidate existing accepted model caches. This contract is tested against historical shared-cache fingerprints.


## KMD-wide model-call cache

All model-derived caches share one canonical root: `/data/var/knowmoredirt/model_cache`. This is intentionally independent of benchmark, suite, run directory, or input corpus. Cache namespaces such as `chunk_drs`, `query_drs`, `frame`, `verifier`, `document_context`, and `evaluation_judge` live beneath that root. Cache keys remain responsible for model, prompt/schema, transport settings, and input identity, so reuse is exact while storage is shared. `KMD_SHARED_MODEL_CACHE_ROOT` may relocate the entire cache. Per-cache directory settings remain supported only as explicit advanced/test overrides; benchmark runners must not invent benchmark-specific model-cache roots.
