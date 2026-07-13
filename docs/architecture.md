# Architecture

The deterministic boundary is limited to raw-file scanning, JSON and table shape discovery, generic record tools, provenance, schema validation, caching, and formatting. Natural-language meaning, source-field selection, relation selection, scope, temporal interpretation, aggregation intent, and answer construction belong to the model.

The query flow is: source catalog, strict model query program, structural validation, generic execution, strict grounded answer. There is no deterministic semantic fallback.
