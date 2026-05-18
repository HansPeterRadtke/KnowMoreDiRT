# Public Contract

KnowMoreDiRT exposes exactly two intended public module functions:

## `initialize(folder_path)`

Input is only a folder path. The folder may contain arbitrary nested subfolders, arbitrary filenames, arbitrary extensions or no extensions, and arbitrary readable text content. Every file is treated as raw text. The caller does not provide schemas, manifests, metadata headers, prepared corpora, benchmark wrappers, or conversion products.

`initialize` scans the tree, reads text files, records natural filesystem metadata, chunks text, and builds the internal index/knowledge state.

## `question(text) -> string`

Input is only a question string. Output is only an answer string at the public boundary. Internal implementations may retain diagnostics, evidence, traces, or graph records, but those are not part of the public outer API.

## Current Phase

The first real implementation is now present. It is deterministic and raw-text-only, with scanner, index, extraction, and answer logic. It remains an early implementation, not the final DRT reasoning system.
