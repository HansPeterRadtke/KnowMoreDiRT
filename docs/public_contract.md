# Public Contract

KnowMoreDiRT exposes exactly two conceptual operations:

## `initialize(folder_path)`

Input is only a folder path. The folder may contain arbitrary nested subfolders, arbitrary filenames, arbitrary extensions or no extensions, and arbitrary readable text content. Every file is treated as raw text. The caller does not provide schemas, manifests, metadata headers, prepared corpora, benchmark wrappers, or conversion products.

## `question(text) -> string`

Input is only a question string. Output is only an answer string at the public boundary. Internal implementations may later retain diagnostics, evidence, traces, or graph records, but those are not part of the public outer contract.

## Current Phase

This repository phase is intentionally test-first. The public module contains a stub session that returns `unknown` for questions. The tests validate the fixture and interface contract; they do not claim real question-answering correctness yet.

