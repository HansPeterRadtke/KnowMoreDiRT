# Conversational Boundary

KnowMoreDiRT is a technical retrieval and reasoning service. A conversational agent belongs above it.

## KnowMoreDiRT responsibilities

KMD accepts a question, retrieves bounded source evidence, executes grounded DRT/DSPG reasoning, detects conflicts or insufficient support, and returns a technical answer state with provenance and diagnostics internally available.

Its decisions should remain stable regardless of whether the user is polite, hostile, vague, childish, malformed, profane, or speaking about nonsense.

## Agent responsibilities

The agent layer handles dialogue, clarification, tone, markdown, explanatory depth, formatting, and user-facing descriptions of uncertainty. It may transform the verified technical answer into natural language, but it may not manufacture evidence or silently replace `unknown` with a guess.

## Nonsense and malformed input

The retrieval layer should treat nonsense like any other literal source material. If a nonsense token or gibberish passage exists in the indexed folder, it remains retrievable. If it does not exist and no grounded semantic relation can be established, the system should report that nothing supported was found.

The conversational benchmark should therefore include database-backed gibberish, absent gibberish, children's writing, fantasy, word salad, malformed grammar, profanity, hostile phrasing, ambiguous names, vague requests, and serious questions about meaningless documents.

These cases test interface robustness without introducing domain-specific semantic handlers.
