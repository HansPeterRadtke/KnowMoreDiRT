# KnowMoreDiRT Requirements from the KMD Voice Specification

## Complete requirements narrative

KnowMoreDiRT must handle difficult question-answering cases in which information is scattered across a corpus or is only true under specific conditions, circumstances, environments, dates, or discourse scopes. The central difficulty is not merely finding a matching sentence. KMD must preserve and recover the circumstances that determine what that sentence means and whether it is valid for the user's question.

A canonical example is a document written by a child, Timmy, describing a dream in which somebody explains a law. A locally retrieved chunk may contain only the law-like statement. If KMD sees that chunk without the dream context, it may incorrectly treat the statement as an official law. Such a mistake can have serious consequences. The system must therefore support standard cases and many extreme cases of this general form, and the test suite must contain both ordinary and difficult examples involving scattered information, conditional truth, special environments, distant context, and misleading locally factual-looking text.

The Timmy example should be treated as a reusable pattern rather than a one-off story. Many questions have evidence that is relevant but does not directly establish the requested real-world proposition. Suppose Timmy's dream contains a rule that bicycles may not drive faster than cars, while the KMD corpus contains no official law text establishing such a rule. If the user asks whether bicycles may drive faster than cars under German law or common traffic law, KMD must not answer yes or no from the language model's pretrained knowledge. It must recognize that the corpus does not establish the requested real-world law. The direct answer is therefore that KMD does not know from the available database or did not find official law text supporting the proposition. At the same time, the dream evidence is relevant and should be reported: KMD can explain that a particular file contains Timmy dreaming of somebody describing exactly such a rule. The dream evidence must always be presented under the correct circumstance and must never be promoted to official real-world law.

This pattern generalizes. Many questions cannot be answered with certainty from the corpus, yet the user may still benefit from related information. KMD must therefore distinguish between the exact proposition requested and nearby evidence that is relevant but differently scoped. A bare yes/no answer can be wrong even when related evidence exists. A bare “no evidence” answer can also be unnecessarily unhelpful when the corpus contains scoped evidence that the user would reasonably want to know about. KMD must be able to return a qualified unknown together with correctly qualified related evidence.

The default interpretation of an underspecified factual question is the ordinary real world. If the user does not explicitly ask about Timmy's dream, a dream proposition cannot directly answer the question. The same principle applies to other special circumstances. Date is also a circumstance. A statement may be valid at one date and not another. More generally, statements can exist under special conditions or environments, while an ordinary document with no established special condition is treated under its ordinary/default real-world context.

Special context need not be expressed by a single explicit sentence. Timmy may never write “this is what I dreamed about.” A text can nevertheless clearly describe a dream. KMD must be able to infer contextual scope from the document as a whole and from its discourse structure. At the same time, KMD must distinguish a passage that is actually established as a dream from one that merely sounds dream-like. The system must record such contextual judgments correctly so that uncertain context is not later treated as certain fact.

Every item of information used by KMD must therefore be marked with enough contextual information to support correct later question answering. Context is not optional descriptive metadata. It is part of the meaning of the information.

All factual answer content must come from entries in the KMD database. Language models used by KMD must not silently answer factual questions from their pretrained data, memories, or judgments unless such external model knowledge is explicitly requested as a separate mode. This grounding rule must be present throughout the relevant prompts and model-call pipeline. The wording should remain simple enough not to confuse the model, but the invariant is strict: models must not hallucinate factual answers from their trained weights, and every factual statement used to answer a normal KMD question must be derived from database evidence.

The benchmark and tester infrastructure must judge answers by meaning rather than exact wording. A gold answer is necessarily written in one specific phrasing. A system may produce a different grammatical structure while preserving exactly the same meaning. That answer must pass. A superficially similar answer with a materially different meaning must fail. The benchmark and agent-system testers therefore require an LLM-based semantic-equivalence judgment rather than literal string equality alone.

Question answering begins with analysis of the user's question. An LLM should determine what information has to be retrieved and what search actions are required. Search is not limited to the literal words supplied by the user. For a broad question such as “what is a cat,” the planner may search for `cat` while also generating related concepts and phrases such as pets, household animals, biology, and other semantically related terms. Query expansion is intended to improve recall when the relevant corpus does not use exactly the same wording as the question.

Vector-embedding-distance search must actually be used. It must not exist only nominally while the real system relies exclusively on literal matching. Vector retrieval requires a similarity or likelihood threshold. That threshold is a system parameter and is difficult to choose exactly. KMD should begin with a reasonable value, perform basic tests, and optimize the value based on measured retrieval behavior. The specification does not prescribe one permanent numeric threshold; it requires a tested threshold appropriate to the active system.

Retrieval may produce a very large result set. A broad question can return one hundred or more relevant or semantically similar chunks. A simple intuitive design would give the model the user question plus every retrieved database result and ask for an answer. That approach may work in many cases, possibly even most cases, but it is not sufficient as KMD's general architecture.

Model context is finite. If one hundred chunks contain roughly one thousand tokens each, the retrieved material alone can fill or nearly fill the context window of many models. Larger or more complicated cases can exceed the context window quickly. At the same time, the model may need information from widely separated parts of a document in order to interpret a matched chunk correctly.

Scope-defining context can occur before or after the matched statement. Timmy may write “then I woke up and it was all a dream” only at the end of the story. That later sentence must be capable of placing earlier statements under dream scope. Conversely, a file header at the beginning may state that the following text is an official German traffic law, establishing the interpretation of later content. KMD must therefore support both forward and retroactive contextual effects.

Surrounding chunks can be necessary even when they do not contain the search terms. If a matching chunk contains `cat`, `traffic law`, or another query-related phrase, KMD may need the immediately previous chunk, several previous chunks, ten previous chunks, following chunks, a file header, or another distant discourse segment before it can determine whether the matched text is a dream, a special condition, an official statement, or an ordinary face-value assertion. Retrieval cannot assume that the query-matching chunk contains all information required to interpret itself.

One possible solution is to put all content from the relevant file into the model context. Another possible solution is for the system's data structures to encode the discourse and contextual relationships so that KMD can retrieve the proposition together with the context needed to interpret it. DRT and DRS are specifically intended to solve this problem. They must correctly represent the information and the circumstances under which it is valid so that retrieval and later question answering preserve the meaning of the source.

DRT and DRS are therefore not optional decorative structures. The difficult examples exist in part to test whether DRT/DRS really work. The system must demonstrate that its discourse representation can encode context that begins before a proposition, context that is established after a proposition, document-level framing, implicit dream descriptions, dated circumstances, surrounding context, and other conditions that determine truth.

## Hard definitions and requirements

### Corpus grounding

1. KMD is corpus-grounded for normal factual question answering.
2. Every factual answer claim must be derived from evidence stored in the KMD database.
3. Models must not silently introduce factual answers from pretrained knowledge, trained weights, memory, or unsupported judgment.
4. Grounding instructions must be present in every model stage that can introduce factual content, not only in the final answer generator.
5. Grounding prompts should be simple and explicit so that the restriction does not unnecessarily confuse the model.
6. Use of general model knowledge is permitted only when explicitly requested as a distinct operating mode.

### Default scope and truth

7. An underspecified factual question defaults to the ordinary real world.
8. A proposition from a dream or another special environment cannot directly establish an ordinary real-world answer unless the user explicitly asks about that environment.
9. If the requested real-world proposition is not established by corpus evidence, KMD must be able to answer that it does not know or did not find sufficient evidence.
10. Relevant evidence from a different scope should still be surfaced when useful, but its circumstance must be stated explicitly.
11. KMD must distinguish the direct answer from related differently scoped evidence.
12. KMD must never promote a conditionally valid, fictional, dreamed, hypothetical, reported, quoted, dated, or otherwise scoped statement into an unconditional fact merely because its local wording appears factual.

### Circumstances and context

13. Context can completely determine the meaning and validity of content.
14. Date is a circumstance and must be represented when it affects applicability.
15. Dream scope is a required supported circumstance.
16. Other special conditions and environments must be representable by the same general mechanism rather than through dream-specific hardcoding.
17. An ordinary document with no established special circumstance is interpreted in its ordinary/default context.
18. Context may be established explicitly or implicitly.
19. KMD must distinguish context that is established by evidence from context that is merely plausible or stylistically suggested.
20. Contextual confidence or uncertainty must be represented correctly.
21. Every information unit must carry enough context to permit correct later question answering.

### Long-range and retroactive scope

22. Scope can be established before the proposition it governs.
23. Scope can be established after the proposition it governs.
24. Retroactive context is required. A later statement such as “then I woke up and it was all a dream” must be able to place earlier content under dream scope.
25. File headers and section headers can establish context for later content.
26. Context needed to interpret a proposition may reside in chunks that do not contain the query terms.
27. KMD must support retrieval of preceding context, following context, document-level context, and discourse-linked context.
28. No fixed assumption may be made that the matched chunk is semantically self-contained.

### Question analysis and query expansion

29. Every question must be analyzed before retrieval.
30. The planner must determine what information needs to be searched.
31. The planner may generate additional semantically related words, phrases, or concepts beyond the literal query wording.
32. Query expansion must preserve the user's actual intent rather than changing the question.

### Vector retrieval

33. Vector-embedding-distance retrieval must be genuinely used in KMD question answering.
34. Vector retrieval must use a similarity or likelihood threshold.
35. The threshold is a configurable system parameter.
36. The threshold must be selected and optimized through empirical testing.
37. No single threshold value is defined as permanently correct independent of embedding model, chunking, corpus, and retrieval behavior.

### Large retrieval result sets and context limits

38. KMD must handle cases in which retrieval returns a very large number of chunks.
39. The architecture must not depend on all retrieved chunks fitting into one model context.
40. The system must handle cases on the order of one hundred chunks and roughly one thousand tokens per chunk, as well as larger cases.
41. A flat prompt containing the question plus all retrieved results is not sufficient as the sole general architecture.
42. The model must still receive or have access to all information necessary to interpret the selected evidence correctly.
43. Context-window management must preserve interpretation-critical context rather than only the most query-similar chunks.

### DRT and DRS

44. DRT and DRS must be operational parts of the system.
45. DRT/DRS must represent the contextual and discourse relationships that determine the meaning of information.
46. DRT/DRS must support retrieval of a proposition together with the context needed to interpret it.
47. DRT/DRS must support context that is distant from the proposition.
48. DRT/DRS must support context established by headers or document structure.
49. DRT/DRS must support retroactive scope.
50. DRT/DRS must support implicit special-context descriptions when the evidence establishes them.
51. Tests must demonstrate that DRT/DRS actually affect difficult cases correctly rather than existing as unused abstractions.

### Qualified unknown behavior

52. KMD must distinguish “the corpus does not establish the requested proposition” from “there is no related information.”
53. If direct evidence is absent, KMD must not invent a yes/no answer from model knowledge.
54. If related evidence exists in another circumstance, KMD should report it with the circumstance explicitly attached.
55. The Timmy bicycle-law case is a canonical acceptance pattern for this behavior.

### Testing

56. The test suite must contain normal/standard use cases.
57. The test suite must also contain many extreme and adversarial context-dependent use cases.
58. Tests must cover scattered information.
59. Tests must cover facts that are only valid under specific circumstances.
60. Tests must cover locally factual-looking chunks whose true context is elsewhere.
61. Tests must cover context appearing before a proposition.
62. Tests must cover context appearing after a proposition.
63. Tests must cover surrounding chunks that do not contain the query terms.
64. Tests must cover implicit dream-like descriptions and distinguish established dream scope from merely suggestive wording.
65. Tests must cover qualified unknown answers plus scoped related evidence.
66. Tests must verify that pretrained model knowledge is not substituted for absent corpus evidence.
67. The Timmy dream/law pattern must remain represented in the acceptance suite together with analogous cases.

### Semantic benchmark judgment

68. Benchmarks and agent-system testers must judge semantic equivalence rather than literal string equality.
69. This requires an LLM-based semantic judge.
70. A correct answer expressed with a different grammatical structure or wording must pass.
71. A materially different or false answer must fail even if its wording is superficially similar.
72. Gold answers represent intended meaning, not a mandatory exact surface string.

## Multiple-choice possibilities left open for implementation

### Whole-file context versus structured context

One possible strategy is to place the entire relevant file into the model context whenever local evidence may depend on distant context. Another strategy is to make DRT/DRS and the persistent data structures encode the required context so that the system can retrieve a scope-complete evidence set without loading the entire file. The requirements strongly motivate the structured approach, but the voice specification explicitly presents both possibilities.

### Amount of neighboring raw context

A matched chunk may require the previous chunk, several previous chunks, approximately ten previous chunks, following chunks, or a much more distant part of the document. No fixed neighbor count is specified.

### Similarity threshold

A reasonable initial threshold should be selected and then tested and optimized. No numeric threshold is mandated.

### Query-expansion breadth

The planner may generate related concepts such as pets, household animals, and biology for a broad query about cats. The exact number and breadth of expansions are not specified.

### Flat raw context versus structured retrieval

Providing the question plus all retrieved database results may be a useful simple baseline and may work in many cases. The final system must nevertheless handle cases where this approach exceeds the context window or loses critical discourse structure. The exact balance between raw evidence and structured representation remains an architectural choice.

### Grounding-prompt wording

The grounding requirement is strict, but the exact wording is not fixed. Prompts should state the source-only constraint as simply and clearly as possible.

### Representation of uncertain implicit context

KMD must distinguish established special scope from text that merely appears to imply that scope. The exact confidence representation, labels, or probability model are not prescribed by the voice specification.

## Open questions requiring explicit system decisions

1. What exact DRT/DRS schema represents scope, discourse relations, provenance, temporal conditions, authority, and uncertainty?
2. What exact propagation rules govern context across chunks, paragraphs, sections, headers, quotations, and document boundaries?
3. How is retroactive scope represented and reconciled when later text changes the interpretation of earlier text?
4. What formal criterion distinguishes an ordinary/default document from a special context when no explicit marker is present?
5. How is uncertain implicit scope represented so that “probably a dream” cannot become “certainly a dream” or an unconditional real-world fact?
6. How are contradictory propositions represented when they occur in different scopes?
7. How are contradictions represented when they occur in the same scope?
8. What exact retrieval pipeline combines vector retrieval, literal retrieval, structured retrieval, graph expansion, and reranking?
9. Which embedding model, distance metric, index, and threshold should production KMD use?
10. How should threshold tuning be evaluated without overfitting to a narrow benchmark?
11. How many query expansions should be generated, and how should intent drift be detected?
12. How should the system determine which neighboring or discourse-linked chunks must accompany a matched proposition?
13. When should KMD load a whole file or large section instead of relying on structured context retrieval?
14. How should a bounded context window prioritize proposition evidence, scope carriers, provenance, contradictions, and background information?
15. What exact answer structure should represent a qualified unknown plus related scoped evidence?
16. What prompt contracts and verification mechanisms best prevent pretrained-memory leakage across all model calls?
17. How should tests detect unsupported facts introduced from model memory?
18. Which model and rubric should serve as the semantic-equivalence judge?
19. How should judge nondeterminism, disagreement, retries, and caching be handled?
20. How should gold answers represent multiple valid semantic formulations and qualified variants?
21. What corpus of normal, difficult, and extreme cases is sufficient to demonstrate that DRT/DRS is operationally correct?
22. What ablation tests demonstrate that DRT/DRS and contextual retrieval materially improve hard cases over a flat question-plus-chunks baseline?
23. What persistence, versioning, invalidation, and migration rules apply to semantic annotations and context structures?
24. What scalability, memory, latency, and context-budget targets should apply to large corpora and large retrieval result sets?

## Summary

KMD must answer questions from its database rather than from hidden model memory. The system must treat context as part of the fact itself: a statement is only usable under the circumstances in which the source establishes it. The default scope for an underspecified factual question is the real world. Dreamed, hypothetical, reported, dated, quoted, or otherwise specially scoped information must never be silently promoted into an unconditional real-world fact, but relevant differently scoped evidence should still be available to the user with explicit qualification.

Question answering begins with LLM-based query analysis and can include bounded semantic query expansion. Vector-embedding retrieval is mandatory and must use an empirically tested similarity threshold. Retrieval can return far more information than a model context can hold, and the context required to interpret a matching chunk can be distant, absent from the matching chunk, or established later in the document. DRT/DRS and the surrounding system structures must therefore encode and recover discourse scope, including headers, implicit contexts, neighboring context, and retroactive scope.

Testing must cover ordinary cases and many difficult/extreme cases, including the Timmy dream/law pattern. Evaluation must judge answer meaning rather than exact wording through an LLM semantic judge. A different phrasing of the same correct answer must pass; a semantically wrong answer must fail. The complete architecture succeeds only if it can retrieve the right information together with the circumstances that determine what that information actually means.
