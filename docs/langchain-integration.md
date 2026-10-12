# LangChain RAG integration

The shared `RagAnswerService` uses LangChain Core 1.6.4 by default, in fake and
live modes. Install project dependencies with `py -3.12 -m pip install -e .[runtime,test]`.
Existing provider configuration and knowledge indexes are unchanged.

## Responsibilities

1. Existing dialogue/routing code resolves intent and conversational references.
   The service retains its context loading, policy clarification and memory failure rules.
2. `HybridKnowledgeRetriever` implements LangChain's async `BaseRetriever`. It
   calls the existing embedding and knowledge ports, including cross-language and
   request-specific model/manual filters. It returns `Document` objects preserving
   chunk IDs, source metadata, ranks and scores; the retrieval chain restores the
   domain contract without changing BM25, vector search or RRF algorithms.
3. Evidence thresholds and clarification decisions still precede generation.
   Empty evidence never invokes the response model.
4. An LCEL sequence builds the existing grounded prompt, invokes the existing
   structured model adapter and runs the existing citation/language/length/leakage
   validator. It returns a domain `RagAnswerResult`, never raw model output.

Chains/retrievers are built per request: `with_knowledge()` overrides and concurrent
conversations cannot retain another request's knowledge scope or prompt builder.
No additional LLM rewrite/retry loop is introduced by this integration. The existing
provider adapters own timeouts, parsing and retry behavior. LangChain is not an
extra model service and does not replace DeepSeek, Qwen Embedding or SQLite.

## Scope and privacy

Web and mini-program adapters reuse this same RAG service. The worker compositions
also use it. Native product web search, business tools, sensitive-intent routing,
session storage and conversation summarization remain application code, not a
LangChain agent. This does not claim a LangGraph workflow or LangChain-managed memory.

LangSmith is a dependency of Core and explicitly pinned because its context API is
used to disable tracing. Application chain invocation uses a context-local
`tracing_context(enabled=False)` even when host tracing environment variables are
set. No LangSmith account/key is needed and no cloud tracing is enabled.

## Verification

- Full Python regression: 652 passed, 10 skipped (external integration prerequisites).
- Mini-program JavaScript regression: 5 passed.
- Dependency check: no broken requirements.
- Added tests for Document round-trip, request scope, concurrent overrides,
  embedding/index/model error mapping, citation validation and tracing suppression.
- Real-provider smoke test through the running mini-program API: Chinese request
  to enable Samsung TV grayscale returned a validated answer with one evidence
  chunk; the same-session English follow-up "How do I turn it off?" resolved the
  grayscale reference and returned a validated English answer with two evidence
  chunks. Both used `general_manual_operation` and disclosed model applicability
  limits. This is an API check, not a new phone UI verification.

These checks establish integration behavior, not improved answer accuracy or
production readiness. Retrieval coverage and web page availability remain limitations.
