# 01 — Multi-path RRF skeleton (Hindsight alignment, v1)

**What to build:** Extract the assembly of weighted RRF fusion inputs out of `Searcher.search()`'s expanded-query path into a single private method, so future retrieval paths (temporal, graph-as-path) become "append one weighted entry" instead of "modify the orchestration". The refactor is behavior-preserving: exactly the same queries, weights, order, scores, and result shape as today. All existing tests stay green; no new files, classes, protocols, concurrency, or placeholders.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [x] In `Searcher`, move the assembly of the RRF fusion inputs (`all_results`) from the expanded-query path of `search()` into one private method that returns the assembled list
- [x] The method wires exactly the current entries in the same order: original query (0.5), LLM semantic rewrite (1.3), LLM keyword expansions (1.0) — behavior and scores identical to today
- [x] `Searcher.search()` signature, result document shape (`{id, text, score, metadata}`), weighted RRF formula, k=50, and query weights all unchanged
- [x] No new classes, protocols, registries, modules, or placeholder files (`temporal.py` / `graph_retriever.py` / `persona.py` are not created)
- [x] No concurrency (thread pool) added — deferred until a second independent retrieval path exists
- [x] Graph expansion stays append-only (ADR-0001 contract untouched); no graph path added to the fusion inputs
- [x] Full existing pytest suite green: `test_search_regression`, `test_search_expansion`, `test_search_reranker`, `test_search_graph`, `test_search_router`, `test_app_search`, `test_app_search_payload`, `test_tools_expansion`, `test_cross_table_agent`, `test_agent`

## Comments

- Spec: `.scratch/hindsight-memory-skeleton/spec.md`
- Background analysis: `docs/hindsight-integration-plan.md`
- Implemented on branch `feature/hindsight-multipath-rrf-skeleton`
- Verification: full suite `tests/` = **109 passed** (incl. permanent regression gate 4/4). Behavior-preserving refactor confirmed.
