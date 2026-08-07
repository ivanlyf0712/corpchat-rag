# Spec: Multi-path RRF skeleton for Hindsight alignment

**Status:** ready-for-agent

**Feature slug:** `hindsight-memory-skeleton`

## Problem Statement

The RAG retrieval stack cannot grow toward a multi-path design without invasive changes each time. Today the inputs to weighted RRF fusion are assembled inline inside `Searcher.search()`'s expanded-query path: the LLM query expansion produces a list of queries (original + semantic rewrite + keyword expansions), each query is searched independently, and each search's result list is collected into a single `all_results` structure that the weighted RRF fusion consumes. There is no place in the code where "the set of retrieval paths feeding RRF fusion" is visible as a unit.

Every future retrieval enhancement — a temporal retrieval path, a graph-as-path, any additional data source — currently requires editing that same method's internals again, and each edit risks disturbing the verified base retrieval behavior. The permanent regression gate (`Searcher.search()` as the single test seam) protects behavior, but the absence of a stable assembly point makes every addition a surgery on the search orchestration rather than a plug.

## Solution

Extract the responsibility "assemble the weighted result lists that feed RRF fusion" out of `Searcher.search()` into a single private method. In this spec, the method produces **exactly the same entries as today** — the original query and each LLM-expanded query, each with its configured weight, in the same order — so retrieval behavior, scores, and ordering are unchanged. This creates a stable seam: a future retrieval path (temporal, graph-as-path) becomes "append one weighted entry to the assembled list" instead of "modify the orchestration".

The change is purely structural and invisible to every caller. `Searcher.search()` keeps its signature, result shape, and fusion math. The graph stays append-only per ADR-0001 and the structural-conversation-graph contract.

## User Stories

1. As a search engineer, I want the fusion inputs assembled in one place, so that adding a future retrieval path is one well-defined addition rather than surgery on the search orchestration.
2. As an end user, I want search results to be identical after this refactor, so that the retrieval quality I rely on is untouched.
3. As a regression maintainer, I want every existing search test to stay green, so that the verified base behavior is provably preserved.
4. As a future feature implementer (e.g. temporal retrieval), I want to add a path by appending a weighted result list, so that I do not need to understand the whole orchestration.
5. As an agent maintainer, I want `Searcher.search()`'s signature and result shape unchanged, so that the agent, tools, CLI, and UI callers are unaffected.
6. As a code reviewer, I want the refactor to introduce no new abstraction without an implementer, so that the codebase stays lean and the POC stays maintainable.
7. As an operator, I want no new runtime dependencies or services, so that the local deployment is unaffected.
8. As a Hindsight roadmap reviewer, I want the multi-path skeleton in place, so that temporal and other retrieval paths plug in cheaply when their tickets land.
9. As a developer, I want the refactor to respect the existing domain contract (graph expansion stays append-only, match surface unchanged), so that no verified design decision is silently overridden.


## Implementation Decisions

1. **Keep the per-query fusion unit (single-level weighted RRF).** The fusion input remains a list of `(result_list, weight)` entries; each entry is produced by one search over one query. Do **not** restructure into path-level two-level fusion (per-path internal fusion followed by a second RRF). Two-level fusion changes every score and would break the regression gate without evidence of benefit.

2. **Extract one private method** on `Searcher` that returns the assembled fusion inputs (the `all_results` structure). In this spec it wires exactly the current entries, in the same order: original query (0.5), LLM semantic rewrite (1.3), LLM keyword expansions (1.0). The method is the single place where the set of retrieval paths is visible.

3. **No new classes, protocols, registries, or files in this spec.** The method is private and structural. A formal retriever protocol is intentionally deferred until the first real second path (temporal) lands, when the interface can be designed against a real consumer.

4. **No concurrency in this spec.** Thread-pool parallelism is deferred until more than one independent retrieval path exists. The method documents that future paths are independent and therefore parallelizable.

5. **`Searcher.search()` signature, result shape, and fusion math unchanged.** The unified result document stays `{id, text, score, metadata}` with optional `rerank_score`. Weighted RRF constants (k=50) and query weights (0.5 / 1.3 / 1.0) unchanged.

6. **Graph remains append-only.** Per ADR-0001 and the graph-expansion contract, no graph path is added to the fusion inputs in this spec.

7. **Match surface and document fetch unchanged.** Retrieval still matches message content plus the curated customer title; structured metadata remains filter/display/LLM-context only.

8. **No placeholder files.** `temporal.py`, `graph_retriever.py`, `persona.py` are not created in this spec.


## Testing Decisions

- **Single test seam: `Searcher.search()`** — the existing, highest project seam, consistent with the "single test seam" decision of the retrieval-base-consolidation effort. A good test asserts external behavior (what a user sees), not implementation details.
- **Verification = the full existing pytest suite stays green before and after the refactor.** Because this spec changes no behavior, the existing suites are the gate: `test_search_regression`, `test_search_expansion`, `test_search_reranker`, `test_search_graph`, `test_search_router`, `test_app_search`, `test_app_search_payload`, `test_tools_expansion`, `test_cross_table_agent`, `test_agent`.
- **No new test file.** A unit test on the extracted private method would test implementation, not behavior, and would lock in structure the method is meant to keep flexible.
- **Prior art:** the deterministic in-memory index pattern (built from the conversation templates with `BAAI/bge-m3`) used by the existing search suites; the regression suite is the permanent gate for every later layer.

## Out of Scope

- **Temporal retrieval** (time-expression parser, temporal retriever, window filtering, pure-temporal branch) — deferred to a separate ticket that includes a txtai-spike and its own tests. Real-product context: roughly 1-in-5 queries reference time (e.g. last year, last month).
- **Graph as a parallel fusion path** — graph expansion stays append-only per ADR-0001.
- **CARA personality layer** (disposition traits, answer stylization) — deferred, no demand evidence yet.
- **Retriever protocol / registry abstraction** — deferred until the first real second path exists.
- **Thread-pool parallelism** — deferred until multiple independent paths exist.
- **Neo4j or any graph database** — explicitly rejected for the POC.
- **Any change to search behavior, scores, ordering, weights, or result shape.**
- **Changes to the agent, tool, CLI, or UI layers.**

## Further Notes

- Background analysis lives in `docs/hindsight-integration-plan.md` (2026-08-07): the component comparison, gap analysis, integration points, and phased roadmap that this spec's v1 scope is cut from.
- This spec is Phase-1-of-selective-adoption only: establish the multi-path RRF skeleton without disturbing the verified base. Temporal and other Hindsight components land as separate tickets with their own spikes and tests.
- When the temporal ticket lands, the agreed direction (to be confirmed in that ticket) is: detect a time window in the query (rule-first, LLM fallback), enlarge the retrieval limit when a window is present and apply the existing post-retrieval date filter, and route pure-temporal queries (no content keywords) through a dedicated branch that bypasses RRF fusion.
