# Spec: Graph parallel fusion path (Hindsight graph-traversal component)

**Status:** ready-for-agent

**Feature slug:** `hindsight-graph-parallel`

## Problem Statement

The structural conversation graph only contributes to retrieval as an append-only enhancement (`_graph_expand`, after RRF fusion): it adds neighbors below the base ranking and never reorders. Hindsight treats graph traversal as a first-class retrieval path that participates in fusion, giving structurally-connected neighbors an independent ranking signal. The retrieval stack has the seam for this — `_retrieve_parallel` assembles the weighted result lists that feed RRF — but no graph path is wired into it.

## Solution

Add an opt-in graph traversal retrieval path (`graph_parallel=True`) that returns a ranked neighbor list for RRF fusion:

1. Run one hybrid search for the query to build the id→relevance map (the existing query-consistency gate).
2. Take the top-N relevant documents as seeds.
3. Traverse the 4 traversal-eligible structural edges (same_conversation, sender_receiver, same_sender, same_company) from each seed, scoring each unique neighbor as `Σ seed_score × hop_discount × neighbor_query_relevance`.
4. Append the ranked neighbor list to the RRF fusion inputs with `GRAPH_RETRIEVAL_WEIGHT`.

The append-only contract (ADR-0001) remains the default: `graph_parallel` defaults to `False`, and `_graph_expand` is untouched. `graph_parallel` requires the expanded-query (Path B / RRF) flow; the append-only `graph_expand` continues to work in both paths.

## User Stories

1. As an end user, I want structurally-connected neighbors to rank on their own merit when graph-parallel mode is on, so that conversation/sender/company context can surface through fusion.
2. As a developer, I want the graph path to be opt-in, so that the verified append-only default (ADR-0001) is preserved for existing callers.
3. As a developer, I want the graph path to reuse the same query-consistency gate as append-only expansion, so that content-irrelevant neighbors never surface.
4. As a regression maintainer, I want every existing search test to stay green, so that the default behavior is provably unchanged.
5. As an operator, I want the graph-path weight configurable, so that it can be tuned or disabled without code changes.

## Implementation Decisions

1. **New `_graph_path_retrieve(query, limit)` on `Searcher`** — returns `List[(doc_id, score)]` ranked by `Σ_seeds seed_score × GRAPH_PARALLEL_HOP_DISCOUNT × relevance`. Skips only the seed set (seeds already run content paths); other query-relevant neighbors may contribute.
2. **Shared `_graph_query_scores(query)` helper** extracted from `_graph_expand` (identical computation: one hybrid search, id→relevance map). Used by both the append-only and parallel paths.
3. **Wired into `_retrieve_parallel`**: when `graph_parallel=True` and a graph exists, the neighbor list is appended as one `(results, GRAPH_RETRIEVAL_WEIGHT)` entry. This fulfills the skeleton seam's stated purpose.
4. **`search()` gains `graph_parallel: bool = False`** — opt-in, default preserves current behavior. Only active in the expanded-query (Path B) RRF flow.
5. **Config**: `GRAPH_RETRIEVAL_WEIGHT` (default 0.8), `GRAPH_PARALLEL_HOP_DISCOUNT` (0.8), `GRAPH_PARALLEL_SEED_LIMIT` (10).
6. **Traversal-eligible relations unchanged** — `same_label` is recorded but never traversed (ADR-0001 / CONTEXT.md).
7. **`_graph_expand` untouched** — the append-only contract remains the default and coexists with the parallel path.

## Testing Decisions

- **Single test seam: `Searcher.search()`** plus direct assertions on `_graph_path_retrieve` (the graph-path unit of behavior), consistent with the existing graph suite.
- **Deterministic graph-enabled index**: the same `CONVERSATION_TEMPLATES` + structural-relationships pattern as `test_search_graph.py`, with a FakeExpander (no LLM) so Path B runs deterministically.
- **Verified behaviors**: `_graph_path_retrieve` returns structural neighbors that pass the query-consistency gate; default-off preserves base behavior; `graph_parallel=True` surfaces graph neighbors in results without dropping the direct match; the gate excludes relevance-0 neighbors.
- **Full existing suite stays green** — the permanent regression gate.
- **Prior art**: `test_search_graph.py` (graph fixture + NetworkX.isquery workaround).

## Out of Scope

- **Auto-enabling the path** (e.g. via `AgenticDecider`) — left as a future toggle; the feature is opt-in via `search(graph_parallel=True)`.
- **CARA personality layer** — still deferred (no demand evidence).
- **Neo4j or any graph database** — rejected for the POC.
- **Multi-hop traversal** — the parallel path traverses 1 hop, matching the append-only path.

## Further Notes

- Background: `docs/hindsight-integration-plan.md` (Phase 2 of the roadmap); the multi-path RRF skeleton (`.scratch/hindsight-memory-skeleton/`); temporal retrieval (`.scratch/hindsight-temporal/`).
- **Test-infra fix landed in the same pass**: the search test suites' bge-m3 fixtures were `scope="session"`, keeping every model alive until session end and exhausting the 16 GiB Mac's MPS GPU budget (~20 GiB) as the suite grew. They are now `scope="module"` and a root `conftest.py` flushes the MPS cache after each test — peak memory is bounded and the full suite runs in ~half the time.
- The graph-path weight (0.8) may need tuning: with RRF the graph entry's weight competes with content entries, and a high weight can surface structurally-connected neighbors ahead of direct matches. It is configurable for exactly this reason.
