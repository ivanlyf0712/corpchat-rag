# 01 — Graph parallel fusion path (Hindsight graph-traversal component)

**What to build:** Add an opt-in graph-traversal retrieval path (`graph_parallel=True`) that returns a ranked structural-neighbor list for RRF fusion via `_retrieve_parallel`. Default stays append-only `_graph_expand` (ADR-0001 contract untouched). New `_graph_path_retrieve` + shared `_graph_query_scores` helper; config `GRAPH_RETRIEVAL_WEIGHT` / `GRAPH_PARALLEL_HOP_DISCOUNT` / `GRAPH_PARALLEL_SEED_LIMIT`. Full suite green.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [x] Extract shared `_graph_query_scores(query)` (id→relevance from one hybrid search) from `_graph_expand`; behavior identical
- [x] New `_graph_path_retrieve(query, limit)` on `Searcher`: seeds = top-N relevant docs; traverse 4 traversal-eligible structural edges (same_label never traversed); score = Σ seed × hop_discount × neighbor_relevance; query-consistency gate excludes relevance-0 neighbors; skips seeds (already on content paths)
- [x] `_retrieve_parallel` appends the graph path as one `(results, GRAPH_RETRIEVAL_WEIGHT)` entry when `graph_parallel=True` and a graph exists
- [x] `search()` gains `graph_parallel: bool = False` — opt-in, default behavior unchanged; active only in the expanded-query (Path B) RRF flow
- [x] Config: `GRAPH_RETRIEVAL_WEIGHT` (0.8), `GRAPH_PARALLEL_HOP_DISCOUNT` (0.8), `GRAPH_PARALLEL_SEED_LIMIT` (10)
- [x] `_graph_expand` (append-only) untouched; coexists with the parallel path
- [x] New test file `tests/test_search_graph_parallel.py`: retriever returns traversal-eligible, gate-passing neighbors; default-off preserves base; `graph_parallel=True` surfaces graph neighbors without dropping the direct match; gate excludes relevance-0 neighbors — 4 tests
- [x] Full existing pytest suite green (124 passed, incl. permanent regression gate 4/4)

## Comments

- Spec: `.scratch/hindsight-graph-parallel/spec.md`
- Background analysis: `docs/hindsight-integration-plan.md` (Phase 2)
- Implemented on branch `feature/hindsight-multipath-rrf-skeleton`
- **Test-infra fix (same pass)**: the search suites' bge-m3 fixtures were `scope="session"` (all models alive until session end → MPS GPU OOM on a 16 GiB Mac as the suite grew). Changed to `scope="module"` (models freed per file) and added root `conftest.py` flushing the MPS cache after each test. Full suite now runs in ~108s (was ~234s) with bounded memory.
- The graph-path weight (0.8) is tunable; a high weight can surface structurally-connected neighbors ahead of direct matches, which is the intended opt-in boost but worth tuning per corpus.
