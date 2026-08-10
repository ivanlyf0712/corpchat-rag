# Spec: Adaptive path activation — using Hindsight's main advantage

**Status:** ready-for-agent

**Feature slug:** `hindsight-adaptive-paths`

## Problem Statement

Hindsight's main advantage — recall via orthogonal evidence paths (content / structural graph / temporal) fused by RRF — is built but not used:

1. `search(graph_parallel=True)` exists but is **unreachable from the agent and UI**: `Agent.process` and `app._run_search` never pass it.
2. `AgenticDecider` decides `mode / expand / graph_expand / use_rerank` but **never decides whether to activate the graph path** for relationship-oriented queries.
3. The synthetic benchmark and the permanent regression gate contain **zero temporal or relationship query classes**, so the advantage cannot be measured or protected.

The temporal path is partially usable (auto-detected inside `Searcher.search`), but graph-parallel is effectively dead code from the product's perspective.

## Solution

Wire adaptive path activation through the decision layer and the agentic pipeline, and prove the advantage with benchmark coverage:

1. **Decide** — `AgenticDecider.decide` also outputs `graph_parallel` (rule-first on relationship keywords, LLM fallback), so the agent activates the structural path for relationship queries.
2. **Thread** — `Agent.process`, `app._run_search`, and the `search_messages` tool accept and pass `graph_parallel` (mirroring `expand` / `use_rerank`), so the decision reaches `Searcher.search` from every entry point.
3. **Measure** — extend the synthetic benchmark and the regression gate with temporal and relationship query classes (assert "time window respected", "structural neighbor recalled").
4. **Tune** — use the benchmark to set `GRAPH_RETRIEVAL_WEIGHT` and temporal behavior so orthogonal evidence helps without overwhelming direct matches.

## User Stories

1. As an end user, I want relationship queries ("跟陳志明聊過的物流", "誰跟鴻海談過合同") to surface structural neighbors, so that the agent recalls conversation context rather than only keyword matches.
2. As an end user, I want combined time + relationship queries ("上個月跟陳志明聊的物流報價") to use both the temporal and graph paths, so that the RRF fusion actually combines orthogonal evidence.
3. As an agent maintainer, I want `AgenticDecider` to decide path activation per query, so that graph/temporal activate adaptively instead of being always-on (noise) or never-on (dead).
4. As a developer, I want the path flags threaded through `Agent.process`, `app._run_search`, and the `search_messages` tool, so the decision reaches `Searcher.search` from every entry point.
5. As a regression maintainer, I want temporal and relationship query classes in the benchmark and regression gate, so that the advantage is visible as a number and protected against regressions.
6. As an operator, I want fusion weights configurable, so that graph/temporal contributions can be tuned per corpus.
7. As a user, I want queries with no relationship/time intent to behave exactly as before, so that the adaptive layer never degrades plain searches.

## Implementation Decisions

1. **Decision layer** — `AgenticDecider.decide` adds `graph_parallel` output. Rule-first on relationship/entity keywords (跟誰/和誰/還有誰/這個人/哪個客戶/發…消息的人/後來/之後/對方/他/她 + cross-entity patterns); LLM fallback mirrors the existing `_llm_decide_mode` pattern. Default `False`.
2. **Threading** — `Agent.process`, `app._run_search`, and `search_messages(query, ..., graph_parallel=False)` gain the flag and pass it to `Searcher.search`. The UI checkbox surfaces it (mirroring the existing Agent/Reranker/Expansion toggles).
3. **Temporal stays auto-detected** — no decision-layer change needed for time; `Searcher.search` already parses time intent. Only the combined case (time + relationship) needs graph_parallel to also be on.
4. **Benchmark** — extend `SYNTHETIC_TEST_QUERIES` and `tests/test_search_regression.py` with temporal classes (窗口被尊重) and relationship classes (結構鄰居被召回), using the deterministic in-memory index pattern.
5. **Weight tuning (D)** — after benchmark coverage lands, set `GRAPH_RETRIEVAL_WEIGHT` (and temporal behavior) so orthogonal evidence boosts without dominating; defaults may move from the current 0.8.
6. **No new modules** — changes live in `agentic.py`, `agent.py`, `app.py`, `tools.py`, `search.py` (benchmark), and tests.

## Testing Decisions

- **Single test seam: `Searcher.search()`** for retrieval behavior; `AgenticDecider.decide` unit-tested for the new `graph_parallel` decision (rule-hit, no-hit, LLM fallback).
- **Regression gate additions**: new temporal and relationship query classes in `tests/test_search_regression.py` (deterministic index) — the permanent gate now covers the advantage.
- **CLI benchmark**: `search.py synthetic-benchmark` extended with the new query classes (label-recall style assertions).
- **Full existing suite stays green** — the permanent regression gate.

## Out of Scope

- **CARA personality layer** — separate feature (`.scratch/hindsight-persona/`).
- **Neo4j** — rejected for the POC.
- **Multi-hop graph traversal** — remains 1 hop.
- **Always-on multi-path retrieval** — the adaptive decision layer is the point; paths activate per query intent.

## Further Notes

- Background: `docs/hindsight-integration-plan.md`; the skeleton (`.scratch/hindsight-memory-skeleton/`), temporal (`.scratch/hindsight-temporal/`), and graph-parallel (`.scratch/hindsight-graph-parallel/`) features this builds on.
- The graph-path weight (0.8) was noted as potentially too high in the graph-parallel ticket; benchmark coverage (ticket 03) is the prerequisite for tuning it safely (ticket 04).
