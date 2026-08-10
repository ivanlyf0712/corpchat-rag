# Spec: Temporal retrieval (Hindsight temporal component)

**Status:** ready-for-agent

**Feature slug:** `hindsight-temporal`

## Problem Statement

Roughly 1-in-5 real user queries reference time (e.g. last year, last month), but the retrieval stack is structurally blind to time: it treats every query as a content-matching problem. Two failure modes result:

1. **Combined queries** ("去年的物流報價") — the time constraint is ignored, so results come back from the wrong period. The existing `date_from`/`date_to` parameters can filter, but they are never derived from the query and, being post-retrieval filters, they starve results: a global top-N search returns few documents inside a narrow window.
2. **Pure-temporal queries** ("最近的消息", "本月有哪些投诉") — there is no content keyword to match, so hybrid search returns nothing at all.

This is an intent-mismatch error, not a ranking error: no amount of reranking can fix it, because the time dimension is absent from the query representation.

## Solution

Add a time-expression parser (rule-first, LLM fallback) and integrate time into `Searcher.search()` with two behaviors:

1. **Combined queries** — detect the window, derive `date_from`/`date_to` from it (unless the caller already supplied explicit dates), and enlarge the retrieval limit while the window is active so the post-retrieval filter does not starve results.
2. **Pure-temporal queries** — route to a dedicated listing branch that scans document `send_time` metadata within the window, returns them newest-first, and bypasses RRF fusion entirely (a pure-temporal list and a content ranking live in different result universes).

When no time intent is present, the parser returns nothing and retrieval behavior is completely unchanged.

## User Stories

1. As an end user, I want "最近/昨天/上周/上月/去年" style queries to return the right period's messages, so that time-sensitive questions are actually answered.
2. As an end user, I want "最近的消息" (no content keywords) to list recent messages, so that pure-time questions do not return empty.
3. As an end user, I want a combined query like "去年的物流報價" to return logistics quotes from last year, so that the time constraint narrows rather than gets ignored.
4. As an end user, I want a query with no time intent ("物流報價") to behave exactly as before, so that existing retrieval quality is untouched.

## Implementation Decisions

1. **Rule-first parser** (`TimeExpressionParser`): handles `最近/近/前 N天|周|月|年`, bare `最近`/`近` (default 7-day window), `前天/昨天/今天`, `上周/本周/这周`, `上月/这个月/上月/本月`, `去年/今年`, and absolute dates (`YYYY-MM-DD`, `YYYY年M月[D日]`, `M月D日`).
2. **LLM fallback only fires when the query contains a time character AND a digit or a range connector** (`之前/以前/之后/以来/期间/从/到/至`). This prevents accidental API calls for bare time words or unrelated queries; failures return None.
3. **Combined queries**: when the parser detects a window and no explicit `date_from`/`date_to` was passed, set them from the window and multiply the retrieval limit by `TEMPORAL_LIMIT_SCALE` (default 5× on top of the existing 3×). This is the agreed "enlarge limit + post-filter" strategy.
4. **Pure-temporal branch**: after removing the matched time expression and temporal filler words, if no meaningful content remains (`< 2` characters), route to a dedicated listing that scans `sections.tags` for `send_time`, filters by window and label, sorts newest-first, and returns without RRF/graph/rerank.
5. **Date-boundary correctness**: all window comparisons normalize `send_time` to its date part (`[:10]`) so a document sent on the window's last day (with a time suffix) is not excluded.
6. **Explicit dates win**: the parser never overrides caller-supplied `date_from`/`date_to`.
7. **No new runtime dependency** beyond the existing stack (rule parsing uses `re`/`datetime`; LLM fallback reuses `LiteLLMClient`).
8. **No behavior change without time intent**: parser returns None → the pipeline is identical to before.

## Testing Decisions

- **Single test seam: `Searcher.search()`** — the existing project seam, consistent with the retrieval-base-consolidation and skeleton efforts. Parser rules are additionally unit-tested with a fixed `now` for determinism.
- **Deterministic index**: docs with `send_time` values relative to the test run (e.g. `now - 2d`, `now - 20d`, `now - 200d`) so relative windows behave deterministically.
- **Verified behaviors**: relative-day windows, bare `最近`, day/week/month/year windows, absolute dates, no-time-intent returns None, pure-temporal listing (newest first), pure-temporal with label filter, combined time+content filtering, unchanged behavior without time intent, explicit dates overriding the parser.
- **Full existing suite stays green** — the permanent regression gate.
- **Prior art**: `test_search_regression.py`'s deterministic in-memory index pattern.

## Out of Scope

- **Temporal as an independent RRF fusion path** — pure-temporal results bypass RRF; combined queries use window filtering, not a separate fusion path.
- **Recency ranking bias for non-temporal queries** — fuzzy recency weighting is explicitly deferred; only explicit windows are handled.
- **Persistent temporal index / side table** — the POC scans `sections.tags` in memory; a `doc_id → send_time` side table is a later optimization if data grows.
- **Graph parallel path and CARA personality layer** — still deferred per the skeleton spec.
- **Neo4j or any graph database** — rejected for the POC.

## Further Notes

- Background: `docs/hindsight-integration-plan.md` and the multi-path RRF skeleton (`.scratch/hindsight-memory-skeleton/`). The temporal feature is the first real plug-in consumer of `Searcher._retrieve_parallel()`'s assembly seam, though in the agreed design it uses window filtering and a dedicated branch rather than an extra RRF path.
- The parser's `_needs_llm` guard keeps the full test suite network-free: rule-covered queries (including all test queries) never call the LLM.

5. As a developer, I want the time parser rule-first so that common expressions resolve in <1ms without an LLM call.
6. As an operator, I want the LLM fallback to be optional and bounded, so that parser failures degrade gracefully.
7. As a regression maintainer, I want every existing search test to stay green, so that the verified base behavior is preserved.
8. As a developer, I want explicit `date_from`/`date_to` to take precedence over the parser, so that callers can still force a window.
