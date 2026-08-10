# 01 — Temporal retrieval (Hindsight temporal component)

**What to build:** Add a time-expression parser (rule-first, LLM fallback) and wire time into `Searcher.search()`: combined queries derive `date_from`/`date_to` from the detected window and enlarge the retrieval limit (5×) to avoid filter starvation; pure-temporal queries (no content keywords) route to a dedicated listing that scans `sections.tags` `send_time`, newest-first, bypassing RRF. No time intent → behavior completely unchanged. Full suite green.

**Blocked by:** None — can start immediately. (Depends on the multi-path RRF skeleton only in the sense that it extends `Searcher`; it does not require the skeleton.)

**Status:** ready-for-agent

- [x] New module `temporal.py`: `TimeWindow` dataclass + `TimeExpressionParser` with rules for `最近/近/前 N天|周|月|年`, bare `最近`, `前天/昨天/今天`, `上周/本周/这周`, `上月/这个月/本月`, `去年/今年`, and absolute dates (`YYYY-MM-DD`, `YYYY年M月[D日]`, `M月D日`)
- [x] LLM fallback gated by `_needs_llm` (time char + digit or range connector) so bare time words and unrelated queries never call the API; failures return None
- [x] `config.py`: `TEMPORAL_LIMIT_SCALE` (default 5.0) and `TEMPORAL_DEFAULT_WINDOW_DAYS` (default 7)
- [x] `Searcher.__init__` accepts optional `temporal_parser` (default: auto-enabled `TimeExpressionParser`); `search()` parses time intent when no explicit `date_from`/`date_to` was passed
- [x] Combined queries: window → `date_from`/`date_to`, retrieval limit × `TEMPORAL_LIMIT_SCALE` via `_retrieve_parallel(scale=...)` and Path A
- [x] Pure-temporal branch: `_is_pure_temporal` + `_temporal_list` (scans `sections.tags`, window+label filter, newest-first, no RRF/graph/rerank)
- [x] Date-boundary correctness: window comparisons normalize `send_time[:10]` in `_filter`, `_passes_filters`, `_temporal_list` (today's docs stay inside windows)
- [x] `TimeExpressionParser` / `TimeWindow` exported from the search package
- [x] New test file `tests/test_search_temporal.py`: parser unit tests (fixed `now`) + `Searcher.search()` integration tests (pure-temporal listing, label filter, combined time+content, no-time-intent unchanged, explicit dates win) — 11 tests
- [x] Full existing pytest suite green (120 passed, incl. permanent regression gate 4/4)

## Comments

- Spec: `.scratch/hindsight-temporal/spec.md`
- Background analysis: `docs/hindsight-integration-plan.md`
- Implemented on branch `feature/hindsight-multipath-rrf-skeleton`
- Test-infra fixes landed in the same pass (test isolation, not feature logic): `test_search_ui.py` now binds a single consistent fake streamlit, stubs `load_agent` so UI-routing tests never load the production index (fixes a full-suite MPS GPU out-of-memory failure), and mocks `_search_router` deterministically.
