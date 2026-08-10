# 02 — Thread path flags through Agent, UI, and tools

**What to build:** Make `graph_parallel` reachable from every entry point: `Agent.process`, `app._run_search`, the UI checkbox, and the `search_messages` LangChain tool — so the decision made in ticket 01 actually reaches `Searcher.search`.

**Blocked by:** 01 (the decision layer) — but can be implemented in parallel since the flag defaults to `False`.

**Status:** ready-for-agent

- [x] `Agent.process(..., graph_parallel: bool = False)` accepts the flag and passes it to `self.searcher.search(...)`
- [x] `app._run_search(..., graph_parallel: bool = False)` accepts and passes the flag to `Searcher.search`
- [x] UI: a "Graph path" checkbox in the Enhancements panel mirrors the existing Agent/Reranker/Expansion toggles and flows into `_run_search` / `Agent.process`
- [x] `search_messages(query, expand, use_rerank)` tool gains `graph_parallel: bool = False` and passes it through the tool's search path (RRF fusion branch)
- [x] `CrossTableAgent` constructor accepts `graph_parallel` and threads it into `search_messages` invocations
- [x] Existing callers unaffected (new flag defaults `False`)

## Comments

- Spec: `.scratch/hindsight-adaptive-paths/spec.md`
- This is the "dead flag" removal: `graph_parallel` currently exists on `Searcher.search` but no caller reaches it (verified: `agent.py:443`, `app.py:323` don't pass it).
- Implemented & verified on branch feature/hindsight-multipath-rrf-skeleton (commits 2a7a54c, a5107fc); full suite 149 passed.
