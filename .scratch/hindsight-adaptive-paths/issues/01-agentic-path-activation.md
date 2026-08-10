# 01 — Agentic path activation (AgenticDecider decides graph_parallel)

**What to build:** Extend `AgenticDecider.decide` to also output `graph_parallel`, so relationship-oriented queries activate the structural graph path adaptively instead of it being always-on or never-on.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [x] `AgenticDecider.decide` gains a `graph_parallel` key (default `False`), rule-first on relationship/entity keywords: 跟誰/和誰/還有誰/這個人/哪個客戶/發…消息的人/後來/之後/對方/他/她 + cross-entity patterns
- [x] LLM fallback mirrors the existing `_llm_decide_mode` pattern (JSON/one-word decision, cache, graceful None on failure)
- [x] Pure relationship query rules do not fire on plain content queries (e.g. "物流報價" → `graph_parallel=False`)
- [x] Decision dict shape stays backward-compatible (`mode/expand/graph_expand/use_rerank` unchanged; new key added)

## Comments

- Spec: `.scratch/hindsight-adaptive-paths/spec.md`
- Builds on: `.scratch/hindsight-graph-parallel/` (`search(graph_parallel=True)`)
- Implemented & verified on branch feature/hindsight-multipath-rrf-skeleton (commits 2a7a54c, a5107fc); full suite 149 passed.
