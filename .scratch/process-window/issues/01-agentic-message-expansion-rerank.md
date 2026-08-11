# 01 — Agentic message-search expansion + rerank

**What to build:** When the agent handles a query with the shared Reranker / LLM-expansion toggles ON, the message search runs full LLM query expansion (semantic rephrase + keywords) with weighted RRF fusion and cross-encoder reranking — treated as an individual search. Contact search stays exact-match. The tool output exposes the expanded queries, hit count, and top-5 previews so the UI can render them.

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [x] `search_messages` accepts expand/use_rerank flags; when ON it expands the query, searches each variant, fuses via weighted RRF, and reranks with the cross-encoder
- [x] `search_contacts` never expands or reranks, even when flags are ON
- [x] The agent accepts `expand` / `use_rerank` constructor options and forwards them to tool invocations
- [x] Tool output / agent result includes expanded queries, hit count, and top-5 previews (content + sender + score)
- [x] When flags are OFF, tool output matches today's behavior (no expansion, no rerank)
- [x] Tests: tool-level (fake expander/reranker) + agent-level, deterministic without a live LLM
