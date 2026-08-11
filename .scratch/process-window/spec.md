# Unified Process Window — Spec

## Problem Statement

The Search page shows two redundant, half-empty result windows per turn:
- **Details** shows the raw-hits table — but in agentic mode it is always empty ("No raw hits available for this turn."), because agentic mode stores `raw_hits = []`.
- **🤖 Agent Process** shows the agent timeline — but only in agentic mode, and its steps contain only a truncated query string, not the actual search output.

A user who turns on agent mode gets *no visibility into what the agent actually searched and found*. In non-agent mode, expanded queries flash by inside a transient `st.status` and are never persisted. Meanwhile a dead "Raw results" toggle block sits unreachable at the bottom of the render function.

## Solution

Merge everything into **one "Process" window per turn**, collapsed by default:

- **Non-agentic turns:** the Process window shows the raw-hits table exactly as today (Message ID / Content / Score / Metadata).
- **Agentic turns:** the Process window shows one **expandable sub-window per tool called**, with compact styling, containing:
  - tool name + the query that was actually searched
  - the LLM-expanded queries (semantic rephrase + keywords) — persisted, shown under the tool
  - hit count + top-5 previews (content + sender + score)

The expander label is `Process`, or `Process (agentic · ✅ · N tools · Xms)` when the agent handled the turn (`⚠️ fallback` in place of `✅` when fallback was used).

While processing, both modes share a **single unified live-activity display** with a fade-in → hold → fade-out animation per stage: `🧠 routing...` → `🔍 using search_messages...` (expanded queries appear beneath) → `👤 using search_contacts...` → `✨ generating answer...`. The stage label holds until the stage *actually completes*, then fades out — no artificial sleeping.

Full LLM query expansion + cross-encoder reranking are enabled **inside `search_messages`** when the shared `Reranker` / `LLM expansion` checkboxes are ON (default ON, user can disable). `search_contacts` stays exact-match (entity queries don't benefit from expansion). Expanded queries are a *search mechanism + transparency display* — they are **never** fed to the LLM answer synthesizer, which sees only clean top-5 results.

The dead "Raw results" toggle block is deleted.

## User Stories

1. As a user searching in non-agent mode, I want the same raw-hits table I get today, so that the merged window doesn't change my current experience.
2. As a user searching in non-agent mode, I want the window collapsed by default, so that my chat stays clean until I opt in to see details.
3. As a user in agent mode, I want one Process window (not two), so that there's no empty/redundant Details tab.
4. As a user in agent mode, I want the label to read "Process (agentic)" when the agent handled my turn, so that I can tell which pipeline ran.
5. As a user in agent mode, I want to see each tool that was called as its own expandable sub-window, so that I can see what the agent actually did.
6. As a user in agent mode, I want to see the exact query used per tool, so that I can verify the agent searched what I asked about.
7. As a user in agent mode, I want to see the LLM-expanded queries per tool, so that I understand what alternative phrasings were searched.
8. As a user in agent mode, I want to see the hit count and a top-5 preview per tool, so that I can see what was actually found without opening a separate view.
9. As a user, I want expanded queries to stay visible after the search completes (persisted), so that I can review them later instead of missing them in a transient flash.
10. As a user, I want the processing animation to fade stages in and out (routing → tool call → answer), so that I get pleasant real-time feedback on what's happening.
11. As a user, I want the expanded Process window to be compact (small fonts, tight spacing), so that a lot of info doesn't blow up the layout.
12. As a user with the Reranker checkbox ON, I want agentic message searches to be reranked, so that the agent gets the same quality as the non-agent pipeline.
13. As a user with the LLM expansion checkbox ON, I want agentic message searches to expand the query, so that recall matches the non-agent pipeline.
14. As a user with the Reranker / LLM expansion checkboxes OFF, I want agentic searches to skip those steps, so that my toggle choices are respected in both modes.
15. As a user querying a contact, I want `search_contacts` to stay exact-match, so that a name or userid isn't fuzzy-rewritten into a worse search.
16. As a user, I want the final LLM answer to be based only on the clean top-5 results, so that expansion noise never distorts what I'm told.
## Implementation Decisions

- **One expander per turn** named "Process"; label suffix `(agentic · ✅ · N tools · Xms)` when agent mode handled the turn, with `⚠️ fallback` when fallback mode was used. Default `expanded=False`.
- **Agentic turns:** per-tool expandable sub-windows inside Process. Each sub-window shows: tool name, query used, expanded queries (persisted), hit count, top-5 previews. Compact styling (smaller font sizes, reduced margins/padding).
- **Non-agentic turns:** Process window renders the existing raw-hits dataframe (Message ID / Content / Score / Metadata columns), unchanged.
- **No combined raw table in agentic mode** — per-tool info replaces it; the empty-Details-tab redundancy disappears.
- **Delete the dead "Raw results" toggle block** (the `show_raw_toggle_<hash>` render loop); nothing sets that flag.
- **Unified live-activity display:** replace the two `st.status` blocks with one shared component. Stage labels animate fade-in (0.3s) → hold (until the stage actually completes) → fade-out (0.3s). No artificial `time.sleep` beyond what the animation needs; stage completion is event-driven.
- **Agentic tool expansion + rerank:** `search_messages` internally runs: LLM semantic rephrase + keyword expansion → per-query txtai search → weighted RRF fusion (reuse the existing fusion logic) → cross-encoder rerank via the existing `Reranker`. Controlled by the shared `use_rerank` / `expand` toggles passed into the agent. `search_contacts` never expands or reranks.
- **Answer/display separation:** the LLM answer synthesizer receives only the clean top-5 tool results; expanded queries are surfaced only in the Process window.
- **Persistence:** each completed turn stores a `process` payload (label suffix data, per-tool sub-window data: query, expanded queries, hit count, previews) so the Process window survives Streamlit reruns.
- The agent's `tool_calls`/steps recording is extended to carry the expanded queries + hit counts + previews that the UI needs.


## Testing Decisions

- **Good tests assert external behavior**, not implementation: given a query and toggle state, the Process window renders the right content; the agent's observable tool output contains/omits expansion data as toggles dictate.
- **Seam 1 — `CrossTableAgent.process()`** (highest seam; prior art `tests/test_app_search_payload.py`): assert `search_messages` tool output contains expanded queries + hit counts when toggles are ON, and does not when OFF; assert `search_contacts` never expands even with toggles ON. Deterministic via an injected fake expander (prior art: `tests/test_search_expansion.py` `FakeExpander`).
- **Seam 2 — UI rendering (`_render_chat_history`)**, via the recording fake-streamlit pattern (prior art `tests/test_search_ui.py`): assert exactly one expander per turn; assert label `Process` vs `Process (agentic · …)`; assert collapsed-by-default; assert non-agentic shows the table, agentic shows per-tool sub-windows; assert no dead "Raw results" block renders.
- **Seam 3 — tool-level unit test** on `search_messages`: fake expander/reranker → assert it expands + reranks + returns a clean result string; assert `search_contacts` stays exact.
- Prior art: `tests/test_search_ui.py`, `tests/test_app_search_payload.py`, `tests/test_search_expansion.py`, `tests/test_cross_table_agent.py`.

## Out of Scope

- Multi-turn conversation memory (separate agent-layer-enhancements spec).
- Changes to the non-agent Searcher expansion/rerank behavior.
- Any change to the answer-generation prompt logic other than ensuring expansion data is excluded.
- Mobile/theme-specific styling work beyond compact Process-window styling.
- Search-result pagination or infinite scroll in the Process window.

## Further Notes

- The repo's "issue tracker" is the local markdown convention: `.scratch/<feature>/issues/NN-*.md` with `**Status:** ready-for-agent`.
- Reuses the existing `fadeInRight` CSS animation pattern already present in app.py.
- All changes backward-compatible: non-agent mode table output is byte-for-byte the current Details table; agentic mode simply replaces the two windows with one richer window.
