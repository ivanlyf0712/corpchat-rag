# Ticket 01: Merge Agentic mode + Cross-table agent into single toggle

**Type:** task
**Status:** ready-for-agent
**Blocked by:** none

## Description
Currently there are two separate checkboxes in the Enhancements panel:
- "Agentic mode" — uses AgenticDecider to choose search params (keyword/semantic/hybrid)
- "Cross-table agent" — uses LangGraph ReAct to search messages AND contacts

These are confusing because cross-table agent is strictly more powerful. When cross-table is enabled, it already handles all search modes internally.

## Acceptance Criteria
1. Remove "Agentic mode" checkbox from the UI
2. Rename "Cross-table agent" → "🤖 Agent" (single toggle)
3. When "🤖 Agent" is ON: ALL queries route through CrossTableAgent.process()
   - Greetings → quick response (no tools)
   - Simple name queries → direct search_contacts
   - Simple message queries → direct search_messages
   - Complex cross-table queries → LangGraph ReAct agent
   - If LangGraph fails or hallucinates → fallback (two-step reasoning)
4. When "🤖 Agent" is OFF: original pipeline (Searcher + AgenticDecider)
5. Toggle state persists across reruns via `st.session_state.agent_enabled`
6. No breaking changes to existing non-agent flow

## Implementation Notes
- In app.py: replace two checkboxes with one, read from session_state
- In cross_table_agent.py: add `process()` logic that routes queries intelligently based on complexity
- Keep the existing `_quick_respond()` for greetings
- Add `_route_query()` method that decides: direct contacts, direct messages, or LangGraph