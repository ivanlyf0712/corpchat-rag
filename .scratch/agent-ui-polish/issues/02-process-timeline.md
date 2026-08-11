# Ticket 02: Add process timeline with timing breakdown

**Type:** task
**Status:** ready-for-agent
**Blocked by:** none

## Description
Users cannot see what the cross-table agent is doing step-by-step. The process should be visible both during execution and after the answer is generated.

## Acceptance Criteria
1. CrossTableAgent.process() returns a `steps` list alongside `output`:
   ```python
   {
     "output": "...",
     "steps": [
       {"icon": "⚡", "label": "Intent check", "duration_ms": 0, "detail": "Greeting → quick response"},
       {"icon": "🔍", "label": "search_messages", "duration_ms": 450, "detail": "Query: '合同已签' → 3 hits"},
       {"icon": "👤", "label": "search_contacts", "duration_ms": 380, "detail": "Query: user_陳志明 → found email"},
       {"icon": "✨", "label": "Answer generation", "duration_ms": 50, "detail": "Combined into answer"},
     ]
   }
   ```
2. Each step records: icon, label, duration in ms, detail string
3. In app.py, render steps inside a `st.expander("🤖 Agent Process")` below the answer
4. The expander stays OPEN after the answer is generated (not auto-collapsed)
5. Show a summary line above the expander: "✅ Agent completed in 1.2s · 2 tools called"
6. For fallback mode, show "⚠️ Fallback mode" badge in the summary
7. For hallucination detection (no tools called), show "🔄 Fallback: agent didn't call tools"

## Implementation Notes
- In cross_table_agent.py: wrap each step in timing, append to `steps` list
- In app.py: render the expander with `st.expander("🤖 Agent Process", expanded=True)`
- Use `time.perf_counter()` for accurate timing