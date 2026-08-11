# Ticket 02: Raw Results Toggle (Optional / Low Priority)

## Summary
Add a UI toggle in Streamlit to switch between LLM-generated answers and raw search results table.

## Description
When the agent produces an LLM answer, users may want to inspect the underlying search hits directly. This ticket adds a small toggle/button in the search page to show/hide the raw results dataframe without expanding the Details panel.

## Acceptance Criteria
- A toggle or button labeled "Show raw results" appears after search completes.
- When activated, a dataframe of raw hits is displayed inline.
- When deactivated, only the LLM answer remains visible.
- Toggle state is per-turn (not global).
- Does not affect search behavior or performance.

## Dependencies
- `apps/corpchat/app.py`
- Existing `_docs_to_tuples` helper.

## Priority
Low — deferred until after multi-turn memory is implemented.