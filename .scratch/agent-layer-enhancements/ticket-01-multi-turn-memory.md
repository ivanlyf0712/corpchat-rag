# Ticket 01: Multi-turn Memory (DB-backed)

## Summary
Add persistent conversation memory to the Agent so context survives page refreshes in Streamlit.

## Description
Currently `Agent` keeps history only in `self.chat_history` (in-memory). When the Streamlit app reruns, the agent forgets prior turns. This ticket adds a PostgreSQL-backed memory layer with automatic fallback to in-memory when the DB is unreachable.

## Acceptance Criteria
- `agent_memory` table exists with columns: `id`, `session_id`, `turn_number`, `user_message`, `bot_message`, `intent`, `created_at`.
- `Agent.process()` loads the last `max_history` turns for the current `session_id` before classifying/routing.
- After generating a response, the turn is appended to `agent_memory`.
- Streamlit generates/uses a UUID `session_id` stored in `st.session_state`.
- CLI remains stateless unless `session_id` is explicitly passed.
- If PostgreSQL is unreachable, the agent silently falls back to in-memory history (no crash).

## Dependencies
- `core/db.py`
- `apps/corpchat/agent.py`

## Priority
Medium — last ticket per grill session notes.