# Agent Layer Enhancements — Spec

## Goal
Enhance the existing `apps/corpchat/agent.py` intent-routing layer with two user-facing features:
1. **Expanded-query visibility** in Streamlit UI with animated reveal.
2. **Multi-turn conversation memory** persisted to the database.

## Scope
- **In scope:** Streamlit UI changes, DB schema additions, agent memory module.
- **Out of scope:** CLI changes for expansion display; raw-results toggle is deferred to a future ticket.

## Requirements

### 1. Expanded-query visibility (streaming animation)
When LLM query expansion is enabled and the model returns alternative phrasings:
- Display each expanded query one-by-one with a right-to-left slide-in + fade animation.
- After animation completes, keep the queries visible inside the `st.status` progress window.
- If expansion fails or LLM is unavailable, show a warning and continue without expansion.

**Acceptance criteria:**
- User sees expanded queries appear sequentially (~100ms apart) during "Query expansion" stage.
- Queries remain visible after the animation ends.
- No regression in search behavior when expansion is disabled.

### 2. Multi-turn memory (DB-backed)
Allow the agent to recall prior turns across separate user sessions by persisting chat history.

**Schema additions:**
```sql
CREATE TABLE IF NOT EXISTS agent_memory (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    turn_number INTEGER NOT NULL,
    user_message TEXT NOT NULL,
    bot_message TEXT NOT NULL,
    intent VARCHAR(32),
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_agent_memory_session ON agent_memory(session_id, turn_number);
```

**Behavior:**
- On `Agent.process()`, load the last N turns for the current `session_id`.
- After generating a response, append the turn to `agent_memory`.
- `session_id` is a UUID stored in `st.session_state` (Streamlit) or generated per CLI invocation.
- `max_history` controls how many recent turns are loaded into context.
- If DB is unavailable, fall back to in-memory history (current behavior).

**Acceptance criteria:**
- Refreshing the Streamlit page preserves conversation context.
- CLI sessions are stateless unless `session_id` is explicitly provided.
- Graceful degradation when PostgreSQL is unreachable.

## Non-goals
- Real-time collaborative memory across multiple users.
- Long-term memory pruning or archival beyond `max_history`.

## Dependencies
- `core/db.py` for PostgreSQL connection.
- Existing `Agent` class in `apps/corpchat/agent.py`.
- Existing Streamlit session state mechanism.