# ──────────────────── CorpChat Persistence Module ────────────────────
"""CorpChat-specific database persistence (agent memory, persona, config, graph).

Candidate 5: split the CorpChat persistence surface out of `core/db.py`, which
also hosts the legacy invoice-OCR functions. This module changes for CorpChat
reasons only; `core/db.py` re-exports these names for backward compatibility.

Connection access goes through `core.db.get_db_connection` *at call time*
(`from core import db as _db`), so tests that monkeypatch
`core.db.get_db_connection` keep working unchanged.
"""

from core import db as _db


def _conn():
    return _db.get_db_connection()


# ── Agent multi-turn memory ──────────────────────────────────────────
def init_agent_memory_table():
    """Create agent_memory table if it doesn't exist."""
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_memory (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                turn_number INTEGER NOT NULL,
                user_message TEXT NOT NULL,
                bot_message TEXT NOT NULL,
                intent VARCHAR(32),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_memory_session
            ON agent_memory(session_id, turn_number)
        """)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def load_agent_memory(session_id: str, max_turns: int = 10):
    """Load recent turns for a session from agent_memory."""
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT turn_number, user_message, bot_message, intent
            FROM agent_memory
            WHERE session_id = %s
            ORDER BY turn_number DESC
            LIMIT %s
        """, (session_id, max_turns))
        rows = cur.fetchall()
        # Reverse to chronological order
        return [{"user": r[1], "bot": r[2], "intent": r[3]} for r in reversed(rows)]
    except Exception:
        return []
    finally:
        cur.close()
        conn.close()


def save_agent_memory(session_id: str, turn_number: int, user_message: str, bot_message: str, intent: str):
    """Append a turn to agent_memory."""
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO agent_memory (session_id, turn_number, user_message, bot_message, intent)
            VALUES (%s, %s, %s, %s, %s)
        """, (session_id, turn_number, user_message, bot_message, intent))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# ── Persona disposition profile persistence ─────────────────────────
def init_disposition_profiles_table():
    """Create disposition_profiles table if it doesn't exist."""
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS disposition_profiles (
                session_id VARCHAR(64) PRIMARY KEY,
                skepticism REAL DEFAULT 0.5,
                literality REAL DEFAULT 0.5,
                empathy REAL DEFAULT 0.5,
                style VARCHAR(16) DEFAULT 'balanced',
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def load_disposition_profile(session_id: str):
    """Load a disposition profile for a session, or None if not set.

    Returns a dict with keys skepticism / literality / empathy / style.
    """
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT skepticism, literality, empathy, style
            FROM disposition_profiles
            WHERE session_id = %s
        """, (session_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "skepticism": float(row[0]),
            "literality": float(row[1]),
            "empathy": float(row[2]),
            "style": row[3],
        }
    except Exception:
        return None
    finally:
        cur.close()
        conn.close()


def save_disposition_profile(session_id: str, profile: dict):
    """Upsert a disposition profile for a session."""
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO disposition_profiles (session_id, skepticism, literality, empathy, style)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
                skepticism = EXCLUDED.skepticism,
                literality = EXCLUDED.literality,
                empathy = EXCLUDED.empathy,
                style = EXCLUDED.style,
                updated_at = NOW()
        """, (
            session_id,
            float(profile.get("skepticism", 0.5)),
            float(profile.get("literality", 0.5)),
            float(profile.get("empathy", 0.5)),
            str(profile.get("style", "balanced")),
        ))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# ── Unified agent config persistence ───────────────────────────────
def init_agent_config_table():
    """Create agent_config table if it doesn't exist."""
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_config (
                session_id VARCHAR(64) PRIMARY KEY,
                config TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def load_agent_config(session_id: str):
    """Load the unified agent config JSON for a session, or None if unset."""
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT config FROM agent_config WHERE session_id = %s", (session_id,))
        row = cur.fetchone()
        if not row or not row[0]:
            return None
        import json
        return json.loads(row[0])
    except Exception:
        return None
    finally:
        cur.close()
        conn.close()


def save_agent_config(session_id: str, config: dict):
    """Upsert the unified agent config JSON for a session."""
    import json
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO agent_config (session_id, config)
            VALUES (%s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
                config = EXCLUDED.config,
                updated_at = NOW()
        """, (session_id, json.dumps(config, ensure_ascii=False)))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# ── Hindsight memory graph persistence ────────────────────────────
def init_memory_graph_table():
    """Create memory_graph table if it doesn't exist."""
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_graph (
                session_id VARCHAR(64) PRIMARY KEY,
                graph TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def load_memory_graph(session_id: str):
    """Load the memory graph JSON for a session, or None if unset."""
    import json
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT graph FROM memory_graph WHERE session_id = %s", (session_id,))
        row = cur.fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])
    except Exception:
        return None
    finally:
        cur.close()
        conn.close()


def save_memory_graph(session_id: str, graph: dict):
    """Upsert the memory graph JSON for a session."""
    import json
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO memory_graph (session_id, graph)
            VALUES (%s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
                graph = EXCLUDED.graph,
                updated_at = NOW()
        """, (session_id, json.dumps(graph, ensure_ascii=False)))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

