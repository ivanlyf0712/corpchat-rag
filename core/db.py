# ──────────────────── Database Module ────────────────────
"""CorpChat database layer.

- `get_db_connection()` is the shared Postgres connection entry point.
- CorpChat persistence lives in `core/corpchat_db.py` (re-exported here for
  backward compatibility).
- The legacy invoice/OCR persistence moved to `core/invoice_db.py` (folder reorg).

Credentials come from the environment only (P0 security fix): no hardcoded
default password; `get_db_connection` fails fast when DB_PASSWORD is unset.
"""
import psycopg2

from core.config import DB_CONFIG


def get_db_connection():
    """Open a PostgreSQL connection.

    Fail-fast when DB_PASSWORD is missing (P0 security fix): the credential
    must come from the environment, never from a committed default. Compose
    passes DB_PASSWORD; local runs must set it in .env.
    """
    if not DB_CONFIG.get("password"):
        raise RuntimeError(
            "DB_PASSWORD is not set — add it to .env (or set DB_PASSWORD) before "
            "connecting to PostgreSQL."
        )
    return psycopg2.connect(**DB_CONFIG)


# ── CorpChat persistence (moved to core/corpchat_db.py) ─────────────
# Re-exported for backward compatibility; new code should import from
# `core.corpchat_db` directly.
from core.corpchat_db import (  # noqa: F401
    init_agent_memory_table,
    load_agent_memory,
    save_agent_memory,
    init_disposition_profiles_table,
    load_disposition_profile,
    save_disposition_profile,
    init_agent_config_table,
    load_agent_config,
    save_agent_config,
    init_memory_graph_table,
    load_memory_graph,
    save_memory_graph,
)
