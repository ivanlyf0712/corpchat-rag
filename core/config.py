# ──────────────────── Configuration ────────────────────
"""CorpChat configuration.

OCR / invoice / Ollama legacy config moved to `core/invoice_db.py` (see the
folder reorg) so this module only carries what the CorpChat POC needs.
"""
import os

# PostgreSQL
# Credentials come from the environment only (P0 security fix): no hardcoded
# default password. core.db.get_db_connection() fails fast when DB_PASSWORD is unset.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("DB_USER", "ocr"),
    "password": os.getenv("DB_PASSWORD", ""),
    "dbname": os.getenv("DB_NAME", "invoices"),
}
