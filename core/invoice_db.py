# ──────────────────── Invoice/OCR Legacy Module ────────────────────
"""OCR + invoice persistence — legacy code kept out of the CorpChat core.

This module exists so `core/db.py` and `core/config.py` stay focused on the
CorpChat POC. It is not imported by the app or the search package; it is kept
for any external tooling that still talks to the `invoices` table.

Config + functions originally lived in core/config.py / core/db.py /
core/embedding.py. The embedding helpers were duplicated across those files;
the canonical copies live here.
"""

import os
import warnings

import pandas as pd
import psycopg2
import requests

from core.config import DB_CONFIG

warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")


# ── OCR config (server + CLI modes) ─────────────────────────────────
OCR_MODE = "server"               # "server" | "cli"
OCR_SERVER_URL = "http://127.0.0.1:8081/v1/chat/completions"
OCR_SERVER_MODEL = "Unlimited-OCR"
OCR_SERVER_PROMPT = "Please OCR the text in this image."
OCR_SERVER_TEMPERATURE = 0.0
OCR_SERVER_MAX_TOKENS = 32768
OCR_SERVER_REPEAT_PENALTY = 1.1

LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-mtmd-cli")
UOCR_MODEL = os.path.expanduser("~/uocr/Unlimited-OCR-Q4_K_M.gguf")
UOCR_MMPROJ = os.path.expanduser("~/uocr/mmproj-Unlimited-OCR-F16.gguf")

MAX_LONG_EDGE = 512              # server can handle larger images
JPEG_QUALITY = 60

OLLAMA_URL = "http://127.0.0.1:11434"
TEXT_MODEL = "qwen2.5:1.5b"       # JSON extraction model
EMBED_MODEL = "mxbai-embed-large"
RAG_MODEL = "qwen2.5:1.5b"        # RAG answer generation model

LLAMA_SERVER_URL = "http://127.0.0.1:8081/v1/chat/completions"

JSON_PROMPT = """Return a single JSON object with these keys:
"invoice_number", "date", "vendor_name", "total_amount", "currency".

Rules:
- Use the exact text from the invoice. Do NOT invent or guess any values.
- If a field is missing, set it to "".
- "total_amount" must contain only the number (e.g. "1250.00"), without currency symbol.
- "currency" must be the three-letter currency code (e.g. "USD").
- Do NOT use nested objects.

Invoice text:
___RAW_TEXT___

JSON:"""

FALLBACK_PROMPT = """Extract these fields from the invoice text.
Do NOT use any of the following words: value, text, string, example, placeholder, xxxx.
Return ONLY a valid JSON object with the keys:
"invoice_number", "date", "vendor_name", "total_amount", "currency".
"total_amount" must be a plain number (e.g. "1250.00").
"currency" must be a three-letter code (e.g. "USD").
If a field is truly missing, leave it as "".

Invoice text:
___RAW_TEXT___

JSON:"""

# ── Invoice persistence ─────────────────────────────────────────────
def insert_invoice(fields: dict, raw_text: str, source_file: str):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO invoices (invoice_number, date, vendor_name,
                                  total_amount, currency, raw_text, source_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            fields.get("invoice_number", ""),
            fields.get("date", ""),
            fields.get("vendor_name", ""),
            fields.get("total_amount", ""),
            fields.get("currency", ""),
            raw_text,
            source_file
        ))
        new_id = cur.fetchone()[0]
        conn.commit()
        return new_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def fetch_all_invoices():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("SELECT id, invoice_number, date, vendor_name, total_amount, currency, source_file, created_at FROM invoices ORDER BY created_at DESC", conn)
    conn.close()
    return df


# ── Embeddings (mxbai-embed-large via Ollama) ───────────────────────
def get_embedding(text: str):
    """Generate embedding via Ollama mxbai-embed-large (1024-dim)."""
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": EMBED_MODEL, "input": text
    })
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def update_embedding(row_id: int):
    """Fetch raw_text for a row, generate its embedding, and update the row."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT raw_text FROM invoices
        WHERE id = %s AND embedding IS NULL
    """, (row_id,))
    row = cur.fetchone()
    if row and row[0]:
        raw_text = row[0].strip()
        if raw_text:
            text_to_embed = raw_text[:4096]
            vec = get_embedding(text_to_embed)
            cur.execute("UPDATE invoices SET embedding = %s WHERE id = %s", (vec, row_id))
            conn.commit()
    cur.close()
    conn.close()



# ── Hybrid search over invoices ─────────────────────────────────────
def search_similar(query, vendor_filter=None, top_k=5,
                   date_from=None, date_to=None, amount_min=None, amount_max=None,
                   keyword_filter=None):
    """
    Hybrid semantic + keyword search over invoices using pgvector cosine similarity
    and PostgreSQL full-text search on raw_text.

    Args:
        query: Natural language search query (embedded via mxbai-embed-large).
        vendor_filter: Optional ILIKE pattern for vendor name.
        top_k: Number of results to return (1-100).
        date_from: Optional start date filter (YYYY-MM-DD).
        date_to: Optional end date filter (YYYY-MM-DD).
        amount_min: Optional minimum total_amount filter.
        amount_max: Optional maximum total_amount filter.
        keyword_filter: Optional keyword/phrase for full-text search on raw_text.

    Returns:
        List of tuples: (id, invoice_number, date, vendor_name, total_amount, currency, similarity)
    """
    query_vec = get_embedding(query)
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    conditions = ["embedding IS NOT NULL"]
    params: list = [query_vec]

    if vendor_filter:
        conditions.append("vendor_name ILIKE %s")
        params.append(f"%{vendor_filter}%")

    if date_from:
        conditions.append("date >= %s")
        params.append(date_from)

    if date_to:
        conditions.append("date <= %s")
        params.append(date_to)

    if amount_min:
        conditions.append("total_amount::numeric >= %s")
        params.append(amount_min)

    if amount_max:
        conditions.append("total_amount::numeric <= %s")
        params.append(amount_max)

    if keyword_filter:
        conditions.append(
            "to_tsvector('english', raw_text) @@ plainto_tsquery('english', %s)"
        )
        params.append(keyword_filter)

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT id, invoice_number, date, vendor_name, total_amount, currency,
               1 - (embedding <=> %s::vector) AS similarity
        FROM invoices
        WHERE {where_clause}
        ORDER BY similarity DESC
        LIMIT %s
    """
    params.append(top_k)

    cur.execute(sql, params)
    results = cur.fetchall()
    cur.close()
    conn.close()

    return results if results else []
