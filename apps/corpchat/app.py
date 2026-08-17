#!/usr/bin/env python3
"""
CorpChat Intelligence – Streamlit App
View contacts, messages, statistics, a chat-style conversation viewer, and semantic search.
"""

import sys
import os
import json
import hashlib
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, timezone

# ── Ensure the project root (ocr/) is on the Python path ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.db import get_db_connection
from core.config import OLLAMA_URL, RAG_MODEL

# ── 明確載入 .env，讓 UI 能讀取 LiteLLM 與搜尋設定 ──
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
except ImportError:
    pass

# ── 從 search.py 匯入搜尋模組 ──
from apps.corpchat.search import (
    load_index,
    Searcher,
    DEFAULT_INDEX_PATH,
)

# ── LiteLLM 設定（密鑰必須從環境變數提供，不可硬編碼）──
import os as _os
LITELLM_API_KEY = _os.getenv("LITELLM_API_KEY", "")   # 從環境變數讀取
LITELLM_BASE_URL = _os.getenv("LITELLM_BASE_URL", "https://your-litellm-proxy.example.com")
LITELLM_MODEL = _os.getenv("LITELLM_MODEL", "dseek-v4-flash")

# ── Page config ──
st.set_page_config(
    page_title="CorpChat Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    :root {
        --page-bg: #f5f7fb;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --border: #dbe4ee;
        --text: #1f2937;
        --muted: #6b7280;
        --accent: #2f6fed;
        --accent-soft: rgba(47, 111, 237, 0.10);
        --success-soft: rgba(16, 185, 129, 0.10);
        --warning-soft: rgba(245, 158, 11, 0.10);
    }

    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, var(--page-bg) 100%);
        color: var(--text);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.25rem;
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.02em;
    }

    [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }

    div[data-baseweb="tab-list"] {
        gap: 0.35rem;
        padding: 0.2rem;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--border);
        border-radius: 16px;
    }

    button[data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 0.55rem 0.85rem !important;
        color: var(--muted) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--accent-soft) !important;
        color: var(--accent) !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid var(--border);
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        color: var(--text);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: rgba(47, 111, 237, 0.35);
        box-shadow: 0 8px 20px rgba(47, 111, 237, 0.10);
        transform: translateY(-1px);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, #3778ff 0%, #2f6fed 100%);
        color: white;
        border-color: #2f6fed;
    }

    .stButton > button[kind="secondary"] {
        background: #ffffff;
    }

    .stDataFrame, .stTable {
        border: 1px solid var(--border);
        border-radius: 16px;
        overflow: hidden;
        background: var(--surface);
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }

    .stExpander {
        border: 1px solid var(--border);
        border-radius: 16px;
        background: var(--surface);
    }

    [data-testid="stChatMessage"] {
        border: 1px solid var(--border);
        border-radius: 16px;
        background: var(--surface);
        padding: 0.25rem 0.5rem;
    }

    .stAlert {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## CorpChat Intelligence")
st.markdown("**Corporate relationship and chat analytics**")

st.caption("A clean workspace for contacts, messages, conversations, and search.")

# ═══════════════════════════════════════ helpers ════════════════════════════════════
@st.cache_data(ttl=30)
def fetch_contacts():
    try:
        conn = get_db_connection()
        df = pd.read_sql(
            "SELECT id, full_name, job_title, company, phone, email, userid, created_at FROM contacts ORDER BY created_at DESC",
            conn
        )
        conn.close()
        return df
    except Exception:
        st.warning("Contacts unavailable — is the database running? Start PostgreSQL, then run the data generator.")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_messages():
    try:
        conn = get_db_connection()
        df = pd.read_sql(
            """SELECT id, msgid, open_kfid, external_userid, send_time, origin, 
                      servicer_userid, msgtype, content, label, created_at 
               FROM messages ORDER BY send_time DESC LIMIT 500""",
            conn
        )
        conn.close()
        return df
    except Exception:
        st.warning("Messages unavailable — is the database running? Start PostgreSQL, then run the data generator.")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM contacts")
        total_contacts = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM messages")
        total_msgs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT open_kfid) FROM messages")
        total_convos = cur.fetchone()[0]
        cur.execute("""
            SELECT label, COUNT(*) 
            FROM messages 
            GROUP BY label 
            ORDER BY COUNT(*) DESC
        """)
        label_counts = cur.fetchall()
        cur.close()
        conn.close()
        return total_contacts, total_msgs, total_convos, label_counts
    except Exception:
        st.warning("Statistics unavailable — is the database running? Start PostgreSQL, then run the data generator.")
        return 0, 0, 0, []

# ── Chat viewer helpers ──
def get_contact_name_map():
    """Return a dict {userid: full_name} for all contacts."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT userid, full_name FROM contacts")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {row[0]: (row[1] if row[1] else row[0]) for row in rows}
    except Exception:
        return {}

def get_conversation_list(label_filter=None, search_term=None):
    """Return a list of distinct conversations with last message info."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Build WHERE clause
        conditions = []
        params = []
        if label_filter:
            conditions.append("label = %s")
            params.append(label_filter)
        if search_term:
            conditions.append("(external_userid ILIKE %s OR servicer_userid ILIKE %s)")
            params.extend([f"%{search_term}%", f"%{search_term}%"])

        where = " AND ".join(conditions) if conditions else "TRUE"

        # Get unique conversations (one row per open_kfid)
        cur.execute(f"""
            SELECT open_kfid,
                   MAX(external_userid) AS external_userid,
                   MAX(servicer_userid) AS servicer_userid,
                   MAX(send_time) AS last_time
            FROM messages
            WHERE {where}
            GROUP BY open_kfid
            ORDER BY last_time DESC
            LIMIT 50
        """, params)

        conversations = []
        name_map = get_contact_name_map()
        for row in cur.fetchall():
            kfid, cust, agent, last_time = row
            # Get last message content
            cur.execute("SELECT content FROM messages WHERE open_kfid = %s ORDER BY send_time DESC LIMIT 1", (kfid,))
            last_msg = cur.fetchone()
            snippet = last_msg[0][:50] + "..." if last_msg and len(last_msg[0]) > 50 else (last_msg[0] if last_msg else "")

            display_name = name_map.get(cust, cust)
            conversations.append({
                "open_kfid": kfid,
                "display_name": display_name,
                "snippet": snippet,
                "last_time": last_time,
                "cust": cust,
                "agent": agent
            })

        cur.close()
        conn.close()
        return conversations
    except Exception as e:
        st.warning(f"Conversations unavailable: {e}")
        return []

def get_messages_for_conversation(open_kfid):
    """Fetch all messages for a given conversation, ordered by time."""
    try:
        conn = get_db_connection()
        df = pd.read_sql(
            "SELECT msgid, external_userid, servicer_userid, send_time, origin, content "
            "FROM messages WHERE open_kfid = %s ORDER BY send_time ASC",
            conn, params=(open_kfid,)
        )
        conn.close()
        return df
    except Exception as e:
        st.warning(f"Messages for conversation unavailable: {e}")
        return pd.DataFrame()

# ════════════════════════════════ 搜尋功能（由 search.py 提供）═══════════════════════
@st.cache_resource
def _load_search_index():
    """載入 search.py 建立的索引（含分塊與豐富化），回傳 txtai Embeddings。"""
    try:
        return load_index(DEFAULT_INDEX_PATH)
    except FileNotFoundError:
        st.warning(
            "search_index 索引不存在。請先執行 `python apps/corpchat/search.py build --force` "
            "來建立含分塊與豐富化的搜尋索引。"
        )
        return None

# ── 重用 search.py 的 _clean_text_from_enriched ──
from apps.corpchat.search import _clean_text_from_enriched as _search_clean_text

def search_messages(
    query: str,
    label_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_k: int = 10,
    use_rerank: bool = True,
    expand: bool = True,
    graph_expand: int = 1,
    agentic: bool = False,
):
    """
    使用 search.py 的 Searcher 執行 **完整鏈路** 搜尋。

    預設啟用所有增強功能（最強模式）：
      - LLM 查詢擴展 + 多查詢加權 RRF 融合（expand=True）
      - 圖一跳鄰居擴展（graph_expand=1）
      - 交叉編碼器重排序（use_rerank=True）

    當 agentic=True 時，使用 AgenticDecider 根據查詢內容自動決定
    mode / expand / graph_expand / use_rerank，覆蓋手動傳入的參數。

    Searcher 已修正：
      - metadata 從 enriched text 的 "Metadata: key=value;..." 後綴中反向解析
      - label 過濾和日期過濾使用正確解析的 metadata
      - 內容文字可透過 _clean_text_from_enriched() 擷取乾淨內容

    回傳格式：[(msgid, content, send_time, external_userid, servicer_userid, label, score), ...]
    """
    embeddings = _load_search_index()
    if embeddings is None:
        return []

    # Agentic 決策：讓 AgenticDecider 依查詢內容選擇 mode / expand / graph / rerank
    if agentic:
        from apps.corpchat.search import AgenticDecider
        decision = AgenticDecider().decide(query)
        mode = decision.get("mode", "hybrid")
        expand = decision.get("expand", expand)
        graph_expand = decision.get("graph_expand", graph_expand)
        use_rerank = decision.get("use_rerank", use_rerank)
    else:
        mode = "hybrid"

    query_expander = None
    if expand:
        from apps.corpchat.search import QueryExpander
        query_expander = QueryExpander()

    reranker = None
    if use_rerank:
        from apps.corpchat.search import Reranker
        reranker = Reranker()
    
    searcher = Searcher(embeddings, expander=query_expander, reranker=reranker)
    results = searcher.search(
        query=query,
        mode=mode,
        limit=top_k,
        expand=expand,
        graph_expand=graph_expand,
        label_filter=label_filter,
        date_from=date_from,
        date_to=date_to,
        use_rerank=use_rerank,
    )

    output = []
    for r in results:
        meta = r.get("metadata", {})
        send_time_str = meta.get("send_time")
        if send_time_str:
            try:
                send_time = datetime.fromisoformat(send_time_str)
            except (ValueError, TypeError):
                send_time = None
        else:
            send_time = None
        output.append((
            r.get("id", ""),
            r.get("text", ""),             # enriched text (含标题+内容+Metadata)
            send_time,
            meta.get("external_userid"),
            meta.get("servicer_userid"),
            meta.get("label"),
            r.get("score", 0.0),
        ))
    return output

# ── 使用 LiteLLM 生成答案（替换原 generate_answer_from_messages）──
def generate_answer_litellm(query, messages):
    """使用 LiteLLM API 生成自然语言答案。"""
    if not messages:
        return "No relevant messages found."

    # 构建上下文：取前 3 条消息的内容
    context_parts = []
    for m in messages[:3]:
        content = m[1] or ""
        if content:
            context_parts.append(content[:1000])
    if not context_parts:
        return "No message content available."

    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""You are a corporate intelligence analyst. Based on the message data below,
answer the user's question in one or two sentences. If you cannot answer, say "Insufficient data."

Question: {query}

Relevant messages:
{context}

Answer:"""

    # 呼叫 LiteLLM（OpenAI 相容端點）
    url = f"{LITELLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LITELLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 128,
        "stream": False
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
KNOWN_LABELS = [
    if label_counts:
def render_chat_view():
    st.subheader("Chat")
    st.caption("Select a conversation on the left and read or respond on the right.")
    if "selected_kfid" not in st.session_state:
        st.session_state.selected_kfid = None

    col_left, col_right = st.columns([1, 2], gap="medium")
    with col_left:
        st.markdown("#### Conversations")
        label_filter = st.selectbox("Label", options=["All"] + KNOWN_LABELS, index=0)
        search_term = st.text_input("Find a person", placeholder="name or user ID")
        label_filter = None if label_filter == "All" else label_filter
        conversations = get_conversation_list(label_filter, search_term if search_term else None)
        if not conversations:
            st.info("No conversations match the filters.")
        else:
            for conv in conversations:
                kfid = conv["open_kfid"]
                name = conv["display_name"]
                snippet = conv["snippet"]
                last_time = conv["last_time"]
                now = datetime.now(timezone.utc)
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=timezone.utc)
                diff = now - last_time
                time_str = "today" if diff.days == 0 else "yesterday" if diff.days == 1 else f"{diff.days}d ago"
                if st.button(f"**{name}**  \n{snippet}  \n_{time_str}_", key=f"chat_{kfid}", use_container_width=True):
                    st.session_state.selected_kfid = kfid

    with col_right:
        st.markdown("#### Conversation")
        if st.session_state.selected_kfid is None:
            st.info("Select a conversation from the list to view the chat.")
        else:
            kfid = st.session_state.selected_kfid
            msgs = get_messages_for_conversation(kfid)
            if msgs.empty:
                st.warning("No messages found for this conversation.")
            else:
                name_map = get_contact_name_map()
                with st.container():
                    last_date = None
                    for _, row in msgs.iterrows():
                        msg_date = row["send_time"].date()
                        if msg_date != last_date:
                            st.markdown(f"**{msg_date.strftime('%Y-%m-%d')}**")
                            last_date = msg_date
                        if row["origin"] == 3:
                            sender_name = name_map.get(row["external_userid"], row["external_userid"])
                            role = "user"
                        else:
                            sender_name = name_map.get(row["servicer_userid"], row.get("servicer_userid", "System"))
                            role = "assistant"
                        with st.chat_message(role):
                            st.markdown(f"**{sender_name}**  \n{row['content']}")
                            st.caption(row["send_time"].strftime("%H:%M"))
            with st.form("reply_form"):
                reply_text = st.text_area("Your message", placeholder="Type a response...")
                submitted = st.form_submit_button("Send")
                if submitted:
                    st.success("Reply drafted. Wire this to your sending flow if needed.")


def render_view_data():
    st.subheader("View Data")
    total_contacts, total_msgs, total_convos, label_counts = fetch_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Contacts", total_contacts)
    with col2:
        st.metric("Messages", total_msgs)
    with col3:
        st.metric("Conversations", total_convos)
    st.markdown("#### Contacts")
    df_contacts = fetch_contacts()
    if not df_contacts.empty:
        st.dataframe(df_contacts, width='stretch')
    else:
        st.info("No contacts found. Run the data generator or OCR pipeline to populate contacts.")


def render_messages():
    st.subheader("Messages")
    df_msgs = fetch_messages()
    if not df_msgs.empty:
        origin_map = {3: "Customer", 4: "System", 5: "Agent"}
        if 'origin' in df_msgs.columns:
            df_msgs['origin'] = df_msgs['origin'].map(origin_map)
        st.dataframe(df_msgs, width='stretch')
    else:
        st.info("No messages found. Generate some conversations first.")


def render_upload_data():
    st.subheader("Upload Data")
    uploaded = st.file_uploader("Upload CSV or JSON files", type=["csv", "json"], accept_multiple_files=True)
    st.info("Uploaded files can be connected to your ingestion flow here.")
    if uploaded:
        st.success(f"{len(uploaded)} file(s) selected.")


def render_search():
    st.subheader("Find Conversations")
    st.caption("Use a simple search box and optional filters to narrow down the right conversations.")
    with st.expander("Filters and settings", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3, gap="medium")
        with col_f1:
            lbl_filter = st.selectbox("Topic", options=["Any topic"] + KNOWN_LABELS, index=0, key="search_label")
        with col_f2:
            date_from = st.text_input("From date", placeholder="2024-01-01", key="search_date_from")
        with col_f3:
            date_to = st.text_input("To date", placeholder="2024-12-31", key="search_date_to")
        col_adv1, col_adv2 = st.columns(2, gap="medium")
        with col_adv1:
            use_rerank = st.checkbox("Better matching", value=True, key="search_rerank")
        with col_adv2:
            top_k = st.slider("How many results", 1, 20, 10, key="search_top_k")
        use_expand = st.checkbox("Use smarter search", value=True, key="search_expand")
        graph_expand = st.number_input("Nearby context", min_value=0, max_value=3, value=1, step=1, key="search_graph_expand")
        use_agentic = st.checkbox("Auto-tune search", value=False, key="search_agentic")

    search_query = st.text_input("What are you looking for?", placeholder="e.g. investment offer, invoice issue, delivery update", key="search_input")
    if st.button("Find Conversations", type="primary", use_container_width=True):
        if not search_query:
            st.warning("Please enter a search query.")
        else:
            st.session_state.rag_answer = None
            lbl = None if lbl_filter == "Any topic" else lbl_filter
            from_date = datetime.strptime(date_from, "%Y-%m-%d").isoformat() if date_from else None
            to_date = datetime.strptime(date_to, "%Y-%m-%d").isoformat() if date_to else None
            with st.spinner("Finding the best matches..."):
                try:
                    results = search_messages(
                        search_query, lbl, from_date, to_date, top_k,
                        use_rerank=use_rerank,
                        expand=use_expand,
                        graph_expand=graph_expand,
                        agentic=use_agentic,
                    )
                    st.session_state.search_results = results
                    st.session_state.search_query = search_query
                except Exception as e:
                    st.error(f"Search error: {e}")
                    st.session_state.search_results = None

    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        if results:
            st.success(f"Found {len(results)} result(s)")
            with st.expander("Search details", expanded=False):
                st.markdown("**Details to review:** search terms, applied filters, timestamps, and settings.")
            name_map = get_contact_name_map()
            df = pd.DataFrame(results, columns=[
                "Message ID", "Content", "Send Time", "Customer ID", "Agent ID", "Label", "Similarity"
            ])
            df["Customer"] = df["Customer ID"].apply(lambda x: name_map.get(x, x) if x else "")
            df["Agent"] = df["Agent ID"].apply(lambda x: name_map.get(x, x) if x else "—")
            df["Send Time"] = df["Send Time"].apply(lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "strftime") else str(t))
            df["Similarity"] = df["Similarity"].apply(lambda x: f"{x:.4f}")
            st.dataframe(df[["Message ID", "Customer", "Agent", "Send Time", "Label", "Similarity", "Content"]], width='stretch')
        else:
            st.info("No matching conversations found. Try a different phrase or a wider date range.")


if "app_view" not in st.session_state:
    st.session_state.app_view = "Chat"

with st.sidebar:
    st.markdown("### Navigation")
    st.radio(
        "Choose a section",
        ["Chat", "View Data", "Messages", "Upload Data", "Search"],
        index=0,
        key="app_view",
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("<div class='nav-card'><strong>Chat</strong><br/><small>Primary workspace for conversation review and replies.</small></div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-card'><strong>View Data</strong><br/><small>Contacts, summary metrics, and label distribution.</small></div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-card'><strong>Messages</strong><br/><small>Recent message records in a table view.</small></div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-card'><strong>Upload Data</strong><br/><small>Import conversation files for processing.</small></div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-card'><strong>Search</strong><br/><small>Find the right conversations with simple filters.</small></div>", unsafe_allow_html=True)

if st.session_state.app_view == "Chat":
    render_chat_view()
elif st.session_state.app_view == "View Data":
    render_view_data()
elif st.session_state.app_view == "Messages":
    render_messages()
elif st.session_state.app_view == "Upload Data":
    render_upload_data()
elif st.session_state.app_view == "Search":
    render_search()

st.markdown("---")
st.caption("CorpChat Intelligence – powered by Unlimited‑OCR & RAG Pipeline")
                width='stretch',
                column_config={
                    "Similarity": st.column_config.TextColumn("Similarity", width="small"),
                    "Label": st.column_config.TextColumn("Label", width="small"),
                }
            )

            st.divider()
            if st.button("🤖 Generate Answer", type="secondary"):
                with st.spinner("Generating answer..."):
                    answer = generate_answer_litellm(
                        st.session_state.search_query, results
                    )
                    st.session_state.rag_answer = answer

            if st.session_state.rag_answer:
                st.markdown(f"**Answer:** {st.session_state.rag_answer}")
        else:
            st.info("找不到符合的訊息。請確認索引已建立：`python apps/corpchat/search.py build --force`")

# ──────────── Tab 6: Chat RAG (hidden) ────────────
# with tab6:
#     st.subheader("🤖 Enterprise Chat RAG")
#
#     # To find the agent ID:
#     #   1. Log into the local chat platform at http://localhost:3000
#     #   2. Navigate to Agents → Edit Agent for "Enterprise Chat RAG"
#     #   3. The agent ID is the numeric portion in the URL (e.g., /agents/3 → agentId=3)
#     AGENT_ID = 1  # <-- REPLACE with the actual agent ID from the agent page
#
#     chat_url = f"http://localhost:3000/app?agentId={AGENT_ID}"
#     st.iframe(src=chat_url, height=700)

st.markdown("---")
st.caption("CorpChat Intelligence – powered by Unlimited‑OCR & RAG Pipeline")