#!/usr/bin/env python3
"""
CorpChat Intelligence – Streamlit App
View contacts, messages, statistics, a chat-style conversation viewer, and semantic search (chatbox).
"""

import sys
import os
import json
import hashlib
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from typing import Any, Optional

# ── Ensure the project root is on the Python path ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.db import get_db_connection

# ── Load .env explicitly so the UI can reach LiteLLM & search config ──
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
except ImportError:
    pass

# ── Import search from search.py ──
from apps.corpchat.search import (
    load_index,
    Searcher,
    DEFAULT_INDEX_PATH,
    LiteLLMClient,
    SearchRouter,
)

# ── Shared Process-window rendering helpers (kept out of app.py) ──
from apps.corpchat import process_window as _process_window

# ── Agentic intent gate (greeting/system_info skip search) ──
from apps.corpchat.agent import (
    Agent,
    IntentClassifier,
    INTENT_GREETING,
    INTENT_SYSTEM_INFO,
    INTENT_CLARIFY,
)

# Module-level intent classifier (cached across renders — deterministic")
_intent_classifier = IntentClassifier()

# ── Initialize DB-backed agent memory table ──
try:
    from core.db import init_agent_memory_table
    init_agent_memory_table()
except Exception:
    pass

# ── Initialize DB-backed disposition profiles table (persona) ──
try:
    from core.db import init_disposition_profiles_table
    init_disposition_profiles_table()
except Exception:
    pass

# ── LiteLLM config ──
import os as _os
LITELLM_API_KEY = _os.getenv("LITELLM_API_KEY", "")
LITELLM_BASE_URL = _os.getenv("LITELLM_BASE_URL", "https://your-litellm-proxy.example.com")
LITELLM_MODEL = _os.getenv("LITELLM_MODEL", "dseek-v4-flash")

# ── Shared LiteLLM client (single instance for all API calls) ──
_llm_client = LiteLLMClient(
    api_base=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    model=LITELLM_MODEL,
)

# ── LLM Router: decides whether to search or chat ──
_search_router = SearchRouter(api_base=LITELLM_BASE_URL, api_key=LITELLM_API_KEY, model=LITELLM_MODEL)

# ═══════════════════════════════════════ page config ════════════════════════════════════
st.set_page_config(
    page_title="CorpChat Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════ professional CSS ════════════════════════════════════
st.markdown("""
<style>
.stApp { background: #0e1117; color: #e6e6e6; font-family: 'Inter','Segoe UI',system-ui,sans-serif; }
.stApp .stMarkdown p, .stApp .stMarkdown li { color: #c9d1d9; }
section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
section[data-testid="stSidebar"] .stRadio > label { color: #58a6ff; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
h1, h2, h3 { color: #f0f6fc !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h1 { border-bottom: 2px solid #30363d; padding-bottom: 0.3em; }
.stChatMessage [data-testid="stChatMessageContent"] { border-radius: 8px; padding: 12px 16px; }
.stDataFrame { border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }
.streamlit-expander { border: 1px solid #30363d; border-radius: 6px; background: #161b22; }
[data-testid="stMetric"] { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
.stChatInput { border-top: 1px solid #30363d; }
@keyframes stageFadeIn { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; transform: none; } }
@keyframes stageFadeOut { from { opacity: 1; } to { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════ sidebar navigation ════════════════════════════════════
with st.sidebar:
    st.markdown("## CorpChat Intelligence")
    st.caption("Corporate Relationship & Chat Analytics")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Search", "Contacts", "Messages", "Overview", "Chat Viewer"],
        index=0,
    )

# ═══════════════════════════════════════ DB helpers ════════════════════════════════════
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
        st.warning("Contacts unavailable — is the database running?")
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
        st.warning("Messages unavailable — is the database running?")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM contacts")
        contacts = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM messages")
        messages = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT open_kfid) FROM messages")
        conversations = cur.fetchone()[0]
        cur.execute("SELECT label, COUNT(*) FROM messages GROUP BY label ORDER BY COUNT(*) DESC")
        labels = cur.fetchall()
        conn.close()
        return contacts, messages, conversations, labels
    except Exception:
        st.warning("Stats unavailable — is the database running?")
        return 0, 0, 0, []

@st.cache_data(ttl=30)
def fetch_conversation(open_kfid):
    try:
        conn = get_db_connection()
        df = pd.read_sql(
            """SELECT id, msgid, external_userid, send_time, origin,
                      servicer_userid, msgtype, content, label, created_at
               FROM messages
               WHERE open_kfid = %s
               ORDER BY send_time ASC""",
            conn,
            params=(open_kfid,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_conversations_for_contact(userid):
    """Fetch all messages involving a contact (as sender or receiver)."""
    try:
        conn = get_db_connection()
        df = pd.read_sql(
            """SELECT id, msgid, open_kfid, external_userid, send_time, origin,
                      servicer_userid, msgtype, content, label, created_at
               FROM messages
               WHERE external_userid = %s OR servicer_userid = %s
               ORDER BY send_time ASC""",
            conn,
            params=(userid, userid)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_contact_name_map():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT userid, full_name FROM contacts")
        rows = cur.fetchall()
        conn.close()
        return {userid: name for userid, name in rows}
    except Exception:
        return {}

# ── Backward-compatible helpers used by tests ──
def get_contact_name_map():
    return fetch_contact_name_map()

def get_conversation_list():
    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT DISTINCT open_kfid FROM messages ORDER BY open_kfid", conn)
        conn.close()
        return df["open_kfid"].tolist()
    except Exception:
        return []

def get_messages_for_conversation(open_kfid: str):
    return fetch_conversation(open_kfid)

# ═══════════════════════════════════ backward-compat seam ════════════════════════════════════
def _docs_to_tuples(docs: list) -> list:
    """Convert Searcher result dicts to the tuple format used by callers."""
    tuples = []
    for doc in docs:
        meta = doc.get("metadata", {})
        tuples.append((
            doc.get("id", ""),
            doc.get("text", ""),
            doc.get("score", 0.0),
            meta.get("customer_name", ""),
            meta.get("company", ""),
            meta.get("label", ""),
        ))
    return tuples

def search_messages(
    query: str,
    top_k: int = 5,
    use_rerank: bool = True,
    expand: bool = True,
    graph_expand: int = 1,
    label_filter: str = "",
    agentic: bool = False,
    graph_parallel: bool = False,
):
    """Backward-compatible search seam used by tests and any external caller.

    Constructs QueryExpander / Reranker / AgenticDecider as needed so the
    wiring tests can verify they are instantiated.
    """
    from apps.corpchat.search import QueryExpander, Reranker, AgenticDecider

    if agentic:
        decider = AgenticDecider()
        decision = decider.decide(query)
        mode = decision.get("mode", "hybrid")
        expand = decision.get("expand", expand)
        graph_expand = decision.get("graph_expand", graph_expand)
        use_rerank = decision.get("use_rerank", use_rerank)
        graph_parallel = decision.get("graph_parallel", graph_parallel)
    else:
        mode = "hybrid"

    expander = QueryExpander() if expand else None
    reranker = Reranker() if use_rerank else None

    try:
        embeddings = _load_search_index()
        searcher = Searcher(embeddings, expander=expander, reranker=reranker)
        raw_results = searcher.search(
            query,
            limit=top_k,
            mode=mode,
            use_rerank=use_rerank,
            expand=expand,
            graph_expand=graph_expand,
            label_filter=label_filter or None,
            graph_parallel=graph_parallel,
        )
        return _docs_to_tuples(raw_results)
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

# ═══════════════════════════════════ LiteLLM helper ════════════════════════════════════
def generate_answer_litellm(query: str, context: str, profile: Optional[Any] = None) -> str:
    if not LITELLM_API_KEY:
        return "LiteLLM API key not configured. Set LITELLM_API_KEY in .env."
    system_content = (
        "You are a helpful assistant answering questions based on retrieved chat messages. "
        "Answer concisely in the same language as the query. "
        "If the context doesn't contain the answer, say so."
    )
    if profile is not None:
        system_content = profile.build_system_prompt(system_content)
    messages = [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        }
    ]
    result = _llm_client.chat(messages, temperature=0.3, max_tokens=300, timeout=15)
    if result:
        return result
    return "Error generating answer: LLM call failed."

# ═══════════════════════════════════════ search logic ════════════════════════════════════
@st.cache_resource
def _load_search_index():
    return load_index(DEFAULT_INDEX_PATH)

@st.cache_data(ttl=30)
def _run_search(query: str, top_k: int, use_rerank: bool, expand: bool, graph_expand: int,
                agentic: bool, label_filter: str, graph_parallel: bool = False):
    if not query.strip():
        return [], []
    try:
        embeddings = _load_search_index()
        searcher = Searcher(embeddings)
        mode = "auto" if agentic else "hybrid"
        raw_results = searcher.search(
            query,
            limit=top_k,
            mode=mode,
            use_rerank=use_rerank,
            expand=expand,
            graph_expand=graph_expand,
            label_filter=label_filter or None,
            graph_parallel=graph_parallel,
        )
        return _docs_to_tuples(raw_results), raw_results
    except Exception as e:
        st.error(f"Search failed: {e}")
        return [], []

def _check_llm_available() -> bool:
    """Quick check if LiteLLM is reachable."""
    return _llm_client.is_available(timeout=5)


def _load_persona_profile():
    """Load the current session's DispositionProfile (persona) or None (neutral)."""
    try:
        from apps.corpchat.search.persona import DispositionProfile
        data = st.session_state.get("persona")
        if data:
            return DispositionProfile.from_dict(data)
    except Exception:
        pass
    return None


def _persist_persona():
    """Persist the session persona to the DB (best-effort, non-fatal)."""
    try:
        from core.db import save_disposition_profile
        data = st.session_state.get("persona")
        if data:
            save_disposition_profile(st.session_state.get("session_id", "default"), data)
    except Exception:
        pass

def _build_agent_process_payload(tool_calls: list, steps: list, turn: dict) -> dict:
    """Backward-compatible wrapper for the Process-window payload builder."""
    return _process_window.build_agent_process_payload(tool_calls, turn)


def _stage_html(label: str, detail: str = "") -> str:
    """Backward-compatible wrapper for stage fade-in HTML."""
    return _process_window.stage_html(label, detail)


def _fade_out_html(label: str) -> str:
    """Backward-compatible wrapper for stage fade-out HTML."""
    return _process_window.fade_out_html(label)


def _animate_stage(slot, label: str, detail: str = ""):
    """Backward-compatible wrapper for animating a stage into `slot`."""
    return _process_window.animate_stage(slot, label, detail)


def _complete_stage(slot, label: str):
    """Backward-compatible wrapper for fading a stage out of `slot`."""
    return _process_window.complete_stage(slot, label)


def _render_chat_history(history: list):
    """Backward-compatible wrapper for rendering the chat history."""
    return _process_window.render_chat_history(history, st, pd)

# ═══════════════════════════════════════ pages ════════════════════════════════════
def _render_search_page():
    """Render the Search page (kept callable so tests can drive it)."""
    st.title("Search")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "searching" not in st.session_state:
        st.session_state.searching = False

    # Check if there's a pending processing turn (from query submission)
    pending_turn = None
    for turn in st.session_state.chat_history:
        if turn.get("status") == "processing":
            pending_turn = turn
            break

    # Two-column layout: chat (wide) + controls (narrow right panel)
    chat_col, ctrl_col = st.columns([3, 1])

    with ctrl_col:
        with st.expander("Enhancements", expanded=False):
            use_rerank = st.checkbox("Reranker", value=True, help="Cross-encoder reranking", disabled=st.session_state.searching)
            expand = st.checkbox("LLM expansion", value=True, help="Expand query via LiteLLM", disabled=st.session_state.searching)
            graph_expand = st.slider("Graph hops", min_value=0, max_value=3, value=1, disabled=st.session_state.searching)
            graph_parallel = st.checkbox(
                "Graph path", value=False,
                help="Graph traversal as a fusion path (relationship queries)",
                disabled=st.session_state.searching,
            )
            # Persist agent toggle across reruns (default: activated)
            if "agent_enabled" not in st.session_state:
                st.session_state.agent_enabled = True

            agent_enabled = st.checkbox(
                "🤖 Agent",
                value=st.session_state.agent_enabled,
                key="agent_cb",
                help="Unified agent: cross-table reasoning + smart search",
                disabled=st.session_state.searching,
            )
            st.session_state.agent_enabled = agent_enabled
        with st.expander("Filters", expanded=False):
            label_filter = st.text_input("Label filter", value="", help="e.g. quotation_request", disabled=st.session_state.searching)
            top_k = st.slider("Top-k", min_value=1, max_value=20, value=5, disabled=st.session_state.searching)
        with st.expander("Persona", expanded=False):
            persona_skepticism = st.slider("懷疑度", 0, 10, 5, help="對證據不足的結論標註不確定性", disabled=st.session_state.searching)
            persona_literality = st.slider("字面性", 0, 10, 5, help="嚴格依據檢索原文回答", disabled=st.session_state.searching)
            persona_empathy = st.slider("共情度", 0, 10, 5, help="先回應情緒再給信息", disabled=st.session_state.searching)
            persona_style = st.selectbox("風格", ["balanced", "concise", "detailed"], disabled=st.session_state.searching)
            st.session_state.persona = {
                "skepticism": persona_skepticism / 10.0,
                "literality": persona_literality / 10.0,
                "empathy": persona_empathy / 10.0,
                "style": persona_style,
            }

    with chat_col:
        _render_chat_history(st.session_state.chat_history)

        # If there's a pending processing turn, handle it now
        if pending_turn:
            query = pending_turn["query"]

            # ── Intent gate: greeting/system_info/clarify skip search ──
            # Use Agent.process() to leverage LLM-generated greetings and DB-backed memory
            if "agent" not in st.session_state:
                try:
                    from apps.corpchat.agent import Agent, load_agent
                    try:
                        st.session_state.agent = load_agent()
                    except FileNotFoundError:
                        # No search index built yet — create agent without searcher
                        # It can still handle greetings and system_info via LLM
                        st.session_state.agent = Agent()
                except Exception:
                    st.session_state.agent = None

            agent = st.session_state.agent
            if agent is None:
                # Fallback: if agent fails to load, still try LLM for natural responses
                intent = _intent_classifier.classify(query)
                if intent == INTENT_GREETING:
                    if _check_llm_available():
                        chat_reply = _llm_client.chat([
                            {"role": "system", "content": "You are a friendly assistant. Reply to greetings naturally and warmly in the same language as the user. Keep it short. Do NOT mention you are an AI."},
                            {"role": "user", "content": query},
                        ], temperature=0.7, max_tokens=60, timeout=5)
                        answer = chat_reply or "Hello! How can I help you today?"
                    else:
                        answer = "Hello! I'm CorpChat Intelligence. How can I help you today?"
                elif intent == INTENT_SYSTEM_INFO:
                    answer = (
                        "I'm **CorpChat Intelligence** — an AI-powered search assistant for corporate chat messages.\n\n"
                        "**Capabilities:**\n"
                        "- Semantic hybrid search (BM25 + vector embeddings)\n"
                        "- LLM query expansion for better recall\n"
                        "- Graph-enhanced retrieval (traverses conversation relationships)\n"
                        "- Cross-encoder reranking for result relevance\n\n"
                        "**Data scope:**\n"
                        "- Indexed corporate WeCom conversations\n"
                        "- Topics: business inquiries, quotations, investments, logistics, tech support,\n"
                        "  invoices, contracts, quality issues, scam/phishing detection\n"
                        "- Bilingual (Chinese + English) messages\n\n"
                        "**Limitations:**\n"
                        "- Can only search within the indexed message corpus\n"
                        "- Cannot access external data or real-time feeds\n"
                        "- LLM-dependent features (query expansion, agentic mode) degrade gracefully\n"
                        "  when the LLM endpoint is unavailable\n\n"
                        "How can I help you today?"
                    )
                elif intent == INTENT_CLARIFY:
                    answer = (
                        "I'd be happy to clarify! Could you rephrase your question or "
                        "provide more specific details about what you're looking for?"
                    )
                else:
                    answer = (
                        "I'd be happy to help! Could you rephrase your question or "
                        "provide more specific details about what you're looking for?"
                    )
                with st.chat_message("assistant"):
                    st.markdown(answer)
                pending_turn["answer"] = answer
                pending_turn["raw_hits"] = []
                pending_turn["status"] = "done"
                st.session_state.searching = False
                st.rerun()
                return

            # Ensure session_id persists across reruns
            if "session_id" not in st.session_state:
                st.session_state.session_id = agent.session_id
            else:
                agent.session_id = st.session_state.session_id

            # ── Persona: persist & load the tuned disposition profile ──
            _persist_persona()
            profile = _load_persona_profile()

            # ── Unified Agent mode ──
            if agent_enabled:
                with st.chat_message("assistant"):
                    with st.status("🤖 Agent processing...", expanded=True) as status:
                        slot = st.empty()
                        stage_labels = []

                        def _on_stage(label, detail=""):
                            stage_labels.append(label)
                            _animate_stage(slot, label, detail)

                        st.write("🤖 Unified agent routing query...")
                        try:
                            from apps.corpchat.search import CrossTableAgent
                            ct_agent = CrossTableAgent(
                                api_base=LITELLM_BASE_URL,
                                api_key=LITELLM_API_KEY,
                                model=LITELLM_MODEL,
                                expand=expand,
                                use_rerank=use_rerank,
                                graph_parallel=graph_parallel,
                                profile=profile,
                            )
                            result = ct_agent.process(query, on_stage=_on_stage)
                            answer = result["output"]
                            tool_calls = result.get("tool_calls", [])
                            steps = result.get("steps", [])
                            for tc in tool_calls:
                                st.write(f"   • {tc.get('tool', '?')}: {str(tc.get('tool_input', ''))[:60]}")
                            if result.get("fallback"):
                                st.write("   ⚠️ Used fallback mode")
                        except Exception as e:
                            st.write(f"   ⚠️ Agent failed: {e}")
                            answer = f"Agent error: {e}"
                            steps = []
                        _complete_stage(slot, stage_labels[-1] if stage_labels else "done")
                        status.update(label="Agent complete", state="complete")
                pending_turn["answer"] = answer
                pending_turn["raw_hits"] = []
                pending_turn["status"] = "done"
                pending_turn["agent_steps"] = steps
                pending_turn["agent_fallback"] = result.get("fallback", False) if 'result' in dir() else False
                # Persist per-tool process payload for the Process window
                pending_turn["process"] = _build_agent_process_payload(tool_calls, steps, pending_turn)
                st.session_state.searching = False
                st.rerun()
                return

            # ── Original pipeline (non-agent mode) ──
            intent, answer, search_results = agent.process(
                query,
                top_k=top_k,
                use_rerank=use_rerank,
                expand=expand,
                graph_expand=graph_expand,
                label_filter=label_filter or None,
                search_mode="hybrid",
                graph_parallel=graph_parallel,
                profile=profile,
            )
            raw_hits = search_results if isinstance(search_results, list) else []

            # Render non-search intents immediately
            if intent in (INTENT_GREETING, INTENT_SYSTEM_INFO, INTENT_CLARIFY):
                with st.chat_message("assistant"):
                    st.markdown(answer)
                pending_turn["answer"] = answer
                pending_turn["raw_hits"] = raw_hits
                pending_turn["status"] = "done"
                st.session_state.searching = False
                st.rerun()
                return

            # ── Real search intent: run the full pipeline ──
            with st.chat_message("assistant"):
                with st.status("Processing query...", expanded=True) as status:
                    slot = st.empty()
                    # Stage 0: LLM router decides whether to search
                    _animate_stage(slot, "0/6 routing...")
                    router_decision = _search_router.decide(query)
                    _complete_stage(slot, "0/6 routing...")
                    st.write(f"   Router: search={router_decision['search']}, query={router_decision['query']!r}")
                    if not router_decision["search"]:
                        # Direct chat reply, no retrieval
                        if llm_ok:
                            chat_reply = _llm_client.chat([
                                {"role": "system", "content": "You are a friendly assistant. Reply concisely in the same language as the user."},
                                {"role": "user", "content": query},
                            ], temperature=0.3, max_tokens=200, timeout=15)
                            answer = chat_reply or "I'm here to help. What would you like to search?"
                        else:
                            answer = "I'm here to help. What would you like to search?"
                        status.update(label="Done", state="complete")
                        pending_turn["answer"] = answer
                        pending_turn["raw_hits"] = []
                        pending_turn["status"] = "done"
                        st.session_state.searching = False
                        st.rerun()
                        return
                    query = router_decision["query"] or query

                    # Stage 1: Query expansion
                    _animate_stage(slot, "1/6 query expansion...")
                    llm_ok = _check_llm_available()
                    expansion_queries = []
                    if expand and llm_ok:
                        try:
                            from apps.corpchat.search import QueryExpander
                            expander = QueryExpander()
                            expansion_queries = expander.expand(query, use_cache=False)
                            if len(expansion_queries) > 1:
                                st.write(f"   Generated {len(expansion_queries) - 1} expanded queries:")
                                # Animation container for expanded queries
                                anim_container = st.container()
                                import time as _time
                                for idx, (eq, weight) in enumerate(expansion_queries[1:], 1):
                                    with anim_container:
                                        st.markdown(f"<div style='animation: fadeInRight 0.5s ease-in {idx * 0.1}s both; padding: 4px 8px; margin: 2px 0; background: #1f2937; border-radius: 4px; border-left: 3px solid #3b82f6;'>{eq}</div>", unsafe_allow_html=True)
                                    _time.sleep(0.1)
                            else:
                                st.write("   No expansion needed")
                        except Exception as e:
                            st.write(f"   ⚠️ Expansion failed: {e}")
                            expand = False
                    elif expand and not llm_ok:
                        st.write("   ⚠️ LLM unavailable — skipping expansion")
                        expand = False

                    # Stage 2: Hybrid search
                    _animate_stage(slot, "2/6 hybrid search (BM25 + vector)...")
                    results, raw_hits = _run_search(
                        query, top_k, use_rerank, expand, graph_expand, False, label_filter,
                        graph_parallel,
                    )
                    _complete_stage(slot, "2/6 hybrid search...")
                    st.write(f"   Found {len(raw_hits)} hits")

                    # Stage 3: RRF fusion
                    _animate_stage(slot, "3/6 RRF fusion...")
                    _complete_stage(slot, "3/6 RRF fusion...")
                    st.write("   Merged expanded queries")

                    # Stage 4: Graph expansion
                    _animate_stage(slot, "4/6 graph expansion...")
                    _complete_stage(slot, "4/6 graph expansion...")
                    if graph_expand > 0:
                        st.write(f"   {graph_expand} hops traversed")
                    else:
                        st.write("   Skipped (0 hops)")

                    # Stage 5: Reranking
                    _animate_stage(slot, "5/6 reranking...")
                    _complete_stage(slot, "5/6 reranking...")
                    if use_rerank:
                        st.write("   Cross-encoder applied")
                    else:
                        st.write("   Skipped")

                    # Stage 6: LLM answer generation
                    _animate_stage(slot, "6/6 generating answer...")
                    context_parts = []
                    for hit in raw_hits[: top_k * 2]:
                        content = hit.get("text", "") if isinstance(hit, dict) else ""
                        if content:
                            context_parts.append(content)
                    context = "\n---\n".join(context_parts) if context_parts else "No relevant context found."

                    if llm_ok:
                        answer = generate_answer_litellm(query, context, profile=profile)
                    else:
                        answer = "LLM is unavailable. Here are the retrieved messages:\n\n" + "\n\n---\n\n".join(context_parts[:3])
                    _complete_stage(slot, "6/6 generating answer...")

                    status.update(label="Search complete!", state="complete")

            # Update the turn with results
            pending_turn["answer"] = answer
            pending_turn["raw_hits"] = raw_hits
            pending_turn["status"] = "done"
            st.session_state.searching = False
            st.rerun()

    # Chat input at page level (full-width, fixed at bottom)
    query = st.chat_input("Ask anything about the conversations...")
    if query and not st.session_state.searching:
        # Add user turn + pending assistant turn
        st.session_state.chat_history.append({
            "query": query,
            "answer": None,
            "raw_hits": [],
            "status": "processing",
        })
        st.session_state.searching = True
        st.rerun()


if page == "Search":
    _render_search_page()

elif page == "Contacts":
    st.title("Contacts")
    df = fetch_contacts()
    if df.empty:
        st.warning("No contacts available.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "Messages":
    st.title("Messages")
    df = fetch_messages()
    if df.empty:
        st.warning("No messages available.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "Overview":
    st.title("Overview")
    contacts, messages, conversations, labels = fetch_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Contacts", contacts)
    col2.metric("Total Messages", messages)
    col3.metric("Conversations", conversations)
    if labels:
        st.subheader("Messages by Label")
        label_df = pd.DataFrame(labels, columns=["Label", "Count"])
        st.bar_chart(label_df.set_index("Label"))

elif page == "Chat Viewer":
    st.title("Chat Viewer")
    name_map = fetch_contact_name_map()
    if not name_map:
        st.warning("No contacts available.")
    else:
        contact_options = {name: uid for uid, name in name_map.items()}
        selected = st.selectbox("Select a contact", list(contact_options.keys()))
        if selected:
            userid = contact_options[selected]
            conv_df = fetch_conversations_for_contact(userid)
            if conv_df.empty:
                st.info("No messages for this contact.")
            else:
                st.caption(f"{len(conv_df)} messages")
                for _, row in conv_df.iterrows():
                    is_user = row["origin"] == "3"
                    with st.chat_message("user" if is_user else "assistant"):
                        st.markdown(f"**{row['external_userid']}** ({row.get('label', '')})\n\n{row['content']}")
