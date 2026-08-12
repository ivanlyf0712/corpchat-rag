#!/usr/bin/env python3
"""
CorpChat Intelligence – Streamlit App
View contacts, messages, statistics, a chat-style conversation viewer, and semantic search (chatbox).
"""

import sys
import os
import json
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from typing import Any, Optional

# ── Ensure the project root is on the Python path ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── HF 离线自动检测: 必须在 import apps.corpchat.search (会拉入 txtai →
# huggingface_hub) 之前调用。huggingface_hub.constants 在 import 时读取
# HF_HUB_OFFLINE, 一旦被提前拉入, 自动离线开关将不生效。──
from apps.corpchat.hf_offline import apply_auto_offline
apply_auto_offline()

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
from apps.corpchat.search.agent_config import (
    PRESET_LABELS,
    SOURCE_OPTIONS,
    STYLE_LABELS,
    apply_preset,
    default_agent_config,
    persona_to_profile_dict,
    preset_index,
    sources_from_labels,
    sources_to_labels,
    style_index,
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
from apps.corpchat.search.utils import _format_citations

# Module-level intent classifier (cached across renders — deterministic")
_intent_classifier = IntentClassifier()

# ── Initialize DB-backed agent memory table ──
try:
    from core.corpchat_db import init_agent_memory_table
    init_agent_memory_table()
except Exception:
    pass

# ── Initialize DB-backed disposition profiles table (persona) ──
try:
    from core.corpchat_db import init_disposition_profiles_table
    init_disposition_profiles_table()
except Exception:
    pass

# ── Initialize DB-backed unified agent config table ──
try:
    from core.corpchat_db import init_agent_config_table
    init_agent_config_table()
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
/* 设置面板: 打开时从上方滑入 */
@keyframes settingsSlideDown { from { opacity: 0; transform: translateY(-14px); } to { opacity: 1; transform: translateY(0); } }
[data-testid="stVerticalBlockBorderWrapper"] { animation: settingsSlideDown 0.28s ease; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════ sidebar navigation ════════════════════════════════════
def _render_sidebar_settings_toggle():
    """左栏底部 ⚙️/✖ 开关: 点击切换右侧设置面板。

    打开 → 右栏变为设置; 再点 (✖) → 退出设置, 聊天占满宽度。
    状态存于 session_state.settings_open (跨 rerun 保留)。
    """
    settings_open = st.session_state.get("settings_open", False)
    label = "✖ Exit settings" if settings_open else "⚙️ Settings"
    if st.button(label, key="settings_toggle", use_container_width=True):
        st.session_state.settings_open = not settings_open
        st.rerun()


with st.sidebar:
    st.markdown("## CorpChat Intelligence")
    st.caption("Corporate Relationship & Chat Analytics")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Search", "Contacts", "Messages", "Overview", "Chat Viewer"],
        index=0,
    )
    st.divider()
    _render_sidebar_settings_toggle()


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
        "If the context doesn't contain the answer, say so. "
        "Only use the provided context — never invent message content, URLs, sender names, "
        "or any detail not present in it."
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
    """从 agent_config 构建当前会话的 DispositionProfile (0-10 → 0-1), 无则 None。

    Hindsight 桥接 (只读镜像): 若配置了 hindsight_bank, 人格完全由 Hindsight bank
    的 disposition 决定 (唯一真源, 在 Hindsight Web UI 调整)。未配置时用本地
    persona 滑杆。两者皆无则返回 None (中性默认)。
    """
    try:
        from apps.corpchat.search.persona import DispositionProfile
        cfg = st.session_state.get("agent_config") or {}

        # 1) Hindsight bank disposition → CARA (唯一真源, 只读)
        hs_bank = cfg.get("persona", {}).get("hindsight_bank") or os.getenv("HINDSIGHT_BANK_ID")
        if hs_bank:
            return DispositionProfile.from_hindsight(hs_bank)

        # 2) 无 Hindsight → 用本地 persona 配置
        if cfg and cfg.get("persona"):
            return DispositionProfile.from_dict(persona_to_profile_dict(cfg["persona"]))
    except Exception:
        pass
    return None


def _retain_search_to_hindsight(query: str, raw_hits: list, bank: Optional[str] = None) -> None:
    """搜索后把查询+命中消息写入 Hindsight 记忆 (跨会话记忆, best-effort)。

    每次搜索把用户问题 + 最相关的几条命中消息 retain 到配置的 bank,
    让 Hindsight 实体图出现新节点, 并供后续会话 recall。
    bank 优先用调用方传入的 UI 配置值, 回退到环境变量/默认。
    """
    try:
        from apps.corpchat.search import hindsight_client as hc
        if not bank:
            bank = os.getenv("HINDSIGHT_BANK_ID") or "test-bank"
        # 组装记忆内容: 查询 + 命中消息摘要 (保留实体可提取性)
        parts = [f"用户查询: {query}"]
        seen = set()
        for h in raw_hits or []:
            if not isinstance(h, dict):
                continue
            mid = str(h.get("id", ""))
            if mid in seen:
                continue
            seen.add(mid)
            text = str(h.get("text", "") or "")[:300]
            if text:
                parts.append(f"- {text}")
            if len(parts) >= 4:
                break
        content = "\n".join(parts)
        if len(content) < 10:
            return
        # 实体锚点: 命中消息的 customer_name/company 作为 tags, 供 Hindsight
        # compact 与未来 recall 保留实体信息 (写入侧保持无条件, 不智能过滤)。
        from apps.corpchat.search.tools import extract_entity_tags
        tags = ["corpchat", "search"] + extract_entity_tags(raw_hits)
        # async_=True: 同步 retain 服务端要跑 ~13s (embedding+实体提取+consolidation),
        # 会阻塞 UI 至 10s 超时 (Agent complete 后答案迟迟不出现)。异步入队 32ms 返回,
        # 由 Hindsight 后台处理, 记忆不丢失。
        hc.retain(content, bank=bank, context=f"corpchat search: {query[:100]}",
                  tags=tags, async_=True)
    except Exception:
        pass


def _load_persisted_config() -> Optional[dict]:
    """从 DB 恢复上次保存的 agent 配置 (跨刷新/跨会话保留)。

    session_id 在 Agent() 构造时随机生成, 刷新后变化, 无法作为稳定 key。
    因此这里读取 agent_config 表中最近更新的那条记录 (全局共享配置),
    而不是按 session_id 精确匹配。无记录时返回 None。
    """
    try:
        from core.db import get_db_connection
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT config FROM agent_config ORDER BY updated_at DESC, id DESC LIMIT 1"
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return None
            import json
            cfg = json.loads(row[0])
            # 兼容旧配置: 缺 hindsight_bank 时补默认 (记住 bank 名)
            if isinstance(cfg, dict):
                cfg.setdefault("persona", {}).setdefault("hindsight_bank", "test-bank")
            return cfg
        finally:
            conn.close()
    except Exception:
        return None


def _persist_agent_config(cfg=None):
    """把统一 agent 配置持久化到 DB (best-effort, non-fatal)。"""
    try:
        from core.corpchat_db import save_agent_config
        cfg = cfg if cfg is not None else st.session_state.get("agent_config")
        if cfg:
            save_agent_config(st.session_state.get("session_id", "default"), cfg)
    except Exception:
        pass


def _node_color(node: dict) -> str:
    """图谱节点配色 (按类型); 高亮(风险)节点用红色。"""
    if node.get("highlighted"):
        return "#f85149"
    return {
        "person": "#58a6ff",
        "company": "#f78166",
        "label": "#3fb950",
        "keyword": "#d29922",
        "message": "#8b949e",
    }.get(node["type"], "#8b949e")


_CONSTELLATION_TPL = """<div id="cg" style="width:100%;height:__HEIGHT__px;background:#161b22;border:1px solid #30363d;border-radius:8px;position:relative;overflow:hidden;">
<svg id="cg-svg" width="100%" height="__HEIGHT__" style="display:block;"></svg>
<div id="cg-tip" style="position:absolute;left:0;top:0;display:none;background:#0d1117;color:#e6e6e6;border:1px solid #58a6ff;padding:3px 8px;border-radius:4px;font:12px 'Segoe UI',sans-serif;pointer-events:none;"></div>
<script>
(function() {
  var NODES = __NODES__;
  var EDGES = __EDGES__;
  var H = __HEIGHT__;
  var box = document.getElementById('cg');
  var W = box.clientWidth || 900;
  var svg = document.getElementById('cg-svg');
  var tip = document.getElementById('cg-tip');
  var NS = 'http://www.w3.org/2000/svg';
  var P = [], IDX = {};
  NODES.forEach(function(n, i) { P.push({i: i, id: n.id, x: Math.random()*W, y: Math.random()*H, vx: 0, vy: 0}); IDX[n.id] = i; });
  var lines = EDGES.map(function() { var l = document.createElementNS(NS, 'line'); l.setAttribute('stroke', '#30363d'); l.setAttribute('stroke-width', '1'); svg.appendChild(l); return l; });
  var circles = NODES.map(function(n) { var c = document.createElementNS(NS, 'circle'); c.setAttribute('r', n.size); c.setAttribute('fill', n.color); c.setAttribute('stroke', '#0d1117'); c.setAttribute('stroke-width', '2'); svg.appendChild(c); return c; });
  var labels = NODES.map(function(n) { var t = document.createElementNS(NS, 'text'); t.setAttribute('fill', '#c9d1d9'); t.setAttribute('font-size', '10'); t.setAttribute('text-anchor', 'middle'); t.textContent = n.label; svg.appendChild(t); return t; });
  var drag = null;
  function pos(ev) { var r = svg.getBoundingClientRect(); return {x: ev.clientX - r.left, y: ev.clientY - r.top}; }
  svg.addEventListener('mousedown', function(ev) {
    var p = pos(ev);
    for (var i = P.length - 1; i >= 0; i--) {
      var d = Math.hypot(p.x - P[i].x, p.y - P[i].y);
      if (d < 22) { drag = P[i]; drag.fx = p.x; drag.fy = p.y; restart(); ev.preventDefault(); return; }
    }
  });
  window.addEventListener('mousemove', function(ev) { if (drag) { var p = pos(ev); drag.fx = p.x; drag.fy = p.y; restart(); } });
  window.addEventListener('mouseup', function() { drag = null; });
  svg.addEventListener('mousemove', function(ev) {
    var p = pos(ev), hit = null;
    for (var i = 0; i < P.length; i++) { if (Math.hypot(p.x - P[i].x, p.y - P[i].y) < 24) hit = P[i]; }
    if (hit) { tip.style.display = 'block'; tip.style.left = (p.x + 12) + 'px'; tip.style.top = (p.y + 12) + 'px'; tip.textContent = NODES[hit.i].label; }
    else { tip.style.display = 'none'; }
  });
  var frame = 0, still = 0, running = true;
  function tick() {
    frame++;
    var energy = 0;
    for (var a = 0; a < P.length; a++) for (var b = a + 1; b < P.length; b++) {
      var A = P[a], B = P[b], dx = B.x - A.x, dy = B.y - A.y, d2 = dx*dx + dy*dy + 1, d = Math.sqrt(d2);
      var f = 4000 / d2;
      A.vx -= dx/d*f; A.vy -= dy/d*f; B.vx += dx/d*f; B.vy += dy/d*f;
    }
    EDGES.forEach(function(e) {
      var A = P[IDX[e.s]], B = P[IDX[e.t]];
      if (A === undefined || B === undefined) return;
      var dx = B.x - A.x, dy = B.y - A.y, d = Math.sqrt(dx*dx + dy*dy + 1), f = (d - 110) * 0.015;
      A.vx += dx/d*f; A.vy += dy/d*f; B.vx -= dx/d*f; B.vy -= dy/d*f;
    });
    var MAXSPEED = 1.5;
    for (var i = 0; i < P.length; i++) {
      var p = P[i];
      if (drag === p) { p.x = p.fx; p.y = p.fy; p.vx = 0; p.vy = 0; }
      else {
        p.vx = (p.vx + (W/2 - p.x)*0.0005) * 0.75; p.vy = (p.vy + (H/2 - p.y)*0.0005) * 0.75;
        p.vx = Math.max(-MAXSPEED, Math.min(MAXSPEED, p.vx));
        p.vy = Math.max(-MAXSPEED, Math.min(MAXSPEED, p.vy));
        p.x += p.vx; p.y += p.vy;
        if (p.x < 10) p.x = 10; if (p.x > W - 10) p.x = W - 10;
        if (p.y < 10) p.y = 10; if (p.y > H - 10) p.y = H - 10;
        energy += p.vx*p.vx + p.vy*p.vy;
      }
    }
    for (var e = 0; e < EDGES.length; e++) {
      var A = P[IDX[EDGES[e].s]], B = P[IDX[EDGES[e].t]];
      if (A !== undefined && B !== undefined) {
        lines[e].setAttribute('x1', A.x); lines[e].setAttribute('y1', A.y);
        lines[e].setAttribute('x2', B.x); lines[e].setAttribute('y2', B.y);
      }
    }
    for (var i = 0; i < P.length; i++) {
      var p = P[i];
      circles[i].setAttribute('cx', p.x); circles[i].setAttribute('cy', p.y);
      labels[i].setAttribute('x', p.x); labels[i].setAttribute('y', p.y - 10);
    }
    // 动画预算: 收敛 (动能连续低) 或达到帧数上限后停止, 释放 CPU (不再 requestAnimationFrame)
    if (drag) { still = 0; }
    else if (energy < 4) { still++; }
    else { still = 0; }
    if (still >= 6 || frame > 160) { running = false; return; }
    requestAnimationFrame(tick);
  }
  // 拖拽时唤醒已停止的动画
  function restart() {
    if (!running) { running = true; frame = 0; still = 0; requestAnimationFrame(tick); }
  }
  tick();
})();
</script>
</div>"""


def _constellation_html(nodes: list, edges: list, height: int = 380) -> str:
    """生成自包含 (无 CDN / 无外部库) 的力导向星座图 HTML。

    经 st.iframe 以 srcdoc 方式隔离渲染 (JS 可运行, 拖拽/悬停/动画), 离线可用。
    节点点击回填搜索由 render 侧 st.button 完成 (iframe 无 Python 桥)。
    """
    ns = json.dumps([
        {"id": str(n["id"]), "label": str(n["label"])[:16], "color": _node_color(n),
         "size": max(6, min(20, int(n.get("size", 10))))}
        for n in nodes
    ], ensure_ascii=False).replace("</", "<\\/")
    es = json.dumps(
        [{"s": str(e["source"]), "t": str(e["target"])} for e in edges],
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return (_CONSTELLATION_TPL
            .replace("__NODES__", ns)
            .replace("__EDGES__", es)
            .replace("__HEIGHT__", str(height)))


def _cap_graph(nodes: list, edges: list, max_nodes: int = 40, max_edges: int = 80):
    """按连接度截断图谱, 限制浏览器端渲染成本。

    优先保留实体节点 (person/company/label/keyword) 和高连接度节点;
    边按两端节点度数之和排序截断。
    """
    from collections import Counter
    degree = Counter()
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    def _sort_key(n):
        return (1 if n.get("type") == "message" else 0, -degree.get(n["id"], 0))

    keep = sorted(nodes, key=_sort_key)[:max_nodes]
    keep_ids = {n["id"] for n in keep}
    kept_edges = [e for e in edges if e["source"] in keep_ids and e["target"] in keep_ids]
    if len(kept_edges) > max_edges:
        kept_edges.sort(key=lambda e: degree.get(e["source"], 0) + degree.get(e["target"], 0),
                        reverse=True)
        kept_edges = kept_edges[:max_edges]
    return keep, kept_edges


def _extract_data_map(max_nodes: int = 60, max_edges: int = 120) -> dict:
    """从 txtai 索引图提取数据地图 (消息/联系人关系, 确定性)。

    返回 {"nodes": [{id,label,type,size}], "edges": [{source,target}]}。
    失败时返回空 dict (降级)。
    """
    try:
        from apps.corpchat.search.searcher import Searcher
        emb = _load_search_index()
        g = emb.graph
        if not g or not g.backend:
            return {}
        nx_graph = g.backend
        # 按连接度排序, 保留高连接节点 (信息密度高)
        degree = {n: nx_graph.degree(n) for n in nx_graph.nodes()}
        all_nodes = sorted(nx_graph.nodes(), key=lambda n: -degree.get(n, 0))
        selected = all_nodes[:max_nodes]
        selected_set = set(selected)
        # 找选中节点间的边
        edges = []
        for src, dst, data in nx_graph.edges(data=True):
            if src in selected_set and dst in selected_set:
                edges.append({"source": src, "target": dst})
                if len(edges) >= max_edges:
                    break
        nodes = []
        for n in selected:
            attrs = nx_graph.nodes[n]
            doc_id = str(attrs.get("id", n))
            nodes.append({
                "id": n,
                "label": doc_id.split("__")[0][:24] if "__" in doc_id else doc_id[:24],
                "type": "person",
                "size": max(6, min(20, 8 + degree.get(n, 0))),
            })
        return {"nodes": nodes, "edges": edges}
    except Exception:
        return {}


def _render_memory_graph(cfg: dict):
    """渲染设置面板下的两个图谱 (tabs 切换)。

    - 📊 数据地图: 从 txtai 索引图提取的全量消息/联系人关系 (确定性, 140/384)。
    - 🧠 记忆视图: Hindsight bank 的记忆实体图 (跨会话, 随交互增长)。
    渲染: 自包含 HTML/SVG 力导向图 (st.iframe srcdoc, 无外部依赖)。
    整个渲染 try/except 兜底 —— 图谱故障不应破坏搜索 UI。
    """
    try:
        tab_data, tab_mem = st.tabs(["📊 Data map", "🧠 Memory view"])

        # ── Tab 1: 数据地图 (txtai graph, 全量确定性) ──
        with tab_data:
            data_map = _extract_data_map()
            if not data_map or not data_map.get("nodes"):
                st.caption("Data map unavailable (index built with graph mode=off).")
            else:
                st.iframe(_constellation_html(data_map["nodes"], data_map["edges"]), height=420)
                st.caption(
                    f"📊 Data map · {len(data_map['nodes'])} nodes · {len(data_map['edges'])} edges"
                    f" · from the txtai graph (deterministic message/contact relations)"
                )

        # ── Tab 2: 记忆视图 (Hindsight 实体图 / 会话检索实体) ──
        with tab_mem:
            hs_bank = cfg.get("persona", {}).get("hindsight_bank") or os.getenv("HINDSIGHT_BANK_ID")
            if hs_bank:
                from apps.corpchat.search import hindsight_client as hc
                hg = hc.get_entity_graph(hs_bank, limit=50)
                if hg.get("nodes"):
                    nodes = []
                    for n in hg["nodes"]:
                        if not isinstance(n, dict) or not n.get("id"):
                            continue
                        nodes.append({
                            "id": str(n.get("id")),
                            "label": str(n.get("label") or n.get("id"))[:16],
                            "type": "person",
                            "color": n.get("color", "#58a6ff"),
                            "size": 12,
                        })
                    edges = []
                    for e in hg.get("edges", []):
                        if not isinstance(e, dict):
                            continue
                        s, t = e.get("source"), e.get("target")
                        if s and t:
                            edges.append({"source": str(s), "target": str(t)})
                    if nodes:
                        st.iframe(_constellation_html(nodes, edges), height=420)
                        st.caption(
                            f"🧠 Hindsight memory view · {hg.get('total_entities', len(nodes))} entities"
                            f" · {hg.get('total_edges', len(edges))} edges"
                            f" · bank {hs_bank} (full version at :9999)"
                        )
                        return
                st.caption("Hindsight memory is empty — it fills in as you search/chat.")

            # 回退: 会话检索实体的本地图
            from apps.corpchat.search.memory_graph import build_entity_graph
            graph_messages = []
            for turn in st.session_state.get("chat_history", []):
                hits = turn.get("raw_hits") or []
                if isinstance(hits, list):
                    graph_messages.extend(h for h in hits if isinstance(h, dict))
            if not graph_messages:
                st.caption("Search first — the memory graph will show entity relationships here.")
                return
            risk = set()
            if cfg["persona"].get("skepticism", 5) >= 7:
                risk = {"old_friend_reconnect", "詐騙", "fraud"}
            graph = build_entity_graph(
                messages=graph_messages,
                sources=cfg["knowledge"].get("sources", ["messages", "contacts"]),
                risk_labels=risk,
            )
            nodes, edges = graph["nodes"], graph["edges"]
            if not nodes:
                st.caption("No entities to draw yet (try searching more messages).")
                return
            if len(nodes) > 40 or len(edges) > 80:
                nodes, edges = _cap_graph(nodes, edges)
            st.iframe(_constellation_html(nodes, edges), height=420)
            st.caption(f"{len(nodes)} nodes · {len(edges)} edges · sources {cfg['knowledge'].get('sources')}")
            # 联动: 点击节点 → 回填搜索框
            clickable = [n for n in nodes if n["type"] in ("person", "company", "label", "keyword")][:12]
            if clickable:
                st.caption("Click a node to fill the search box:")
                cols = st.columns(min(3, len(clickable)))
                for i, n in enumerate(clickable):
                    with cols[i % len(cols)]:
                        if st.button(n["label"], key=f"graph_node_{i}", use_container_width=True):
                            st.session_state.search_query = n["label"]
                            st.rerun()
    except Exception as e:
        st.caption(f"Memory graph render failed: {e}")

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
def _render_settings_panel(cfg: dict):
    """渲染右侧设置面板 (人格特質 / 搜索策略 / 知識範圍)。

    即时生效, 会话级持久化 (session_state.agent_config)。显隐由左栏
    ⚙️/✖ 开关控制 (settings_open)。
    """
    with st.expander("🎛️ Configure Agent", expanded=False):
        with st.expander("🧠 Personality (CARA)", expanded=False):
            # ── Hindsight 桥接 (只读镜像模式) ──
            # 填入 bank ID 后: 人格由 Hindsight 驱动, 滑杆只读显示 Hindsight 真实值。
            # 修改请到 Hindsight Web UI (链接下方), 或点"刷新"重新拉取。
            hs_bank = st.text_input(
                "Hindsight memory bank (optional)",
                value=cfg["persona"].get("hindsight_bank", ""),
                help="Enter a Hindsight bank ID (e.g. test-bank) to drive the persona from that "
                     "bank's disposition (read-only mirror); leave empty to use the local sliders below.",
                disabled=st.session_state.searching,
            )
            cfg["persona"]["hindsight_bank"] = hs_bank.strip()
            hs_driven = bool(hs_bank.strip())

            # 当前生效的人格 (Hindsight 驱动时为 Hindsight 值, 否则为本地滑杆值)
            # session 级缓存: 只在 bank 变更或点"刷新"时重新拉取, 避免每次 rerun 打 API。
            effective = None
            if hs_driven:
                cache_key = f"hs_profile_{hs_bank.strip()}"
                if (st.session_state.get("hs_bank") != hs_bank.strip()
                        or st.session_state.get("hs_profile_ts") is None):
                    try:
                        from apps.corpchat.search.persona import DispositionProfile
                        effective = DispositionProfile.from_hindsight(hs_bank.strip())
                        st.session_state[cache_key] = effective
                        st.session_state.hs_bank = hs_bank.strip()
                        st.session_state.hs_profile_ts = True
                    except Exception as _hs_err:
                        st.caption(f"⚠️ Cannot reach Hindsight: {type(_hs_err).__name__}: {_hs_err}")
                else:
                    effective = st.session_state.get(cache_key)

            col_link, col_refresh = st.columns([3, 1])
            with col_link:
                if hs_driven:
                    st.link_button(
                        "🔗 Adjust in Hindsight",
                        f"http://localhost:9999/banks/{hs_bank.strip()}/",
                        type="secondary",
                        use_container_width=True,
                    )
                else:
                    st.caption("Not connected to Hindsight — using the local sliders below")
            with col_refresh:
                if hs_driven and st.button(
                    "🔄 Refresh", use_container_width=True,
                    disabled=st.session_state.searching,
                ):
                    # 清除 session 缓存标记 → 下一轮从 Hindsight 重新拉取 disposition
                    st.session_state.hs_profile_ts = None
                    st.rerun()

            if hs_driven and effective is not None:
                st.caption(
                    f"🔗 Personality driven by Hindsight: skepticism {effective.skepticism:.0%} · "
                    f"literality {effective.literality:.0%} · empathy {effective.empathy:.0%}"
                    f" (adjust in the Hindsight Web UI)"
                )

            # 滑杆: Hindsight 驱动时只读显示 Hindsight 值; 否则本地可编辑
            hs_ro = hs_driven and effective is not None
            if hs_ro:
                # Hindsight 0-1 → 0-10 显示刻度 (只读)
                sk_val = int(round(effective.skepticism * 10))
                li_val = int(round(effective.literality * 10))
                em_val = int(round(effective.empathy * 10))
            else:
                sk_val = int(cfg["persona"].get("skepticism", 5))
                li_val = int(cfg["persona"].get("literality", 5))
                em_val = int(cfg["persona"].get("empathy", 5))

            preset_label = st.selectbox(
                "Preset mode", list(PRESET_LABELS.keys()),
                index=preset_index(cfg["persona"].get("preset", "custom")),
                disabled=st.session_state.searching or hs_ro,
            )
            apply_preset(cfg, preset_label)

            cfg["persona"]["skepticism"] = st.slider(
                "Skepticism", 0, 10, sk_val,
                help="Mark uncertainty on conclusions with insufficient evidence",
                disabled=st.session_state.searching or hs_ro,
            )
            cfg["persona"]["literality"] = st.slider(
                "Literality", 0, 10, li_val,
                help="Answer strictly from the retrieved text",
                disabled=st.session_state.searching or hs_ro,
            )
            cfg["persona"]["empathy"] = st.slider(
                "Empathy", 0, 10, em_val,
                help="Acknowledge tone/feelings before giving information",
                disabled=st.session_state.searching or hs_ro,
            )
            style_label = st.selectbox(
                "Answer length", list(STYLE_LABELS.keys()),
                index=style_index(cfg["persona"].get("style", "standard")),
                disabled=st.session_state.searching or hs_ro)
            cfg["persona"]["style"] = STYLE_LABELS[style_label]

        with st.expander("⚙️ Search strategy", expanded=False):
            depth_label = st.selectbox(
                "Search depth", ["Simple", "Auto", "Deep"],
                index={"simple": 0, "auto": 1}.get(cfg["search"].get("depth", "deep"), 2),
                help="Simple = single-step search; Auto = rule-detected agent escalation "
                     "(multi-hop / cross-session / time only); Deep = always agent",
                disabled=st.session_state.searching)
            cfg["search"]["depth"] = {"Simple": "simple", "Auto": "auto"}.get(depth_label, "deep")
            cfg["search"]["expand"] = st.checkbox(
                "Query expansion", value=cfg["search"].get("expand", True),
                help="LLM semantic rephrase + keywords", disabled=st.session_state.searching)
            cfg["search"]["rerank"] = st.checkbox(
                "Rerank", value=cfg["search"].get("rerank", True),
                help="Cross-encoder reranking", disabled=st.session_state.searching)
            cfg["search"]["graph_hops"] = st.slider(
                "Graph hops", 0, 3, int(cfg["search"].get("graph_hops", 1)),
                disabled=st.session_state.searching)
            cfg["search"]["graph_parallel"] = st.checkbox(
                "Graph path", value=cfg["search"].get("graph_parallel", False),
                help="Traverse the graph as a fusion path (relationship queries)", disabled=st.session_state.searching)
            cfg["search"]["top_k"] = st.slider(
                "Top-k", 1, 20, int(cfg["search"].get("top_k", 5)),
                disabled=st.session_state.searching)
            cfg["search"]["label_filter"] = st.text_input(
                "Label filter", value=cfg["search"].get("label_filter", ""),
                help="e.g. quotation_request", disabled=st.session_state.searching)

        with st.expander("📚 Knowledge scope", expanded=False):
            source_labels = st.multiselect(
                "Data sources", SOURCE_OPTIONS,
                default=sources_to_labels(cfg["knowledge"].get("sources", ["messages", "contacts"])),
                disabled=st.session_state.searching)
            cfg["knowledge"]["sources"] = sources_from_labels(source_labels)
            cfg["knowledge"]["citations"] = st.checkbox(
                "Citations", value=cfg["knowledge"].get("citations", False),
                help="Attach sources to answers", disabled=st.session_state.searching)


def _render_search_page():
    """Render the Search page (kept callable so tests can drive it)."""
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

    # ── 统一 agent 配置 (设置面板关闭时仍保留上次值) ──
    cfg = st.session_state.get("agent_config")
    if cfg is None:
        # 刷新/新会话 → 从 DB 恢复上次配置 (避免每次刷新重置)
        cfg = _load_persisted_config()
        if cfg is None:
            cfg = default_agent_config()
        st.session_state.agent_config = cfg

    # 提交搜索时自动退出编辑模式 (让聊天区显示处理过程与结果)
    if pending_turn:
        st.session_state.settings_open = False
    settings_open = st.session_state.get("settings_open", False)

    st.title("⚙️ Settings" if settings_open else "Search")

    # ── 派生局部变量 (供下游搜索/agent 使用; 即时生效) ──
    st.session_state.agent_config = cfg
    expand = cfg["search"]["expand"]
    use_rerank = cfg["search"]["rerank"]
    graph_expand = cfg["search"]["graph_hops"]
    graph_parallel = cfg["search"]["graph_parallel"]
    top_k = cfg["search"]["top_k"]
    label_filter = cfg["search"]["label_filter"]
    depth = cfg["search"].get("depth", "deep")
    if depth == "auto":
        # 规则检测器 (multi-hop / cross-session / time) 在 query 可用后决定是否
        # 升级到 agent (见下方 pending_turn 分支); 此处仅作默认占位。
        agent_enabled = False
    else:
        agent_enabled = (depth == "deep")
    st.session_state.agent_enabled = agent_enabled

    # ── 布局: 编辑模式 → 配置面板替代聊天面板 (图谱在配置底部) ──
    if settings_open:
        with st.container(key="settings_panel", border=True):
            _render_settings_panel(cfg)
            # 滑杆/输入改动即时持久化 → 刷新后也能恢复
            _persist_agent_config(cfg)
        st.divider()
        st.markdown("#### 🕸️ Memory Graph")
        _render_memory_graph(cfg)
    else:
        _render_chat_history(st.session_state.chat_history)

        # If there's a pending processing turn, handle it now
        if pending_turn:
            query = pending_turn["query"]
            if depth == "auto":
                # Auto 深度: 规则检测器决定是否升级到 agent (检索优先为默认)
                from apps.corpchat.search import detect_agent_mode
                agent_enabled = detect_agent_mode(query) == "agent"
                st.session_state.agent_enabled = agent_enabled

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

            # ── 问候/系统问题快路: 不走 agent 搜索, 避免 "Hi!" 触发工具调用 ──
            from apps.corpchat.search.cross_table_agent import _is_greeting_query, _SYSTEM_KEYWORDS
            _q_lower = (query or "").strip().lower()
            if _is_greeting_query(_q_lower) or any(kw in _q_lower for kw in _SYSTEM_KEYWORDS):
                if _check_llm_available():
                    chat_reply = _llm_client.chat([
                        {"role": "system", "content": (
                            "You are a friendly assistant. Reply to greetings naturally and "
                            "warmly in the same language as the user. Keep it short. Do NOT "
                            "mention you are an AI or list capabilities.")},
                        {"role": "user", "content": query},
                    ], temperature=0.7, max_tokens=60, timeout=5)
                    answer = chat_reply or "Hello! How can I help you today?"
                else:
                    answer = "Hello! I'm CorpChat Intelligence. How can I help you today?"
                with st.chat_message("assistant"):
                    st.markdown(answer)
                pending_turn["answer"] = answer
                pending_turn["raw_hits"] = []
                pending_turn["status"] = "done"
                st.session_state.searching = False
                st.rerun()
                return

            # ── Persona: persist & load the tuned disposition profile ──
            _persist_agent_config()
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
                            try:
                                status.update(label=f"{label} {detail}".strip())
                            except Exception:
                                pass

                        def _on_tool(tool_name, tool_args):
                            """Live per-tool stage: 显示 agent 正在调用的精确工具与参数。"""
                            label = f"🔍 {tool_name}"
                            try:
                                args_str = json.dumps(tool_args, ensure_ascii=False)[:80]
                            except Exception:
                                args_str = ""
                            stage_labels.append(label)
                            _animate_stage(slot, label, args_str)
                            try:
                                status.update(label=f"{label} {args_str}".strip())
                            except Exception:
                                pass

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
                                sources=cfg["knowledge"].get("sources"),
                                hindsight_bank=cfg["persona"].get("hindsight_bank") or None,
                            )
                            # 会话历史: chat_history (本会话) + DB agent_memory
                            # (跨页面刷新可恢复, 决策 A4-i 指代解析)。
                            _history = [
                                {"query": t.get("query"), "answer": t.get("answer")}
                                for t in st.session_state.get("chat_history", [])
                                if t.get("answer")
                            ]
                            try:
                                from core.corpchat_db import load_agent_memory
                                _sid = st.session_state.get("session_id")
                                if _sid:
                                    _known = {(h["query"], h["answer"]) for h in _history}
                                    for _m in load_agent_memory(_sid, max_turns=6):
                                        _q, _a = _m.get("user"), _m.get("bot")
                                        if _q and (_q, _a) not in _known:
                                            _history.insert(0, {"query": _q, "answer": _a})
                                            _known.add((_q, _a))
                            except Exception:
                                pass
                            result = ct_agent.process(query, on_stage=_on_stage, on_tool=_on_tool, history=_history)
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
                # 引用来源 (agent 模式统一生效): 开启时追加来源块 (sender · 日期 · label)
                if cfg["knowledge"]["citations"] and 'result' in dir():
                    answer = answer + _format_citations(result.get("raw_hits", []))
                pending_turn["answer"] = answer
                pending_turn["raw_hits"] = result.get("raw_hits", []) if 'result' in dir() else []
                # Hindsight 参与度 (供 Process 窗显示):
                #   recall = 命中触发词, 记忆已注入; skip = gate 跳过 (无触发词); none = 未配置
                _hs_bank = cfg["persona"].get("hindsight_bank") or os.getenv("HINDSIGHT_BANK_ID")
                _hs_fired = any(s.get("label") == "Hindsight memory" for s in steps)
                pending_turn["hindsight"] = ("recall" if _hs_fired else "skip") if _hs_bank else "none"
                _retain_search_to_hindsight(pending_turn.get("query", ""), pending_turn["raw_hits"],
                                            bank=cfg["persona"].get("hindsight_bank") or None)
                # ── 持久化到 agent_memory (DB 多轮记忆, 跨页面刷新可恢复) ──
                if not str(answer).startswith("Agent error"):
                    try:
                        from core.corpchat_db import load_agent_memory, save_agent_memory
                        _sid = st.session_state.get("session_id") or "default"
                        _n = len(load_agent_memory(_sid, max_turns=100000))
                        save_agent_memory(_sid, _n + 1, query, answer, "agent")
                    except Exception:
                        pass
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
                    # 引用来源: 开启时追加来源块 (sender · 日期 · label)
                    if cfg["knowledge"]["citations"]:
                        answer = answer + _format_citations(raw_hits)
                    _complete_stage(slot, "6/6 generating answer...")

                    status.update(label="Search complete!", state="complete")

            # Update the turn with results
            pending_turn["answer"] = answer
            pending_turn["raw_hits"] = raw_hits
            # 非 agent 模式不跑 recall, 只做 retain; 未配置 bank → none
            _hs_bank = cfg["persona"].get("hindsight_bank") or os.getenv("HINDSIGHT_BANK_ID")
            pending_turn["hindsight"] = "retain" if _hs_bank else "none"
            _retain_search_to_hindsight(query, raw_hits,
                                        bank=cfg["persona"].get("hindsight_bank") or None)
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
