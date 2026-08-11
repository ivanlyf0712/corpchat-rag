"""
CorpChat Search — Cross-Table Agent Tools
===========================================
Defines LangChain tools that the ReAct agent can call:
  - search_messages: 搜索内部消息 (txtai 向量索引)
  - search_contacts: 按姓名或 userid 搜索联系人

Each tool returns structured text that the LLM can reason over.

search_messages supports full LLM query expansion + cross-encoder rerank
(as an individual search, mirroring the non-agent Searcher pipeline) when
the shared Reranker / LLM-expansion toggles are enabled. The structured
metadata of the most recent call is exposed for the Process window.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import txtai
from langchain_core.tools import tool

from .config import (
    CONTACTS_INDEX_PATH,
    DEFAULT_INDEX_PATH,
    MAX_SEARCH_LIMIT,
    logger,
)
from .utils import _segment

# txtai 的 SQLite connection 是 check_same_thread=False, 线程安全必须由外部处理
# (txtai sqlite.py: "Thread locking must be handled externally")。Streamlit 每次
# rerun 在不同线程运行, 并发访问同一 txtai 实例会触发 "Recursive use of cursors"。
# 用一把模块级 RLock 串行化所有 txtai 索引访问 (search / metadata / scan)。
_TXTAI_LOCK = threading.RLock()


# ── Module-level index cache ─────────────────────────────────────
_contacts_embeddings: Optional[txtai.Embeddings] = None
_messages_embeddings: Optional[txtai.Embeddings] = None

# ── Structured metadata of the most recent tool call ─────────────
# Captures what was actually searched so the Process window can show
# per-tool details (query, expanded queries, hit count, previews).
#
# These remain module-level dicts as a *backward-compatible channel* for
# direct tool invocations and tests. The agent path does NOT rely on them:
# CrossTableAgent snapshots the meta per tool call at execution time
# (snapshot_meta) so each tool call is attributed its own result even under
# concurrent Streamlit sessions. All reads/writes are serialized on
# _TXTAI_LOCK (per HANDOFF.md: reuse the shared lock, don't add new ones).
_last_msg_meta: Dict[str, Any] = {
    "query": "",
    "expanded_queries": [],
    "hit_count": 0,
    "previews": [],
    "raw_hits": [],
}
_last_contact_meta: Dict[str, Any] = {
    "query": "",
    "hit_count": 0,
    "previews": [],
}

# ── Module-level tool configuration ──────────────────────────────
# Retrieval-tuning toggles (expand / rerank / graph_parallel) are NOT
# part of the tool schema the LLM sees — they are system-level settings
# injected by CrossTableAgent.configure_search() at agent-build time.
_active_config: Dict[str, Any] = {
    "expand": False,
    "use_rerank": False,
    "graph_parallel": False,
}


def configure_search(expand: bool = False, use_rerank: bool = False,
                     graph_parallel: bool = False) -> None:
    """Set the module-level retrieval configuration for search tools.

    Called by CrossTableAgent at agent-build / process time so the tools
    honor the agent's toggles without exposing them in the LLM-visible schema.
    """
    global _active_config
    _active_config = {
        "expand": bool(expand),
        "use_rerank": bool(use_rerank),
        "graph_parallel": bool(graph_parallel),
    }


def get_search_config() -> Dict[str, Any]:
    """Return the current module-level tool configuration."""
    return dict(_active_config)


def get_last_search_meta() -> Dict[str, Any]:
    """Return structured metadata of the most recent search_messages call."""
    with _TXTAI_LOCK:
        return dict(_last_msg_meta)


def get_last_contact_meta() -> Dict[str, Any]:
    """Return structured metadata of the most recent search_contacts call."""
    with _TXTAI_LOCK:
        return dict(_last_contact_meta)


# ── Lock-guarded meta channel (compat + per-call snapshot) ────────
def _set_msg_meta(d: Dict[str, Any]) -> None:
    """Replace the message-search meta (lock-guarded, thread-safe)."""
    with _TXTAI_LOCK:
        _last_msg_meta.clear()
        _last_msg_meta.update(d)


def _update_msg_meta(**kwargs: Any) -> None:
    """Patch the message-search meta in place (lock-guarded)."""
    with _TXTAI_LOCK:
        _last_msg_meta.update(kwargs)


def _set_contact_meta(d: Dict[str, Any]) -> None:
    """Replace the contact-search meta (lock-guarded, thread-safe)."""
    with _TXTAI_LOCK:
        _last_contact_meta.clear()
        _last_contact_meta.update(d)


def _update_contact_meta(**kwargs: Any) -> None:
    """Patch the contact-search meta in place (lock-guarded)."""
    with _TXTAI_LOCK:
        _last_contact_meta.update(kwargs)


def snapshot_meta(tool_name: str) -> Optional[Dict[str, Any]]:
    """Snapshot the meta written by the most recent call of a tool.

    Called by the agent's tool wrapper immediately after the tool function
    returns (same thread, back-to-back with the tool's own write), so the
    snapshot is attributable to that exact tool call — even when several
    sessions run concurrently on different threads. Returns None for tools
    that carry no structured meta.
    """
    with _TXTAI_LOCK:
        if tool_name == "search_messages" or tool_name == "search_messages_where":
            return dict(_last_msg_meta)
        if tool_name == "search_contacts":
            return dict(_last_contact_meta)
    return None


def extract_entity_tags(raw_hits: list) -> List[str]:
    """从搜索命中消息的 metadata 提取实体名 (Hindsight retain 实体锚点)。

    复用消息索引已携带的字段, 无新提取依赖:
      - customer_name: 客户实体 (缺失时回退到 external_userid 稳定标识)
      - company: 组织实体
    去重保序, 最多返回 5 个。查询侧的实体名由 Hindsight 自身的实体提取
    从 content/context 中解析, 这里不重复造轮子。
    """
    entities: List[str] = []
    seen = set()
    for h in raw_hits or []:
        if not isinstance(h, dict):
            continue
        meta = h.get("metadata") or {}
        if not isinstance(meta, dict):
            continue
        name = str(meta.get("customer_name") or "").strip()
        if not name:
            name = str(meta.get("external_userid") or "").strip()
        company = str(meta.get("company") or "").strip()
        for val in (name, company):
            if val and val not in seen:
                seen.add(val)
                entities.append(val)
    return entities[:5]



def _load_messages_index() -> txtai.Embeddings:
    """Lazy-load messages txtai index (线程安全: 加载也在锁内)."""
    global _messages_embeddings
    if _messages_embeddings is not None:
        return _messages_embeddings
    with _TXTAI_LOCK:
        if _messages_embeddings is not None:
            return _messages_embeddings
        path = os.environ.get("INDEX_PATH", DEFAULT_INDEX_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Messages index not found at {path}. Run `python search.py build` first.")
        _messages_embeddings = txtai.Embeddings()
        _messages_embeddings.load(path)
        logger.info(f"Loaded messages index ({_messages_embeddings.count()} chunks)")
    return _messages_embeddings


def _load_contacts_index() -> txtai.Embeddings:
    """Lazy-load contacts txtai index (线程安全: 加载也在锁内)."""
    global _contacts_embeddings
    if _contacts_embeddings is not None:
        return _contacts_embeddings
    with _TXTAI_LOCK:
        if _contacts_embeddings is not None:
            return _contacts_embeddings
        path = os.environ.get("CONTACTS_INDEX_PATH", CONTACTS_INDEX_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Contacts index not found at {path}. Run `python search.py build-contacts` first.")
        _contacts_embeddings = txtai.Embeddings()
        _contacts_embeddings.load(path)
        logger.info(f"Loaded contacts index ({_contacts_embeddings.count()} contacts)")
    return _contacts_embeddings


def _locked_search(embeddings: txtai.Embeddings, *args, **kwargs):
    """Thread-safe wrapper around embeddings.search (txtai 非线程安全连接)."""
    with _TXTAI_LOCK:
        return embeddings.search(*args, **kwargs)


def _fetch_doc_metadata(embeddings: txtai.Embeddings, doc_id: str) -> Dict[str, Any]:
    """Fetch metadata (tags) for a document by ID from the txtai SQLite store."""
    with _TXTAI_LOCK:
        try:
            db = embeddings.database
            if db is None:
                return {}
            conn = db.connection
            cur = conn.cursor()
            try:
                cur.execute("SELECT tags FROM sections WHERE id = ?", (doc_id,))
                row = cur.fetchone()
                if row and row[0]:
                    return json.loads(row[0]) if isinstance(row[0], str) else dict(row[0])
            finally:
                cur.close()
        except Exception:
            pass
    return {}


def _parse_msg_docs(embeddings: txtai.Embeddings, raw: List[Any]) -> List[Dict[str, Any]]:
    """Parse txtai message results into {id, text, score, metadata} dicts."""
    docs = []
    for item in raw:
        doc_id = item.get("id", "") if isinstance(item, dict) else (item[0] if isinstance(item, tuple) else "")
        text = item.get("text", "") if isinstance(item, dict) else (item[1] if isinstance(item, tuple) else "")
        score = item.get("score", 0.0) if isinstance(item, dict) else (
            item[3] if isinstance(item, tuple) and len(item) >= 4 else 0.0)
        if not doc_id:
            continue
        docs.append({
            "id": doc_id,
            "text": text,
            "score": score,
            "metadata": _fetch_doc_metadata(embeddings, doc_id),
        })
    return docs


def _weighted_rrf_fuse(embeddings: txtai.Embeddings,
                       queries_with_weights: List[Tuple[str, float]],
                       limit: int = 10,
                       graph_parallel: bool = False,
                       where: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search each expanded query and merge via weighted RRF fusion.

    Thin shim over the single retrieval assembly seam (`Searcher.search_queries`):
    an agentic tool call is treated as an individual full search, riding the
    exact same per-query loop / RRF / graph-parallel path as `Searcher.search()`.

    graph_parallel: append the structural graph-traversal path (Hindsight
    graph evidence) to the fusion inputs, mirroring Searcher.search(...,
    graph_parallel=True).

    where: optional txtai SQL WHERE fragment (e.g. the sender/receiver
    metadata filter). When set, every expanded-query search is filtered
    through it, and the graph path is skipped (graph traversal cannot
    apply a metadata filter without leaking excluded chunks).
    """
    from .searcher import Searcher

    return Searcher(embeddings).search_queries(
        queries_with_weights,
        limit=limit,
        graph_parallel=graph_parallel,
        where=where,
    )



def _resolve_name_to_userid(name: Optional[str]) -> Tuple[Optional[str], str]:
    """通过姓名查找 userid，支持精确匹配。

    解析策略 (确定性, 不依赖 LLM):
      1. 输入已是 userid (user_ 前缀) → 直接返回。
      2. 精确匹配: 扫描 contacts 索引的 tags.full_name, 精确比对。
      3. 命中多个 → 返回 (None, 候选列表文本), 由 agent 决定向用户澄清。
      4. 命中 0 个 → 返回 (None, "未找到联系人")。

    Returns:
        (userid, note)。userid 为 None 时 note 是对 LLM 的说明文本。
    """
    if not name:
        return None, ""
    name = (name or "").strip()
    if re.match(r"^user_[\w\u4e00-\u9fff_]+$", name):
        return name, ""

    try:
        embeddings = _load_contacts_index()
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        logger.warning(f"Contacts index load failed in name resolution: {e}")
        return None, ""

    matches: List[Tuple[str, str]] = []  # (userid, full_name)
    try:
        db = embeddings.database
        if db is None:
            return None, ""
        with _TXTAI_LOCK:
            cur = db.connection.cursor()
            try:
                cur.execute("SELECT id, tags FROM sections")
                for row in cur.fetchall():
                    tags_raw = row[1]
                    try:
                        tags = json.loads(tags_raw) if isinstance(tags_raw, str) else dict(tags_raw or {})
                    except Exception:
                        continue
                    if (tags.get("full_name") or "") == name:
                        matches.append((row[0], tags.get("full_name", name)))
            finally:
                cur.close()
    except Exception as e:
        logger.warning(f"Exact contact name lookup failed: {e}")

    if len(matches) == 1:
        return matches[0][0], ""
    if len(matches) > 1:
        candidates = "、".join(f"{uid} ({fn})" for uid, fn in matches)
        return None, f"姓名「{name}」匹配到 {len(matches)} 个联系人: {candidates}，请澄清具体是哪一位。"
    return None, f"未找到联系人「{name}」。"


def _resolve_userid_to_name(userid: str) -> str:
    """通过 userid 反查联系人姓名 (确定性, 精确匹配 contacts 索引)。"""
    try:
        embeddings = _load_contacts_index()
        db = embeddings.database
        if db is not None:
            # 用独立连接避免与遍历中的游标冲突 (SQLite: recursive cursors not allowed)
            conn = db.connection
            with _TXTAI_LOCK:
                cur = conn.cursor()
                try:
                    cur.execute("SELECT tags FROM sections WHERE id = ?", (userid,))
                    row = cur.fetchone()
                finally:
                    cur.close()
            if row:
                tags_raw = row[0]
                try:
                    tags = json.loads(tags_raw) if isinstance(tags_raw, str) else dict(tags_raw or {})
                    return tags.get("full_name") or userid
                except Exception:
                    return userid
    except Exception as e:
        logger.warning(f"Contact name reverse lookup failed ({userid}): {e}")
    return userid


def _find_conversation_partners(userid: str, limit: int = 20) -> List[Tuple[str, str]]:
    """找出某人聊过天的对侧联系人 (会话级, 不涉及消息方向)。

    strategy:
      1. 首选 PostgreSQL: 按 open_kfid 找该 userid 参与的所有会话, 取对侧去重。
      2. DB 不可用时回退: 扫描 txtai 消息索引 tags, 收集同一 open_kfid 的对侧。

    Returns: [(userid, full_name), ...] 已去重。
    """
    partners: Dict[str, str] = {}

    # ── 1) PostgreSQL 主路径 ──
    try:
        from core.db import get_db_connection
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT DISTINCT
                    CASE
                        WHEN external_userid = %s THEN servicer_userid
                        ELSE external_userid
                    END AS partner
                FROM messages
                WHERE (external_userid = %s OR servicer_userid = %s)
                  AND content IS NOT NULL AND content != ''
                """,
                (userid, userid, userid),
            )
            rows = cur.fetchall()
            for (partner_uid,) in rows:
                if partner_uid and partner_uid != userid:
                    partners.setdefault(partner_uid, _resolve_userid_to_name(partner_uid))
        finally:
            conn.close()
        if partners:
            return list(partners.items())[:limit]
    except Exception as e:
        logger.warning(f"PostgreSQL conversation-partner query failed: {e}")

    # ── 2) txtai 索引扫描回退 ──
    try:
        embeddings = _load_messages_index()
        db = embeddings.database
        if db is None:
            return list(partners.items())[:limit]
        # 一次性取出全部 tags, 关闭游标后再做反查 (避免 SQLite 嵌套游标冲突)
        with _TXTAI_LOCK:
            cur = db.connection.cursor()
            try:
                cur.execute("SELECT id, tags FROM sections")
                rows = cur.fetchall()
            finally:
                cur.close()
        for row in rows:
            tags_raw = row[1]
            try:
                tags = json.loads(tags_raw) if isinstance(tags_raw, str) else dict(tags_raw or {})
            except Exception:
                continue
            ext = tags.get("external_userid")
            serv = tags.get("servicer_userid")
            if ext == userid and serv and serv != userid:
                partners.setdefault(serv, _resolve_userid_to_name(serv))
            elif serv == userid and ext and ext != userid:
                partners.setdefault(ext, _resolve_userid_to_name(ext))
    except Exception as e:
        logger.warning(f"Index conversation-partner scan failed: {e}")

    return list(partners.items())[:limit]



# ═══════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════


@tool
def search_messages(query: Optional[str] = None, sender: Optional[str] = None,
                    receiver: Optional[str] = None, limit: int = 10) -> str:
    """搜索内部聊天消息，返回消息内容和发送者信息。支持按发送者/接收者过滤。

    Use this when the user asks about conversations, chat records, messages,
    or anything that was said or discussed in chats. You can filter by who
    sent the messages (sender) or who received them (receiver).

    Args:
        query: The content search query (e.g. "合同已签", "诈骗链接", "物流报价").
            Omit / leave empty when the user only wants messages from or to a
            specific person (pure metadata filter).
        sender: Sender's full name or userid (e.g. "陳志明"). Only messages
            SENT by this person are returned.
        receiver: Receiver's full name or userid (e.g. "李雅婷"). Only
            messages RECEIVED by this person are returned.
        limit: Max number of results (default 10, max 100).

    Returns:
        A formatted string with search results, each containing:
        message content, sender (userid), label, and score.
    """
    _set_msg_meta({"query": query or "", "expanded_queries": [], "hit_count": 0, "previews": [], "raw_hits": []})

    # ── 姓名 → userid 解析 (确定性, 工具内部完成; 不依赖 LLM) ──────
    sender_uid, sender_note = _resolve_name_to_userid(sender) if sender else (None, "")
    if sender and sender_uid is None:
        return sender_note or f"未找到发送者「{sender}」。"
    receiver_uid, receiver_note = _resolve_name_to_userid(receiver) if receiver else (None, "")
    if receiver and receiver_uid is None:
        return receiver_note or f"未找到接收者「{receiver}」。"

    # ── 构建 origin 感知的过滤谓词 (方向性: origin 3=客户发言, 5=客服发言) ──
    # sender    = (origin=3 AND external_userid=X) OR (origin=5 AND servicer_userid=X)
    # receiver  = (origin=3 AND servicer_userid=X) OR (origin=5 AND external_userid=X)
    filters = []
    if sender_uid:
        filters.append(
            f"(json_extract(tags, '$.origin') = 3 AND json_extract(tags, '$.external_userid') = '{sender_uid}' "
            f"OR json_extract(tags, '$.origin') = 5 AND json_extract(tags, '$.servicer_userid') = '{sender_uid}')"
        )
    if receiver_uid:
        filters.append(
            f"(json_extract(tags, '$.origin') = 3 AND json_extract(tags, '$.servicer_userid') = '{receiver_uid}' "
            f"OR json_extract(tags, '$.origin') = 5 AND json_extract(tags, '$.external_userid') = '{receiver_uid}')"
        )
    where = " AND ".join(filters) if filters else None

    cfg = get_search_config()
    expand = cfg.get("expand", False)
    use_rerank = cfg.get("use_rerank", False)
    graph_parallel = cfg.get("graph_parallel", False)

    try:
        embeddings = _load_messages_index()
    except FileNotFoundError as e:
        return str(e)

    try:
        content = (query or "").strip()
        if not content and not where:
            return "请提供搜索内容关键词，或指定 sender/receiver 过滤条件。"

        if not content:
            # ── 纯过滤模式 (filter-only): 无内容关键词, 跳过 similar() ──
            sql = "select id, text from txtai where " + where
            try:
                raw = _locked_search(embeddings, sql, limit=min(int(limit), MAX_SEARCH_LIMIT))
            except Exception as e:
                logger.warning(f"Filter-only search failed: {e}")
                raw = []
            docs = _parse_msg_docs(embeddings, raw)
        elif expand:
            from .query_expander import QueryExpander
            try:
                expander = QueryExpander()
                queries_with_weights = expander.expand(content)
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
                queries_with_weights = [(content, 1.0)]
            _update_msg_meta(expanded_queries=[q for q, _ in queries_with_weights if q != content])
            docs = _weighted_rrf_fuse(embeddings, queries_with_weights,
                                      limit=int(limit) or 10, graph_parallel=graph_parallel, where=where)
        else:
            if graph_parallel:
                docs = _weighted_rrf_fuse(embeddings, [(content, 1.0)], limit=int(limit) or 10,
                                          graph_parallel=True, where=where)
            else:
                if where:
                    sql = ("select id, text, score from txtai where similar(:q) and " + where)
                    raw = _locked_search(embeddings, sql, parameters={"q": _segment(content)},
                                            limit=min(int(limit), MAX_SEARCH_LIMIT))
                else:
                    segmented = _segment(content)
                    raw = _locked_search(embeddings, segmented, limit=min(int(limit), MAX_SEARCH_LIMIT))
                docs = _parse_msg_docs(embeddings, raw)

        if use_rerank:
            from .reranker import Reranker
            try:
                reranker = Reranker()
                docs = reranker.rerank(content or (query or ""), docs)
            except Exception as e:
                logger.warning(f"Rerank failed: {e}")
    except Exception as e:
        logger.warning(f"Messages search failed: {e}")
        return f"Search failed: {e}"

    if not docs:
        return "No relevant messages found."

    _update_msg_meta(
        hit_count=len(docs),
        previews=[
            {
                "text": (d.get("text", "") or "")[:200],
                "sender": (d.get("metadata", {}).get("customer_name")
                           or d.get("metadata", {}).get("external_userid") or "?"),
                "score": round(float(d.get("score", 0.0)), 4),
            }
            for d in docs[:5]
        ],
    )
    # 原始结果 (含 metadata) — 供 Hindsight 记忆图谱等下游使用
    _update_msg_meta(raw_hits=[
        {"id": d.get("id", ""), "text": (d.get("text", "") or "")[:300],
         "score": float(d.get("score", 0.0)),
         "metadata": d.get("metadata", {}) or {}}
        for d in docs[:10]
    ])

    lines = ["【消息搜索结果】"]
    for i, d in enumerate(docs, 1):
        meta = d.get("metadata", {})
        label = meta.get("label", "-")
        text = d.get("text", "")
        score = d.get("score", 0.0)
        # 真实发送者: origin=3 → 客户(external_userid); origin=5 → 客服(servicer_userid)。
        # customer_name 是 external 侧的 join 名, 不能直接当发送者。
        origin = meta.get("origin")
        try:
            is_customer = int(origin) == 3
        except (TypeError, ValueError):
            is_customer = True
        if is_customer:
            sender_name = meta.get("customer_name") or meta.get("external_userid", "?")
            userid = meta.get("external_userid", "")
        else:
            sender_name = _resolve_userid_to_name(meta.get("servicer_userid") or "") or meta.get("servicer_userid") or "?"
            userid = meta.get("servicer_userid", "")
        lines.append(
            f"\n{i}. [Score: {score:.4f}] {sender_name} (userid: {userid}) "
            f"[Label: {label}]\n"
            f"   {text[:200]}"
        )

    return "\n".join(lines)


@tool
def search_contacts(query: str) -> str:
    """搜索联系人信息，返回邮箱、公司、职位、电话等。

    Use this when the user asks about a person's contact details such as
    email, phone number, company name, or job title. Also use this when
    you need to find a userid by name, or look up who a person is.

    Args:
        query: The search query (e.g. "李雅婷", "陳志明 email", "johnsonj")

    Returns:
        A formatted string with contact details: full_name, userid,
        email, company, phone, job_title.
    """
    try:
        embeddings = _load_contacts_index()
    except FileNotFoundError as e:
        return str(e)

    _set_contact_meta({"query": query, "hit_count": 0, "previews": []})

    try:
        segmented = _segment(query)
        raw = _locked_search(embeddings, segmented, limit=5)
    except Exception as e:
        logger.warning(f"Contacts search failed: {e}")
        return f"Search failed: {e}"

    if not raw:
        return "No matching contacts found."

    previews = []
    lines = ["【联系人搜索结果】"]
    for i, item in enumerate(raw, 1):
        doc_id = item.get("id", "") if isinstance(item, dict) else (item[0] if isinstance(item, tuple) else "")
        score = item.get("score", 0.0) if isinstance(item, dict) else (item[3] if isinstance(item, tuple) and len(item) >= 4 else 0.0)
        meta = _fetch_doc_metadata(embeddings, doc_id) if doc_id else {}
        if i <= 5:
            previews.append({
                "name": meta.get("full_name", "?"),
                "email": meta.get("email", "-"),
                "score": round(float(score), 4),
            })
        lines.append(
            f"\n{i}. [Score: {score:.4f}] {meta.get('full_name', '?')} "
            f"(userid: {meta.get('userid', '?')})\n"
            f"   Email: {meta.get('email', '-')}\n"
            f"   Company: {meta.get('company', '-')}\n"
            f"   Phone: {meta.get('phone', '-')}\n"
            f"   Job Title: {meta.get('job_title', '-')}"
        )

    _update_contact_meta(hit_count=len(raw), previews=previews)
    return "\n".join(lines)


@tool
def search_conversation_partners(person: str) -> str:
    """查询某人与谁有过对话 (会话级关系, 返回对侧联系人列表)。

    Use this when the user asks "who did X talk to", "跟谁聊过", "和谁对话",
    or wants to know the OTHER party of X's conversations. This is a
    relationship query, not a content search — it looks up all conversations
    X participated in (as either customer or servicer) and returns the
    distinct people on the other side.

    Args:
        person: 姓名或 userid (e.g. "陳志明")

    Returns:
        A formatted list of distinct conversation partners (name + userid).
    """
    userid, note = _resolve_name_to_userid(person)
    if userid is None:
        return note or f"未找到联系人「{person}」。"

    partners = _find_conversation_partners(userid, limit=20)
    if not partners:
        return f"未找到 {person} 的会话记录。"

    lines = ["【会话关系查询】"]
    for i, (partner_uid, partner_name) in enumerate(partners, 1):
        lines.append(f"\n{i}. {partner_name} (userid: {partner_uid})")
    return "\n".join(lines)


# ── SQL 结构化检索 (messages 精确过滤, 不走向量语义) ──────────────
_SQL_TABLES = ("messages", "contacts")


def _run_messages_sql(sql: str) -> List[Dict[str, Any]]:
    """Execute a read-only SELECT on the messages table; returns rows as dicts."""
    from core.db import get_db_connection
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


def _validate_sql(sql: str) -> Optional[str]:
    """Validate an LLM-generated SELECT: single SELECT, allowed tables, read-only.

    Returns the cleaned SQL (with a LIMIT guard) or None when unsafe.
    """
    if not sql:
        return None
    s = sql.strip().rstrip(";").strip()
    low = s.lower()
    if not low.startswith("select"):
        return None
    for bad in ("insert", "update", "delete", "drop", "alter", "grant",
                "create", "truncate", "copy", "--", "/*"):
        if bad in low:
            return None
    tables = re.findall(r"\bfrom\s+(\w+)", low)
    if not tables or any(t not in _SQL_TABLES for t in tables):
        return None
    if "limit" not in low:
        s += " LIMIT 20"
    return s


def _condition_to_sql(condition: str) -> Optional[str]:
    """规则映射自然语言条件 → SQL (确定性, 不依赖 LLM)。"""
    c = condition.lower()
    url_words = ("link", "url", "http", "www", "網址", "網址", "链接", "連結", "网站", "網站", "网络", "網路")
    if any(w in c for w in url_words):
        return (
            "SELECT msgid, external_userid, send_time, label, content "
            "FROM messages "
            "WHERE content ILIKE '%http://%' OR content ILIKE '%https://%' "
            "OR content ILIKE '%www.%' OR content ~ '(https?://|www\\.)[^ ]*' "
            "ORDER BY send_time DESC LIMIT 20"
        )
    m = re.search(r"(?:label|标签|標籤)\s*[:=]?\s*['\"]?([a-z_]+)", c)
    if m:
        return (
            "SELECT msgid, external_userid, send_time, label, content "
            f"FROM messages WHERE label = '{m.group(1)}' "
            "ORDER BY send_time DESC LIMIT 20"
        )
    return None


def _llm_condition_to_sql(condition: str) -> Optional[str]:
    """Text-to-SQL 兜底: LLM 生成 messages 表的 SELECT, 经严格校验后执行。"""
    try:
        from .litellm_client import LiteLLMClient
        client = LiteLLMClient()
        result = client.chat(
            [
                {"role": "system", "content": (
                    "You convert a natural-language filter into a PostgreSQL SELECT. "
                    "Table: messages(msgid, open_kfid, external_userid, send_time, origin, "
                    "servicer_userid, msgtype, content, label). "
                    "Reply with ONLY the SQL statement. Read-only SELECT only, no semicolons."
                )},
                {"role": "user", "content": condition},
            ],
            temperature=0.0, max_tokens=160, timeout=8,
        )
        return _validate_sql(result or "")
    except Exception as e:
        logger.warning(f"Text-to-SQL failed: {e}")
        return None


def _scan_messages_index(condition: str) -> List[Dict[str, Any]]:
    """DB 不可用时的回退: 扫描 txtai 消息索引, 按正则精确匹配 (URL/链接)。

    与语义向量检索不同, 这是对全量索引的确定性模式匹配。
    """
    pattern = None
    c = condition.lower()
    url_words = ("link", "url", "http", "www", "網址", "網址", "链接", "連結", "网站", "網站")
    if any(w in c for w in url_words):
        pattern = re.compile(r"(https?://|www\.)", re.I)
    if pattern is None:
        return []

    embeddings = _load_messages_index()
    try:
        docs = _locked_search(embeddings, "*", limit=embeddings.count() + 1)
    except Exception:
        docs = _locked_search(embeddings, "", limit=embeddings.count() + 1)

    rows = []
    for d in docs:
        text = str(d.get("text", "") or "")
        if pattern.search(text):
            meta = _fetch_doc_metadata(embeddings, d.get("id", "")) if d.get("id") else {}
            rows.append({
                "msgid": d.get("id", ""),
                "external_userid": (meta.get("external_userid") or meta.get("customer_name") or ""),
                "send_time": meta.get("send_time", ""),
                "label": meta.get("label", ""),
                "content": text,
            })
    return rows


@tool
def search_messages_where(condition: str) -> str:
    """按结构化条件精确检索消息 (SQL 过滤, 非语义向量)。

    Use this when the user asks for messages that match an exact property:
    contains a link/URL (含链接/網址), a specific label, a specific sender,
    or any condition expressible as a SQL filter. Returns matching message
    rows (sender, label, time, content) — grounded in the DB, no guessing.

    Args:
        condition: 自然语言条件, e.g. "messages containing a link",
                   "label = fraud", "含網址的消息"
    """
    sql = _condition_to_sql(condition)
    if sql is None:
        sql = _llm_condition_to_sql(condition)
    if sql is None:
        return "无法理解该结构化条件。"

    _set_msg_meta({"query": condition, "expanded_queries": [], "hit_count": 0, "previews": [], "raw_hits": []})

    try:
        rows = _run_messages_sql(sql)
    except Exception as e:
        logger.warning(f"Structured messages SQL failed ({e}) — falling back to index scan")
        rows = _scan_messages_index(condition)

    if not rows:
        return "No matching messages found."

    _update_msg_meta(
        hit_count=len(rows),
        previews=[
            {"text": (r.get("content", "") or "")[:200],
             "sender": r.get("external_userid", "?"),
             "score": 0.0}
            for r in rows[:5]
        ],
    )
    _update_msg_meta(raw_hits=[
        {"id": str(r.get("msgid", "")), "text": (r.get("content", "") or "")[:300],
         "score": 0.0,
         "metadata": {"customer_name": r.get("external_userid", ""), "label": r.get("label", ""),
                      "send_time": r.get("send_time")}}
        for r in rows[:10]
    ])

    lines = ["【结构化匹配】"]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"\n{i}. [Match] {r.get('external_userid', '?')} "
            f"(userid: {r.get('external_userid', '?')}) "
            f"[Label: {r.get('label', '-')}] [{r.get('send_time', '')}]\n"
            f"   {r.get('content', '')}"
        )
    return "\n".join(lines)


# ── Export tool list ─────────────────────────────────────────────
# 主路径工具集: 3 个 (DeepSeek 通过 LangGraph 绑定)。search_messages_where
# 保留用于降级路径 (text-to-SQL), 不暴露给主路径 LLM (路由边界更清晰)。
CROSS_TABLE_TOOLS = [search_messages, search_contacts, search_conversation_partners]