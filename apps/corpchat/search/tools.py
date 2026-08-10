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
from typing import Any, Dict, List, Optional, Tuple

import txtai
from langchain_core.tools import tool

from .config import (
    CONTACTS_INDEX_PATH,
    DEFAULT_INDEX_PATH,
    logger,
)
from .utils import _segment


# ── Module-level index cache ─────────────────────────────────────
_contacts_embeddings: Optional[txtai.Embeddings] = None
_messages_embeddings: Optional[txtai.Embeddings] = None

# ── Structured metadata of the most recent tool call ─────────────
# Captures what was actually searched so the Process window can show
# per-tool details (query, expanded queries, hit count, previews).
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


def get_last_search_meta() -> Dict[str, Any]:
    """Return structured metadata of the most recent search_messages call."""
    return dict(_last_msg_meta)


def get_last_contact_meta() -> Dict[str, Any]:
    """Return structured metadata of the most recent search_contacts call."""
    return dict(_last_contact_meta)



def _load_messages_index() -> txtai.Embeddings:
    """Lazy-load messages txtai index."""
    global _messages_embeddings
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
    """Lazy-load contacts txtai index."""
    global _contacts_embeddings
    if _contacts_embeddings is not None:
        return _contacts_embeddings
    path = os.environ.get("CONTACTS_INDEX_PATH", CONTACTS_INDEX_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Contacts index not found at {path}. Run `python search.py build-contacts` first.")
    _contacts_embeddings = txtai.Embeddings()
    _contacts_embeddings.load(path)
    logger.info(f"Loaded contacts index ({_contacts_embeddings.count()} contacts)")
    return _contacts_embeddings


def _fetch_doc_metadata(embeddings: txtai.Embeddings, doc_id: str) -> Dict[str, Any]:
    """Fetch metadata (tags) for a document by ID from the txtai SQLite store."""
    try:
        db = embeddings.database
        if db is None:
            return {}
        conn = db.connection
        cur = conn.cursor()
        cur.execute("SELECT tags FROM sections WHERE id = ?", (doc_id,))
        row = cur.fetchone()
        if row and row[0]:
            return json.loads(row[0]) if isinstance(row[0], str) else dict(row[0])
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
                       graph_parallel: bool = False) -> List[Dict[str, Any]]:
    """Search each expanded query and merge via weighted RRF fusion.

    Mirrors the non-agent Searcher pipeline so an agentic tool call is
    treated as an individual full search.

    graph_parallel: append the structural graph-traversal path (Hindsight
    graph evidence) to the fusion inputs, mirroring Searcher.search(...,
    graph_parallel=True).
    """
    from .searcher import Searcher

    all_results: List[Tuple[List[Tuple[str, float]], float]] = []
    for q, q_weight in queries_with_weights:
        try:
            raw = embeddings.search(_segment(q), limit=limit * 3)
        except Exception as e:
            logger.warning(f"Expanded query search failed ({q!r}): {e}")
            raw = []
        result_list: List[Tuple[str, float]] = []
        for item in raw:
            doc_id = item.get("id", "") if isinstance(item, dict) else (item[0] if isinstance(item, tuple) else "")
            score = item.get("score", 0.0) if isinstance(item, dict) else (
                item[3] if isinstance(item, tuple) and len(item) >= 4 else 0.0)
            if doc_id:
                result_list.append((doc_id, score))
        all_results.append((result_list, q_weight))

    # 图并行检索路 (opt-in): 结构邻居作为独立证据参与融合
    if graph_parallel:
        original_query = queries_with_weights[0][0] if queries_with_weights else ""
        entry = Searcher(embeddings)._graph_parallel_entry(original_query, limit=limit)
        if entry is not None:
            all_results.append(entry)

    fused = Searcher._weighted_rrf_fusion(all_results)
    output: List[Dict[str, Any]] = []
    seen = set()
    for doc_id, _ in fused:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc = {"id": doc_id, "text": "", "score": 0.0, "metadata": _fetch_doc_metadata(embeddings, doc_id)}
        try:
            db = embeddings.database
            if db is not None:
                cur = db.connection.cursor()
                cur.execute("SELECT text FROM sections WHERE id = ?", (doc_id,))
                row = cur.fetchone()
                if row:
                    doc["text"] = row[0] or ""
        except Exception:
            pass
        output.append(doc)
        if len(output) >= limit:
            break
    return output



# ═══════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════


@tool
def search_messages(query: str, expand: bool = False, use_rerank: bool = False,
                    graph_parallel: bool = False) -> str:
    """搜索内部聊天消息，返回消息内容和发送者信息。

    Use this when the user asks about conversations, chat records, messages,
    or anything that was said or discussed in chats. Returns message content,
    sender name, label, and userid for each result.

    Args:
        query: The search query (e.g. "合同已签", "诈骗链接", "物流报价")
        expand: Whether to run LLM query expansion + RRF fusion (optional).
        use_rerank: Whether to cross-encoder rerank the results (optional).
        graph_parallel: Treat graph traversal as a parallel RRF fusion path
            (structural neighbors; works with or without expand).

    Returns:
        A formatted string with search results, each containing:
        message content, sender (userid), label, and score.
    """
    global _last_msg_meta
    _last_msg_meta = {"query": query, "expanded_queries": [], "hit_count": 0, "previews": []}

    try:
        embeddings = _load_messages_index()
    except FileNotFoundError as e:
        return str(e)

    try:
        if expand:
            from .query_expander import QueryExpander
            try:
                expander = QueryExpander()
                queries_with_weights = expander.expand(query)
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
                queries_with_weights = [(query, 1.0)]
            _last_msg_meta["expanded_queries"] = [q for q, _ in queries_with_weights if q != query]
            docs = _weighted_rrf_fuse(embeddings, queries_with_weights, limit=10, graph_parallel=graph_parallel)
        else:
            if graph_parallel:
                # 与 Searcher Path A 一致: 直接结果 + 图路 RRF 融合 (无扩展也可用图路)
                docs = _weighted_rrf_fuse(embeddings, [(query, 1.0)], limit=10, graph_parallel=True)
            else:
                segmented = _segment(query)
                raw = embeddings.search(segmented, limit=10)
                docs = _parse_msg_docs(embeddings, raw)

        if use_rerank:
            from .reranker import Reranker
            try:
                reranker = Reranker()
                docs = reranker.rerank(query, docs)
            except Exception as e:
                logger.warning(f"Rerank failed: {e}")
    except Exception as e:
        logger.warning(f"Messages search failed: {e}")
        return f"Search failed: {e}"

    if not docs:
        return "No relevant messages found."

    _last_msg_meta["hit_count"] = len(docs)
    _last_msg_meta["previews"] = [
        {
            "text": (d.get("text", "") or "")[:200],
            "sender": (d.get("metadata", {}).get("customer_name")
                       or d.get("metadata", {}).get("external_userid") or "?"),
            "score": round(float(d.get("score", 0.0)), 4),
        }
        for d in docs[:5]
    ]
    # 原始结果 (含 metadata) — 供 Hindsight 记忆图谱等下游使用
    _last_msg_meta["raw_hits"] = [
        {"id": d.get("id", ""), "text": (d.get("text", "") or "")[:300],
         "score": float(d.get("score", 0.0)),
         "metadata": d.get("metadata", {}) or {}}
        for d in docs[:10]
    ]

    lines = ["【消息搜索结果】"]
    for i, d in enumerate(docs, 1):
        meta = d.get("metadata", {})
        customer_name = meta.get("customer_name", meta.get("external_userid", "?"))
        label = meta.get("label", "-")
        userid = meta.get("external_userid", "")
        text = d.get("text", "")
        score = d.get("score", 0.0)
        lines.append(
            f"\n{i}. [Score: {score:.4f}] {customer_name} (userid: {userid}) "
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

    global _last_contact_meta
    _last_contact_meta = {"query": query, "hit_count": 0, "previews": []}

    try:
        segmented = _segment(query)
        raw = embeddings.search(segmented, limit=5)
    except Exception as e:
        logger.warning(f"Contacts search failed: {e}")
        return f"Search failed: {e}"

    if not raw:
        return "No matching contacts found."

    _last_contact_meta["hit_count"] = len(raw)
    _last_contact_meta["previews"] = []
    lines = ["【联系人搜索结果】"]
    for i, item in enumerate(raw, 1):
        doc_id = item.get("id", "") if isinstance(item, dict) else (item[0] if isinstance(item, tuple) else "")
        score = item.get("score", 0.0) if isinstance(item, dict) else (item[3] if isinstance(item, tuple) and len(item) >= 4 else 0.0)
        meta = _fetch_doc_metadata(embeddings, doc_id) if doc_id else {}
        if i <= 5:
            _last_contact_meta["previews"].append({
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

    return "\n".join(lines)


# ── Export tool list ─────────────────────────────────────────────
CROSS_TABLE_TOOLS = [search_messages, search_contacts]