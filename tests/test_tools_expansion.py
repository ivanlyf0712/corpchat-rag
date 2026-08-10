"""
Ticket 01 — Agentic message-search expansion + rerank.

Tests that:
  - search_messages accepts expand/use_rerank and runs expansion + rerank
    when enabled, exposing expanded queries / hit count / previews.
  - search_contacts never expands, even when toggles are ON.
  - CrossTableAgent forwards expand/use_rerank to search_messages only.

Run:
    conda run -n ocr pytest tests/test_tools_expansion.py -v
"""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from apps.corpchat.search import tools as tools_module
from apps.corpchat.search.tools import search_messages, search_contacts, search_messages_where


# ── Fake txtai embeddings ─────────────────────────────────────────
class _FakeCursor:
    def __init__(self):
        self._sql = ""

    def execute(self, sql, *args):
        self._sql = sql
        return None

    def fetchone(self):
        if "tags" in self._sql:
            return (json.dumps({
                "label": "sample_request",
                "customer_name": "陳志明",
                "external_userid": "user_1",
                "full_name": "陳志明",
                "userid": "user_1",
                "email": "weiyao@example.org",
                "company": "聯成電腦",
                "phone": "0912345678",
                "job_title": "採購專員",
            }, ensure_ascii=False),)
        return ("合同已签，请确认后安排付款。",)

    def fetchall(self):
        return []

    def close(self):
        pass


class _FakeDB:
    connection = type("Conn", (), {"cursor": staticmethod(lambda: _FakeCursor())})


class _FakeEmbeddings:
    database = _FakeDB()

    def search(self, q, limit=10, weights=None):
        return [
            {"id": "doc_1", "text": "合同已签，请确认后安排付款。", "score": 0.61},
            {"id": "doc_2", "text": "合同已签，下一步做什么？", "score": 0.55},
            {"id": "doc_3", "text": "报价已发送，请查收。", "score": 0.40},
        ]

    def count(self):
        return 3


# ── Fakes for expander / reranker ────────────────────────────────
class _RecordingExpander:
    """Deterministic expander that records calls and returns expansions."""

    def __init__(self):
        self.calls = []
        self._expansions = [(("合同已签", 0.5), ("合约确认", 1.3), ("合同 签署", 1.0))]

    def expand(self, query, use_cache=True):
        self.calls.append(query)
        return list(self._expansions[0])


class _RecordingReranker:
    def __init__(self):
        self.calls = 0
        self.enabled = True

    def rerank(self, query, docs):
        self.calls += 1
        return list(docs)


@pytest.fixture(autouse=True)
def _use_fake_index(monkeypatch):
    monkeypatch.setattr(tools_module, "_load_messages_index", lambda: _FakeEmbeddings())
    monkeypatch.setattr(tools_module, "_load_contacts_index", lambda: _FakeEmbeddings())


@pytest.fixture
def fake_expander():
    return _RecordingExpander()


@pytest.fixture
def fake_reranker():
    return _RecordingReranker()


# ── search_messages tool ─────────────────────────────────────────
class TestSearchMessagesTool:
    def test_default_no_expand_no_rerank(self):
        """Defaults: expansion and rerank disabled, meta has no expansions."""
        result = search_messages.invoke({"query": "合同已签"})
        meta = tools_module.get_last_search_meta()
        assert "合同已签" in result
        assert meta["expanded_queries"] == []
        assert meta["hit_count"] == 3
        assert len(meta["previews"]) == 3

    def test_expand_enabled_records_expanded_queries(self, monkeypatch, fake_expander, fake_reranker):
        """expand=True → expanded queries appear in meta; expander was called."""
        monkeypatch.setattr("apps.corpchat.search.query_expander.QueryExpander",
                            lambda *a, **k: fake_expander)
        search_messages.invoke({"query": "合同已签", "expand": True, "use_rerank": False})
        meta = tools_module.get_last_search_meta()
        assert fake_expander.calls == ["合同已签"], f"Expander not called: {fake_expander.calls}"
        assert meta["expanded_queries"] == ["合约确认", "合同 签署"]
        assert meta["hit_count"] == 3

    def test_rerank_enabled_calls_reranker(self, monkeypatch, fake_reranker):
        """use_rerank=True → Reranker invoked; results still returned."""
        monkeypatch.setattr("apps.corpchat.search.reranker.Reranker",
                            lambda *a, **k: fake_reranker)
        result = search_messages.invoke({"query": "合同已签", "expand": False, "use_rerank": True})
        assert fake_reranker.calls == 1, "Reranker not called"
        assert "合同已签" in result

    def test_expansion_failure_falls_back_to_plain_query(self, monkeypatch):
        """Expander raising → graceful fallback to single-query search."""
        class _BoomExpander:
            def expand(self, query, use_cache=True):
                raise RuntimeError("LLM down")

        monkeypatch.setattr("apps.corpchat.search.query_expander.QueryExpander",
                            lambda *a, **k: _BoomExpander())
        result = search_messages.invoke({"query": "合同已签", "expand": True, "use_rerank": False})
        meta = tools_module.get_last_search_meta()
        assert "合同已签" in result
        assert meta["expanded_queries"] == []
        assert meta["hit_count"] == 3

# ── search_contacts tool ─────────────────────────────────────────
class TestSearchContactsTool:
    def test_contacts_never_expands(self, monkeypatch, fake_expander, fake_reranker):
        """search_contacts has no expand/use_rerank knobs and stays exact."""
        monkeypatch.setattr("apps.corpchat.search.query_expander.QueryExpander",
                            lambda *a, **k: fake_expander)
        monkeypatch.setattr("apps.corpchat.search.reranker.Reranker",
                            lambda *a, **k: fake_reranker)
        result = search_contacts.invoke({"query": "陳志明"})
        meta = tools_module.get_last_contact_meta()
        assert fake_expander.calls == [], "Contacts must not expand"
        assert fake_reranker.calls == 0, "Contacts must not rerank"
        assert "陳志明" in result
        assert meta["hit_count"] == 3


# ── CrossTableAgent forwarding ───────────────────────────────────
class TestAgentForwarding:
    def test_agent_forwards_expand_and_rerank_to_messages_only(self, monkeypatch):
        """CrossTableAgent forwards expand/use_rerank to search_messages but not contacts."""
        import types
        from apps.corpchat.search.cross_table_agent import CrossTableAgent

        msg_kwargs = {}
        contact_kwargs = {}

        def _fake_msg_invoke(payload):
            msg_kwargs.update(payload)
            return "【消息搜索结果】\n1. [Score: 0.61] 陳志明 (userid: user_1) [Label: sample_request]\n   合同已签"

        def _fake_contact_invoke(payload):
            contact_kwargs.update(payload)
            return "【联系人搜索结果】\n1. [Score: 0.9] 陳志明 (userid: user_1)\n   Email: x@y.org"

        # The agent does `from .tools import search_messages` inside process(),
        # so patching the tools module attributes is picked up.
        monkeypatch.setattr(tools_module, "search_messages", types.SimpleNamespace(invoke=_fake_msg_invoke))
        monkeypatch.setattr(tools_module, "search_contacts", types.SimpleNamespace(invoke=_fake_contact_invoke))

        agent = CrossTableAgent(expand=True, use_rerank=True)
        agent._extract_search_query = lambda q: "合同已签"
        result = agent.process("帮我查一下合同已签的消息")

        assert msg_kwargs.get("expand") is True, f"search_messages not told to expand: {msg_kwargs}"
        assert msg_kwargs.get("use_rerank") is True, f"search_messages not told to rerank: {msg_kwargs}"
        assert "expand" not in contact_kwargs, f"search_contacts got expand: {contact_kwargs}"
        assert result.get("success") is True

    def test_agent_disabled_toggles_not_forwarded(self, monkeypatch):
        """expand/use_rerank False → search_messages invoked with False."""
        import types
        from apps.corpchat.search.cross_table_agent import CrossTableAgent

        msg_kwargs = {}

        def _fake_msg_invoke(payload):
            msg_kwargs.update(payload)
            return "【消息搜索结果】\n1. [Score: 0.61] 陳志明 (userid: user_1) [Label: sample_request]\n   合同已签"

        def _fake_contact_invoke(payload):
            return "【联系人搜索结果】\n1. [Score: 0.9] 陳志明 (userid: user_1)\n   Email: x@y.org"

        monkeypatch.setattr(tools_module, "search_messages", types.SimpleNamespace(invoke=_fake_msg_invoke))
        monkeypatch.setattr(tools_module, "search_contacts", types.SimpleNamespace(invoke=_fake_contact_invoke))

        agent = CrossTableAgent(expand=False, use_rerank=False)
        agent._extract_search_query = lambda q: "合同已签"
        result = agent.process("帮我查一下合同已签的消息")

        assert msg_kwargs.get("expand") is False
        assert msg_kwargs.get("use_rerank") is False


# ═══════════════ Ticket 02: graph_parallel threading (tool) ═══════════════
def test_search_messages_accepts_and_forwards_graph_parallel(monkeypatch):
    """search_messages 接受 graph_parallel 并透传给 RRF 融合路径 (expand 分支)。"""
    calls = []

    def _recording_fuse(embeddings, queries_with_weights, limit=10, **kw):
        calls.append(kw)
        return []

    monkeypatch.setattr(tools_module, "_weighted_rrf_fuse", _recording_fuse)
    search_messages.invoke({"query": "跟誰聊過物流", "expand": True, "graph_parallel": True})

    assert calls, "expand 分支应调用 _weighted_rrf_fuse"
    assert calls[0].get("graph_parallel") is True, f"graph_parallel 未透传: {calls[0]}"


def test_cross_table_agent_forwards_graph_parallel(monkeypatch):
    """CrossTableAgent(graph_parallel=True) → search_messages 带 graph_parallel=True。"""
    import types
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    msg_kwargs = {}

    def _fake_msg_invoke(payload):
        msg_kwargs.update(payload)
        return "【消息搜索结果】\n1. [Score: 0.61] 陳志明 (userid: user_1) [Label: sample_request]\n   合同已签"

    monkeypatch.setattr(tools_module, "search_messages", types.SimpleNamespace(invoke=_fake_msg_invoke))
    monkeypatch.setattr(tools_module, "search_contacts", types.SimpleNamespace(invoke=lambda p: ""))

    agent = CrossTableAgent(graph_parallel=True)
    agent._extract_search_query = lambda q: "跟誰聊過物流"
    agent.process("跟誰聊過物流")

    assert msg_kwargs.get("graph_parallel") is True, f"graph_parallel 未转发: {msg_kwargs}"


def test_format_citations_builds_source_block():
    """_format_citations 从结果 metadata 构建来源块; 空结果返回空串。"""
    from apps.corpchat.search.utils import _format_citations

    results = [{
        "id": "m1", "text": "物流報價 100 元", "score": 0.9,
        "metadata": {"customer_name": "陳志明", "send_time": "2026-08-01T10:00:00", "label": "product_inquiry"},
    }]
    block = _format_citations(results)
    assert "陳志明" in block and "2026-08-01" in block and "product_inquiry" in block
    assert _format_citations([]) == ""


def test_cross_table_agent_contacts_gated_by_sources(monkeypatch):
    """sources 不含 contacts → 不调用 search_contacts (跨表查询仍走消息)。"""
    import types
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    calls = []

    def _fake_msg_invoke(payload):
        calls.append("msg")
        return "【消息搜索结果】\n1. [Score: 0.61] 陳志明 (userid: user_1)\n   合同已签"

    def _fake_contact_invoke(payload):
        calls.append("contact")
        return "【联系人搜索结果】"

    monkeypatch.setattr(tools_module, "search_messages", types.SimpleNamespace(invoke=_fake_msg_invoke))
    monkeypatch.setattr(tools_module, "search_contacts", types.SimpleNamespace(invoke=_fake_contact_invoke))

    agent = CrossTableAgent(sources=["messages"])
    agent._extract_search_query = lambda q: "發'合同已簽'消息的人"
    agent.process("發'合同已簽'消息的人，他的聯絡方式")

    assert "msg" in calls, "messages 源应被搜索"
    assert "contact" not in calls, f"sources 排除 contacts 仍调用了 search_contacts: {calls}"


def test_contact_name_query_routes_to_contacts():
    """'姓名/联系人' 类查询应路由到 search_contacts, 而非默认回退 search_messages。"""
    from apps.corpchat.search.cross_table_agent import _LiteLLMWrapper

    w = _LiteLLMWrapper(api_base="", api_key="", model="test")
    queries = [
        "Give me one example of male's name in the contact",
        "给我一个男生的姓名",
        "联系人的名字",
    ]
    for q in queries:
        names = [c["name"] for c in w._decide_tool_calls(q)]
        assert "search_contacts" in names, f"{q} 应路由到 contacts: {names}"
        assert "search_messages" not in names, f"{q} 不应路由到 messages: {names}"


def test_search_messages_graph_parallel_without_expand_calls_graph_path(monkeypatch):
    """expand=False + graph_parallel=True 也调用图并行路 (与 Searcher Path A 一致)。

    修复: 之前工具在 expand=False 时静默忽略 graph_parallel, 与 Searcher 不一致。
    """
    from apps.corpchat.search.searcher import Searcher

    called = []

    def _fake_entry(self, query, limit):
        called.append(query)
        return None  # 无图结果 → 回退直接文档

    monkeypatch.setattr(Searcher, "_graph_parallel_entry", _fake_entry)

    result = search_messages.invoke({"query": "跟誰聊過物流", "graph_parallel": True})
    assert called == ["跟誰聊過物流"], f"图并行入口应被调用: {called}"
    assert "【消息搜索结果】" in result, "图路为空时应回退直接搜索"


# ═══════════════ Memory graph data source (raw_hits) ═══════════════
def test_search_messages_meta_includes_raw_hits():
    """search_messages 的 meta 含 raw_hits (带 metadata), 供记忆图谱使用。"""
    from apps.corpchat.search.tools import get_last_search_meta

    search_messages.invoke({"query": "合同已签"})
    meta = get_last_search_meta()
    assert "raw_hits" in meta, "meta 应暴露 raw_hits"
    assert meta["raw_hits"], "raw_hits 应被填充"
    first = meta["raw_hits"][0]
    assert first.get("metadata", {}).get("customer_name") == "陳志明"


def test_cross_table_agent_exposes_raw_hits(monkeypatch):
    """CrossTableAgent.process 返回 search_messages 的原始结果 (含 metadata)。

    确保 agent 路径 (深度模式) 的记忆图谱也有数据来源。
    """
    import types
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    def _fake_msg_invoke(payload):
        return "【消息搜索结果】\n1. [Score: 0.61] 高健銘 (userid: user_1) [Label: old_friend_reconnect]\n   合同已签"

    def _fake_contact_invoke(payload):
        return ""

    monkeypatch.setattr(tools_module, "search_messages", types.SimpleNamespace(invoke=_fake_msg_invoke))
    monkeypatch.setattr(tools_module, "search_contacts", types.SimpleNamespace(invoke=_fake_contact_invoke))
    monkeypatch.setattr(tools_module, "_last_msg_meta", {
        "query": "合同已签", "expanded_queries": [], "hit_count": 1, "previews": [],
        "raw_hits": [{"id": "m1", "text": "合同已签", "score": 0.61,
                      "metadata": {"customer_name": "高健銘", "company": "DCH",
                                   "label": "old_friend_reconnect", "open_kfid": "k1"}}],
    })

    agent = CrossTableAgent(expand=False, use_rerank=False)
    agent._extract_search_query = lambda q: "合同已签"
    result = agent.process("帮我查一下合同已签的消息")

    hits = result.get("raw_hits", [])
    assert len(hits) == 1, f"agent 应暴露 raw_hits: {hits}"
    assert hits[0]["metadata"]["customer_name"] == "高健銘"
    assert result.get("success") is True


# ═══════════════ SQL 结构化检索 (search_messages_where) ═══════════════
def test_structured_link_query_routes_to_search_messages_where():
    """含 link/URL 的消息查询 → SQL 结构化检索, 不走语义向量。"""
    from apps.corpchat.search.cross_table_agent import _LiteLLMWrapper
    w = _LiteLLMWrapper(api_base="", api_key="", model="test")

    for q in ("search for messages containing a link please",
              "search for messages containing a URL please",
              "找一下含網址的消息"):
        names = [c["name"] for c in w._decide_tool_calls(q)]
        assert "search_messages_where" in names, f"{q!r} should route to structured tool: {names}"

    # 非结构化查询不受影响
    assert w._decide_tool_calls("李雅婷的邮箱是什么")[0]["name"] == "search_contacts"


def test_search_messages_where_returns_db_rows(monkeypatch):
    """结构化工具经 SQL 返回真实匹配消息 (含链接), 不依赖语义检索。"""
    import core.db

    class _Cur:
        description = [("msgid",), ("external_userid",), ("send_time",), ("label",), ("content",)]
        def execute(self, sql):
            pass
        def fetchall(self):
            return [
                ("m1", "user_高健銘_i-chunli", "2026-01-01T10:00:00", "old_friend_reconnect",
                 "看看这个链接 https://tinyurl.com/2p9demo"),
            ]
        def close(self):
            pass

    class _Conn:
        def cursor(self):
            return _Cur()
        def close(self):
            pass

    monkeypatch.setattr(core.db, "get_db_connection", lambda: _Conn())

    res = search_messages_where.invoke({"condition": "messages containing a link"})

    assert "https://tinyurl.com/2p9demo" in res, f"链接消息未返回: {res}"
    meta = tools_module.get_last_search_meta()
    assert meta["raw_hits"] and meta["raw_hits"][0]["text"].startswith("看看这个链接")


def test_validate_sql_blocks_unsafe_statements():
    """text-to-SQL 校验器拒绝非 SELECT / 危险语句, 并附加 LIMIT。"""
    from apps.corpchat.search.tools import _validate_sql

    assert _validate_sql("DROP TABLE messages") is None
    assert _validate_sql("INSERT INTO messages VALUES (1)") is None
    assert _validate_sql("DELETE FROM messages") is None
    assert _validate_sql("SELECT content FROM messages WHERE label='fraud'") is not None
    ok = _validate_sql("SELECT content FROM messages WHERE content ILIKE '%http%'")
    assert ok is not None and ok.lower().endswith("limit 20")


def test_search_messages_where_falls_back_to_index_scan_when_db_down(monkeypatch):
    """DB 不可用时回退到 txtai 索引扫描 (确定性正则匹配 URL)。"""
    import core.db

    def _down():
        raise ConnectionError("db down")

    monkeypatch.setattr(core.db, "get_db_connection", _down)

    class _UrlEmbeddings:
        database = _FakeDB
        def count(self):
            return 3
        def search(self, q, limit=10, weights=None):
            return [
                {"id": "m1", "text": "看看 https://tinyurl.com/2p9demo", "score": 0.0},
                {"id": "m2", "text": "好的 收到 谢谢", "score": 0.0},
                {"id": "m3", "text": "www.example.com 请查收", "score": 0.0},
            ]

    monkeypatch.setattr(tools_module, "_load_messages_index", lambda: _UrlEmbeddings())

    res = search_messages_where.invoke({"condition": "messages containing a link"})

    assert "https://tinyurl.com/2p9demo" in res, f"索引扫描未返回链接消息: {res}"
    assert "www.example.com" in res
    meta = tools_module.get_last_search_meta()
    assert meta["hit_count"] == 2


def test_cross_table_agent_uses_structured_tool_for_link_query(monkeypatch):
    """Agent 对含链接查询调用 search_messages_where 并暴露 raw_hits。"""
    import types
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    calls = []

    def _fake_where_invoke(payload):
        calls.append(payload.get("condition"))
        return ("【结构化匹配】\n1. [Match] user_高健銘_i-chunli (userid: user_高健銘_i-chunli) "
                "[Label: old_friend_reconnect] [2026-01-01T10:00:00]\n   看看 https://tinyurl.com/2p9demo")

    monkeypatch.setattr(tools_module, "search_messages_where", types.SimpleNamespace(invoke=_fake_where_invoke))
    monkeypatch.setattr(tools_module, "search_messages", types.SimpleNamespace(invoke=lambda p: ""))
    monkeypatch.setattr(tools_module, "search_contacts", types.SimpleNamespace(invoke=lambda p: ""))
    monkeypatch.setattr(tools_module, "_last_msg_meta", {
        "query": "", "expanded_queries": [], "hit_count": 1, "previews": [],
        "raw_hits": [{"id": "m1", "text": "https://tinyurl.com/2p9demo", "score": 0.0,
                      "metadata": {"customer_name": "高健銘", "label": "old_friend_reconnect"}}],
    })

    agent = CrossTableAgent(expand=False, use_rerank=False)
    agent._llm_summarize = lambda q, msg, contact: "含链接的消息: 高健銘"
    result = agent.process("search for messages containing a link please")

    assert calls, "structured tool 未被调用"
    assert result.get("raw_hits"), "结构化结果应暴露 raw_hits 供图谱使用"
    assert result.get("success") is True

