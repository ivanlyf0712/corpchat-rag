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
import types

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
        if "text, tags" in self._sql:
            # 真实 txtai sections 表: SELECT text, tags → 两列
            return ("合同已签，请确认后安排付款。",
                    json.dumps({
                        "label": "sample_request",
                        "customer_name": "陳志明",
                        "external_userid": "user_1",
                        "full_name": "陳志明",
                        "userid": "user_1",
                        "email": "weiyao@example.org",
                        "company": "聯成電腦",
                        "phone": "0912345678",
                        "job_title": "採購專員",
                    }, ensure_ascii=False))
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


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset the module-level retrieval config between tests (决策 12)."""
    tools_module.configure_search(expand=False, use_rerank=False, graph_parallel=False)
    yield
    tools_module.configure_search(expand=False, use_rerank=False, graph_parallel=False)


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
        """expand=True (via configure_search) → expanded queries in meta; expander called."""
        monkeypatch.setattr("apps.corpchat.search.query_expander.QueryExpander",
                            lambda *a, **k: fake_expander)
        tools_module.configure_search(expand=True, use_rerank=False)
        search_messages.invoke({"query": "合同已签"})
        meta = tools_module.get_last_search_meta()
        assert fake_expander.calls == ["合同已签"], f"Expander not called: {fake_expander.calls}"
        assert meta["expanded_queries"] == ["合约确认", "合同 签署"]
        assert meta["hit_count"] == 3

    def test_rerank_enabled_calls_reranker(self, monkeypatch, fake_reranker):
        """use_rerank=True (via configure_search) → Reranker invoked; results returned."""
        monkeypatch.setattr("apps.corpchat.search.reranker.Reranker",
                            lambda *a, **k: fake_reranker)
        tools_module.configure_search(expand=False, use_rerank=True)
        result = search_messages.invoke({"query": "合同已签"})
        assert fake_reranker.calls == 1, "Reranker not called"
        assert "合同已签" in result

    def test_expansion_failure_falls_back_to_plain_query(self, monkeypatch):
        """Expander raising → graceful fallback to single-query search."""
        class _BoomExpander:
            def expand(self, query, use_cache=True):
                raise RuntimeError("LLM down")

        monkeypatch.setattr("apps.corpchat.search.query_expander.QueryExpander",
                            lambda *a, **k: _BoomExpander())
        tools_module.configure_search(expand=True, use_rerank=False)
        result = search_messages.invoke({"query": "合同已签"})
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


# ── CrossTableAgent forwarding (via configure_search, 决策 12) ────
class TestAgentForwarding:
    def test_agent_injects_expand_and_rerank_config(self, monkeypatch):
        """CrossTableAgent 构建时把 expand/use_rerank 注入模块级配置 (决策 12)。"""
        from apps.corpchat.search.cross_table_agent import CrossTableAgent

        injected = {}
        monkeypatch.setattr(tools_module, "configure_search",
                            lambda **kw: injected.update(kw))

        agent = CrossTableAgent(expand=True, use_rerank=True)
        agent._init_agent()

        assert injected.get("expand") is True, f"expand not injected: {injected}"
        assert injected.get("use_rerank") is True, f"use_rerank not injected: {injected}"

    def test_agent_disabled_toggles_injected_false(self, monkeypatch):
        """expand/use_rerank False → configure_search 收到 False。"""
        from apps.corpchat.search.cross_table_agent import CrossTableAgent

        injected = {}
        monkeypatch.setattr(tools_module, "configure_search",
                            lambda **kw: injected.update(kw))

        agent = CrossTableAgent(expand=False, use_rerank=False)
        agent._init_agent()

        assert injected.get("expand") is False
        assert injected.get("use_rerank") is False


# ═══════════════ Ticket 02: graph_parallel threading (tool) ═══════════════
def test_search_messages_accepts_and_forwards_graph_parallel(monkeypatch):
    """search_messages 读取模块级 graph_parallel 并透传给 RRF 融合路径 (expand 分支)。"""
    calls = []

    def _recording_fuse(embeddings, queries_with_weights, limit=10, **kw):
        calls.append(kw)
        return []

    monkeypatch.setattr(tools_module, "_weighted_rrf_fuse", _recording_fuse)
    tools_module.configure_search(expand=True, graph_parallel=True)
    search_messages.invoke({"query": "跟誰聊過物流"})

    assert calls, "expand 分支应调用 _weighted_rrf_fuse"
    assert calls[0].get("graph_parallel") is True, f"graph_parallel 未透传: {calls[0]}"


def test_cross_table_agent_injects_graph_parallel(monkeypatch):
    """CrossTableAgent(graph_parallel=True) → configure_search 收到 graph_parallel=True。"""
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    injected = {}
    monkeypatch.setattr(tools_module, "configure_search",
                        lambda **kw: injected.update(kw))

    agent = CrossTableAgent(graph_parallel=True)
    agent._init_agent()

    assert injected.get("graph_parallel") is True, f"graph_parallel 未注入: {injected}"


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
    """sources 不含 contacts → 绑定给 agent 的工具集不含 search_contacts。"""
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    agent = CrossTableAgent(sources=["messages"])
    # 让 create_react_agent 不真正构建 (monkeypatch), 只验证工具筛选
    captured = {}
    monkeypatch.setattr(
        "apps.corpchat.search.cross_table_agent.create_react_agent",
        lambda model, tools, prompt=None: captured.update({"tools": tools}) or object(),
    )
    agent._init_agent()

    names = [t.name for t in captured.get("tools", [])]
    assert "search_messages" in names, f"messages 源应有 search_messages: {names}"
    assert "search_conversation_partners" in names, f"messages 源应有关系工具: {names}"
    assert "search_contacts" not in names, f"sources 排除 contacts 仍绑定: {names}"


def test_contact_name_query_uses_search_contacts_tool():
    """联系人姓名类查询由 search_contacts 工具服务 (主路径由 DeepSeek 路由)。"""
    from apps.corpchat.search.tools import CROSS_TABLE_TOOLS

    names = [t.name for t in CROSS_TABLE_TOOLS]
    assert "search_contacts" in names, f"工具集应包含 search_contacts: {names}"


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
    tools_module.configure_search(expand=False, graph_parallel=True)

    result = search_messages.invoke({"query": "跟誰聊過物流"})
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
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    monkeypatch.setattr(tools_module, "_last_msg_meta", {
        "query": "合同已签", "expanded_queries": [], "hit_count": 1, "previews": [],
        "raw_hits": [{"id": "m1", "text": "合同已签", "score": 0.61,
                      "metadata": {"customer_name": "高健銘", "company": "DCH",
                                   "label": "old_friend_reconnect", "open_kfid": "k1"}}],
    })
    monkeypatch.setattr(tools_module, "_last_contact_meta",
                        {"query": "", "hit_count": 0, "previews": []})

    class _FakeAgent:
        def invoke(self, state):
            return {"messages": [
                HumanMessage(content="帮我查一下合同已签的消息"),
                AIMessage(content="", tool_calls=[{
                    "name": "search_messages", "args": {"query": "合同已签"}, "id": "c1",
                }]),
                ToolMessage(content="【消息搜索结果】", tool_call_id="c1"),
                AIMessage(content="找到了 1 条相关消息"),
            ]}

    agent = CrossTableAgent(expand=False, use_rerank=False)
    agent._agent = _FakeAgent()
    result = agent.process("帮我查一下合同已签的消息")

    hits = result.get("raw_hits", [])
    assert len(hits) == 1, f"agent 应暴露 raw_hits: {hits}"
    assert hits[0]["metadata"]["customer_name"] == "高健銘"
    assert result.get("success") is True


# ═══════════════ SQL 结构化检索 (search_messages_where, 降级路径) ═══════════════
def test_structured_link_query_schema_available():
    """search_messages_where 仍可导入 (降级路径保留), 但不在主路径工具集。"""
    from apps.corpchat.search.tools import search_messages_where, CROSS_TABLE_TOOLS

    names = [t.name for t in CROSS_TABLE_TOOLS]
    assert "search_messages_where" not in names, "结构化工具不应暴露给主路径 LLM"
    assert search_messages_where.name == "search_messages_where"


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


def test_cross_table_agent_fallback_keeps_structured_tool_available(monkeypatch):
    """search_messages_where 保留在降级路径 (可导入, 不被主路径绑定)。"""
    from apps.corpchat.search.tools import search_messages_where
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    # 降级路径 (_fallback_process) 仍使用 search_messages_where 的能力:
    # 它通过 configure_search 注入 + search_messages 走语义检索; 结构化工具
    # 依然可独立调用 (含链接查询)。
    res = search_messages_where.invoke({"condition": "messages containing a link"})
    assert isinstance(res, str)
    assert res.strip(), "结构化工具降级路径仍应可用"


# ── Ticket 02: Hindsight retain 实体锚点 (extract_entity_tags) ─────
class TestExtractEntityTags:
    """从 raw_hits metadata 提取实体名, 供 Hindsight retain tags 使用。"""

    def _hits(self, *metas):
        return [{"id": f"m{i}", "text": "x", "score": 0.5, "metadata": m}
                for i, m in enumerate(metas)]

    def test_customer_and_company_extracted(self):
        from apps.corpchat.search.tools import extract_entity_tags
        hits = self._hits({"customer_name": "陳志明", "company": "聯成電腦"})
        assert extract_entity_tags(hits) == ["陳志明", "聯成電腦"]

    def test_dedup_preserves_order(self):
        from apps.corpchat.search.tools import extract_entity_tags
        hits = self._hits(
            {"customer_name": "陳志明", "company": "聯成電腦"},
            {"customer_name": "陳志明", "company": "聯成電腦"},
        )
        assert extract_entity_tags(hits) == ["陳志明", "聯成電腦"]

    def test_customer_name_fallback_to_external_userid(self):
        from apps.corpchat.search.tools import extract_entity_tags
        hits = self._hits({"external_userid": "user_1", "company": ""})
        assert extract_entity_tags(hits) == ["user_1"]

    def test_cap_at_five(self):
        from apps.corpchat.search.tools import extract_entity_tags
        metas = [{"customer_name": f"客戶{i}", "company": f"公司{i}"} for i in range(5)]
        tags = extract_entity_tags(self._hits(*metas))
        assert len(tags) == 5
        assert len(set(tags)) == len(tags), "不应有重复"

    def test_empty_or_malformed_inputs(self):
        from apps.corpchat.search.tools import extract_entity_tags
        assert extract_entity_tags([]) == []
        assert extract_entity_tags(None) == []
        assert extract_entity_tags([{"id": "m", "text": "x", "score": 0.1}]) == []
        assert extract_entity_tags([None, "junk", {"metadata": "not-dict"}]) == []

    def test_blank_values_skipped(self):
        from apps.corpchat.search.tools import extract_entity_tags
        hits = self._hits({"customer_name": "  ", "company": "  聯成電腦  "})
        assert extract_entity_tags(hits) == ["聯成電腦"]


# ── HF 缓存自动离线检测 (apps.corpchat.hf_offline) ─────────────────
class TestHfCacheHasModel:
    """纯文件系统检测: 缓存里有非空快照 → 自动离线; 缺失/空 → 在线。"""

    @staticmethod
    def _seed_cache(tmp_path, model_id, with_file=True):
        """构造 hub/models--<id>/snapshots/<sha>/ 结构 (with_file=False 时空快照)。"""
        snap = tmp_path / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        if with_file:
            (snap / "config.json").write_text("{}")
        return tmp_path

    def test_detects_cached_model(self, tmp_path, monkeypatch):
        from apps.corpchat.hf_offline import _hf_cache_has_model
        self._seed_cache(tmp_path, "BAAI/bge-m3")
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
        assert _hf_cache_has_model("BAAI/bge-m3") is True

    def test_missing_model_returns_false(self, tmp_path, monkeypatch):
        from apps.corpchat.hf_offline import _hf_cache_has_model
        self._seed_cache(tmp_path, "BAAI/bge-m3")
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
        assert _hf_cache_has_model("BAAI/bge-m3") is True
        assert _hf_cache_has_model("other/model") is False

    def test_empty_snapshot_not_cached(self, tmp_path, monkeypatch):
        from apps.corpchat.hf_offline import _hf_cache_has_model
        self._seed_cache(tmp_path, "BAAI/bge-m3", with_file=False)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
        assert _hf_cache_has_model("BAAI/bge-m3") is False

    def test_no_cache_dir_returns_false(self, tmp_path, monkeypatch):
        from apps.corpchat.hf_offline import _hf_cache_has_model
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "nope"))
        assert _hf_cache_has_model("BAAI/bge-m3") is False

    def test_apply_auto_offline_sets_env_when_cached(self, tmp_path, monkeypatch):
        from apps.corpchat.hf_offline import apply_auto_offline
        self._seed_cache(tmp_path, "BAAI/bge-m3")
        self._seed_cache(tmp_path, "BAAI/bge-reranker-base")
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        apply_auto_offline()
        assert os.environ.get("HF_HUB_OFFLINE") == "1"

    def test_apply_auto_offline_respects_explicit_env(self, tmp_path, monkeypatch):
        from apps.corpchat.hf_offline import apply_auto_offline
        self._seed_cache(tmp_path, "BAAI/bge-m3")
        self._seed_cache(tmp_path, "BAAI/bge-reranker-base")
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")  # 显式在线 → 不覆盖
        apply_auto_offline()
        assert os.environ.get("HF_HUB_OFFLINE") == "0"

    def test_apply_auto_offline_no_env_when_missing(self, tmp_path, monkeypatch):
        from apps.corpchat.hf_offline import apply_auto_offline
        self._seed_cache(tmp_path, "BAAI/bge-m3")  # 只缓存一个 → 不设离线
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        apply_auto_offline()
        assert os.environ.get("HF_HUB_OFFLINE") is None

