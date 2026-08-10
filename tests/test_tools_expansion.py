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
from apps.corpchat.search.tools import search_messages, search_contacts


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

