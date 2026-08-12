"""
Unit tests for CrossTableAgent helper functions (no index / LLM required).

Covers the two robustness fixes:
  - _detect_language: mixed-language queries (English + Chinese name)
    should be answered in English when Latin letters dominate.
  - _format_fallback_answer: message preview must be the actual message
    content line, not the 【消息搜索结果】 section header.
"""
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from apps.corpchat.search.cross_table_agent import CrossTableAgent


# ── _detect_language ─────────────────────────────────────────────
class TestDetectLanguage:
    def test_english_only(self):
        assert CrossTableAgent._detect_language("what is the logistics quote") == "en"

    def test_simplified_chinese(self):
        assert CrossTableAgent._detect_language("李雅婷的邮箱是什么？") == "zh-CN"

    def test_traditional_chinese(self):
        assert CrossTableAgent._detect_language("李雅婷的郵箱是什麼？") == "zh-TW"

    def test_mixed_english_dominant(self):
        """'who did 何建明 spoke to?' is mostly English → answer in English."""
        assert CrossTableAgent._detect_language("who did 何建明 spoke to?") == "en"

    def test_mixed_english_name_only(self):
        """'email for 李雅婷' is English-dominant → answer in English."""
        assert CrossTableAgent._detect_language("email for 李雅婷") == "en"

    def test_chinese_english_balanced_english_wins(self):
        """Equal-weight CJK names in an English sentence → English."""
        assert CrossTableAgent._detect_language("who is 陈志明 and 李雅婷?") == "en"


# ── _format_fallback_answer message preview ──────────────────────
class TestFormatFallbackAnswer:
    def _agent(self):
        return CrossTableAgent()

    def test_msg_preview_skips_header(self):
        """The message preview should be real content, not the section header."""
        msg_result = (
            "【消息搜索结果】\n"
            "\n"
            "1. [Score: 0.6122] 陳志明 (userid: user_陳志明_johnsonj) [Label: sample_request]\n"
            "   合同已签，请确认后安排付款。\n"
            "\n"
            "2. [Score: 0.4122] 李雅婷 (userid: user_李雅婷_tiffanyli) [Label: sample_request]\n"
            "   报价已发送，请查收。\n"
        )
        contact_result = (
            "【联系人搜索结果】\n"
            "\n"
            "1. [Score: 0.9] 陳志明 (userid: user_陳志明_johnsonj)\n"
            "   Email: weiyao@example.org\n"
            "   Company: 聯成電腦\n"
            "   Phone: 0912345678\n"
            "   Job Title: 採購專員\n"
        )
        answer = self._agent()._format_fallback_answer(
            "发'合同已签'消息的人，他的邮箱是什么？", msg_result, contact_result
        )
        assert "合同已签" in answer, f"Preview should contain message content, got: {answer}"
        assert "【消息搜索结果】" not in answer, (
            "Preview must not be the section header"
        )
        assert "weiyao@example.org" in answer, "Email should be present"
        assert "陳志明" in answer, "Sender name should be present"

    def test_msg_preview_empty_results(self):
        """No results → graceful 'not found' message, no crash."""
        answer = self._agent()._format_fallback_answer(
            "不存在的关键词", "", ""
        )
        assert answer.strip(), "Expected a non-empty response"
        assert "抱歉" in answer or "Sorry" in answer

    def test_msg_preview_content_regex_variant(self):
        """'Content:' prefixed results still parse."""
        msg_result = (
            "【消息搜索结果】\n"
            "1. [Score: 0.5] A (userid: user_a_x)\n"
            "   Content: 明天开会讨论预算\n"
        )
        contact_result = (
            "【联系人搜索结果】\n"
            "1. [Score: 0.9] A (userid: user_a_x)\n"
            "   Email: a@example.org\n"
        )
        answer = self._agent()._format_fallback_answer(
            "A 的消息是什么？", msg_result, contact_result
        )
        assert "明天开会" in answer, f"Expected content line, got: {answer}"


# ── Greeting intent: LLM classify + LLM-generated reply ────────────────────
class TestGreetingHandling:
    """'how are you' and friends must route as greetings (no tools), and
    greeting replies must be LLM-generated when the LLM is available."""

    def _agent(self, **kw):
        return CrossTableAgent(**kw)

    def test_how_are_you_no_tools_llm_down(self, monkeypatch):
        """'how are you?' must NOT trigger a search (rules path, LLM down)."""
        agent = self._agent()
        monkeypatch.setattr(agent, "_check_llm", lambda: False)
        result = agent.process("how are you?")
        assert result.get("fallback") is False
        assert len(result.get("tool_calls", [])) == 0, "greeting must not call tools"
        assert result.get("output")

    def test_greeting_variants_no_tools(self, monkeypatch):
        agent = self._agent()
        monkeypatch.setattr(agent, "_check_llm", lambda: False)
        for q in ("Hi", "hello", "hey", "how are you", "how's it going",
                  "good morning", "你好", "哈囉", "早安", "最近怎么样"):
            result = agent.process(q)
            assert len(result.get("tool_calls", [])) == 0, f"{q!r} should be a greeting"
            assert result.get("output"), f"{q!r} should produce an answer"

    def test_decide_tool_calls_how_are_you_no_tools(self):
        """问候词走规则快路, 不触发工具路由 (LLM classify 已移除, 直接进 agent)。"""
        from apps.corpchat.search.cross_table_agent import _is_greeting_query
        assert _is_greeting_query("how are you?")

    def test_decide_tool_calls_hi_inside_word_not_greeting(self):
        """'hi' inside 'this' must not be treated as a greeting (whole-word matching)."""
        from apps.corpchat.search.cross_table_agent import _is_greeting_query
        assert not _is_greeting_query("this is it")

    def test_quick_respond_llm_classifies_greeting(self, monkeypatch):
        """非典型问候直接进入 agent 路径 (规则未命中 → None), 不再走 LLM 分类。"""
        agent = self._agent(api_base="http://fake", api_key="k")
        monkeypatch.setattr(agent, "_check_llm", lambda: True)
        # 规则未命中的非典型问候 → 直接返回 None, 交给 LangGraph agent
        assert agent._quick_respond("how is everything going my good friend today") is None

    def test_quick_respond_llm_generates_greeting(self, monkeypatch):
        """Greeting answer comes from the LLM, not the preset string."""
        agent = self._agent(api_base="http://fake", api_key="k")
        monkeypatch.setattr(agent, "_check_llm", lambda: True)
        monkeypatch.setattr(
            "apps.corpchat.search.litellm_client.LiteLLMClient.chat",
            lambda self, messages, **kw: "Hello! How can I help you search today?",
        )
        out = agent._quick_respond("how are you?")
        assert out == "Hello! How can I help you search today?"
        assert out != agent._PRESET_GREETING_EN

    def test_quick_respond_preset_when_llm_generation_fails(self, monkeypatch):
        """Preset fallback when the LLM call fails mid-greeting."""
        agent = self._agent(api_base="http://fake", api_key="k")
        monkeypatch.setattr(agent, "_check_llm", lambda: True)

        def boom(self, messages, **kw):
            raise RuntimeError("llm down")

        monkeypatch.setattr("apps.corpchat.search.litellm_client.LiteLLMClient.chat", boom)
        out = agent._quick_respond("hello")
        assert "CorpChat" in out  # preset fallback

    def test_quick_respond_llm_down_preset(self, monkeypatch):
        agent = self._agent()
        monkeypatch.setattr(agent, "_check_llm", lambda: False)
        assert agent._quick_respond("hello") == agent._PRESET_GREETING_EN

    def test_search_hint_skips_llm_classify(self, monkeypatch):
        """搜索查询直接进入 agent 路径, 不再有独立的 LLM classify 往返。"""
        agent = self._agent(api_base="http://fake", api_key="k")
        monkeypatch.setattr(agent, "_check_llm", lambda: True)
        assert agent._quick_respond("李雅婷的邮箱是什么？") is None


def test_extract_search_query_strips_english_noise():
    """英文口语噪声 (try again / who is / please find) 应从查询中剥离。"""
    agent = CrossTableAgent()
    assert agent._extract_search_query("try again. Who is 李雅婷?") == "李雅婷"
    assert agent._extract_search_query("please find 陳志明 email") == "陳志明 email"
    assert agent._extract_search_query("search for messages containing a link please") == "messages containing a link"


class _FakeAgent:
    """Minimal fake LangGraph agent for main-path process() tests (决策 10c)."""

    def __init__(self, messages):
        self._messages = messages

    def invoke(self, state):
        return {"messages": self._messages}


def test_contact_only_query_uses_fake_agent(monkeypatch):
    """主路径: agent 返回联系人搜索 + 最终答案, process() 正确透传。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    import types
    from apps.corpchat.search import tools as tools_module

    monkeypatch.setattr(tools_module, "_last_contact_meta",
                        {"query": "", "hit_count": 1, "previews": []})
    monkeypatch.setattr(tools_module, "_last_msg_meta",
                        {"query": "", "expanded_queries": [], "hit_count": 0,
                         "previews": [], "raw_hits": []})

    agent = CrossTableAgent(expand=False, use_rerank=False)
    msgs = [
        HumanMessage(content="Who is 李雅婷?"),
        AIMessage(content="", tool_calls=[{
            "name": "search_contacts", "args": {"query": "李雅婷"}, "id": "c1",
        }]),
        ToolMessage(content=("【联系人搜索结果】\n1. [Score: 0.9] 李雅婷 "
                             "(userid: user_李雅婷_dliang)\n   Email: hsin-ihu@example.org"),
                    tool_call_id="c1"),
        AIMessage(content="李雅婷 (userid: user_李雅婷_dliang) 邮箱: hsin-ihu@example.org"),
    ]
    agent._agent = _FakeAgent(msgs)

    result = agent.process("Who is 李雅婷?")

    assert result.get("success") is True
    assert result.get("fallback") is False
    assert "hsin-ihu@example.org" in result["output"]
    assert [tc["tool"] for tc in result.get("tool_calls", [])] == ["search_contacts"]


# ── Hindsight 按需 recall gate (决策 16: 记忆触达词) ───────────────
class TestNeedsHindsightRecall:
    """记忆触达词 gate: 命中显式跨会话引用词才需要 Hindsight recall。"""

    def test_cn_trigger_words(self):
        from apps.corpchat.search.cross_table_agent import _needs_hindsight_recall
        for q in ("记得上次说的报价吗", "上次聊的那个客户是谁", "之前说的合同",
                  "以前谈的物流价格", "当时说的方案", "上回那件事怎么样了",
                  "还记得我们聊过什么吗"):
            assert _needs_hindsight_recall(q), f"{q!r} should trigger recall"

    def test_en_trigger_words(self):
        from apps.corpchat.search.cross_table_agent import _needs_hindsight_recall
        for q in ("what did we discuss last time", "do you remember the quote",
                  "as I said before", "previously discussed pricing",
                  "like we discussed earlier"):
            assert _needs_hindsight_recall(q), f"{q!r} should trigger recall"

    def test_plain_query_no_trigger(self):
        from apps.corpchat.search.cross_table_agent import _needs_hindsight_recall
        for q in ("李雅婷的邮箱是什么？", "客户喜欢什么沟通方式",
                  "陈志明昨天发了什么消息", "who is 李雅婷?"):
            assert not _needs_hindsight_recall(q), f"{q!r} must NOT trigger recall"

    def test_in_session_pronoun_no_trigger(self):
        """会话内指代 ('她的邮箱') 走历史注入, 不触发 Hindsight recall。"""
        from apps.corpchat.search.cross_table_agent import _needs_hindsight_recall
        for q in ("那她的邮箱是什么？", "这个客户的电话呢", "那他呢"):
            assert not _needs_hindsight_recall(q), f"{q!r} must NOT trigger recall"

    def test_english_word_boundary(self):
        """英文单词整词匹配: 'before' 不命中 'beforehand'。"""
        from apps.corpchat.search.cross_table_agent import _needs_hindsight_recall
        assert not _needs_hindsight_recall("search beforehand please")


def test_hindsight_recall_gated_in_process(monkeypatch):
    """决策 16: process() 只对命中触达词的查询调用 Hindsight recall。"""
    from langchain_core.messages import AIMessage, HumanMessage
    from apps.corpchat.search import hindsight_client, tools as tools_module

    calls: list = []

    def fake_recall(query, bank=None, max_results=5):
        calls.append(query)
        return []

    monkeypatch.setattr(hindsight_client, "recall", fake_recall)
    monkeypatch.setattr(tools_module, "_last_contact_meta",
                        {"query": "", "hit_count": 0, "previews": []})
    monkeypatch.setattr(tools_module, "_last_msg_meta",
                        {"query": "", "expanded_queries": [], "hit_count": 0,
                         "previews": [], "raw_hits": []})

    agent = CrossTableAgent(hindsight_bank="test-bank")
    agent._agent = _FakeAgent([HumanMessage(content="x"),
                               AIMessage(content="done")])

    # 命中触达词 → recall 被调用
    agent.process("记得上次说的报价吗")
    assert calls, "trigger-word query must call Hindsight recall"
    assert calls[0] == "记得上次说的报价吗"

    # 普通查询 → recall 被跳过
    calls.clear()
    agent.process("李雅婷的邮箱是什么？")
    assert not calls, "plain query must skip Hindsight recall"

    # 会话内指代 → 不触发 (走会话历史注入)
    calls.clear()
    agent.process("那她的邮箱呢？")
    assert not calls, "in-session pronoun must skip Hindsight recall"


def test_hindsight_recall_injects_memory_when_gated(monkeypatch):
    """命中 gate 且 bank 有相关记忆 → 记忆注入到 agent 输入。"""
    from langchain_core.messages import AIMessage, HumanMessage
    from apps.corpchat.search import hindsight_client, tools as tools_module

    captured: dict = {}

    def fake_recall(query, bank=None, max_results=5):
        return [{"content": "客户偏好电话沟通"}]

    monkeypatch.setattr(hindsight_client, "recall", fake_recall)
    monkeypatch.setattr(tools_module, "_last_contact_meta",
                        {"query": "", "hit_count": 0, "previews": []})
    monkeypatch.setattr(tools_module, "_last_msg_meta",
                        {"query": "", "expanded_queries": [], "hit_count": 0,
                         "previews": [], "raw_hits": []})

    class _RecordingFake(_FakeAgent):
        def invoke(self, state):
            captured["input"] = state["messages"][0].content
            return super().invoke(state)

    agent = CrossTableAgent(hindsight_bank="test-bank")
    agent._agent = _RecordingFake([HumanMessage(content="x"),
                                   AIMessage(content="done")])
    agent.process("记得上次说的报价吗")
    assert "客户偏好电话沟通" in captured["input"], "gated recall must inject memory"
    assert "Hindsight bank" in captured["input"]


# ── 工具实时通知 (on_tool: UI 显示精确工具名+参数) ─────────────────
def test_notifying_tool_wrapper_fires_on_tool_callback():
    """包装后的工具: 真实调用时触发 on_tool(name, args), 原结果与 schema 不变。"""
    from langchain_core.tools import tool
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    def _fake_search(query: str = "x") -> str:
        """Fake search tool."""
        return f"result:{query}"

    agent = CrossTableAgent()
    wrapped = agent._notifying_tool(tool(_fake_search))
    assert wrapped.name == "_fake_search"
    assert sorted(wrapped.args_schema.model_fields.keys()) == ["query"]

    calls = []
    agent._on_tool_callback = lambda name, args: calls.append((name, args))
    assert wrapped.invoke({"query": "y"}) == "result:y"
    assert calls == [("_fake_search", {"query": "y"})], f"回调应收到工具名与参数: {calls}"

    # 未设置回调 (默认 None) → 静默, 不影响结果
    agent._on_tool_callback = None
    calls.clear()
    assert wrapped.invoke({"query": "z"}) == "result:z"
    assert calls == []


def test_init_agent_wraps_tools_with_notifying(monkeypatch):
    """_init_agent 用 _notifying_tool 重建包装每个主路径工具 (传给 create_react_agent)。"""
    from apps.corpchat.search import cross_table_agent as cta
    from apps.corpchat.search import tools as tools_module

    captured = {}

    def fake_create_react_agent(model, tools=None, prompt=None, **kw):
        captured["tools"] = list(tools or [])
        return object()

    monkeypatch.setattr(cta, "create_react_agent", fake_create_react_agent)
    agent = cta.CrossTableAgent(api_base="http://fake", api_key="k")
    agent._init_agent()

    originals = list(tools_module.CROSS_TABLE_TOOLS)
    wrapped = captured["tools"]
    assert len(wrapped) == len(originals), f"工具数量不变, 应包装全部 {len(originals)} 个"
    for w, o in zip(wrapped, originals):
        assert w.name == o.name
        assert w.func is not o.func, f"{o.name} 应被 _notifying_tool 重建包装"
        assert sorted(w.args_schema.model_fields.keys()) == sorted(o.args_schema.model_fields.keys())


# ── Per-call tool-result attribution (candidate 1: no global meta) ─
class TestPerCallMetaAttribution:
    """真实工具执行时, process() 对每个工具调用归属各自的结果 meta。

    Regression guard for the thread-safety / per-tool Process-window fix:
    two search_messages calls in one turn must each carry their own meta,
    not the last call's.
    """

    def _wrapped(self, agent):
        """Build a _notifying_tool-wrapped stub that records per-call meta.

        与真实工具一致: 用 @tool 装饰器生成 args_schema, 再经 _notifying_tool 包装。
        """
        from apps.corpchat.search import tools as tools_module
        from langchain_core.tools import tool

        @tool("search_messages")
        def _stub_tool(query: str = "") -> str:
            """Stub message search."""
            tools_module._set_msg_meta({"query": query, "expanded_queries": [],
                                        "hit_count": 1, "previews": [],
                                        "raw_hits": [{"id": f"m_{query}", "text": query,
                                                      "score": 0.5, "metadata": {}}]})
            return f"res:{query}"

        return agent._notifying_tool(_stub_tool)

    def test_two_calls_get_distinct_meta(self):
        from apps.corpchat.search.cross_table_agent import CrossTableAgent

        agent = CrossTableAgent()
        wrapped = self._wrapped(agent)

        wrapped.invoke({"query": "合同已签"})
        wrapped.invoke({"query": "物流报价"})

        assert len(agent._tool_meta_log) == 2
        queries = [e["meta"]["query"] for e in agent._tool_meta_log]
        assert queries == ["合同已签", "物流报价"], f"meta 应按调用顺序逐次归属: {queries}"

    def test_process_attributes_meta_per_executed_call(self, monkeypatch):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        from apps.corpchat.search import tools as tools_module
        from apps.corpchat.search.cross_table_agent import CrossTableAgent

        monkeypatch.setattr(tools_module, "_last_msg_meta",
                            {"query": "fallback", "expanded_queries": [],
                             "hit_count": 0, "previews": [], "raw_hits": []})
        monkeypatch.setattr(tools_module, "_last_contact_meta",
                            {"query": "", "hit_count": 0, "previews": []})

        agent = CrossTableAgent()
        wrapped = self._wrapped(agent)

        class _FakeAgent:
            def invoke(self, state):
                # 真实工具执行路径: wrapper 在工具返回后逐调用快照 meta
                wrapped.invoke({"query": "合同已签"})
                wrapped.invoke({"query": "物流报价"})
                return {"messages": [
                    HumanMessage(content="查两个词"),
                    AIMessage(content="", tool_calls=[
                        {"name": "search_messages", "args": {"query": "合同已签"}, "id": "c1"},
                        {"name": "search_messages", "args": {"query": "物流报价"}, "id": "c2"},
                    ]),
                    ToolMessage(content="res:合同已签", tool_call_id="c1"),
                    ToolMessage(content="res:物流报价", tool_call_id="c2"),
                    AIMessage(content="完成"),
                ]}

        agent._agent = _FakeAgent()

        result = agent.process("查两个词")

        metas = [tc["meta"] for tc in result["tool_calls"]]
        assert [m["query"] for m in metas] == ["合同已签", "物流报价"], \
            f"每个 search_messages 调用应归属各自的 meta: {metas}"
        assert [h["id"] for h in result["raw_hits"]] == ["m_物流报价"], \
            "raw_hits 应取最后一次 search_messages 调用的结果"

    def test_snapshot_meta_locked_and_copy(self):
        from apps.corpchat.search import tools as tools_module

        tools_module._set_msg_meta({"query": "q", "hit_count": 3, "previews": [],
                                    "expanded_queries": [], "raw_hits": []})
        snap = tools_module.snapshot_meta("search_messages")
        snap["query"] = "mutated"
        assert tools_module.get_last_search_meta()["query"] == "q", \
            "snapshot_meta 应返回副本, 外部修改不影响全局通道"



# ── Ticket 04: structured fallback rendering (regex-scraping removed) ─
class TestStructuredFallback:
    def _agent(self):
        return CrossTableAgent()

    def test_structured_fallback_renders_from_hits(self):
        """结构化 hits 渲染: 邮箱/公司/预览直接来自 metadata, 无 regex。"""
        agent = self._agent()
        msg_hits = [{"id": "m1",
                     "text": "陳志明 (sample_request)\n---\n合同已签，请确认后安排付款。",
                     "metadata": {"customer_name": "陳志明", "label": "sample_request"}}]
        contact_hits = [{"id": "c1", "text": "陳志明",
                         "metadata": {"full_name": "陳志明", "userid": "user_陳志明_johnsonj",
                                      "email": "weiyao@example.org", "company": "聯成電腦",
                                      "phone": "0912345678"}}]
        answer = agent._structured_fallback_answer(
            "发'合同已签'消息的人，他的邮箱是什么？", "", "", msg_hits, contact_hits)
        assert "合同已签" in answer
        assert "weiyao@example.org" in answer
        assert "陳志明" in answer
        assert "聯成電腦" in answer

    def test_structured_fallback_empty_hits_delegates_legacy(self):
        """无结构化 hits → 回退 legacy 格式化字符串解析 (向后兼容)。"""
        agent = self._agent()
        answer = agent._structured_fallback_answer("不存在的关键词", "", "", [], [])
        assert "抱歉" in answer or "Sorry" in answer

    def test_extract_userid_from_hits_structured(self):
        """从结构化 hits 直接读 userid (不依赖格式化字符串里的 '(userid: ...)')。"""
        hits = [{"id": "m1", "text": "x",
                 "metadata": {"external_userid": "user_陳志明_johnsonj",
                              "servicer_userid": "user_許志豪_yongtang"}}]
        assert CrossTableAgent._extract_userid_from_hits(hits) == "user_陳志明_johnsonj"
        assert CrossTableAgent._extract_userid_from_hits([]) is None
        assert CrossTableAgent._extract_userid_from_hits([{"id": "m", "metadata": {}}]) is None


# ── Single source for intent words (candidate 4: 三份拷贝收敛为一处) ─
def test_intent_words_single_source():
    """greeting/system/clarify 词表与生成器只有一份实现, 双 agent 共享。"""
    from apps.corpchat import agent as legacy_agent
    from apps.corpchat.search import cross_table_agent as cta
    from apps.corpchat.search import intent_words

    assert cta._GREETING_KEYWORDS is intent_words.GREETING_KEYWORDS
    assert cta._SYSTEM_KEYWORDS is intent_words.SYSTEM_KEYWORDS
    assert legacy_agent.GREETING_KEYWORDS is intent_words.GREETING_KEYWORDS
    assert legacy_agent.SYSTEM_INFO_KEYWORDS is intent_words.SYSTEM_KEYWORDS
    assert legacy_agent.CLARIFY_KEYWORDS is intent_words.CLARIFY_KEYWORDS
    # 同一匹配实现: cross_table_agent 的 _is_greeting_query 即 intent_words 的
    assert cta._is_greeting_query is intent_words.is_greeting_query
