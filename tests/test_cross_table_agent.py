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
        from apps.corpchat.search.cross_table_agent import _LiteLLMWrapper
        w = _LiteLLMWrapper(api_base="", api_key="", model="test")
        assert w._decide_tool_calls("how are you?") == []

    def test_decide_tool_calls_hi_inside_word_not_greeting(self):
        """'hi' inside 'this' must not swallow a search (whole-word matching)."""
        from apps.corpchat.search.cross_table_agent import _LiteLLMWrapper
        w = _LiteLLMWrapper(api_base="", api_key="", model="test")
        names = [c["name"] for c in w._decide_tool_calls("this is it")]
        assert "search_messages" in names

    def test_quick_respond_llm_classifies_greeting(self, monkeypatch):
        """LLM drives greeting intent when available (ambiguous query only)."""
        agent = self._agent(api_base="http://fake", api_key="k")
        seen = []
        monkeypatch.setattr(agent, "_check_llm", lambda: True)
        monkeypatch.setattr(agent, "_llm_classify_intent",
                            lambda u: seen.append(u) or "greeting")
        monkeypatch.setattr(agent, "_generate_greeting", lambda u, lang: "LLM GREETING")
        # 规则未命中的非典型问候 → 走 LLM 分类
        assert agent._quick_respond("how is everything going my good friend today") == "LLM GREETING"
        assert seen == ["how is everything going my good friend today"]

    def test_quick_respond_llm_generates_greeting(self, monkeypatch):
        """Greeting answer comes from the LLM, not the preset string."""
        agent = self._agent(api_base="http://fake", api_key="k")
        monkeypatch.setattr(agent, "_check_llm", lambda: True)
        monkeypatch.setattr(agent, "_llm_classify_intent", lambda u: "greeting")
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
        monkeypatch.setattr(agent, "_llm_classify_intent", lambda u: "greeting")

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
        """Obvious search queries skip the LLM classify round-trip."""
        agent = self._agent(api_base="http://fake", api_key="k")
        classified = []
        monkeypatch.setattr(agent, "_check_llm", lambda: True)
        monkeypatch.setattr(agent, "_llm_classify_intent",
                            lambda u: classified.append(u) or "search")
        assert agent._quick_respond("李雅婷的邮箱是什么？") is None
        assert classified == [], "search-hint query must not hit LLM classify"


def test_extract_search_query_strips_english_noise():
    """英文口语噪声 (try again / who is / please find) 应从查询中剥离。"""
    agent = CrossTableAgent()
    assert agent._extract_search_query("try again. Who is 李雅婷?") == "李雅婷"
    assert agent._extract_search_query("please find 陳志明 email") == "陳志明 email"
    assert agent._extract_search_query("search for messages containing a link please") == "messages containing a link"


def test_contact_only_query_uses_deterministic_answer(monkeypatch):
    """纯联系人查询走确定性格式化, 不调用 LLM 总结 (避免小模型幻觉)。"""
    import types
    from apps.corpchat.search import tools as tools_module

    def _fake_contact_invoke(payload):
        return ("【联系人搜索结果】\n1. [Score: 0.9] 李雅婷 (userid: user_李雅婷_dliang)\n"
                "   Email: hsin-ihu@example.org\n   Company: 富邦金控\n"
                "   Phone: (925)853-4192x832\n   Job Title: 人力資源主管")

    monkeypatch.setattr(tools_module, "search_contacts", types.SimpleNamespace(invoke=_fake_contact_invoke))
    monkeypatch.setattr(tools_module, "search_messages", types.SimpleNamespace(invoke=lambda p: ""))
    monkeypatch.setattr(tools_module, "search_messages_where", types.SimpleNamespace(invoke=lambda p: ""))
    monkeypatch.setattr(tools_module, "_last_contact_meta",
                        {"query": "", "hit_count": 1, "previews": []})

    agent = CrossTableAgent(expand=False, use_rerank=False)
    called = []
    agent._llm_summarize = lambda q, msg, contact: called.append(1) or "LLM ANSWER"

    result = agent.process("Who is 李雅婷?")

    assert called == [], "contact-only query must not call LLM summarize"
    assert "李雅婷" in result["output"] and "hsin-ihu@example.org" in result["output"]
    assert result.get("success") is True
