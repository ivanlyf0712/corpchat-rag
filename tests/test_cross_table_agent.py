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
