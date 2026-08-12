#!/usr/bin/env python3
"""Tests for the deterministic answer-path helpers (agent-smartness-p0 tickets 01–04).

Covers the pure, LLM-free building blocks in apps.corpchat.search.answer_path:
  - derive_search_filter / extract_label_filter (ticket 01)
  - extract_keywords / extract_names / evidence_passes / compute_confidence (ticket 02)
  - resolve_party_detail / first_party_detail / party_detail_text /
    party_answer_text / is_party_detail_question (ticket 03)
  - detect_agent_mode (ticket 04)

No LLM / index / network required (prior art: tests/test_eval_qa.py).
"""
import sys
import os
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from apps.corpchat.search.answer_path import (
    NOT_FOUND_ANSWER,
    compute_confidence,
    derive_search_filter,
    detect_agent_mode,
    evidence_passes,
    extract_keywords,
    extract_label_filter,
    extract_names,
    first_party_detail,
    is_party_detail_question,
    party_answer_text,
    party_detail_text,
    resolve_party_detail,
)

FIXED_NOW = datetime(2026, 8, 7, 12, 0, 0)
LABELS = ["product_inquiry", "investment_opportunity", "warranty_claim", "tech_support"]


# ── Ticket 01: label filter + search filter derivation ────────────
class TestDeriveSearchFilter:
    def test_bare_yyyy_mm_window(self):
        f = derive_search_filter("2026-07 关于 product_inquiry 有什么消息？",
                                 known_labels=LABELS, now=FIXED_NOW)
        assert f["label_filter"] == "product_inquiry"
        assert f["date_from"] == "2026-07-01"
        assert f["date_to"] == "2026-07-31"

    def test_no_time_intent_no_filter(self):
        f = derive_search_filter("物流報價 方案", known_labels=LABELS, now=FIXED_NOW)
        assert f["label_filter"] is None
        assert f["date_from"] is None and f["date_to"] is None

    def test_known_label_in_question(self):
        assert extract_label_filter("2026-08 关于 warranty_claim 的消息",
                                    known_labels=LABELS) == "warranty_claim"
        assert extract_label_filter("随便聊聊", known_labels=LABELS) is None

    def test_yyyy_mm_does_not_shadow_yyyy_mm_dd(self):
        f = derive_search_filter("2025-03-01 的订单", known_labels=LABELS, now=FIXED_NOW)
        assert f["date_from"] == "2025-03-01" and f["date_to"] == "2025-03-01"


# ── Ticket 02: keywords, names, evidence gate, confidence ─────────
class TestEvidenceGate:
    def _hit(self, hid, text, sender, label="x", uid="user_x"):
        return {"id": hid, "text": f"{sender} ({label})\n---\n{text}",
                "metadata": {"customer_name": sender, "external_userid": uid,
                             "label": label}}

    def test_message_content_gate_passes(self):
        hits = [self._hit("m1", "國榮 ， 我們 台達 三廠 的 生產 設備", "胡志強")]
        assert evidence_passes("胡志強 说了什么关于 國榮 的内容？", hits) is True

    def test_message_content_gate_fails_wrong_sender(self):
        # 人名与主题分属不同 hit → 单 hit 覆盖规则拒绝 (弱证据)
        hits = [self._hit("m1", "報價 已 發送", "李雅婷"),
                self._hit("m2", "國榮 ， 我們 台達 三廠", "胡志強")]
        assert evidence_passes("李雅婷 说了什么关于 國榮 的内容？", hits) is False

    def test_negation_no_hits(self):
        assert evidence_passes("有没有人提到过 退款？", []) is False

    def test_negation_unrelated_hits(self):
        hits = [self._hit("m1", "報價 已 發送", "李雅婷")]
        assert evidence_passes("有没有人提到过 退款？", hits) is False

    def test_temporal_label_in_metadata(self):
        hits = [{"id": "w1", "text": "怡萱 你好 請問是幾台問題",
                 "metadata": {"label": "warranty_claim"}}]
        assert evidence_passes("2026-08 关于 warranty_claim 有什么消息？", hits) is True

    def test_empty_question_empty_hits(self):
        assert evidence_passes("", []) is False

    def test_no_hits_always_false(self):
        assert evidence_passes("胡志強 说了什么关于 國榮 的内容？", []) is False

    def test_party_detail_question_gate_relaxed(self):
        # 多跳问题: 发送者出现即通过 (公司来自联系人表, 主题非必需)
        hits = [self._hit("m1", "珮琪 抱歉 抱歉", "廖珮琪")]
        assert evidence_passes("发过关于 這個 的消息的 廖珮琪，他的公司是？", hits) is True


class TestConfidence:
    def _hit(self, hid, text, sender):
        return {"id": hid, "text": f"{sender} (x)\n---\n{text}",
                "metadata": {"customer_name": sender, "label": "x"}}

    def test_low_when_gate_fails(self):
        assert compute_confidence(False, "有没有人提到过 退款？", []) == "low"

    def test_high_when_top_hit_covers_keyword(self):
        hits = [self._hit("m1", "國榮 ， 我們 台達 三廠 的 生產 設備", "胡志強")]
        assert compute_confidence(True, "胡志強 说了什么关于 國榮 的内容？", hits) == "high"

    def test_medium_when_keyword_lower(self):
        hits = [self._hit("m1", "報價 已 發送", "李雅婷"),
                self._hit("m2", "國榮 ， 我們 台達 三廠", "胡志強")]
        assert evidence_passes("胡志強 说了什么关于 國榮 的内容？", hits) is True
        assert compute_confidence(True, "胡志強 说了什么关于 國榮 的内容？", hits) == "medium"

        f = derive_search_filter("2025-03-01 的订单", known_labels=LABELS, now=FIXED_NOW)
        assert f["date_from"] == "2025-03-01" and f["date_to"] == "2025-03-01"

# ── Ticket 03: cross-table party resolver ─────────────────────────
_CONTACTS = [
    {"full_name": "廖珮琪", "userid": "user_廖珮琪_pchen", "company": "勤業眾信",
     "email": "williamanderson@example.com", "phone": "0912345678"},
    {"full_name": "胡志強", "userid": "user_胡志強_h", "company": "台達電子",
     "email": "chen@example.net"},
]


class TestPartyResolver:
    def _hit(self, sender, uid, text="內容"):
        return {"id": "m1", "text": f"{sender} (x)\n---\n{text}",
                "metadata": {"customer_name": sender, "external_userid": uid, "label": "x"}}

    def test_resolve_by_customer_name(self):
        c = resolve_party_detail(self._hit("廖珮琪", "user_廖珮琪_pchen"), _CONTACTS)
        assert c is not None and c["company"] == "勤業眾信"

    def test_resolve_by_userid(self):
        c = resolve_party_detail(self._hit("廖珮琪", "user_廖珮琪_pchen"), _CONTACTS)
        assert c["email"] == "williamanderson@example.com"

    def test_unknown_sender_returns_none(self):
        h = self._hit("王小明", "user_王小明_x")
        assert resolve_party_detail(h, _CONTACTS) is None

    def test_empty_contacts_graceful(self):
        assert resolve_party_detail(self._hit("廖珮琪", "u"), []) is None
        assert first_party_detail("x", [self._hit("廖珮琪", "u")], []) is None

    def test_first_party_prefers_question_sender(self):
        hits = [self._hit("胡志強", "user_胡志強_h", "三年約的價格不錯")]
        c = first_party_detail("发过关于 三年 的消息的 胡志強，他的公司是？", hits, _CONTACTS)
        assert c["company"] == "台達電子"

    def test_party_detail_text_and_answer_text(self):
        assert "勤業眾信" in party_detail_text(_CONTACTS[0])
        ans = party_answer_text(_CONTACTS[0])
        assert ans == "廖珮琪 的公司是 勤業眾信, 邮箱 williamanderson@example.com"

    def test_is_party_detail_question(self):
        assert is_party_detail_question("发过关于 三年 的消息的 胡志強，他的公司是？") is True
        assert is_party_detail_question("胡志強 说了什么关于 國榮 的内容？") is False
        assert is_party_detail_question("有没有人提到过 退款？") is False

    def test_is_party_detail_question_fang_surname(self):
        """姓氏补集回归: 方志遠 的 '方' 必须能被识别为发送者。"""
        q = "发过关于 麻煩 的消息的 方志遠，他的公司是？"
        assert extract_names(q) == ["方志遠"]
        assert is_party_detail_question(q) is True


# ── Ticket 04: rule detector ──────────────────────────────────────
class TestDetectAgentMode:
    def test_multi_hop_escalates(self):
        assert detect_agent_mode("发过关于 合同 的消息的 陳志明，他的公司是？") == "agent"

    def test_cross_session_escalates(self):
        assert detect_agent_mode("之前 鍾佩珊 说过关于 王經理 的内容吗？") == "agent"

    def test_time_escalates(self):
        assert detect_agent_mode("2026-07 关于 product_inquiry 有什么消息？") == "agent"

    def test_default_pipeline(self):
        assert detect_agent_mode("物流報價 方案 多少錢") == "pipeline"
        assert detect_agent_mode("hello how are you") == "pipeline"
        assert detect_agent_mode("") == "pipeline"


# ── 常量 ───────────────────────────────────────────────────────────
def test_not_found_answer_constant():
    assert NOT_FOUND_ANSWER == "没有找到相关证据"


def test_extract_keywords_frames_stripped():
    assert extract_keywords("胡志強 说了什么关于 國榮 的内容？") == ["胡志強", "國榮"]
    assert extract_keywords("发过关于 這個 的消息的 廖珮琪，他的公司是？") == ["廖珮琪"]
    assert extract_names("发过关于 這個 的消息的 廖珮琪，他的公司是？") == ["廖珮琪"]

