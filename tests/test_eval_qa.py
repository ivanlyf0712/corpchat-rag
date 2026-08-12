#!/usr/bin/env python3
"""Tests for the eval harness deterministic parts: QA generation, judge parsing,
and LiteLLM usage capture. No LLM / index / network required.
"""
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from eval.qa_generator import generate_qa, _meta, _clean_content


# ── Corpus fixture (metadata-shaped, like the txtai index returns) ──
def _msg(msgid, text, sender, company, label, send_time, kfid):
    return {
        "id": msgid,
        "text": f"{sender} ({label})\n---\n{text}",
        "score": 0.0,
        "metadata": {
            "msgid": msgid, "customer_name": sender, "company": company,
            "label": label, "send_time": send_time, "open_kfid": kfid,
            "servicer_userid": "agent1", "origin": "3",
        },
    }


_CORPUS = [
    _msg("m1", "合同已签，请确认后安排付款。", "陳志明", "鴻海精密工業股份有限公司", "contract_renewal", "2026-01-05T10:00:00", "k1"),
    _msg("m2", "报价已发送，请查收。", "陳志明", "鴻海精密工業股份有限公司", "product_inquiry", "2026-02-03T09:00:00", "k1"),
    _msg("m3", "物流方案报价需要尽快确认。", "張偉強", "長榮航空", "product_inquiry", "2026-02-10T11:00:00", "k2"),
    _msg("m4", "ERP timeout error 影响作业。", "張偉強", "長榮航空", "tech_support", "2026-03-12T14:00:00", "k2"),
    _msg("m5", "请问折扣有什么方案？", "林怡君", "台積電", "price_negotiation", "2026-03-20T15:00:00", "k3"),
]

_CONTACTS = [
    {"full_name": "陳志明", "company": "鴻海精密工業股份有限公司", "email": "chen@example.com"},
    {"full_name": "張偉強", "company": "長榮航空", "email": "zhang@example.com"},
    {"full_name": "林怡君", "company": "台積電", "email": "lin@example.com"},
]


def test_generate_qa_deterministic_and_grounded():
    qa1 = generate_qa(_CORPUS, contacts=_CONTACTS, seed=7, n=20)
    qa2 = generate_qa(_CORPUS, contacts=_CONTACTS, seed=7, n=20)
    assert qa1 == qa2, "same seed must give the same QA set"
    assert qa1, "should produce QA pairs"
    for item in qa1:
        assert item["question"] and item["expected"]
        assert item["type"] in ("message_content", "temporal_window", "multi_hop_entity",
                                "cross_session", "disambiguation", "negation")
        # every non-negation item must cite real evidence ids
        if item["type"] != "negation":
            assert item["evidence"], f"non-negation item missing evidence: {item}"
            ids = {m["id"] for m in _CORPUS}
            assert all(e in ids for e in item["evidence"]), f"evidence not in corpus: {item}"


def test_qa_types_cover_adversarial_cases():
    qa = generate_qa(_CORPUS, contacts=_CONTACTS, seed=3, n=100)
    types = {i["type"] for i in qa}
    assert "message_content" in types
    assert "multi_hop_entity" in types
    assert "temporal_window" in types


def test_negation_requires_absent_term():
    """If every rare term is present in the corpus, negation items can't be generated."""
    from eval import qa_generator as qg
    from eval.qa_generator import _RARE_TERMS
    # corpus contains "折扣" (m5) — generator must skip that term for negation
    items = generate_qa(_CORPUS, contacts=_CONTACTS, seed=3, n=100)
    for item in items:
        if item["type"] == "negation":
            term = item["question"].split("提到过 ")[1].strip("？")
            assert term not in _RARE_TERMS or term != "折扣"


def test_meta_and_clean_content():
    m = _CORPUS[0]
    assert _meta(m)["sender"] == "陳志明"
    assert _clean_content(m) == "合同已签，请确认后安排付款。"


# ── Judge parsing ──────────────────────────────────────────────────
def test_judge_parse_json():
    from eval.judge import _parse_judgment
    parsed = _parse_judgment('{"correct": true, "grounded": false, "hallucination": true, "rationale": "invented"}')
    assert parsed["correct"] is True and parsed["hallucination"] is True
    wrapped = _parse_judgment('Sure: {"correct": false, "grounded": true, "hallucination": false, "rationale": "x"}')
    assert wrapped["correct"] is False
    assert _parse_judgment("no json here") is None


def test_judge_answer_mock_failure_path():
    """A throwing client must degrade to a conservative (not-correct) verdict."""
    from eval.judge import judge_answer

    class _Boom:
        def chat(self, *a, **k):
            raise RuntimeError("network")

    verdict = judge_answer("q", "e", "a", [], _Boom())
    assert verdict["correct"] is False
    assert verdict["hallucination"] is True


# ── LiteLLM usage capture ──────────────────────────────────────────
def test_usage_capture():
    from apps.corpchat.search.litellm_client import LiteLLMClient, reset_usage, usage_total

    client = LiteLLMClient()
    reset_usage()
    client._record_usage({"prompt_tokens": 100, "completion_tokens": 50})
    client._record_usage({"prompt_tokens": 10, "completion_tokens": 5})
    assert client.usage == {"prompt_tokens": 110, "completion_tokens": 55, "calls": 2}
    total = usage_total()
    assert total["prompt_tokens"] == 110 and total["calls"] == 2
    # malformed usage is ignored, not raised
    client._record_usage(None)
    client._record_usage({"prompt_tokens": "x", "completion_tokens": None})
    assert client.usage["calls"] == 2


# ── Slot-based content variation (10k corpus is genuinely different) ─
def test_slot_fill_varies_content_per_repeat():
    """同模板重复生成时内容不同 (数字/产品槽位随机化), 且同一 seed 确定性。"""
    import random
    from apps.corpchat.gen_fake_msg import _randomize_text

    template = "方案报价 8-12% 年化, 最低 300 片, 5000元, 物流方案 細節"
    rng = random.Random(7)
    variants = {_randomize_text(template, rng) for _ in range(20)}
    assert len(variants) > 10, f"槽位填充应产生不同内容: {len(variants)} 变体"

    # 确定性: 同 seed → 同输出序列
    a = [_randomize_text(template, random.Random(3)) for _ in range(3)]
    b = [_randomize_text(template, random.Random(3)) for _ in range(3)]
    assert a == b
    assert a[0] != template, "有数字/产品词的模板应被改写"
