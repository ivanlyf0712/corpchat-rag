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
# ── Eval answer path (agent-smartness-p0: gate + party resolver) ────
class _FakeSearcher:
    """Deterministic Searcher stand-in for answer_question tests."""

    def __init__(self, hits, **kwargs):
        self.hits = hits
        self.last_kwargs = None

    def search(self, query, **kwargs):
        self.last_kwargs = kwargs
        return list(self.hits)


class _CountingLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, *a, **k):
        self.calls += 1
        return "合成的答案"


def test_answer_question_gate_fail_short_circuits_synthesizer():
    """门控失败 → 直接返回 NOT_FOUND_ANSWER, 不调用 synthesizer (无成本)。"""
    from eval.run_baseline import answer_question
    from apps.corpchat.search.answer_path import NOT_FOUND_ANSWER

    # 检索命中与问题不匹配 (错误发送者/主题) → 门控失败
    hits = [{"id": "m1", "text": "李雅婷 (product_inquiry)\n---\n報價已發送",
             "metadata": {"customer_name": "李雅婷", "label": "product_inquiry"}}]
    searcher = _FakeSearcher(hits)
    llm = _CountingLLM()

    resp = answer_question("胡志強 说了什么关于 國榮 的内容？", searcher, llm,
                           known_labels=["product_inquiry"], contacts=[])
    assert resp["answer"] == NOT_FOUND_ANSWER
    assert resp["raw_hits"] == []
    assert resp["citations"] == []
    assert resp["confidence"] == "low"
    assert resp["evidence_gate"] is False
    assert llm.calls == 0, "门控失败时 synthesizer 不应被调用"


def test_answer_question_party_detail_deterministic():
    """party-detail 问题 → 确定性 resolver 一步回答, 不调 synthesizer。"""
    from eval.run_baseline import answer_question

    hits = [{"id": "m1", "text": "廖珮琪 (payment_reminder)\n---\n珮琪 抱歉 抱歉",
             "metadata": {"customer_name": "廖珮琪", "label": "payment_reminder"}}]
    contacts = [{"full_name": "廖珮琪", "userid": "user_廖珮琪_pchen",
                 "company": "勤業眾信", "email": "williamanderson@example.com"}]
    searcher = _FakeSearcher(hits)
    llm = _CountingLLM()

    resp = answer_question("发过关于 這個 的消息的 廖珮琪，他的公司是？", searcher, llm,
                           known_labels=["payment_reminder"], contacts=contacts)
    assert "勤業眾信" in resp["answer"]
    assert "williamanderson@example.com" in resp["answer"]
    assert resp["party_deterministic"] is True
    assert resp["confidence"] == "high"
    assert llm.calls == 0, "party-detail 确定性路径不应调用 synthesizer"
    assert resp["raw_hits"][0]["id"].startswith("contact:"), "联系人记录应作为证据"


def test_answer_question_passes_label_and_window_filter():
    """answer path 把 derive_search_filter 的 label + 窗口传给检索 seam。"""
    from eval.run_baseline import answer_question

    searcher = _FakeSearcher([])
    llm = _CountingLLM()
    resp = answer_question("2026-07 关于 product_inquiry 有什么消息？", searcher, llm,
                           known_labels=["product_inquiry"], contacts=[])
    assert searcher.last_kwargs["label_filter"] == "product_inquiry"
    assert searcher.last_kwargs["date_from"] == "2026-07-01"
    assert searcher.last_kwargs["date_to"] == "2026-07-31"
    # 空 hits → 门控失败 → 诚实 not-found
    assert resp["answer"] == "没有找到相关证据"


# ── Contract-domain eval set (ticket 05) ───────────────────────────
def test_generate_contract_qa_deterministic_and_grounded():
    from eval.contract_qa import generate_contract_qa, _CONTRACT_LABELS

    qa1 = generate_contract_qa(_CORPUS, contacts=_CONTACTS, seed=7, n=30)
    qa2 = generate_contract_qa(_CORPUS, contacts=_CONTACTS, seed=7, n=30)
    assert qa1 == qa2, "same seed must give the same contract QA set"
    assert qa1, "should produce contract QA pairs"
    types = {i["type"] for i in qa1}
    assert types <= {"contract_party", "contract_company", "contract_amount",
                     "contract_date", "contract_clause", "contract_negation"}
    ids = {m["id"] for m in _CORPUS}
    for item in qa1:
        assert item["question"] and item["expected"]
        if item["type"] != "contract_negation":
            assert item["evidence"], f"non-negation missing evidence: {item}"
            assert all(e in ids for e in item["evidence"]), f"evidence not in corpus: {item}"
        # only contract-like labels are used
        assert item.get("label", "") in _CONTRACT_LABELS or item["type"] == "contract_negation"


def test_contract_amount_normalized():
    from eval.contract_qa import _normalize_amount, _extract_amount
    assert _normalize_amount("¥ 320 , 000") == "¥320,000"
    assert _extract_amount("合約 上 是 ¥ 320 , 000") == "¥320,000"
    assert _extract_amount("調漲 8%") == "8%"
    assert _extract_amount("沒有金額的訊息") is None


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


def test_write_spotcheck_no_crash(tmp_path):
    """_write_spotcheck writes a table and never crashes (regression for the
    stray-print NameError that killed the 200-q run before --out JSON)."""
    from eval.run_baseline import _write_spotcheck

    results = [{"id": "qa_1", "type": "negation", "question": "q?", "expected": "e",
                "answer": "没有找到", "correct": True, "grounded": True,
                "rationale": "ok"}]
    out = tmp_path / "spot.md"
    _write_spotcheck(results, str(out), 1)
    text = out.read_text(encoding="utf-8")
    assert "Human spot-check" in text
    assert "qa_1" in text


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
