#!/usr/bin/env python3
"""
Tests for the temporal feature (hindsight-temporal ticket 01): time-expression
parsing and time-sensitive retrieval through the Searcher.search() seam.

Deterministic in-memory index from crafted docs with send_times relative to
the test run, so relative windows (最近N天 / 最近 / 上周 / 去年) behave
deterministically. Uses the production embedding model (BAAI/bge-m3).

Run:
    conda run -n ocr pytest tests/test_search_temporal.py -v
"""
import json
import os
import sys
from datetime import datetime, timedelta

import pytest
import txtai

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.search import Searcher, TimeExpressionParser

EMBEDDING_MODEL = "BAAI/bge-m3"

# 解析器单元测试使用固定 now, 保证确定性
FIXED_NOW = datetime(2026, 8, 7, 12, 0, 0)


# ── TimeExpressionParser 单元测试 ───────────────────────────────
def _win(p, q, now=FIXED_NOW):
    w = p.parse(q, now)
    return (w.start, w.end) if w else None


def test_parser_relative_days():
    p = TimeExpressionParser()
    assert _win(p, "最近7天的消息") == ("2026-07-31", "2026-08-07")


def test_parser_bare_recent_defaults_to_window_days():
    p = TimeExpressionParser()
    assert _win(p, "最近的报价") == ("2026-07-31", "2026-08-07")


def test_parser_day_week_month_year():
    p = TimeExpressionParser()
    assert _win(p, "昨天") == ("2026-08-06", "2026-08-07")
    assert _win(p, "上周") == ("2026-07-27", "2026-08-02")
    assert _win(p, "本月") == ("2026-08-01", "2026-08-07")
    assert _win(p, "上月") == ("2026-07-01", "2026-07-31")
    assert _win(p, "去年") == ("2025-01-01", "2025-12-31")


def test_parser_absolute_date():
    p = TimeExpressionParser()
    assert _win(p, "2025-03-01") == ("2025-03-01", "2025-03-01")
    assert _win(p, "2025年3月") == ("2025-03-01", "2025-03-31")


def test_parser_bare_yyyy_mm():
    """ticket 01: 裸 YYYY-MM 形式必须有规则 (不依赖 LLM 回退)。"""
    p = TimeExpressionParser()
    assert _win(p, "2026-07") == ("2026-07-01", "2026-07-31")
    assert _win(p, "2025-03") == ("2025-03-01", "2025-03-31")
    # 更具体的 YYYY-MM-DD 不被裸月规则误吞
    assert _win(p, "2025-03-01") == ("2025-03-01", "2025-03-01")


def test_parser_bare_yyyy_mm_allow_llm_off_is_pure_rule():
    """allow_llm=False 时裸 YYYY-MM 也走规则, 无 LLM 依赖 (eval 确定性)。"""
    p = TimeExpressionParser(allow_llm=False)
    assert _win(p, "2026-07 关于 product_inquiry 有什么消息？") == ("2026-07-01", "2026-07-31")
    assert _win(p, "物流報價") is None


def test_parser_no_time_intent_returns_none():
    p = TimeExpressionParser()
    assert _win(p, "物流報價") is None
    # 含时间字 (周) 但非时间意图 → 规则不命中, 且不触发 LLM (无数字/范围词)
    assert _win(p, "周末安排") is None


def test_parser_today_inclusive_of_today_docs():
    p = TimeExpressionParser()
    assert _win(p, "今天") == ("2026-08-07", "2026-08-07")


# ── Fixture: 确定性内存索引 (send_time 相对运行时间) ────────────
def _ts(days_ago):
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S")


def _build_temporal_index(tmp_path):
    docs = [
        ("recent_log", "李雅婷 (product_inquiry)\n---\n物流報價方案 100 元 含運費",
         {"label": "product_inquiry", "customer_name": "李雅婷", "company": "鴻海", "send_time": _ts(2)}),
        ("mid_log", "李雅婷 (product_inquiry)\n---\n物流報價方案 120 元 含運費",
         {"label": "product_inquiry", "customer_name": "李雅婷", "company": "鴻海", "send_time": _ts(20)}),
        ("old_log", "李雅婷 (product_inquiry)\n---\n物流報價方案 80 元 含運費",
         {"label": "product_inquiry", "customer_name": "李雅婷", "company": "鴻海", "send_time": _ts(200)}),
        ("recent_inv", "羅思婷 (investment_opportunity)\n---\n投資方案 高回報 每月5%",
         {"label": "investment_opportunity", "customer_name": "羅思婷", "company": "投資顧問", "send_time": _ts(3)}),
        ("old_inv", "羅思婷 (investment_opportunity)\n---\n投資方案 穩健回報 年化8%",
         {"label": "investment_opportunity", "customer_name": "羅思婷", "company": "投資顧問", "send_time": _ts(300)}),
    ]
    embeddings = txtai.Embeddings(
        {"path": EMBEDDING_MODEL, "content": True, "objects": True,
         "hybrid": True, "scoring": {"method": "bm25"}}
    )
    embeddings.index([(d[0], d[1], json.dumps(d[2], default=str)) for d in docs])
    idx_path = os.path.join(tmp_path, "temporal_idx")
    embeddings.save(idx_path)
    return idx_path


@pytest.fixture(scope="module")
def temporal_index(tmp_path_factory):
    return _build_temporal_index(tmp_path_factory.mktemp("temporal"))


@pytest.fixture(scope="module")
def searcher(temporal_index):
    embeddings = txtai.Embeddings()
    embeddings.load(temporal_index)
    return Searcher(embeddings)


# ── Searcher 集成测试 (通过 search() seam) ──────────────────────
def test_search_pure_temporal_returns_recent_docs(searcher):
    """最近N天窗口内文档按时间倒序返回, 不进 RRF。"""
    window_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    results = searcher.search("最近的消息", mode="hybrid", limit=5, expand=False, use_rerank=False)
    assert results, "纯时序查询应返回窗口内文档"
    for r in results:
        assert str(r["metadata"]["send_time"])[:10] >= window_start, "纯时序应只返回最近7天文档"
    times = [str(r["metadata"]["send_time"]) for r in results]
    assert times == sorted(times, reverse=True), "应按时间倒序 (最新在前)"


def test_search_pure_temporal_respects_label_filter(searcher):
    window_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    results = searcher.search("最近的消息", mode="hybrid", limit=5, expand=False,
                              use_rerank=False, label_filter="investment_opportunity")
    assert results
    for r in results:
        assert r["metadata"]["label"] == "investment_opportunity"
        assert str(r["metadata"]["send_time"])[:10] >= window_start


def test_search_combined_time_and_content_filters_window(searcher):
    """时间+内容组合: 放大检索量后按窗口过滤, 窗口外旧文档被排除。"""
    window_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    results = searcher.search("最近的物流報價", mode="hybrid", limit=5, expand=False, use_rerank=False)
    assert results, "组合查询应返回结果"
    for r in results:
        assert str(r["metadata"]["send_time"])[:10] >= window_start, "窗口外文档应被过滤"
    texts = [r["text"] for r in results]
    assert any("物流報價" in t for t in texts), "应包含窗口内的物流报价文档"


def test_search_unchanged_without_time_intent(searcher):
    """无时间意图: 行为与历史一致, 不做窗口过滤。"""
    results = searcher.search("物流報價", mode="hybrid", limit=5, expand=False, use_rerank=False)
    assert results, "无时间意图的查询行为不变"
    texts = [r["text"] for r in results]
    assert any("物流報價" in t for t in texts)


def test_search_explicit_date_overrides_parser(searcher):
    """显式 date_from/date_to 优先, 解析器不覆盖。"""
    # 显式收窄到旧窗口: 只应返回 200 天前的物流文档
    date_from = (datetime.now() - timedelta(days=210)).strftime("%Y-%m-%d")
    date_to = (datetime.now() - timedelta(days=190)).strftime("%Y-%m-%d")
    results = searcher.search("物流報價", mode="hybrid", limit=5, expand=False,
                              use_rerank=False, date_from=date_from, date_to=date_to)
    assert results
    for r in results:
        st = str(r["metadata"]["send_time"])[:10]
        assert date_from <= st <= date_to, "显式日期窗口应生效"
