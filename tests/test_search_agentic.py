#!/usr/bin/env python3
"""
Tests for ticket 01 of hindsight-adaptive-paths: AgenticDecider decides when to
activate the graph-parallel retrieval path for relationship-oriented queries.

Seam: `AgenticDecider.decide(query)` — the public decision interface.
Offline-deterministic: the LLM client is stubbed to return nothing, so only the
rule-first path is exercised.

Run:
    conda run -n ocr pytest tests/test_search_agentic.py -v
"""
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.search.agentic import AgenticDecider


def _decider() -> AgenticDecider:
    """Offline AgenticDecider: LLM returns nothing, so rules are authoritative."""
    d = AgenticDecider()
    d._client.chat = lambda *a, **k: ""
    return d


def test_graph_parallel_enabled_for_relationship_query():
    """'跟誰聊過物流' 是关系查询 → 激活图并行路。"""
    assert _decider().decide("跟誰聊過物流")["graph_parallel"] is True


def test_graph_parallel_enabled_for_who_sent_message():
    """'發...消息的人' 需要关系上下文 → 激活图并行路。"""
    assert _decider().decide("發'合同已簽'消息的人")["graph_parallel"] is True


def test_graph_parallel_enabled_for_combined_time_relationship():
    """'上個月跟陳志明聊的物流報價' 时序+关系 → 激活图并行路。"""
    assert _decider().decide("上個月跟陳志明聊的物流報價")["graph_parallel"] is True


def test_graph_parallel_disabled_for_plain_content_query():
    """纯内容查询不激活图并行路 (默认 False)。"""
    assert _decider().decide("物流報價 方案")["graph_parallel"] is False


def test_decision_dict_backward_compatible():
    """决策字典保留既有键, 新增键不破坏调用方。"""
    decision = _decider().decide("物流報價")
    for key in ("mode", "expand", "graph_expand", "use_rerank", "graph_parallel"):
        assert key in decision
