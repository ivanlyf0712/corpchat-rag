#!/usr/bin/env python3
"""
Tests for ticket 03 of agent-config-panel: Hindsight memory graph.

Seams:
  - `memory_graph.build_entity_graph()` — pure entity/relationship extraction
  - `core.db.save_memory_graph` / `load_memory_graph` — persistence (mocked conn)

Run:
    conda run -n ocr pytest tests/test_memory_graph.py -v
"""
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.search.memory_graph import build_entity_graph

MESSAGES = [
    {"id": "m1", "text": "物流報價方案 100 元，含運費",
     "metadata": {"customer_name": "陳志明", "company": "鴻海", "label": "product_inquiry",
                  "external_userid": "user_chen", "open_kfid": "kf_1", "send_time": "2026-08-01T10:00:00"}},
    {"id": "m2", "text": "好的，物流報價稍後發送",
     "metadata": {"customer_name": "許志豪", "company": "鴻海", "label": "product_inquiry",
                  "external_userid": "user_hsu", "open_kfid": "kf_1", "send_time": "2026-08-01T10:05:00"}},
    {"id": "m3", "text": "這是詐騙連結，請勿點擊",
     "metadata": {"customer_name": "羅思婷", "company": "投資顧問", "label": "old_friend_reconnect",
                  "external_userid": "user_lo", "open_kfid": "kf_2", "send_time": "2026-08-02T09:00:00"}},
]


def _graph(**kw):
    return build_entity_graph(MESSAGES, **kw)


def test_graph_has_person_company_label_nodes():
    g = _graph()
    types = {n["type"] for n in g["nodes"]}
    assert "person" in types and "company" in types and "label" in types
    labels = {n["label"] for n in g["nodes"]}
    assert "陳志明" in labels and "鴻海" in labels and "product_inquiry" in labels


def test_graph_person_company_association_edge():
    g = _graph()
    rels = {(e["source"], e["target"], e["type"]) for e in g["edges"]}
    assert ("person:陳志明", "company:鴻海", "association") in rels


def test_graph_person_person_same_conversation_edge():
    g = _graph()
    rels = {(e["source"], e["target"]) for e in g["edges"]}
    assert ("person:陳志明", "person:許志豪") in rels, "同会话人-人关联边应存在"


def test_graph_mention_and_reference_edges():
    g = _graph()
    types = {e["type"] for e in g["edges"]}
    assert "mention" in types and "reference" in types
    refs = {(e["source"], e["target"]) for e in g["edges"] if e["type"] == "reference"}
    assert ("label:old_friend_reconnect", "m3") in refs


def test_graph_keyword_nodes_from_text():
    g = _graph()
    kws = [n["label"] for n in g["nodes"] if n["type"] == "keyword"]
    assert any("物流" in k or "報價" in k for k in kws), f"关键词节点缺失: {kws}"


def test_graph_sources_gate_contacts():
    """sources 排除 contacts → 无 person/company 节点。"""
    g = _graph(sources=["messages"])
    types = {n["type"] for n in g["nodes"]}
    assert "person" not in types and "company" not in types
    assert "label" in types


def test_graph_risk_labels_highlighted_when_skeptical():
    """risk_labels 中标签在怀疑度模式下被高亮。"""
    g = _graph(risk_labels={"old_friend_reconnect"})
    hi = [n["label"] for n in g["nodes"] if n.get("highlighted")]
    assert "old_friend_reconnect" in hi


def test_graph_node_size_reflects_degree():
    g = _graph()
    sizes = {n["label"]: n["size"] for n in g["nodes"] if n["type"] == "person"}
    # 陈志明与多人/公司/标签/消息相连 → size 大于孤立节点
    assert sizes.get("陳志明", 0) > 0
    assert sizes.get("許志豪", 0) > 0


def test_memory_graph_persistence(monkeypatch):
    """save/load memory_graph round-trip via core.db (mocked connection)."""
    import core.db as db_module

    store = {}

    class _FakeCur:
        def __init__(self):
            self._row = None

        def execute(self, sql, *args):
            params = args[0] if args else ()
            if sql.strip().upper().startswith("SELECT"):
                sid = params[0]
                self._row = (store.get(sid),) if sid in store else None
            else:
                self._row = None
                store[params[0]] = params[1]

        def fetchone(self):
            return self._row

        def close(self):
            pass

    class _FakeConn:
        def cursor(self):
            return _FakeCur()

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(db_module, "get_db_connection", lambda: _FakeConn())

    from core.db import load_memory_graph, save_memory_graph

    g = _graph()
    save_memory_graph("sess_g", g)
    loaded = load_memory_graph("sess_g")
    assert loaded and len(loaded["nodes"]) == len(g["nodes"])
    assert load_memory_graph("sess_missing") is None
