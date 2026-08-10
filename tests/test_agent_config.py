#!/usr/bin/env python3
"""
Tests for ticket 01 of agent-config-panel: unified agent configuration model.

Seams:
  - `agent_config.default_agent_config()` / `apply_preset()` — pure config logic
  - `core.db.load_agent_config` / `save_agent_config` — persistence (mocked conn)
  - `Searcher`/answer points — unchanged (config maps onto existing pipeline)

Run:
    conda run -n ocr pytest tests/test_agent_config.py -v
"""
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.search.agent_config import (
    CARA_PRESETS,
    apply_preset,
    default_agent_config,
)


def test_default_config_shape():
    """默认配置含 persona/search/knowledge 三区块且与现有默认行为一致。"""
    cfg = default_agent_config()
    assert set(cfg.keys()) == {"persona", "search", "knowledge"}
    # 现有默认行为: expand=True, rerank=True, depth=deep, graph_hops=1
    assert cfg["search"]["expand"] is True
    assert cfg["search"]["rerank"] is True
    assert cfg["search"]["depth"] == "deep"
    assert cfg["search"]["graph_hops"] == 1
    assert cfg["persona"]["style"] == "standard"
    assert cfg["knowledge"]["sources"] == ["messages", "contacts"]
    assert cfg["knowledge"]["citations"] is False


def test_default_config_is_deep_copied():
    """default_agent_config() 每次返回独立副本, 互不影响。"""
    a = default_agent_config()
    b = default_agent_config()
    a["search"]["expand"] = False
    assert b["search"]["expand"] is True


def test_apply_audit_preset():
    cfg = apply_preset(default_agent_config(), "审计助手")
    p = cfg["persona"]
    assert p["skepticism"] == 8 and p["literality"] == 7 and p["empathy"] == 3
    assert p["preset"] == "audit"


def test_apply_service_preset():
    cfg = apply_preset(default_agent_config(), "客服助手")
    assert cfg["persona"]["empathy"] == 8
    assert cfg["persona"]["preset"] == "service"


def test_apply_research_preset():
    cfg = apply_preset(default_agent_config(), "研究助理")
    assert cfg["persona"]["style"] == "detailed"
    assert cfg["persona"]["preset"] == "research"


def test_custom_preset_keeps_values():
    """'自訂' 不改动 persona 值, 只标记 preset=custom。"""
    cfg = default_agent_config()
    cfg["persona"]["skepticism"] = 9
    apply_preset(cfg, "自訂")
    assert cfg["persona"]["skepticism"] == 9
    assert cfg["persona"]["preset"] == "custom"


def test_presets_are_0_to_10():
    """预设值处于 0-10 滑杆范围。"""
    for name, vals in CARA_PRESETS.items():
        for key in ("skepticism", "literality", "empathy"):
            assert 0 <= vals[key] <= 10, f"{name}.{key} out of range"


def test_persona_to_profile_dict_scales_to_unit():
    """persona_to_profile_dict 把 0-10 换算为 0-1 (DispositionProfile 输入)。"""
    from apps.corpchat.search.agent_config import persona_to_profile_dict

    d = persona_to_profile_dict({"skepticism": 8, "literality": 7, "empathy": 3, "style": "detailed"})
    assert d["skepticism"] == 0.8
    assert d["literality"] == 0.7
    assert d["empathy"] == 0.3
    assert d["style"] == "detailed"


def test_sources_label_key_roundtrip():
    """数据源标签↔内部 key 互转; multiselect default 必须 ⊆ options (回归 #575)。"""
    from apps.corpchat.search.agent_config import (
        SOURCE_OPTIONS,
        sources_from_labels,
        sources_to_labels,
    )

    labels = sources_to_labels(["messages", "contacts"])
    assert labels == ["消息", "联系人"]
    # Streamlit 约束: 所有 default 值必须在 options 中
    assert all(l in SOURCE_OPTIONS for l in labels), f"default 超出 options: {labels}"
    assert sources_from_labels(labels) == ["messages", "contacts"]
    assert sources_to_labels(["messages"]) == ["消息"]
    assert sources_from_labels(["联系人"]) == ["contacts"]


def test_agent_config_persistence(monkeypatch):
    """save/load agent_config round-trip via core.db (mocked connection)."""
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
            else:  # INSERT ... ON CONFLICT
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

    from core.db import load_agent_config, save_agent_config

    cfg = default_agent_config()
    cfg["search"]["expand"] = False
    save_agent_config("sess_cfg", cfg)
    loaded = load_agent_config("sess_cfg")
    assert loaded["search"]["expand"] is False
    assert loaded["persona"]["style"] == "standard"
    assert load_agent_config("sess_missing") is None
