#!/usr/bin/env python3
"""
Tests for the persona layer (hindsight-persona ticket 01): tunable disposition
profile that conditions answer generation via system-prompt injection.

Seams: `DispositionProfile.build_system_prompt(base_prompt)` (unit) and the
three answer points (integration, mocked LLM).

Run:
    conda run -n ocr pytest tests/test_persona.py -v
"""
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.search.persona import DispositionProfile


def test_default_profile_is_neutral():
    """中性默认 (0.5, balanced) 不追加实质性性格指令。"""
    prompt = DispositionProfile().build_system_prompt("You are a helpful assistant.")
    assert "You are a helpful assistant." in prompt
    # 中性不应包含 trait 专属指令
    assert "不確定" not in prompt and "原文" not in prompt


def test_high_skepticism_adds_uncertainty_instruction():
    prompt = DispositionProfile(skepticism=0.9).build_system_prompt("base")
    assert "不確定" in prompt or "不确定性" in prompt


def test_high_literality_adds_literal_instruction():
    prompt = DispositionProfile(literality=0.9).build_system_prompt("base")
    assert "原文" in prompt


def test_high_empathy_adds_empathetic_instruction():
    prompt = DispositionProfile(empathy=0.9).build_system_prompt("base")
    assert "情緒" in prompt or "语气" in prompt


def test_detailed_style_adds_detail_instruction():
    prompt = DispositionProfile(style="detailed").build_system_prompt("base")
    assert "詳細" in prompt


def test_profile_serializes_to_dict_and_back():
    p = DispositionProfile(skepticism=0.7, literality=0.2, empathy=0.9, style="concise")
    d = p.to_dict()
    p2 = DispositionProfile.from_dict(d)
    assert p2.skepticism == 0.7
    assert p2.style == "concise"


def test_disposition_profile_persistence(monkeypatch):
    """save/load round-trip via core.db (mocked psycopg2 connection)."""
    import core.db as db_module

    store = {}

    class _FakeCur:
        def __init__(self):
            self._row = None

        def execute(self, sql, *args):
            params = args[0] if args else ()
            if sql.strip().upper().startswith("SELECT"):
                sid = params[0]
                self._row = None
                if sid in store:
                    p = store[sid]
                    self._row = (p["skepticism"], p["literality"], p["empathy"], p["style"])
            else:  # INSERT ... ON CONFLICT
                sid, sk, li, em, st = params
                store[sid] = {"skepticism": sk, "literality": li, "empathy": em, "style": st}

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

    from core.db import load_disposition_profile, save_disposition_profile

    save_disposition_profile(
        "sess_1", {"skepticism": 0.9, "literality": 0.2, "empathy": 0.8, "style": "concise"}
    )
    loaded = load_disposition_profile("sess_1")
    assert loaded["skepticism"] == 0.9
    assert loaded["style"] == "concise"
    assert load_disposition_profile("sess_missing") is None


# ── Injection at the three answer points ─────────────────────────
def test_app_generate_answer_applies_profile(monkeypatch):
    """app.generate_answer_litellm 带 profile → system prompt 含性格指令。"""
    from apps.corpchat import app as app_module

    captured = {}

    def _fake_chat(messages, **kw):
        captured["system"] = messages[0]["content"]
        return "answer"

    monkeypatch.setattr(app_module._llm_client, "chat", _fake_chat)

    ans = app_module.generate_answer_litellm("測試問題", "ctx", profile=DispositionProfile(skepticism=0.9))
    assert ans == "answer"
    assert "不確定" in captured["system"]


def test_agent_generate_answer_applies_profile(monkeypatch):
    """agent._generate_answer 带 profile → system prompt 含性格指令。"""
    from apps.corpchat import agent as agent_module
    from apps.corpchat.agent import Agent, IntentClassifier

    captured = {}

    def _fake_chat(messages, **kw):
        captured["system"] = messages[0]["content"]
        return "answer"

    monkeypatch.setattr(agent_module._llm_client, "chat", _fake_chat)

    a = Agent(searcher=None, classifier=IntentClassifier(lite_llm_available=False))
    ans = a._generate_answer("問題", "ctx", profile=DispositionProfile(literality=0.9))
    assert ans == "answer"
    assert "原文" in captured["system"]


def test_cross_table_summarize_applies_profile(monkeypatch):
    """CrossTableAgent 带 profile → _llm_summarize 的 system prompt 含性格指令。"""
    from apps.corpchat.search.cross_table_agent import CrossTableAgent
    from apps.corpchat.search.litellm_client import LiteLLMClient

    captured = {}

    def _fake_chat(self, messages, **kw):
        captured["system"] = messages[0]["content"]
        return "answer"

    monkeypatch.setattr(LiteLLMClient, "chat", _fake_chat)

    agent = CrossTableAgent(profile=DispositionProfile(empathy=0.9))
    out = agent._llm_summarize("問題", "msg結果", "contact結果")
    assert out == "answer"
    assert "情緒" in captured["system"]
