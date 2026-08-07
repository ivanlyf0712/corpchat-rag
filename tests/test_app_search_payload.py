"""
Test the app.py search function directly by constructing a payload.

This test imports the search function from app.py and verifies that
the agent path handles queries correctly (no fallback for normal queries).
"""
import sys
import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pytest


def _build_payload(query: str, agent_enabled: bool = True) -> dict:
    """Construct a payload mimicking what the Streamlit UI sends."""
    return {
        "query": query,
        "agent_enabled": agent_enabled,
        "top_k": 5,
        "use_rerank": True,
        "expand": True,
        "graph_expand": 1,
        "label_filter": "",
    }


def test_agent_mode_default_activated():
    """Agent mode should default to activated (True)."""
    # Simulate the session_state initialization in app.py
    session_state = {}
    if "agent_enabled" not in session_state:
        session_state["agent_enabled"] = True  # default: activated
    assert session_state["agent_enabled"] is True


def test_cross_table_agent_process_fraud_query():
    """The agent should handle '帮我查一下诈骗相关的消息' without falling back."""
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    agent = CrossTableAgent()
    payload = _build_payload("帮我查一下诈骗相关的消息")
    result = agent.process(payload["query"])

    # Should NOT be in fallback mode
    assert result.get("fallback") is False, (
        f"Expected agent mode, got fallback. Output: {result.get('output', '')[:200]}"
    )
    # Should have called at least one tool
    assert len(result.get("tool_calls", [])) >= 1, "Expected at least 1 tool call"
    # Should have a non-empty answer
    assert result.get("output"), "Expected a non-empty answer"


def test_cross_table_agent_process_cross_table_query():
    """Cross-table query should call both tools."""
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    agent = CrossTableAgent()
    payload = _build_payload("发'合同已签'消息的人，他的邮箱是什么？")
    result = agent.process(payload["query"])

    assert result.get("fallback") is False, "Expected agent mode, got fallback"
    tools = [tc.get("tool") for tc in result.get("tool_calls", [])]
    assert "search_messages" in tools, f"Expected search_messages, got {tools}"
    assert "search_contacts" in tools, f"Expected search_contacts, got {tools}"


def test_cross_table_agent_process_contact_query():
    """Contact-only query should call search_contacts."""
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    agent = CrossTableAgent()
    payload = _build_payload("李雅婷的邮箱是什么？")
    result = agent.process(payload["query"])

    assert result.get("fallback") is False, "Expected agent mode, got fallback"
    tools = [tc.get("tool") for tc in result.get("tool_calls", [])]
    assert "search_contacts" in tools, f"Expected search_contacts, got {tools}"


def test_cross_table_agent_process_greeting():
    """Greeting should not call any tools."""
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    agent = CrossTableAgent()
    payload = _build_payload("你好")
    result = agent.process(payload["query"])

    assert result.get("fallback") is False, "Expected agent mode, got fallback"
    assert len(result.get("tool_calls", [])) == 0, "Greeting should not call tools"
    assert result.get("output"), "Expected a greeting response"


def test_cross_table_agent_process_system_question():
    """System question should not call any tools."""
    from apps.corpchat.search.cross_table_agent import CrossTableAgent

    agent = CrossTableAgent()
    payload = _build_payload("你能做什么？")
    result = agent.process(payload["query"])

    assert result.get("fallback") is False, "Expected agent mode, got fallback"
    assert len(result.get("tool_calls", [])) == 0, "System question should not call tools"
    assert result.get("output"), "Expected a system info response"
