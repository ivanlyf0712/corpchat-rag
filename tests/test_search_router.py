"""
Tests for the LLM-based search router.
"""

import pytest
from unittest.mock import MagicMock

from apps.corpchat.search import SearchRouter, LiteLLMClient


class TestSearchRouter:
    """Test SearchRouter decision logic."""

    def test_router_returns_search_true_for_factual_query(self, monkeypatch):
        """Factual questions should route to search=true."""
        fake_client = MagicMock()
        fake_client.chat.return_value = '{"search": true, "query": "logistics quotation"}'
        monkeypatch.setattr("apps.corpchat.search.router.LiteLLMClient", lambda *a, **k: fake_client)
        router = SearchRouter(api_base="http://localhost", api_key="k", model="m")
        decision = router.decide(" logistics quotation ")
        assert decision["search"] is True
        assert decision["query"] == "logistics quotation"

    def test_router_returns_search_false_for_greeting(self, monkeypatch):
        """Greetings should route to search=false."""
        fake_client = MagicMock()
        fake_client.chat.return_value = '{"search": false, "query": "Hello!"}'
        monkeypatch.setattr("apps.corpchat.search.router.LiteLLMClient", lambda *a, **k: fake_client)
        router = SearchRouter(api_base="http://localhost", api_key="k", model="m")
        decision = router.decide("Hello!")
        assert decision["search"] is False
        assert decision["query"] == "Hello!"

    def test_router_defaults_to_search_on_json_parse_failure(self, monkeypatch):
        """Garbage LLM output → safe default search=true."""
        fake_client = MagicMock()
        fake_client.chat.return_value = "I am not sure."
        monkeypatch.setattr("apps.corpchat.search.router.LiteLLMClient", lambda *a, **k: fake_client)
        router = SearchRouter(api_base="http://localhost", api_key="k", model="m")
        decision = router.decide("vague input")
        assert decision["search"] is True
        assert decision["query"] == "vague input"

    def test_router_handles_json_in_text_wrapper(self, monkeypatch):
        """LLM wrapped JSON in extra prose → still parses."""
        fake_client = MagicMock()
        fake_client.chat.return_value = 'Sure: {"search": false, "query": "thanks"}'
        monkeypatch.setattr("apps.corpchat.search.router.LiteLLMClient", lambda *a, **k: fake_client)
        router = SearchRouter(api_base="http://localhost", api_key="k", model="m")
        decision = router.decide("thanks")
        assert decision["search"] is False
        assert decision["query"] == "thanks"

    def test_router_falls_back_when_client_raises(self, monkeypatch):
        """LLM exception → safe default search=true."""
        fake_client = MagicMock()
        fake_client.chat.side_effect = RuntimeError("network")
        monkeypatch.setattr("apps.corpchat.search.router.LiteLLMClient", lambda *a, **k: fake_client)
        router = SearchRouter(api_base="http://localhost", api_key="k", model="m")
        decision = router.decide("anything")
        assert decision["search"] is True
        assert decision["query"] == "anything"
        assert decision["raw"] == ""
