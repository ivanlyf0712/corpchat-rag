"""
CorpChat Search — LLM Router
================================
Wraps the existing Searcher with an LLM-based routing decision.

Decision format: {"search": true/false, "query": "..."}

Behavior:
  - JSON parse failure → safe default: search=true
  - search=true  → run search pipeline → summarize results via LLM
  - search=false → return LLM's direct reply without search
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import LITELLM_MODEL, logger
from .litellm_client import LiteLLMClient


class SearchRouter:
    """LLM-driven gate that decides whether to search or chat."""

    _SYSTEM_PROMPT = (
        "You are a corporate chat search assistant. "
        "Decide whether the user's message requires searching the message corpus. "
        "Respond with ONLY a JSON object: {\"search\": true/false, \"query\": \"...\"}. "
        "Rules: "
        "1) Greetings, thanks, small talk, emotions → search=false. "
        "2) Factual questions about messages, contracts, records, scams, logistics → search=true. "
        "3) Vague or ambiguous input → search=false and keep the original query. "
        "4) If search=true, rewrite the query to a concise retrieval query."
    )

    def __init__(self, api_base: Optional[str] = None, api_key: Optional[str] = None, model: str = LITELLM_MODEL):
        self._client = LiteLLMClient(api_base=api_base, api_key=api_key, model=model)

    def decide(self, user_message: str) -> Dict[str, Any]:
        """
        Return a decision dict with keys:
          - search: bool
          - query: str
          - raw: str (original LLM text)
        """
        decision = {"search": True, "query": user_message, "raw": ""}
        try:
            raw = self._client.chat(
                [
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=120,
                timeout=10,
            )
            decision["raw"] = raw or ""
            parsed = self._parse_json(raw or "")
            if parsed is not None:
                decision["search"] = bool(parsed.get("search", True))
                q = parsed.get("query")
                decision["query"] = q if isinstance(q, str) and q.strip() else user_message
        except Exception as e:
            logger.debug(f"Router decision failed: {e}")
        return decision

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON parse from noisy LLM output."""
        if not text:
            return None
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None