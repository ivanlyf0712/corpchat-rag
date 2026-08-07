"""
CorpChat Search — LiteLLM Client
=================================
Single client for all LiteLLM API calls. Encapsulates the HTTP request,
error handling, and availability check so callers don't duplicate the
same requests.post() pattern.
"""

from typing import Dict, List, Optional

import requests

from .config import (
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LITELLM_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    logger,
)


class LiteLLMClient:
    """Thin wrapper around the LiteLLM OpenAI-compatible chat completions API.

    Falls back to DeepSeek (OpenAI-compatible) when the primary LLM is
    unreachable or returns an error, so the app degrades gracefully.
    """

    def __init__(self, api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: str = LITELLM_MODEL):
        self.api_base = (api_base or LITELLM_BASE_URL).rstrip("/")
        self.api_key = api_key or LITELLM_API_KEY
        self.model = model

    def _deepseek_chat(self, messages: List[Dict], temperature: float,
                       max_tokens: int, timeout: int) -> str:
        """Call DeepSeek's OpenAI-compatible endpoint as a fallback LLM."""
        if not DEEPSEEK_API_KEY:
            logger.warning("DeepSeek fallback skipped: DEEPSEEK_API_KEY not set")
            return ""
        try:
            resp = requests.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                json={
                    "model": DEEPSEEK_MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                logger.info("DeepSeek fallback LLM used successfully")
                return content
            logger.warning(f"DeepSeek fallback failed ({resp.status_code}): {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"DeepSeek fallback exception: {e}")
        return ""

    def chat(self, messages: List[Dict], temperature: float = 0.1,
             max_tokens: int = 200, timeout: int = 15) -> str:
        """
        Send a chat completion request and return the assistant's content.

        Tries the primary OpenAI-compatible endpoint first, then falls back to
        Ollama's native `/api/chat`, then finally to DeepSeek.
        Returns empty string on any failure (graceful degradation).
        """
        # 1) Try OpenAI-compatible endpoint
        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            if resp.status_code != 404:
                logger.warning(f"LLM 调用失败 ({resp.status_code}): {resp.text[:200]}")
                # Non-404 error → try DeepSeek fallback
                return self._deepseek_chat(messages, temperature, max_tokens, timeout)
        except requests.exceptions.RequestException as e:
            logger.warning(f"LLM 调用异常: {e}")
            # Network error → try DeepSeek fallback
            return self._deepseek_chat(messages, temperature, max_tokens, timeout)

        # 2) Fallback: Ollama native /api/chat
        try:
            resp = requests.post(
                f"{self.api_base}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.warning(f"LLM 调用失败 (Ollama fallback): {e}")
            # Ollama also failed → try DeepSeek fallback
            return self._deepseek_chat(messages, temperature, max_tokens, timeout)

    def is_available(self, timeout: int = 3) -> bool:
        """Quick check if the LiteLLM endpoint is reachable."""
        if not self.api_key:
            return False
        try:
            resp = requests.get(
                f"{self.api_base}/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False

