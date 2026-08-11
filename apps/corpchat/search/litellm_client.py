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

# 进程级 token 用量累计 (成本/用量观测, 供 eval 与监控使用)。
# 每次成功的 chat 响应把 usage 累加进来; reset_usage() 可清零。
_USAGE_TOTAL: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def reset_usage() -> None:
    """清零进程级 token 用量累计。"""
    _USAGE_TOTAL["prompt_tokens"] = 0
    _USAGE_TOTAL["completion_tokens"] = 0
    _USAGE_TOTAL["calls"] = 0


def usage_total() -> Dict[str, int]:
    """返回进程级累计用量 (副本)。"""
    return dict(_USAGE_TOTAL)


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
        # 单实例用量 (与进程级累计并存)
        self.usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def _record_usage(self, usage: Optional[Dict]) -> None:
        """把一次响应的 usage 计入实例与进程级累计。"""
        if not usage:
            return
        try:
            p = int(usage.get("prompt_tokens", 0) or 0)
            c = int(usage.get("completion_tokens", 0) or 0)
        except (TypeError, ValueError):
            return
        self.usage["prompt_tokens"] += p
        self.usage["completion_tokens"] += c
        self.usage["calls"] += 1
        _USAGE_TOTAL["prompt_tokens"] += p
        _USAGE_TOTAL["completion_tokens"] += c
        _USAGE_TOTAL["calls"] += 1

    def chat(self, messages: List[Dict], temperature: float = 0.1,
             max_tokens: int = 200, timeout: int = 15) -> str:
        """
        Send a chat completion request and return the assistant's content.

        Tries the primary OpenAI-compatible endpoint first, then falls back to
        Ollama's native `/api/chat`, then finally to DeepSeek.
        Returns empty string on any failure (graceful degradation).
        """
        result = self.chat_message(messages, tools=None, temperature=temperature,
                                   max_tokens=max_tokens, timeout=timeout)
        return (result or {}).get("content") or ""

    def chat_message(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
                     temperature: float = 0.1, max_tokens: int = 200,
                     timeout: int = 15) -> Optional[Dict]:
        """Send a chat completion and return the full assistant message dict.

        Unlike chat(), this returns the raw message object so the caller can
        inspect `tool_calls` (for native tool-calling agents). Falls back to
        DeepSeek when the primary endpoint fails. Returns None on failure.

        Returns: message dict with keys "content" (str|None) and optionally
        "tool_calls" (list) when the model requested tool calls.
        """
        # 1) Try OpenAI-compatible endpoint (primary; DeepSeek by default)
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                self._record_usage(data.get("usage"))
                return {
                    "content": msg.get("content"),
                    "tool_calls": msg.get("tool_calls"),
                }
            if resp.status_code != 404:
                logger.warning(f"LLM 调用失败 ({resp.status_code}): {resp.text[:200]}")
                # Non-404 error → try DeepSeek fallback
                return self._deepseek_chat_message(messages, tools, temperature, max_tokens, timeout)
        except requests.exceptions.RequestException as e:
            logger.warning(f"LLM 调用异常: {e}")
            # Network error → try DeepSeek fallback
            return self._deepseek_chat_message(messages, tools, temperature, max_tokens, timeout)

        # 2) Fallback: Ollama native /api/chat (no tools support)
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
            return {"content": resp.json().get("message", {}).get("content", "").strip(),
                    "tool_calls": None}
        except Exception as e:
            logger.warning(f"LLM 调用失败 (Ollama fallback): {e}")
            # Ollama also failed → try DeepSeek fallback
            return self._deepseek_chat_message(messages, tools, temperature, max_tokens, timeout)

    def _deepseek_chat_message(self, messages: List[Dict], tools: Optional[List[Dict]],
                               temperature: float, max_tokens: int, timeout: int) -> Optional[Dict]:
        """Call DeepSeek's OpenAI-compatible endpoint; return message dict or None."""
        if not DEEPSEEK_API_KEY:
            logger.warning("DeepSeek fallback skipped: DEEPSEEK_API_KEY not set")
            return None
        try:
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            resp = requests.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                logger.info("DeepSeek fallback LLM used successfully")
                self._record_usage(data.get("usage"))
                return {
                    "content": msg.get("content"),
                    "tool_calls": msg.get("tool_calls"),
                }
            logger.warning(f"DeepSeek fallback failed ({resp.status_code}): {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"DeepSeek fallback exception: {e}")
        return None

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

