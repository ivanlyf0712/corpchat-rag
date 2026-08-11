"""CorpChat Search — Intent word tables & greeting generation (single source).

Candidate 4: the greeting/system/clarify keyword tables and the LLM greeting
generator previously existed in up to three copies (legacy `agent.py`,
`cross_table_agent.py`, inline in `app.py`) with diverging content. This module
is the one place they live; both agents re-export/import from here so intent
routing can never drift between the legacy and cross-table pipelines.
"""

import re
from typing import Optional

# ── Greeting keywords (union of legacy agent.py + cross_table_agent.py) ──
GREETING_KEYWORDS = (
    "hi", "hello", "hey", "hiya", "howdy", "greetings", "yo",
    "how are you", "how's it going", "how are you doing", "how do you do",
    "what's up", "whats up", "good morning", "good afternoon", "good evening",
    "long time no see", "nice to meet you", "good to see you",
    "嗨", "你好", "哈囉", "好嗎", "早安", "午安", "晚安",
    "久久", "怎麼樣", "最近怎么样", "最近怎樣", "最近怎麼樣",
    "吃了嗎", "吃了吗",
)

# ── System-info keywords (what can you do / who are you) ───────────
SYSTEM_KEYWORDS = (
    "你是誰", "你是谁", "what is your name", "who are you", "叫什麼名字",
    "what can you do", "能做什麼", "能做什么", "能做", "can you help",
    "功能", "能力", "作用", "什麼功能", "多少功能", "help", "幫助",
    "使用說明", "搜索範圍", "scope", "what can you search", "can you search",
    "你能搜尋", "搜什麼", "資料範圍", "what can you access", "can you access",
    "access", "知道些什麼", "知道什么", "了解什麼", "了解什么",
    "你会什么", "你會什麼", "你会做什么", "你會做什麼",
)

# ── Clarify keywords (rephrase / explain more) ─────────────────────
CLARIFY_KEYWORDS = (
    "能再說", "再說一遍", "不是很懂", "不太清楚", "clarify", "explain more",
    "詳細一些", "細節", "what do you mean", "什麼意思", "不太明白",
    "再解釋", "不太理解", "看不懂", "具體點",
)

_LANG_NAMES = {
    "en": "English",
    "zh-TW": "Traditional Chinese",
    "zh-CN": "Simplified Chinese",
}


def is_greeting_query(q: str) -> bool:
    """Greeting detection shared by the tool router and the quick-respond gate.

    Single tokens match whole-word (so "hi" never fires inside "which"/"this"),
    multi-word phrases match as substrings, and CJK phrases match on their own
    boundaries. Kept conservative: courteous-prefix greetings ("您好", "在嗎")
    are deliberately excluded — the LLM classify path covers those, and a
    greeting word must never swallow a search request.
    """
    if len(q) >= 20:
        return False
    for g in GREETING_KEYWORDS:
        if " " in g or "'" in g or "-" in g:
            if g in q:
                return True
        elif re.search(rf"(^|[^a-z]){re.escape(g)}([^a-z]|$)", q):
            return True
    return False


def is_system_query(q: str) -> bool:
    """System-capability query detection (substring)."""
    return any(kw in q for kw in SYSTEM_KEYWORDS)


def is_clarify_query(q: str) -> bool:
    """Clarify intent detection (substring)."""
    return any(kw in q for kw in CLARIFY_KEYWORDS)


def generate_greeting(client, user_input: str, lang: str = "en",
                      fallback: str = "") -> str:
    """LLM-generated greeting reply with preset fallback (shared by both agents).

    Args:
        client: the LiteLLM client to call (each agent injects its own).
        user_input: the user's greeting text.
        lang: "en" | "zh-TW" | "zh-CN" (drives the reply language).
        fallback: preset reply used when the LLM is unavailable.
    """
    lang_name = _LANG_NAMES.get(lang, "English")
    try:
        result = client.chat(
            [
                {"role": "system", "content": (
                    f"You are CorpChat Intelligence, a friendly enterprise chat-search "
                    f"assistant. Reply to the user's greeting naturally in {lang_name}. "
                    f"Keep it short, warm, and context-aware. Do NOT mention that you are "
                    f"an AI or list capabilities. Briefly invite them to search chat "
                    f"messages or contacts if they need help."
                )},
                {"role": "user", "content": user_input or "Hi!"},
            ],
            temperature=0.7,
            max_tokens=60,
            timeout=5,
        )
        return result or fallback
    except Exception:
        return fallback
