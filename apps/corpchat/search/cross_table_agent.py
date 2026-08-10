"""
CorpChat Search — Cross-Table Agent (LangChain ReAct)
=======================================================
LangChain ReAct Agent with LiteLLM as the LLM backend.
Autonomous tool routing across messages and contacts indices.

Architecture:
  User Input → Agent (ReAct) → Tool Router → [search_messages | search_contacts]
                                     ↓
                              Result Integration → Natural Language Answer

Fallback: If LangChain is unavailable, falls back to the original Agent pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LITELLM_MODEL,
    logger,
)
from .persona import DispositionProfile

# ── LangChain imports ────────────────────────────────────────────
_LANGCHAIN_AVAILABLE = False
try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langgraph.prebuilt import create_react_agent
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available — cross-table agent will use fallback mode")


# ── Intent keyword tables (rule fast-path & LLM-down fallback) ────
_GREETING_KEYWORDS = (
    "hi", "hello", "hey", "hiya", "howdy", "greetings",
    "how are you", "how's it going", "how are you doing", "how do you do",
    "what's up", "whats up", "good morning", "good afternoon", "good evening",
    "long time no see", "nice to meet you", "good to see you",
    "嗨", "你好", "哈囉", "早安", "午安", "晚安",
    "最近怎么样", "最近怎樣", "吃了嗎", "吃了吗",
)

_SYSTEM_KEYWORDS = (
    "你是誰", "你是谁", "what can you do", "能做什麼", "能做什么",
    "能做", "功能", "能力", "help", "使用說明", "搜索範圍",
    "你会什么", "你會什麼", "你会做什么", "你會做什麼",
)

_CONTACT_KEYWORDS = (
    "邮箱", "郵箱", "email", "电话", "電話", "phone", "公司", "职位", "職位",
    "联系方式", "聯繫方式", "userid", "who is", "who are", "是谁", "是誰",
    "contact", "contacts", "联系人", "聯絡人", "名字", "姓名", "name",
    "male", "female", "男生", "女生", "先生", "女士",
)

_MESSAGE_KEYWORDS = (
    "消息", "说了", "說了", "说", "說", "聊", "对话", "對話", "聊天",
    "spoke", "talked", "said", "sent", "message", "conversation",
    "发", "發", "收到", "内容", "內容", "what did", "who did",
)

_CROSS_TABLE_TERMS = (
    "spoke to", "talked to", "跟谁", "和谁", "跟誰", "和誰",
    "发", "發", "消息的人", "訊息的人",
)

# Per-(api_base, api_key) LLM availability cache — avoids a /v1/models
# probe on every query (app.py constructs a fresh CrossTableAgent per request).
_LLM_AVAILABILITY_CACHE: Dict[Tuple[str, str], bool] = {}


def _is_greeting_query(q: str) -> bool:
    """Greeting detection shared by the tool router and the quick-respond gate.

    Single tokens match whole-word (so "hi" never fires inside "which"/"this"),
    multi-word phrases match as substrings, and CJK phrases match on their own
    boundaries. Kept conservative: courteous-prefix greetings ("您好", "在嗎")
    are deliberately excluded — the LLM classify path covers those, and a
    greeting word must never swallow a search request.
    """
    if len(q) >= 20:
        return False
    for g in _GREETING_KEYWORDS:
        if " " in g or "'" in g or "-" in g:
            if g in q:
                return True
        elif re.search(rf"(^|[^a-z]){re.escape(g)}([^a-z]|$)", q):
            return True
    return False


class _LiteLLMWrapper(BaseChatModel):
    """Minimal BaseChatModel wrapper around our existing LiteLLMClient."""
    
    def __init__(self, api_base: str, api_key: str, model: str, **kwargs):
        super().__init__(**kwargs)
        self._api_base = api_base
        self._api_key = api_key
        self._model = model
        self._client = None
        self._bound_tools = []
    
    def _get_client(self):
        if self._client is None:
            from .litellm_client import LiteLLMClient
            self._client = LiteLLMClient(api_base=self._api_base, api_key=self._api_key, model=self._model)
        return self._client
    
    def bind_tools(self, tools, **kwargs):
        """Required by LangGraph for tool-calling models."""
        self._bound_tools = list(tools)
        return self
    
    def _decide_tool_calls(self, user_input: str) -> List[Dict]:
        """
        Decide which tools to call based on the user query.
        Returns a list of tool-call dicts in LangChain format:
          [{"name": ..., "args": {...}, "id": ...}]
        """
        q = user_input.lower().strip()

        # Contact-detail questions → search_contacts
        wants_contact = any(kw in q for kw in _CONTACT_KEYWORDS)
        # Message questions → search_messages
        wants_message = any(kw in q for kw in _MESSAGE_KEYWORDS)

        # Cross-table: "who did X spoke to" / "发'X'消息的人" → both
        cross_table = any(t in q for t in _CROSS_TABLE_TERMS)

        # 结构化过滤查询 (含链接/URL/标签等精确条件) → SQL 精确检索 (不走语义向量)
        struct_kws = ("link", "url", "http", "www", "網址", "網址", "链接", "連結", "网站", "網站")
        msg_ctx = ("message" in q or "messages" in q or "msg" in q or "消息" in q or "訊息" in q
                   or "sent" in q or "contain" in q or "含" in q or "有" in q
                   or "找" in q or "search" in q or "查" in q or "发" in q or "發" in q)
        if any(kw in q for kw in struct_kws) and msg_ctx:
            return [{
                "name": "search_messages_where",
                "args": {"condition": user_input},
                "id": "call_struct_1",
            }]

        # Greeting / system questions → no tools.
        # Checked AFTER search keywords so an explicit search request wins
        # over a courtesy greeting word embedded in it.
        if not (wants_contact or wants_message or cross_table) and _is_greeting_query(q):
            return []
        if not (wants_contact or wants_message or cross_table) and any(kw in q for kw in _SYSTEM_KEYWORDS):
            return []

        tool_calls = []
        if wants_message or cross_table:
            tool_calls.append({
                "name": "search_messages",
                "args": {"query": user_input},
                "id": "call_msg_1",
            })
        if wants_contact or cross_table:
            tool_calls.append({
                "name": "search_contacts",
                "args": {"query": user_input},
                "id": "call_contact_1",
            })

        # Default: if nothing matched, search messages (most common intent)
        if not tool_calls:
            tool_calls.append({
                "name": "search_messages",
                "args": {"query": user_input},
                "id": "call_msg_1",
            })
        return tool_calls

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Call LiteLLM and return ChatResult.

        For the first turn (user query), we emit tool calls so LangGraph
        actually executes the search tools. On subsequent turns (after tools
        have run), we call the LLM to synthesize the final answer.
        """
        client = self._get_client()
        # Convert LangChain messages to dicts
        raw_messages = []
        has_tool_result = False
        user_query = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                raw_messages.append({"role": "user", "content": msg.content})
                if not user_query:
                    user_query = str(msg.content)
            elif isinstance(msg, SystemMessage):
                raw_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, AIMessage):
                raw_messages.append({"role": "assistant", "content": msg.content})
            elif hasattr(msg, "type") and msg.type == "tool":
                has_tool_result = True
                raw_messages.append({"role": "user", "content": f"Tool result: {msg.content}"})
            else:
                raw_messages.append({"role": "user", "content": str(msg.content)})

        # If tools haven't run yet, emit tool calls so the agent executes them
        if not has_tool_result and user_query:
            tool_calls = self._decide_tool_calls(user_query)
            if tool_calls:
                message = AIMessage(content="", tool_calls=tool_calls)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])

        # Tools have run (or no tools needed) → synthesize final answer with LLM
        content = client.chat(raw_messages, temperature=kwargs.get("temperature", 0.1),
                             max_tokens=kwargs.get("max_tokens", 2048), timeout=30)

        message = AIMessage(content=content or "")
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    
    @property
    def _llm_type(self) -> str:
        return "litellm-wrapper"
    
    @property
    def _identifying_params(self):
        return {"model": self._model}

# ── System prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个企业微信聊天记录智能搜索助手，可以跨数据源检索信息。

你有以下工具可用：

1. **search_messages(query)** — 搜索内部聊天消息
   - 用于查找用户对话、消息内容、聊天记录
   - 返回消息内容、发送者姓名、userid、标签等
   - 示例查询: "合同已签", "诈骗链接", "物流报价"

2. **search_contacts(query)** — 搜索联系人信息
   - 用于查找联系人邮箱、电话、公司、职位
   - 返回全名、userid、邮箱、公司、电话、职位
   - 示例查询: "李雅婷", "陳志明 email", "johnsonj", "user_陳志明"

工作流程:
1. 分析用户问题，判断需要哪些数据
2. 如果需要查消息 → 调用 search_messages
3. 如果需要查联系人 → 调用 search_contacts
4. 如果需要结合多个来源 → 先查消息获取 userid，再用 userid 查联系人
5. 用中文整合所有结果，给出清晰的自然语言答案

关键规则:
- 如果问题是闲聊、问候、系统能力询问 → 直接回答，无需调用工具
- 如果问题需要跨表查询（如"发'合同已签'消息的人，他的邮箱是什么？"）→
  先调用 search_messages 查消息 → 从结果提取 userid → 再用 userid 调用 search_contacts 查联系人
- 用中文回答，保持简洁准确
- 如果搜索结果为空，如实告知用户
"""


class CrossTableAgent:
    """
    LangChain ReAct Agent that can autonomously route queries across
    messages and contacts data sources.

    Features:
      - Autonomous tool selection and chaining
      - Two-step reasoning (messages → extract userid → contacts)
      - Graceful degradation to fallback on LangChain failure
      - Streaming thought process for debugging
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = LITELLM_MODEL,
        expand: bool = False,
        use_rerank: bool = False,
        graph_parallel: bool = False,
        profile: Optional[DispositionProfile] = None,
        sources: Optional[List[str]] = None,
    ):
        self.api_base = api_base or LITELLM_BASE_URL
        self.api_key = api_key or LITELLM_API_KEY
        self.model = model
        self.expand = expand
        self.use_rerank = use_rerank
        self.graph_parallel = graph_parallel
        self.profile = profile  # DispositionProfile (persona), optional
        # 数据源门控: 子集 of {"messages", "contacts"}; None = 全部启用
        self.sources = sources
        self._agent: Optional[Any] = None
        self._last_thoughts: List[str] = []
        self._last_tool_calls: List[Dict[str, Any]] = []
        self._steps: List[Dict[str, Any]] = []

    def _get_llm(self):
        """Create a LangChain-compatible LLM wrapper around LiteLLMClient."""
        return _LiteLLMWrapper(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
        )

    def _init_agent(self):
        """Lazy-initialize the LangGraph ReAct agent."""
        if self._agent is not None:
            return

        if not _LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain is not installed. Run: pip install langchain langchain-community")

        from .tools import search_messages, search_contacts, search_messages_where

        model = self._get_llm()

        self._agent = create_react_agent(
            model,
            tools=[search_messages, search_contacts, search_messages_where],
            prompt=SYSTEM_PROMPT,
        )

    def _add_step(self, icon: str, label: str, duration_ms: int, detail: str = "") -> None:
        """Record a process step for the timeline."""
        self._steps.append({
            "icon": icon,
            "label": label,
            "duration_ms": duration_ms,
            "detail": detail,
        })

    @staticmethod
    def _detect_language(user_input: str) -> str:
        """
        Detect the input language.
        Returns: "en", "zh-TW", or "zh-CN".
        English is the default fallback.
        """
        # Traditional Chinese characters (distinctive)
        trad_chars = "維認體機關係臺灣門東車馬為與從來個們說時後這那裡會對過還進動開點學問發約訊郵簽話電號碼員務"
        # Simplified Chinese characters (distinctive)
        simp_chars = "维认体机关系台湾门东车马为与从来个们说时候这里会对过还进动开点学问发约讯邮签话电号码员务"


        has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in user_input)
        if not has_cjk:
            return "en"

        # Mixed-language queries (e.g. "who did 何建明 spoke to?") are
        # predominantly English with an embedded Chinese name. If Latin
        # letters outnumber CJK characters, treat the query as English so
        # the answer language matches the bulk of the question.
        latin_count = sum(1 for ch in user_input if 'a' <= ch.lower() <= 'z')
        cjk_count = sum(1 for ch in user_input if '\u4e00' <= ch <= '\u9fff')
        if latin_count > cjk_count:
            return "en"

        trad_count = sum(1 for ch in user_input if ch in trad_chars)
        simp_count = sum(1 for ch in user_input if ch in simp_chars)

        if trad_count > simp_count:
            return "zh-TW"
        return "zh-CN"


    def process(self, user_input: str, on_stage: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process a user query through the cross-table agent.

        Uses a manual ReAct loop: decide which tools to call → execute them →
        synthesize the final answer with the LLM. This is more reliable than
        relying on LangGraph's tool-calling protocol, which requires the model
        to emit structured tool calls.

        Args:
            user_input: The user's query string.
            on_stage: Optional callback `on_stage(label: str, detail: str = "")`
                invoked as each processing stage starts. The UI uses it to
                drive the fade-in/out stage animation.

        Returns:
            Dict with keys:
              - output: str — the final answer
              - thoughts: List[str] — agent reasoning steps (for debugging)
              - tool_calls: List[Dict] — tools that were called
              - steps: List[Dict] — process timeline (icon, label, duration_ms, detail)
              - success: bool — whether the agent succeeded
              - fallback: bool — whether fallback was used
        """
        import time as _time

        def _stage(label: str, detail: str = ""):
            if on_stage:
                try:
                    on_stage(label, detail)
                except Exception:
                    pass

        self._last_thoughts = []
        self._last_tool_calls = []
        self._steps = []
        _start = _time.perf_counter()

        # ── Step 1: Quick check — is this a greeting or system question? ──
        _stage("🧠", "routing...")
        quick_result = self._quick_respond(user_input)
        if quick_result:
            self._add_step("⚡", "Intent check", 0, "Greeting/system → quick response")
            return {
                "output": quick_result,
                "thoughts": ["Quick response: no tools needed"],
                "tool_calls": [],
                "raw_hits": [],
                "steps": self._steps,
                "success": True,
                "fallback": False,
            }

        # ── Step 2: Manual ReAct loop (reliable, no LangGraph dependency) ──
        try:
            from .tools import search_messages, search_contacts, search_messages_where

            # Decide which tools to call based on the query
            wrapper = _LiteLLMWrapper(
                api_base=self.api_base,
                api_key=self.api_key,
                model=self.model,
            )
            tool_calls = wrapper._decide_tool_calls(user_input)
            # 数据源门控: 按 knowledge.sources 过滤工具
            if self.sources:
                tool_calls = [
                    tc for tc in tool_calls
                    if (tc["name"] == "search_messages" and "messages" in self.sources)
                    or (tc["name"] == "search_messages_where" and "messages" in self.sources)
                    or (tc["name"] == "search_contacts" and "contacts" in self.sources)
                ]
            self._add_step("🧠", "Agent routing", 0, f"Decided {len(tool_calls)} tool call(s)")

            # Extract a clean search query for better tool results
            search_query = self._extract_search_query(user_input)

            # Execute tools
            msg_result = ""
            contact_result = ""
            executed_calls = []
            for tc in tool_calls:
                name = tc["name"]
                _t0 = _time.perf_counter()
                actual_query = search_query
                if name == "search_messages":
                    _stage("🔍", f"using search_messages... query: {search_query}")
                    msg_result = search_messages.invoke(
                        {"query": search_query, "expand": self.expand, "use_rerank": self.use_rerank,
                         "graph_parallel": self.graph_parallel}
                    )
                    # If no results with the extracted query, retry with the original query
                    if self._is_empty_result(msg_result) and search_query != user_input:
                        msg_result = search_messages.invoke(
                            {"query": user_input, "expand": self.expand, "use_rerank": self.use_rerank,
                             "graph_parallel": self.graph_parallel}
                        )
                        actual_query = user_input
                    _t1 = _time.perf_counter()
                    self._add_step("🔍", "search_messages", int((_t1 - _t0) * 1000), f"Query: '{actual_query}'")
                    from .tools import get_last_search_meta
                    meta = get_last_search_meta()
                elif name == "search_messages_where":
                    _stage("🗄️", f"using search_messages_where... condition: {search_query}")
                    msg_result = search_messages_where.invoke({"condition": search_query})
                    _t1 = _time.perf_counter()
                    self._add_step("🗄️", "search_messages_where", int((_t1 - _t0) * 1000),
                                   f"Condition: '{search_query}'")
                    from .tools import get_last_search_meta
                    meta = get_last_search_meta()
                elif name == "search_contacts":
                    _stage("👤", f"using search_contacts... query: {search_query}")
                    contact_result = search_contacts.invoke({"query": search_query})
                    if self._is_empty_result(contact_result) and search_query != user_input:
                        contact_result = search_contacts.invoke({"query": user_input})
                        actual_query = user_input
                    _t1 = _time.perf_counter()
                    self._add_step("👤", "search_contacts", int((_t1 - _t0) * 1000), f"Query: '{actual_query}'")
                    from .tools import get_last_contact_meta
                    meta = get_last_contact_meta()
                executed_calls.append({
                    "tool": name,
                    "tool_input": actual_query,
                    "observation": (msg_result if name in ("search_messages", "search_messages_where") else contact_result)[:200],
                    "meta": meta,
                })



            self._last_tool_calls = executed_calls

            # ── Step 3: Synthesize answer ──
            _stage("✨", "generating answer...")
            self._add_step("✨", "Answer generation", 0, "Combining results")
            # 联系人结构化查询 → 确定性格式化 (小模型易幻觉, 不冒 LLM 总结风险)
            only_contacts = bool(executed_calls) and all(
                tc.get("tool") == "search_contacts" for tc in executed_calls)
            try:
                if only_contacts:
                    output = self._format_fallback_answer(user_input, msg_result, contact_result)
                else:
                    output = self._llm_summarize(user_input, msg_result, contact_result)
            except Exception:
                output = self._format_fallback_answer(user_input, msg_result, contact_result)

            # 汇总 search_messages / search_messages_where 的原始结果 (含 metadata) — 供记忆图谱使用
            graph_raw_hits = []
            for tc in executed_calls:
                if tc.get("tool") in ("search_messages", "search_messages_where"):
                    graph_raw_hits.extend(tc.get("meta", {}).get("raw_hits", []))

            return {
                "output": output,
                "thoughts": [f"Agent routed to {len(tool_calls)} tool(s)"],
                "tool_calls": executed_calls,
                "raw_hits": graph_raw_hits,
                "steps": self._steps,
                "success": True,
                "fallback": False,
            }

        except Exception as e:
            logger.warning(f"Cross-table agent failed: {e}")
            self._add_step("⚠️", "Agent error", 0, str(e)[:100])
            # ── Step 4: Fallback — use two-step reasoning ──
            return self._fallback_process(user_input, error=str(e))


    def _quick_respond(self, user_input: str) -> Optional[str]:
        """Handle greetings and system questions without invoking tools.

        Rule-first fast path (keyword gates are <1ms):
          greeting → LLM-generated greeting response (or preset if LLM down)
          system   → self-description
          search   → proceed to tool routing

        The LLM intent classification runs ONLY for ambiguous queries that
        no rule matched — known greetings/system and obvious searches never
        pay an extra LLM round-trip (keeps search latency low).
        """
        q = user_input.lower().strip()
        lang = self._detect_language(user_input)
        llm_ok = self._check_llm()

        # Fast keyword gates FIRST — known greetings never pay an LLM classify
        if _is_greeting_query(q):
            if llm_ok:
                return self._generate_greeting(user_input, lang)
            return self._PRESET_GREETING_EN if lang == "en" else self._PRESET_GREETING_ZH
        if any(kw in q for kw in _SYSTEM_KEYWORDS):
            return self._system_info_text(lang)

        # Clearly a search query → proceed to tool routing (no LLM classify)
        if self._rule_search_hint(q):
            return None

        # Ambiguous query → LLM intent classification (bounded timeout)
        if llm_ok:
            intent = self._llm_classify_intent(user_input)
            if intent == "greeting":
                return self._generate_greeting(user_input, lang)
            if intent == "system":
                return self._system_info_text(lang)
            # "search" or None → proceed to search

        return None

    # -- Preset replies -- used ONLY when the LLM is unavailable -----
    _PRESET_GREETING_EN = (
        "Hello! I'm CorpChat Intelligence. I can search chat messages and "
        "contact info. How can I help?"
    )
    _PRESET_GREETING_ZH = (
        "你好！我是 CorpChat 智能搜索助手。我可以帮你搜索聊天记录和联系人信息。"
        "请问有什么可以帮你的？"
    )

    @staticmethod
    def _rule_search_hint(q: str) -> bool:
        """Cheap 'clearly a search query' hint - skips the LLM classify round-trip."""
        return (
            any(kw in q for kw in _CONTACT_KEYWORDS)
            or any(kw in q for kw in _MESSAGE_KEYWORDS)
            or any(t in q for t in _CROSS_TABLE_TERMS)
        )

    def _check_llm(self) -> bool:
        """Whether the LLM endpoint is reachable (cached per base+key)."""
        if not self.api_key:
            return False
        cache_key = (self.api_base, self.api_key)
        if cache_key in _LLM_AVAILABILITY_CACHE:
            return _LLM_AVAILABILITY_CACHE[cache_key]
        try:
            from .litellm_client import LiteLLMClient
            ok = bool(
                LiteLLMClient(api_base=self.api_base, api_key=self.api_key).is_available(timeout=3)
            )
        except Exception:
            ok = False
        _LLM_AVAILABILITY_CACHE[cache_key] = ok
        return ok

    def _llm_classify_intent(self, user_input: str) -> Optional[str]:
        """Classify intent with the LLM: 'greeting' | 'system' | 'search'.

        Returns None when the LLM is unavailable or returns something
        unrecognized, so the caller can fall back to keyword rules.
        """
        try:
            from .litellm_client import LiteLLMClient
            client = LiteLLMClient(api_base=self.api_base, api_key=self.api_key)
            result = client.chat(
                [
                    {"role": "system", "content": (
                        "You are an intent classifier for a chat-search assistant. "
                        "Classify the user's message into exactly one category:\n"
                        "- greeting: casual greeting or small talk "
                        "(hi, hello, how are you, good morning, 你好, 最近怎么样)\n"
                        "- system: asking about the assistant's capabilities or identity "
                        "(who are you, what can you do, 你会什么)\n"
                        "- search: anything requesting information from chat messages or contacts\n"
                        "Reply with ONLY the category name: greeting, system, or search."
                    )},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.0,
                max_tokens=8,
                timeout=3,
            )
            result = (result or "").strip().lower()
            for intent in ("greeting", "system", "search"):
                if intent in result:
                    return intent
        except Exception as e:
            logger.debug(f"LLM intent classification failed: {e}")
        return None

    def _generate_greeting(self, user_input: str, lang: str) -> str:
        """LLM-generated greeting response; preset fallback when LLM is down."""
        try:
            from .litellm_client import LiteLLMClient
            client = LiteLLMClient(api_base=self.api_base, api_key=self.api_key)
            lang_name = {
                "en": "English",
                "zh-TW": "Traditional Chinese",
                "zh-CN": "Simplified Chinese",
            }.get(lang, "English")
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
                max_tokens=80,
                timeout=5,
            )
            if result and result.strip():
                return result.strip()
        except Exception as e:
            logger.debug(f"LLM greeting generation failed: {e}")
        return self._PRESET_GREETING_EN if lang == "en" else self._PRESET_GREETING_ZH

    def _system_info_text(self, lang: str) -> str:
        """Self-description answer. Kept static - only greetings are LLM-generated."""
        if lang == "en":
            return (
                "I'm **CorpChat Intelligence**, a cross-source search assistant.\n\n"
                "**Capabilities:**\n"
                "- 🔍 **Search chat messages** — find conversations, content, labels\n"
                "- 👤 **Search contacts** — find email, phone, company, title\n"
                "- 🔗 **Cross-table reasoning** — find sender from message, then contact details\n\n"
                "**Examples:**\n"
                "- \"What is 李雅婷's email?\"\n"
                "- \"Who sent the '合同已签' message and what's their email?\"\n"
                "- \"Any new messages today?\"\n"
                "- \"Find messages about fraud\"\n"
            )
        return (
            "我是 **CorpChat 智能搜索助手**，可以跨数据源检索信息。\n\n"
            "**可用能力：**\n"
            "- 🔍 **搜索聊天消息** — 查找对话、消息内容、标签等\n"
            "- 👤 **搜索联系人** — 查找邮箱、电话、公司、职位\n"
            "- 🔗 **跨表推理** — 先查消息找到发送者，再查联系人获取详细信息\n\n"
            "**示例：**\n"
            "- \"李雅婷的邮箱是什么？\"\n"
            "- \"发'合同已签'消息的人，他的邮箱是什么？\"\n"
            "- \"今天有什么新消息？\"\n"
            "- \"帮我查一下诈骗相关的消息\"\n"
        )

    def _fallback_process(self, user_input: str, error: str = "") -> Dict[str, Any]:
        """
        Fallback when LangChain agent fails.
        Uses TWO-STEP reasoning: extract query → search messages → extract userid → search contacts.
        """
        import time as _time
        logger.info("Using fallback mode for cross-table agent")

        from .tools import search_messages, search_contacts

        # ── Step 1: Extract the actual search query from the user's question ──
        search_query = self._extract_search_query(user_input)
        self._add_step("🔍", "Query extraction", 0, f"Extracted: '{search_query}'")

        # ── Step 2: Try two-step reasoning (messages → contacts) ──
        msg_result = ""
        contact_result = ""
        tool_calls = []

        try:
            # First: search messages with the extracted query
            _t0 = _time.perf_counter()
            msg_result = search_messages.invoke(
                {"query": search_query, "expand": self.expand, "use_rerank": self.use_rerank,
                 "graph_parallel": self.graph_parallel}
            )
            _t1 = _time.perf_counter()
            self._add_step("🔍", "search_messages", int((_t1 - _t0) * 1000), f"Query: '{search_query}'")
            from .tools import get_last_search_meta
            tool_calls.append({
                "tool": "search_messages",
                "tool_input": search_query,
                "observation": msg_result[:200],
                "meta": get_last_search_meta(),
            })

            # Second: extract userid from message results and search contacts
            userid = self._extract_userid_from_result(msg_result)
            if userid:
                # Search contacts by the exact userid
                _t0 = _time.perf_counter()
                contact_result = search_contacts.invoke({"query": userid})
                _t1 = _time.perf_counter()
                self._add_step("👤", "search_contacts", int((_t1 - _t0) * 1000), f"Query: '{userid}'")
                from .tools import get_last_contact_meta
                tool_calls.append({
                    "tool": "search_contacts",
                    "tool_input": userid,
                    "observation": contact_result[:200],
                    "meta": get_last_contact_meta(),
                })
            else:
                # If no userid found in messages, also try searching contacts directly
                name = self._extract_name_from_question(user_input)
                if name:
                    contact_result = search_contacts.invoke({"query": name})
                else:
                    contact_result = search_contacts.invoke({"query": search_query})
                self._add_step("👤", "search_contacts", 0, f"Query: '{name or search_query}'")
                from .tools import get_last_contact_meta
                tool_calls.append({
                    "tool": "search_contacts",
                    "tool_input": search_query,
                    "observation": contact_result[:200],
                    "meta": get_last_contact_meta(),
                })
        except Exception as e:
            logger.warning(f"Two-step fallback failed: {e}")
            # Ultra-fallback: try both tools with original query
            try:
                if not msg_result:
                    msg_result = search_messages.invoke({"query": search_query})
                if not contact_result:
                    contact_result = search_contacts.invoke({"query": search_query})
            except Exception:
                pass

        # ── Step 3: Try LLM summarization ──
        try:
            summary = self._llm_summarize(user_input, msg_result, contact_result)
        except Exception:
            summary = self._format_fallback_answer(user_input, msg_result, contact_result)

        self._add_step("✨", "Answer generation", 0, "Combined results into answer")

        # 汇总 search_messages 的原始结果 (含 metadata) — 供记忆图谱使用
        fallback_raw_hits = []
        for tc in tool_calls:
            if tc.get("tool") == "search_messages":
                fallback_raw_hits.extend(tc.get("meta", {}).get("raw_hits", []))

        return {
            "output": summary,
            "thoughts": [
                f"LangChain agent error: {error[:100] if error else 'unknown'}",
                f"Fallback: extracted query='{search_query}', userid='{userid if 'userid' in dir() else 'N/A'}'",
            ],
            "tool_calls": tool_calls,
            "raw_hits": fallback_raw_hits,
            "steps": self._steps,
            "success": True,
            "fallback": True,
        }

    @staticmethod
    def _extract_search_query(user_input: str) -> str:
        """
        Extract the actual search key from a user question.
        E.g. "发'合同已签'消息的人，他的邮箱是什么？" → "合同已签"
             "李雅婷的邮箱是什么？" → "李雅婷"
        """
        # Remove common question words/phrases
        q = user_input
        # Try to extract quoted text first (most reliable)
        import re
        quoted = re.findall(r"['\"'\"''](.+?)['\"'\"'']", q)
        if quoted:
            return quoted[0]

        # Remove question suffixes
        for suffix in ["的邮箱是什么", "的邮箱是", "是谁", "是什么", "他的邮箱是什么",
                       "她的邮箱是什么", "邮箱是什么", "的电话是什么", "的电话是",
                       "联系方式是什么", "的公司是什么", "的职位是什么"]:
            q = q.replace(suffix, "")

        # Remove leading context words
        for prefix in ["发", "找", "查", "搜索", "帮我查", "帮我找", "请问",
                       "发'", "找'", "查'"]:
            if q.startswith(prefix):
                q = q[len(prefix):]
                break

        # Remove filler words that add noise to the search query
        for filler in ["一下", "相关的", "有关", "关于", "关于的", "的消息", "的訊息",
                       "帮我", "请", "请问", "查一下", "找一下", "搜索一下"]:
            q = q.replace(filler, "")

        # Strip English conversational noise / instructions (e.g. "try again. Who is 李雅婷")
        for pat in (r"^try again[\.!\s]*", r"^please[,\s]+", r"^find (me |out )?",
                    r"^(who is|who are|what is|what are|tell me about|give me|show me)\s+",
                    r"^i (want|need) to (know|find|search)\s+",
                    r"^can you (tell me|find|search|look up)\s+",
                    r"^do you know\s+", r"^search for\s+", r"^look (for|up)\s+"):
            q = re.sub(pat, "", q, flags=re.IGNORECASE)
        q = re.sub(r"\s*(please|thanks|thank you)[.!]*\s*$", "", q, flags=re.IGNORECASE)
        # Remove trailing punctuation
        q = q.strip("'\"'?？，,。.!！")
        return q.strip() or user_input[:30]


    @staticmethod
    def _is_empty_result(result: str) -> bool:
        """Check if a tool result is empty or indicates no results found."""
        if not result or not result.strip():
            return True
        empty_markers = ["没有找到", "未找到", "没有相关", "no results", "no relevant",
                         "not found", "无法找到", "找不到", "没有关于", "No messages found",
                         "No contacts found", "没有消息", "没有联系人"]
        return any(marker in result for marker in empty_markers)

    @staticmethod
    def _extract_userid_from_result(result: str) -> Optional[str]:

        """Extract userid from tool result text. E.g. '(userid: user_陳志明_johnsonj)'."""
        import re
        match = re.search(r'userid:\s*([^\s\)\]]+)', result)
        if match:
            return match.group(1)

        # Also try: the second key-value after the name
        match = re.search(r'\(userid:\s*(\S+)\)', result)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _extract_name_from_question(user_input: str) -> Optional[str]:
        """Extract a person's name from the question."""
        import re
        # Common Chinese surnames
        surnames = "李王張陳劉楊黃吳趙周徐孫馬朱胡林郭何高羅鄭梁謝宋唐許韓馮鄧曹彭曾蕭田董潘袁蔡蔣余于杜葉程蘇魏呂丁任姚盧沈鍾姜崔譚"
        match = re.search(f'([{surnames}][\\u4e00-\\u9fff]{{1,3}})', user_input)
        if match:
            return match.group(1)
        return None

    def _llm_summarize(self, query: str, msg_result: str, contact_result: str) -> str:
        """Use LiteLLM to summarize combined results."""
        from .litellm_client import LiteLLMClient

        lang = self._detect_language(query)
        client = LiteLLMClient(api_base=self.api_base, api_key=self.api_key, model=self.model)

        if lang == "en":
            prompt = (
                f"User question: {query}\n\n"
                f"Message search results:\n{msg_result}\n\n"
                f"Contact search results:\n{contact_result}\n\n"
                "Based on the above information, give a concise and accurate answer in English. "
                "If information is insufficient, say so honestly. "
                "Only use the provided results — never invent message content, URLs, "
                "sender names, or any other detail that is not in the results."
            )
            system = "You are a corporate chat search assistant. Answer in English, concise and accurate."
        else:
            prompt = (
                f"用户问题：{query}\n\n"
                f"消息搜索结果：\n{msg_result}\n\n"
                f"联系人搜索结果：\n{contact_result}\n\n"
                "请根据以上信息，给出一个简洁准确的答案。如果信息不足，请如实告知。"
                "只能使用给定的搜索结果——严禁编造消息内容、URL、发送者姓名等结果中没有的信息。"
            )
            system = "你是一个企业聊天记录搜索助手。用中文回答，简洁准确。"

        if self.profile is not None:
            system = self.profile.build_system_prompt(system)

        result = client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
            timeout=15,
        )
        if not result:
            return self._format_fallback_answer(query, msg_result, contact_result)

        # If the LLM says it found nothing but we have search results, show the raw results
        no_info_markers = ["没有找到", "未找到", "没有相关", "no information", "no relevant",
                           "not found", "无法找到", "找不到", "没有关于",
                           "do not include", "does not include", "cannot provide",
                           "not available", "not in the results", "no contact",
                           "no person", "无法提供", "没有这个人", "结果中不包含"]
        has_results = bool(msg_result.strip() or contact_result.strip())
        if has_results and any(marker in result for marker in no_info_markers):
            logger.info("LLM reported no info but search returned results — showing raw results")
            return self._format_fallback_answer(query, msg_result, contact_result)

        return result


    def _format_fallback_answer(self, query: str, msg_result: str, contact_result: str) -> str:
        """Smart fallback: extract email from contacts result for a direct answer.

        For cross-table queries (message → contact), the answer is structured as:
          1. WHO sent the message (name + userid)
          2. WHAT they sent (message preview)
          3. THEN the email / contact details
        """
        lang = self._detect_language(query)

        # ── Extract contact details ──
        email = None
        name = None
        userid = None
        company = None
        phone = None
        if contact_result:
            email_m = re.search(r'Email:\s*([^\s\n,]+)', contact_result)
            if email_m:
                email = email_m.group(1)
            name_m = re.search(r'\[Score:.+?\]\s+([\u4e00-\u9fff]+)', contact_result)
            if name_m:
                name = name_m.group(1)
            userid_m = re.search(r'userid:\s*([^\s\)\]]+)', contact_result)
            if userid_m:
                userid = userid_m.group(1)
            company_m = re.search(r'Company:\s*([^\n,]+?)(?:\s+Phone:|\s*$)', contact_result)
            if company_m:
                company = company_m.group(1).strip()
            phone_m = re.search(r'Phone:\s*([^\s\n,]+)', contact_result)
            if phone_m:
                phone = phone_m.group(1)

        # ── Extract message preview ──
        msg_preview = None
        if msg_result:
            # Try to extract the message content from the search result
            content_m = re.search(r'Content:\s*([^\n]+)', msg_result)
            if content_m:
                msg_preview = content_m.group(1).strip()
                # Trim trailing metadata like "(userid: ...)"
                msg_preview = re.sub(r'\s*\(userid:\s*[^)]*\)\s*$', '', msg_preview).strip()
            else:
                # Fallback: take the first meaningful message-content line.
                # Search results are formatted as:
                #   【消息搜索结果】
                #   1. [Score: 0.61] 陳志明 (userid: ...) [Label: ...]
                #      <actual message content>
                # Skip the section header, the numbered score line, and the
                # sender metadata — keep only the indented content line.
                for line in msg_result.split("\n"):
                    line = line.strip()
                    if not line or len(line) <= 5:
                        continue
                    if "【" in line or "】" in line or "Score" in line or "userid" in line:
                        continue
                    if line.startswith(("1.", "2.", "3.", "4.", "5.",
                                        "6.", "7.", "8.", "9.", "0.")):
                        continue
                    msg_preview = line[:120]
                    break

        # ── Build structured answer ──
        if email:
            if lang == "en":
                result = f"✅ Found: **{name or 'Contact'}**"
                if userid:
                    result += f" ({userid})"
                result += "\n\n"
                if msg_preview:
                    result += f"   📩 Sent: \"{msg_preview}\"\n\n"
                result += f"   📧 Email: **{email}**\n"
                if company:
                    result += f"   🏢 Company: {company}\n"
                if phone:
                    result += f"   📱 Phone: {phone}\n"
            else:
                result = f"✅ 找到：**{name or '联系人'}**"
                if userid:
                    result += f" ({userid})"
                result += "\n\n"
                if msg_preview:
                    result += f"   📩 发送内容：\"{msg_preview}\"\n\n"
                result += f"   📧 邮箱：**{email}**\n"
                if company:
                    result += f"   🏢 公司：{company}\n"
                if phone:
                    result += f"   📱 电话：{phone}\n"
            return result

        # ── No email found — show what we have ──
        parts = []
        if contact_result and "没有" not in contact_result and "No" not in contact_result:
            if lang == "en":
                parts.append(f"👤 **Contact info:**\n{contact_result}")
            else:
                parts.append(f"👤 **联系人信息：**\n{contact_result}")
        elif msg_result and "没有" not in msg_result and "No" not in msg_result:
            if lang == "en":
                parts.append(f"📝 **Message search results:**\n{msg_result}")
            else:
                parts.append(f"📝 **消息搜索结果：**\n{msg_result}")
        if not parts:
            if lang == "en":
                return f"Sorry, no information found related to '{query}'."
            return f"抱歉，没有找到与「{query}」相关的信息。"
        return "\n\n".join(parts)

    # ── Thought capture callback ─────────────────────────────────
    class _ThoughtCapture:
        """Simple callback to capture agent reasoning steps."""

        def on_llm_start(self, serialized, prompts, **kwargs):
            pass

        def on_llm_end(self, response, **kwargs):
            pass

        def on_tool_start(self, serialized, input_str, **kwargs):
            pass

        def on_tool_end(self, output, **kwargs):
            pass

        def on_chain_start(self, serialized, inputs, **kwargs):
            pass

        def on_chain_end(self, outputs, **kwargs):
            pass

    @property
    def last_thoughts(self) -> List[str]:
        return self._last_thoughts

    @property
    def last_tool_calls(self) -> List[Dict[str, Any]]:
        return self._last_tool_calls


# ── Standalone helper ────────────────────────────────────────────
def cross_table_chat(user_input: str, **kwargs) -> Dict[str, Any]:
    """
    One-shot convenience function.

    Args:
        user_input: The user query.
        **kwargs: Passed to CrossTableAgent constructor.

    Returns:
        Result dict from CrossTableAgent.process()
    """
    agent = CrossTableAgent(**kwargs)
    return agent.process(user_input)


def is_cross_table_available() -> bool:
    """Check if cross-table agent dependencies are met."""
    return _LANGCHAIN_AVAILABLE