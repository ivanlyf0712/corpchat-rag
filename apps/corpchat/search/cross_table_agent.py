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
import os
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
# 单一来源: apps.corpchat.search.intent_words (候选 4)。这里保留旧私有名
# 作向后兼容别名 (app.py / 测试仍从本模块 import)。
from .intent_words import (
    GREETING_KEYWORDS as _GREETING_KEYWORDS,
    SYSTEM_KEYWORDS as _SYSTEM_KEYWORDS,
    is_greeting_query as _is_greeting_query,
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


# ── Hindsight 按需 recall 判定 (决策 16: 记忆触达词 gate) ──────────
# 用户拍板: 只命中显式跨会话引用词才注入 Hindsight 记忆, 平时不调 recall。
# - 刻意不用裸指代词 (她/他/这个/那个): 会话内指代由 process() 的
#   会话历史注入 (A4-i) 解析, 触发 recall 只会浪费一次调用。
# - 代价: 无触达词的隐性记忆查询 ("客户喜欢什么沟通方式") 会漏掉,
#   退回纯工具回答 —— 等价于接入 Hindsight 之前的行为, 可接受。
# gate 谓词归属 Hindsight 适配器 (hindsight_client.needs_recall), 这里
# 仅保留向后兼容别名 (测试与旧引用仍可 import _needs_hindsight_recall)。
from .hindsight_client import (  # noqa: F401  (向后兼容别名)
    _MEMORY_TRIGGER_KEYWORDS,
    needs_recall as _needs_hindsight_recall,
)


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
        """Store tool schemas so _generate() can send them to the LLM."""
        self._bound_tools = list(tools)
        return self

    def _messages_to_openai(self, messages) -> List[Dict]:
        """Convert LangChain messages to OpenAI chat-format dicts."""
        raw = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                raw.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage):
                raw.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                entry: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                            },
                        }
                        for tc in tool_calls
                    ]
                raw.append(entry)
            elif getattr(msg, "type", "") == "tool":
                raw.append({
                    "role": "tool",
                    "tool_call_id": getattr(msg, "tool_call_id", ""),
                    "content": str(msg.content),
                })
            else:
                raw.append({"role": "user", "content": str(getattr(msg, "content", ""))})
        return raw

    @staticmethod
    def _tools_to_schema(tools) -> List[Dict]:
        """Convert LangChain tools to OpenAI function-calling schema."""
        schema = []
        for tool in tools:
            try:
                name = tool.name if hasattr(tool, "name") else getattr(tool, "func", tool).__name__
            except Exception:
                name = str(tool)
            try:
                args_schema = tool.args_schema
            except Exception:
                args_schema = None
            if args_schema is not None:
                try:
                    parameters = args_schema.model_json_schema()
                except Exception:
                    parameters = args_schema.schema()
            else:
                parameters = {"type": "object", "properties": {}}
            schema.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description if hasattr(tool, "description") else "",
                    "parameters": parameters,
                },
            })
        return schema

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Call the LLM with real tool calling; return ChatResult.

        When tools are bound, the schema is sent with the request. If the
        model responds with tool_calls, they are parsed and returned as
        AIMessage.tool_calls (LangGraph executes them and calls back with the
        results). Otherwise the model's content is the final answer.
        """
        client = self._get_client()
        raw_messages = self._messages_to_openai(messages)
        tools_schema = self._tools_to_schema(self._bound_tools) if self._bound_tools else None

        result = client.chat_message(
            raw_messages,
            tools=tools_schema,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 2048),
            timeout=kwargs.get("timeout", 30),
        )
        if result is None:
            # LLM unreachable → empty AIMessage ends the loop gracefully;
            # CrossTableAgent.process() then falls back.
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        content = result.get("content") or ""
        raw_tool_calls = result.get("tool_calls") or []

        if raw_tool_calls:
            parsed_tool_calls = []
            for tc in raw_tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw)
                except (json.JSONDecodeError, TypeError) as e:
                    # 决策 15: 非法 tool_calls → 抛异常, 让 LangGraph 重试本轮。
                    raise ValueError(
                        f"LLM returned malformed tool arguments for {name!r}: {args_raw!r} ({e})"
                    )
                parsed_tool_calls.append({
                    "name": name,
                    "args": args if isinstance(args, dict) else {},
                    "id": tc.get("id", ""),
                })
            message = AIMessage(content="", tool_calls=parsed_tool_calls)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

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

1. **search_messages(query, sender, receiver, limit)** — 搜索内部聊天消息
   - query: 内容关键词 (如 "合同已签", "物流报价")。当用户只要某人发/收的消息
     (无内容关键词) 时，可以省略 query。
   - sender: 发送者姓名或 userid。指定后只返回此人发送的消息。
     "来自陳志明" / "陳志明发了什么" → sender="陳志明"
   - receiver: 接收者姓名或 userid。指定后只返回发给此人的消息。
     "发给李雅婷" / "谁发过消息给李雅婷" → receiver="李雅婷"
   - sender 和 receiver 可同时使用 (如 "來自陳志明發給李雅婷的消息")。
   - limit: 返回条数 (默认 10，最多 100)。"列出所有"类查询可调大。

2. **search_contacts(query)** — 搜索联系人信息
   - 用于查找联系人邮箱、电话、公司、职位
   - 返回全名、userid、邮箱、公司、电话、职位
   - 示例查询: "李雅婷", "陳志明 email", "johnsonj", "user_陳志明"

3. **search_conversation_partners(person)** — 查询某人跟谁聊过天
   - 用于 "who did X talk to"、"X 跟谁聊过"、"X 和谁对话" 这类关系查询
   - 返回 X 参与过的所有会话的对侧联系人列表 (不含 X 自己)
   - 示例: person="陳志明"

工作流程:
1. 分析用户问题，判断需要哪些数据和过滤条件
2. 涉及消息内容/发送者/接收者 → search_messages (填 query/sender/receiver)
3. 涉及联系人资料 → search_contacts
4. 问"谁跟谁聊过" → search_conversation_partners
5. 需要结合多个来源 → 先查消息获取 userid，再用 userid 查联系人
6. 用中文整合所有结果，给出清晰的自然语言答案

关键规则:
- 如果问题是闲聊、问候、系统能力询问 → 直接回答，无需调用工具
- 解析"来自 X" → sender=X；解析"发给 Y" → receiver=Y；两者同时出现 → 都填
- 如果工具返回"匹配到多个联系人"的澄清提示，向用户询问具体是哪一位
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
        hindsight_bank: Optional[str] = None,
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
        # Hindsight 记忆银行 (跨会话记忆后端): None = 禁用记忆注入
        self.hindsight_bank = hindsight_bank or os.getenv("HINDSIGHT_BANK_ID")
        self._agent: Optional[Any] = None
        self._last_thoughts: List[str] = []
        self._last_tool_calls: List[Dict[str, Any]] = []
        self._steps: List[Dict[str, Any]] = []
        # 每次 process() 调用可注入的 on_tool 回调 (UI 实时工具显示)。
        # 工具包装器在真实调用时读取它, 因此缓存的 agent 可跨调用复用。
        self._on_tool_callback: Optional[callable] = None
        # 真实工具调用的逐次结果快照 (tool → meta), 供 process() 做逐调用归属。
        # 与 tools._last_msg_meta 全局兼容通道不同, 该日志随 agent 实例走,
        # 不跨会话共享, 并保留每次调用的顺序。
        self._tool_meta_log: List[Dict[str, Any]] = []

    def _get_llm(self):
        """Create a LangChain-compatible LLM wrapper around LiteLLMClient."""
        return _LiteLLMWrapper(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
        )

    def _init_agent(self):
        """Lazy-initialize the LangGraph ReAct agent.

        Binds the 3-tool main-path toolset (search_messages / search_contacts /
        search_conversation_partners), injects the module-level retrieval
        config (expand/rerank/graph_parallel), applies the persona profile to
        the system prompt, and gates tools by knowledge.sources.
        """
        if self._agent is not None:
            return

        if not _LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain is not installed. Run: pip install langchain langchain-community")

        from .tools import CROSS_TABLE_TOOLS, configure_search

        configure_search(expand=self.expand, use_rerank=self.use_rerank,
                         graph_parallel=self.graph_parallel)

        tools = list(CROSS_TABLE_TOOLS)
        if self.sources:
            tools = [
                t for t in tools
                if (t.name == "search_messages" and "messages" in self.sources)
                or (t.name == "search_conversation_partners" and "messages" in self.sources)
                or (t.name == "search_contacts" and "contacts" in self.sources)
            ]
        if not tools:
            raise RuntimeError(f"No tools available for sources={self.sources}")

        # 包装工具: 每次真实调用时触发 on_tool 回调 (UI 实时显示精确工具+参数)。
        # 包装只读 self._on_tool_callback (每次 process() 调用可更新), 因此
        # 缓存的 agent (self._agent) 可跨调用安全复用。
        tools = [self._notifying_tool(t) for t in tools]

        model = self._get_llm()
        prompt = SYSTEM_PROMPT
        if self.profile is not None:
            prompt = self.profile.build_system_prompt(prompt)

        self._agent = create_react_agent(model, tools=tools, prompt=prompt)

    def _notifying_tool(self, tool):
        """Wrap a LangChain tool so each real invocation notifies the UI.

        Rebuilds the tool via StructuredTool.from_function preserving its name,
        description and args_schema (so model tool-calling schemas are unchanged).
        The wrapper fires `self._on_tool_callback(tool_name, tool_args)` (set per
        process() call) right before the underlying function runs, then returns
        the original result untouched.
        """
        import functools
        from langchain_core.tools import StructuredTool

        func = tool.func

        @functools.wraps(func)
        def _run(*args, **kwargs):
            cb = self._on_tool_callback
            if cb is not None:
                try:
                    cb(tool.name, dict(kwargs))
                except Exception:
                    pass
            result = func(*args, **kwargs)
            # 逐调用结果快照: 在工具返回后立即(同线程)抓取该次调用的结构化 meta,
            # 供 process() 做逐调用归属 (多 search_messages 调用各得各的 meta,
            # 并发会话互不污染)。
            try:
                from .tools import snapshot_meta
                meta = snapshot_meta(tool.name)
                if meta is not None:
                    self._tool_meta_log.append({"tool": tool.name, "meta": meta})
            except Exception:
                pass
            return result

        return StructuredTool.from_function(
            func=_run,
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
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


    def process(self, user_input: str, on_stage: Optional[callable] = None,
                on_tool: Optional[callable] = None,
                history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process a user query through the cross-table agent.

        Args:
            user_input: The user's query string.
            on_stage: Optional callback `on_stage(label: str, detail: str = "")`
                invoked as each processing stage starts. The UI uses it to
                drive the fade-in/out stage animation.
            on_tool: Optional callback `on_tool(tool_name: str, tool_args: dict)`
                invoked each time the agent actually calls a search tool, so the
                UI can show the exact tool + arguments live (per-tool stage).
            history: Optional list of prior conversation turns, each with
                {"query": str, "answer": str}. Injected as session context so
                the agent can resolve references ("her" → 李雅婷) and avoid
                redundant tool calls on follow-ups. Kept concise (summaries)
                to limit token cost; result tables are preserved verbatim.

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
        self._tool_meta_log = []
        # 每次 process() 调用可携带新的 on_tool 回调 (UI 实时工具显示)
        self._on_tool_callback = on_tool
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

        # ── Step 2: Real LangGraph ReAct loop (DeepSeek native tool calling) ──
        try:
            from .tools import configure_search
            configure_search(expand=self.expand, use_rerank=self.use_rerank,
                             graph_parallel=self.graph_parallel)
            self._init_agent()

            _stage("🧠", "agent thinking...")
            _t0 = _time.perf_counter()

            # ── Hindsight 跨会话记忆注入 ──
            # recall 与当前查询相关的历史记忆, 作为上下文附在用户问题后,
            # 让 agent 能利用此前会话存入的记忆 (跨会话/跨轮次上下文)。
            # ── 会话内历史注入 (A4-i: 让 LLM 自己解析指代) ──
            # 把前几轮对话摘要附在用户问题前, 让 agent 能解析 "her"→李雅婷
            # 等指代, 并在追问时避免重复调工具。结果表格保留原文 (A5-b)。
            hist_parts = []
            for t in (history or [])[-6:]:
                if not isinstance(t, dict):
                    continue
                q = str(t.get("query") or "")
                a = str(t.get("answer") or "")
                if q:
                    hist_parts.append(f"用户问: {q[:300]}")
                if a:
                    hist_parts.append(f"助手答: {a[:500]}")
            if hist_parts:
                agent_input = (
                    "[对话历史 (本次会话前几轮, 用于理解指代如'她'):]\n"
                    + "\n".join(hist_parts)
                    + f"\n\n[当前问题]: {user_input}"
                )
            else:
                agent_input = user_input

            # ── Hindsight 跨会话记忆注入 (决策 16: 按需 recall gate) ──
            # 仅当查询命中显式跨会话引用词 (上次/之前/记得/以前/当时/上回 等)
            # 才调 recall; 普通查询跳过, 避免污染上下文 + 增加延迟。
            # 裸指代词 (她/他/这个) 不触发 —— 会话内指代已由上面的历史注入解析。
            if self.hindsight_bank and _needs_hindsight_recall(user_input):
                _mem_t0 = _time.perf_counter()
                try:
                    from .hindsight_client import recall as hs_recall
                    mems = hs_recall(user_input, bank=self.hindsight_bank, max_results=5)
                    if mems:
                        lines = []
                        for m in mems:
                            c = m.get("content") or m.get("text") or ""
                            if c:
                                lines.append(f"- {c[:200]}")
                        if lines:
                            agent_input = (
                                f"{agent_input}\n\n[相关历史记忆 (Hindsight bank"
                                f" {self.hindsight_bank})]:\n" + "\n".join(lines)
                                + "\n[请结合以上记忆回答; 若与当前问题无关可忽略]"
                            )
                except Exception as _mems_err:
                    logger.debug(f"Hindsight recall failed: {_mems_err}")
                self._add_step("🧠", "Hindsight memory",
                               int((_time.perf_counter() - _mem_t0) * 1000),
                               f"recall on '{user_input[:40]}'")

            response = self._agent.invoke(
                {"messages": [HumanMessage(content=agent_input)]}
            )
            _t1 = _time.perf_counter()
            self._add_step("🧠", "Agent reasoning", int((_t1 - _t0) * 1000), "LangGraph ReAct loop")
            _stage("✨", "assembling answer...")

            # ── Parse the agent's message history ──
            messages = response.get("messages", [])
            executed_calls: List[Dict[str, Any]] = []
            msg_result = ""
            contact_result = ""
            # 收集工具调用与观察 (供 UI/记忆图谱使用)。
            # 逐调用归属: 真实执行时每个工具调用各得各的 meta (来自 agent 实例的
            # _tool_meta_log, 按执行顺序); 直接注入的 fake agent (测试) 没有真实
            # 工具执行, 回退到模块级兼容通道 get_last_*_meta。
            from .tools import get_last_search_meta, get_last_contact_meta
            last_msg_meta = get_last_search_meta()
            last_contact_meta = get_last_contact_meta()
            msg_log = [e["meta"] for e in self._tool_meta_log if e["tool"] == "search_messages"]
            contact_log = [e["meta"] for e in self._tool_meta_log if e["tool"] == "search_contacts"]
            # 记忆图谱数据源: 真实 search_messages 最后一次调用的 raw_hits 优先。
            graph_raw_hits = (msg_log[-1].get("raw_hits", []) if msg_log
                              else last_msg_meta.get("raw_hits", []))
            for m in messages:
                tool_calls = getattr(m, "tool_calls", None) or []
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("args", {}) or {}
                    if name == "search_messages":
                        msg_result = f"search_messages({args})"
                    elif name == "search_contacts":
                        contact_result = f"search_contacts({args})"
                    if name == "search_messages":
                        meta = msg_log.pop(0) if msg_log else last_msg_meta
                    elif name == "search_contacts":
                        meta = contact_log.pop(0) if contact_log else last_contact_meta
                    else:
                        meta = {}
                    executed_calls.append({
                        "tool": name,
                        "tool_input": args,
                        "observation": "",
                        "meta": meta,
                    })
            self._last_tool_calls = executed_calls

            # 最终答案 = 最后一个 assistant 消息的内容
            final_content = ""
            for m in reversed(messages):
                if getattr(m, "type", "") == "ai" and m.content:
                    final_content = str(m.content)
                    break
            if not final_content:
                raise RuntimeError("Agent returned no final answer")

            return {
                "output": final_content,
                "thoughts": [f"Agent executed {len(executed_calls)} tool call(s)"],
                "tool_calls": executed_calls,
                "raw_hits": graph_raw_hits,
                "steps": self._steps,
                "success": True,
                "fallback": False,
            }

        except Exception as e:
            logger.warning(f"Cross-table agent failed: {e}")
            self._add_step("⚠️", "Agent error", 0, str(e)[:100])
            # ── Step 3: Fallback — two-step reasoning (degraded, deterministic) ──
            return self._fallback_process(user_input, error=str(e))


    def _quick_respond(self, user_input: str) -> Optional[str]:
        """Handle greetings and system questions without invoking tools.

        Rule-first fast path (keyword gates are <1ms):
          greeting → LLM-generated greeting response (or preset if LLM down)
          system   → self-description
          search   → proceed to the LangGraph agent

        Per decision 8, the LLM intent-classification half was removed: any
        query that is not a rule-detected greeting/system goes straight to the
        LangGraph agent (DeepSeek decides whether tools are needed).

        The LLM availability probe is deferred until a greeting is actually
        hit — search queries (the common path) never pay the /v1/models
        round-trip.
        """
        q = user_input.lower().strip()
        lang = self._detect_language(user_input)

        # Fast keyword gates FIRST — known greetings never pay an LLM round-trip
        if _is_greeting_query(q):
            # 仅问候命中时才探测 LLM (探测结果按 base+key 缓存)
            llm_ok = self._check_llm()
            if llm_ok:
                return self._generate_greeting(user_input, lang)
            return self._PRESET_GREETING_EN if lang == "en" else self._PRESET_GREETING_ZH
        if any(kw in q for kw in _SYSTEM_KEYWORDS):
            return self._system_info_text(lang)

        # 其余一律进入 LangGraph agent (模糊查询由 DeepSeek 自己决定是否需要工具)
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

    def _generate_greeting(self, user_input: str, lang: str) -> str:
        """LLM-generated greeting response; preset fallback when LLM is down.

        委托 intent_words.generate_greeting (单一实现, 候选 4)。
        """
        try:
            from .litellm_client import LiteLLMClient
            from .intent_words import generate_greeting
            client = LiteLLMClient(api_base=self.api_base, api_key=self.api_key)
            fallback = self._PRESET_GREETING_EN if lang == "en" else self._PRESET_GREETING_ZH
            return generate_greeting(client, user_input, lang, fallback=fallback)
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

        from .tools import search_messages, search_contacts, configure_search
        configure_search(expand=self.expand, use_rerank=self.use_rerank,
                         graph_parallel=self.graph_parallel)

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
            msg_result = search_messages.invoke({"query": search_query})
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
            # (结构化优先: 从工具 meta 的 raw_hits 读 userid; 无结构化时才回退 regex)
            msg_struct = {}
            try:
                from .tools import get_structured_result
                msg_struct = get_structured_result("search_messages") or {}
            except Exception:
                pass
            userid = self._extract_userid_from_hits(msg_struct.get("hits") or [])
            if not userid:
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

        # ── 收集结构化 hits (ticket 04: fallback 答案路径消费结构化结果) ──
        msg_hits = (msg_struct.get("hits") or []) if "msg_struct" in dir() else []
        contact_struct = {}
        try:
            from .tools import get_structured_result
            contact_struct = get_structured_result("search_contacts") or {}
        except Exception:
            pass
        contact_hits = contact_struct.get("hits") or []

        # ── Step 3: Try LLM summarization ──
        try:
            summary = self._llm_summarize(user_input, msg_result, contact_result,
                                          msg_hits=msg_hits, contact_hits=contact_hits)
        except Exception:
            summary = self._structured_fallback_answer(user_input, msg_result, contact_result,
                                                       msg_hits, contact_hits)

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
    def _extract_userid_from_hits(raw_hits: List[Dict]) -> Optional[str]:
        """从结构化 message hits 中提取 userid (确定性, 无 regex 解析格式化字符串)。

        每条 hit 的 metadata 携带 external_userid (客户) / servicer_userid (客服),
        直接读取即可, 不再从格式化显示文本里抓 '(userid: ...)'。
        """
        for h in raw_hits or []:
            if not isinstance(h, dict):
                continue
            meta = h.get("metadata") or {}
            if not isinstance(meta, dict):
                continue
            uid = meta.get("external_userid") or meta.get("servicer_userid")
            if uid:
                return str(uid)
        return None

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

    def _llm_summarize(self, query: str, msg_result: str, contact_result: str,
                       msg_hits: Optional[List[Dict]] = None,
                       contact_hits: Optional[List[Dict]] = None) -> str:
        """Use LiteLLM to summarize combined results.

        msg_hits/contact_hits: 结构化工具 hits (ticket 04)。LLM 不可用或判断
        无信息时, fallback 渲染优先消费结构化 hits (无 regex 解析格式化字符串)。
        """
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
            return self._structured_fallback_answer(query, msg_result, contact_result,
                                                    msg_hits or [], contact_hits or [])

        # If the LLM says it found nothing but we have search results, show the raw results
        no_info_markers = ["没有找到", "未找到", "没有相关", "no information", "no relevant",
                           "not found", "无法找到", "找不到", "没有关于",
                           "do not include", "does not include", "cannot provide",
                           "not available", "not in the results", "no contact",
                           "no person", "无法提供", "没有这个人", "结果中不包含"]
        has_results = bool(msg_result.strip() or contact_result.strip())
        if has_results and any(marker in result for marker in no_info_markers):
            logger.info("LLM reported no info but search returned results — showing raw results")
            return self._structured_fallback_answer(query, msg_result, contact_result,
                                                    msg_hits or [], contact_hits or [])

        return result


    def _structured_fallback_answer(self, query: str, msg_result: str, contact_result: str,
                                    msg_hits: List[Dict], contact_hits: List[Dict]) -> str:
        """Fallback 渲染: 有结构化 hits 时直接渲染结构化结果 (ticket 04, 无 regex),
        否则回退到 legacy 格式化字符串解析 (保持测试向后兼容)。"""
        if msg_hits or contact_hits:
            return self._format_structured_answer(query, msg_hits or [], contact_hits or [],
                                                  self._detect_language(query))
        return self._format_fallback_answer(query, msg_result, contact_result)



    def _format_fallback_answer(self, query: str, msg_result: str, contact_result: str) -> str:
        """Legacy formatted-string parsing (kept for backward-compat unit tests;
        the production fallback path calls `_format_structured_answer` with the
        tools' structured hits — ticket 04 removes regex-scraping from that path).
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

    def _format_structured_answer(self, query: str, msg_hits: List[Dict],
                                  contact_hits: List[Dict], lang: str) -> str:
        """从结构化工具 hits 渲染 fallback 答案 (纯渲染, 无 regex 解析)。

        contact_hits[0] / msg_hits[0] 的 metadata 直接携带联系人字段与
        发送者 userid — 答案路径不再从格式化字符串里抓 Email:/Company:。
        """
        contact: Dict = {}
        if contact_hits and isinstance(contact_hits[0], dict):
            c = contact_hits[0]
            contact = c.get("metadata") if isinstance(c.get("metadata"), dict) else c
        email = contact.get("email")
        name = contact.get("full_name") or contact.get("name")
        userid = contact.get("userid")
        company = contact.get("company")
        phone = contact.get("phone")

        msg_preview = None
        if msg_hits and isinstance(msg_hits[0], dict):
            text = str(msg_hits[0].get("text") or "")
            if "\n---\n" in text:
                msg_preview = text.split("\n---\n", 1)[1].strip()[:120]
            else:
                msg_preview = text.strip()[:120]

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
        if msg_preview:
            if lang == "en":
                parts.append(f"📝 **Message search results:**\n{msg_preview}")
            else:
                parts.append(f"📝 **消息搜索结果：**\n{msg_preview}")
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