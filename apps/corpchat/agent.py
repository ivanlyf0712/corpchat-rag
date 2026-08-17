#!/usr/bin/env python3
"""
CorpChat Agentic Intelligence Layer
=====================================

Purpose:
    Add an intelligent agent layer on top of the existing Searcher, enabling
    intent classification, routing, and self-description — without modifying
    search.py.

Design (per grill session):
    - 5 intent categories: greeting, system_info, search, clarify, fallback
    - Classification: rule-based first (<1ms) → LLM fallback (2s timeout)
      → default to "search" when LLM unavailable (safe degradation)
    - Routing:
        greeting     → static greeting message
        system_info  → static self-description
        search       → call Searcher.search() with enhancement params from caller
        clarify      → ask user to rephrase
        fallback     → treat as search (safe default)
    - Multi-turn context: simple last-N turns history
    - Performance: rules <1ms, LLM <2s, total agent overhead <500ms (excluding search)

Integration:
    The agent is used by app.py's chat flow and can also be called from CLI:
        from apps.corpchat.agent import Agent, load_agent
        agent = load_agent()
        intent, response, search_results = agent.process("物流報價 方案", top_k=5)
"""

import os
import re
import time
import json
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("corpchat-agent")

# ── Configuration ────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
import sys as _sys
if ROOT_DIR not in _sys.path:
    _sys.path.insert(0, ROOT_DIR)

# Import search layer (no modifications to search.py)
from apps.corpchat.search import (
    Searcher,
    QueryExpander,
    Reranker,
    AgenticDecider,
    LiteLLMClient,
    DispositionProfile,
    load_index,
    DEFAULT_INDEX_PATH,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LITELLM_MODEL,
)

# ── Shared LiteLLM client ──────────────────────────────────────────────
_llm_client = LiteLLMClient(
    api_base=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    model=LITELLM_MODEL,
)

# ── Intent categories ────────────────────────────────────────────────────
INTENT_GREETING = "greeting"
INTENT_SYSTEM_INFO = "system_info"
INTENT_SEARCH = "search"
INTENT_CLARIFY = "clarify"
INTENT_FALLBACK = "fallback"

CONTRACT_TYPE_KEYWORDS = (
    "contract type", "type", "agreement type", "service contract", "purchase contract",
    "sales contract", "nda", "non-disclosure", "employment contract", "lease contract",
    "contract類型", "合同類型", "合約類型", "类型", "類型",
)
CONTRACT_STATUS_KEYWORDS = (
    "contract status", "status", "active", "inactive", "draft", "signed",
    "pending", "approved", "rejected", "expired", "terminated", "completed",
    "進行中", "已簽署", "已签署", "草稿", "已完成", "失效", "終止", "終止",
)

CONTRACT_TYPE_ALIASES = {
    "nda": "NDA",
    "non-disclosure": "NDA",
    "non disclosure": "NDA",
    "service contract": "Service Contract",
    "purchase contract": "Purchase Contract",
    "sales contract": "Sales Contract",
    "employment contract": "Employment Contract",
    "lease contract": "Lease Contract",
}

# ── Label aliases (query term → canonical corpus label) ─────────
# 语料标签共 31 类 (见 gen_fake_msg.py)。别名表把用户自然用语
# (中/英) 映射到确切标签值, 供 _extract_label_filter 精确过滤。
# 值可以是 str 或 tuple — tuple 用于一个别名对应多个候选标签的情况
# (如 fraud 语料里可能写作 fraud 或 詐騙)。
LABEL_ALIASES = {
    # English
    "after service": "after_service",
    "after-service": "after_service",
    "after sales": "after_service",
    "annual review": "annual_review",
    "business proposal": "business_proposal",
    "proposal": "business_proposal",
    "contract renewal": "contract_renewal",
    "renewal": "contract_renewal",
    "coordination": "coordination",
    "delivery status": "delivery_status",
    "delivery": "delivery_status",
    "equipment maintenance": "equipment_maintenance",
    "maintenance": "equipment_maintenance",
    "equipment quote": "equipment_quote",
    "factory audit": "factory_audit",
    "audit": "factory_audit",
    "investment opportunity": "investment_opportunity",
    "investment": "investment_opportunity",
    "invoice issue": "invoice_issue",
    "invoice": "invoice_issue",
    "marketing campaign": "marketing_campaign",
    "marketing": "marketing_campaign",
    "campaign": "marketing_campaign",
    "meeting schedule": "meeting_schedule",
    "meeting": "meeting_schedule",
    "old friend": "old_friend_reconnect",
    "reconnect": "old_friend_reconnect",
    "order change": "order_change",
    "order confirmation": "order_confirmation",
    "order": "order_confirmation",
    "partnership": "partnership_discussion",
    "payment reminder": "payment_reminder",
    "payment": "payment_reminder",
    "product demo": "product_demo",
    "demo": "product_demo",
    "product inquiry": "product_inquiry",
    "inquiry": "product_inquiry",
    "quality issue": "quality_issue",
    "quality": "quality_issue",
    "quotation request": "quotation_request",
    "quotation": "quotation_request",
    "quote": "quotation_request",
    "recruitment": "recruitment",
    "hiring": "recruitment",
    "sample request": "sample_request",
    "sample": "sample_request",
    "software license": "software_license",
    "license": "software_license",
    "system upgrade": "system_upgrade",
    "upgrade": "system_upgrade",
    "tech support": "tech_support",
    "support": "tech_support",
    "training program": "training_program",
    "training": "training_program",
    "vendor evaluation": "vendor_evaluation",
    "vendor": "vendor_evaluation",
    "warehouse transfer": "warehouse_transfer",
    "warehouse": "warehouse_transfer",
    "warranty claim": "warranty_claim",
    "warranty": "warranty_claim",
    # 中文 (繁/简)
    "售後": "after_service", "售后": "after_service",
    "年度回顧": "annual_review", "年度回顾": "annual_review",
    "提案": "business_proposal",
    "續約": "contract_renewal", "续约": "contract_renewal",
    "協調": "coordination", "协调": "coordination",
    "物流": "delivery_status", "配送": "delivery_status",
    "交貨": "delivery_status", "交货": "delivery_status",
    "維護": "equipment_maintenance", "维护": "equipment_maintenance",
    "設備報價": "equipment_quote", "设备报价": "equipment_quote",
    "驗廠": "factory_audit", "验厂": "factory_audit",
    "投資": "investment_opportunity", "投资": "investment_opportunity",
    "發票": "invoice_issue", "发票": "invoice_issue",
    "行銷": "marketing_campaign", "营销": "marketing_campaign",
    "會議": "meeting_schedule", "会议": "meeting_schedule",
    "老朋友": "old_friend_reconnect", "敘舊": "old_friend_reconnect",
    "叙旧": "old_friend_reconnect",
    "改單": "order_change", "改单": "order_change",
    "訂單變更": "order_change", "订单变更": "order_change",
    "訂單確認": "order_confirmation", "订单确认": "order_confirmation",
    "下單": "order_confirmation", "下单": "order_confirmation",
    "合作": "partnership_discussion",
    "催款": "payment_reminder", "付款": "payment_reminder",
    "演示": "product_demo",
    "產品詢問": "product_inquiry", "产品询问": "product_inquiry",
    "品質": "quality_issue", "品质": "quality_issue",
    "報價": "quotation_request", "报价": "quotation_request",
    "詢價": "quotation_request", "询价": "quotation_request",
    "招聘": "recruitment",
    "樣品": "sample_request", "样品": "sample_request",
    "授權": "software_license", "授权": "software_license",
    "升級": "system_upgrade", "升级": "system_upgrade",
    "技術支援": "tech_support", "技术支持": "tech_support",
    "培訓": "training_program", "培训": "training_program",
    "供應商": "vendor_evaluation", "供应商": "vendor_evaluation",
    "倉庫": "warehouse_transfer", "仓库": "warehouse_transfer",
    "調撥": "warehouse_transfer", "调拨": "warehouse_transfer",
    "保固": "warranty_claim", "保修": "warranty_claim",
    # 风险标签 (语料中诈骗消息的 label 值依部署可能是 詐騙 或 fraud)
    "詐騙": ("詐騙", "fraud"), "诈骗": ("詐騙", "fraud"),
    "fraud": ("詐騙", "fraud"), "scam": ("詐騙", "fraud"),
    "phishing": ("詐騙", "fraud"),
}

# ── 过滤意图词 (label 别名 + 这些词同时出现 → 硬过滤) ─────────────
# 单独的 "物流報價" 是内容查询不是过滤意图; 只有 "只看物流 / only invoices /
# filter by label" 这类表达才应缩小召回, 否则标签仅用于结果分组排序。
FILTER_INTENT_KEYWORDS = (
    "只看", "只要", "仅看", "僅看", "限于", "限定", "限於",
    "only", "just", "filter", "筛選", "篩選", "筛选",
    "類別", "类别", "category", "label", "標籤", "标签",
)

CONTRACT_STATUS_ALIASES = {
    "active": "Active",
    "inactive": "Inactive",
    "draft": "Draft",
    "signed": "Signed",
    "pending": "Pending",
    "approved": "Approved",
    "rejected": "Rejected",
    "expired": "Expired",
    "terminated": "Terminated",
    "completed": "Completed",
    "進行中": "Active",
    "已簽署": "Signed",
    "已签署": "Signed",
    "草稿": "Draft",
    "已完成": "Completed",
    "失效": "Expired",
    "終止": "Terminated",
}

# ── Rule-based keyword sets ──────────────────────────────────────────────
# 单一来源: apps.corpchat.search.intent_words (候选 4 — 与 cross_table_agent
# 共享同一套词表, 避免三份拷贝漂移)。保留旧列表名作向后兼容别名。
from apps.corpchat.search.intent_words import (  # noqa: E402
    CLARIFY_KEYWORDS,
    GREETING_KEYWORDS,
    SYSTEM_KEYWORDS as SYSTEM_INFO_KEYWORDS,
)

# LLM classification timeout (seconds) — per design spec §2.7
LLM_INTENT_TIMEOUT = 2.0


class IntentClassifier:
    """
    Intent classification using rule-based first, LLM fallback.

    Classification flow:
        1. Rule matching (keywords) — <1ms, catches 80% of common cases
        2. LLM classification (if available) — <2s, handles 20% edge cases
        3. Default to "search" — when LLM is unavailable (safe degradation)

    Intent categories:
        greeting, system_info, search, clarify, fallback

    The fallback → search mapping is a safe design choice: if we can't
    classify the intent, it's better to try a search than to refuse to act.
    """

    # Static LLM classification prompt
    _LLM_PROMPT = (
        "Classify the user's intent into ONE of: greeting, system_info, search, clarify, fallback. "
        "greeting = casual hello, system_info = asking about the system's capabilities/identity, "
        "search = looking for information in chat messages, clarify = asking for more detail/explanation. "
        "If unsure, return 'fallback'. "
        "Reply with ONLY the category name."
    )

    def __init__(self, lite_llm_available: Optional[bool] = None):
        """
        Args:
            lite_llm_available: If None, auto-detect. If False, rules-only.
        """
        self._llm_check_done = False
        self._llm_available = lite_llm_available

    def _check_llm(self) -> bool:
        """Check if LiteLLM endpoint is reachable (cached)."""
        if self._llm_check_done:
            return self._llm_available if self._llm_available is not None else False

        if self._llm_available is not None:
            self._llm_check_done = True
            return self._llm_available

        if not LITELLM_API_KEY:
            self._llm_check_done = True
            self._llm_available = False
            return False

        self._llm_available = _llm_client.is_available(timeout=3)

        self._llm_check_done = True
        return self._llm_available

    def _rule_classify(self, query: str) -> Optional[str]:
        """
        Rule-based classification using keyword matching.
        Returns intent string or None if no rule matches.
        Complexity: O(n * m) where n=len(keywords), m=len(query) — <1ms.
        """
        q_lower = query.lower().strip()

        # System info: asking "what can you do / access / know" type questions
        # Check BEFORE greeting because these often contain greeting-like substrings
        for kw in SYSTEM_INFO_KEYWORDS:
            if kw in q_lower:
                return INTENT_SYSTEM_INFO
        if "do you know" in q_lower or "can you access" in q_lower or "what can you" in q_lower:
            return INTENT_SYSTEM_INFO

        # Greeting: typically short, single word
        if len(q_lower) <= 15:
            for kw in GREETING_KEYWORDS:
                if kw in q_lower:
                    return INTENT_GREETING

        # Clarify: asking for more detail
        for kw in CLARIFY_KEYWORDS:
            if kw in q_lower:
                return INTENT_CLARIFY

        # Explicit search intent keywords
        search_kws = ["找", "搜尋", "搜索", "查", "查詢", "找找", "搜", "找一下",
                       "search", "find", "查一下", "幫我找", "協助搜尋"]
        q_words = q_lower.split()
        if any(kw in q_lower for kw in search_kws):
            return INTENT_SEARCH

        return None

    def _llm_classify(self, query: str) -> Optional[str]:
        """
        LLM-based classification as fallback.
        2s timeout per design spec. Returns None if LLM unavailable or fails.
        """
        if not self._check_llm():
            return None

        try:
            result = _llm_client.chat(
                [
                    {"role": "system", "content": self._LLM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=10,
                timeout=LLM_INTENT_TIMEOUT,
            )
            result = result.strip().lower()
            # Normalize: map synonyms
            for intent in [INTENT_GREETING, INTENT_SYSTEM_INFO, INTENT_SEARCH,
                           INTENT_CLARIFY, INTENT_FALLBACK]:
                if intent in result:
                    return intent
            return None
        except Exception as e:
            logger.debug(f"LLM classification failed: {e}")
            return None

    def classify(self, query: str) -> str:
        """
        Classify user intent.

        Flow: rules → LLM → default "search"

        Returns one of: greeting, system_info, search, clarify, fallback
        """
        # Step 1: Rule-based (fast, <1ms)
        t0 = time.perf_counter()
        result = self._rule_classify(query)
        rule_time = (time.perf_counter() - t0) * 1000

        if result:
            logger.debug(f"Intent '{result}' via rules ({rule_time:.1f}ms)")
            return result

        # Step 2: LLM fallback (<2s)
        t0 = time.perf_counter()
        result = self._llm_classify(query)
        llm_time = (time.perf_counter() - t0) * 1000

        if result:
            logger.debug(f"Intent '{result}' via LLM ({llm_time:.1f}ms)")
            return result

        # Step 3: Safe default — treat as search
        logger.debug(f"Intent 'search' (default, rules={rule_time:.1f}ms, llm={llm_time:.1f}ms)")
        return INTENT_SEARCH


class Agent:
    """
    Agentic intelligence layer wrapping Searcher.

    Provides:
        - Intent classification (rule + LLM fallback)
        - Intent routing (static replies, search, clarification)
        - Multi-turn context memory (last N turns, DB-backed with in-memory fallback)
        - Graceful degradation when LLM is unavailable
        - Dynamic LLM-generated greetings
        - Static system info response (always available without LLM)

    The agent does NOT modify search.py — it wraps Searcher.search().
    """

    # Static fallback greeting when LLM is unavailable
    _GREETING_RESPONSE = (
        "Hello! I'm **CorpChat Intelligence** — your corporate chat search assistant. "
        "I can search through conversations for logistics, investments, scams, and more. "
        "How can I help you today?"
    )

    # Static response for system info (always available without LLM)
    _SYSTEM_INFO_RESPONSE = (
        "I'm **CorpChat Intelligence** — an AI-powered search assistant for corporate chat messages.\n\n"
        "**Capabilities:**\n"
        "- Semantic hybrid search (BM25 + vector embeddings)\n"
        "- LLM query expansion for better recall\n"
        "- Graph-enhanced retrieval (traverses conversation relationships)\n"
        "- Cross-encoder reranking for result relevance\n\n"
        "**Data scope:**\n"
        "- Indexed corporate WeCom conversations\n"
        "- Topics: business inquiries, quotations, investments, logistics, tech support,\n"
        "  invoices, contracts, quality issues, scam/phishing detection\n"
        "- Bilingual (Chinese + English) messages\n\n"
        "**Limitations:**\n"
        "- Can only search within the indexed message corpus\n"
        "- Cannot access external data or real-time feeds\n"
        "- LLM-dependent features (query expansion, agentic mode) degrade gracefully\n"
        "  when the LLM endpoint is unavailable\n\n"
        "How can I help you today?"
    )

    def __init__(
        self,
        searcher: Optional[Searcher] = None,
        classifier: Optional[IntentClassifier] = None,
        max_history: int = 10,
        session_id: Optional[str] = None,
    ):
        """
        Args:
            searcher: A pre-constructed Searcher instance. If None, must call
                        set_searcher() before processing search queries.
            classifier: Intent classifier. If None, creates a default one.
            max_history: Maximum number of turns to keep in context memory.
            session_id: Optional session ID for DB-backed memory. If None, a new
                        UUID is generated. Pass an explicit ID to resume a session.
        """
        self.searcher = searcher
        self.classifier = classifier or IntentClassifier()
        self.max_history = max_history
        self.session_id = session_id or str(uuid.uuid4())
        self.chat_history: List[Dict[str, Any]] = []
        self._turn_counter = 0
        self._load_memory_from_db()

    def set_searcher(self, searcher: Searcher):
        """Set or replace the Searcher instance (lazy loading support)."""
        self.searcher = searcher

    def _load_memory_from_db(self):
        """Load recent turns from DB if available; fall back to empty."""
        try:
            from core.corpchat_db import load_agent_memory
            turns = load_agent_memory(self.session_id, max_turns=self.max_history)
            if turns:
                self.chat_history = turns[-self.max_history:]
                self._turn_counter = len(turns)
        except Exception:
            # DB unavailable — use in-memory only
            pass

    def _persist_turn(self, user_msg: str, bot_msg: str, intent: str):
        """Append the current turn to DB memory if possible."""
        try:
            from core.corpchat_db import save_agent_memory
            self._turn_counter += 1
            save_agent_memory(
                session_id=self.session_id,
                turn_number=self._turn_counter,
                user_message=user_msg,
                bot_message=bot_msg,
                intent=intent,
            )
        except Exception:
            # DB unavailable — silently keep in-memory only
            pass

    def _add_to_history(self, user_msg: str, bot_msg: str):
        """Add a turn to the multi-turn context memory."""
        self.chat_history.append({"user": user_msg, "bot": bot_msg})
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def _get_context(self, query: str) -> str:
        """
        Build context from chat history + current query.
        For clarification queries, include the last turn for context.
        """
        if not self.chat_history:
            return query

        # Include last 3 turns as context
        recent = self.chat_history[-3:]
        context_parts = [f"Previous conversation:"]
        for turn in recent:
            context_parts.append(f"  User: {turn['user']}")
            context_parts.append(f"  Assistant: {turn['bot'][:100]}...")
        context_parts.append(f"Current query: {query}")
        return "\n".join(context_parts)

    def process(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        expand: bool = True,
        graph_expand: int = 1,
        label_filter: Optional[str] = None,
        search_mode: str = "hybrid",
        graph_parallel: bool = False,
        profile: Optional[DispositionProfile] = None,
    ) -> Tuple[str, str, List[Dict]]:
        """
        Process a user query through the agentic pipeline.

        Flow:
            1. Classify intent (rule-based → LLM fallback → default search)
            2. Route to handler based on intent
            3. For search intent: call Searcher.search() with params
            4. Return (intent, response, search_results)

        Args:
            query: User's input string.
            top_k: Number of search results to return (for search intent).
            use_rerank: Whether to use cross-encoder reranking.
            expand: Whether to use LLM query expansion.
            graph_expand: Number of graph expansion hops.
            label_filter: Optional label to filter results by.
            search_mode: "hybrid", "keyword", or "semantic".
            graph_parallel: Treat graph traversal as a parallel RRF fusion path
                (Hindsight graph-traversal evidence; defaults off).

        Returns:
            Tuple of (intent, response_text, search_results)
            - intent: the classified intent string
            - response_text: text to show the user (static reply or LLM answer)
            - search_results: list of result dicts (empty for non-search intents)
        """
        t0 = time.perf_counter()

        # Step 1: Classify intent
        intent = self.classifier.classify(query)
        classify_time = (time.perf_counter() - t0) * 1000

        # Step 2: Route based on intent
        if intent == INTENT_GREETING:
            # Use LLM to generate a context-aware greeting
            greeting = self._generate_greeting(user_query=query)
            self._add_to_history(query, greeting)
            self._persist_turn(query, greeting, intent)
            return intent, greeting, []

        elif intent == INTENT_SYSTEM_INFO:
            self._add_to_history(query, self._SYSTEM_INFO_RESPONSE)
            self._persist_turn(query, self._SYSTEM_INFO_RESPONSE, intent)
            return intent, self._SYSTEM_INFO_RESPONSE, []

        elif intent == INTENT_CLARIFY:
            response = "I'd be happy to clarify! Could you rephrase your question or provide more specific details about what you're looking for?"
            self._add_to_history(query, response)
            self._persist_turn(query, response, intent)
            return intent, response, []

        elif intent == INTENT_FALLBACK:
            # Fallback → search (safe default per design)
            # Fall through to search with a note
            intent = INTENT_SEARCH

        # Step 3: Search intent (covers INTENT_SEARCH and INTENT_FALLBACK)
        if self.searcher is None:
            response = "Search system is not initialized. Please load the index first."
            self._add_to_history(query, response)
            return intent, response, []

        contract_facets = self._extract_contract_facets(query)
        # 显式 label_filter 参数优先; 否则要求查询带明确过滤意图词才硬过滤,
        # 避免 "物流報價" 这类内容查询被标签缩小召回 (回归: test_logistics_query)
        query_label_filter = self._extract_label_filter(query)
        hard_from_query = query_label_filter if self._has_filter_intent(query) else None
        effective_label_filter = label_filter or hard_from_query

        try:
            results = self.searcher.search(
                query,
                mode=search_mode,
                limit=top_k,
                expand=expand,
                graph_expand=graph_expand,
                label_filter=effective_label_filter,
                type_filter=contract_facets.get("type_filter"),
                status_filter=contract_facets.get("status_filter"),
                use_rerank=use_rerank,
                graph_parallel=graph_parallel,
            )

            # Build response from results
            if results:
                # Extract context for potential LLM answer
                context_parts = [r.get("text", "") for r in results[:top_k]]
                context = "\n---\n".join(context_parts)

                # Try LLM answer if available
                llm_ok = self.classifier._check_llm()
                if llm_ok and LITELLM_API_KEY:
                    answer = self._generate_answer(query, context, profile=profile)
                    if answer is None:
                        # LLM failed — fall back to formatted results
                        answer = self._format_results_as_answer(query, results)
                else:
                    # Fallback: show top results
                    answer = self._format_results_as_answer(query, results)
            else:
                answer = "I couldn't find any relevant messages in the conversation corpus."

            if hard_from_query and not label_filter:
                applied = (
                    " / ".join(hard_from_query)
                    if isinstance(query_label_filter, tuple)
                    else str(hard_from_query)
                )
                answer = "🔖 Label filter applied: [" + applied + "]\n\n" + answer

            routing_time = (time.perf_counter() - t0) * 1000
            logger.debug(
                f"Agent processed '{query[:30]}...' → intent={intent}, "
                f"{len(results)} results, total={routing_time:.1f}ms"
            )

            self._add_to_history(query, answer)
            self._persist_turn(query, answer, intent)
            return intent, answer, results

        except Exception as e:
            logger.error(f"Search failed in agent: {e}")
            response = f"Search encountered an error: {e}. Showing keyword-based results."
            # Retry with keyword mode if hybrid fails
            try:
                results = self.searcher.search(
                    query, mode="keyword", limit=top_k,
                    expand=False, graph_expand=0,
                    label_filter=effective_label_filter,
                    type_filter=contract_facets.get("type_filter"),
                    status_filter=contract_facets.get("status_filter"),
                    use_rerank=False,
                )
                if results:
                    answer = self._format_results_as_answer(query, results)
                    self._add_to_history(query, answer)
                    self._persist_turn(query, answer, intent)
                    return INTENT_SEARCH, answer, results
            except Exception:
                pass

            self._add_to_history(query, response)
            self._persist_turn(query, response, intent)
            return intent, response, []

    def _generate_greeting(self, user_query: str = "") -> str:
        """Generate a friendly, natural greeting via LLM, with fast fallback."""
        # Respect classifier's LLM availability to keep greeting fast when LLM is down
        if not self.classifier._check_llm():
            return self._GREETING_RESPONSE
        from apps.corpchat.search.intent_words import generate_greeting
        try:
            return generate_greeting(_llm_client, user_query, "en",
                                     fallback=self._GREETING_RESPONSE)
        except Exception as e:
            logger.debug(f"LLM greeting generation failed: {e}")
        return self._GREETING_RESPONSE


    @staticmethod
    def _has_filter_intent(query: str) -> bool:
        """查询是否带明确过滤意图词 (决定标签是硬过滤还是仅分组排序)。"""
        q = (query or "").lower()
        return any(kw in q for kw in FILTER_INTENT_KEYWORDS)

    def _extract_label_filter(self, query: str):
        """从查询中识别标签过滤意图, 返回 canonical label (str)、候选标签
        tuple 或 None。

        匹配规则 (与 intent_words 一致):
          - 别名按长度降序匹配, 优先更具体的短语 ("equipment quote" 先于 "quote");
          - 单个英文词用整词边界匹配 ("order" 不会命中 "border");
          - 中文别名/多词短语用子串匹配 (CJK 无词边界)。
        """
        q = (query or "").lower().strip()
        if not q:
            return None
        for alias in sorted(LABEL_ALIASES, key=len, reverse=True):
            target = LABEL_ALIASES[alias]
            if " " in alias or "-" in alias:
                if alias in q:
                    return target
            elif all(ord(c) < 128 for c in alias):
                if re.search(rf"(^|[^a-z]){re.escape(alias)}([^a-z]|$)", q):
                    return target
            else:
                if alias in q:
                    return target
        return None

    def _extract_contract_facets(self, query: str) -> Dict[str, Optional[str]]:
        q = (query or "").lower()
        facets = {"type_filter": None, "status_filter": None}
        for kw, val in CONTRACT_TYPE_ALIASES.items():
            if kw in q:
                facets["type_filter"] = val
                break
        for kw, val in CONTRACT_STATUS_ALIASES.items():
            if kw in q:
                facets["status_filter"] = val
                break
        return facets

    def _generate_answer(self, query: str, context: str,
                         profile: Optional[DispositionProfile] = None) -> Optional[str]:
        """Generate LLM answer (requires LiteLLM). profile: DispositionProfile (optional)."""
        try:
            system = (
                "You are a helpful assistant answering questions based on retrieved chat messages. "
                "Answer concisely in the same language as the query. "
                "If the context doesn't contain the answer, say so."
            )
            if profile is not None:
                system = profile.build_system_prompt(system)
            result = _llm_client.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"},
                ],
                temperature=0.3,
                max_tokens=300,
                timeout=10,
            )
            if result:
                return result
            return None
        except Exception as e:
            logger.warning(f"LLM answer generation failed: {e}")
            return None

    def _format_results_as_answer(self, query: str, results: List[Dict]) -> str:
        """Format search results as a readable answer (fallback when LLM is down).

        结果按 label 分组整理 (sort them out):
          - 组内按 score 降序;
          - 组之间按组内最高 score 降序;
          - 单一 label (通常是过滤后) 时不重复显示组头。
        """
        if not results:
            return "No relevant messages found."

        groups: Dict[str, List[Dict]] = {}
        for r in results:
            label = r.get("metadata", {}).get("label") or "unlabeled"
            groups.setdefault(label, []).append(r)

        for label in groups:
            groups[label].sort(key=lambda x: x.get("score", 0), reverse=True)
        ordered = sorted(
            groups.items(),
            key=lambda kv: kv[1][0].get("score", 0) if kv[1] else 0,
            reverse=True,
        )

        # 查询提到的标签组置顶 (display-only; 未带过滤意图词, 不影响召回,
        # 召回侧由 process() 的 _has_filter_intent 闸门决定)
        mentioned = self._extract_label_filter(query)
        if mentioned:
            mentioned_set = set(mentioned if isinstance(mentioned, tuple) else (mentioned,))
            if any(k in mentioned_set for k in groups):
                ordered = sorted(ordered, key=lambda kv: 0 if kv[0] in mentioned_set else 1)

        multi_group = len(ordered) > 1
        header = f"Found {len(results)} relevant messages"
        if multi_group:
            header += f" across {len(ordered)} labels"
        parts = [header + ":"]
        shown = 0
        for label, items in ordered:
            if multi_group:
                parts.append(f"\n### {label} ({len(items)})")
            for r in items:
                if shown >= 5:
                    break
                shown += 1
                meta = r.get("metadata", {})
                sender = meta.get("customer_name", meta.get("external_userid", "?"))
                text = r.get("text", "")[:200]
                if multi_group:
                    parts.append(f"\n{shown}. {sender} → {text}")
                else:
                    parts.append(f"\n{shown}. [{label}] {sender} → {text}")
            if shown >= 5:
                break
        return "\n".join(parts)

    def reset(self):
        """Clear conversation history."""
        self.chat_history = []


# ── Convenience: lazy-load agent with index ────────────────────────────────
_agent_instance: Optional[Agent] = None
_index_loaded: Optional[Any] = None


def load_agent(index_path: Optional[str] = None) -> Agent:
    """
    Load (or reuse cached) Agent with the search index.

    Uses module-level caching so repeated calls are fast.
    """
    global _agent_instance

    if _agent_instance is not None:
        return _agent_instance

    global _index_loaded
    if _index_loaded is None:
        _index_loaded = load_index(index_path or DEFAULT_INDEX_PATH)

    searcher = Searcher(_index_loaded)
    _agent_instance = Agent(searcher=searcher)
    return _agent_instance


def get_or_create_agent(searcher: Optional[Searcher] = None) -> Agent:
    """Get a fresh Agent (useful in tests for isolation)."""
    classifier = IntentClassifier()
    return Agent(searcher=searcher, classifier=classifier)
