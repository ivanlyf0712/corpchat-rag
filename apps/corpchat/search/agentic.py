"""
CorpChat Search — Agentic Decider
==================================
Rule-first + LLM-fallback decision maker for search parameters
(mode, expand, graph_expand, use_rerank, graph_parallel).
"""

from typing import Any, Dict, Optional

from .config import LITELLM_MODEL, logger
from .litellm_client import LiteLLMClient


class AgenticDecider:
    def __init__(self, api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: str = LITELLM_MODEL):
        self._client = LiteLLMClient(api_base=api_base, api_key=api_key, model=model)
        self._mode_cache: Dict[str, str] = {}
        self._graph_cache: Dict[str, bool] = {}

    # 关系/实体查询关键词 → 激活图并行检索路 (结构邻居作为独立证据)
    RELATIONSHIP_KWS = (
        "跟誰", "跟谁", "和誰", "和谁", "還有誰", "还有谁",
        "這個人", "这个人", "哪個客戶", "哪个客户",
        "消息的人", "對方", "对方", "後來", "后来", "之後", "之后",
        "聊", "談", "对话", "對話",
    )

    def decide(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        q_len = len(query.split())
        decision = {
            "mode": "hybrid", "expand": True, "graph_expand": 0,
            "use_rerank": True, "graph_parallel": False,
        }
        question_kws = {"谁", "什么", "何时", "where", "when", "who", "哪个", "如何"}
        similarity_kws = {"类似", "相关", "similar", "related", "like"}
        if any(kw in q_lower for kw in question_kws):
            decision["mode"] = "keyword"; decision["expand"] = False
        elif any(kw in q_lower for kw in similarity_kws):
            decision["mode"] = "semantic"; decision["expand"] = True
        if q_len > 5 or any(c in q_lower for c in ["和", "以及", "对比", "比较", "vs"]):
            decision["graph_expand"] = 1; decision["use_rerank"] = True
        elif q_len <= 2:
            decision["use_rerank"] = False
        # 关系/实体查询 → 图并行路 (Hindsight graph-traversal evidence)
        if any(kw in q_lower for kw in self.RELATIONSHIP_KWS):
            decision["graph_parallel"] = True
        try:
            mode_from_llm = self._llm_decide_mode(query)
            if mode_from_llm:
                decision["mode"] = mode_from_llm
        except Exception:
            pass
        try:
            graph_from_llm = self._llm_decide_graph_parallel(query)
            if graph_from_llm is not None:
                decision["graph_parallel"] = graph_from_llm
        except Exception:
            pass
        return decision

    def _llm_decide_mode(self, query: str) -> Optional[str]:
        cache_key = query.lower()[:100]
        if cache_key in self._mode_cache:
            return self._mode_cache[cache_key]
        try:
            result = self._client.chat(
                [{"role": "user",
                  "content": f'For query "{query}", pick ONE: keyword, semantic, hybrid. Reply ONE word.'}],
                temperature=0,
                max_tokens=10,
                timeout=10,
            )
            choice = result.strip().lower()
            for mode in ["keyword", "semantic", "hybrid"]:
                if mode in choice:
                    self._mode_cache[cache_key] = mode
                    return mode
        except Exception:
            pass
        return None

    def _llm_decide_graph_parallel(self, query: str) -> Optional[bool]:
        cache_key = query.lower()[:100]
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]
        try:
            result = self._client.chat(
                [{"role": "user",
                  "content": (
                      f'For query "{query}", does answering require conversation/relationship '
                      "context (who talked to whom, later messages in a conversation, a specific "
                      "customer's interactions)? Reply ONE word: yes or no."
                  )}],
                temperature=0,
                max_tokens=10,
                timeout=10,
            )
            choice = result.strip().lower()
            if "yes" in choice:
                self._graph_cache[cache_key] = True
                return True
            if "no" in choice:
                self._graph_cache[cache_key] = False
                return False
        except Exception:
            pass
        return None