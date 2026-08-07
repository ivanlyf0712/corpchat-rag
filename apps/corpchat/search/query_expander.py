"""
CorpChat Search — Query Expander
=================================
Uses LLM to generate semantic rephrases and keyword expansions for
improved recall in hybrid search.
"""

from typing import Dict, List, Optional, Tuple

from .config import (
    LITELLM_MODEL,
    LLM_KEYWORD_QUERY_WEIGHT,
    LLM_SEMANTIC_QUERY_WEIGHT,
    ORIGINAL_QUERY_WEIGHT,
    logger,
)
from .litellm_client import LiteLLMClient


class QueryExpander:
    """使用 LLM 生成语义重写和关键词扩展查询。"""

    def __init__(self, api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: str = LITELLM_MODEL):
        self._client = LiteLLMClient(api_base=api_base, api_key=api_key, model=model)
        self._cache: Dict[str, List[Tuple[str, float]]] = {}

    def _call_llm(self, messages: List[Dict], max_tokens: int = 200) -> str:
        return self._client.chat(messages, temperature=0.1, max_tokens=max_tokens)

    def _semantic_rephrase(self, query: str) -> Optional[str]:
        system_msg = (
            "You reformulate user queries into standalone semantic search queries. "
            "Output ONLY the reformulated query, no extra text."
        )
        user_msg = (
            f"Rewrite this query into a standalone semantic search query. "
            f"In most cases keep it identical. Only add missing context or remove "
            f"non-search instructions.\n\nQuery: {query}\n\nSemantic query:"
        )
        result = self._call_llm([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])
        if result and result != query:
            return result
        return None

    def _keyword_expand(self, query: str) -> List[str]:
        system_msg = (
            "You reformulate user queries into keyword-only queries. "
            "Output ONLY the keywords, one set per line (max 3 lines)."
        )
        user_msg = (
            f"Extract up to 3 keyword-only search queries from the user query. "
            f"Each line should contain one set of keywords.\n\nQuery: {query}\n\nKeywords:"
        )
        result = self._call_llm([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])
        if not result:
            return []
        keywords = [
            line.strip() for line in result.split("\n")
            if line.strip() and len(line.strip()) > 1
        ]
        return keywords[:3]

    def expand(self, query: str, use_cache: bool = True) -> List[Tuple[str, float]]:
        cache_key = query[:100]
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        results: List[Tuple[str, float]] = [(query, ORIGINAL_QUERY_WEIGHT)]

        try:
            semantic = self._semantic_rephrase(query)
            if semantic and semantic.lower() != query.lower():
                results.append((semantic, LLM_SEMANTIC_QUERY_WEIGHT))
        except Exception as e:
            logger.debug(f"语义重写失败: {e}")

        try:
            kw_queries = self._keyword_expand(query)
            for kw in kw_queries:
                existing = {q.lower() for q, _ in results}
                if kw.lower() not in existing:
                    results.append((kw, LLM_KEYWORD_QUERY_WEIGHT))
        except Exception as e:
            logger.debug(f"关键词扩展失败: {e}")

        self._cache[cache_key] = results
        return results