"""
CorpChat Search — Reranker
===========================
Cross-encoder reranking for improved result relevance.
"""

from typing import Dict, List, Optional

from .config import DEFAULT_RERANKER_MODEL, DEFAULT_RERANK_TOP_N, logger


class Reranker:
    """交叉编码器重排序, 仅对前 rerank_top_n 个结果重排。"""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL,
                 top_n: int = DEFAULT_RERANK_TOP_N):
        self.enabled = False
        self.model = None
        self.model_name = model_name
        self.top_n = top_n
        try:
            from sentence_transformers import CrossEncoder
            self.enabled = True
        except ImportError:
            logger.warning("sentence_transformers 未安装, 重排序已禁用")

    def _ensure_model(self) -> None:
        if self.model is None and self.enabled:
            from sentence_transformers import CrossEncoder
            logger.info(f"加载交叉编码器: {self.model_name}")
            self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        if not self.enabled or not results:
            return results
        if self.model is None:
            try:
                self._ensure_model()
            except Exception as e:
                logger.warning(f"重排序模型加载失败: {e}")
                return results

        if len(results) <= self.top_n:
            to_rerank = results
            rest = []
        else:
            to_rerank = results[:self.top_n]
            rest = results[self.top_n:]

        pairs = [(query, item.get("text", "")) for item in to_rerank]
        try:
            scores = self.model.predict(pairs)
            for i, score in enumerate(scores):
                to_rerank[i]["rerank_score"] = float(score)
                # Keep original score (RRF or hybrid) for display; use rerank_score for sorting only
            to_rerank.sort(key=lambda x: float(x.get("rerank_score", 0)), reverse=True)
        except Exception as e:
            logger.warning(f"重排序失败: {e}")

        return to_rerank + rest