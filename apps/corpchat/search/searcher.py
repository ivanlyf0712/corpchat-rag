"""
CorpChat Search — Searcher
===========================
Multi-mode search engine: keyword / semantic / hybrid + graph expansion
+ reranking. Implements weighted RRF fusion and structural graph traversal.
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import txtai

from .config import (
    MAX_SEARCH_LIMIT,
    ORIGINAL_QUERY_WEIGHT,
    RRF_K_VALUE,
    logger,
)
from .query_expander import QueryExpander
from .reranker import Reranker
from .utils import _segment


class Searcher:
    """
    多模式搜索器: keyword / semantic / hybrid + 图增强 + 重排序。

    可直接被 app.py 导入使用:
      from apps.corpchat.search import Searcher, load_index
      searcher = Searcher(load_index())
    """

    def __init__(self, embeddings: txtai.Embeddings,
                 expander: Optional[QueryExpander] = None,
                 reranker: Optional[Reranker] = None):
        self.embeddings = embeddings
        self.expander = expander
        self.reranker = reranker

    # ── 加权 RRF 融合 (§2.8) ─────────────────────────────────
    @staticmethod
    def _weighted_rrf_fusion(
        all_results: List[Tuple[List[Tuple[str, float]], float]],
        k: int = RRF_K_VALUE
    ) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = defaultdict(float)
        source_rank: Dict[str, int] = {}
        source_idx: Dict[str, int] = {}

        for q_idx, (result_list, weight) in enumerate(all_results):
            for rank, (doc_id, _) in enumerate(result_list, start=1):
                if not doc_id:
                    continue
                rrf_score = weight / (k + rank)
                scores[doc_id] += rrf_score
                if doc_id not in source_rank:
                    source_rank[doc_id] = rank
                    source_idx[doc_id] = q_idx

        sorted_ids = sorted(
            scores.keys(),
            key=lambda did: (-scores[did], source_rank.get(did, 999), source_idx.get(did, 999))
        )
        return [(did, scores[did]) for did in sorted_ids]

    # ── 多路检索组装 (RRF 融合输入) ─────────────────────────────
    def _retrieve_parallel(
        self,
        query: str,
        weights: Optional[Tuple[float, float]],
        limit: int,
        expand: bool = True,
    ) -> List[Tuple[List[Tuple[str, float]], float]]:
        """
        组装加权 RRF 融合输入: 每条扩展查询一路结果。

        当前唯一的路来源是 LLM 查询扩展:
          原始查询 (ORIGINAL_QUERY_WEIGHT)
          语义重写 (LLM_SEMANTIC_QUERY_WEIGHT)
          关键词扩展 (LLM_KEYWORD_QUERY_WEIGHT)
        每条路 = (结果列表, 权重)。未来新增检索路 (时序 / 图并行) 在此追加
        一个 (结果列表, 权重) 条目即可, 无需修改调用方。各路相互独立,
        未来可并行执行。
        """
        queries_with_weights: List[Tuple[str, float]] = [(query, ORIGINAL_QUERY_WEIGHT)]
        try:
            if expand and self.expander:
                queries_with_weights = self.expander.expand(query)
        except Exception as e:
            logger.warning(f"查询扩展失败: {e}")

        all_results: List[Tuple[List[Tuple[str, float]], float]] = []
        for q, q_weight in queries_with_weights:
            raw = self.embeddings.search(
                _segment(q), limit=min(limit * 3, MAX_SEARCH_LIMIT), weights=weights
            )
            result_list: List[Tuple[str, float]] = []
            for item in raw:
                parsed = self._parse_txtai_result(item)
                if parsed:
                    result_list.append((parsed["id"], parsed["score"]))
            all_results.append((result_list, q_weight))

        return all_results

    # ── 图扩展 (纯结构, 直接 backend API) ──────────────────────
    def _graph_expand(self, results: List[Dict], max_expand: int = 3,
                       hop_discount: float = 0.8, limit: int = 20,
                       query: str = "", label_filter: Optional[str] = None,
                       date_from: Optional[str] = None, date_to: Optional[str] = None) -> List[Dict]:
        """
        从 base 结果出发, 遍历 4 种 traversal-eligible 结构边,
        将邻居追加到 base 结果下方 (不重排 base).

        score = parent_score × hop_discount × neighbor_query_relevance
        其中 neighbor_query_relevance 通过 already-loaded 索引对邻居文本做一次
        轻量 hybrid search 获得, 实现 query-consistency gate.
        """
        graph = self.embeddings.graph
        if not graph or not results:
            return results

        # Build doc-id -> node-key map once
        id_to_key = {}
        for key, attrs in graph.scan(data=True):
            id_to_key[attrs["id"]] = key

        # Only traverse these 4 relation types; same_label is recorded but never traversed
        TRAVERSAL_RELATIONS = {
            "same_conversation", "sender_receiver",
            "same_sender", "same_company",
        }

        # Query-consistency gate: run the query search ONCE, build id -> score map.
        # Avoids an N+1 full search per neighbor.
        query_scores: Dict[str, float] = {}
        if query:
            try:
                q_raw = self.embeddings.search(_segment(query), limit=MAX_SEARCH_LIMIT)
                for item in q_raw:
                    parsed = self._parse_txtai_result(item)
                    if parsed:
                        query_scores[parsed["id"]] = parsed.get("score", 0.0)
            except Exception:
                query_scores = {}

        def _passes_filters(doc: Dict) -> bool:
            meta = doc.get("metadata", {})
            if label_filter and meta.get("label") != label_filter:
                return False
            send_time = meta.get("send_time", "")
            if date_from and send_time and str(send_time) < date_from:
                return False
            if date_to and send_time and str(send_time) > date_to:
                return False
            return True

        expanded_ids = {r["id"] for r in results}
        expanded = list(results)

        seeds = results[:min(max_expand, len(results))]
        for r in seeds:
            seed_id = r["id"]
            seed_key = id_to_key.get(seed_id)
            if seed_key is None:
                continue

            neighbors = graph.edges(seed_key)
            if not neighbors:
                continue

            for neighbor_key, edge_attrs in neighbors.items():
                relation = edge_attrs.get("relation", "")
                if relation not in TRAVERSAL_RELATIONS:
                    continue

                # Resolve neighbor doc id from node attributes
                neighbor_attrs = graph.node(neighbor_key)
                if not neighbor_attrs:
                    continue
                neighbor_id = neighbor_attrs.get("id")
                if not neighbor_id or neighbor_id in expanded_ids:
                    continue

                # Query-consistency gate: look up the precomputed relevance score.
                neighbor_query_relevance = query_scores.get(neighbor_id, 0.0)

                final_score = r.get("score", 0.0) * hop_discount * neighbor_query_relevance
                if final_score <= 0.0:
                    # Irrelevant neighbor: balanced out, do not surface
                    continue

                neighbor_doc = self._fetch_one_doc(neighbor_id)
                if not neighbor_doc:
                    continue

                # Re-apply label/date filters
                if not _passes_filters(neighbor_doc):
                    continue

                expanded_ids.add(neighbor_id)
                expanded.append({
                    "id": neighbor_id,
                    "text": neighbor_doc.get("text", ""),
                    "score": final_score,
                    "metadata": {
                        **neighbor_doc.get("metadata", {}),
                        "_graph_relation": relation,
                        "_from_node": seed_id[:30],
                    },
                })

        # Append-only: base order preserved; expanded docs sorted below by score
        base_part = [d for d in expanded if not d.get("metadata", {}).get("_graph_relation")]
        extra_part = [d for d in expanded if d.get("metadata", {}).get("_graph_relation")]
        extra_part.sort(key=lambda x: x.get("score", 0), reverse=True)
        return (base_part + extra_part)[:limit]

    # ── 从 txtai 获取单个文档并提取 metadata ───────────────
    @staticmethod
    def _parse_txtai_result(item: Any) -> Optional[Dict]:
        """
        将 txtai 搜索结果统一解析为 {id, text, score, metadata} 格式。

        txtai 返回格式:
          - dict: {id, text, score, tags(optional)}
          - tuple: (id, text, tags_json, score)

        注意: txtai 在 content=True 且 objects=True 的配置下, search() 返回的
        dict 中不包含 tags 字段。metadata 通过 _fetch_one_doc 从 sections.tags
        列按 id 查询获取 (见 _fetch_one_doc)。
        """
        doc_id = ""
        text = ""
        score = 0.0

        if isinstance(item, dict):
            doc_id = item.get("id", "")
            text = item.get("text", "")
            score = item.get("score", 0.0)
        elif isinstance(item, tuple) and len(item) >= 4:
            doc_id = item[0]
            text = item[1]
            score = item[3]
        else:
            return None

        if not doc_id:
            return None

        return {
            "id": doc_id,
            "text": text,
            "score": score,
            "metadata": {},
        }

    def _fetch_one_doc(self, doc_id: str) -> Optional[Dict]:
        """
        通过 doc_id 从索引的 sections 表取出文档文本与结构化 tags。

        修复: 旧实现用 embeddings.search(f"id:{doc_id}") 做文本搜索,
        那是一次错误的 BM25 查询, 会返回错误文档。这里改为按 id 直接
        查询 SQLite sections 表, 同时取回 tags 元数据。
        """
        try:
            db = self.embeddings.database
            if db is None:
                return None
            conn = db.connection
            cur = conn.cursor()
            cur.execute("SELECT text, tags FROM sections WHERE id = ?", (doc_id,))
            row = cur.fetchone()
            if not row:
                return None
            text, tags_json = row
            metadata = {}
            if tags_json:
                try:
                    metadata = json.loads(tags_json) if isinstance(tags_json, str) else dict(tags_json)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            return {
                "id": doc_id,
                "text": text,
                "score": 0.0,
                "metadata": metadata,
            }
        except Exception as e:
            logger.debug(f"按 id 获取文档失败 {doc_id}: {e}")
            return None

    # ── 搜索主入口 ──────────────────────────────────────────
    def search(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 10,
        expand: bool = True,
        graph_expand: int = 0,
        label_filter: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        use_rerank: bool = True,
    ) -> List[Dict]:
        """
        执行搜索 (默认启用全链路搜索)。

        全链路 = LLM 查询扩展 + 混合搜索 + RRF 融合 + 交叉编码器重排序。

        当 expand=True 且 self.expander 可用时:
          - 生成语义重写 + 关键词扩展查询
          - 每条查询独立执行 txtai hybrid search
          - 加权 RRF 融合所有结果
          - Reranker 对 RRF 融合后的 top-N 重排序
        返回 RRF 分数 (小数值)。

        当 expand=False 或 expander 不可用时:
          - 直接执行 txtai hybrid search
          - 分数 0~1 (原生向量+BM25)

        use_rerank=True 且 reranker 可用时:
          - 对最终 top-20 结果用交叉编码器重排序
        """
        weight_map = {
            "keyword": (0.0, 1.0),
            "semantic": (1.0, 0.0),
            "hybrid": None,
        }
        weights = weight_map.get(mode, None)

        def _filter(item: Dict) -> bool:
            meta = item.get("metadata", {})
            if label_filter and meta.get("label") != label_filter:
                return False
            send_time = meta.get("send_time", "")
            if date_from and send_time and str(send_time) < date_from:
                return False
            if date_to and send_time and str(send_time) > date_to:
                return False
            return True

        # 中文分词: 查询与索引使用同一 jieba 分词, 保证 BM25 匹配一致
        segmented_query = _segment(query)

        # ── 路径 A: 直接 txtai 搜索 ──
        if not expand or not self.expander:
            raw = self.embeddings.search(segmented_query, limit=min(limit * 3, MAX_SEARCH_LIMIT), weights=weights)
            output = []
            for item in raw:
                parsed = self._parse_txtai_result(item)
                if parsed:
                    # 按 id 取回结构化 tags 元数据 (过滤/展示用)
                    doc = self._fetch_one_doc(parsed["id"])
                    if doc:
                        parsed["metadata"] = doc["metadata"]
                    if _filter(parsed):
                        output.append(parsed)

            if graph_expand > 0 and self.embeddings.graph:
                output = self._graph_expand(
                    output[:limit], max_expand=3, limit=limit * 2,
                    query=query, label_filter=label_filter,
                    date_from=date_from, date_to=date_to,
                )
            if use_rerank and self.reranker and self.reranker.enabled:
                output = self.reranker.rerank(query, output)
            # _graph_expand truncates to limit*2 so graph hits can surface below base
            return output

        # ── 路径 B: 多查询扩展 + RRF ──
        all_results = self._retrieve_parallel(query, weights, limit, expand)
        fused = self._weighted_rrf_fusion(all_results)
        output: List[Dict] = []
        seen_ids = set()
        for doc_id, _ in fused:
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            doc = self._fetch_one_doc(doc_id)
            if doc and _filter(doc):
                output.append(doc)

        if graph_expand > 0 and self.embeddings.graph:
            output = self._graph_expand(
                output[:limit], max_expand=3, limit=limit * 2,
                query=query, label_filter=label_filter,
                date_from=date_from, date_to=date_to,
            )
        if use_rerank and self.reranker and self.reranker.enabled:
            output = self.reranker.rerank(query, output)

        # _graph_expand truncates to limit*2 so graph hits can surface below base
        return output

    # ── 图查询 ──────────────────────────────────────────────
    def graph_query(self, cypher: str, limit: int = 20) -> List[Dict]:
        if not self.embeddings.graph:
            raise RuntimeError("图未启用")
        results = self.embeddings.graph.search(cypher)
        output = []
        for i, row in enumerate(results[:limit]):
            item = {"row": i + 1}
            if isinstance(row, (tuple, list)):
                for j, val in enumerate(row):
                    item[f"col_{j}"] = str(val)[:80]
            else:
                item["result"] = str(row)[:80]
            output.append(item)
        return output