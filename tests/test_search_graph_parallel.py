#!/usr/bin/env python3
"""
Tests for the graph-parallel fusion path (hindsight-graph-parallel ticket 01):
structural graph traversal as an opt-in RRF fusion path (graph_parallel=True).

Default remains append-only graph expansion (ADR-0001 contract). When enabled,
`_graph_path_retrieve` returns a ranked neighbor list that participates in
weighted RRF fusion with GRAPH_RETRIEVAL_WEIGHT.

Uses the same deterministic graph-enabled index pattern as test_search_graph.py,
plus a FakeExpander (no LLM) so the expanded-query (Path B) flow runs
deterministically.

Run:
    conda run -n ocr pytest tests/test_search_graph_parallel.py -v
"""
import json
import os
import sys

import pytest
import txtai

# Same workaround as test_search_graph.py: txtai graph .isquery() requires
# GrandCypher, which is not installed here.
try:
    from txtai.graph.networkx import NetworkX
    if not hasattr(NetworkX, "original_isquery"):
        NetworkX.original_isquery = NetworkX.isquery
        NetworkX.isquery = lambda self, queries: False
except Exception:
    pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.search import Searcher
from apps.corpchat.gen_fake_msg import CONVERSATION_TEMPLATES, CONTACTS
from apps.corpchat.search.config import ORIGINAL_QUERY_WEIGHT

EMBEDDING_MODEL = "BAAI/bge-m3"


class FakeExpander:
    """Deterministic expander: no LLM calls in tests."""

    def __init__(self, mapping):
        self._mapping = mapping

    def expand(self, query, use_cache=True):
        return self._mapping.get(query, [(query, ORIGINAL_QUERY_WEIGHT)])


def _compute_structural_relationships(docs):
    """Compute the five structural edge descriptors for each chunk (tuple format)."""
    metas = {}
    for doc_id, _text, tags_json in docs:
        metas[doc_id] = json.loads(tags_json)

    relationships = {doc_id: [] for doc_id, _, _ in docs}
    doc_ids = list(metas.keys())

    for i, a_id in enumerate(doc_ids):
        a = metas[a_id]
        for b_id in doc_ids[i + 1:]:
            b = metas[b_id]
            rels = set()

            if a.get("open_kfid") and a["open_kfid"] == b.get("open_kfid"):
                rels.add("same_conversation")

            if a.get("open_kfid") and a["open_kfid"] == b.get("open_kfid"):
                if a["external_userid"] == b.get("servicer_userid") or \
                   b["external_userid"] == a.get("servicer_userid"):
                    rels.add("sender_receiver")

            if a.get("external_userid") and a["external_userid"] == b.get("external_userid"):
                rels.add("same_sender")

            if a.get("company") and a["company"] == b.get("company"):
                rels.add("same_company")

            if a.get("label") and a["label"] == b.get("label"):
                rels.add("same_label")

            for rel in rels:
                relationships[a_id].append({"id": b_id, "relation": rel})
                relationships[b_id].append({"id": a_id, "relation": rel})

    return relationships


# ── Fixture: deterministic in-memory graph-enabled index ────────
def _build_test_index(tmp_path):
    docs = []
    for conv_idx, conv in enumerate(CONVERSATION_TEMPLATES):
        label = conv["label"]
        init_name = CONTACTS[conv["initiator"]]["name"]
        resp_name = CONTACTS[conv["responder"]]["name"]
        open_kfid = f"kf_{label}_{conv_idx}"
        for i, (speaker_idx, text) in enumerate(conv["turns"]):
            speaker_name = CONTACTS[speaker_idx]["name"]
            doc_id = f"{label}_{i}"
            title = f"{speaker_name} ({label})"
            match_text = f"{title}\n---\n{text}"
            tags = {
                "label": label,
                "customer_name": speaker_name,
                "company": CONTACTS[speaker_idx]["company"],
                "send_time": "2026-01-01T00:00:00",
                "external_userid": speaker_name,
                "servicer_userid": resp_name if speaker_name == init_name else init_name,
                "msgid": doc_id,
                "origin": "3" if speaker_name == init_name else "5",
                "chunk_index": i,
                "open_kfid": open_kfid,
            }
            docs.append((doc_id, match_text, json.dumps(tags, default=str)))

    relationships = _compute_structural_relationships(docs)
    graph_docs = []
    for doc_id, match_text, tags_json in docs:
        graph_docs.append((
            doc_id,
            {"text": match_text, "relationships": relationships.get(doc_id, [])},
            tags_json,
        ))

    embeddings = txtai.Embeddings(
        {
            "path": EMBEDDING_MODEL,
            "content": True,
            "objects": True,
            "hybrid": True,
            "scoring": {"method": "bm25"},
            "graph": True,
            "columns": {"relationships": "relationships"},
        }
    )
    embeddings.index(graph_docs)

    idx_path = os.path.join(tmp_path, "test_idx")
    embeddings.save(idx_path)
    return idx_path


@pytest.fixture(scope="module")
def graph_index(tmp_path_factory):
    return _build_test_index(tmp_path_factory.mktemp("graph_parallel"))


@pytest.fixture(scope="module")
def searcher(graph_index):
    embeddings = txtai.Embeddings()
    embeddings.load(graph_index)
    return Searcher(embeddings)


# ── Tests ───────────────────────────────────────────────────────
def test_graph_path_retrieve_returns_relevant_neighbors(searcher):
    """图并行检索路返回与查询相关的结构邻居 (query-consistency gate)。"""
    query = "物流報價 方案"
    results = searcher._graph_path_retrieve(query, limit=10)
    assert results, "图并行检索应返回结构邻居"

    graph = searcher.embeddings.graph
    id_to_key = {}
    for key, attrs in graph.scan(data=True):
        id_to_key[attrs["id"]] = key

    # 每个返回的邻居都必须与某个相关种子存在 traversal-eligible 结构边
    for neighbor_id, _score in results:
        assert neighbor_id in id_to_key, f"邻居 {neighbor_id} 不在图中"
        key = id_to_key[neighbor_id]
        edges = graph.edges(key) or {}
        rels = {e.get("relation") for e in edges.values()}
        assert rels.intersection(
            {"same_conversation", "sender_receiver", "same_sender", "same_company"}
        ), f"{neighbor_id} 应通过 traversal-eligible 边连接"


def test_graph_parallel_default_off_preserves_base(searcher):
    """默认 (graph_parallel=False) 行为与无图路一致: base 检索不被图路影响。"""
    query = "物流報價 方案"
    fake = FakeExpander({query: [(query, ORIGINAL_QUERY_WEIGHT)]})
    s = Searcher(searcher.embeddings, expander=fake)
    results = s.search(query, mode="hybrid", limit=10, expand=True, graph_expand=0, use_rerank=False)
    assert results
    top_texts = [r["text"] for r in results[:5]]
    assert any("物流" in t and "報價" in t for t in top_texts), f"Base degraded: {top_texts}"


def test_graph_parallel_surfaces_graph_neighbors(searcher):
    """graph_parallel=True 把结构邻居带入 RRF 融合结果。"""
    query = "物流報價 方案"
    fake = FakeExpander({query: [(query, ORIGINAL_QUERY_WEIGHT)]})
    s = Searcher(searcher.embeddings, expander=fake)

    graph_ids = {doc_id for doc_id, _ in s._graph_path_retrieve(query, limit=10)}
    assert graph_ids, "前置条件: 图路应有邻居"

    on = s.search(query, mode="hybrid", limit=10, expand=True, graph_expand=0,
                  use_rerank=False, graph_parallel=True)
    assert on
    surfaced = [r for r in on if r["id"] in graph_ids]
    assert surfaced, "graph_parallel=True 应把图邻居带入结果"
    # 直接匹配的物流文档不被图路挤掉 (product_inquiry_0 是客户询问物流報價的种子)
    assert any(r["id"] == "product_inquiry_0" for r in on), (
        "直接匹配的物流文档不应丢失"
    )


def test_graph_parallel_query_consistency_gate(searcher):
    """query-consistency gate: 图路只返回查询相关的邻居 (relevance > 0)。

    结构性相连但内容无关的邻居 (relevance=0, 未进入查询 top-100) 不得 surfacing。
    """
    query = "物流報價 方案"
    results = searcher._graph_path_retrieve(query, limit=10)
    assert results
    query_scores = searcher._graph_query_scores(query)
    for neighbor_id, _score in results:
        assert query_scores.get(neighbor_id, 0.0) > 0, (
            f"{neighbor_id} 未通过 query-consistency gate (relevance=0)"
        )
