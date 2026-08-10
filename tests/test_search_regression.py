#!/usr/bin/env python3
"""
Regression tests for ticket 01 — Chinese-capable hybrid base.

Tests the Searcher.search() seam with a deterministic in-memory index
built from the conversation templates. Uses the production embedding
model (BAAI/bge-m3) so tests exercise exactly what runs in production.

Run:
    conda run -n ocr pytest tests/test_search_regression.py -v
"""
import json
import os
import sys
import tempfile

import pytest
import txtai

# Ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.search import (
    Searcher,
    QueryExpander,
    Reranker,
    DEFAULT_INDEX_PATH,
)
from apps.corpchat.gen_fake_msg import CONVERSATION_TEMPLATES, CONTACTS


# ── Fixture: deterministic in-memory index ──────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"


def _build_test_index(tmp_path):
    """Build a tiny txtai index from conversation templates."""
    docs = []
    for conv in CONVERSATION_TEMPLATES:
        label = conv["label"]
        init_name = CONTACTS[conv["initiator"]]["name"]
        resp_name = CONTACTS[conv["responder"]]["name"]
        for i, (speaker_idx, text) in enumerate(conv["turns"]):
            speaker_name = CONTACTS[speaker_idx]["name"]
            doc_id = f"{label}_{i}"
            title = f"{speaker_name} ({label})"
            # Match surface: content + curated title only
            match_text = f"{title}\n---\n{text}"
            # Structured metadata (filter/display/LLM-context only)
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
            }
            docs.append((doc_id, match_text, json.dumps(tags, default=str)))

    embeddings = txtai.Embeddings(
        {
            "path": EMBEDDING_MODEL,
            "content": True,
            "objects": True,
            "hybrid": True,
            "scoring": {"method": "bm25"},
        }
    )
    embeddings.index(docs)

    idx_path = os.path.join(tmp_path, "test_idx")
    embeddings.save(idx_path)
    return idx_path


@pytest.fixture(scope="module")
def test_index(tmp_path_factory):
    """Session-scoped deterministic index."""
    tmp = tmp_path_factory.mktemp("corpchat")
    idx = _build_test_index(tmp)
    yield idx


@pytest.fixture(scope="module")
def searcher(test_index):
    embeddings = txtai.Embeddings()
    embeddings.load(test_index)
    return Searcher(embeddings)


# ── Regression assertions ───────────────────────────────────────
def test_logistics_quotation_returns_relevant_message(searcher):
    """物流報價 方案 must return the logistics quotation message, not just any 方案 doc."""
    results = searcher.search("物流報價 方案", mode="hybrid", limit=5, expand=False, graph_expand=0, use_rerank=False)
    assert results, "No results returned"
    top_texts = [r.get("text", "") for r in results[:3]]
    # The top result must contain the logistics content, not merely 方案
    assert any("物流" in t and "報價" in t for t in top_texts), (
        f"Top results don't contain 物流+報價 context: {top_texts}"
    )


def test_investment_bond_bluechip_returns_keyword_message(searcher):
    """投資美國債券跟藍籌股 must return the investment message containing those keywords."""
    results = searcher.search("投資美國債券跟藍籌股", mode="hybrid", limit=5, expand=False, graph_expand=0, use_rerank=False)
    assert results, "No results returned"
    top_texts = [r.get("text", "") for r in results[:3]]
    assert any("債券" in t and "藍籌" in t for t in top_texts), (
        f"Top results don't contain keywords 債券+藍籌: {top_texts}"
    )


def test_bare_label_does_not_rank_all_label_docs(searcher):
    """Searching a bare label (product_inquiry) must not rank all product_inquiry docs."""
    results = searcher.search("product_inquiry", mode="hybrid", limit=10, expand=False, graph_expand=0, use_rerank=False)
    assert results, "No results returned"
    # Count how many top-10 results are product_inquiry
    labels = [r.get("metadata", {}).get("label") for r in results]
    pi_count = sum(1 for l in labels if l == "product_inquiry")
    # With the fix, label-only search should NOT rank every product_inquiry doc
    assert pi_count < len(results), (
        f"Bare label search ranked all {pi_count}/{len(results)} docs as product_inquiry"
    )


def test_label_filter_scopes_correctly(searcher):
    """Label filter investment_opportunity returns only investment_opportunity docs."""
    results = searcher.search("投資", mode="hybrid", limit=10, expand=False, graph_expand=0, use_rerank=False, label_filter="investment_opportunity")
    assert results, "No results returned"
    labels = [r.get("metadata", {}).get("label") for r in results]
    assert all(l == "investment_opportunity" for l in labels), (
        f"Label filter leaked other labels: {labels}"
    )


# ── Ticket: temporal & relationship recall (Hindsight advantage) ──
# 永久回归门新增: 时序窗口被尊重 / 结构邻居被召回 / 组合查询不破坏内容。
# 确定性图+时间索引: 会话按 conv_idx 偏移 send_time, 结构边来自真实会话关系。
from datetime import datetime, timedelta  # noqa: E402

try:  # noqa: E402 — 同 test_search_graph.py 的 GrandCypher workaround
    from txtai.graph.networkx import NetworkX
    if not hasattr(NetworkX, "original_isquery"):
        NetworkX.original_isquery = NetworkX.isquery
        NetworkX.isquery = lambda self, queries: False
except Exception:
    pass


def _ts(days_ago: int) -> str:
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S")


def _compute_structural_relationships(docs):
    """Five structural edge descriptors for tuple-format docs (per test_search_graph)."""
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


def _build_graph_time_index(tmp_path):
    """图+时间索引: send_time 按会话偏移 (conv_idx*3 天前), 结构图启用。"""
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
                "send_time": _ts(conv_idx * 3),
                "external_userid": speaker_name,
                "servicer_userid": resp_name if speaker_name == init_name else init_name,
                "msgid": doc_id,
                "origin": "3" if speaker_name == init_name else "5",
                "chunk_index": i,
                "open_kfid": open_kfid,
            }
            docs.append((doc_id, match_text, json.dumps(tags, default=str)))

    relationships = _compute_structural_relationships(docs)
    graph_docs = [
        (doc_id, {"text": text, "relationships": relationships.get(doc_id, [])}, tags_json)
        for doc_id, text, tags_json in docs
    ]
    embeddings = txtai.Embeddings(
        {"path": EMBEDDING_MODEL, "content": True, "objects": True, "hybrid": True,
         "scoring": {"method": "bm25"}, "graph": True, "columns": {"relationships": "relationships"}}
    )
    embeddings.index(graph_docs)
    idx_path = os.path.join(tmp_path, "graph_time_idx")
    embeddings.save(idx_path)
    return idx_path


@pytest.fixture(scope="module")
def graph_time_searcher(tmp_path_factory):
    idx = _build_graph_time_index(tmp_path_factory.mktemp("graph_time"))
    embeddings = txtai.Embeddings()
    embeddings.load(idx)
    return Searcher(embeddings)


def test_regression_temporal_window_respected(graph_time_searcher):
    """纯时序查询 '最近的消息' 只返回 7 天窗口内的文档。"""
    window_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    results = graph_time_searcher.search(
        "最近的消息", mode="hybrid", limit=10, expand=False, use_rerank=False
    )
    assert results, "纯时序查询应返回窗口内文档"
    for r in results:
        assert str(r["metadata"]["send_time"])[:10] >= window_start, (
            f"窗口外文档被召回: {r['id']}"
        )


def test_regression_relationship_neighbor_recalled(graph_time_searcher):
    """关系查询 + graph_parallel=True → 引入直接搜索没有的结构邻居。"""
    query = "物流報價 方案"
    off = graph_time_searcher.search(
        query, mode="hybrid", limit=10, expand=False, graph_expand=0, use_rerank=False
    )
    on = graph_time_searcher.search(
        query, mode="hybrid", limit=10, expand=False, graph_expand=0, use_rerank=False,
        graph_parallel=True,
    )
    assert off and on
    off_ids = {r["id"] for r in off}
    new_ids = [r["id"] for r in on if r["id"] not in off_ids]
    assert new_ids, "graph_parallel 应引入结构邻居"


def test_regression_combined_time_relationship(graph_time_searcher):
    """组合查询 '最近的物流報價': 时间窗口被尊重且内容不被破坏。"""
    window_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    results = graph_time_searcher.search(
        "最近的物流報價", mode="hybrid", limit=10, expand=False, use_rerank=False
    )
    assert results, "组合查询应返回窗口内结果"
    for r in results:
        assert str(r["metadata"]["send_time"])[:10] >= window_start, (
            f"窗口外文档被召回: {r['id']}"
        )
    labels = {r["metadata"]["label"] for r in results}
    assert "product_inquiry" in labels, "窗口内物流对话应被召回"