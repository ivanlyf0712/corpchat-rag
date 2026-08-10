"""
CorpChat Search Package
========================
Focused modules for the CorpChat RAG search engine.

Public interface (re-exported for backward compatibility):
  - IndexBuilder, Searcher, QueryExpander, Reranker, AgenticDecider
  - load_index, DEFAULT_INDEX_PATH, CONTACTS_INDEX_PATH and all config constants
  - CrossTableAgent, cross_table_chat (cross-table reasoning)
  - search_messages, search_contacts (LangChain tools)
"""

import os
from typing import Optional

import txtai

from .config import (
    CONTACTS_INDEX_PATH,
    DB_CONFIG,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_INDEX_PATH,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_TOP_N,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LITELLM_MODEL,
    LLM_KEYWORD_QUERY_WEIGHT,
    LLM_SEMANTIC_QUERY_WEIGHT,
    MAX_SEARCH_LIMIT,
    ORIGINAL_QUERY_WEIGHT,
    RRF_K_VALUE,
    _EMBED_MODEL,
    _METADATA_MARKER,
    logger,
)
from .index_builder import IndexBuilder
from .query_expander import QueryExpander
from .reranker import Reranker
from .searcher import Searcher
from .agentic import AgenticDecider
from .litellm_client import LiteLLMClient
from .router import SearchRouter
from .temporal import TimeExpressionParser, TimeWindow
from .persona import DispositionProfile
from .tools import CROSS_TABLE_TOOLS, search_messages, search_contacts
from .cross_table_agent import CrossTableAgent, cross_table_chat, is_cross_table_available
from .utils import _clean_text_from_enriched, _compute_structural_relationships, _segment


def load_index(index_path: Optional[str] = None) -> txtai.Embeddings:
    path = index_path or DEFAULT_INDEX_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"索引不存在: {path}。请先运行 python search.py build")
    embeddings = txtai.Embeddings()
    embeddings.load(path)
    return embeddings


def load_contacts_index(contacts_index_path: Optional[str] = None) -> txtai.Embeddings:
    """加载联系人索引。"""
    path = contacts_index_path or CONTACTS_INDEX_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"联系人索引不存在: {path}。请先运行 python search.py build-contacts")
    embeddings = txtai.Embeddings()
    embeddings.load(path)
    return embeddings


__all__ = [
    "AgenticDecider",
    "CONTACTS_INDEX_PATH",
    "CROSS_TABLE_TOOLS",
    "CrossTableAgent",
    "DB_CONFIG",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_HYBRID_ALPHA",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_RERANK_TOP_N",
    "DispositionProfile",
    "IndexBuilder",
    "LITELLM_API_KEY",
    "LITELLM_BASE_URL",
    "LITELLM_MODEL",
    "LLM_KEYWORD_QUERY_WEIGHT",
    "LLM_SEMANTIC_QUERY_WEIGHT",
    "LiteLLMClient",
    "MAX_SEARCH_LIMIT",
    "SearchRouter",
    "TimeExpressionParser",
    "TimeWindow",
    "ORIGINAL_QUERY_WEIGHT",
    "QueryExpander",
    "RRF_K_VALUE",
    "Reranker",
    "Searcher",
    "_EMBED_MODEL",
    "_METADATA_MARKER",
    "_clean_text_from_enriched",
    "_compute_structural_relationships",
    "_segment",
    "cross_table_chat",
    "is_cross_table_available",
    "load_contacts_index",
    "load_index",
    "logger",
    "search_contacts",
    "search_messages",
]