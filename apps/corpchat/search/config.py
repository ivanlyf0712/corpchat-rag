"""
CorpChat Search — Configuration & Constants
============================================
Centralizes all configuration values, constants, and environment variables
used across the search package. Keeps the rest of the package free of
environment-dependent setup.
"""

import os
import sys
import logging

# ── 环境变量 (.env) ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))
except ImportError:
    pass

# ── 路径 & 配置 ──────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from core.config import DB_CONFIG
except ImportError:
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "dbname": os.getenv("DB_NAME", "invoices"),
        "user": os.getenv("DB_USER", "ocr"),
        "password": os.getenv("DB_PASSWORD", "***REMOVED***"),
    }

# ── 日志 ─────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("corpchat-search")

# ── 中文分词 (jieba) ─────────────────────────────────────────────
try:
    import jieba
    jieba.setLogLevel(20)  # silence jieba's build-dict logging
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

# ── 嵌入模型: 本地缓存优先 (默认 bge-m3 — 中文检索能力) ──────────
_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
_LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, "models", "bge-m3")
if os.path.isdir(_LOCAL_MODEL_PATH):
    _EMBED_MODEL = _LOCAL_MODEL_PATH

# ── 索引路径 ─────────────────────────────────────────────────────
DEFAULT_INDEX_PATH = os.getenv(
    "INDEX_PATH",
    os.path.join(os.path.dirname(__file__), "..", "search_index"),
)
CONTACTS_INDEX_PATH = os.getenv(
    "CONTACTS_INDEX_PATH",
    os.path.join(os.path.dirname(__file__), "..", "contacts_index"),
)

# ── 分块参数 (§2.2) ─────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))

# ── 搜索参数 (§2.5, §2.8) ────────────────────────────────────────
DEFAULT_HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))
RRF_K_VALUE = 50
MAX_SEARCH_LIMIT = 100

# ── 重排序模型: 中文能力 (BAAI/bge-reranker-base) ────────────────
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
DEFAULT_RERANK_TOP_N = 20

# ── 查询权重 (§2.7 的 constants.py) ──────────────────────────────
ORIGINAL_QUERY_WEIGHT = 0.5
LLM_SEMANTIC_QUERY_WEIGHT = 1.3
LLM_KEYWORD_QUERY_WEIGHT = 1.0

# ── 时序检索 (Hindsight temporal-retrieval component) ───────────
# 检测到时间窗口时放大检索量, 避免窗口过滤饿死结果 (放大 limit + 后置过滤)。
TEMPORAL_LIMIT_SCALE = float(os.getenv("TEMPORAL_LIMIT_SCALE", "5.0"))
# bare "最近/近" 的默认窗口 (天)。
TEMPORAL_DEFAULT_WINDOW_DAYS = int(os.getenv("TEMPORAL_DEFAULT_WINDOW_DAYS", "7"))

# ── 图并行检索 (Hindsight graph-traversal path, 默认关闭) ───────
# 图遍历作为独立检索路参与 RRF 融合时的权重 (opt-in: search(graph_parallel=True))。
# 默认关闭, 保留 append-only 图扩展 (ADR-0001) 为默认行为。
GRAPH_RETRIEVAL_WEIGHT = float(os.getenv("GRAPH_RETRIEVAL_WEIGHT", "0.8"))
GRAPH_PARALLEL_HOP_DISCOUNT = float(os.getenv("GRAPH_PARALLEL_HOP_DISCOUNT", "0.8"))
GRAPH_PARALLEL_SEED_LIMIT = int(os.getenv("GRAPH_PARALLEL_SEED_LIMIT", "10"))

# ── LiteLLM 配置 (密钥必须从环境变量提供, 不硬编码) ──────────────
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://your-litellm-proxy.example.com/")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "dseek-v4-flash")

# ── DeepSeek 备用 LLM (当主 LLM 不可用时回退) ────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


# ── 富文本 Metadata 格式标记 (已弃用 — 元数据现存于 sections.tags 列) ──
_METADATA_MARKER = "\n---\nMetadata: "