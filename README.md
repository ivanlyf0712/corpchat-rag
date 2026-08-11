# CorpChat RAG

企业微信对话智能搜索与 RAG 系统。基于 txtai 混合搜索 + LLM 查询扩展 + 交叉编码器重排序的搜索框架。

## 功能特性

- 🔍 **混合搜索**: BM25 + 向量 + LLM 查询扩展 + 加权 RRF 融合
- 🕸️ **图增强搜索**: 基于 txtai 图的一跳邻居扩展
- ⚡ **交叉编码器重排序**: 使用 BAAI/bge-reranker-base (中文/多语言)
- 💬 **Streamlit 交互界面**: 联系人、消息、聊天记录、语义搜索一体化
- 🤖 **RAG 问答**: 基于 LiteLLM 的自然语言答案生成

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据库

凭据只从环境变量读取 (不再硬编码)。创建 `.env` (已 gitignore):

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=invoices
export DB_USER=ocr
export DB_PASSWORD=***REMOVED***
export DEEPSEEK_API_KEY=sk-...   # 必需: docker-compose 缺失即启动失败
```

### 3. Docker compose 一键启动 (推荐)

```bash
make up        # 校验 .env → 创建外部卷 → docker compose up -d --build
make ps        # 查看服务状态
make logs      # 跟随日志
make down      # 停止 (保留数据卷)
```

### 4. 生成测试数据

```bash
python apps/corpchat/gen_fake_msg.py
```

### 5. 构建搜索索引

```bash
python apps/corpchat/search.py build --force
```

### 6. 启动 Streamlit 应用

```bash
streamlit run apps/corpchat/app.py
```

## 搜索 CLI

```bash
# 构建索引
python apps/corpchat/search.py build --force --graph-mode auto

# 搜索 (全链路模式)
python apps/corpchat/search.py search "诈骗" --mode hybrid --expand --rerank

# 合成测试基准
python apps/corpchat/search.py synthetic-benchmark
```

## 项目结构

```
corpchat-rag/
├── Makefile                   # make up/down/logs/test (env 校验 + 卷引导)
├── docker-compose.yml         # postgres + hindsight + corpchat 一键栈
├── apps/corpchat/
│   ├── app.py                 # Streamlit 交互界面
│   ├── process_window.py      # Process 窗口渲染 (UI 辅助)
│   ├── hf_offline.py          # HF 离线自动检测
│   ├── warmup.py              # 容器启动预热
│   ├── search.py              # CLI 入口 (薄封装)
│   ├── search/                # 搜索核心引擎包
│   │   ├── searcher.py        # Searcher (混合搜索 + RRF + 图扩展)
│   │   ├── index_builder.py   # IndexBuilder (分块 + 丰富化)
│   │   ├── query_expander.py  # QueryExpander (LLM 查询扩展)
│   │   ├── reranker.py        # Reranker (交叉编码器)
│   │   ├── agentic.py         # AgenticDecider (参数决策)
│   │   ├── litellm_client.py  # LiteLLMClient (统一 API 调用)
│   │   ├── hindsight_client.py# Hindsight 记忆适配器
│   │   ├── intent_words.py    # 意图词表与问候生成 (单一来源)
│   │   ├── persona.py         # CARA 人格层
│   │   ├── cross_table_agent.py # LangGraph 跨表 Agent
│   │   ├── config.py          # 配置与常量
│   │   └── utils.py           # 共享工具
│   ├── build_index.py         # 索引构建脚本
│   ├── gen_fake_msg.py        # 测试数据生成
│   └── search_index/          # 预构建的 txtai 索引
├── core/
│   ├── config.py              # CorpChat 配置 (DB, 凭据 env-only)
│   ├── db.py                  # 共享数据库连接
│   ├── corpchat_db.py         # CorpChat 持久化 (agent_memory/配置/图谱)
│   └── invoice_db.py          # OCR/发票遗留代码 (独立隔离)
├── requirements.txt
└── README.md