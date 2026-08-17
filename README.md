# CorpChat RAG

企業微信對話智能搜尋與 RAG 系統。基於 txtai 混合搜尋 + LLM 查詢擴展 + 交叉編碼器重排序的搜尋框架。

## 功能特性

- 🔍 **高级搜尋**: 混合搜尋 (BM25 + 向量) + LLM 查詢擴展 + 加權 RRF 融合
- 🕸️ **圖增強搜尋**: 基於 txtai 圖的一跳鄰居擴展
- ⚡ **交叉编码器重排序**: 使用 cross-encoder/ms-marco-MiniLM-L-6-v2
- 💬 **Streamlit 互動介面**: 聯絡人、訊息、聊天記錄、語意搜尋一體化
- 🤖 **RAG 問答**: 基於 LiteLLM 的自然語言答案生成

## 快速开始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 配置資料庫

编辑 `core/config.py` 或设置环境变量:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=invoices
export DB_USER=ocr
export DB_PASSWORD=***REMOVED***
```

### 3. 生成測試資料

```bash
python apps/corpchat/gen_fake_msg.py
```

### 4. 建立搜尋索引

```bash
python apps/corpchat/search.py build --force
```

### 5. 啟動 Streamlit 應用

```bash
streamlit run apps/corpchat/app.py
```

## 搜尋 CLI

```bash
# 建立索引
python apps/corpchat/search.py build --force --graph-mode auto

# 搜尋（完整鏈路模式）
python apps/corpchat/search.py search "诈骗" --mode hybrid --expand --rerank

# 合成測試基準
python apps/corpchat/search.py synthetic-benchmark
```

## 项目结构

```
corpchat-rag/
├── apps/corpchat/
│   ├── app.py                # Streamlit 交互界面
│   ├── search.py             # 搜索核心引擎 (IndexBuilder, Searcher, Reranker, etc.)
│   ├── build_index.py        # 索引构建脚本
│   ├── gen_fake_msg.py       # 测试数据生成
│   ├── ingest.py             # 数据导入
│   └── search_index/         # 预构建的 txtai 索引
├── core/
│   ├── config.py             # 数据库与 API 配置
│   ├── db.py                 # 数据库连接
│   └── embedding.py          # 嵌入工具
├── requirements.txt
└── README.md