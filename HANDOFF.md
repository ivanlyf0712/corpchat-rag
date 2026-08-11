# CorpChat ↔ Hindsight 集成 — 会话交接说明

## 一、系统现状（一切可运行）

**三个服务通过 docker compose 运行中（`cd ~/Desktop/corpchat-rag && docker compose ps`）**：
- **corpchat-rag** (`:8501`) — Streamlit RAG UI + LangGraph agent
- **hindsight** (`:8888` API / `:9999` Web UI) — 记忆层，DeepSeek LLM，本地 bge embeddings
- **postgres** (`:5432`) — pgvector，140 消息 + 30 联系人

**技术栈**：Python 3.10（conda env `ocr`），txtai 9.12 + LangGraph + DeepSeek API（key 在 `.env`，LITELLM_* 指向 api.deepseek.com）。

## 二、已完成的架构（重要，先读这些再动代码）

1. **Agent 架构**（`apps/corpchat/search/cross_table_agent.py`）：
   - `_LiteLLMWrapper` 是真实 tool calling（`bind_tools` → DeepSeek 发 schema → 解析 tool_calls）
   - `process()` 走 LangGraph ReAct 循环；`_quick_respond` 规则快路拦问候（decision 8）
   - 工具集：`search_messages(query, sender, receiver, limit)` + `search_contacts(query)` + `search_conversation_partners(person)`（3 个，`CROSS_TABLE_TOOLS`）
2. **方向性**（核心）：`external_userid` ≠ 发送方；sender/receiver 用 `origin` 双分支谓词过滤（origin=3 客户发言，5 客服发言）
3. **Hindsight 桥接**（`apps/corpchat/search/hindsight_client.py` + `persona.py`）：
   - `DispositionProfile.from_hindsight()` / `sync_to_hindsight()` — bank disposition (1-5) ↔ CARA 人格 (0-1)
   - UI 设置面板的 CARA 是**只读镜像**（Hindsight 驱动，滑杆不可拖，有"在 Hindsight 调整"链接 + 刷新按钮）
   - 搜索后自动 retain 到 Hindsight（`app.py::_retain_search_to_hindsight`）
4. **多轮上下文**：
   - 会话内：`process(history=[...])` 注入前几轮摘要（A4-i，让 LLM 解析"her"→李雅婷）
   - 跨会话：`process()` 内无条件 `hindsight_client.recall()` 注入记忆（**这就是当前要改的点**）
5. **持久化**：`core/config.py` DB_CONFIG 读环境变量（compose 传 DB_HOST=postgres）；`app.py::_load_persisted_config` 从 DB 恢复 agent 配置（含 hindsight_bank 默认 test-bank）
6. **线程安全**（已修，重要）：txtai SQLite 非线程安全（源码注明"Thread locking must be handled externally"），所有 txtai 访问（`tools.py` + `searcher.py`）已用模块级 `_TXTAI_LOCK`（`threading.RLock`）串行化，索引加载也在锁内。**不要再加自己的锁，复用 `_TXTAI_LOCK`**

## 三、当前卡点（新会话要接手的事）

**用户要求：按需调用 Hindsight——只有查询确实需要历史记忆时才 recall，平时不调。**

现状（`cross_table_agent.py` process() 约 507-525 行）：只要 `self.hindsight_bank` 配置了，**每次查询都无条件 `hs_recall()` 注入**。问题：污染上下文 + 增加延迟 + 可能让 agent 过度依赖记忆而少调真实工具。

**我（上一会话）正在拷问用户，但还没得到答案——请新会话继续**：
用户要的"查询需要记忆"的判定信号，我提出了 5 个候选（指代词检测 / 有历史才查 / 向量预检 / LLM 判定 / 全不注入），并指出了各自的缺陷：
- 指代词会漏掉"客户喜欢什么沟通方式"（无指代词但需跨会话记忆），又误触发"她的邮箱"（会话内指代，该走 history 而非 Hindsight）
- 有历史才查会漏"记得上次说的报价吗"（首次会话无本地历史但需 Hindsight）
- 向量预检 = 已经调了 recall，没省
- LLM 判定 = 多一次 LLM 调用，比 recall 还慢
- 全不注入 = Hindsight 记忆形同虚设

**待用户拍板判定信号后**：实现"按需 recall"——在 `process()` 里加 gate，命中判定才调 `hs_recall`，否则跳过。改动应该集中在 `cross_table_agent.py::process()`。

## 四、约束与已验证事实

- 202 个 pytest 全过：`/Users/ivanlee/miniconda3/envs/ocr/bin/python -m pytest tests/ -q`（改完代码必须跑）
- 容器重建：`docker compose build corpchat && docker compose up -d corpchat`
- **不要用本地 `streamlit run`**（系统 streamlit 是 Python 3.13，与项目不匹配）；验证用 docker compose 或容器内 `docker exec -e PYTHONPATH=/app corpchat-rag python ...`
- Hindsight API 实测正常：retain/recall/实体图/read disposition 都验证过
- 项目根：`/Users/ivanlee/Desktop/corpchat-rag`；conda env：`ocr`
