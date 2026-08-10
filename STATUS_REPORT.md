# CorpChat 状态报告 (Status Report)

> 生成时间: 2026-08-10  |  分支: `feature/hindsight-multipath-rrf-skeleton`  |  提交: `f8c92d1`
> 范围: `corpchat-rag/` 全库（代码、数据库、索引、测试、配置；未含 `oa-rag`）

---

## 1. 执行摘要

CorpChat 是一个基于 txtai 混合检索（BM25 + bge-m3 向量 + 加权 RRF + 交叉编码器重排）+ LiteLLM/LangChain Agent 的企业微信消息/联系人 RAG 系统。当前处于 **Hindsight Memory 整合的中后期**：时序检索、图并行检索路、CARA 个性层、统一配置面板、记忆图谱均已落地并通过 **202 项测试**；数据库与索引数据**干净完整**（140 条消息、30 个联系人、无空值、无孤儿发送者）。**主要风险**是：(1) 主 LLM 为本地小模型 `qwen2.5:1.5b`，在多跳总结上易幻觉（已通过确定性联系人回答 + 反幻觉提示 + SQL 结构化检索缓解）；(2) 冷启动延迟（索引/模型/LLM 首次加载约 10–15s）；(3) 技术债集中在密钥硬编码、索引仅支持全量重建、大文件、无结构化日志/监控。

---

## 2. 数据模型与索引

### 2.1 表结构（PostgreSQL，`dbname=invoices`）

**`messages`（140 行）**

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | integer | 主键 |
| `msgid` | varchar | 消息 ID（如 `msg_kf_product_inquiry_0_0000`） |
| `open_kfid` | varchar | 会话/群 ID（对话维度） |
| `external_userid` | varchar | 客户发送者 userid（如 `user_陳志明_johnsonj`） |
| `servicer_userid` | varchar | 客服/服务方 userid |
| `send_time` | timestamptz | 发送时间 |
| `origin` | integer | 3=客户发言 / 5=客服发言 |
| `msgtype` | varchar | 消息类型（均非空） |
| `content` | text | 消息正文（无空值） |
| `raw_json` | jsonb | 原始消息 JSON |
| `embedding` | USER-DEFINED | 向量（pgvector，未在应用检索中使用） |
| `label` | varchar | 会话标签（31 种，如 `quotation_request`, `fraud` 类） |
| `created_at` | timestamp | 创建时间 |

**`contacts`（30 行）**

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | integer | 主键 |
| `full_name` | varchar | 姓名（无空值） |
| `job_title` / `company` / `phone` / `email` | varchar | 职位/公司/电话/邮箱（无空值） |
| `website` / `address` | varchar/text | 网站/地址（**未建索引，未用于检索**） |
| `userid` | varchar | userid（与 `messages.external_userid` 关联） |
| `embedding` / `raw_text` / `source_file` / `created_at` | - | 冗余/来源字段 |

**关联**: `messages.external_userid → contacts.userid`（LEFT JOIN）。**无 `receiver_id`/`receiver_name` 列** —— "接收者"只能由 `open_kfid` + `servicer_userid` 推断，因此"发给某人"类查询能力受限。

### 2.2 索引

| 索引 | 路径 | 模型 | 维度 | 后端 | 混合 | 图 | 规模 |
|---|---|---|---|---|---|---|---|
| 消息索引 | `apps/corpchat/search_index/` | `BAAI/bge-m3` | 1024 | faiss | BM25 | ✅ | 140 文档（每消息 1 块） |
| 联系人索引 | `apps/corpchat/contacts_index/` | `BAAI/bge-m3` | 1024 | faiss | BM25 | ❌ | 30 文档 |

### 2.3 `_enrich_chunk` 匹配面

- **匹配面（match surface）**：`标题 "customer_name (label)" + 消息正文（jieba 分词后）`，格式 `{title}\n---\n{_segment(text)}`。
- **结构化元数据**（label/时间/发送者/msgid/open_kfid/company）**不拼入匹配文本**，存于 txtai SQLite `sections.tags`（已验证样例：`{"msgid":..., "send_time":..., "external_userid":..., "servicer_userid":..., "label":...}`），用于过滤/展示/LLM 上下文。
- **分块**：`chonkie.SentenceChunker`（中文 1 字≈0.5 token 的 token 计数器），chunk_size=256、overlap=0；chonkie 不可用时用正则句子切分回退。当前 140 条消息 → 140 块（短消息未拆分）。

### 2.4 构建脚本与增量更新

- CLI：`python apps/corpchat/search.py build [--force] [--graph-mode ...]`、`build-contacts`。
- **仅支持全量重建**：`build()` 若索引已存在且未 `--force` 则直接加载；否则从 DB 全量重灌。**无增量/增量追加**（新消息需要全量 rebuild）。
- 生成测试数据：`gen_fake_msg.py`（Faker，30 联系人 + 140 消息）。

---

## 3. Agent 工具与技能

### 3.1 工具清单（`apps/corpchat/search/tools.py`）

| 工具 | 输入 | 输出 | 路由触发 |
|---|---|---|---|
| `search_messages(query, expand, use_rerank, graph_parallel)` | 检索词 | 格式化文本（【消息搜索结果】+ Score/sender/userid/Label/内容）| 消息类关键词 |
| `search_contacts(query)` | 姓名/userid/细节 | 联系人卡片（姓名/userid/Email/Company/Phone/Job Title）| 联系人关键词（邮箱/电话/公司/职位/姓名等） |
| `search_messages_where(condition)` | 自然语言条件 | 【结构化匹配】SQL 精确结果 | 含 link/url/網址/链接 + 消息语境 |

### 3.2 语义解析能力

- **"来自某人/发给某人"**：跨表推理（`search_messages` → 从结果提取 `userid` → `search_contacts`）。由于无 receiver 列，**"发给某人"仅能靠 `servicer_userid`/`open_kfid` 推断**，准确性有限。
- **工具路由**：`_LiteLLMWrapper._decide_tool_calls` 为规则路由（greeting/system/contact/message/cross-table/structured），搜索意图优先于问候词；LLM 不参与工具选择（仅 `_quick_respond` 对规则无法判定的模糊查询做 greeting/system/search 分类）。
- **查询清洗**：`_extract_search_query` 剥离中英文口语噪声（"try again/please/who is/帮我查" 等）。

### 3.3 错误处理与回退

- 主路径：手动 ReAct 循环（`CrossTableAgent.process`）→ 工具执行 → `_llm_summarize`；异常 → `_fallback_process`（两段式：提取查询→搜消息→提取 userid→搜联系人）。
- LLM 不可用 → 规则/预设回复；`LiteLLMClient.chat()` 依次回退 OpenAI 兼容端 → Ollama 原生 `/api/chat` → DeepSeek，全失败返回空串。
- **反幻觉**：`_llm_summarize`/`generate_answer_litellm` 提示词禁止编造内容/URL/姓名；"没找到"标记（含 `do not include`/`cannot provide`）命中且实际有结果时回退到确定性格式化。
- **确定性联系人回答**：纯 `search_contacts` 查询不再走 LLM 总结，直接用 `_format_fallback_answer`（避免小模型幻觉）。
- **SQL 工具安全**：`_validate_sql` 只允许 SELECT、限定表（messages/contacts）、拒绝 DML/注释、自动 LIMIT；DB 不可用时回退到 txtai 索引正则扫描。

---

## 4. 搜索管道性能

### 4.1 混合检索与融合

- **混合检索**：`Searcher.search()` 支持 keyword/semantic/hybrid（权重 `(0,1)/(1,0)/None`，alpha 默认 0.5）；BM25 与向量均经 jieba 分词对齐，**工作正常**。
- **RRF 融合**：`_weighted_rrf_fusion(k=50)` 原生支持 N 路带权（原始 0.5 / LLM 语义 1.3 / 关键词 1.0 / 图并行路 0.8）。
- **重排序**：`BAAI/bge-reranker-base`（top_n=20）。**近期已加进程级 CrossEncoder 缓存** —— 实测搜索从 ~14s/次 降到 ~0.2s/次（此前每次搜索都重新加载模型）。
- **图增强**：append-only `_graph_expand`（默认）+ opt-in `graph_parallel` 独立融合路（默认关闭，ADR-0001 保持默认行为）。

### 4.2 查询扩展（LLM）

- `QueryExpander`：每查询 1 次语义重写 + 1 次关键词生成（本地 Ollama 约 0.6s），结果按查询缓存。扩展失败 → 回退单路原始查询。
- **代价**：扩展使每次搜索多 1 个 LLM 调用（原始 0.5/语义 1.3/关键词 1.0 三路都进 RRF），且 rerank 开销叠加 → 默认路径单次搜索 ~10s（含扩展+重排）属正常水平。

### 4.3 时序检索

- `TimeExpressionParser`：规则优先（<1ms）解析 最近N天/周/月、昨天/今天、绝对日期等；时间窗口存在时检索量放大 5× 再后置过滤，纯时序查询走 SQLite 扫描直接返回。
- 规模小（140 文档）时全量扫描可接受；`_temporal_list` 注释已注明数据量大后需导出 doc_id→send_time 边表。

### 4.4 已知延迟/超时

| 场景 | 量级 | 说明 |
|---|---|---|
| 首次消息搜索（冷） | ~15–17s | bge-m3 索引 + 模型加载（进程级缓存，仅一次） |
| 首次联系人搜索（冷） | ~15s | 同上；当前 DB 与索引均可访问 |
| 热搜索（重排缓存后） | ~0.2s | 搜索本体 |
| LLM 冷启动 | 数秒–30s+ | Ollama 闲置后卸载模型，下一次调用重新加载 |
| 完整 agent 查询 | 1–14s | 含 classify(可选) + 扩展 + 重排 + 总结 |

> 无硬编码超时问题；各 LLM 调用均有 timeout（3–15s），失败走回退链。

---

## 5. 用户交互与 UI

### 5.1 页面与功能（`apps/corpchat/app.py`，Streamlit）

| 页面 | 功能 |
|---|---|
| **Search** | 聊天式搜索（agent/非 agent 双路径）、问候语 LLM 生成、Process 窗口（工具调用时间线）、引用来源块（可开关）、聊天历史、**记忆图谱**（配置页底部） |
| **Contacts / Messages** | `st.dataframe` 只读表格（DB 直查，30s TTL 缓存） |
| **Overview** | 指标卡（联系人/消息/会话数）+ 按 label 柱状图 |
| **Chat Viewer** | 按联系人查看其全部消息（`fetch_conversations_for_contact`） |

### 5.2 配置面板（"⚙️ Settings"）

- 入口：左栏底部 ⚙️/✖ 开关（编辑模式**替换**聊天面板，CSS 滑入动画，标题显示 "⚙️ Settings"）。
- **CARA 个性**：預設模式（preset）、懷疑度/字面性/共情度滑杆（0–10）、回答長度；DB 持久化。
- **搜索策略**：檢索深度（简单/深度 → agent 开关）、查詢擴展、重排序、Graph hops、Graph path、Top-k、Label filter。
- **知識範圍**：數據源多选（消息/联系人）、引用來源开关。

### 5.3 答案格式与错误提示

- 答案：Markdown 文本；可选【來源】引用块（sender · 日期 · label）；**不支持表格输出、流式输出**。
- 错误提示：空结果 → "抱歉，没有找到…"；图谱故障 → caption 显示"記憶圖譜渲染失敗: {e}"；DB 不可用 → warning。
- 联系人确定性回答为结构化卡片（✅ Found + 邮箱/公司/电话）。

---

## 6. 数据质量问题

| 检查项 | 结果 |
|---|---|
| 联系人重复（同名） | 0 条 |
| 联系人缺 email/phone/full_name | 0 条 |
| 消息空 content | 0 条 |
| 消息缺 external_userid / servicer_userid | 0 条 |
| 发送者不在联系人表（孤儿） | 0 条 |
| 消息缺 msgtype / send_time | 0 条 |
| **receiver 字段** | **不存在**（"发给某人"受限） |
| **性别字段** | **不存在**（"male's name" 类查询无法按性别过滤 —— 产品级限制） |
| 不确定性处理 | 结果显示 Score；CARA 懷疑度(≥7) 在记忆图谱中红色高亮风险标签；答案不臆断缺失属性 |

**索引一致性**：消息索引 140 块与 DB 140 行对应；联系人索引 30 与 DB 30 对应；`website`/`address` 字段存在但未索引、未用于检索。

---


---

## 7. 代码质量与技术债务

### 7.1 结构与规范

- 结构清晰：`apps/corpchat/search/` 按职责分包（searcher/index_builder/query_expander/reranker/tools/cross_table_agent/agentic/temporal/memory_graph/...）；`core/` 提供 DB/配置。
- **大文件**：`app.py` 1110 行、`cross_table_agent.py` 1106 行、`searcher.py` 622 行 —— 接近单体，难以维护。
- PEP8 大体遵循（无 linter 配置；`import` 顺序/行长有小瑕疵）。

### 7.2 硬编码与敏感信息

- **`core/config.py` 硬编码 DB 口令 `***REMOVED***`**（`DB_CONFIG`），`search/config.py` 也有同款回退默认值 —— 风险：仓库泄露即凭据泄露（.env 已 gitignore，但代码内默认口令仍在）。
- `.env` 含真实 `LITELLM_API_KEY`、`DEEPSEEK_API_KEY`（已 gitignore，未入库）。

### 7.3 异常处理

- 全链路 try/except：agent 循环、LLM 回退链、DB 访问、图谱渲染均有兜底，**无未捕获会崩溃的路径**（搜索/图谱失败以文案提示）。
- `_fallback_process` 两段式推理 + 超兜底（两工具重试）。

### 7.4 测试

- **19 个测试文件 / 207 个测试函数 / 全量 202 通过**（~2.5 分钟，含真实 bge-m3 索引构建）。
- `conftest.py`：模块级模型 fixture + MPS 缓存冲刷（解决 16GB Mac MPS OOM）。
- 覆盖：搜索回归门、扩展/重排/图/时序/agentic/UI 流程/工具/SQL 结构化/greeting 确定性/图谱。
- **缺口**：无 SQL 工具的性能/边界 fuzz、无前端组件级测试、无端到端（DB→索引→UI）集成测试。

### 7.5 依赖

- `requirements.txt` 全量锁定；`streamlit==1.59.2`（**环境实装 1.60.0** —— 版本漂移）；langchain 1.3.14 / langgraph 1.2.10（较新）；txtai 9.12.0；sentence-transformers 5.6.1。
- 混入非本应用依赖（fpdf2/pymupdf/OCR 相关），`core/config.py` 有大量发票 OCR 配置 —— 与 CorpChat 检索无关的遗留。

### 7.6 其他

- 未跟踪杂项：`_test_fix.py`、`.scratch/agent-layer-enhancements/`、`.scratch/agent-ui-polish/`、`.scratch/process-window/`（部分 scratch 已入库，部分未入库，状态不一致）。
- `docs/adr/0001-structural-conversation-graph.md` + `docs/hindsight-integration-plan.md` 提供了设计文档；`.scratch/*/issues/*.md` 为按票单跟踪的规范。

---

## 8. 日志与监控

- **日志**：`search/config.py` 全局 `logging.basicConfig(INFO)`；`logger` 贯穿 search 包。粒度以 `logger.warning`/`debug` 为主（searcher/tools/app 合计约 12 处显式日志），记录扩展失败、LLM 回退、SQL 回退等。**无请求级日志**（无 query/tool-call/耗时结构化记录）。
- **监控指标**：**无**。搜索请求量、响应时间、索引大小、工具调用次数均无统计；`perf_counter` 仅用于 agent `steps` 时间线（展示在 UI Process 窗口，未聚合导出）。
- 仓库内**无日志文件**（`*.log` 不存在），日志仅输出到终端。

---

## 9. 已知问题与待办事项

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | 主 LLM 为 `qwen2.5:1.5b`（本地 Ollama），多跳总结易幻觉 | 曾编造不存在的 URL、把 5 个联系人混为一谈 | 已缓解：确定性联系人回答 + 反幻觉提示 + no-info 标记兜底 + SQL 结构化检索 |
| 2 | 冷启动延迟（bge-m3 索引/模型 + Ollama 冷加载） | 首次查询 15s+，闲置后 LLM 调用慢 | 部分缓解：CrossEncoder 进程缓存；索引/LLM 冷启动未解决 |
| 3 | 数据无 receiver / gender 字段 | "发给某人"、"male's name" 类查询能力受限 | 产品级限制，需数据侧补充 |
| 4 | 索引仅全量重建，无增量 | 新消息需手动 rebuild（~分钟级） | 待办 |
| 5 | `core/config.py` 硬编码 DB 口令 | 安全风险 | 待办（改为纯环境变量） |
| 6 | 无结构化日志/监控 | 难排障、无性能趋势 | 待办 |
| 7 | `app.py`/`cross_table_agent.py` 超 1000 行 | 可维护性 | 待办 |
| 8 | `requirements.txt` streamlit 1.59.2 vs 环境 1.60.0 | 环境漂移 | 待办 |
| 9 | 未跟踪杂项（`_test_fix.py`、部分 `.scratch/`） | 仓库脏 | 待办（清理或入库） |
| 10 | 搜索默认路径含 LLM 扩展+重排，单次 ~10s | 感知慢 | 已知权衡（热搜索本体 0.2s） |

---

## 10. 改进建议（按优先级）

**P0 — 安全与可靠**
1. **去除硬编码 DB 口令**：`core/config.py` 与 `search/config.py` 的 `DB_CONFIG` 默认值改为仅环境变量（启动时缺失则报错）。
2. **结构化请求日志 + 指标**：为每次搜索/工具调用记录 `query / tool / status / duration_ms / hit_count`；聚合响应时间与索引规模（可轻量落到 SQLite/CSV）。

**P1 — 数据与能力**
3. **增量索引**：`IndexBuilder.build()` 支持按 `created_at/send_time` 增量追加文档（txtai `index()` 支持追加），避免全量 rebuild。
4. **数据模型补齐**：为 contacts 增加 `gender`（或结构化属性表），为 messages 增加 `receiver_id`（或明确会话参与方模型）以解锁"发给某人"/"male name" 查询。
5. **预加载索引/模型**：应用启动后台线程预热消息+联系人索引与 CrossEncoder，把首次查询冷启动从 15s 摊到启动期。

**P2 — 性能与体验**
6. **查询扩展延迟兜底**：为 `QueryExpander` 增加更短超时 + 结果缓存持久化；对简单名词查询（<=2 token）跳过扩展。
7. **搜索路径去重**：agent 非结构查询与 CLI 共享同一 `Searcher.search()` 参数面，减少重复实现。
8. **答案渲染增强**：支持表格/流式（可选，需升级回答生成方式）。

**P3 — 工程化**
9. **拆分大文件**：`app.py`（页面/组件化）、`cross_table_agent.py`（路由/执行/总结分层）。
10. **版本对齐**：`requirements.txt` 与实装环境对齐（streamlit 1.60）；清理无关 OCR 依赖。
11. **仓库卫生**：删除/入库 `_test_fix.py` 与未跟踪 `.scratch/`；补 `.gitignore` 规则。
12. **测试补强**：SQL 工具 fuzz（非法注入）、端到端（DB→索引→agent）集成测试。

---

*报告基于 2026-08-10 实机检查：DB（postgres localhost:5432）可读、消息索引/联系人索引可加载、全量测试 202 通过。日志/监控为未实现项（报告中已注明）。*

