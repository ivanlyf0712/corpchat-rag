# Hindsight Memory 整合方案 — CorpChat RAG 分析报告

> 版本: v1.0  |  日期: 2026-08-07  |  范围: `corpchat-rag/`（不含 `oa-rag`）
>
> 目标: 评估在现有 CorpChat RAG（txtai 混合检索 + LLM 查询扩展 + 交叉编码器重排序 + LangChain Agent）之上，
> 引入 **Hindsight Memory** 的"四路并行检索 + RRF + 重排序 + CARA 个性层"机制，给出差距分析、整合点、优先级与分阶段实施路线。

---

## 0. 分析结论速览（TL;DR）

| 维度 | 结论 |
|---|---|
| 已具备 | 语义检索、关键词检索、RRF 融合(k=50)、交叉编码器重排序、结构图（txtai） |
| 缺少 | 时序检索（Temporal）、图遍历作为**独立并行检索路**、CARA 个性驱动层 |
| 最高杠杆点 | `Searcher._weighted_rrf_fusion()` 已原生支持任意路数+权重，**无需重构即可接入第 3、4 路检索** |
| 快速引入（Phase 1） | 时序检索 + RRF 多路泛化（1~2 周） |
| 中期（Phase 2） | 图遍历升级为独立检索路，实现 4 路 RRF（2~3 周） |
| 后期（Phase 3） | CARA 个性驱动层，接入 3 个答案生成点（3~4 周） |
| 图数据库 | **POC 不建议引入 Neo4j**；复用 txtai graph（已支持 Cypher，结构边确定性，单机维护成本低） |
| 关键约束 | `tests/test_search_regression.py` 是**永久回归门**；新检索路必须默认关闭或保持现有排序不变 |

---

## 1. 现有系统架构概览

### 1.1 当前搜索管道（实测代码，非设计文档）

```
用户查询
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 入口层                                                       │
│  · app.py (Streamlit) / search.py (CLI) / agent.py (Agent)  │
│  · SearchRouter.decide()        → 是否搜索 / 直接聊天        │
│  · IntentClassifier._rule_classify() → greeting/system/... │
│  · AgenticDecider.decide()      → mode / expand / graph /   │
│                                   use_rerank 参数决策       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 查询扩展层  QueryExpander.expand(query)                     │
│   · 原始查询                     weight=0.5  (ORIGINAL_QUERY_WEIGHT)  │
│   · LLM 语义重写                 weight=1.3  (LLM_SEMANTIC_QUERY_WEIGHT)│
│   · LLM 关键词扩展 (最多3条)      weight=1.0  (LLM_KEYWORD_QUERY_WEIGHT)│
└──────────────────────────┬──────────────────────────────────┘
                           ▼  (每个扩展查询独立执行)
┌─────────────────────────────────────────────────────────────┐
│ 混合检索层  embeddings.search(q, weights=(α, 1-α))          │
│   · 语义向量 bge-m3  (txtai, faiss, 1024维)                  │
│   · BM25   jieba 中文分词  (scoring.method=bm25)             │
│   · mode: keyword / semantic / hybrid                       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 加权 RRF 融合  Searcher._weighted_rrf_fusion(all_results, k=50)│
│   对"每条扩展查询的结果列表"按权重求和: score += w/(k+rank)    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 图扩展（追加式，可选）  Searcher._graph_expand()              │
│   · 从 base 结果出发沿 4 种结构边 1 跳遍历                    │
│   · score = parent × hop_discount(0.8) × query_relevance     │
│   · 追加在 base 结果下方，绝不重排 base                       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 交叉编码器重排序  Reranker.rerank(query, results)            │
│   · BAAI/bge-reranker-base，仅对前 top_n=20 重排              │
│   · 保留原 score，用 rerank_score 排序                        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 答案生成层（Agent）                                          │
│  · Agent._generate_answer()      (context → LLM answer)     │
│  · CrossTableAgent._llm_summarize()  (工具结果整合)          │
│  · app.generate_answer_litellm()  (RAG QA)                  │
└─────────────────────────────────────────────────────────────┘
```

> 说明：图扩展发生在 **RRF 融合之后、重排序之前**，且是"追加式"而非"独立并行检索路"。
> 这与 Hindsight 的"图遍历检索作为第四路、在融合**之前**并行执行"是核心架构差异。

### 1.2 代码位置清单

| 功能 | 文件 | 关键符号 |
|---|---|---|
| 查询扩展 | `apps/corpchat/search/query_expander.py` | `QueryExpander.expand()` |
| 混合检索 | `apps/corpchat/search/searcher.py` | `Searcher.search()` 路径 A/B |
| 中文分词 | `apps/corpchat/search/utils.py` | `_segment()` |
| RRF 融合 | `apps/corpchat/search/searcher.py` | `Searcher._weighted_rrf_fusion()` |
| 图扩展 | `apps/corpchat/search/searcher.py` | `Searcher._graph_expand()` / `graph_query()` |
| 图构建 | `apps/corpchat/search/index_builder.py` + `utils.py` | `IndexBuilder.build()` / `_compute_structural_relationships()` |
| 重排序 | `apps/corpchat/search/reranker.py` | `Reranker.rerank()` |
| Agent 参数决策 | `apps/corpchat/search/agentic.py` | `AgenticDecider.decide()` |
| 意图路由 | `apps/corpchat/agent.py` | `IntentClassifier` / `Agent.process()` |
| LLM 封装 | `apps/corpchat/search/litellm_client.py` | `LiteLLMClient.chat()` |
| ReAct Agent | `apps/corpchat/search/cross_table_agent.py` | `CrossTableAgent.process()` |
| LangChain 工具 | `apps/corpchat/search/tools.py` | `search_messages` / `search_contacts` |
| 数据源 | `core/db.py` / `apps/corpchat/search/index_builder.py` | PostgreSQL (`psycopg2`) |

> **注意**：任务背景描述数据源为 MySQL，但 `corpchat-rag` 实际使用 **PostgreSQL**（`psycopg2`，见 `core/db.py`、`index_builder.py`）。
> `oa-rag` 才使用 `pymysql`。本报告按实际代码（PostgreSQL）分析；若生产确为 MySQL，仅需替换连接层，不影响检索架构。

---

## 2. Hindsight 组件对照表

| Hindsight 组件 | CorpChat 现状 | 结论 |
|---|---|---|
| **语义检索 (Vector)** | bge-m3 向量检索（txtai hybrid / semantic 模式） | ✅ 已有 |
| **关键词检索 (BM25)** | jieba 分词 + txtai BM25（`scoring.method=bm25`） | ✅ 已有（中文增强） |
| **图遍历检索 (Graph)** | txtai 结构图 + 1 跳追加式扩展（`_graph_expand`，4 种可遍历边） | ⚠️ 部分 — 是融合**后**增强，不是融合**前**独立检索路 |
| **时序检索 (Temporal)** | 仅 `date_from` / `date_to` 过滤参数 | ❌ 缺失 — 无查询时间解析、无 recency 打分 |
| **RRF 融合 (k=50)** | `_weighted_rrf_fusion(all_results, k=50)`，支持任意路数 + 每路权重 | ✅ 已有，**与 Hindsight 的 k=50 一致** |
| **交叉编码器重排序** | `Reranker`（bge-reranker-base, top_n=20） | ✅ 已有 |
| **CARA 个性驱动层**（怀疑度/字面性/共情度/风格化输出） | 无任何 disposition/trait 概念 | ❌ 缺失 |
| **记忆 (Memory)** | `agent_memory` 表（session_id + turn 级历史，`Agent` 多轮上下文） | ⚠️ 部分是对话记忆，非检索记忆 |

---

## 3. 差距分析

### 3.1 现有实现与 Hindsight 的差异

| 差异点 | Hindsight | CorpChat 现状 | 影响 |
|---|---|---|---|
| 检索路组织 | 4 路**并行独立**检索 → RRF | 单路 hybrid + 扩展查询多路 → RRF；图是**融合后**追加 | 图/时序无法作为独立信号参与融合 |
| 图检索 | 图遍历是一等检索路（返回排序列表） | 追加式、不重排 base（`CONTEXT.md` 明确定义为 append-only） | 架构级差异，Phase 2 解决 |
| 时序检索 | 时间表达式解析 + 时间相关排序 | 仅过滤参数，无解析、无打分 | 时间敏感查询（"最近/今天/上周"）召回差 |
| 权重体系 | 每路独立权重 | 每路权重已有（0.5/1.3/1.0），但仅用于扩展查询 | 只需补充新路的权重常量 |
| 重排顺序 | 融合后统一重排 | 融合 → 图扩展 → 重排 | 若图变为融合前路径，重排顺序自然对齐 Hindsight |
| 个性层 | 融合后 CARA 驱动风格化 | 无 | Phase 3 |
| 重排一致性 | 无 | rerank 分数与 RRF 分数并存（display 用原分数） | 无需改动 |

### 3.2 已具备（可直接复用）

1. **RRF 融合器**：`_weighted_rrf_fusion` 是静态方法，输入 `List[Tuple[List[Tuple[str, float]], float]]`（每路结果列表 + 权重），输出 `List[Tuple[str, float]]`。**天生支持任意路数**，接入第 3、4 路零重构。
2. **txtai graph**：已支持 Cypher（`graph_query`）、结构边确定性构建、`graph.edges()`/`graph.node()` API。Phase 2 可复用其遍历能力。
3. **元数据体系**：每个文档在 `sections.tags` 中已存 `send_time / send_time_str / label / company / open_kfid` 等结构化字段，时序检索与图遍历的数据基础已就位。
4. **LLM 封装**：`LiteLLMClient`（含 DeepSeek/Ollama 双回退），时序解析、个性提示都可复用。
5. **回归测试体系**：`tests/test_search_regression.py` 用确定性内存索引（bge-m3）测试 `Searcher.search()` seam —— 新特性的验证基准。

### 3.3 缺少（需新建）

1. **时序检索模块**（Temporal Retriever）
2. **图并行检索模块**（Graph Retriever，区别于现有 `_graph_expand`）
3. **CARA 个性驱动层**（Disposition Profile + 风格化提示词构建）
4. **检索路注册表 / 并行执行器**（Retriever Registry + 可选 ThreadPool 并行）



---

## 4. 整合点识别（代码位置 + 修改建议）

### 4.1 关键可插拔接口（现状签名）

| 接口 | 位置 | 签名 | 可插拔性 |
|---|---|---|---|
| RRF 融合 | `searcher.py` | `_weighted_rrf_fusion(all_results: List[Tuple[List[Tuple[str,float]], float]], k=50) -> List[Tuple[str,float]]` | **极高** — 已支持 N 路 |
| 搜索入口 | `searcher.py` | `Searcher.search(query, mode, limit, expand, graph_expand, label_filter, date_from, date_to, use_rerank) -> List[Dict]` | 高 — 参数化、无侵入 |
| 查询扩展 | `query_expander.py` | `QueryExpander.expand(query) -> List[Tuple[str,float]]` | 高 — 可追加时间/实体查询 |
| 重排 | `reranker.py` | `Reranker.rerank(query, results: List[Dict]) -> List[Dict]` | 高 — 融合后统一执行 |
| 图后端 | `searcher.py` | `graph_query(cypher, limit)` / `embeddings.graph.edges()` | 高 — txtai Cypher |
| Agent 答案 | `agent.py` | `Agent._generate_answer(query, context)` | 高 — 个性层注入点 |
| ReAct 答案 | `cross_table_agent.py` | `_llm_summarize(query, msg_result, contact_result)` + `SYSTEM_PROMPT` | 高 — 个性层注入点 |
| UI 答案 | `app.py` | `generate_answer_litellm(query, context)` | 高 — 个性层注入点 |
| Agent 工具 | `tools.py` | `search_messages(query, expand=False, use_rerank=False) -> str` | 中 — 工具级调用 |
| 索引构建 | `index_builder.py` | `IndexBuilder.build(force, enable_graph, graph_mode)` | 中 — 建索引期挂接时序/图字段 |

**结果文档结构（统一契约）**：
```python
{
    "id": str,            # txtai sections.id
    "text": str,          # 干净消息内容（不含 title/metadata）
    "score": float,       # RRF 分数或原生 0~1 分数
    "metadata": Dict,     # tags 解析出的结构化字段
    # 可选: "rerank_score": float  # Reranker 追加，仅用于排序
}
```
`metadata` 字段键：`label, customer_name, external_userid, servicer_userid, send_time, send_time_str, open_kfid, origin, company, chunk_index, msgid`

### 4.2 问题一：RRF 融合模块如何扩展支持更多路检索结果？

**现状**：`_weighted_rrf_fusion` 的输入 `all_results` 已是 `List[(结果列表, 权重)]`，天然支持任意路数。
当前 `Searcher.search()` 路径 B 只是把"每条扩展查询"的结果放入该列表。

**建议**：将 `all_results` 的语义从"每条扩展查询一路"推广为"每个检索器一路"，
提取一个 `_retrieve_parallel()` 辅助方法，集中组装各路结果：

```python
# searcher.py (示意，非完整实现)
def _retrieve_parallel(self, query, weights, limit,
                       semantic=True, keyword=True,
                       graph_parallel=False, temporal=False) -> List[Tuple[List[Tuple[str,float]], float]]:
    all_results = []
    for q, q_weight in self.expander.expand(query):          # 语义/关键词扩展路
        all_results.append((self._hybrid_search(q, weights, limit), q_weight))
    if graph_parallel and self._graph_retriever:             # 图遍历路
        all_results.append((self._graph_retriever.retrieve(query, limit),
                            GRAPH_RETRIEVAL_WEIGHT))
    if temporal and self._temporal_retriever:                # 时序路
        all_results.append((self._temporal_retriever.retrieve(query, limit),
                            TEMPORAL_WEIGHT))
    return all_results
```

- `_weighted_rrf_fusion` **无需任何修改**。
- 新增权重常量：`GRAPH_RETRIEVAL_WEIGHT`、`TEMPORAL_WEIGHT`（建议 0.8~1.2 区间，与现有 0.5/1.0/1.3 同量级，先在回归基准上调参）。
- 各检索路相互独立，可用 `concurrent.futures.ThreadPoolExecutor` 并行执行（txtai 搜索是 CPU/本地调用，线程并行即可）。

### 4.3 问题二：图遍历检索是否需要引入图数据库（Neo4j）？

**结论：POC 阶段不需要。** 理由：

1. **txtai graph 已满足需求**：现有 `_graph_expand` 已经实现 1 跳结构边遍历（`graph.edges()` + `graph.node()`），并带 query-consistency 门控。把这段逻辑从"追加式"改造成"返回排序列表"即可得到 `GraphRetriever`。
2. **支持 Cypher**：`Searcher.graph_query()` 已封装 txtai 图查询，具备未来扩展查询语言的入口。
3. **运维成本**：本地单机 POC，Neo4j 增加 JVM/服务部署、数据同步、权限维护成本，违背"优先实施可行性"约束。
4. **数据规模**：单会话结构图（每消息一块为一节点），本地索引可承载。

**何时再评估 Neo4j**：
- 需要跨会话、跨用户的长链路图推理（如"A 推荐 B → B 推荐 C → C 成交"）；
- 需要多跳（>2 hop）实体级（人物/公司/标签作为独立节点）遍历；
- 图数据量超过单机 txtai 承载（数十万节点）。

若未来需要，现有 `GraphRetriever` 接口（`retrieve(query) -> List[(id, score)]`）可平滑替换后端为 Neo4j Cypher，`Searcher` 侧无感知。

### 4.4 问题三：时序检索是否需要时间解析模块？

**结论：需要，但轻量实现即可，无需独立服务。**

**建议新增 `apps/corpchat/search/temporal.py`**，包含两个组件：

```python
class TemporalQueryParser:
    """从查询中解析时间窗口。规则优先（<1ms），LLM 回退。"""
    def parse(self, query: str) -> Optional[TimeWindow]:
        # 规则: 最近N天/周/月, 昨天, 今天, 上周, 上个月, 本月, 今年,
        #       YYYY-MM-DD, YYYY年M月, MM/DD 等（正则 + python-dateutil）
        # 回退: LiteLLMClient 抽取 {start, end} 时间范围
        # 返回: TimeWindow(start_iso, end_iso, recency_bias) 或 None

class TemporalRetriever:
    """按时间窗口过滤 + recency 加权排序，返回 List[(doc_id, score)]。"""
    def __init__(self, embeddings): ...          # 复用 embeddings.database (sections.tags)
    def retrieve(self, query: str, limit: int) -> List[Tuple[str, float]]:
        # 1) parser.parse(query) → 无时间意图则返回 []
        # 2) 在内存/索引中扫 tags.send_time 过滤窗口内文档
        # 3) score = recency_decay(now - send_time) 或 基础相关×decay
        # 4) 返回排序列表，供 RRF 融合
```

**实现要点**：
- `send_time` 已在 `sections.tags` 中，POC 语料规模小，可内存扫描（`graph.scan(data=True)` 或遍历 `database`）；数据量增大时在建索引期导出 `doc_id → send_time` 边表（txtai 自定义列）即可。
- recency 打分建议 `decay = exp(-λ·age_days)` 或分段衰减（近 24h/7d/30d），λ 作为配置。
- 时序路 **权重应低**（如 0.8），避免无时间意图的查询被时序结果淹没；`parse()` 返回 None 时整路为空，RRF 自然忽略。

### 4.5 问题四：个性驱动层如何与现有 Agent 系统集成？

**结论：作为"提示词条件化层"接入 3 个答案生成点，不改变检索与 Agent 控制流。**

**建议新增 `apps/corpchat/search/persona.py`**：

```python
class DispositionProfile:
    """CARA 个性画像。"""
    def __init__(self, skepticism: float, literality: float,
                 empathy: float, style: str = "concise"): ...
    def build_system_prompt(self, base_prompt: str) -> str:
        # 按 trait 追加风格指令，例如:
        #   怀疑度高 → "对检索证据不足的结论要明确标注不确定性，避免臆断"
        #   字面性高 → "严格依据检索到的原文回答，不添加推测"
        #   共情度高 → "以温和、体谅的语气回答，先回应情绪再给信息"
        #   风格 → concise / detailed / structured
```

**集成点（3 处，均为"注入系统提示词"）**：

| 集成点 | 文件/函数 | 改法 |
|---|---|---|

---

## 5. 分阶段实施路线图

### Phase 1 — RRF 多路泛化 + 时序检索（1~2 周）

**目标**：从"扩展查询多路"升级为"检索器多路"，新增时序检索路，实现 3 路 RRF。

**改动范围**：
- `apps/corpchat/search/searcher.py`：抽取 `_retrieve_parallel()`；`search()` 新增 `temporal: bool = False` 参数（默认关，保回归门绿）
- 新增 `apps/corpchat/search/temporal.py`：`TemporalQueryParser` + `TemporalRetriever`
- `apps/corpchat/search/config.py`：新增 `TEMPORAL_WEIGHT`、`RECENCY_HALF_LIFE_DAYS`
- 新增 `tests/test_search_temporal.py`（沿用确定性内存索引模式）

**预期收益**：
- "最近/今天/上周"等时间敏感客服查询显著提升
- 验证 RRF 多路机制，为 Phase 2 铺路
- 回归测试门保持绿色

### Phase 2 — 图遍历升级为独立检索路（2~3 周）

**目标**：新增 `GraphRetriever`，图从"融合后追加"改为"融合前并行路"，实现 4 路 RRF。

**改动范围**：
- 新增 `apps/corpchat/search/graph_retriever.py`：复用结构边 + query-consistency 门控，返回 `List[(id, score)]`（对齐 Hindsight 图遍历检索）
- `searcher.py`：`search()` 新增 `graph_parallel: bool = False`；与现有 `_graph_expand`（append-only 兼容模式）并存，`AgenticDecider` 决策是否启用
- `config.py`：新增 `GRAPH_RETRIEVAL_WEIGHT`
- 新增 `tests/test_search_graph_parallel.py`

**预期收益**：
- 架构对齐 Hindsight 四路检索
- 跨会话/同客户上下文召回增强（图作为独立信号参与融合）

### Phase 3 — CARA 个性驱动层（3~4 周）

**目标**：实现 disposition-trait 画像 + 回答风格化输出。

**改动范围**：
- 新增 `apps/corpchat/search/persona.py`：`DispositionProfile`
- `core/db.py`：新增 `disposition_profiles` 表 CRUD（复用 psycopg2 连接）
- 接入 3 个答案生成点（见 §4.5）
- `app.py` UI：画像调整控件 + 当前画像展示
- 新增 `tests/test_persona.py`

**预期收益**：
- 个性化回答体验，不同会话获得一致的语气/风格
- 为后续"画像驱动的检索加权"预留接口

---

## 6. 风险评估

| 风险 | 等级 | 说明 | 缓解 |
|---|---|---|---|
| 回归门被破坏 | 高 | `test_search_regression.py` 断言 `Searcher.search()` 排序语义；新检索路若默认开启可能改变结果顺序 | 新特性**默认关闭**；仅通过显式参数/`AgenticDecider` 开启；在回归基准上迭代权重 |
| 多路检索延迟 | 中 | 每增加一路多一次 txtai 搜索 | 各路相互独立，用 `ThreadPoolExecutor` 并行；时序路内存扫描 <10ms |
| 时序解析误判 | 中 | 规则优先漏掉口语化时间表达；LLM 回退增加延迟 | 规则覆盖高频表达（今天/昨天/最近N天/上周/本月）；LLM 回退设超时（10s 内）；解析失败整路为空，不降级其他路 |
| 图并行检索与追加扩展语义冲突 | 中 | `CONTEXT.md` 明确图扩展为 append-only；并行化改变该契约 | Phase 2 保持 append-only 为默认，`graph_parallel` 显式 opt-in；同步更新 CONTEXT.md/ADR |
| reranker 对时序/图结果降权 | 中 | bge-reranker-base 对"时间相关"查询不一定友好 | 在合成基准上验证 rerank 前后时序命中是否保持；必要时 rerank 仅作用于混合+图路 |
| 个性层风格漂移 | 低 | 风格指令可能被模型忽略或过度发挥 | trait 数值离散化（低/中/高）+ 明确的风格指令模板；回归测试校验 system prompt 包含 trait 指令 |
| LLM 依赖（扩展/时序/个性） | 中 | 本地 POC 的 LLM 端点可能不可用 | 全部走 `LiteLLMClient` 优雅降级：时序规则优先、个性层在 LLM 不可用时返回原文摘要 |
| 数据源差异 | 低 | 任务描述 MySQL vs 实际 PostgreSQL | 已在 §1.2 标注；时序/画像表按现有 psycopg2 模式建表，迁移 MySQL 只需换连接层 |

---

## 7. 技术决策建议

1. **不引入 Neo4j（POC）**：复用 txtai graph。txtai 已支持 Cypher（`graph_query`）、确定性结构边、单机低维护。仅在多跳/实体级图推理成为硬需求时评估迁移，届时 `GraphRetriever` 接口保证后端可替换。
2. **时序检索用轻量规则 + dateutil + LLM 回退**：不引入独立时序数据库、不引入 NLP 时间解析服务；`send_time` 已存在于文档元数据，首版内存扫描即可。
3. **RRF 泛化是最高杠杆动作**：`_weighted_rrf_fusion` 已是 k=50、支持 N 路 + 权重，与 Hindsight 一致；新增检索路只需"往 `all_results` 里加一路"，这是成本最低、收益最直接的切入点。
4. **CARA 个性层以提示词条件化落地**：不做用户建模 ML，不做模型微调；画像表 + 系统提示词注入，维护成本最低，符合本地 POC 约束。
5. **严格守护永久回归门**：所有新检索路默认关闭、显式开启；每个 Phase 以 `tests/test_search_regression.py` 绿为标准验收。
6. **新增模块遵循现有结构**：检索器（`temporal.py` / `graph_retriever.py` / `persona.py`）放 `apps/corpchat/search/` 包内，配置常量入 `config.py`，导出走 `__init__.py`，与现有 `QueryExpander`/`Reranker` 的插拔风格一致。

---

## 附录 A：现有数据结构与函数签名速查

### 统一结果文档
```python
{"id": str, "text": str, "score": float, "metadata": Dict, "rerank_score": float(可选)}
```

### 核心函数签名
```python
# searcher.py
Searcher(embeddings: txtai.Embeddings, expander=None, reranker=None)
_weighted_rrf_fusion(all_results: List[Tuple[List[Tuple[str,float]], float]], k=50) -> List[Tuple[str,float]]   # static
search(query: str, mode="hybrid", limit=10, expand=True, graph_expand=0,
       label_filter=None, date_from=None, date_to=None, use_rerank=True) -> List[Dict]
_graph_expand(results, max_expand=3, hop_discount=0.8, limit=20, query="",
              label_filter=None, date_from=None, date_to=None) -> List[Dict]
_fetch_one_doc(doc_id: str) -> Optional[Dict]
graph_query(cypher: str, limit=20) -> List[Dict]

# query_expander.py
QueryExpander.expand(query: str, use_cache=True) -> List[Tuple[str, float]]   # (查询, 权重)

# reranker.py
Reranker.rerank(query: str, results: List[Dict]) -> List[Dict]

# agentic.py
AgenticDecider.decide(query: str) -> Dict[str, Any]   # {mode, expand, graph_expand, use_rerank}

# tools.py
search_messages(query: str, expand=False, use_rerank=False) -> str   # LangChain @tool
search_contacts(query: str) -> str

# cross_table_agent.py
CrossTableAgent.process(user_input: str, on_stage=None) -> Dict  # {output, thoughts, tool_calls, steps, success, fallback}

# agent.py
Agent.process(query, top_k=..., ...) -> Tuple[intent, response, results]

# index_builder.py
IndexBuilder.build(force=False, enable_graph=True, graph_mode="auto") -> txtai.Embeddings

# utils.py
_segment(text: str) -> str
_compute_structural_relationships(chunks) -> Dict[str, List[Dict]]   # 5 种结构关系
```

### 关键配置常量（config.py）
```
RRF_K_VALUE = 50
ORIGINAL_QUERY_WEIGHT = 0.5
LLM_SEMANTIC_QUERY_WEIGHT = 1.3
LLM_KEYWORD_QUERY_WEIGHT = 1.0
DEFAULT_HYBRID_ALPHA = 0.5
DEFAULT_RERANK_TOP_N = 20
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
```

### 图结构（ADR-0001）
- 5 种结构边：`same_conversation / sender_receiver / same_sender / same_company / same_label`
- 可遍历 4 种：前 4 种；`same_label` 仅记录、不遍历
- 构建：`IndexBuilder` 建索引时经 `relationships` 列写入 txtai graph

| 单轮 RAG QA | `app.py` → `generate_answer_litellm()` | system prompt 前拼接 `profile.build_system_prompt()` |
| Agent 意图搜索 | `agent.py` → `Agent._generate_answer()` | 同上 |
| ReAct 跨表 Agent | `cross_table_agent.py` → `SYSTEM_PROMPT` / `_llm_summarize()` | 在 SYSTEM_PROMPT 尾部追加风格指令 |

**画像来源（POC 简化）**：
- 新增 `disposition_profiles` 表（`session_id, skepticism, literality, empathy, style, updated_at`），或直接扩展现有 `agent_memory`；
- UI 提供默认画像调整；后续可让 LLM 从对话历史推断画像（Phase 3 延伸）。

**可选的检索侧影响（默认关闭）**：高怀疑度画像可提升图/时序路权重以强化证据链 —— 但为控制 POC 复杂度，首版只做回答风格化。
