# Spec: Agent 配置面板 + Hindsight 记忆图谱

**Status:** ready-for-agent

**Feature slug:** `agent-config-panel`

## Problem Statement

Agent 的个性与行为目前分散在多个不连贯的 UI 控件（Enhancements / Filters / Persona 三个独立 expander），且:

1. **CARA 人格缺少预设模式** — 只有三个滑杆，没有"审计助手 / 客服助手 / 研究助理"一键预设。
2. **搜索策略无会话持久化** — expand/rerank/graph/top-k 等参数仅存于 UI 控件，刷新即丢；persona 虽已持久化但体系割裂。
3. **知识范围与输出控制缺失** — 无数据源筛选（消息/联系人）、无引用来源、回答长度仅隐式存在（persona style）。
4. **无记忆图谱** — Hindsight 的"星座视图"（实体关系图）完全缺失；现有 `visualize_graph.py` 是 CLI（chunk 级结构图），未接入 UI，更无实体级记忆图。

目标: 提供一个统一的"🎛️ 配置代理"面板（CARA 人格 / 搜索策略 / 知识范围与输出 / 记忆图谱四区块），所有配置即时生效、会话级持久化，并集成 Hindsight 风格的可交互实体记忆图谱。

## Solution

1. **统一配置面板**：把现有三个 expander 收敛为单一 `🎛️ 配置代理` 面板，四区块（人格/搜索/知识/图谱），所有参数汇入 `st.session_state.agent_config` 单一结构，render 时即时读取（天然即时生效）。
2. **预设模式**：CARA 预设表（审计/客服/研究助理）一键写入三值。
3. **持久化**：新增 `agent_config` 表（session_id PK + 三区块 JSON），`_load_agent_config` / `_persist_agent_config` 镜像现有 `_persist_persona` 模式；启动/切换 session 时恢复。
4. **参数映射**：面板每个控件映射到已有 Agent 管道（详见下表），无行为重构。
5. **记忆图谱**：新 `memory_graph` 模块从消息元数据 + `agent_memory` 抽取实体（人/公司/标签/关键词）与关系（提及/关联/引用），落新表；Streamlit 用 `streamlit-agraph` 渲染星座视图（点击节点→回填搜索框）。

## User Stories

1. As a user, I want a single "配置代理" panel with CARA / search / knowledge / graph sections, so that I tune the agent in one place.
2. As a user, I want one-click CARA presets (审计助手 / 客服助手 / 研究助理), so that common personas are applied instantly.
3. As a user, I want my config to survive page reloads and new sessions, so that I don't re-tune every time.
4. As a user, I want to pick data sources (消息 / 联系人), so that the agent only searches what I choose.
5. As a user, I want citations on/off, so that answers can show their sources.
6. As a user, I want answer length (简洁 / 标准 / 详细), so that output verbosity matches my need.
7. As a user, I want an interactive entity constellation (people/companies/labels/keywords + mention/association/reference links), so that I can see the memory landscape.
8. As a user, I want clicking a graph node to fill the search box and trigger a query, so that I can drill into an entity's memories.
9. As a user, I want the graph to react to my config (skepticism highlights risk nodes; knowledge scope filters entities), so that the constellation reflects my current persona.
10. As a regression maintainer, I want all existing search/agent tests to stay green, so that config changes never degrade retrieval.


## Implementation Decisions

### 1. 配置数据模型（单一来源）

所有面板参数汇入一个结构，存于 `st.session_state.agent_config`，并按 session 持久化到 `agent_config` 表：

```python
# st.session_state.agent_config
{
    "persona": {
        "preset": "audit",                 # audit | service | research | custom
        "skepticism": 8, "literality": 7, "empathy": 3,   # 0-10
        "style": "standard",               # concise | standard | detailed (= 回答长度)
    },
    "search": {
        "depth": "deep",                   # simple | deep
        "expand": True, "rerank": True,
        "graph_hops": 1, "graph_parallel": False,
        "top_k": 5, "label_filter": "",
    },
    "knowledge": {
        "sources": ["messages", "contacts", "contracts"],   # multi-select
        "citations": True,
    },
}
```

### 2. UI 布局（Streamlit 组件）

```
🎛️ 配置代理  (st.expander, 默认展开)
 ├─ 🧠 人格特质 (CARA)
 │   ├─ 怀疑度/字面性/共情度  st.slider(0..10)  → /10 写入 DispositionProfile
 │   ├─ 预设模式 st.selectbox(审计助手/客服助手/研究助理/自訂)
 │   └─ 回答长度 st.selectbox(简洁/标准/详细)  → persona.style
 ├─ ⚙️ 搜索策略
 │   ├─ 检索深度 st.selectbox(简单/深度)
 │   ├─ 查询扩展 st.checkbox  → Searcher.search(expand=)
 │   ├─ 重排序   st.checkbox  → use_rerank
 │   ├─ (既有) Graph hops / Graph path / Top-k / Label filter
 ├─ 📚 知识范围
 │   ├─ 数据源 st.multiselect(消息/联系人)
 │   └─ 引用来源 st.checkbox
 └─ 🕸️ 记忆图谱
     └─ streamlit_agraph (节点=实体, 边=关系, 点击→搜索)
```

### 3. 参数 → Agent 行为映射

| 面板参数 | UI | → 管道 | 现状 |
|---|---|---|---|
| 怀疑度/字面性/共情度 | slider 0-10 | `DispositionProfile` → `build_system_prompt` (3 答案点) | ✅ 已实现 |
| 预设模式 | selectbox | preset dict → 三值 | ❌ 新增 |
| 回答长度 | selectbox | `persona.style` (concise/standard/detailed) | ⚠️ 已存在(style) 需并入面板 |
| 检索深度 简单/深度 | selectbox | simple→非 agent 单步搜索; deep→`CrossTableAgent` 多步 | ⚠️ 需映射 |
| 查询扩展/重排序 | checkbox | `search(expand=)` / `use_rerank` | ✅ 已实现 |
| Graph hops/path/top-k/label | 既有控件 | `search(graph_expand=, graph_parallel=, ...)` | ✅ 已实现 |
| 数据源 | multiselect | 门控: 搜索工具白名单 + 图谱过滤 | ⚠️ messages/contacts 有, contracts 缺 |
| 引用来源 | checkbox | 答案后处理追加 `【來源】` | ❌ 新增 |
| Agent toggle | checkbox | `agent_enabled` → CrossTableAgent | ✅ 已实现 |

### 4. 持久化方案

- **session 级（主）**：`st.session_state.agent_config` — render 即读，天然即时生效（无需重启）。
- **跨会话（辅）**：新 `agent_config` 表（`session_id VARCHAR PK, config TEXT`），`load_agent_config(session_id)` / `save_agent_config(session_id, config)` 镜像 `disposition_profiles` 模式；应用启动 + session 切换时 `_load_agent_config()` 恢复。
- **迁移**：现有 `disposition_profiles` 保留为兼容（persona 区块仍写它）；新搜索/知识区块写 `agent_config`。

### 5. 检索深度映射（简单 vs 深度）

- **简单**：`agent_enabled=False` 等价 —— 单步 `Searcher.search()`（非 agent），不触发跨表。
- **深度**：`agent_enabled=True` —— `CrossTableAgent.process()` 多步推理（消息→userid→联系人）。
- 实现：面板把 `depth` 映射为 `agent_enabled`，UI 的 Agent 复选框与此统一。

### 6. 引用来源

答案生成后，若 `citations=True`，在答案尾部追加来源行（取自结果 metadata 的 sender/send_time/label）：

```python
def _format_citations(results, max_sources=3) -> str:
    lines = []
    for r in results[:max_sources]:
        meta = r.get("metadata", {})
        sender = meta.get("customer_name") or meta.get("external_userid", "?")
        ts = str(meta.get("send_time", ""))[:10]
        lines.append(f"- {sender} · {ts} · [{meta.get('label', '-')}]")
    return "\n【來源】\n" + "\n".join(lines) if lines else ""
```
接入点：`app.py` 6-stage 答案组装处 + `Agent.process` 返回前。

### 7. 记忆图谱（Hindsight 星座视图）

- **实体抽取（`apps/corpchat/search/memory_graph.py`）**：从消息元数据（`customer_name/company/label/open_kfid`）+ `agent_memory` 文本：
  - 节点类型: `person` / `company` / `label` / `keyword`（关键词用 jieba 抽取 top-N）
  - 边类型: `mention`(实体↔消息)、`association`(人↔公司、人↔标签、同会话人↔人)、`reference`(消息↔标签)
- **持久化**：`memory_graph` 表（`session_id PK, nodes JSON, edges JSON, updated_at`）。
- **渲染**：`streamlit-agraph`（新增依赖）——节点大小=连接密度、颜色=类型/关系、支持点击回调。
- **联动**：
  - 高怀疑度(`>=7`) → 风险相关节点(label∈{诈骗,old_friend_reconnect} 等)高亮
  - 数据源 → 图谱只显示选定来源的实体
  - 点击节点 → `st.session_state.search_query = node` + 触发搜索

## 关键代码片段（原型级）

```python
# apps/corpchat/search/memory_graph.py (示意)
CARA_PRESETS = {
    "audit":    {"skepticism": 8, "literality": 7, "empathy": 3, "style": "standard"},
    "service":  {"skepticism": 3, "literality": 4, "empathy": 8, "style": "concise"},
    "research": {"skepticism": 6, "literality": 6, "empathy": 5, "style": "detailed"},
}

def build_entity_graph(session_id, messages, agent_memory, sources) -> dict:
    nodes, edges = [], []
    # person/company from metadata, label/keyword from text
    # association: person-company (contacts), person-label (messages), person-person (same open_kfid)
    # mention: entity -> message id; reference: message -> label
    return {"nodes": nodes, "edges": edges}
```

```python
# app.py 面板骨架
with st.expander("🎛️ 配置代理", expanded=True):
    with st.expander("🧠 人格特質 (CARA)", expanded=True):
        preset = st.selectbox("預設模式", ["審計助手", "客服助手", "研究助理", "自訂"])
        if preset != "自訂":
            p = CARA_PRESETS[PRESET_KEYS[preset]]
            cfg["persona"].update(p)
        cfg["persona"]["skepticism"] = st.slider("懷疑度", 0, 10, cfg["persona"]["skepticism"]) / 10.0
        ...
    with st.expander("⚙️ 搜索策略", expanded=True):
        cfg["search"]["depth"] = st.selectbox("檢索深度", ["简单", "深度"])
        cfg["search"]["expand"] = st.checkbox("查詢擴展", value=cfg["search"]["expand"])
        ...

## Testing Decisions

- **Seams（预约定）**：`DispositionProfile.build_system_prompt`（预设映射）、`_load_agent_config`/`_persist_agent_config`（持久化）、`Searcher.search`（深度/数据源映射的检索行为）、答案点函数（引用注入）、`build_entity_graph`（实体抽取）、配置面板 render（UI 接线，fake streamlit）。
- **确定性**：单元测试无真实 LLM 调用（复用 `_DeterministicExpander` 模式）；实体抽取用确定性元数据。
- **回归门**：全量既有套件保持绿（面板默认值 = 现有行为）。
- **UI 测试**：沿用 `test_search_ui.py` 的 fake streamlit 模式断言面板控件接线。

## 测试用例清单

### 配置面板
- [ ] 预设模式"审计助手"→ skepticism=8/literality=7/empathy=3；"客服助手"→ empathy=8；"研究助理"→ style=detailed
- [ ] 滑杆 0-10 → `DispositionProfile` 值 = 滑杆/10
- [ ] 回答长度 → persona.style 映射（简洁→concise, 标准→standard, 详细→detailed）
- [ ] 检索深度: 简单→`agent_enabled=False`（非 agent 单步）；深度→`agent_enabled=True`（CrossTableAgent）
- [ ] 数据源取消勾选"联系人"→ `search_contacts` 不被调用 / 联系人索引被门控
- [ ] 引用来源开→答案含 `【來源】` + sender/time；关→不含

### 持久化
- [ ] `save_agent_config` / `load_agent_config` round-trip（mock 连接，镜像 persona 测试）
- [ ] 刷新后 `_load_agent_config` 恢复三区块值
- [ ] 默认值 = 现有行为（expand=True, rerank=True, depth=deep）→ 回归门不红

### 记忆图谱
- [ ] `build_entity_graph`: 从元数据抽出 person/company/label 节点；person-company 关联边；同会话 person-person 边
- [ ] 关键词节点: jieba 抽取 top-N
- [ ] 数据源过滤: 只含选定来源的实体
- [ ] 高怀疑度 → 风险标签节点标记 `highlighted`
- [ ] 图谱渲染: streamlit-agraph 收到正确 nodes/edges
- [ ] 点击节点 → `st.session_state.search_query` 被填充并触发搜索

### 回归
- [ ] 全量 `tests/` 绿（面板默认值不改变现有检索/agent 行为）

## Out of Scope

- **合同数据源**：产品决策 — 合同不出本期范围；数据源多选仅含 消息 / 联系人。
- **图谱实体抽取的 LLM 增强**：POC 用规则 + jieba；LLM 实体抽取是后续优化。
- **配置版本化 / 多用户配置中心**：单 session 持久化足够。
- **CARA 检索侧加权**（高怀疑度提升图/时序权重）——仍为未来扩展。
- **Neo4j**：POC 拒绝，图谱存 `memory_graph` 表 + streamlit-agraph 渲染。

## Further Notes

- 背景：`docs/hindsight-integration-plan.md`；已有组件（persona 层、adaptive paths、temporal、graph-parallel）是此面板的"后端"，面板只是统一它们 + 新增预设/引用/图谱。
- 依赖新增：`streamlit-agraph`（图谱交互）——需加入 `requirements.txt`。
- 数据源多选仅含 消息 / 联系人（合同按产品决策排除）。

```
