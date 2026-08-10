# 03 — Memory graph: entity extraction + constellation view

**What to build:** Hindsight 星座视图。新 `memory_graph` 模块从消息元数据 + `agent_memory` 抽取实体（人/公司/标签/关键词）与关系（提及/关联/引用），落 `memory_graph` 表；Streamlit 用 `streamlit-agraph` 渲染交互图谱（点击节点→回填搜索框），并联动人格（高怀疑度高亮风险节点）与数据源（过滤）。

**Blocked by:** 01（面板容器）— 图谱区块挂进统一面板。

**Status:** ready-for-agent

- [x] 新增依赖 `streamlit-agraph`（requirements.txt）
- [x] `apps/corpchat/search/memory_graph.py`:
  - `build_entity_graph(session_id, messages, agent_memory, sources)` → `{nodes, edges}`
  - 节点: person/company/label/keyword（jieba top-N）
  - 边: mention / association（person-company、person-label、同会话 person-person）/ reference（message-label）
- [x] `memory_graph` 表（session_id PK + nodes/edges JSON）+ load/save
- [x] 图谱渲染: `st.components` + streamlit-agraph `Config`（节点大小=连接密度、颜色=类型）
- [x] 联动 1: 高怀疑度(≥7) → 风险标签节点（诈骗/old_friend_reconnect 等）`highlighted`
- [x] 联动 2: 数据源 → 图谱只含选定来源实体
- [x] 联动 3: 点击节点 → `st.session_state.search_query = node` + 触发搜索
- [x] 测试: 实体/边抽取（确定性元数据）、关键词 top-N、数据源过滤、怀疑度高亮、渲染接收正确 nodes/edges、点击回填、全量回归绿

## Comments

- Spec: `.scratch/agent-config-panel/spec.md`
- 参考: Hindsight 星座视图；现有 `visualize_graph.py`（CLI，pyvis）可复用图提取思路但需改为实体级。
- 若 `streamlit-agraph` 引入困难，降级方案: pyvis HTML iframe（渲染）但点击→Streamlit 需 hack；streamlit-agraph 是首选（原生点击回调）。
- Implemented & verified on branch feature/hindsight-multipath-rrf-skeleton; full suite 172 passed.
