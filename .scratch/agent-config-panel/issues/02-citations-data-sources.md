# 02 — Knowledge scope & citations: data-source gating + answer citations

**What to build:** 知识范围区块：数据源多选（消息/联系人）门控搜索工具与图谱；引用来源开关 → 答案追加 `【來源】`。

**Blocked by:** 01（面板容器）— 知识范围区块挂进统一面板。

**Status:** ready-for-agent

- [x] 数据源多选（`knowledge.sources` ∈ {消息, 联系人}）→ 门控 `search_messages` / `search_contacts` 调用（取消勾选"联系人"则跳过 contacts 索引）
- [x] 图谱侧数据源过滤接口（`build_entity_graph(sources=...)` 只含选定来源实体）
- [x] `_format_citations(results, max_sources=3)` 助手（sender/send_time/label）
- [x] 引用注入: `app.py` 6-stage 答案 + `Agent.process` 返回前，`citations=True` 时追加
- [x] 测试: 数据源门控（contacts 取消 → search_contacts 不被调用）、引用开/关、全量回归绿

## Comments

- Spec: `.scratch/agent-config-panel/spec.md`
- 合同数据源按产品决策排除（Out of Scope）。

- Implemented & verified on branch feature/hindsight-multipath-rrf-skeleton; full suite 163 passed.
