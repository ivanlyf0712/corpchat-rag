# 02 — Knowledge scope & citations: data-source gating + answer citations

**What to build:** 知识范围区块：数据源多选（消息/联系人/合同）门控搜索工具与图谱；引用来源开关 → 答案追加 `【來源】`。合同源依赖新的 contracts 索引/工具（独立子任务）。

**Blocked by:** None — 可先做消息/联系人门控 + 引用；合同源单独依赖。

**Status:** ready-for-agent

- [ ] 数据源多选（`knowledge.sources`）→ 门控 `search_messages` / `search_contacts` 调用（取消勾选"联系人"则跳过 contacts 索引）
- [ ] 图谱侧数据源过滤接口（`build_entity_graph(sources=...)` 只含选定来源实体）
- [ ] `_format_citations(results, max_sources=3)` 助手（sender/send_time/label）
- [ ] 引用注入: `app.py` 6-stage 答案 + `Agent.process` 返回前，`citations=True` 时追加
- [ ] 测试: 数据源门控（contacts 取消 → search_contacts 不被调用）、引用开/关、全量回归绿

## Comments

- Spec: `.scratch/agent-config-panel/spec.md`
- 合同源依赖: 当前只有 `invoices` DB 表 + `search_similar`（pgvector）；需要 contracts 索引或 `search_contracts` 工具才能作为数据源 —— 列为后续独立 ticket（本 ticket 只做门控框架 + 消息/联系人）。
