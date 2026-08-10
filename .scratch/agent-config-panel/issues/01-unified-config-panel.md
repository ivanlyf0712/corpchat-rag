# 01 — Unified config panel: presets, persistence, depth & answer-length mapping

**What to build:** 把现有三个 expander 收敛为单一 `🎛️ 配置代理` 面板（人格/搜索/知识/图谱四区块），新增 CARA 预设模式、`agent_config` 表持久化、检索深度（简单/深度）映射、回答长度→persona.style 映射。默认值 = 现有行为，回归门不红。

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `st.session_state.agent_config` 单一结构（persona/search/knowledge），render 即时读取
- [ ] CARA 预设表（审计/客服/研究助理）→ 一键写三值 + style；"自訂"保留滑杆覆盖
- [ ] 回答长度 select（简洁/标准/详细）→ `persona.style`（concise/standard/detailed）
- [ ] 检索深度 select（简单/深度）→ 简单=非 agent 单步；深度=`agent_enabled=True`（CrossTableAgent）
- [ ] `agent_config` 表 + `load_agent_config`/`save_agent_config`（core/db.py，镜像 disposition_profiles）
- [ ] `_load_agent_config`（启动/session 切换恢复）/ `_persist_agent_config`（render 或 turn 完成时）
- [ ] 面板默认值 = 现有行为（depth=deep, expand=True, rerank=True, graph_hops=1）
- [ ] 测试: 预设映射、滑杆/10、回答长度映射、深度→agent_enabled、持久化 round-trip、刷新恢复、全量回归绿

## Comments

- Spec: `.scratch/agent-config-panel/spec.md`
- 复用: persona 面板（`DispositionProfile`）、Enhancements/Filters 控件、`AgenticDecider`
