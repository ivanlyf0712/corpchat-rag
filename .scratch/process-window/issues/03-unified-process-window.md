# 03 — Unified "Process" window UI

**What to build:** Replace the two expanders (`Details` + `🤖 Agent Process`) with one collapsed "Process" expander per turn. Non-agent turns show the existing raw-hits table. Agent turns show compact, expandable per-tool sub-windows (tool name, query, expanded queries, hit count, top-5 previews). Label reads `Process` or `Process (agentic · ✅ · N tools · Xms)`. Delete the dead "Raw results" toggle block.

**Blocked by:** 02 — Persist per-tool process payload per turn

**Status:** ready-for-agent

- [x] Exactly one expander per turn, default collapsed
- [x] Label: `Process` (non-agent) vs `Process (agentic · ✅ · N tools · Xms)` / `⚠️ fallback` (agent)
- [x] Non-agent turns render the raw-hits table with the existing column config (Message ID / Content / Score / Metadata)
- [x] Agent turns render one compact expandable sub-window per tool with query, expanded queries, hit count, top-5 previews
- [x] No combined raw table in agentic mode; the empty-Details-tab redundancy is gone
- [x] Dead "Raw results" toggle block (show_raw_toggle flag loop) removed
- [x] Tests: exactly one expander, correct labels, collapsed default, table-vs-sub-windows per mode, dead block gone
