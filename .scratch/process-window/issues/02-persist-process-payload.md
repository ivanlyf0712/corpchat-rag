# 02 — Persist per-tool process payload per turn

**What to build:** Each completed turn stores a structured process payload so the Process window survives Streamlit reruns: per-tool info (name, query used, expanded queries, hit count, top-5 previews) plus the window label data (agentic/fallback badge, tool count, total ms). Non-agent turns store the existing raw_hits table.

**Blocked by:** 01 — Agentic message-search expansion + rerank

**Status:** ready-for-agent

- [x] Agentic turns persist a `process` payload keyed per turn in session history
- [x] Payload contains per-tool sub-window data: tool name, query used, expanded queries, hit count, top-5 previews
- [x] Payload contains label data: agentic vs fallback badge, number of tools
- [x] Non-agent turns keep storing raw_hits (table unchanged)
- [x] Payload survives a Streamlit rerun (no data loss)
- [x] Tests: turn persistence via the recording-fake-streamlit seam
