# 04 — Structured results + output schema + adaptive escalation

**What to build:** the enablers that make fixes 01–03 clean rather than hacks.
Tools return structured hits (`{hits, expanded_queries, filter_used}`) instead
of formatted strings; every answer carries `{answer, citations, confidence}`;
and the default path stays retrieval-first with agent escalation gated by a
cheap rule detector.

**Blocked by:** 01, 02, 03 (consumes their building blocks)

**Status:** done (tools expose structured hits; answers carry
`{answer, citations, confidence}`; rule detector gates agent escalation)

- [x] Message/contact tools expose structured results; the formatted display
      string becomes a rendering concern (regex-scraping in the fallback
      answer path removed)
- [x] Deterministic output schema: `{answer, citations: [hit_id], confidence}`
      — confidence derived from the evidence gate + hit scores, never an LLM
      claim
- [x] Rule detector (multi-hop / cross-session / time keywords) selects agent
      mode; retrieval-first is the default, keeping cost/latency bounded
- [x] Unit tests for the schema + detector (deterministic, no LLM)
- [x] Full existing suite green

## Comments
- Spec: `.scratch/agent-smartness-p0/spec.md`
- The structured-results change is the same seam family as the earlier
  tool-result-channel work (per-call meta); this extends it to the answer path.
