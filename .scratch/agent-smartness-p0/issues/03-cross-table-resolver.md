# 03 — Cross-table resolver (message → contact party detail)

**What to build:** party-detail questions ("发过关于 X 的消息的 Y，他的公司
是？") answer in one step. After a message hit that names a sender, a
deterministic resolver looks up sender → contact company/email from the
contacts index and appends a structured "party detail" block — no agent loop
needed for the common pattern.

**Blocked by:** None — can start immediately.

**Status:** done (multi_hop_entity 34% → 100%)

- [x] Pure resolver over a message hit + contacts index:
      sender name → company/email (deterministic, fake-index tests, prior art
      `test_tools_expansion.py`)
- [x] The eval answer path calls the resolver after a message hit whose sender
      is a known contact; the structured "party detail" block is appended
- [x] Baseline re-run: multi_hop_entity correctness rises from 34% (acceptance
      gate) — measured 100%
- [x] No behavior change when the sender is not a known contact (graceful)
- [x] Full existing suite green

## Comments
- Spec: `.scratch/agent-smartness-p0/spec.md`
- Root cause from baseline: the single-search answer path cannot chain
  message → contact; the agent's `search_contacts` tool exists but is never
  called in the non-agent path.
