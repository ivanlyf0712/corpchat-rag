# 04 — Gate mode config toggle (always / trigger-only / never)

**What to build:** The recall gate's mode is switchable without code changes — `always` (pre-gate behavior), `trigger-only` (default, current), or `never` (recall disabled) — so real-usage tuning can adjust gate sensitivity without a redeploy. Wired through the same config surface that already carries `hindsight_bank`.

**Blocked by:** 01 — On-demand Hindsight recall gate (needs the landed predicate seam).

**Status:** ready-for-agent

- [ ] Mode configurable (config/env surface); default is `trigger-only`
- [ ] `always` restores pre-gate behavior; `never` disables recall entirely
- [ ] Gate predicate is untouched by the mode switch (single seam)
- [ ] Tests cover all three modes
