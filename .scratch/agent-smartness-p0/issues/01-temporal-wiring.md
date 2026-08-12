# 01 — Temporal wiring (bare YYYY-MM rule + label filter pass-through)

**What to build:** time-windowed questions ("2026-07 关于 product_inquiry")
return the right period. The rule-first time parser learns the bare `YYYY-MM`
form, and the answer path extracts a known label from the question and passes
it as a metadata filter so the existing date-filter + enlarged-limit behavior
fires.

**Blocked by:** None — can start immediately.

**Status:** done (baseline: temporal_window 16% → 24%)

- [x] `TimeExpressionParser` gains a bare `YYYY-MM` rule (alongside `YYYY年M月`
      and `YYYY-MM-DD`); unit-tested with a fixed `now` (prior art
      `test_search_temporal.py`)
- [x] A pure helper derives `{label_filter, date_from, date_to}` from a
      question (known label token + time window)
- [x] The eval answer path consumes the derived filter (window + label), so the
      retrieval seam returns windowed, label-scoped hits
- [x] Baseline re-run: temporal_window correctness rises from 16% (acceptance
      gate; ~$0.10 / ~30 min)
- [x] Full existing suite green

## Comments
- Spec: `.scratch/agent-smartness-p0/spec.md`
- Root cause from baseline: bare `2026-07` had no parser rule → no date filter;
  label never used as filter.
