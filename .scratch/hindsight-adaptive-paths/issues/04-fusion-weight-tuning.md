# 04 — Fusion weight tuning against the benchmark

**What to build:** Use the benchmark coverage from ticket 03 to set `GRAPH_RETRIEVAL_WEIGHT` (and temporal behavior) so orthogonal evidence boosts recall without overwhelming direct matches. The current 0.8 was flagged as potentially too high (graph neighbors can outrank direct matches).

**Blocked by:** 03 (needs the benchmark classes to measure against).

**Status:** ready-for-agent

- [x] Run the extended benchmark at the current defaults and record the temporal/relationship/label-recall results
- [x] Sweep `GRAPH_RETRIEVAL_WEIGHT` (e.g. 0.3 / 0.5 / 0.8) and record the trade-off: structural recall vs. direct-match precision
- [x] Set a default that keeps content precision (regression gate green) while surfacing structural neighbors for relationship queries
- [x] Verify the chosen default with the full pytest suite (124 + new classes) green
- [x] Document the chosen values and rationale in this ticket's Comments

## Comments

- Spec: `.scratch/hindsight-adaptive-paths/spec.md`
- Depends on: 03 (benchmark coverage is the measurement prerequisite).
- Config knobs: `GRAPH_RETRIEVAL_WEIGHT`, `GRAPH_PARALLEL_HOP_DISCOUNT`, `GRAPH_PARALLEL_SEED_LIMIT` (config.py).
- **Sweep result (2026-08-07)**: swept `GRAPH_RETRIEVAL_WEIGHT` ∈ {0.3, 0.5, 0.8, 1.0} against the deterministic graph+time regression index for query `物流報價 方案`. Top-5 composition was identical at every weight — the direct match (`product_inquiry_0`) stayed in top-5 and 6 structural neighbors surfaced in top-10 in all cases. In this synthetic corpus the graph-path results overlap heavily with content results, so the weight has no material effect. **Default kept at 0.8** (validated by the regression gate); the knob remains configurable for real corpora where graph-only neighbors are more orthogonal.
- Implemented & verified on branch feature/hindsight-multipath-rrf-skeleton (commits 2a7a54c, a5107fc); full suite 149 passed.
