# 04 — Fusion weight tuning against the benchmark

**What to build:** Use the benchmark coverage from ticket 03 to set `GRAPH_RETRIEVAL_WEIGHT` (and temporal behavior) so orthogonal evidence boosts recall without overwhelming direct matches. The current 0.8 was flagged as potentially too high (graph neighbors can outrank direct matches).

**Blocked by:** 03 (needs the benchmark classes to measure against).

**Status:** ready-for-agent

- [ ] Run the extended benchmark at the current defaults and record the temporal/relationship/label-recall results
- [ ] Sweep `GRAPH_RETRIEVAL_WEIGHT` (e.g. 0.3 / 0.5 / 0.8) and record the trade-off: structural recall vs. direct-match precision
- [ ] Set a default that keeps content precision (regression gate green) while surfacing structural neighbors for relationship queries
- [ ] Verify the chosen default with the full pytest suite (124 + new classes) green
- [ ] Document the chosen values and rationale in this ticket's Comments

## Comments

- Spec: `.scratch/hindsight-adaptive-paths/spec.md`
- Depends on: 03 (benchmark coverage is the measurement prerequisite).
- Config knobs: `GRAPH_RETRIEVAL_WEIGHT`, `GRAPH_PARALLEL_HOP_DISCOUNT`, `GRAPH_PARALLEL_SEED_LIMIT` (config.py).
