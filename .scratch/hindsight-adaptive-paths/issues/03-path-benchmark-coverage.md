# 03 — Benchmark coverage for temporal & relationship recall

**What to build:** Prove Hindsight's advantage in numbers. Add temporal and relationship query classes to the synthetic benchmark (`search.py synthetic-benchmark`) and to the permanent regression gate, so "time window respected" and "structural neighbor recalled" are asserted and protected.

**Blocked by:** None — can start immediately (uses the already-built temporal + graph-parallel paths).

**Status:** ready-for-agent

- [ ] Extend `SYNTHETIC_TEST_QUERIES` (search.py) with temporal classes: e.g. "上個月的物流報價" (asserts window-respecting recall, expected label + time-window)
- [ ] Add relationship classes: e.g. "發'合同已簽'消息的人還聊了什麼" / "跟陳志明聊過物流的對話" (asserts a structural neighbor is recalled)
- [ ] Add a combined class: time + relationship ("上個月跟陳志明聊的物流報價") — the case only the fused paths can answer
- [ ] `tests/test_search_regression.py` gains the same query classes against the deterministic index (permanent gate), asserting: window respected / structural neighbor present / no-time intent unchanged
- [ ] Benchmark CLI output reports temporal & relationship class results alongside label recall

## Comments

- Spec: `.scratch/hindsight-adaptive-paths/spec.md`
- Prior art: the existing `synthetic-benchmark` label-recall classes in `search.py` and the regression-gate pattern in `tests/test_search_regression.py`.
