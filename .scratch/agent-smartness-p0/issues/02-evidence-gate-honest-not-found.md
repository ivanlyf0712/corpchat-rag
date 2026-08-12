# 02 — Evidence gate + honest "not found"

**What to build:** the synthesizer never guesses from weak evidence. A pure
predicate checks the question's key entities/keywords against the top-k hit
texts before synthesis; on failure the answer path returns "没有找到相关证据"
directly (no synthesizer call). This is the hallucination control.

**Blocked by:** None — can start immediately.

**Status:** done (message_content hallucination 13% → 0%, ≤3% gate)

- [x] Pure predicate `evidence_passes(question, hits) -> bool` (entity/keyword
      extraction vs. top-k hit texts; unit-tested without an LLM, prior art
      `test_eval_qa.py`)
- [x] The eval answer path short-circuits to the honest "not found" reply when
      the gate fails (no synthesizer call, no cost)
- [x] message_content hallucination falls from 13% to ≤3% on the baseline
      (acceptance gate) — measured 0%
- [x] Deterministic output: gate outcome feeds `confidence`
      (low/medium/high), not an LLM claim
- [x] Full existing suite green

## Comments
- Spec: `.scratch/agent-smartness-p0/spec.md`
- Root cause from baseline: 13% halluc on message_content — the "only use
  context" prompt alone doesn't stop a cheap model padding from weak evidence.
