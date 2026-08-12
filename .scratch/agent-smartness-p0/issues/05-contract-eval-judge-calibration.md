# 05 — Contract-domain eval set + judge calibration

**What to build:** the readiness artifact for the oa-rag purpose. A 30–50
question eval set over contract-like data (parties, clauses, amounts, dates,
obligations) reusing the existing harness, plus human calibration of the judge
(the baseline judge is the same cheap-tier model as the answerer — 20 spot-check
samples must be reviewed by a human before the headline numbers are trusted).

**Blocked by:** 01–04 (the pipeline fixes make the eval meaningful)

**Status:** done (36 contract QA pairs, baseline run, human calibration 20/20)

- [x] 30–50 contract-domain QA pairs (rule-generated from contract-like
      documents, evidence-grounded, following the adversarial pattern of
      `eval/qa_generator.py`)
- [x] Run the baseline on the contract eval set; report correctness /
      hallucination / latency / cost per type
- [x] Human calibration pass on the 20 spot-check samples; record the judge's
      agreement rate and any rubric adjustments
- [x] Set the acceptance bar from the calibrated baseline (e.g., ≥70% correct /
      ≤3% hallucination / <5s p50)
- [x] Artifact written to `eval/results/` and summarized in the spec's Further
      Notes

## Comments
- Spec: `.scratch/agent-smartness-p0/spec.md`
- This is the P2 item from the plan: it answers "smart-enough for oa-rag?",
  not "smart-enough for the chat corpus?"
