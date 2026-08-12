# Baseline — 10k synthetic corpus (current pipeline, cheap-tier judge)

Date: 2026-08-12 · index: 9,100 chunks (bge-m3) · contacts: 30 · seed: 42
Command: `python eval/run_baseline.py --index apps/corpchat/search_index \
  --contacts-index apps/corpchat/contacts_index --qa-count 200 --seed 42 --spot-check 20`

## Headline

| metric | value |
|---|---|
| questions | 200 |
| **answer correctness** | **45.0%** |
| grounded | 97.5% |
| hallucination rate | 3.0% |
| latency p50 / p95 / avg | 3378 / 4608 / 3275 ms |
| tokens (in/out/calls) | 255,938 / 24,698 / 697 |
| est. cost | $0.096 (≈ $0.48 / 1k queries) |

## By question type

| type | n | correct | grounded | halluc | p50 |
|---|---|---|---|---|---|
| cross_session | 49 | 47% | 98% | 2% | 3587ms |
| message_content | 39 | 62% | 90% | 13% | 3380ms |
| multi_hop_entity | 41 | 34% | 100% | 0% | 3380ms |
| negation | 26 | 85% | 100% | 0% | 1705ms |
| temporal_window | 45 | 16% | 100% | 0% | 3169ms |

## Reading

- **Never invents, but often misses:** hallucination 3% overall but 13% on
  message_content (the one type where the LLM pads from weak evidence);
  temporal_window is the worst (16%) — the "last month about label X" questions
  rarely land the right evidence with the current label-only query.
- **Retrieval is the lever, not the LLM:** grounded is ~100% on the hard types,
  which means the answerer stays honest; correctness is capped by recall.
- **The judge is self-referential** (DeepSeek grades DeepSeek). See
  `spotcheck-10k.md` — 20 samples need human calibration before this number is
  trusted.
- **Corpus caveat:** 9,100 rows / 438 unique contents (slot filler raises the
  140-template ceiling; low-slot templates still collide ~20×). The number is
  optimistic vs. a truly-diverse 10k.

## Next (per the "measure first, set the bar" agreement)

Target the worst types first, then set the acceptance bar from this baseline:
1. temporal_window — route time expressions to the temporal path (already built)
   instead of the label-only query; add date filter + larger limit.
2. message_content — retrieval precision (the 13% hallucination is a grounding
   problem: tighten the synthesis prompt / evidence gate).
3. multi_hop_entity — cross-table join on the answer path (contact lookup after
   the message hit).
4. Human-calibrate the judge on the 20 spot-check samples; then decide the bar
   (e.g. ≥70% correct / ≤3% hallucination at <5s p50) and re-run.
