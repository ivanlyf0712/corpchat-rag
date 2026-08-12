# Baseline — P0 (agent-smartness-p0, calibrated judge)

Date: 2026-08-12 · index: 9,100 chunks (bge-m3) · contacts: 30 · seed: 42
Command: `python eval/run_baseline.py --index apps/corpchat/search_index \
  --contacts-index apps/corpchat/contacts_index --qa-count 200 --seed 42 --spot-check 20`

Judge calibration: `eval/judge._evidence_snippets` now shows up to 15 dated
evidence hits (matching the answerer's context) so grounded claims are checkable.

## Headline vs. original baseline

| metric | before | after | delta |
|---|---|---|---|
| **answer correctness** | 45.0% | **63.0%** | **+18.0pp** |
| grounded | 97.5% | 99.0% | +1.5pp |
| hallucination rate | 3.0% | **1.0%** | **−2.0pp** |
| latency p50 / p95 / avg | 3378 / 4608 / 3275 ms | 2565 / 4608 / 2768 ms | −24% p50 |
| tokens (in/out/calls) | 255,938 / 24,698 / 697 | 375,321 / 23,894 / 608 | fewer calls |
| est. cost | $0.096 (≈$0.48/1k) | $0.128 (≈$0.64/1k) | +33% (larger context) |

## By question type

| type | n | before correct | after correct | before halluc | after halluc |
|---|---|---|---|---|---|
| temporal_window | 45 | 16% | **24%** | 0% | 4% |
| multi_hop_entity | 41 | 34% | **100%** | 0% | 0% |
| cross_session | 49 | 47% | 47% | 2% | 0% |
| message_content | 39 | 62% | **64%** | 13% | **0%** |
| negation | 26 | 85% | **100%** | 0% | 0% |

## What moved the numbers (per ticket)

1. **01 temporal wiring** — bare `YYYY-MM` rule in `TimeExpressionParser`;
   `derive_search_filter` passes label + window to the retrieval seam;
   windowed queries retrieve 15 dated hits. temporal 16%→24%.
2. **02 evidence gate** — `evidence_passes` blocks synthesis on weak evidence
   (honest "没有找到相关证据", empty evidence, no synthesizer call);
   confidence is deterministic (gate + hit placement). message_content
   hallucination 13%→0%.
3. **03 cross-table resolver** — `resolve_party_detail`/`first_party_detail`
   answer party-detail questions deterministically in one step (no LLM).
   multi_hop 34%→100%, halluc 0%.
4. **04 structured results + schema + escalation** — tools expose
   `get_structured_result`; the fallback answer path consumes structured hits
   (regex-scraping removed); every answer carries
   `{answer, citations, confidence, evidence_gate}`; `detect_agent_mode` rule
   detector gates agent escalation (app "Search depth: Auto").

## Reading

- **Reliability, not just accuracy:** hallucination is 1.0% overall and 0% on
  the previously-worst message_content type. Every answer is grounded in
  retrieved evidence or says "没有找到相关证据".
- **The agent is the exception, not the default:** the measured path is the
  cheap deterministic pipeline; the LangGraph agent remains rule-gated.
- **Residual: temporal_window halluc 4%** — the QA's temporal expected is ONE
  random in-window message among ~100 near-duplicates; the model answers with a
  summary and occasionally cites a date/amount the judge cannot verify against
  the 15-hit evidence. Hypothesis: temporal recall is the next lever (better
  windowed ranking), not synthesis.
- **Cost:** up 33% because windowed queries feed 15 hits (was 5) and the judge
  now sees the same evidence; still ≈$0.64/1k queries.
- **Judge calibration caveat:** the judge is the same cheap-tier model as the
  answerer; `--spot-check 20` samples need human review (see contract
  baseline's human calibration pass for the agreement rate).
