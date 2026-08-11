# 01 — On-demand Hindsight recall gate (trigger-word signal)

**What to build:** Every query no longer injects Hindsight memories unconditionally. Recall fires only when the query contains an explicit cross-session reference word (上次/之前/记得/以前/当时/上回, last time/remember/before/...); bare pronouns never trigger. Plain lookups and same-session follow-ups pay zero recall latency and zero memory noise; cross-session explicit references still get memory injected. The timeline shows a "Hindsight memory" step when recall fires, and stays silent when it skips.

**Blocked by:** None — can start immediately.

**Status:** resolved

- [x] Plain queries ("李雅婷的邮箱是什么？") skip recall — verified by integration test (recall client invoked zero times)
- [x] Same-session pronouns ("那她的邮箱呢？") skip recall — verified by integration test
- [x] Cross-session explicit references ("记得上次说的报价吗") fire recall and inject memory into the agent input
- [x] Gate predicate is a pure module-level function, unit-tested for CN/EN trigger words, no-trigger queries, and English whole-word boundary (`before` ≠ `beforehand`)
- [x] Full suite green: 209 passed (202 pre-existing + 7 new), 0 failures — run 2026-08-11
