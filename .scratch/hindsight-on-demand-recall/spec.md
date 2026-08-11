# Spec: On-demand Hindsight recall gate (memory trigger-word signal)

**Status:** ready-for-agent

**Feature slug:** `hindsight-on-demand-recall`

## Problem Statement

Every query processed by the cross-table agent injects Hindsight cross-session memories **unconditionally** whenever a `hindsight_bank` is configured. This has three costs:

- **Context pollution**: unrelated memories are appended to the agent input (up to 5×200 chars), with an instruction to ignore irrelevant ones — tokens spent on noise.
- **Latency**: every query pays a Hindsight recall round-trip. Measured on this machine: **~0.4 s steady state** (2.8 s cold), even when nothing in the bank could help.
- **Tool dependency**: injected memories bias the agent toward leaning on memory instead of calling the real search tools.

Not every query needs cross-session memory. Same-session references are resolved by in-session history injection (0 ms, verbatim); the bank's unique value is **rare cross-session recall**. The decision "does this query need memory?" must be made cheaply — without an extra LLM call (slower than recall itself) and without pre-running recall (which saves nothing).

## Solution

Gate Hindsight recall behind a cheap rule-based predicate: recall fires **only when the query contains an explicit cross-session reference word** (上次 / 上回 / 上上次 / 之前 / 以前 / 记得 / 还记得 / 当时 / 那次 / 上次说 / 上次聊 / 上回说 / 上回聊 / 之前说 / 之前聊 / 以前说 / 以前聊 / 那件事, and English `last time` / `remember` / `before` / `previously` / `earlier` / `as i said` / `we discussed`). **Bare pronouns (她 / 他 / 这个 / 那个) deliberately do NOT trigger** — in-session reference resolution is the job of the in-session history injection, and firing recall on them wastes a call.

The user's real usage validates this split:

- **Most common usage is same-session follow-ups** whose value is subject identification — served by in-session history injection (0 ms, verbatim Q+A), not by Hindsight (measured ~0.4 s, fuzzy, redundant content for the same session).
- **Cross-session recall is rare**; when it happens it is usually explicit ("记得上次说的报价吗"), so a trigger-word gate fires in exactly the rare cases the bank exists for.

**Asymmetry is deliberate**: recall (read) is gated; retain (write) stays unconditional — you cannot recall what you never retained, and the gate already removes the latency/context cost that motivated "on-demand".

## User Stories

1. As a user asking a plain lookup ("李雅婷的邮箱是什么？"), I want Hindsight recall skipped, so that I get the answer without extra latency or memory noise.
2. As a user asking a same-session follow-up ("那她的邮箱呢？"), I want the agent to resolve the reference from in-session history, so that Hindsight is not consulted redundantly.
3. As a user asking a cross-session explicit reference ("记得上次说的报价吗"), I want Hindsight memories injected, so that the agent can answer from prior sessions.
4. As a user asking an implicit memory query ("客户喜欢什么沟通方式"), I want graceful degradation to pure tool search, so that the answer is no worse than before Hindsight existed.
5. As a user asking in English ("do you remember the quote", "what did we discuss last time"), I want recall triggered too, so that cross-session memory works in both languages.
6. As a user, I want "beforehand" not to be matched by the word "before", so that unrelated queries don't pay recall (whole-word matching for single English words).
7. As an operator, I want the gate to be a pure rule with no LLM call, so that the gate itself is faster than the recall it guards.
8. As a maintainer, I want the gate predicate isolated as a pure function, so that it is unit-testable without an index or LLM.
9. As a UI user, I want the timeline to show a "Hindsight memory" step when recall fires (with latency), so that memory usage is observable.
10. As a UI user, I want the skip case to stay silent, so that the timeline is not noisy on the normal path.
11. As a maintainer, I want retain to remain unconditional, so that the memory pool stays rich for the rare cross-session recall.
12. As an operator, I want zero-hit / near-empty retains dropped, so that the bank does not accumulate pure-noise entries.
13. As a regression maintainer, I want the full test suite to stay green plus new gate tests, so that the verified base behavior is provably preserved.

## Implementation Decisions

1. **Signal = explicit cross-session reference words only (decision 16).** Recall fires iff the query contains a trigger word. Bare pronouns are excluded: "她的邮箱" is an in-session reference and must route through history injection, not Hindsight. An implicit memory query ("客户喜欢什么沟通方式") with no trigger word is a **known, accepted miss** — it degrades to pure tool search, which is exactly the pre-Hindsight behavior.

2. **Gate implementation shape.** A module-level keyword tuple plus a pure predicate `_needs_hindsight_recall(query) -> bool` in the cross-table agent module. Matching reuses the existing greeting-gate pattern: multi-word phrases match as substrings; single words (incl. CJK) match on word boundaries. The agent's `process()` calls recall only when `hindsight_bank` is configured **and** the predicate is true, and records a timeline step ("Hindsight memory", with measured latency) when recall fires.

3. **Same-session memory is owned by history injection (Q4 ruling).** History injection (last 6 turns of verbatim query+answer summaries) stays always-on and is the sole same-session memory channel. Hindsight recall is strictly a cross-session channel. Rationale: history is 0 ms and verbatim-precise; Hindsight recall of the same-session content is a redundant fuzzy re-injection at ~0.4 s.

4. **Retain stays unconditional (Q3 ruling); recall is the only gate.** Writes remain best-effort and async, and the existing empty-content guard (`len < 10 → skip`) stays: zero-hit searches are not persisted. The bank's noise is absorbed by Hindsight-side relevance scoring + `max_results=5` truncation at recall time.

5. **Word-list boundaries (Q5 ruling).** (a) Time words (昨天/前天/上周/上个月) are deliberately **excluded** — the corpus temporal-search path owns those queries; including them would double-recall. (b) 之前 stays broad — it fires on corpus-answerable queries too, and the cost (one ~0.4 s recall, results ignored) is accepted. (c) English trigger words included, single English words whole-word matched.

6. **UI transparency (Q6 ruling).** The hit case shows a "Hindsight memory" timeline step; the skip case is silent. The skip is the designed normal path and must not add timeline noise.

7. **History-window hole is out of scope (Q2 ruling).** A subject established in turn 1 and followed up in turn 8 falls outside the 6-turn window; history cannot resolve it and the gate will not fire. Accepted as a known limitation — sessions longer than 6 turns are not expected.

## Testing Decisions

- **Test external behavior only.** The gate predicate is tested by its observable return value; `process()` is tested by what the agent input contains and whether the recall client is invoked — never by internal flags.
- **Unit seam: the gate predicate** (`_needs_hindsight_recall`) — pure, no I/O, no index, no LLM. Cases: CN trigger words, EN trigger words, plain queries (no trigger), in-session pronouns (no trigger), English whole-word boundary (`before` ≠ `beforehand`).
- **Integration seam: `process()`** — monkeypatch the Hindsight recall client and inject the existing fake LangGraph agent (same seam as the existing main-path `process()` tests). Cases: trigger-word query → recall invoked with the exact user input and the memory text reaches the agent input; plain query and in-session pronoun → recall invoked zero times.
- **Prior art**: the existing `process()` main-path tests in the cross-table-agent test module (fake agent + monkeypatched tool metadata) and the greeting-gate keyword tests (same matching pattern). The 7 new tests landed in the same test module.
- **Full suite**: 209 passed (202 pre-existing + 7 new), 0 failures.

## Out of Scope

- **History-window expansion** (6 → N turns) — same-session out-of-window reference resolution; belongs to the history-injection layer, not this gate.
- **Entity-anchor enrichment of retain** (adding person/entity names to `tags` on retain) — deferred follow-up; no evidence yet that Hindsight compaction drops entity anchors.
- **Time-word triggers** (昨天 / 上周 …) — corpus temporal search owns those queries.
- **Config toggle** for gate mode (always / trigger-only / never) — the pure predicate is a clean seam; a toggle is trivial to add later if real usage demands it.
- **LLM-based judgment** and **vector pre-check** as the signal — rejected: one is slower than recall, the other saves nothing.

## Further Notes

- Measured recall latency on this machine: ~0.4 s steady state, 2.8 s on first call (embedding model warm-up). The gate removes this cost from every non-trigger query.
- The next token lever, if ever needed, is the history injection itself (up to ~4.8K chars injected per turn when history exists) — entity-anchored compaction of old turns, a separate future task.
- Retain already writes `context` and `tags=["corpchat","search"]`; entity-name tags are the natural future enrichment.
- Testing gotcha recorded for future sessions: background pytest jobs started from a shell can be STOPPED by job control (STAT `T`, 0% CPU, empty log). Launch detached with a new session (`start_new_session=True`) and stream output to a log.
