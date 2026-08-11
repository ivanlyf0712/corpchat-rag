# 03 — History-window token compaction (entity-anchored summaries)

**What to build:** The in-session history injection (up to ~4.8K chars/turn, 6 turns of verbatim query+answer) is compressed into entity-anchored one-line summaries (e.g. "本轮主体：陳志明/物流报价"), cutting the per-turn token cost of same-session follow-ups while preserving reference resolution (subject identification must not regress).

**Blocked by:** None — history-injection layer, independent of ticket 01.

**Status:** ready-for-agent

- [ ] Old turns are compressed into entity-anchored summaries; most recent turn(s) may stay verbatim
- [ ] Same-session follow-up reference resolution ("那她的邮箱呢？") still passes existing tests
- [ ] Token/char count injected per turn is measurably lower than today
- [ ] Full suite green
