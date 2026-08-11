# 04 — Unified fade-in/out processing animation

**What to build:** Both modes share one unified live-activity display. During processing, stage labels animate in with a fade effect, hold while the stage actually runs, then fade out as the next stage starts: `routing...` → `using search_messages...` (expanded queries appear beneath) → `using search_contacts...` → `generating answer...`. The two separate `st.status` code paths collapse into one.

**Blocked by:** 03 — Unified "Process" window UI

**Status:** ready-for-agent

- [x] One shared status component drives both agent and non-agent processing
- [x] Stage labels fade in → hold until the stage completes → fade out (no artificial sleeping)
- [x] Agent mode shows: routing → using search_messages (with expanded queries beneath) → using search_contacts (only if that tool runs) → generating answer
- [x] Non-agent mode shows the equivalent pipeline stages
- [x] Reuses the existing fadeInRight CSS animation pattern
- [x] Tests: processing renders stage labels; both modes use the same status component
