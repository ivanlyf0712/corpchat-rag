# 01 — Persona profile & answer stylization

**What to build:** Add a tunable `DispositionProfile` (skepticism / literality / empathy / style) that conditions answer generation at all three answer points, persisted per session, with UI tuning and graceful degradation.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] New `apps/corpchat/search/persona.py`: `DispositionProfile` dataclass (skepticism, literality, empathy, style) + `build_system_prompt(base_prompt) -> str` appending trait instructions; neutral default (0.5, balanced) appends nothing substantive
- [ ] Persistence: `load_profile(session_id)` / `save_profile(session_id, profile)` in `core/db.py` (extend `agent_memory` or add `disposition_profiles` table, psycopg2 pattern)
- [ ] Inject at `app.py::generate_answer_litellm`: prepend `profile.build_system_prompt(...)` to the system prompt
- [ ] Inject at `agent.py::Agent._generate_answer`: same prepend
- [ ] Inject at `cross_table_agent.py` (SYSTEM_PROMPT / `_llm_summarize`): append trait instructions
- [ ] UI: "Persona" expander with sliders (怀疑度/字面性/共情度) + style select, persisted to the session profile
- [ ] Graceful degradation: no profile / LLM down → today's neutral prompts and fallback formatting unchanged
- [ ] Tests: persona unit tests (trait instructions present per setting; default neutral) + integration (mocked LLM: non-default profile → system prompt contains trait instruction) + full suite green

## Comments

- Spec: `.scratch/hindsight-persona/spec.md`
- Background analysis: `docs/hindsight-integration-plan.md` §4.5 (integration points).
- Previously deferred for lack of demand evidence; explicit product demand now exists (personality tuning requested).
