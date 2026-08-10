# Spec: Persona layer — tunable disposition & answer style

**Status:** ready-for-agent

**Feature slug:** `hindsight-persona`

## Problem Statement

Every answer is generated in the same one-size-fits-all voice: the RAG answer prompt, the Agent answer prompt, and the cross-table agent's system prompt all instruct a neutral assistant. There is no way to tune how the agent answers — skepticism toward weak evidence, literal adherence to retrieved text, empathy toward the customer, or output style. Hindsight's CARA layer (disposition traits driving stylized output) is the piece the product now wants: the ability to tune the agent's personality.

## Solution

Introduce a `DispositionProfile` (skepticism, literality, empathy, style) and condition answer generation on it at the three existing answer points. Profiles are persisted per session (reusing the existing `agent_memory` DB pattern), exposed through the UI as tunable sliders, and degrade gracefully to today's neutral prompts when the LLM is unavailable or a profile is absent.

## User Stories

1. As an operator, I want to tune the agent's skepticism/literality/empathy/style per session, so that answers match the intended persona.
2. As an end user, I want the tuned persona applied consistently across all three answer paths (RAG QA, Agent, cross-table), so that the voice does not vary by entry point.
3. As an operator, I want the profile persisted per session, so that tuning survives page reloads and follow-up turns.
4. As a developer, I want persona to be prompt conditioning only (no fine-tuning, no new runtime services), so that the POC stays maintainable.
5. As an operator, I want graceful degradation: if the LLM is down or no profile is set, answers fall back to today's neutral prompts.

## Implementation Decisions

1. **`DispositionProfile`** (new module `apps/corpchat/search/persona.py`): dataclass with `skepticism` (0..1), `literality` (0..1), `empathy` (0..1), `style` (`concise`/`balanced`/`detailed`), and `build_system_prompt(base_prompt) -> str` that appends trait-specific instructions (e.g. high skepticism → "对检索证据不足的結論要標註不確定性"; high literality → "嚴格依據檢索原文回答"; high empathy → "先回應情緒再給資訊"; style → 詳略/結構指令).
2. **Persistence** — extend the existing `agent_memory` table (or add a `disposition_profiles` table) keyed by `session_id`, with `load_profile`/`save_profile` in `core/db.py` following the existing psycopg2 pattern.
3. **Three injection points** — `app.py::generate_answer_litellm`, `agent.py::Agent._generate_answer`, and `cross_table_agent.py` (SYSTEM_PROMPT / `_llm_summarize`): prepend `profile.build_system_prompt(...)` to the existing system prompt.
4. **UI** — a "Persona" expander with sliders (怀疑度/字面性/共情度) and a style select, saved to the session profile.
5. **Default profile** — neutral (all 0.5, `balanced`) so existing answers are unchanged unless tuned.
6. **Graceful degradation** — profile is applied only when the LLM call happens; if the LLM is down, the existing fallback answer formatting is unchanged.

## Testing Decisions

- **Persona unit tests** — `DispositionProfile.build_system_prompt` includes trait-specific instructions for each setting; default profile appends nothing substantive.
- **Integration** — the three answer points prepend the profile; assert via mocked LLM client that the system prompt contains the trait instruction when a non-default profile is set, and the neutral prompt when not.
- **Full existing suite stays green** — the permanent regression gate (default neutral profile = no behavior change).
- **Prior art**: the deterministic in-memory index pattern and the existing `agent_memory` persistence tests.

## Out of Scope

- **Model fine-tuning or user-modeling ML** — prompt conditioning only.
- **Profile auto-inference from conversation history** — manual tuning first; inference is a future extension.
- **Persona affecting retrieval weights** (e.g. high skepticism boosting graph/temporal evidence) — noted as a future extension, not in this spec.
- **Neo4j** — rejected for the POC.

## Further Notes

- Background: `docs/hindsight-integration-plan.md` (CARA was deferred in the selective-adoption discussion; the product now has explicit demand for personality tuning, so it is being delivered).
- The three injection points were identified in the integration analysis (§4.5): `app.py::generate_answer_litellm`, `agent.py::_generate_answer`, `cross_table_agent.py` SYSTEM_PROMPT / `_llm_summarize`.
