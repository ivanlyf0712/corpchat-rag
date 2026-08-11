# Agent UI Polish — Map

## Notes
Feature to polish the cross-table agent UX based on user feedback.

## Decisions-so-far
- Merged "Agentic mode" and "Cross-table agent" into single "🤖 Agent" toggle (R1)
- Added process timeline with timing breakdown (R2)
- Language-aware responses: en > zh-TW > zh-CN (R3)
- Improved answer formatting: person identity first, then email (R4)

## Fog
- Need to verify LangGraph hallucination detection works in production
- Need to ensure fallback mode produces structured output consistently