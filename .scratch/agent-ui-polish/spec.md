# Agent UI Polish — Spec

## Objective
Polish the cross-table agent UX: merge redundant controls, add process visibility, use language-aware responses, and improve answer formatting.

## Requirements

### R1: Merge agent controls
Replace "Agentic mode" + "Cross-table agent" checkboxes with a single unified "🤖 Agent" toggle. When enabled, ALL queries route through CrossTableAgent. When disabled, the original pipeline runs.

### R2: Process timeline visible after answer
CrossTableAgent.process() returns a `steps` list. app.py renders these inside a `st.expander("🤖 Agent Process")` that stays open below the answer. Each step shows icon, label, duration, and detail text.

### R3: Language-aware responses
Detect input language (English / Traditional Chinese / Simplified Chinese). All system prompts, tool descriptions, and answer formatting respect detected language. English is the default fallback.

### R4: Improved answer formatting for cross-table queries
For queries like "发'合同已签'消息的人，他的邮箱是什么？":
1. State WHO sent the message (name + userid)
2. Show WHAT they sent (message preview)
3. THEN answer the email

## Motivation
- Users confused by two separate agent checkboxes
- No visibility into agent reasoning steps
- Answers don't clearly identify the person before showing their email
- Hallucinated answers when LangGraph fails silently

## Notes
- See individual tickets for implementation details
- All changes are backward-compatible
- Fallback mode remains the safety net