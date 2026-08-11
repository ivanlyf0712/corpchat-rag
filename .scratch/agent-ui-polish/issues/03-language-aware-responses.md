# Ticket 03: Language-aware responses (en > zh-TW > zh-CN)

**Type:** task
**Status:** ready-for-agent
**Blocked by:** none

## Description
All responses should be in the same language as the user's input. The priority is: English > Traditional Chinese > Simplified Chinese.

## Acceptance Criteria
1. Detect input language in CrossTableAgent.process()
2. System prompt and tool descriptions use detected language
3. Answer formatting respects detected language
4. English is the default fallback
5. Language detection heuristic:
   - If any CJK character present → Chinese
   - If Traditional Chinese chars detected (維, 認, 體, 機, 關, 係, etc.) → zh-TW
   - Otherwise → zh-CN (Simplified) or en (no CJK chars)
6. Quick responses (greetings, system info) are also language-aware

## Implementation Notes
- Add `_detect_language(user_input) -> str` method to CrossTableAgent
- Return value: "en", "zh-TW", or "zh-CN"
- Store language in the result dict: `{"output": "...", "language": "zh-TW", ...}`
- System prompt should be selected based on detected language
- Keep English as default for ambiguous cases