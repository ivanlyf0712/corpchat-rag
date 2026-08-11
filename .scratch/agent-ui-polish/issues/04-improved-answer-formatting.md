# Ticket 04: Improved answer formatting for cross-table queries

**Type:** task
**Status:** ready-for-agent
**Blocked by:** none

## Description
For cross-table queries like "发'合同已签'消息的人，他的邮箱是什么？", the answer should clearly identify WHO sent the message before answering the email.

## Acceptance Criteria
For a cross-table query (message → contact):
1. First state the person's name and userid
2. Show a preview of the message they sent
3. Then answer the specific question (email)

Example output:
```
✅ Found: **陳志明** (user_陳志明_johnsonj)

   📩 Sent: "陳經理你好，當然可以。請把合約電子檔寄給我..."

   📧 Email: **weiyao@example.org**
   🏢 Company: 鴻海精密工業股份有限公司
   📱 Phone: 13818196001
```

For direct contact queries (e.g., "李雅婷的邮箱是什么？"):
```
✅ **李雅婷**

   📧 Email: **hsin-ihu@example.org**
   🏢 Company: Example Corp
   📱 Phone: 1234567890
```

## Implementation Notes
- Modify `_format_fallback_answer()` to produce structured output with person identity first
- Modify `_llm_summarize()` prompt to include message preview + person identity before email
- When LangGraph succeeds, check if the result includes contact info and format accordingly
- Use markdown bold for key values (email, name)
- Include message preview as a quoted block