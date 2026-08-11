"""
Vigorous integration test for app.py intent classification and tool calling.

Runs in the conda 'ocr' environment where all dependencies are installed.
Tests the CrossTableAgent.process() path that app.py uses for agent mode.
"""
import sys
import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

from apps.corpchat.search.cross_table_agent import CrossTableAgent

PASS = 0
FAIL = 0
FAILURES = []

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        FAILURES.append((name, detail))
        print(f"  ❌ {name}: {detail}")

def test_intent_classification():
    """Test that greetings/system questions are handled without tools."""
    print("\n=== Intent Classification Tests ===")
    agent = CrossTableAgent()

    # Greetings → no tools, quick response
    for greeting in ["你好", "hello", "hi", "嗨", "哈囉"]:
        result = agent.process(greeting)
        check(f"Greeting '{greeting}' → no fallback", result.get("fallback") is False,
              f"got fallback={result.get('fallback')}")
        check(f"Greeting '{greeting}' → no tools", len(result.get("tool_calls", [])) == 0,
              f"got {len(result.get('tool_calls', []))} tools")
        check(f"Greeting '{greeting}' → has output", bool(result.get("output")),
              "empty output")

    # System questions → no tools, quick response
    for sys_q in ["你能做什么？", "what can you do?", "你是谁", "help"]:
        result = agent.process(sys_q)
        check(f"System '{sys_q}' → no fallback", result.get("fallback") is False,
              f"got fallback={result.get('fallback')}")
        check(f"System '{sys_q}' → no tools", len(result.get("tool_calls", [])) == 0,
              f"got {len(result.get('tool_calls', []))} tools")
        check(f"System '{sys_q}' → has output", bool(result.get("output")),
              "empty output")

def test_tool_calling():
    """Test that search queries trigger the right tools."""
    print("\n=== Tool Calling Tests ===")
    agent = CrossTableAgent()

    # Message search → search_messages
    result = agent.process("帮我查一下诈骗相关的消息")
    tools = [tc.get("tool") for tc in result.get("tool_calls", [])]
    check("Fraud query → no fallback", result.get("fallback") is False,
          f"got fallback={result.get('fallback')}, output={result.get('output','')[:100]}")
    check("Fraud query → search_messages called", "search_messages" in tools,
          f"got tools={tools}")
    check("Fraud query → has output", bool(result.get("output")), "empty output")

    # Contact query → search_contacts
    result = agent.process("李雅婷的邮箱是什么？")
    tools = [tc.get("tool") for tc in result.get("tool_calls", [])]
    check("Contact query → no fallback", result.get("fallback") is False,
          f"got fallback={result.get('fallback')}")
    check("Contact query → search_contacts called", "search_contacts" in tools,
          f"got tools={tools}")

    # Cross-table query → both tools
    result = agent.process("发'合同已签'消息的人，他的邮箱是什么？")
    tools = [tc.get("tool") for tc in result.get("tool_calls", [])]
    check("Cross-table query → no fallback", result.get("fallback") is False,
          f"got fallback={result.get('fallback')}")
    check("Cross-table query → search_messages called", "search_messages" in tools,
          f"got tools={tools}")
    check("Cross-table query → search_contacts called", "search_contacts" in tools,
          f"got tools={tools}")

    # English cross-table
    result = agent.process("who did 何建明 spoke to?")
    tools = [tc.get("tool") for tc in result.get("tool_calls", [])]
    check("English cross-table → no fallback", result.get("fallback") is False,
          f"got fallback={result.get('fallback')}")
    check("English cross-table → both tools", "search_messages" in tools and "search_contacts" in tools,
          f"got tools={tools}")

    # Message-only query
    result = agent.process("今天有什么新消息？")
    tools = [tc.get("tool") for tc in result.get("tool_calls", [])]
    check("Message-only → no fallback", result.get("fallback") is False,
          f"got fallback={result.get('fallback')}")
    check("Message-only → search_messages called", "search_messages" in tools,
          f"got tools={tools}")

def test_answer_quality():
    """Test that answers are non-empty and relevant."""
    print("\n=== Answer Quality Tests ===")
    agent = CrossTableAgent()

    # Fraud query should produce a meaningful answer
    result = agent.process("帮我查一下诈骗相关的消息")
    output = result.get("output", "")
    check("Fraud answer non-empty", bool(output), "empty output")
    check("Fraud answer not 'no info'", "没有找到" not in output and "没有相关" not in output,
          f"got: {output[:150]}")

    # Contact query should produce contact info
    result = agent.process("李雅婷的邮箱是什么？")
    output = result.get("output", "")
    check("Contact answer non-empty", bool(output), "empty output")
    check("Contact answer has email", "邮箱" in output or "Email" in output or "@" in output,
          f"got: {output[:150]}")

    # Cross-table query should produce structured answer
    result = agent.process("发'合同已签'消息的人，他的邮箱是什么？")
    output = result.get("output", "")
    check("Cross-table answer non-empty", bool(output), "empty output")
    check("Cross-table answer has email", "邮箱" in output or "Email" in output or "@" in output,
          f"got: {output[:150]}")

def test_process_timeline():
    """Test that the process timeline is populated."""
    print("\n=== Process Timeline Tests ===")
    agent = CrossTableAgent()
    result = agent.process("帮我查一下诈骗相关的消息")
    steps = result.get("steps", [])
    check("Timeline has steps", len(steps) > 0, f"got {len(steps)} steps")
    check("Timeline has agent routing", any(s.get("label") == "Agent routing" for s in steps),
          f"labels: {[s.get('label') for s in steps]}")
    check("Timeline has search_messages", any(s.get("label") == "search_messages" for s in steps),
          f"labels: {[s.get('label') for s in steps]}")
    check("Timeline has answer generation", any(s.get("label") == "Answer generation" for s in steps),
          f"labels: {[s.get('label') for s in steps]}")

if __name__ == "__main__":
    print("=" * 60)
    print("CorpChat App Integration Tests (conda ocr env)")
    print("=" * 60)

    test_intent_classification()
    test_tool_calling()
    test_answer_quality()
    test_process_timeline()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if FAILURES:
        print("\nFailures:")
        for name, detail in FAILURES:
            print(f"  ❌ {name}: {detail}")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
