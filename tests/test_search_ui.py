#!/usr/bin/env python3
"""
UI-flow tests for the Search page (`apps.corpchat.app._render_search_page`).

These tests catch four user-reported regressions:

  1. "Search was interrupted" is shown spuriously for a turn that is actively
     being processed (fresh `processing` turn rendered by `_render_chat_history`).
  2. Enhancement/Filters expanders auto-expand after a search completes
     (`expanded=not st.session_state.searching` pops them open).
  3. A greeting query still triggers a full search (no intent gate).
  4. The 6-stage progress window is shown for a greeting (same root cause as #3,
     but observable independently).

The seam: `_render_search_page()` is now a plain callable, so tests can drive it
with a recording fake `streamlit` module.

Run:
    conda run -n ocr pytest tests/test_search_ui.py -v
"""
import json
import os
import sys
import types

import pandas as pd
import pytest

# Ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ── Recording fake streamlit ────────────────────────────────────────────────
class _Recorder:
    """Captures what the UI called, so tests can assert on rendered output."""

    def __init__(self):
        self.infos = []          # st.info(...) messages
        self.expander_expanded = []  # expanded= values passed to st.expander
        self.expander_labels = []    # label= values passed to st.expander
        self.status_labels = []  # status.update(label=...) values
        self.search_calls = []   # _run_search invocations
        self.writes = []         # st.write(...) messages
        self.reruns = 0

    def reset(self):
        self.infos.clear()
        self.expander_expanded.clear()
        self.expander_labels.clear()
        self.status_labels.clear()
        self.search_calls.clear()
        self.writes.clear()
        self.reruns = 0


def _make_fake_streamlit(recorder: _Recorder, search_impl=None):
    """Build a fake streamlit module backed by a _Recorder."""
    st = types.ModuleType("streamlit")

    class _Ctx:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def markdown(self, *a, **k):
            return None

    def _noop(*a, **k):
        return None

    def _noop_ctx(*a, **k):
        return _Ctx()

    class _FakeStatus(_Ctx):
        def __init__(self):
            super().__init__()
        def update(self, **kwargs):
            recorder.status_labels.append(kwargs.get("label"))

    # Page setup
    st.set_page_config = _noop
    st.title = _noop
    st.markdown = _noop
    st.caption = _noop
    st.subheader = _noop
    st.info = lambda *a, **k: recorder.infos.append(a[0]) if a else None
    st.warning = _noop
    st.error = _noop
    st.success = _noop
    st.divider = _noop
    st.write = lambda *a, **k: recorder.writes.extend(str(x) for x in a if x)

    # Layout context managers
    st.empty = lambda *a, **k: _Ctx()
    st.expander = lambda *a, **k: (_Ctx(),
                                   recorder.expander_labels.append(a[0] if a else k.get("label", "")),
                                   recorder.expander_expanded.append(k.get("expanded")))[0]
    st.chat_message = _noop_ctx
    st.status = lambda *a, **k: _FakeStatus()
    st.tabs = lambda *a, **k: [_Ctx() for _ in (a[0] if a and isinstance(a[0], (list, tuple)) else a)]
    st.columns = lambda *a, **k: [_Ctx() for _ in range(len(a[0]) if a and isinstance(a[0], (list, tuple)) else (a[0] if a else 1))]

    # Widgets
    st.button = lambda *a, **k: False
    # Respect the `value=` param so tests can opt out of agent mode by setting
    # ss["agent_enabled"] = False. Other checkboxes use value=True → still True.
    st.checkbox = lambda *a, **k: k.get("value", True)
    st.radio = lambda *a, **k: (a[1][0] if len(a) > 1 and a[1] else None)
    st.selectbox = lambda *a, **k: (a[1][0] if len(a) > 1 and a[1] else None)
    st.text_input = lambda *a, **k: k.get("value", "")
    st.multiselect = lambda *a, **k: k.get("default", [])
    st.chat_input = lambda *a, **k: None
    st.number_input = lambda *a, **k: 1
    st.slider = lambda *a, **k: 10
    st.dataframe = _noop
    st.metric = _noop
    st.bar_chart = _noop
    st.iframe = _noop
    st.rerun = lambda: setattr(recorder, "reruns", recorder.reruns + 1)

    # Column config for dataframe column_config= (TextColumn / NumberColumn)
    class _ColumnConfig:
        def TextColumn(self, *a, **k):
            return a[0] if a else "text"
        def NumberColumn(self, *a, **k):
            return a[0] if a else "number"
    st.column_config = _ColumnConfig()

    # Session state
    class _FS(dict):
        def __getattr__(self, name):
            if name in self:
                return self[name]
            raise AttributeError(name)
        def __setattr__(self, name, value):
            self[name] = value
    st.session_state = _FS()

    st.cache_data = lambda *a, **k: (a[0] if a else (lambda f: f))
    st.cache_resource = lambda *a, **k: (a[0] if a else (lambda f: f))

    # Sidebar
    class _FakeSidebar:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        title = caption = divider = _noop
        radio = lambda *a, **k: "Search"
        checkbox = lambda *a, **k: True
        slider = lambda *a, **k: 10
        text_input = lambda *a, **k: ""
        def expander(self, *a, **k): return _Ctx()
    st.sidebar = _FakeSidebar()

    return st


# ── Install fake streamlit + import app ─────────────────────────────────────
_recorder = _Recorder()
_FAKE_ST = _make_fake_streamlit(_recorder)
sys.modules["streamlit"] = _FAKE_ST

# Mock DB so app.py imports cleanly
import core.db as core_db_module
class _FakeCursor:
    def execute(self, sql, *args): return None
    def fetchone(self): return (0, 0, 0, [])
    def fetchall(self): return []
    def close(self): return None
class _FakeConn:
    def cursor(self): return _FakeCursor()
    def close(self): return None
core_db_module.get_db_connection = lambda: _FakeConn()
pd.read_sql = lambda *a, **k: pd.DataFrame()

from apps.corpchat import app as app_module  # noqa: E402


@pytest.fixture(autouse=True)
def _bind_recording_st(monkeypatch):
    """Bind `app_module.st` AND `sys.modules['streamlit']` to this file's fake.

    Other test files (e.g. test_app_search.py) also replace
    `sys.modules['streamlit']` with their own fakes; whichever imported last wins
    the global slot, so `app_module.st` can point at a foreign fake with a
    different session_state and widget behavior. Rebinding both to THIS file's
    fake (`_FAKE_ST`) for every test keeps reads/writes consistent regardless of
    cross-file import order.
    """
    monkeypatch.setitem(sys.modules, "streamlit", _FAKE_ST)
    monkeypatch.setattr(app_module, "st", _FAKE_ST)
    yield
    monkeypatch.undo()


# ── Fixture: session state with a fresh processing turn ─────────────────────
def _fresh_session(query: str):
    """Return a session_state with one fresh processing turn (as if just submitted)."""
    ss = type("SS", (dict,), {})()
    ss["chat_history"] = [{"query": query, "answer": None, "raw_hits": [], "status": "processing"}]
    ss["searching"] = True
    # These tests target the ORIGINAL non-agent pipeline; agent mode defaults ON
    # in app.py, so opt out explicitly for deterministic routing.
    ss["agent_enabled"] = False
    return ss


@pytest.fixture(autouse=True)
def _stub_agent_loading(monkeypatch):
    """Don't load the real production index in UI-routing tests.

    `_render_search_page` builds `st.session_state.agent` via `load_agent()`,
    which loads the production txtai index — heavy, and under the full suite
    exhausts MPS GPU memory (RuntimeError: MPS backend out of memory), making
    later tests order/timing dependent. These tests target UI routing, not
    retrieval: a stub agent that classifies intent (rules-only, no search, no
    LLM) preserves the routing behavior while removing the production-index and
    GPU-memory dependency.
    """
    import apps.corpchat.agent as agent_module

    def _stub_agent():
        return types.SimpleNamespace(
            session_id="ui-test-session",
            # 规则纯分类 (_rule_classify): 无网络、无 LLM、无索引依赖
            process=lambda query, **kw: (app_module._intent_classifier._rule_classify(query) or "search", "", []),
        )

    monkeypatch.setattr(agent_module, "load_agent", lambda *a, **k: _stub_agent())
    monkeypatch.setattr(agent_module, "Agent", lambda *a, **k: _stub_agent())
    yield


@pytest.fixture(autouse=True)
def _clean_recorder():
    _recorder.reset()
    yield
    _recorder.reset()


# ── Bug 1: spurious "Search was interrupted" ─────────────────────────────────
def test_no_spurious_interrupted_for_fresh_processing_turn(monkeypatch):
    """A freshly-submitted processing turn must NOT render 'Search was interrupted'."""
    from streamlit import session_state as ss
    ss["chat_history"] = [{"query": "Hi", "answer": None, "raw_hits": [], "status": "processing"}]
    ss["searching"] = True
    ss["agent_enabled"] = False  # test the original non-agent pipeline

    # Stub out the search/LLM so the pending-turn handler is fast & deterministic
    monkeypatch.setattr(app_module, "_run_search", lambda *a, **k: ([], []))
    monkeypatch.setattr(app_module, "_check_llm_available", lambda: False)
    monkeypatch.setattr(app_module, "generate_answer_litellm", lambda q, c: "fallback")
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    # The write() in the status block will call write(); ensure Searcher isn't
    # reached by _run_search (already stubbed). Render the page.
    app_module._render_search_page()

    assert not _recorder.infos, (
        f"'Search was interrupted' (or other st.info) rendered spuriously: {_recorder.infos}"
    )


# ── Bug 2: filters/expanders must not auto-expand after search ───────────────
def test_expanders_not_expanded_when_idle(monkeypatch):
    """When not searching, Enhancement/Filters expanders must be collapsed."""
    from streamlit import session_state as ss
    ss["chat_history"] = []
    ss["searching"] = False
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    assert _recorder.expander_expanded, "No expander calls recorded"
    assert all(e is False for e in _recorder.expander_expanded), (
        f"Expanders auto-expanded when idle: {_recorder.expander_expanded}"
    )


# ── Bug 3: greeting must not trigger search ──────────────────────────────────
def test_greeting_does_not_call_search(monkeypatch):
    """A greeting query must NOT invoke _run_search (no wasted search)."""
    from streamlit import session_state as ss
    ss["chat_history"] = [{"query": "Hi", "answer": None, "raw_hits": [], "status": "processing"}]
    ss["searching"] = True
    ss["agent_enabled"] = False  # test the original non-agent pipeline

    # Spy on _run_search — if the greeting path is correct, it is never called.
    calls = []
    def _spy_run_search(*a, **k):
        calls.append(a[0])
        return ([], [])
    monkeypatch.setattr(app_module, "_run_search", _spy_run_search)
    monkeypatch.setattr(app_module, "_check_llm_available", lambda: False)
    monkeypatch.setattr(app_module, "generate_answer_litellm", lambda q, c: "")
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    assert calls == [], f"_run_search called for greeting: {calls}"


def test_greeting_does_not_show_progress_window(monkeypatch):
    """A greeting must not render the 6-stage progress status window."""
    from streamlit import session_state as ss
    ss["chat_history"] = [{"query": "Hi", "answer": None, "raw_hits": [], "status": "processing"}]
    ss["searching"] = True
    ss["agent_enabled"] = False  # test the original non-agent pipeline

    monkeypatch.setattr(app_module, "_run_search", lambda *a, **k: ([], []))
    monkeypatch.setattr(app_module, "_check_llm_available", lambda: False)
    monkeypatch.setattr(app_module, "generate_answer_litellm", lambda q, c: "")
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    # No status window should have appeared for a greeting
    assert _recorder.status_labels == [], (
        f"Progress status shown for greeting: {_recorder.status_labels}"
    )


# ── Search intent still works (no regression) ───────────────────────────────
def test_build_agent_process_payload_structures_tools():
    """Agent process payload captures per-tool query, expansions, hit count, previews."""
    tool_calls = [
        {
            "tool": "search_messages",
            "tool_input": "合同已签",
            "observation": "...",
            "meta": {
                "expanded_queries": ["合约确认", "合同 签署"],
                "hit_count": 10,
                "previews": [{"text": "合同已签...", "sender": "陳志明", "score": 0.61}],
            },
        },
        {
            "tool": "search_contacts",
            "tool_input": "陳志明",
            "observation": "...",
            "meta": {"hit_count": 5, "previews": [{"name": "陳志明", "email": "x@y.org", "score": 0.9}]},
        },
    ]
    payload = app_module._build_agent_process_payload(
        tool_calls, [], {"agent_fallback": False}
    )
    assert payload["agentic"] is True
    assert payload["fallback"] is False
    assert len(payload["tools"]) == 2
    assert payload["tools"][0]["name"] == "search_messages"
    assert payload["tools"][0]["expanded_queries"] == ["合约确认", "合同 签署"]
    assert payload["tools"][0]["hit_count"] == 10
    assert payload["tools"][1]["name"] == "search_contacts"
    assert payload["tools"][1]["previews"][0]["email"] == "x@y.org"


def test_build_agent_process_payload_fallback_flag():
    """Fallback badge data flows through the payload."""
    payload = app_module._build_agent_process_payload(
        [], [], {"agent_fallback": True}
    )
    assert payload["fallback"] is True


def test_search_query_still_calls_search(monkeypatch):
    """A real search query (with explicit search keyword) must still invoke _run_search."""
    from streamlit import session_state as ss
    original_query = "找物流報價 方案"
    # "找" is an explicit search keyword → rule-classified as search, deterministic
    ss["chat_history"] = [{"query": original_query, "answer": None, "raw_hits": [], "status": "processing"}]
    ss["searching"] = True
    ss["agent_enabled"] = False  # test the original non-agent pipeline

    calls = []
    def _spy_run_search(*a, **k):
        calls.append(a[0])
        return ([], [])
    monkeypatch.setattr(app_module, "_run_search", _spy_run_search)
    monkeypatch.setattr(app_module, "_check_llm_available", lambda: False)
    monkeypatch.setattr(app_module, "generate_answer_litellm", lambda q, c: "")
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)
    # 确定性: 屏蔽真实 LLM 路由调用。该测试验证的是"搜索查询必须到达
    # _run_search", 不应依赖外部 LLM 网络的即时响应 (search:true/false)。
    monkeypatch.setattr(
        app_module, "_search_router",
        types.SimpleNamespace(decide=lambda q: {"search": True, "query": q, "raw": ""}),
    )

    app_module._render_search_page()

    assert len(calls) == 1, f"Search query did not reach _run_search: {calls}"


# ── Ticket 01: unified config panel ──────────────────────────────
def test_config_panel_writes_session_config(monkeypatch):
    """配置代理面板把值写入 agent_config 并派生 agent_enabled (深度→agent)。"""
    from streamlit import session_state as ss
    ss["chat_history"] = []
    ss["searching"] = False
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    cfg = ss.get("agent_config")
    assert cfg is not None, "面板应写入 session_state.agent_config"
    assert {"persona", "search", "knowledge"} <= set(cfg.keys())
    # fake selectbox 返回第一项 "简单" → depth=simple → agent_enabled=False
    assert cfg["search"]["depth"] == "simple"
    assert ss["agent_enabled"] is False

# ── Ticket 03: Unified Process window ─────────────────────────────
def test_process_window_single_expander_agentic(monkeypatch):
    """Agentic turn renders ONE Process expander with agentic label, collapsed."""
    from streamlit import session_state as ss
    ss["chat_history"] = [{
        "query": "发'合同已签'消息的人",
        "answer": "陈志明 email is x",
        "raw_hits": [],
        "status": "done",
        "agent_fallback": False,
        "process": {
            "agentic": True,
            "fallback": False,
            "tools": [
                {"name": "search_messages", "query": "合同已签",
                 "expanded_queries": ["合约确认"], "hit_count": 3,
                 "previews": [{"text": "合同已签...", "sender": "陳志明", "score": 0.61}]},
            ],
        },
    }]
    ss["searching"] = False
    ss["agent_enabled"] = True
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    process_labels = [l for l in _recorder.expander_labels if str(l).startswith("Process")]
    assert len(process_labels) == 1, f"Expected ONE Process expander, got {process_labels}"
    assert "agentic" in process_labels[0], f"Label should mark agentic: {process_labels[0]}"
    assert "✅" in process_labels[0]
    assert _recorder.expander_expanded, "Expanders recorded"


def test_process_window_collapsed_by_default(monkeypatch):
    """Process window defaults to collapsed (expanded=False)."""
    from streamlit import session_state as ss
    ss["chat_history"] = [{
        "query": "q", "answer": "a", "raw_hits": [{"id": "1", "text": "t", "score": 0.5, "metadata": {}}],
        "status": "done",
    }]
    ss["searching"] = False
    ss["agent_enabled"] = False
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    idx = None
    for i, l in enumerate(_recorder.expander_labels):
        if l == "Process":
            idx = i
            break
    assert idx is not None, "Process expander not found"
    assert _recorder.expander_expanded[idx] is False, "Process window must be collapsed by default"


def test_process_window_agent_shows_tool_subwindows(monkeypatch):
    """Agentic turn → per-tool sub-windows inside Process (expandable)."""
    from streamlit import session_state as ss
    ss["chat_history"] = [{
        "query": "q", "answer": "a", "raw_hits": [],
        "status": "done",
        "process": {
            "agentic": True,
            "fallback": False,
            "tools": [
                {"name": "search_messages", "query": "合同已签",
                 "expanded_queries": ["合约确认", "合同 签署"], "hit_count": 3,
                 "previews": [{"text": "合同已签...", "sender": "陳志明", "score": 0.61}]},
                {"name": "search_contacts", "query": "陳志明", "expanded_queries": [],
                 "hit_count": 1,
                 "previews": [{"name": "陳志明", "email": "x@y.org", "score": 0.9}]},
            ],
        },
    }]
    ss["searching"] = False
    ss["agent_enabled"] = True
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    tool_labels = [l for l in _recorder.expander_labels if "search_" in str(l)]
    assert len(tool_labels) == 2, f"Expected 2 tool sub-windows, got {tool_labels}"
    assert any("search_messages" in str(l) for l in tool_labels)
    assert any("search_contacts" in str(l) for l in tool_labels)


def test_process_window_fallback_badge(monkeypatch):
    """Fallback agent turn → ⚠️ in the Process label."""
    from streamlit import session_state as ss
    ss["chat_history"] = [{
        "query": "q", "answer": "a", "raw_hits": [],
        "status": "done",
        "process": {"agentic": True, "fallback": True, "tools": []},
    }]
    ss["searching"] = False
    ss["agent_enabled"] = True
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    process_labels = [l for l in _recorder.expander_labels if str(l).startswith("Process")]
    assert process_labels, "Process expander not found"
    assert "⚠️" in process_labels[0], f"Fallback badge missing: {process_labels[0]}"


def test_process_window_no_dead_raw_results_block(monkeypatch):
    """The dead 'Raw results' toggle block must not render."""
    from streamlit import session_state as ss
    ss["chat_history"] = [{
        "query": "q", "answer": "a", "raw_hits": [{"id": "1", "text": "t", "score": 0.5}],
        "status": "done",
    }]
    ss["searching"] = False
    ss["agent_enabled"] = False
    ss["show_raw_toggle_" + "x"] = True
    monkeypatch.setattr(app_module, "_load_search_index", lambda: None)

    app_module._render_search_page()

    assert "Raw results" not in _recorder.expander_labels, (
        f"Dead Raw results block still renders: {_recorder.expander_labels}"
    )


# ── Ticket 04: Unified fade-in/out processing animation ──────────
def test_stage_helpers_produce_animation_html():
    """_stage_html / _fade_out_html embed the fade animations."""
    html = app_module._stage_html("🧠 routing...", "deciding")
    assert "stageFadeIn" in html
    assert "🧠 routing..." in html
    assert "deciding" in html
    fade = app_module._fade_out_html("🧠 routing...")
    assert "stageFadeOut" in fade


def test_on_stage_callback_receives_stage_labels(monkeypatch):
    """CrossTableAgent.process(on_stage=...) invokes the callback per stage."""
    import types
    from apps.corpchat.search.cross_table_agent import CrossTableAgent
    from apps.corpchat.search import tools as tools_module

    def _fake_msg_invoke(payload):
        return "【消息搜索结果】\n1. [Score: 0.61] 陳志明 (userid: user_1) [Label: sample_request]\n   合同已签"

    def _fake_contact_invoke(payload):
        return "【联系人搜索结果】\n1. [Score: 0.9] 陳志明 (userid: user_1)\n   Email: x@y.org"

    monkeypatch.setattr(tools_module, "search_messages", types.SimpleNamespace(invoke=_fake_msg_invoke))
    monkeypatch.setattr(tools_module, "search_contacts", types.SimpleNamespace(invoke=_fake_contact_invoke))

    stages = []
    agent = CrossTableAgent(expand=False, use_rerank=False)
    agent._extract_search_query = lambda q: "合同已签"
    result = agent.process("帮我查一下合同已签的消息", on_stage=lambda l, d="": stages.append(l))

    assert "🧠" in stages, f"Routing stage missing: {stages}"
    assert "🔍" in stages, f"Tool stage missing: {stages}"
    assert "✨" in stages, f"Answer stage missing: {stages}"
    assert result.get("success") is True


def test_on_stage_callback_never_breaks_processing(monkeypatch):
    """A throwing on_stage callback must not break the agent."""
    import types
    from apps.corpchat.search.cross_table_agent import CrossTableAgent
    from apps.corpchat.search import tools as tools_module

    def _fake_msg_invoke(payload):
        return "【消息搜索结果】\n1. [Score: 0.61] 陳志明 (userid: user_1) [Label: sample_request]\n   合同已签"

    def _fake_contact_invoke(payload):
        return "【联系人搜索结果】\n1. [Score: 0.9] 陳志明 (userid: user_1)\n   Email: x@y.org"

    monkeypatch.setattr(tools_module, "search_messages", types.SimpleNamespace(invoke=_fake_msg_invoke))
    monkeypatch.setattr(tools_module, "search_contacts", types.SimpleNamespace(invoke=_fake_contact_invoke))

    def _boom(label, detail=""):
        raise RuntimeError("ui failure")

    agent = CrossTableAgent(expand=False, use_rerank=False)
    agent._extract_search_query = lambda q: "合同已签"
    result = agent.process("帮我查一下合同已签的消息", on_stage=_boom)
    assert result.get("success") is True

