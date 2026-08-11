"""
CorpChat Search — Agent Configuration Model
=============================================
Unified agent configuration for the "配置代理" panel.

Single source of truth for CARA personality, search strategy, and knowledge
scope. Stored in `st.session_state.agent_config` (session-level, immediate
effect) and persisted per session via the `agent_config` DB table.

Values:
  - persona.skepticism / literality / empathy: 0-10 (UI slider scale)
  - persona.style: concise | standard | detailed (= 回答长度)
  - search.depth: simple | deep  (= agent_enabled: single-step vs cross-table)
  - knowledge.sources: subset of {messages, contacts}
  - knowledge.citations: bool
"""

from copy import deepcopy
from typing import Dict

# CARA 预设 (0-10 滑杆值 + 回答长度)
CARA_PRESETS: Dict[str, Dict] = {
    "audit":    {"skepticism": 8, "literality": 7, "empathy": 3, "style": "standard"},
    "service":  {"skepticism": 3, "literality": 4, "empathy": 8, "style": "concise"},
    "research": {"skepticism": 6, "literality": 6, "empathy": 5, "style": "detailed"},
}

PRESET_LABELS = {
    "审计助手": "audit",
    "客服助手": "service",
    "研究助理": "research",
    "自訂": "custom",
}

# 回答长度 (UI 标签 → style key), 顺序即下拉选项顺序
STYLE_LABELS = {
    "简洁": "concise",
    "标准": "standard",
    "详细": "detailed",
}

# 数据源 (UI 标签 ↔ 内部 key); multiselect 的 options 与 default 都必须用标签
SOURCE_LABELS = {
    "消息": "messages",
    "联系人": "contacts",
}
SOURCE_OPTIONS = list(SOURCE_LABELS.keys())


def sources_to_labels(sources) -> list:
    """内部 key 列表 → 显示标签列表 (multiselect default 用)。"""
    rev = {v: k for k, v in SOURCE_LABELS.items()}
    return [rev.get(s, s) for s in (sources or [])]


def sources_from_labels(labels) -> list:
    """显示标签列表 → 内部 key 列表 (存回 config 用)。"""
    return [SOURCE_LABELS.get(l, l) for l in (labels or [])]


def preset_index(preset_key: str) -> int:
    """预设 key → 下拉选项索引 (用于 st.selectbox index=)。"""
    for i, (label, key) in enumerate(PRESET_LABELS.items()):
        if key == preset_key:
            return i
    return len(PRESET_LABELS) - 1  # 自訂


def style_index(style: str) -> int:
    """style key → 下拉选项索引。"""
    for i, (label, key) in enumerate(STYLE_LABELS.items()):
        if key == style:
            return i
    return 1  # 标准

_DEFAULT_CONFIG: Dict = {
    "persona": {
        "preset": "custom",
        "skepticism": 5,
        "literality": 5,
        "empathy": 5,
        "style": "standard",
        "hindsight_bank": "test-bank",
    },
    "search": {
        "depth": "deep",
        "expand": True,
        "rerank": True,
        "graph_hops": 1,
        "graph_parallel": False,
        "top_k": 5,
        "label_filter": "",
    },
    "knowledge": {
        "sources": ["messages", "contacts"],
        "citations": False,
    },
}


def default_agent_config() -> Dict:
    """返回默认配置的独立副本 (不共享可变状态)。"""
    return deepcopy(_DEFAULT_CONFIG)


def apply_preset(config: Dict, preset_label: str) -> Dict:
    """将 CARA 预设写入 config["persona"] (0-10 值); '自訂' 不改值。"""
    key = PRESET_LABELS.get(preset_label, "custom")
    if key != "custom" and key in CARA_PRESETS:
        config["persona"].update(CARA_PRESETS[key])
    config["persona"]["preset"] = key
    return config


def persona_to_profile_dict(persona: Dict) -> Dict:
    """把 0-10 persona 值换算为 0-1 的 DispositionProfile 字典。"""
    return {
        "skepticism": float(persona.get("skepticism", 5)) / 10.0,
        "literality": float(persona.get("literality", 5)) / 10.0,
        "empathy": float(persona.get("empathy", 5)) / 10.0,
        "style": persona.get("style", "standard"),
    }
