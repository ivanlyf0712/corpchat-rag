"""Shared Process-window rendering helpers for the CorpChat UI.

This module keeps the Streamlit-specific rendering logic out of app.py so the
page module can stay focused on orchestration.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List


def build_agent_process_payload(tool_calls: List[Dict[str, Any]], turn: Dict[str, Any]) -> Dict[str, Any]:
    """Build the persisted Process-window payload for an agentic turn."""
    tools = []
    for tc in tool_calls or []:
        meta = tc.get("meta", {}) or {}
        tools.append({
            "name": tc.get("tool", "?"),
            "query": tc.get("tool_input", ""),
            "expanded_queries": meta.get("expanded_queries") or [],
            "hit_count": meta.get("hit_count", 0),
            "previews": meta.get("previews", []),
        })
    return {
        "agentic": True,
        "fallback": bool(turn.get("agent_fallback", False)),
        "tools": tools,
    }


def stage_html(label: str, detail: str = "") -> str:
    """HTML for a fade-in stage label with optional detail (compact)."""
    html = f"<div style='font-size:0.85rem;animation:stageFadeIn 0.3s ease-in both;'>{label}"
    if detail:
        html += f" <span style='color:#6b7280;'>{detail}</span>"
    return html + "</div>"


def fade_out_html(label: str) -> str:
    """HTML for a fade-out stage label (used before the next stage replaces it)."""
    return (
        f"<div style='font-size:0.85rem;animation:stageFadeOut 0.3s ease-out both;'>"
        f"{label}</div>"
    )


def animate_stage(slot, label: str, detail: str = ""):
    """Fade in a stage label into `slot` (an st.empty())."""
    slot.markdown(stage_html(label, detail), unsafe_allow_html=True)


def complete_stage(slot, label: str):
    """Fade out the current stage label after its work completed (0.3s)."""
    slot.markdown(fade_out_html(label), unsafe_allow_html=True)
    time.sleep(0.3)


def render_turn_process_window(turn: Dict[str, Any], st_module, pd_module) -> None:
    """Render the unified Process window for a completed turn."""
    process = turn.get("process") or {}
    agentic = bool(process.get("agentic")) or bool(turn.get("agent_steps"))
    fallback = bool(process.get("fallback", turn.get("agent_fallback", False)))
    steps = turn.get("agent_steps", [])
    total_ms = sum(s.get("duration_ms", 0) for s in steps)
    n_tools = len(process.get("tools", [])) or sum(
        1 for s in steps if s.get("label") in ("search_messages", "search_contacts")
    )

    if agentic:
        badge = "⚠️ fallback" if fallback else "✅"
        tool_names = [t.get("name", "?") for t in process.get("tools", []) if t.get("name")]
        label = f"Process (agentic · {badge} · {n_tools} tools · {total_ms}ms"
        if tool_names:
            label += f" · {', '.join(tool_names)}"
        label += ")"
    else:
        label = "Process"

    with st_module.expander(label, expanded=False):
        # ── Hindsight 参与度 (recall/skip/retain/none) ──
        hs = turn.get("hindsight")
        if hs:
            hs_text = {
                "recall": "🧠 Hindsight: memory recall used · retain async",
                "skip": "🧠 Hindsight: recall skipped (no memory trigger word) · retain async",
                "retain": "🧠 Hindsight: retain only (recall is agent-mode)",
                "none": "🧠 Hindsight: not configured",
            }.get(hs)
            if hs_text:
                st_module.markdown(
                    f"<div style='font-size:0.8rem;color:#6b7280;margin-bottom:4px;'>{hs_text}</div>",
                    unsafe_allow_html=True,
                )
        if agentic:
            tools = process.get("tools", [])
            for t in tools:
                t_name = t.get("name", "?")
                t_query = t.get("query", "")
                expanded_qs = t.get("expanded_queries", [])
                hit_count = t.get("hit_count", 0)
                previews = t.get("previews", [])
                with st_module.expander(
                    f"{'🔍' if t_name == 'search_messages' else '👤'} {t_name} · {hit_count} hits",
                    expanded=False,
                ):
                    st_module.markdown(
                        f"<div style='font-size:0.85rem;color:#9ca3af;'>Query: "
                        f"<code>{t_query}</code></div>",
                        unsafe_allow_html=True,
                    )
                    if expanded_qs:
                        st_module.markdown(
                            "<div style='font-size:0.8rem;color:#6b7280;'>Expanded queries:</div>",
                            unsafe_allow_html=True,
                        )
                        for eq in expanded_qs:
                            st_module.markdown(
                                f"<div style='font-size:0.8rem;padding:1px 8px;margin:1px 0;"
                                f"background:#1f2937;border-radius:4px;border-left:3px solid #3b82f6;'>{eq}</div>",
                                unsafe_allow_html=True,
                            )
                    if previews:
                        for p in previews[:5]:
                            sender = p.get("sender") or p.get("name") or "?"
                            text = p.get("text", "")
                            score = p.get("score", "")
                            score_str = f" · {score}" if score != "" else ""
                            st_module.markdown(
                                f"<div style='font-size:0.8rem;padding:1px 0;'><b>{sender}</b>{score_str} — {text[:120]}</div>",
                                unsafe_allow_html=True,
                            )
        else:
            if turn.get("raw_hits"):
                st_module.dataframe(
                    pd_module.DataFrame(turn["raw_hits"]),
                    column_config={
                        "id": st_module.column_config.TextColumn("Message ID"),
                        "text": st_module.column_config.TextColumn("Content"),
                        "score": st_module.column_config.NumberColumn("Score"),
                        "metadata": st_module.column_config.TextColumn("Metadata"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st_module.caption("No raw hits available for this turn.")


def render_chat_history(history: List[Dict[str, Any]], st_module, pd_module) -> None:
    for turn in history:
        with st_module.chat_message("user"):
            st_module.markdown(turn["query"])
        with st_module.chat_message("assistant"):
            if turn.get("interrupted"):
                st_module.info("Search was interrupted. This turn has no results.")
            elif turn.get("status") == "processing":
                st_module.markdown("_Processing your request…_")
            elif turn.get("answer"):
                st_module.markdown(turn["answer"])
                render_turn_process_window(turn, st_module, pd_module)