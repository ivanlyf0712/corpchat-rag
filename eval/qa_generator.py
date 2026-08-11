"""CorpChat eval — adversarial QA generation (deterministic, evidence-grounded).

Synthetic question-answer pairs are generated FROM the corpus's own structured
metadata, so the ground truth ("expected") is known and the evidence message ids
are recorded. The question types are deliberately adversarial — the kinds that
break naive retrieval — not easy lookups:

  - message_content    "what did X say about <topic>?"  (with distractors)
  - temporal_window    "<window> about <label/topic>?"  (time-filtered)
  - multi_hop_entity   message -> contact company/email  (cross-table)
  - cross_session      same sender, earlier conversation (recall-like)
  - coreference        "what did they say about..." (context turn provided)
  - disambiguation     same-company sender, one topic  (entity disambiguation)
  - negation           "<rare term>?"  -> answer must be "no evidence found"

Determinism: `random.Random(seed)` — same corpus in, same QA set out.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional

_RARE_TERMS = ["退款", "refund", "折扣", "discount", "保固", "warranty", "違約金", "违约金"]


# ── Normalization ───────────────────────────────────────────────────
def _meta(msg: Dict) -> Dict:
    """Normalize a message dict into the fields the generator reads."""
    if isinstance(msg.get("metadata"), dict):
        m = msg["metadata"]
        return {
            "msgid": str(msg.get("id") or m.get("msgid") or ""),
            "text": str(msg.get("text") or ""),
            "sender": str(m.get("customer_name") or m.get("external_userid") or ""),
            "servicer": str(m.get("servicer_userid") or ""),
            "company": str(m.get("company") or ""),
            "label": str(m.get("label") or ""),
            "send_time": str(m.get("send_time") or ""),
            "open_kfid": str(m.get("open_kfid") or ""),
            "origin": str(m.get("origin") or ""),
        }
    return {
        "msgid": str(msg.get("msgid") or ""),
        "text": str(msg.get("content") or msg.get("text") or ""),
        "sender": str(msg.get("customer_name") or msg.get("external_userid") or ""),
        "servicer": str(msg.get("servicer_userid") or ""),
        "company": str(msg.get("company") or ""),
        "label": str(msg.get("label") or ""),
        "send_time": str(msg.get("send_time") or ""),
        "open_kfid": str(msg.get("open_kfid") or ""),
        "origin": str(msg.get("origin") or ""),
    }


def _clean_content(msg: Dict) -> str:
    """Strip the title/match-surface prefix from message text (keeps the content)."""
    text = _meta(msg)["text"]
    if "\n---\n" in text:
        return text.split("\n---\n", 1)[1]
    return text


def _month_key(send_time: str) -> Optional[str]:
    """'YYYY-MM' for window grouping; None if unparseable."""
    m = re.search(r"(\d{4})-(\d{2})", send_time or "")
    return f"{m.group(1)}-{m.group(2)}" if m else None


# ── QA generation ───────────────────────────────────────────────────
def generate_qa(messages: List[Dict], contacts: Optional[List[Dict]] = None,
                seed: int = 42, n: int = 200) -> List[Dict]:
    """Generate `n` adversarial QA pairs from the corpus.

    Args:
        messages: message dicts (metadata-shaped or {msgid, content, ...}).
        contacts: optional contact dicts (name -> company/email) for multi-hop.
        seed: deterministic RNG seed.
        n: target number of QA pairs.

    Returns: list of {"id", "type", "question", "expected", "evidence", "context"}.
    """
    rng = random.Random(seed)
    norm = [_meta(m) for m in messages]
    norm = [m for m in norm if m["text"] and m["msgid"]]

    contact_map: Dict[str, Dict] = {}
    for c in (contacts or []):
        name = str(c.get("full_name") or c.get("name") or "")
        if name:
            contact_map[name] = {
                "company": str(c.get("company") or ""),
                "email": str(c.get("email") or ""),
            }

    by_month: Dict[str, List[Dict]] = {}
    by_sender: Dict[str, List[Dict]] = {}
    by_label: Dict[str, List[Dict]] = {}
    for m in norm:
        mk = _month_key(m["send_time"])
        if mk:
            by_month.setdefault(mk, []).append(m)
        if m["sender"]:
            by_sender.setdefault(m["sender"], []).append(m)
        if m["label"]:
            by_label.setdefault(m["label"], []).append(m)

    # same-company groups (for disambiguation)
    by_company: Dict[str, List[str]] = {}
    for name, info in contact_map.items():
        if info["company"]:
            by_company.setdefault(info["company"], []).append(name)

    qa: List[Dict] = []
    used_evidence: set = set()
    generators = [
        _gen_message_content, _gen_temporal_window, _gen_multi_hop,
        _gen_cross_session, _gen_disambiguation, _gen_negation,
    ]

    while len(qa) < n:
        made = False
        rng.shuffle(generators)
        for gen in generators:
            item = gen(norm, by_month, by_sender, by_label, by_company,
                       contact_map, rng, used_evidence)
            if item is not None:
                qa.append(item)
                used_evidence.update(item["evidence"])
                made = True
                break
        if not made:
            break  # corpus exhausted for these generators

    return qa[:n]


# ── Generators ──────────────────────────────────────────────────────
def _pick(rng, lst):
    return lst[rng.randrange(len(lst))] if lst else None


def _gen_message_content(norm, by_month, by_sender, by_label, by_company,
                         contact_map, rng, used_evidence):
    """'What did <sender> say about <topic>?' — content lookup with distractors."""
    senders = [s for s, ms in by_sender.items() if len(ms) >= 2]
    sender = _pick(rng, senders)
    if not sender:
        return None
    msgs = [m for m in by_sender[sender] if m["msgid"] not in used_evidence]
    if not msgs:
        return None
    target = _pick(rng, msgs)
    content = _clean_content(target)
    if len(content) < 8:
        return None
    # topic = a distinctive token from the content
    topic = _distinctive_token(content)
    if not topic:
        return None
    return {
        "type": "message_content",
        "question": f"{sender} 说了什么关于 {topic} 的内容？",
        "expected": content[:160],
        "evidence": [target["msgid"]],
        "context": None,
    }


def _gen_temporal_window(norm, by_month, by_sender, by_label, by_company,
                         contact_map, rng, used_evidence):
    """'<last month> 关于 <label> 有什么消息？' — time-filtered recall."""
    months = [mk for mk, ms in by_month.items() if len(ms) >= 2]
    mk = _pick(rng, months)
    if not mk:
        return None
    cands = [m for m in by_month[mk] if m["msgid"] not in used_evidence]
    if not cands:
        return None
    target = _pick(rng, cands)
    content = _clean_content(target)
    label = target["label"] or "消息"
    return {
        "type": "temporal_window",
        "question": f"{mk} 关于 {label} 有什么消息？",
        "expected": content[:160],
        "evidence": [target["msgid"]],
        "context": None,
    }


def _gen_multi_hop(norm, by_month, by_sender, by_label, by_company,
                   contact_map, rng, used_evidence):
    """'<sender> 说了 <topic>, 他的公司/邮箱是？' — message → contact cross-table."""
    senders = [s for s in by_sender if s in contact_map and contact_map[s].get("company")]
    sender = _pick(rng, senders)
    if not sender:
        return None
    msgs = [m for m in by_sender[sender] if m["msgid"] not in used_evidence]
    if not msgs:
        return None
    target = _pick(rng, msgs)
    topic = _distinctive_token(_clean_content(target))
    if not topic:
        return None
    company = contact_map[sender]["company"]
    email = contact_map[sender].get("email")
    expected = f"{sender} 的公司是 {company}"
    if email:
        expected += f", 邮箱 {email}"
    return {
        "type": "multi_hop_entity",
        "question": f"发过关于 {topic} 的消息的 {sender}，他的公司是？",
        "expected": expected,
        "evidence": [target["msgid"]],
        "context": None,
    }


def _gen_cross_session(norm, by_month, by_sender, by_label, by_company,
                       contact_map, rng, used_evidence):
    """Earlier conversation with same sender — recall-style cross-session."""
    senders = [s for s, ms in by_sender.items() if len(ms) >= 3]
    sender = _pick(rng, senders)
    if not sender:
        return None
    msgs = sorted(
        [m for m in by_sender[sender] if m["msgid"] not in used_evidence],
        key=lambda m: m["send_time"],
    )
    if len(msgs) < 2:
        return None
    earlier = msgs[0]
    later = msgs[-1]
    if earlier["open_kfid"] == later["open_kfid"]:
        return None
    content = _clean_content(earlier)
    topic = _distinctive_token(content)
    if not topic:
        return None
    return {
        "type": "cross_session",
        "question": f"之前 {sender} 说过关于 {topic} 的内容吗？",
        "expected": content[:160],
        "evidence": [earlier["msgid"]],
        "context": None,
    }


def _gen_disambiguation(norm, by_month, by_sender, by_label, by_company,
                        contact_map, rng, used_evidence):
    """Two senders from the same company; ask about one of them by company+topic."""
    companies = [c for c, names in by_company.items() if len(names) >= 2]
    company = _pick(rng, companies)
    if not company:
        return None
    names = by_company[company]
    sender = _pick(rng, names)
    msgs = [m for m in by_sender.get(sender, []) if m["msgid"] not in used_evidence]
    if not msgs:
        return None
    target = _pick(rng, msgs)
    topic = _distinctive_token(_clean_content(target))
    if not topic:
        return None
    return {
        "type": "disambiguation",
        "question": f"{company} 的员工里，谁提到了 {topic}？",
        "expected": f"{sender} ({company})",
        "evidence": [target["msgid"]],
        "context": None,
    }


def _gen_negation(norm, by_month, by_sender, by_label, by_company,
                  contact_map, rng, used_evidence):
    """A rare term that (probably) isn't in the corpus — tests honest 'I don't know'."""
    term = _pick(rng, _RARE_TERMS)
    if not term:
        return None
    # only emit if the term truly is absent
    if any(term in _clean_content(m) for m in norm):
        return None
    return {
        "type": "negation",
        "question": f"有没有人提到过 {term}？",
        "expected": "没有找到相关证据",
        "evidence": [],
        "context": None,
    }


def _distinctive_token(text: str) -> Optional[str]:
    """Pick a salient token from content (skips stopwords/punctuation)."""
    STOP = {"的", "了", "和", "是", "在", "有", "我", "你", "他", "她", "它",
            "这", "那", "也", "就", "都", "而", "及", "与", "或", "请", "等",
            "a", "an", "the", "to", "of", "and", "or", "for", "with"}
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9._-]{2,}", text)
    for t in tokens:
        if t not in STOP:
            return t
    return None

