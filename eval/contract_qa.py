"""CorpChat eval — contract-domain QA set (oa-rag readiness artifact, ticket 05).

Rule-generated from the corpus's contract-like conversations (labels:
contract_renewal / payment_reminder / invoice_issue / quotation_request /
order_confirmation / warranty_claim / software_license / equipment_quote).
Questions are framed around contract concepts:

  - contract_party      "who sent the message about <topic>?"  (sender attribution)
  - contract_company    message -> contact company/email (cross-table, deterministic)
  - contract_amount     the amount mentioned in the message
  - contract_date       the send date of the message
  - contract_clause     quote what the message says (evidence quote)
  - contract_negation   a rare contract term that is absent -> honest not-found

Every expected answer is derived from a real message's content/metadata, so the
set is evidence-grounded. Determinism: `random.Random(seed)` — same corpus in,
same QA set out (mirrors eval/qa_generator.py). The set reuses the run_baseline
harness via `--qa-file`.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional

from .qa_generator import _clean_content, _distinctive_token, _meta, _pick

_CONTRACT_LABELS = (
    "contract_renewal", "payment_reminder", "invoice_issue", "quotation_request",
    "order_confirmation", "warranty_claim", "software_license", "equipment_quote",
)

_RARE_TERMS = ["保密协议", "违约金", "仲裁条款", "管辖法院", "第三方托管", "独家授权", "竞业禁止"]

# 金额 pattern: "¥ 320 , 000" / "$1,500" / "8%" / "5000元"
_AMOUNT_RE = re.compile(r"[¥￥$]\s*[\d][\d\s.,]*|\d+(?:\.\d+)?\s*(?:%|％|元|萬|万|美元|USD)", re.I)


def _normalize_amount(raw: str) -> str:
    """把 '¥ 320 , 000' 归一化成 '¥320,000' (去空白, 保留符号/千分位)。"""
    return re.sub(r"\s+", "", raw).strip()


def _extract_amount(content: str) -> Optional[str]:
    m = _AMOUNT_RE.search(content)
    if not m:
        return None
    norm = _normalize_amount(m.group(0))
    if len(norm) < 2:
        return None
    return norm


def _gen_party(norm, by_sender, rng, used):
    """'关于 <topic> 的消息是谁发送的?' — 发送者归属 (message -> sender)。"""
    senders = [s for s, ms in by_sender.items() if len(ms) >= 2]
    sender = _pick(rng, senders)
    if not sender:
        return None
    msgs = [m for m in by_sender[sender] if m["msgid"] not in used]
    if not msgs:
        return None
    target = _pick(rng, msgs)
    topic = _distinctive_token(_clean_content(target))
    if not topic:
        return None
    return {
        "type": "contract_party",
        "question": f"关于 {topic} 的消息是谁发送的？",
        "expected": sender,
        "evidence": [target["msgid"]],
        "context": None,
    }


def _gen_company(norm, by_sender, contact_map, rng, used):
    """'发过关于 <topic> 的消息的 <sender> 的公司是哪一家?' — 跨表 party detail。"""
    senders = [s for s in by_sender if s in contact_map and contact_map[s].get("company")]
    sender = _pick(rng, senders)
    if not sender:
        return None
    msgs = [m for m in by_sender[sender] if m["msgid"] not in used]
    if not msgs:
        return None
    target = _pick(rng, msgs)
    topic = _distinctive_token(_clean_content(target))
    if not topic:
        return None
    c = contact_map[sender]
    expected = f"{sender} 的公司是 {c['company']}"
    if c.get("email"):
        expected += f", 邮箱 {c['email']}"
    return {
        "type": "contract_company",
        "question": f"发过关于 {topic} 的消息的 {sender}，他的公司是哪一家？",
        "expected": expected,
        "evidence": [target["msgid"]],
        "context": None,
    }


def _gen_amount(norm, by_sender, rng, used):
    """'关于 <topic> 的消息里提到的金额是多少?' — 金额提取。"""
    senders = [s for s, ms in by_sender.items() if len(ms) >= 2]
    rng.shuffle(senders)
    for sender in senders:
        msgs = [m for m in by_sender[sender] if m["msgid"] not in used]
        rng.shuffle(msgs)
        for target in msgs:
            content = _clean_content(target)
            amount = _extract_amount(content)
            topic = _distinctive_token(content)
            if amount and topic:
                used.add(target["msgid"])
                return {
                    "type": "contract_amount",
                    "question": f"关于 {topic} 的消息里提到的金额是多少？",
                    "expected": amount,
                    "evidence": [target["msgid"]],
                    "context": None,
                }
    return None


def _gen_date(norm, by_sender, rng, used):
    """'关于 <topic> 的消息的发送日期是哪天?' — 发送日期。"""
    senders = [s for s, ms in by_sender.items() if len(ms) >= 2]
    sender = _pick(rng, senders)
    if not sender:
        return None
    msgs = [m for m in by_sender[sender] if m["msgid"] not in used]
    if not msgs:
        return None
    target = _pick(rng, msgs)
    content = _clean_content(target)
    topic = _distinctive_token(content)
    if not topic or not target.get("send_time"):
        return None
    return {
        "type": "contract_date",
        "question": f"关于 {topic} 的消息的发送日期是哪天？",
        "expected": str(target["send_time"])[:10],
        "evidence": [target["msgid"]],
        "context": None,
    }


def _gen_clause(norm, by_sender, rng, used):
    """'关于 <topic> 的消息里具体说了什么?' — 证据引用 (message_content 同族)。"""
    senders = [s for s, ms in by_sender.items() if len(ms) >= 2]
    sender = _pick(rng, senders)
    if not sender:
        return None
    msgs = [m for m in by_sender[sender] if m["msgid"] not in used]
    if not msgs:
        return None
    target = _pick(rng, msgs)
    content = _clean_content(target)
    if len(content) < 8:
        return None
    topic = _distinctive_token(content)
    if not topic:
        return None
    return {
        "type": "contract_clause",
        "question": f"关于 {topic} 的消息里具体说了什么？",
        "expected": content[:160],
        "evidence": [target["msgid"]],
        "context": None,
    }


def _gen_negation(norm, rng):
    """'合約中是否有提到 <rare term>?' — 语料中确实不存在的合同术语。"""
    term = _pick(rng, _RARE_TERMS)
    if not term:
        return None
    if any(term in _clean_content(m) for m in norm):
        return None
    return {
        "type": "contract_negation",
        "question": f"合約中是否有提到 {term}？",
        "expected": "没有找到相关证据",
        "evidence": [],
        "context": None,
    }


def generate_contract_qa(messages: List[Dict], contacts: Optional[List[Dict]] = None,
                         seed: int = 7, n: int = 40) -> List[Dict]:
    """Generate `n` contract-domain QA pairs from the corpus (deterministic).

    Args:
        messages: message dicts (metadata-shaped, like the txtai index returns).
        contacts: optional contact dicts (name -> company/email) for party-detail.
        seed: deterministic RNG seed.
        n: target number of QA pairs.

    Returns: list of {"id", "type", "question", "expected", "evidence", "context"}.
    """
    rng = random.Random(seed)
    norm = [_meta(m) for m in messages]
    norm = [m for m in norm if m["text"] and m["msgid"] and m["label"] in _CONTRACT_LABELS]

    contact_map: Dict[str, Dict] = {}
    for c in (contacts or []):
        name = str(c.get("full_name") or c.get("name") or "")
        if name:
            contact_map[name] = {
                "company": str(c.get("company") or ""),
                "email": str(c.get("email") or ""),
            }

    by_sender: Dict[str, List[Dict]] = {}
    for m in norm:
        if m["sender"]:
            by_sender.setdefault(m["sender"], []).append(m)

    qa: List[Dict] = []
    used: set = set()
    generators = [_gen_party, _gen_company, _gen_amount, _gen_date,
                  _gen_clause, _gen_negation]
    target_per_type = max(1, n // len(generators)) if n >= len(generators) else n
    counts: Dict[str, int] = {g.__name__: 0 for g in generators}

    while len(qa) < n:
        made = False
        rng.shuffle(generators)
        for gen in generators:
            if counts[gen.__name__] >= target_per_type:
                continue
            if gen is _gen_negation:
                item = _gen_negation(norm, rng)
            elif gen is _gen_company:
                item = _gen_company(norm, by_sender, contact_map, rng, used)
            elif gen is _gen_amount:
                item = _gen_amount(norm, by_sender, rng, used)
            elif gen is _gen_date:
                item = _gen_date(norm, by_sender, rng, used)
            elif gen is _gen_clause:
                item = _gen_clause(norm, by_sender, rng, used)
            else:
                item = _gen_party(norm, by_sender, rng, used)
            if item is not None:
                item["id"] = f"contract_{len(qa) + 1:03d}"
                counts[gen.__name__] += 1
                # evidence 的 label (供测试断言: 只用 contract-like 标签)
                ev_label = ""
                for _m in norm:
                    if _m["msgid"] == (item["evidence"][0] if item["evidence"] else None):
                        ev_label = _m["label"]
                        break
                item["label"] = ev_label
                qa.append(item)
                used.update(item["evidence"])
                made = True
                break
        if not made:
            break

    return qa[:n]

