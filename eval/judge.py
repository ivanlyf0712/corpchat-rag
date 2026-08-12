"""CorpChat eval — LLM-as-judge for answer correctness + grounding.

Judge rubric (mirrors the milestone-1 metric):
  - correct: the answer is factually right against the expected ground truth.
  - grounded: every claim in the answer is derivable from the retrieved evidence
    (a "no evidence found" answer is grounded iff evidence is empty).
  - hallucination: any claim NOT derivable from the retrieved evidence.

The judge is the same cheap-tier model as the answerer (DeepSeek-chat) — a
self-referential eval, so `run_baseline.py --spot-check N` exports a sample for
human review to calibrate the judge.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

_JUDGE_SYSTEM = (
    "You grade a customer-service search assistant's answer. "
    "Given the QUESTION, the EXPECTED answer (ground truth from the corpus), "
    "the CANDIDATE answer, and the RETRIEVED EVIDENCE, judge two things:\n"
    "1. correct: does the CANDIDATE answer state the same fact as EXPECTED? "
    "   Phrasing differences are fine; a wrong or missing key fact is incorrect. "
    "   If EXPECTED is 'no evidence found', the candidate is correct only if it "
    "   also says nothing was found.\n"
    "2. grounded: is every claim in the CANDIDATE answer derivable from the "
    "   RETRIEVED EVIDENCE? If evidence is empty, a 'not found' answer is "
    "   grounded and any invented claim is not.\n"
    "Reply with ONLY a JSON object: "
    '{"correct": true/false, "grounded": true/false, "hallucination": true/false, '
    '"rationale": "one short sentence"}'
)


def _evidence_snippets(raw_hits: List[Dict], limit: int = 15) -> str:
    """Format the retrieved evidence for the judge.

    limit=15 aligns the judge's evidence view with the answerer's context
    (the answer path feeds up to `retrieve_k=15` dated hits to the synthesizer
    for windowed queries). Each line includes the sender AND the send date, so
    claims citing a message's date/period are checkable — otherwise the judge
    would flag grounded statements as hallucinated (ticket 05 calibration).
    """
    lines = []
    for h in (raw_hits or [])[:limit]:
        text = str(h.get("text") or "")[:220]
        meta = h.get("metadata") or {}
        sender = meta.get("customer_name") or meta.get("external_userid") or "?"
        ts = str(meta.get("send_time") or "")[:10]
        if not text:
            continue
        prefix = f"[{sender}]"
        if ts:
            prefix += f" ({ts})"
        lines.append(f"- {prefix} {text}")
    return "\n".join(lines) or "(no evidence retrieved)"


def _parse_judgment(text: str) -> Optional[Dict]:
    """Best-effort JSON extraction from the judge's reply."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def judge_answer(question: str, expected: str, answer: str,
                 raw_hits: List[Dict], llm_client) -> Dict:
    """Judge one answer; returns {correct, grounded, hallucination, rationale}."""
    user_msg = (
        f"QUESTION: {question}\n\n"
        f"EXPECTED: {expected}\n\n"
        f"CANDIDATE ANSWER: {answer}\n\n"
        f"RETRIEVED EVIDENCE:\n{_evidence_snippets(raw_hits)}\n\n"
        "JSON:"
    )
    try:
        text = llm_client.chat(
            [{"role": "system", "content": _JUDGE_SYSTEM},
             {"role": "user", "content": user_msg}],
            temperature=0.0, max_tokens=120, timeout=15,
        )
    except Exception as e:
        return {"correct": False, "grounded": False, "hallucination": True,
                "rationale": f"judge failed: {e}"}
    parsed = _parse_judgment(text or "")
    if parsed is None:
        return {"correct": False, "grounded": False, "hallucination": True,
                "rationale": f"unparseable judge output: {(text or '')[:80]}"}
    return {
        "correct": bool(parsed.get("correct", False)),
        "grounded": bool(parsed.get("grounded", False)),
        "hallucination": bool(parsed.get("hallucination", False)),
        "rationale": str(parsed.get("rationale") or "")[:200],
    }
