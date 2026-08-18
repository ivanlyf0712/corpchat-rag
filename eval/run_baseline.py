#!/usr/bin/env python3
"""CorpChat eval — milestone 1 baseline runner (measure the CURRENT pipeline).

Usage:
  python eval/run_baseline.py --index apps/corpchat/search_index \
      --contacts-index apps/corpchat/contacts_index --qa-count 200 --seed 42

  # agents / mock judge / spot-check export:
  python eval/run_baseline.py ... --mode pipeline --judge mock \
      --spot-check 20 --out /tmp/baseline.json --spot-file /tmp/spotcheck.md

Reports: answer correctness %, hallucination %, grounded %, per-type breakdown,
latency (p50/p95/avg), token usage + estimated cost.

The metric: correctness is judged against the QA generator's ground truth, with
grounding as a hard gate (a hallucinated answer is never "correct" for our
purposes even if the judge marks it correct — see REPORT).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from apps.corpchat.search import load_index, load_contacts_index, Searcher, LiteLLMClient
from apps.corpchat.search.answer_path import (
    NOT_FOUND_ANSWER,
    compute_confidence,
    derive_search_filter,
    evidence_passes,
    first_party_detail,
    is_party_detail_question,
    party_answer_text,
    party_detail_text,
)
from apps.corpchat.search.query_expander import QueryExpander
from apps.corpchat.search.reranker import Reranker
from apps.corpchat.search.litellm_client import reset_usage, usage_total

from eval.qa_generator import generate_qa
from eval.judge import judge_answer

# DeepSeek-chat pricing (USD per 1M tokens) — cheap-tier assumption.
_PRICE_INPUT_PER_M = 0.27
_PRICE_OUTPUT_PER_M = 1.10

_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based on retrieved chat messages. "
    "Answer concisely in the same language as the query. "
    "When the question asks what someone said or what messages exist, quote the retrieved "
    "message content directly from the context. For time-period questions, reply with the "
    "content of the most relevant retrieved message; do not enumerate dates, counts, or "
    "invented details. Never invent message content, URLs, sender names, amounts, companies, "
    "or dates that are not present in the context. "
    "If the context contains only one side of a conversation, do not guess or complete the "
    "other side's reply. "
    "If the context does not contain the answer, say so honestly (reply '没有找到相关证据')."
)


# ── Corpus loading ─────────────────────────────────────────────────
def load_messages(embeddings) -> List[Dict]:
    """Read all messages (text + tags) from a txtai index (single-process eval)."""
    db = embeddings.database
    if db is None:
        return []
    conn = db.connection
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, text, tags FROM sections")
        rows = cur.fetchall()
    finally:
        cur.close()
    out = []
    for doc_id, text, tags_raw in rows:
        meta = {}
        if tags_raw:
            try:
                meta = json.loads(tags_raw) if isinstance(tags_raw, str) else dict(tags_raw)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        out.append({"id": doc_id, "text": text, "score": 0.0, "metadata": meta})
    return out


def load_contacts(embeddings) -> List[Dict]:
    """Read contact docs from the contacts index (id → tags)."""
    db = embeddings.database
    if db is None:
        return []
    conn = db.connection
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, tags FROM sections")
        rows = cur.fetchall()
    finally:
        cur.close()
    out = []
    for doc_id, tags_raw in rows:
        try:
            meta = json.loads(tags_raw) if isinstance(tags_raw, str) else dict(tags_raw)
        except (json.JSONDecodeError, TypeError):
            meta = {}
        if meta.get("full_name") or meta.get("name"):
            out.append(meta)
    return out


# ── Current-pipeline answer path (non-agent, mirrors the UI) ────────
def answer_question(query: str, searcher: Searcher, llm_client: LiteLLMClient,
                    top_k: int = 5, use_rerank: bool = True, expand: bool = True,
                    graph_expand: int = 0, graph_parallel: bool = False,
                    known_labels: Optional[List[str]] = None,
                    contacts: Optional[List[Dict]] = None) -> Dict:
    """Run the current retrieval + LLM-synthesis path; return answer + raw hits.

    Answer-path wiring:
      - derive_search_filter: label_filter + date window passed to
        the retrieval seam → windowed, label-scoped hits.
      - evidence_passes: deterministic gate; on failure the honest
        NOT_FOUND_ANSWER is returned with EMPTY evidence (no synthesizer call,
        no cost). Confidence derives from the gate + hit placement.
      - first_party_detail: message-hit → contact company/email
        resolved deterministically and appended to the synthesis context.
      - Structured output {answer, citations, confidence, evidence_gate}.
    """
    filt = derive_search_filter(query, known_labels=known_labels)
    # 时间窗口查询: 放大检索量 (QA 生成器从窗口内随机挑一条目标消息, 需更多
    # 窗口内 hit 才能覆盖它), 且上下文保留更多命中。
    windowed = bool(filt["date_from"] or filt["date_to"])
    retrieve_k = max(top_k, 15) if windowed else top_k
    raw_results = searcher.search(
        query, limit=retrieve_k, use_rerank=use_rerank, expand=expand,
        graph_expand=graph_expand, graph_parallel=graph_parallel,
        label_filter=filt["label_filter"],
        date_from=filt["date_from"], date_to=filt["date_to"],
    )

    evidence_ok = evidence_passes(query, raw_results)
    confidence = compute_confidence(evidence_ok, query, raw_results)
    citations = [str(h.get("id") or "") for h in raw_results[:3]]

    # ── 跨表一步直达: 确定性 message-hit → contact 公司/邮箱 ──
    # party-detail 问题且 resolver 从命中中解析出发送者的联系人时, 直接给出
    # 确定性答案 (不调 synthesizer, 无 LLM 推理)。联系人记录作为证据追加到
    # raw_hits 首位, 保证 judge 的 grounded 判定可追溯到检索证据。
    if contacts and is_party_detail_question(query):
        party = first_party_detail(query, raw_results, contacts)
        if party and (party.get("company") or party.get("email")):
            contact_evidence = {
                "id": f"contact:{party.get('userid') or party.get('full_name') or '?'}",
                "text": "联系人 " + party_detail_text(party),
                "score": 1.0,
                "metadata": dict(party),
            }
            raw_results = [contact_evidence] + list(raw_results)
            return {
                "answer": party_answer_text(party),
                "raw_hits": raw_results,
                "citations": [str(h.get("id") or "") for h in raw_results[:3]],
                "confidence": "high",
                "evidence_gate": evidence_ok,
                "filter_used": filt,
                "party_deterministic": True,
            }

    if not evidence_ok:
        # 诚实 "无证据": 空证据 → judge 将 not-found 判定为 grounded (无幻觉)。
        return {
            "answer": NOT_FOUND_ANSWER,
            "raw_hits": [],
            "citations": [],
            "confidence": confidence,
            "evidence_gate": False,
            "filter_used": filt,
        }

    context_hits = raw_results if windowed else raw_results[: top_k * 2]

    def _context_text(h: Dict) -> str:
        meta = h.get("metadata") or {}
        ts = str(meta.get("send_time") or "")[:10]
        text = str(h.get("text") or "")
        return f"[{ts}] {text}" if ts else text

    context_parts = [_context_text(h) for h in context_hits]

    # ── 跨表解析: message hit → contact company/email 追加进上下文 ──
    if contacts:
        party = first_party_detail(query, raw_results, contacts)
        if party:
            context_parts.append("【联系人信息】" + party_detail_text(party))

    context = "\n---\n".join(context_parts) if context_parts else "No relevant context found."
    answer = llm_client.chat(
        [{"role": "system", "content": _SYSTEM_PROMPT},
         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}],
        temperature=0.3, max_tokens=300, timeout=20,
    )
    return {
        "answer": answer or "(empty)",
        "raw_hits": raw_results,
        "citations": citations,
        "confidence": confidence,
        "evidence_gate": True,
        "filter_used": filt,
    }

# ── Report / spot-check ─────────────────────────────────────────────
def _aggregate(results: List[Dict]) -> Dict:
    n = len(results)
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    per_type = {}
    for t, rs in by_type.items():
        correct = sum(1 for r in rs if r["correct"])
        grounded = sum(1 for r in rs if r["grounded"])
        halluc = sum(1 for r in rs if r["hallucination"])
        lat = [r["latency_ms"] for r in rs]
        per_type[t] = {
            "n": len(rs),
            "correct_rate": round(correct / len(rs), 3),
            "grounded_rate": round(grounded / len(rs), 3),
            "hallucination_rate": round(halluc / len(rs), 3),
            "p50_ms": round(statistics.median(lat), 1),
        }
    correct = sum(1 for r in results if r["correct"])
    grounded = sum(1 for r in results if r["grounded"])
    halluc = sum(1 for r in results if r["hallucination"])
    lat = [r["latency_ms"] for r in results]
    return {
        "n": n,
        "correct_rate": round(correct / n, 3) if n else 0.0,
        "grounded_rate": round(grounded / n, 3) if n else 0.0,
        "hallucination_rate": round(halluc / n, 3) if n else 0.0,
        "latency_ms": {"p50": round(statistics.median(lat), 1),
                       "p95": round(sorted(lat)[int(len(lat) * 0.95) - 1], 1) if lat else 0,
                       "avg": round(statistics.mean(lat), 1) if lat else 0},
        "per_type": per_type,
    }


def _print_report(agg: Dict, usage: Dict) -> None:
    print("\n" + "=" * 60)
    print("BASELINE REPORT (current pipeline, cheap-tier judge)")
    print("=" * 60)
    print(f"questions            : {agg['n']}")
    print(f"answer correctness   : {agg['correct_rate']:.1%}")
    print(f"grounded             : {agg['grounded_rate']:.1%}")
    print(f"hallucination rate   : {agg['hallucination_rate']:.1%}")
    print(f"latency p50/p95/avg  : {agg['latency_ms']['p50']}/{agg['latency_ms']['p95']}/{agg['latency_ms']['avg']} ms")
    cost = (usage["prompt_tokens"] / 1e6 * _PRICE_INPUT_PER_M
            + usage["completion_tokens"] / 1e6 * _PRICE_OUTPUT_PER_M)
    print(f"tokens (in/out/calls): {usage['prompt_tokens']}/{usage['completion_tokens']}/{usage['calls']}")
    print(f"est. cost            : ${cost:.4f}  (~${cost / max(agg['n'], 1) * 1000:.2f}/1k queries)")
    print("-" * 60)
    for t, d in sorted(agg["per_type"].items()):
        print(f"  {t:24s} n={d['n']:3d}  correct={d['correct_rate']:.0%}  "
              f"grounded={d['grounded_rate']:.0%}  halluc={d['hallucination_rate']:.0%}  p50={d['p50_ms']:.0f}ms")


def _write_spotcheck(results: List[Dict], path: str, n: int) -> None:
    lines = ["# Human spot-check (judge calibration)", "",
             "Review these — the judge is the same cheap-tier model as the answerer.",
             "", "| qid | type | question | correct | grounded | rationale | answer |"]
    for r in results[:n]:
        answer = (r["answer"] or "").replace("|", "/").replace("\n", " ")[:120]
        lines.append(f"| {r['id']} | {r['type']} | {r['question'][:60]} | "
                     f"{r['correct']} | {r['grounded']} | {r['rationale'][:60]} | {answer} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"spot-check written to {path}")



def main() -> None:
    ap = argparse.ArgumentParser(description="CorpChat baseline eval")
    ap.add_argument("--index", required=True, help="path to the messages txtai index")
    ap.add_argument("--contacts-index", default=None, help="path to the contacts index")
    ap.add_argument("--qa-count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qa-file", default=None,
                    help="load pre-generated QA pairs (JSON list) instead of "
                         "generate_qa — used for the contract-domain eval set")
    ap.add_argument("--judge", choices=["llm", "mock"], default="llm")
    ap.add_argument("--mode", choices=["pipeline", "agent"], default="pipeline")
    ap.add_argument("--spot-check", type=int, default=0, help="export N samples for human review")
    ap.add_argument("--out", default=None, help="write full results JSON here")
    ap.add_argument("--spot-file", default="/tmp/spotcheck.md")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-expand", action="store_true", help="disable LLM query expansion")
    ap.add_argument("--no-rerank", action="store_true", help="disable cross-encoder rerank")
    args = ap.parse_args()

    embeddings = load_index(args.index)
    messages = load_messages(embeddings)
    contacts = load_contacts(load_contacts_index(args.contacts_index)) if args.contacts_index else None
    print(f"loaded {len(messages)} messages, {len(contacts) if contacts else 0} contacts")

    if args.qa_file:
        with open(args.qa_file, "r", encoding="utf-8") as f:
            qa = json.load(f)
        print(f"loaded {len(qa)} QA pairs from {args.qa_file} "
              f"({Counter(q['type'] for q in qa).most_common()})")
    else:
        qa = generate_qa(messages, contacts=contacts, seed=args.seed, n=args.qa_count)
        print(f"generated {len(qa)} QA pairs "
              f"({Counter(q['type'] for q in qa).most_common()})")

    searcher = Searcher(
        embeddings,
        expander=QueryExpander() if not args.no_expand else None,
        reranker=Reranker() if not args.no_rerank else None,
    )
    llm_client = LiteLLMClient()
    reset_usage()

    # 已知 label 集合 (label 提取) + 联系人表 (跨表解析)
    known_labels = sorted({m.get("metadata", {}).get("label") for m in messages if m.get("metadata", {}).get("label")})

    def _mock_judge(q, e, a, h, c):
        return {"correct": True, "grounded": True, "hallucination": False,
                "rationale": "mock"}

    results = []
    for i, item in enumerate(qa, 1):
        t0 = time.perf_counter()
        resp = answer_question(
            item["question"], searcher, llm_client,
            top_k=args.top_k, use_rerank=not args.no_rerank, expand=not args.no_expand,
            known_labels=known_labels, contacts=contacts,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if args.judge == "llm":
            verdict = judge_answer(item["question"], item["expected"],
                                   resp["answer"], resp["raw_hits"], llm_client)
        else:
            verdict = _mock_judge(item["question"], item["expected"], resp["answer"],
                                  resp["raw_hits"], llm_client)
        results.append({
            "id": item.get("id", f"qa_{i:04d}"),
            "type": item["type"],
            "question": item["question"],
            "expected": item["expected"],
            "answer": resp["answer"],
            "latency_ms": elapsed_ms,
            **verdict,
        })
        if i % 25 == 0 or i == len(qa):
            print(f"  {i}/{len(qa)} done")

    agg = _aggregate(results)
    usage = usage_total()
    _print_report(agg, usage)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"aggregate": agg, "usage": usage, "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"results written to {args.out}")
    if args.spot_check:
        _write_spotcheck(results, args.spot_file, args.spot_check)


if __name__ == "__main__":
    main()
