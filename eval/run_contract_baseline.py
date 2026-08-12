"""CorpChat eval — run the contract-domain baseline and write the artifact.

Usage (repo root):
  /Users/ivanlee/miniconda3/envs/ocr/bin/python eval/run_contract_baseline.py \
      --index apps/corpchat/search_index \
      --contacts-index apps/corpchat/contacts_index

Runs `run_baseline.py --qa-file eval/results/contract-qa.json` with the
calibrated judge, then writes a per-type report to eval/results/contract-baseline.md.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))  # eval/
DEFAULT_QA = f"{ROOT}/results/contract-qa.json"
DEFAULT_OUT = f"{ROOT}/results/contract-baseline.json"


def _aggregate(results):
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
    n = len(results)
    return {
        "n": n,
        "correct_rate": round(sum(1 for r in results if r["correct"]) / n, 3) if n else 0.0,
        "grounded_rate": round(sum(1 for r in results if r["grounded"]) / n, 3) if n else 0.0,
        "hallucination_rate": round(sum(1 for r in results if r["hallucination"]) / n, 3) if n else 0.0,
        "latency_ms": {"p50": round(statistics.median([r["latency_ms"] for r in results]), 1) if results else 0},
        "per_type": per_type,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Contract-domain baseline")
    ap.add_argument("--index", required=True)
    ap.add_argument("--contacts-index", required=True)
    ap.add_argument("--qa-file", default=DEFAULT_QA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--spot-check", type=int, default=20)
    args = ap.parse_args()

    # 复用同一 harness (run_baseline) — 校准后的 judge 与 answer path 一致
    cmd = [
        sys.executable, f"{ROOT}/run_baseline.py",
        "--index", args.index, "--contacts-index", args.contacts_index,
        "--qa-file", args.qa_file, "--seed", "7",
        "--spot-check", str(args.spot_check), "--out", args.out,
    ]
    subprocess.run(cmd, check=True)

    with open(args.out, "r", encoding="utf-8") as f:
        data = json.load(f)
    agg = _aggregate(data["results"])
    usage = data.get("usage", {})
    cost = (usage.get("prompt_tokens", 0) / 1e6 * 0.27
            + usage.get("completion_tokens", 0) / 1e6 * 1.10)

    lines = [
        "# Contract-domain baseline (oa-rag readiness artifact)",
        "",
        f"Date: run of `run_contract_baseline.py` · index: {args.index} · "
        f"QA set: {args.qa_file} ({agg['n']} questions) · calibrated judge",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "|---|---|",
        f"| questions | {agg['n']} |",
        f"| **answer correctness** | **{agg['correct_rate']:.1%}** |",
        f"| grounded | {agg['grounded_rate']:.1%} |",
        f"| hallucination rate | {agg['hallucination_rate']:.1%} |",
        f"| latency p50 | {agg['latency_ms']['p50']} ms |",
        f"| est. cost | ${cost:.4f} |",
        "",
        "## By question type",
        "",
        "| type | n | correct | grounded | halluc | p50 |",
        "|---|---|---|---|---|---|",
    ]
    for t, d in sorted(agg["per_type"].items()):
        lines.append(
            f"| {t} | {d['n']} | {d['correct_rate']:.0%} | {d['grounded_rate']:.0%} | "
            f"{d['hallucination_rate']:.0%} | {d['p50_ms']:.0f}ms |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- Contract types: parties (contract_party), company (contract_company, "
        "deterministic resolver), amounts (contract_amount), dates (contract_date), "
        "clauses (contract_clause), negation (contract_negation).",
        "- The judge is the calibrated DeepSeek tier; spot-check samples are exported "
        "to `/tmp/spotcheck.md` for human review.",
    ]
    report_path = args.out.replace(".json", ".md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"contract baseline report written to {report_path}")


if __name__ == "__main__":
    main()
