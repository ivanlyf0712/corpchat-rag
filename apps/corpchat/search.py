#!/usr/bin/env python3
"""
CorpChat Search CLI — Command-line interface
============================================
Thin CLI wrapper over the `apps.corpchat.search` package.

Usage:
  python apps/corpchat/search.py build [--force] [--graph-mode auto|llm|off]
  python apps/corpchat/search.py build-contacts
  python apps/corpchat/search.py search "诈骗" --mode hybrid --expand
  python apps/corpchat/search.py benchmark --runs 20
  python apps/corpchat/search.py synthetic-benchmark
  python apps/corpchat/search.py agent-chat "你的邮箱是什么？"
"""

import os
import sys
import logging
import statistics
import time
from typing import Dict, List, Optional

import click
from tabulate import tabulate

# ── 路径 & 配置 ──────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.search import (  # noqa: E402
    AgenticDecider,
    CONTACTS_INDEX_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_RERANK_TOP_N,
    LITELLM_MODEL,
    QueryExpander,
    Reranker,
    IndexBuilder,
    Searcher,
    load_index,
    logger,
    _clean_text_from_enriched,
)


# ═══════════════════════════════════════════════════════════════════
# 8. 合成测试数据
# ═══════════════════════════════════════════════════════════════════

SYNTHETIC_TEST_QUERIES = [
    {"query": "物流方案报价", "expected_labels": ["product_inquiry"],
     "description": "鴻海陳志明詢問物流系統報價"},
    {"query": "ERP timeout error", "expected_labels": ["tech_support"],
     "description": "長榮張偉強反映 ERP timeout"},
    {"query": "invoice discrepancy", "expected_labels": ["invoice_issue"],
     "description": "勤業廖珮琪核對發票金額差異"},
    {"query": "Microsoft 365 E5 授權價格", "expected_labels": ["software_license"],
     "description": "趨勢謝明宏詢價 M365 E5"},
    {"query": "品質不良率 3%", "expected_labels": ["quality_issue"],
     "description": "鴻準蕭國榮反應零件 3% 不良率"},
    {"query": "聯合促銷活動合作", "expected_labels": ["marketing_campaign"],
     "description": "統一劉德華提聯合促銷"},
    {"query": "Surface Pro 電池續航", "expected_labels": ["warranty_claim"],
     "description": "微軟周怡萱處理 Surface Pro 保固"},
    {"query": "合約續約租金調漲", "expected_labels": ["contract_renewal"],
     "description": "和碩鍾佩珊確認續約條件"},
    {"query": "訂單數量增加300片", "expected_labels": ["order_change"],
     "description": "華碩江柏翰調整訂單數量"},
    {"query": "年度業務檢討會議", "expected_labels": ["annual_review"],
     "description": "聯發吳佳穎預約年度檢討"},
    {"query": "詐騙連結", "expected_labels": ["old_friend_reconnect"],
     "description": "高健銘假冒老同學發送釣魚連結"},
    {"query": "投資方案高回報", "expected_labels": ["investment_opportunity"],
     "description": "羅思婷推銷高回報投資方案"},
]


# ═══════════════════════════════════════════════════════════════════
# 9. CLI (click)
# ═══════════════════════════════════════════════════════════════════


def _format_results(results: List[Dict], show_len: int = 100) -> str:
    if not results:
        return "没有找到结果。\n"
    rows = []
    for i, r in enumerate(results, 1):
        text = r.get("text", "")
        meta = r.get("metadata", {})
        text_preview = _clean_text_from_enriched(text)[:show_len] + "..." if len(text) > show_len else _clean_text_from_enriched(text)
        graph_info = ""
        if meta.get("_graph_relation"):
            graph_info = f"🕸️ {meta['_graph_relation']}"
        if r.get("rerank_score") is not None:
            graph_info += f" [Rerank: {r['rerank_score']:.4f}]"

        rows.append([
            i,
            r["id"][:25],
            f"{r.get('score', 0):.4f}",
            str(meta.get("customer_name", "") or meta.get("external_userid", ""))[:12],
            str(meta.get("label", "-")),
            text_preview,
            graph_info,
        ])

    return tabulate(
        rows,
        headers=["#", "ID", "Score", "From", "Label", "Content", "Info"],
        tablefmt="simple_grid",
        maxcolwidths=[None, 18, None, 10, 12, 55, 25],
    )


TEST_QUERIES = [
    {"query": "诈骗", "expected_ids": [], "description": "scam-related"},
    {"query": "合作方案", "expected_ids": [], "description": "cooperation plan"},
    {"query": "product inquiry", "expected_ids": [], "description": "product_inquiry label"},
    {"query": "出货", "expected_ids": [], "description": "shipping logistics"},
    {"query": "投诉", "expected_ids": [], "description": "complaints"},
]


def _calc_mrr(predictions: List[str], expected: List[str]) -> float:
    for i, pid in enumerate(predictions, 1):
        if pid in expected:
            return 1.0 / i
    return 0.0


@click.group()
@click.option("--debug", is_flag=True)
def cli(debug: bool):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command("build")
@click.option("--force", is_flag=True)
@click.option("--graph-mode", type=click.Choice(["auto", "llm", "off"]), default="auto")
@click.option("--index-path", default=DEFAULT_INDEX_PATH)
@click.option("--chunk-size", default=256, type=int)
def build_cmd(force, graph_mode, index_path, chunk_size):
    try:
        enable_graph = graph_mode != "off"
        builder = IndexBuilder(index_path, chunk_size=chunk_size)
        embeddings = builder.build(force=force, enable_graph=enable_graph, graph_mode=graph_mode)
        click.echo(f"✅ 索引就绪 — {embeddings.count()} 个块 | 图: {'✅' if embeddings.graph else '❌'}")
    except Exception as e:
        logger.exception("构建失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


@cli.command("build-contacts")
@click.option("--contacts-index-path", default=None)
def build_contacts_cmd(contacts_index_path):
    """为 contacts 表构建向量索引 (跨表推理需要)。"""
    try:
        path = contacts_index_path or CONTACTS_INDEX_PATH
        embeddings = IndexBuilder.build_contacts_index(path)
        click.echo(f"✅ Contacts 索引就绪 — {embeddings.count()} 条联系人")
    except Exception as e:
        logger.exception("Contacts 索引构建失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


@cli.command("search")
@click.argument("query")
@click.option("--mode", default="hybrid", type=click.Choice(["keyword", "semantic", "hybrid"]))
@click.option("--limit", default=10, type=int)
@click.option("--expand/--no-expand", default=False)
@click.option("--graph-expand", default=0, type=int)
@click.option("--label", default=None)
@click.option("--date-from", default=None)
@click.option("--date-to", default=None)
@click.option("--rerank", is_flag=True)
@click.option("--agentic/--no-agentic", default=False)
@click.option("--api-base", default=None)
@click.option("--api-key", default=None)
@click.option("--model", default=LITELLM_MODEL)
@click.option("--index-path", default=DEFAULT_INDEX_PATH)
def search_cmd(query, mode, limit, expand, graph_expand, label,
               date_from, date_to, rerank, agentic, api_base, api_key,
               model, index_path):
    try:
        if not os.path.exists(index_path):
            click.echo(f"❌ 索引不存在: {index_path}", err=True)
            sys.exit(1)

        embeddings = load_index(index_path)
        click.echo(f"📊 索引: {embeddings.count()} 个块 | 图: {'✅' if bool(embeddings.graph) else '❌'}")

        if agentic:
            decider = AgenticDecider(api_base=api_base, api_key=api_key, model=model)
            decision = decider.decide(query)
            mode = decision["mode"]
            expand = decision.get("expand", expand)
            graph_expand = decision.get("graph_expand", graph_expand)
            rerank = decision.get("use_rerank", rerank)
            click.echo(f"🤖 Agentic: mode={mode}, expand={expand}, graph={graph_expand}, rerank={rerank}")

        expander = QueryExpander(api_base=api_base, api_key=api_key, model=model) if expand else None
        reranker = Reranker(top_n=DEFAULT_RERANK_TOP_N) if rerank else None
        searcher = Searcher(embeddings, expander=expander, reranker=reranker)

        t0 = time.perf_counter()
        results = searcher.search(
            query=query, mode=mode, limit=limit,
            expand=expand, graph_expand=graph_expand,
            label_filter=label, date_from=date_from, date_to=date_to,
            use_rerank=rerank,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        click.echo(f"🔍 模式: {mode} | 查询: \"{query}\" | expand={'✅' if expand else '❌'} | {len(results)} 条 | {elapsed:.1f}ms\n")
        click.echo(_format_results(results))

    except Exception as e:
        logger.exception("搜索失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


@cli.command("graph-query")
@click.argument("cypher")
@click.option("--limit", default=20, type=int)
@click.option("--index-path", default=DEFAULT_INDEX_PATH)
def graph_query_cmd(cypher, limit, index_path):
    try:
        if not os.path.exists(index_path):
            click.echo(f"❌ 索引不存在: {index_path}", err=True)
            sys.exit(1)
        embeddings = load_index(index_path)
        if not embeddings.graph:
            click.echo("❌ 图未启用", err=True)
            sys.exit(1)
        searcher = Searcher(embeddings)
        results = searcher.graph_query(cypher, limit)
        if results:
            click.echo(tabulate(results, headers="keys", tablefmt="simple_grid"))
        else:
            click.echo("无结果。")
    except Exception as e:
        logger.exception("图查询失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


@cli.command("benchmark")
@click.option("--runs", default=20, type=int)
@click.option("--index-path", default=DEFAULT_INDEX_PATH)
def benchmark_cmd(runs, index_path):
    try:
        if not os.path.exists(index_path):
            click.echo(f"❌ 索引不存在: {index_path}", err=True)
            sys.exit(1)
        embeddings = load_index(index_path)
        searcher = Searcher(embeddings)
        queries = ["诈骗", "合作方案", "project report", "urgent", "投资"]
        click.echo(f"📊 {embeddings.count()} 个块, 每个查询 {runs} 次\n")
        all_latencies: List[float] = []
        rows = []
        for q in queries:
            latencies: List[float] = []
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = searcher.search(q, mode="hybrid", limit=10, expand=False)
                latencies.append((time.perf_counter() - t0) * 1000)
            all_latencies.extend(latencies)
            avg = statistics.mean(latencies)
            p50 = statistics.median(latencies)
            p95 = sorted(latencies)[int(len(latencies) * 0.95)]
            p99 = sorted(latencies)[int(len(latencies) * 0.99)]
            rows.append([q, f"{avg:.1f}", f"{p50:.1f}", f"{p95:.1f}", f"{p99:.1f}"])
        click.echo(tabulate(rows, headers=["Query", "Avg(ms)", "P50(ms)", "P95(ms)", "P99(ms)"], tablefmt="simple_grid"))
        if all_latencies:
            sorted_all = sorted(all_latencies)
            click.echo(f"\n📈 总体: Avg={statistics.mean(all_latencies):.1f}ms | P50={statistics.median(all_latencies):.1f}ms | P95={sorted_all[int(len(sorted_all)*0.95)]:.1f}ms | P99={sorted_all[int(len(sorted_all)*0.99)]:.1f}ms")
    except Exception as e:
        logger.exception("基准测试失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


@cli.command("synthetic-benchmark")
@click.option("--index-path", default=DEFAULT_INDEX_PATH)
def synthetic_benchmark_cmd(index_path):
    try:
        if not os.path.exists(index_path):
            click.echo(f"❌ 索引不存在: {index_path}", err=True)
            sys.exit(1)
        embeddings = load_index(index_path)
        searcher = Searcher(embeddings)
        click.echo("🧪 合成测试查询基准\n")
        rows = []
        hit_count = 0
        for test in SYNTHETIC_TEST_QUERIES:
            results = searcher.search(test["query"], mode="hybrid", limit=10, expand=False)
            found_labels = set()
            for r in results:
                lbl = r.get("metadata", {}).get("label", "")
                if lbl:
                    found_labels.add(lbl)
            matched = any(el in found_labels for el in test["expected_labels"])
            if matched:
                hit_count += 1
            rows.append([
                "✅" if matched else "❌",
                test["query"][:25],
                test["description"][:30],
                ", ".join(test["expected_labels"]),
                ", ".join(sorted(found_labels)[:4]) or "-",
            ])
        click.echo(tabulate(rows, headers=["", "Query", "Description", "Expected Labels", "Found Labels"], tablefmt="simple_grid"))
        total = len(SYNTHETIC_TEST_QUERIES)
        click.echo(f"\n📊 召回率: {hit_count}/{total} = {hit_count/total*100:.1f}%")
    except Exception as e:
        logger.exception("合成基准失败")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()