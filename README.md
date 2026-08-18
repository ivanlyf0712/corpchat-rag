# CorpChat RAG

Semantic search and RAG over WeCom (企业微信) customer-service conversations — hybrid retrieval, graph-aware expansion, and a LangGraph agent with long-term memory.

Built on txtai hybrid search, LLM query expansion, cross-encoder reranking, and LiteLLM.

## Features

- 🔍 **Hybrid retrieval** — BM25 + dense vectors (bge-m3) + LLM query expansion, fused with weighted RRF
- 🕸️ **Graph expansion** — one-hop neighbor expansion over a structural conversation graph (same conversation / sender / receiver / company edges)
- ⚡ **Cross-encoder reranking** — `BAAI/bge-reranker-base` (Chinese / multilingual)
- 🧠 **Hindsight memory integration** — multi-path retrieval (content + temporal + graph) with an on-demand recall gate and a persona layer
- 💬 **Streamlit UI** — contacts, messages, conversation history, and semantic search in one place
- 🤖 **Agentic answers** — LangGraph ReAct agent with tool calling, deterministic evidence gating, and cross-table resolution
- 📊 **Eval harness** — adversarial QA generation with LLM-judged correctness / grounding metrics

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Credentials are read from environment variables only. Create a `.env` (gitignored):

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=invoices
export DB_USER=ocr
export DB_PASSWORD=changeme
export DEEPSEEK_API_KEY=sk-...   # required — docker-compose fails fast without it
```

### 3. Start the stack (Docker)

```bash
make up        # validate .env → create volumes → docker compose up -d --build
make ps        # service status
make logs      # follow logs
make down      # stop (data volumes preserved)
```

This brings up three services: **postgres** (pgvector), **hindsight** (memory layer), and **corpchat** (Streamlit UI on `:8501`).

### 4. Seed data, build the index, run the app

```bash
python apps/corpchat/gen_fake_msg.py          # synthetic test corpus
python apps/corpchat/search.py build --force  # build the search index
streamlit run apps/corpchat/app.py            # or use the compose service
```

## Search CLI

```bash
# Build the index
python apps/corpchat/search.py build --force --graph-mode auto

# Full-pipeline search (hybrid + expansion + rerank)
python apps/corpchat/search.py search "诈骗" --mode hybrid --expand --rerank

# Synthetic benchmark
python apps/corpchat/search.py synthetic-benchmark
```

## Evaluation

The eval harness generates adversarial QA pairs from the corpus (multi-hop, temporal, cross-conversation, disambiguation, negation), runs the retrieval + synthesis pipeline, and scores answers with an LLM judge on correctness and grounding.

```bash
# Smoke run on the current index
python eval/run_baseline.py --index apps/corpchat/search_index \
    --contacts-index apps/corpchat/contacts_index --qa-count 200 --seed 42

# Full baseline on a 10k synthetic corpus
python apps/corpchat/gen_fake_msg.py --count 10000
python apps/corpchat/search.py build --force
python eval/run_baseline.py --index apps/corpchat/search_index \
    --contacts-index apps/corpchat/contacts_index --qa-count 200 \
    --spot-check 20 --out /tmp/baseline.json

# Retrieval cost/latency only (mock judge — no judge LLM calls)
python eval/run_baseline.py ... --judge mock
```

Reports include answer correctness %, grounded %, hallucination %, per-type breakdowns, latency p50/p95/avg, and token usage with estimated cost. `--spot-check N` exports a table for manual review of the judge's calls.

### Deterministic answer-path controls

- **Label filter + time window** (`derive_search_filter`) — questions like "2026-07 关于 product_inquiry" are scoped by month window and label.
- **Evidence gate** (`evidence_passes`) — when key entities/keywords are absent from the retrieved hits, the system answers "没有找到相关证据" instead of synthesizing (hallucination control).
- **Cross-table resolution** (`resolve_party_detail`) — party-detail questions ("the person who sent X — what's their company?") are answered in one deterministic step via contact resolution, with citations and confidence.

## Project Structure

```
corpchat-rag/
├── Makefile                    # up/down/logs/test — env validation + volume bootstrap
├── docker-compose.yml          # postgres + hindsight + corpchat stack
├── apps/corpchat/
│   ├── app.py                  # Streamlit UI
│   ├── search.py               # CLI entry (thin wrapper)
│   ├── search/                 # Search engine package
│   │   ├── searcher.py         # Hybrid search + RRF + graph expansion
│   │   ├── index_builder.py    # Chunking + enrichment
│   │   ├── query_expander.py   # LLM query expansion
│   │   ├── reranker.py         # Cross-encoder reranker
│   │   ├── answer_path.py      # Filters, evidence gate, party-detail resolver
│   │   ├── temporal.py         # Temporal retrieval path
│   │   ├── memory_graph.py     # Entity memory graph
│   │   ├── cross_table_agent.py# LangGraph cross-table agent
│   │   ├── hindsight_client.py # Hindsight memory adapter
│   │   ├── persona.py          # CARA persona layer
│   │   ├── router.py           # Query routing
│   │   ├── tools.py            # Agent tool definitions
│   │   ├── litellm_client.py   # Unified LLM API client
│   │   └── config.py           # Configuration and constants
│   ├── build_index.py          # Index build script
│   └── gen_fake_msg.py         # Synthetic corpus generator
├── core/                       # DB config and persistence (env-only credentials)
├── eval/                       # QA generation, judging, baseline runner
├── tests/                      # pytest suite
├── docs/                       # ADRs and design notes
└── requirements.txt
```

## Testing

```bash
make test        # or: python -m pytest tests/ -q
```

