# CorpChat-RAG — Code Structure, Agentic-Search Review & OA-RAG Integration Plan

> Generated 2026-08-10. Scope: analyze the newly-pulled `~/corpchat-rag`, review its
> agentic RAG search design, and plan how to port the **same agentic search function**
> into `~/oa-rag`.

---

## 1. Executive Summary

`corpchat-rag` is a mature, **modular** agentic RAG search framework. Its search engine
lives in a dedicated package `apps/corpchat/search/` and is layered as:

```
Retrieval base (txtai hybrid BM25+vector, jieba-segmented, sentence-chunked)
  -> QueryExpander (LLM semantic rephrase + keyword expansion)
  -> Weighted RRF fusion (k=50; weights 0.5 / 1.3 / 1.0)
  -> Graph expansion (structural, append-only, query-consistency gate)
  -> Reranker (BAAI/bge-reranker-base cross-encoder, top-20)
Agentic layer:
  -> AgenticDecider   - picks mode/expand/graph_expand/use_rerank (rules -> LLM)
  -> SearchRouter     - LLM gate: search-vs-chat, rewrites the retrieval query
  -> Agent            - intent classification + routing + multi-turn memory
  -> CrossTableAgent  - manual-ReAct tool router across messages & contacts indices
```

**Key finding about `oa-rag`:** it is an **older fork of corpchat-rag** (shared git
history up to commit `66798dd`). Its entire search stack is a **single 1282-line
monolith** `apps/search.py` that contains an *older copy* of `IndexBuilder`,
`QueryExpander`, `Reranker`, `Searcher`, and `AgenticDecider`. It is **missing** the
modular `search/` package, `LiteLLMClient` (with DeepSeek fallback), `SearchRouter`,
`CrossTableAgent`, and the top-level `agent.py` intent layer. Its `tests/test_agent.py`
is **stale** (it still imports `apps.corpchat.*`, which does not exist in oa-rag).

The integration therefore = **port the modular agentic package + agent layer into
oa-rag, adapted to the contract domain (MySQL, contract relationships, no contacts
table).**

---

## 2. CorpChat-RAG Code Structure

```
corpchat-rag/
├── AGENTS.md / CLAUDE.md / CONTEXT.md   # agent conventions + domain glossary
├── README.md
├── requirements.txt                     # txtai 9.12, sentence-transformers, langgraph,
│                                        #   langchain, chonkie, jieba, streamlit, psycopg2
├── core/
│   ├── config.py                        # DB_CONFIG (Postgres), env loading
│   ├── db.py                            # Postgres connection helpers
│   └── embedding.py                     # embedding utilities
├── apps/corpchat/
│   ├── app.py                           # Streamlit UI (731 lines) — contacts, messages,
│   │                                    #   chat viewer, semantic search, process window
│   ├── agent.py                         # ★ Agent + IntentClassifier (600 lines)
│   ├── process_window.py                # UI: agent "thinking" timeline renderer
│   ├── search.py                        # CLI entry (thin wrapper over search/ package)
│   ├── build_index.py                   # index build script
│   ├── gen_fake_msg.py                  # synthetic test-data generator
│   ├── run_agent.py / test_search.py / visualize_graph.py
│   └── search/                          # ★★ THE SEARCH ENGINE PACKAGE (the port target)
│       ├── __init__.py                  # public API re-exports + load_index/load_contacts_index
│       ├── config.py                    # all constants, env, jieba init, embed model, LLM cfg
│       ├── litellm_client.py            # ★ LiteLLMClient — OpenAI-compatible + Ollama +
│       │                                #   DeepSeek fallback, availability check
│       ├── index_builder.py             # IndexBuilder — fetch→chunk(chonkie)→enrich→graph
│       ├── searcher.py                  # ★ Searcher — multi-mode search + weighted RRF +
│       │                                #   graph expansion + rerank orchestration
│       ├── query_expander.py            # QueryExpander — LLM rephrase + keyword expand (cached)
│       ├── reranker.py                  # Reranker — lazy CrossEncoder, top-N re-sort
│       ├── agentic.py                   # ★ AgenticDecider — rule-first + LLM-fallback params
│       ├── router.py                    # ★ SearchRouter — LLM search/chat gate + query rewrite
│       ├── tools.py                     # ★ LangChain tools: search_messages, search_contacts
│       │                                #   (+ last-call metadata capture for process window)
│       ├── cross_table_agent.py         # ★ CrossTableAgent — manual-ReAct multi-source router
│       └── utils.py                     # text cleaning, jieba _segment, structural rels
├── docs/
│   ├── adr/0001-structural-conversation-graph.md
│   └── agents/ (domain.md, issue-tracker.md, triage-labels.md, agentic-layer-report.md)
├── lib/                                 # front-end JS/CSS (vis-network, tom-select)

---

## 3. Agentic Search — Component Review

### 3.1 Retrieval base — `Searcher` (`searcher.py`, 386 lines)
- `search(query, mode, limit, expand, graph_expand, label_filter, date_from/to, use_rerank)`.
- **Two paths**: (A) direct txtai search when `expand=False`; (B) multi-query expansion +
  weighted RRF when `expand=True`. RRF `_weighted_rrf_fusion` uses `k=50` and per-query
  weights (original 0.5 / semantic 1.3 / keyword 1.0); ties broken by best source rank.
- **jieba segmentation** applied identically to index & query so BM25 matches Chinese phrases.
- Metadata fetched per-id directly from txtai's SQLite `sections.tags` (`_fetch_one_doc`) —
  *fixed an earlier bug* that ran an erroneous BM25 `id:` query.
- Filters (label, date range) applied post-hoc on structured metadata only.

**Review:** clean separation of matching vs. metadata; the two-path design keeps a fast
"base retrieval" gate. The RRF fusion is deterministic and well-tested (regression gate).

### 3.2 Graph expansion — `Searcher._graph_expand`
- **Structural-only, append-only.** Walks 4 traversal-eligible edges
  (`same_conversation, sender_receiver, same_sender, same_company`); `same_label` is
  recorded but never traversed (a label never becomes a match signal).
- **Query-consistency gate:** runs the query once to build `id→score`; a neighbor is only
  surfaced if `parent_score × hop_discount(0.8) × neighbor_query_relevance > 0`. This avoids
  an N+1 search and guarantees graph hits stay query-relevant.
- Base order is preserved; graph hits are appended below, sorted by score, truncated to
  `limit*2`.

**Review:** a disciplined, well-reasoned design (documented in ADR-0001). Edges are
deterministic/structural, never vector-inferred — this is the project's core insight.

### 3.3 `QueryExpander` (`query_expander.py`)
- LLM produces 1 semantic rephrase + up to 3 keyword-only queries; each carries a weight.
- Result cached by `query[:100]`. Each LLM call is wrapped in try/except so expansion
  degrades to "original query only" when the LLM is down.

### 3.4 `Reranker` (`reranker.py`)
- Lazy-loads `BAAI/bge-reranker-base` only on first use; `enabled=False` if
  `sentence-transformers` is missing. Re-sorts only the top-20, keeps the original
  (RRF/hybrid) score for display, stores a separate `rerank_score` for sorting.

### 3.5 `LiteLLMClient` (`litellm_client.py`) — *new vs. oa-rag*
- Single choke-point for all LLM HTTP calls. **3-tier graceful degradation**:
  OpenAI-compatible `/chat/completions` → Ollama native `/api/chat` → DeepSeek fallback.
- `is_available()` probe. Every caller gets an empty string on failure (never raises) —
  this is what makes the whole agentic layer **degrade gracefully**.

### 3.6 `AgenticDecider` (`agentic.py`)
- Rule-first: question keywords → `keyword` mode + no expand; similarity keywords →
  `semantic` + expand; long/comparative queries → `graph_expand=1` + rerank; very short →
  no rerank. Then an LLM call (cached, 10s timeout) may override only the `mode`.
- Never lets an LLM failure break the decision.

### 3.7 `SearchRouter` (`router.py`) — *new vs. oa-rag*
- LLM gate returning `{"search": bool, "query": "..."}`. Decides search-vs-chat and
  **rewrites** the message into a concise retrieval query. Robust best-effort JSON parsing;
  parse failure → safe default `search=true`.

### 3.8 `Agent` + `IntentClassifier` (`agent.py`) — *new vs. oa-rag*
- 5 intents: `greeting, system_info, search, clarify, fallback`. Classification is
  **rule-first (<1ms) → LLM fallback (2s timeout) → default "search"** (safe degradation).
- Routing: greeting→(LLM or static) greeting; system_info→static self-description;
  clarify→ask to rephrase; **fallback→search**; search→`Searcher.search()` then LLM answer
  (or formatted results when LLM down). On hybrid failure it retries in plain `keyword` mode.
- Multi-turn memory: last-N turns (in-memory + optional DB persist).

### 3.9 `CrossTableAgent` + `tools.py` (`cross_table_agent.py`) — *new vs. oa-rag*
- LangChain `@tool`s `search_messages` / `search_contacts` that reuse the full
  expansion+rerank pipeline and capture per-call metadata for the **process window**.
- `_LiteLLMWrapper(BaseChatModel)` + a **manual ReAct loop** (chosen over LangGraph's
  tool-calling protocol for reliability with models that don't emit structured tool calls):
  quick greeting/system short-circuit → keyword-based tool routing (`_decide_tool_calls`) →
  execute tools (with empty-result retry using the raw query) → LLM synthesis. Language
  detection (en / zh-TW / zh-CN) drives the answer language.

### 3.10 Overall assessment
**Strengths:** deterministic-first with LLM as an *enhancement* never a dependency;
graceful degradation at every layer; strong test coverage (regression gate on the base);
clean glossary/ADR discipline; metadata strictly separated from the match surface.
**Watch-outs:** the manual ReAct loop and rule tables are domain-tuned to WeCom
messages/contacts and will need re-tuning for contracts; `AgenticDecider`'s keyword sets and
`tools.py`'s two-index (messages+contacts) model do not map 1:1 onto oa-rag's single
contract index.

├── tests/                               # 11 test files incl. router, cross_table, tools,
│                                        #   expansion, graph, reranker, regression, agent
└── .scratch/                            # feature specs/tickets (issue tracker)
```

`★` = a component that is part of the **agentic search function** to be ported.

---

## 4. CorpChat vs OA-RAG — Gap Analysis

| Capability | corpchat-rag | oa-rag (`apps/search.py` monolith) | Action |
|---|---|---|---|
| IndexBuilder | `search/index_builder.py` | older copy inline (`IndexBuilder`) | refresh + modularize |
| QueryExpander | `search/query_expander.py` (uses LiteLLMClient) | older copy, **raw `requests`** | port LiteLLMClient-backed version |
| Reranker | `search/reranker.py` | near-identical inline | port |
| Searcher (RRF+graph) | `search/searcher.py` | near-identical inline (`Searcher`) | port; **relation names differ** |
| AgenticDecider | `search/agentic.py` (uses LiteLLMClient) | older copy inline, raw `requests` | port |
| **LiteLLMClient** (DeepSeek/Ollama fallback) | ✅ `search/litellm_client.py` | ❌ **missing** (inline `requests.post`) | **port** |
| **SearchRouter** (search/chat gate + rewrite) | ✅ `search/router.py` | ❌ **missing** | **port** |
| **Agent + IntentClassifier** | ✅ `agent.py` | ❌ **missing** (`tests/test_agent.py` stale) | **port (adapt)** |
| **CrossTableAgent + tools** | ✅ messages+contacts | ❌ **missing**; oa-rag has **one** contract index + a separate `risk_search.py` planner | **adapt** (see §5.3) |
| Modular `search/` package | ✅ | ❌ (single file) | **create** |
| DB | Postgres (`psycopg2`), `messages`+`contacts` | MySQL (`pymysql`), `formtable_main_385` contracts | keep oa-rag's `core/db.py` |
| Graph relationships | conversation edges | contract edges (`same_contract, same_counterparty, same_department, same_contract_type`) | keep oa-rag's |

**Bottom line:** the retrieval base in oa-rag is already ~90% aligned. The real gap is the
**agentic layer**: `LiteLLMClient`, `SearchRouter`, `Agent`/`IntentClassifier`, and the
modular packaging. The `CrossTableAgent` does **not** map directly (oa-rag has no contacts
index) and should be adapted rather than copied verbatim.

---

## 5. Integration Plan — Agentic Search for OA-RAG

### 5.0 Guiding principles
- **Preserve oa-rag's domain specifics**: MySQL `core/db.py`, contract chunking/enrichment,
  and the 4 contract relationship types. Only the *agentic/search-engine* layer is ported.
- **Modularize** the monolith into a `search/` package mirroring corpchat-rag (this is the
  biggest structural win and makes future syncs trivial).
- **Deterministic-first, LLM-optional**: every LLM touchpoint must degrade gracefully.
- **Don't break existing behavior**: `apps/search.py` CLI, `apps/app.py`, and
  `apps/risk_search.py` keep working via thin re-export shims.


### 5.1 Step 1 — Create the `apps/search/` package (modularize the monolith)
Split `apps/search.py` into focused modules (contract-adapted):

```
oa-rag/apps/search/
├── __init__.py        # re-exports + load_index()
├── config.py          # constants; keep oa-rag DB + EMBED_MODEL + LITELLM_* env
├── litellm_client.py  # ★ PORT from corpchat (DeepSeek/Ollama fallback) — unchanged
├── utils.py           # _clean_text_from_enriched, _segment, _compute_contract_relationships
├── index_builder.py   # from monolith (contract fetch/chunk/enrich/graph) — mostly as-is
├── query_expander.py  # refactor to use LiteLLMClient (replace raw requests)
├── reranker.py        # from monolith (already aligned)
├── searcher.py        # from monolith; KEEP contract TRAVERSAL_RELATIONS
├── agentic.py         # refactor to use LiteLLMClient
└── router.py          # ★ PORT from corpchat; rewrite SYSTEM_PROMPT for contracts
```
Then make `apps/search.py` a **thin CLI shim** that imports from `apps.search` (keeps the
`python apps/search.py build/search/benchmark` commands working). Add a package-alias guard
so `from apps.search import Searcher, load_index` (used by `app.py`) resolves to the package.

### 5.2 Step 2 — Port the agentic decision layer
1. **`LiteLLMClient`** — copy `litellm_client.py` verbatim; it is domain-agnostic. This
   immediately upgrades `QueryExpander` and `AgenticDecider` from brittle inline `requests`
   to the 3-tier fallback client.
2. **`SearchRouter`** — copy `router.py`; change `_SYSTEM_PROMPT` from "corporate chat" to
   "contract / agreement corpus" and adjust the search=true examples (contracts, clauses,
   counterparties, renewal, breach, liability).
3. **`AgenticDecider`** — port the LiteLLMClient-backed version; retune keyword sets for
   contract language (e.g. add "续签/违约/赔偿/采购/对比合同" cues) while keeping the
   rules→LLM→default structure.

### 5.3 Step 3 — Port the top-level `Agent` (adapt to contracts)
Create `oa-rag/apps/agent.py` from corpchat's `agent.py` with these adaptations:
- Import from `apps.search` (not `apps.corpchat.search`).
- Replace WeCom wording in `_GREETING_RESPONSE` / `_SYSTEM_INFO_RESPONSE` with
  contract-screening capabilities; reuse `risk_search.py`'s vocabulary.
- Keep the 5-intent model and rules→LLM→search-default flow.
- `label_filter` → oa-rag's filters (`contract_type`, `department`, `counterparty_name`,
  date range) — map the agent's `search()` kwargs accordingly.
- **Decide the CrossTableAgent strategy** (oa-rag has a single contract index, no contacts):
  - **Option A (recommended):** skip `CrossTableAgent`/`tools.py` initially; instead expose
    the agent's search as one tool and wire the **existing `risk_search.py` RiskPlanner** as
    a second tool ("contract risk screening"). This gives oa-rag a *two-tool* agentic
    search (general contract search ∪ risk screening) that matches its domain.
  - **Option B:** port `tools.py` + `cross_table_agent.py` and build a second "counterparty"
    index so the manual-ReAct router can hop contracts↔counterparties (heavier lift).
- Replace the stale `tests/test_agent.py` (it currently imports `apps.corpchat.*`) with a
  contract-domain version that builds a small in-memory contract index.


### 5.4 Step 4 — UI & tests
- Surface the agent in `apps/app.py` as a new "Ask (Agent)" view alongside
  Search / Risk Search / Browse / Dashboard: intent badge, search/chat router decision,
  process timeline (port `process_window.py` if the animated timeline is wanted).
- Port the relevant corpchat tests → contract domain: router, agentic decider, expansion,
  graph, reranker, and the agent intent/routing/degradation tests.
- Keep `test_risk_search.py` and the regression gate green.

### 5.5 Step 5 — Validation
1. Rebuild the oa-rag index (graph enabled) via the new package.
2. Run the full pytest suite (existing + ported).
3. Manually verify graceful degradation with `LITELLM_API_KEY` unset (rules-only path).
4. Confirm the CLI (`apps/search.py`) and Streamlit app still work end-to-end.

### 5.6 Suggested implementation order (incremental, each independently shippable)
1. `search/config.py` + `search/utils.py` + `search/litellm_client.py` (foundation).
2. `search/index_builder.py` + `search/reranker.py` + `search/searcher.py` (retrieval base).
3. `search/query_expander.py` + `search/agentic.py` refactored onto LiteLLMClient.
4. `search/router.py` + `search/__init__.py` + slim `apps/search.py` shim.
5. `apps/agent.py` (contract-adapted) + tests; optional risk-tool wiring (Option A).
6. UI "Ask (Agent)" view + process window; full validation.

---

## 6. Risks & Notes
- **Package-vs-file name collision**: `apps/search.py` and `apps/search/` cannot both be
  imported as `apps.search`. Resolve by making `apps/search.py` a shim that explicitly
  loads the package, or rename the CLI to `apps/search_cli.py`. **Decide this first.**
- **Domain re-tuning**: router prompts, AgenticDecider keywords, and the ReAct tool-routing
  tables are WeCom-tuned; budget time to tune them on real contract queries.
- **No contacts table**: the literal cross-table (messages↔contacts) feature does not
  translate; the value to carry over is the *manual-ReAct tool-router pattern*, re-aimed at
  contract-search ∪ risk-screening tools.
- **Keep the regression gate**: port corpchat's `test_search_regression.py` philosophy so
  the retrieval base stays measurable after the refactor.

