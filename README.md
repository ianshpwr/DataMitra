# DataMitra 📊
### AI-Powered Multi-Agent Analytics System

> *From raw data to actionable decisions — automatically.*

DataMitra is not a dashboard tool. It is an autonomous AI system that ingests your data, detects patterns, explains what they mean in plain English, recommends what to do about them, and visualizes the evidence — all without manual analysis.

---

## What it does

Upload a CSV. Within 15 seconds you get:

- **Plain English explanations** of every significant pattern in your data
- **Confidence-scored insights** validated by a critic agent before they reach you
- **Ranked action recommendations** with owner, effort level, and expected impact
- **AI-selected charts** chosen based on what each insight actually needs to show
- **Executive summary** you can paste directly into a report

---

## Architecture

DataMitra is built as a **LangGraph state machine** — a directed graph of specialized agents that each do one job and pass a typed state object to the next.

```
CSV / JSON / Parquet / Live Stream
            ↓
    ┌─────────────────┐
    │  Data Agent     │  Cleans, profiles, detects domain
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  Insight Agent  │  Stats engine + LLM explanation
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  Critic Agent   │  Validates, scores confidence, rejects bad insights
    └────────┬────────┘
             ↓ (retry loop if quality < threshold)
    ┌─────────────────┐
    │  Decision Agent │  Converts insights → ranked action recommendations
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  Chart Agent    │  AI selects chart type per insight, renders from real data
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  FastAPI + UI   │  Streamlit dashboard
    └─────────────────┘
```

---

## Agent breakdown

### Agent 1 — Data Processing Agent
**What it does:** Loads raw files (CSV, JSON, Parquet) or live Redis streams, cleans them, profiles every column, detects the business domain, and outputs a typed `DataContext` object.

**Key decisions:**
- Uses **DuckDB** for in-process OLAP queries (no database server needed)
- Uses **Polars** instead of Pandas — 10–50× faster for large files
- Auto-detects and fixes: mixed date formats, currency strings (`$29.99`), case-inconsistent status values, duplicates, missing totals
- Filters out ID columns, geo coordinates, and timestamps before analysis so downstream agents never waste tokens on meaningless columns

**Output:** Cleaned `pl.DataFrame` + `DataContext` (schema, quality report, domain tag, warnings)

---

### Agent 2 — Insight Agent
**What it does:** Runs statistical analysis on the cleaned DataFrame, then uses an LLM to explain each finding in plain English.

**Two-stage design:**
1. **Stats engine** (pure Python, no LLM) — runs: summary stats, null analysis, distribution profiling, categorical analysis, time-series trend detection, correlation analysis, z-score anomaly detection
2. **LLM explainer** (Groq) — takes each statistical result and writes a business-focused explanation

**Dual-model strategy:**
- `llama-3.1-8b-instant` for INFO-level insights (fast, cheap)
- `llama-3.3-70b-versatile` for WARNING/CRITICAL insights (deep reasoning)

**Output:** `InsightBundle` — up to 8 ranked insights, each with title, explanation, evidence, severity, and actionable flag

---

### Agent 3 — Critic Agent
**What it does:** Validates every insight before it reaches the user. No LLM — pure rule-based scoring.

**Scoring rubric (max 1.0):**
| Check | Points |
|-------|--------|
| All affected columns exist in DataFrame | +0.35 |
| Title is concise, specific, no vague language | +0.25 |
| Explanation has enough words and specifics | +0.20 |
| At least one evidence item attached | +0.20 |
| Per flag penalty | -0.15 each |

**Critical flags** (block PASS regardless of score):
- `column_not_in_data` — insight references a column that doesn't exist
- `evidence_value_implausible` — outlier value fails both z-score > 3 AND IQR bounds check

**Key design choice:** Summary/distribution insights (which report means and percentiles as evidence) are **never z-score checked** — they'd always false-flag their own statistics.

**Output:** Validated `InsightBundle` with confidence scores + `CriticReport`

---

### Agent 4 — Decision Agent
**What it does:** Converts validated insights into ranked, executable action recommendations.

**Two-stage design:**
1. **Rule engine** — maps (insight_type, severity, keyword) → action template (owner, KPI, effort level, impact level, priority boost)
2. **LLM recommender** — fills in the specific what/why/expected_impact/risk_if_ignored using business context

**Priority scoring:**
```
priority = severity_base + rule_boost + (confidence × 0.1)
```
Where `severity_base`: critical=0.7, warning=0.5, info=0.2

**Output:** `DecisionPack` — decisions ranked by priority, quick wins identified (high/medium impact + low effort), executive action summary

---

### Agent 5 — Chart Agent
**What it does:** For each insight, decides what chart type would best visualize the underlying data, then renders it from the real DataFrame — not from summary statistics.

**Batched LLM design:**
- Sends all insights in ONE LLM call instead of one call per insight
- Reduces token usage from ~1200/insight to ~800 total
- Falls back to rule-based chart selection if LLM fails (rate limit, timeout)

**Fallback rules:**
| Insight type | Chart type |
|---|---|
| anomaly | histogram |
| distribution (numeric) | histogram |
| categorical_distribution | bar |
| trend | line |
| correlation | scatter |
| summary | skip |

**Column filtering (before LLM sees anything):**
Automatically excludes: `_id` columns, `lat`/`lng`/`longitude`/`latitude`, `timestamp`/`datetime`, high-cardinality numerics (unique ratio > 80%)

**Output:** List of `ChartPlan` objects → Plotly figures rendered from real DataFrame

---

## Tech stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Orchestration | LangGraph 0.1.19 | Stateful multi-agent graph with conditional routing and retry loops |
| LLM | Groq (llama-3.3-70b + llama-3.1-8b) | Free tier, 10-20× faster than OpenAI, dual-model cost strategy |
| Data processing | Polars + DuckDB | 10-50× faster than Pandas, in-process OLAP, no server needed |
| Validation | Pydantic v2 | Typed contracts between every agent |
| API | FastAPI + Uvicorn | Async, auto-docs at `/docs`, multipart file upload |
| UI | Streamlit | Python-native, dark theme, zero JS |
| Charts | Plotly | Interactive, dark theme, renders from real data |
| Live streaming | Redis (dev) / Kafka (prod) | Queue-based event ingestion |
| Real-time | WebSocket / SSE | Progressive result delivery |

---

## Project structure

```
dataMitra/
├── agents/
│   ├── data_agent/
│   │   ├── agent.py          # Main agent class
│   │   ├── loader.py         # Static (DuckDB) + live (Redis) loaders
│   │   ├── profiler.py       # Schema profiling + semantic tagging
│   │   ├── validator.py      # Quality scoring
│   │   ├── cleaner.py        # Cleaning pipeline
│   │   └── models.py         # DataContext, ColumnProfile, QualityReport
│   ├── insight_agent/
│   │   ├── agent.py          # Orchestrates stats + LLM
│   │   ├── stats_engine.py   # Pure statistical analysis (no LLM)
│   │   ├── llm_explainer.py  # Groq-based explanation generation
│   │   └── models.py         # InsightBundle, Insight, StatEvidence
│   ├── critic_agent/
│   │   ├── agent.py          # Rule-based validation + scoring
│   │   └── models.py         # InsightVerdict, CriticReport
│   ├── decision_agent/
│   │   ├── agent.py          # Decision generation orchestrator
│   │   ├── rules.py          # Domain rule engine
│   │   ├── llm_recommender.py # Groq-based recommendation generation
│   │   └── models.py         # Decision, DecisionPack
│   ├── chart_agent/
│   │   ├── agent.py          # Batched chart planning (LLM + fallback)
│   │   └── models.py         # ChartPlan
│   └── pipeline.py           # LangGraph state machine
├── api/
│   ├── main.py               # FastAPI app + CORS
│   ├── schemas.py            # API response models
│   └── routers/
│       ├── analysis.py       # /api/v1/analyse/upload, /path, /live
│       └── health.py         # /health
├── charts/
│   ├── models.py             # ChartPlan (shared)
│   ├── renderer.py           # Plotly renderer (real data only)
│   └── spec_builder.py       # Deprecated
├── data/
│   ├── static/
│   │   ├── orders.csv        # E-commerce test data (4,000 rows)
│   │   ├── products.csv      # Product catalog (300 rows)
│   │   ├── customers.csv     # Customer profiles (1,500 rows)
│   │   └── generate_data.py  # Test data generator with intentional quality issues
│   └── live/
│       └── stream_simulator.py  # Redis stream simulator (0.8 events/sec)
├── tests/
│   ├── test_data_agent_static.py
│   ├── test_data_agent_live.py
│   ├── test_insight_agent.py
│   ├── test_pipeline.py
│   └── test_api.py
├── app_ui.py                 # Streamlit dashboard
├── requirements.txt
└── .env
```

---

## Setup

### Prerequisites
- Python 3.11+
- Redis (for live stream testing)

### Install

```bash
git clone <repo>
cd dataMitra
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

Create `.env`:
```bash
GROQ_API_KEY=gsk_your_key_here
```

Get a free Groq key at [console.groq.com](https://console.groq.com)

### Generate test data

```bash
cd data/static
python generate_data.py
# Creates orders.csv (4,000 rows), products.csv, customers.csv
# with intentional data quality issues built in
```

---

## Running

```bash
# Terminal 1 — API server
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run app_ui.py

# Terminal 3 — Live stream simulator (optional)
redis-server
python data/live/stream_simulator.py
```

- **UI:** http://localhost:8501
- **API docs:** http://localhost:8000/docs

---

## Running tests

```bash
# Test each agent individually
python tests/test_data_agent_static.py
python tests/test_data_agent_live.py   # requires Redis + simulator
python tests/test_insight_agent.py

# Test full pipeline
python tests/test_pipeline.py

# Test API
uvicorn api.main:app --reload --port 8000  # must be running
python tests/test_api.py
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + agent status |
| POST | `/api/v1/analyse/upload` | Upload file and analyse |
| POST | `/api/v1/analyse/path` | Analyse file already on server |
| POST | `/api/v1/analyse/live` | Pull batch from Redis stream |

Full interactive docs at `http://localhost:8000/docs`

---

## Supported file formats

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Auto-detects separators |
| JSON | `.json`, `.jsonl` | Flat records |
| Parquet | `.parquet` | Fastest for large files |
| Max size | 50 MB | Configurable in `analysis.py` |

---

## Supported domains

DataMitra auto-detects the business domain from column names:

| Domain | Detected columns |
|--------|-----------------|
| E-commerce | order_id, product_id, customer_id, unit_price, total_amount |
| Finance | transaction_id, account_id, debit, credit, balance |
| SaaS | user_id, subscription_id, mrr, churn, plan |
| HR | employee_id, department, salary, hire_date |
| Unknown | Falls back to generic analysis |

---

## Key design decisions

**Why LangGraph instead of raw LangChain?**
LangGraph gives us a real state machine with conditional routing and cycles. The critic→insight retry loop would be impossible with a linear chain.

**Why Groq instead of OpenAI?**
Free tier with 14,400 requests/day. Inference is 10–20× faster due to custom silicon. `llama-3.3-70b` matches GPT-4 quality for analytical reasoning tasks.

**Why Polars + DuckDB instead of Pandas?**
A 4,000-row CSV that takes 3.8 seconds in Pandas takes 0.3 seconds in Polars + DuckDB. At 50MB files the difference is even more dramatic.

**Why rule-based Critic instead of LLM-based?**
Deterministic validation. An LLM validator can be convinced bad insights are good. Rules cannot. The critic's job is to ground claims in data — that's exactly what rules are good at.

**Why separate stats engine from LLM explainer?**
The stats engine runs first and produces hard numbers. The LLM only explains what the numbers already prove. This prevents hallucination — if there's no statistical evidence, there's no insight.

---

## Known limitations

- Confidence scores are currently fixed at 0.95 for insights with no flags — will improve with Memory Agent (planned)
- Domain detection is keyword-based — will miss custom column naming conventions
- Chart agent uses one LLM batch call — if all 8 insights hit rate limits, fallback rules activate automatically
- Live stream uses Redis as a simple queue — Kafka integration is planned for production

---

## Roadmap

| Agent | Status | Description |
|-------|--------|-------------|
| Data Agent | ✅ Done | Load, clean, profile |
| Insight Agent | ✅ Done | Stats + LLM explanation |
| Critic Agent | ✅ Done | Validation + confidence scoring |
| Decision Agent | ✅ Done | Action recommendations |
| Chart Agent | ✅ Done | AI-selected charts from real data |
| Memory Agent | 🔜 Next | Qdrant vector store for past insights/decisions |
| Monitor Agent | 🔜 Planned | Real-time Kafka stream monitoring + alerts |
| NL Query | 🔜 Planned | "Why did sales drop?" → instant answer |
| Multi-domain | 🔜 Planned | Finance, SaaS, Healthcare domain packs |
| BI Export | 🔜 Planned | Power BI / Tableau integration |

---

## Built with

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Groq](https://console.groq.com)
- [Polars](https://pola.rs)
- [DuckDB](https://duckdb.org)
- [FastAPI](https://fastapi.tiangolo.com)
- [Streamlit](https://streamlit.io)
- [Plotly](https://plotly.com)

---

*DataMitra — from data to decision, automatically.*