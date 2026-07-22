# AI Company Analysis Framework

[![CI](https://github.com/changsunglim/ai-company-analysis-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/changsunglim/ai-company-analysis-framework/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Automated pipeline that turns manual company research into structured, AI-powered reports. Works with any OpenAI-compatible LLM + async processing for financial, competitive, and strategic analysis.

**Results:** cuts company-analysis time from ~120 minutes of manual work to ~35 minutes automated. Runs **free** on Gemini or Groq free-tier models (~$0 per run), or ~$0.005–0.02 per run on OpenAI `gpt-4o-mini`. Used to analyze 70+ companies. Full examples: **[BlackRock](examples/blackrock.md)** (generated free on Gemini, $0.00) · [Apple](examples/sample_report.md).

## Try it

**[Open the web app](#)** — enter a ticker and your own OpenAI API key, get a report in your browser. No install. Your key is used only for that request and never stored.

*(Deployed via [Streamlit Community Cloud](https://streamlit.io/cloud) — replace the link above with your app's URL after deploying.)*

Or run the UI locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

```
                    ┌─────────────────────────────────────────────────┐
                    │        AI Company Analysis Pipeline             │
                    └─────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │   COLLECT    │───▶│  PREPROCESS  │───▶│   ANALYZE    │───▶│   REPORT     │
  │              │    │              │    │              │    │              │
  │ • Financial  │    │ • Clean      │    │ • Financial  │    │ • Markdown   │
  │ • News       │    │ • Deduplicate│    │ • Sentiment  │    │ • JSON       │
  │ • Industry   │    │ • Chunk      │    │ • Competitive│    │ • Metadata   │
  │              │    │ • Tokenize   │    │ • Risk       │    │ • Sources    │
  └──────────────┘    └──────────────┘    │ • Growth     │    └──────────────┘
        ▲                   ▲             │ • Summary    │          │
        │                   │             └──────────────┘          │
   Yahoo Finance       tiktoken              ▲                     ▼
   Google News          NLP          OpenAI-compatible LLM     output/*.md
                                     (OpenAI · Gemini · Groq)
```

## What it does

- Collects financial data (yfinance), news (Google News RSS), and competitor info concurrently
- Preprocesses with deduplication, relevance filtering, and token-optimized chunking
- Runs 5 analysis modules: financial health, news sentiment, competitive position, risk assessment, growth outlook
- Rate-limited async API calls with token bucket + exponential backoff
- Generates Markdown reports with executive summaries

## Quick Start

```bash
git clone https://github.com/changsunglim/ai-company-analysis-framework.git
cd ai-company-analysis-framework

pip install -r requirements.txt

# Configure a provider (see .env.example for free Gemini/Groq options)
cp .env.example .env
# edit .env

python -m src.main AAPL --company "Apple Inc"
```

## Providers

Works with any OpenAI-compatible endpoint — set these in `.env`:

| Provider | Cost | Config |
|---|---|---|
| OpenAI | ~$0.005–0.02/run | `OPENAI_API_KEY=sk-...` |
| **Gemini** (free tier) | **$0** | `LLM_API_KEY` + `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/` + `LLM_MODEL=gemini-2.0-flash` |
| **Groq** (free tier) | **$0** | `LLM_API_KEY` + `LLM_BASE_URL=https://api.groq.com/openai/v1` + `LLM_MODEL=llama-3.3-70b-versatile` |

Free tiers are rate-limited — dial concurrency down with `LLM_MAX_CONCURRENT=1` and `LLM_MAX_RPM=10` if you hit 429s. The report's cost line shows `$0.00` on free providers. The [BlackRock example](examples/blackrock.md) was generated this way.

## Usage

```bash
# Basic
python -m src.main AAPL

# With company name (helps news search)
python -m src.main MSFT --company "Microsoft"

# Korean stocks
python -m src.main 005930.KS --company "Samsung Electronics"

# Specific modules only
python -m src.main TSLA --modules financial_analysis news_sentiment

# Custom output dir
python -m src.main NVDA --output-dir ./reports
```

## Project Structure

```
ai-company-analysis-framework/
├── src/
│   ├── main.py              # CLI entry point
│   ├── pipeline.py          # Pipeline orchestrator
│   ├── collector/           # Data collection
│   │   ├── base.py          #   Collector interface
│   │   ├── financial.py     #   Yahoo Finance
│   │   ├── news.py          #   Google News RSS
│   │   └── industry.py      #   Competitor data
│   ├── preprocessor/
│   │   └── cleaner.py       #   Dedup, clean, chunk
│   ├── analyzer/
│   │   ├── llm_engine.py    #   Async OpenAI + rate limiting
│   │   └── prompts.py       #   Prompt templates
│   ├── reporter/
│   │   └── generator.py     #   Markdown/JSON reports
│   └── utils/
│       ├── async_queue.py   #   Rate-limited task queue
│       └── logger.py        #   Logging
├── config/config.yaml
├── examples/sample_report.md
├── tests/
├── requirements.txt
└── README.md
```

## How it works

### Pipeline

4-stage pipeline, each stage independently configurable:

1. **Collect** — Three collectors run concurrently (`asyncio.gather`). Financial metrics from yfinance, news from Google News RSS, competitor data from yfinance. All implement `BaseCollector` interface.

2. **Preprocess** — Filter by reliability score → deduplicate with normalized fingerprinting → clean text → chunk with `tiktoken` for accurate token counting. Goal is to minimize tokens sent to the API.

3. **Analyze** — LLM engine routes chunks to modules based on a relevance map (e.g. financial data goes to financial analysis, not sentiment). Runs through `AsyncTaskQueue` with rate limiting and retries. Executive summary is generated last from all other results.

4. **Report** — Jinja2 templates assemble everything into Markdown with metadata and source attribution.

### Rate Limiting

The `AsyncTaskQueue` was the trickiest part to get right. It does:
- Token bucket rate limiting (requests/minute)
- Semaphore for concurrency control
- Exponential backoff on failures
- Priority ordering
- Usage/cost tracking

### Cost

Free (**$0**) on Gemini/Groq free-tier models. With OpenAI `gpt-4o-mini`, a full analysis runs about **$0.005–$0.02** per company. The preprocessing pipeline keeps costs down by deduplicating, filtering, and only sending relevant data to each module.

## Configuration

Edit `config/config.yaml`:

```yaml
analyzer:
  rate_limit:
    max_requests_per_minute: 20
    retry_attempts: 3
    exponential_backoff: true

collector:
  financial:
    enabled: true
  news:
    enabled: true
    max_articles: 15
```

## Tests

```bash
python -m pytest tests/ -v
```

## Deploying the web app

1. Push this repo to your own GitHub (already done if you're reading this on GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, "New app".
3. Point it at this repo, branch `main`, main file `streamlit_app.py`. Deploy.
4. Update the "Try it" link at the top of this README to your new app's URL.

No secrets to configure — each visitor pastes their own OpenAI key into the app's sidebar at runtime; it's never persisted.

## Tech Stack

- Python 3.11+ (async/await, type hints)
- OpenAI API (gpt-4o-mini)
- yfinance, aiohttp, tiktoken, Rich, Jinja2, tenacity

## Author

**Isaac Lim** — CS at University of Liverpool

## License

MIT
