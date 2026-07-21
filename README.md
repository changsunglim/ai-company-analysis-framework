# AI Company Analysis Framework

[![CI](https://github.com/changsunglim/ai-company-analysis-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/changsunglim/ai-company-analysis-framework/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Automated pipeline that turns manual company research into structured, AI-powered reports. Uses OpenAI API + async processing for financial, competitive, and strategic analysis.

**Results:** cuts company-analysis time from ~120 minutes of manual work to ~35 minutes automated, at $0.005–0.02 per run (gpt-4o-mini). Used to analyze 70+ companies. See a full example in [`examples/sample_report.md`](examples/sample_report.md).

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
   Google News          NLP               OpenAI API          output/*.md
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

# Set your OpenAI API key
cp .env.example .env
# edit .env

python -m src.main AAPL --company "Apple Inc"
```

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

With `gpt-4o-mini`, a full analysis runs about **$0.005–$0.02** per company. The preprocessing pipeline keeps costs down by deduplicating, filtering, and only sending relevant data to each module.

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

## Tech Stack

- Python 3.11+ (async/await, type hints)
- OpenAI API (gpt-4o-mini)
- yfinance, aiohttp, tiktoken, Rich, Jinja2, tenacity

## Author

**Isaac Lim** — CS at University of Liverpool

## License

MIT
