# AI Company Analysis Research Framework

An automated pipeline that transforms hours of manual company research into structured, AI-powered analysis reports. Built with Python, OpenAI API, and async processing to deliver comprehensive financial, competitive, and strategic analysis.

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

## Features

- **Multi-source data collection** — Financial data (yfinance), news (Google News RSS), and competitor analysis run concurrently
- **Smart preprocessing** — Deduplication, relevance filtering, and token-optimized chunking to minimize API costs
- **5 analysis modules** — Financial health, news sentiment, competitive position, risk assessment, and growth outlook
- **Rate-limited async engine** — Token bucket rate limiting, exponential backoff, and priority queuing for reliable API usage
- **Automated report generation** — Jinja2-templated Markdown reports with executive summaries and full metadata

## Quick Start

```bash
# Clone the repository
git clone https://github.com/isaaclim221b/ai-company-analysis-framework.git
cd ai-company-analysis-framework

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key

# Run analysis
python -m src.main AAPL --company "Apple Inc"
```

## Usage

```bash
# Basic analysis (uses ticker as company name)
python -m src.main AAPL

# Specify company name for better news search
python -m src.main MSFT --company "Microsoft"

# Korean stocks
python -m src.main 005930.KS --company "Samsung Electronics"

# Run specific modules only
python -m src.main TSLA --modules financial_analysis news_sentiment

# Custom config
python -m src.main GOOGL --config my_config.yaml

# Custom output directory
python -m src.main NVDA --output-dir ./reports
```

## Project Structure

```
ai-company-analysis-framework/
├── src/
│   ├── main.py              # CLI entry point
│   ├── pipeline.py          # Pipeline orchestrator
│   ├── collector/           # Data collection modules
│   │   ├── base.py          #   Abstract collector interface
│   │   ├── financial.py     #   Yahoo Finance data
│   │   ├── news.py          #   Google News RSS
│   │   └── industry.py      #   Competitor analysis
│   ├── preprocessor/        # Data preprocessing
│   │   └── cleaner.py       #   Dedup, clean, chunk, tokenize
│   ├── analyzer/            # LLM analysis engine
│   │   ├── llm_engine.py    #   Async OpenAI with rate limiting
│   │   └── prompts.py       #   Prompt engineering templates
│   ├── reporter/            # Report generation
│   │   └── generator.py     #   Markdown/JSON report builder
│   └── utils/
│       ├── async_queue.py   #   Rate-limited async task queue
│       └── logger.py        #   Rich console logging
├── config/
│   └── config.yaml          # Pipeline configuration
├── examples/
│   └── sample_report.md     # Example output report
├── tests/
│   ├── test_preprocessor.py
│   └── test_async_queue.py
├── .env.example
├── requirements.txt
└── README.md
```

## Architecture

### Pipeline Design

The framework follows a **staged pipeline architecture** where each stage is independently configurable and testable:

1. **Collect** — Three collectors run concurrently via `asyncio.gather()`, gathering financial metrics, news articles, and competitor data. Each collector implements the `BaseCollector` interface for consistency.

2. **Preprocess** — Raw data passes through reliability filtering (configurable threshold), content deduplication (normalized fingerprinting), text cleaning, and token-aware chunking using `tiktoken` for accurate GPT token counting.

3. **Analyze** — The LLM engine routes preprocessed chunks to appropriate analysis modules based on a relevance map. Modules run through an `AsyncTaskQueue` with token bucket rate limiting, exponential backoff retry, and concurrent execution control. After individual analyses complete, an executive summary synthesizes all findings.

4. **Report** — Jinja2 templates assemble analysis results into formatted Markdown reports with metadata, timestamps, and source attribution.

### Rate Limiting Strategy

API efficiency was a core design constraint. The `AsyncTaskQueue` implements:
- **Token bucket** rate limiting (configurable requests/minute)
- **Semaphore-based** concurrency control
- **Exponential backoff** with configurable retry attempts
- **Priority queuing** for critical analysis modules
- **Usage tracking** for cost monitoring

### Prompt Engineering

Each analysis module uses carefully engineered prompts (`src/analyzer/prompts.py`) designed through iterative experimentation to:
- Extract maximum analytical depth per API call
- Produce structured, parseable output
- Include specific data citations
- Distinguish facts from interpretation

## Configuration

All pipeline parameters are configurable via `config/config.yaml`:

```yaml
# Rate limiting (adjust based on your API tier)
analyzer:
  rate_limit:
    max_requests_per_minute: 20
    max_tokens_per_minute: 60000
    retry_attempts: 3
    exponential_backoff: true

# Control which collectors and modules run
collector:
  financial:
    enabled: true
  news:
    enabled: true
    max_articles: 15
```

## Sample Output

See [examples/sample_report.md](examples/sample_report.md) for a complete analysis report generated for Apple Inc. (AAPL).

## Running Tests

```bash
python -m pytest tests/ -v
```

## Cost Efficiency

Using `gpt-4o-mini`, a typical full analysis costs approximately **$0.005–$0.02** per company. The preprocessing pipeline minimizes token usage through:
- Deduplication of overlapping data sources
- Relevance-based filtering
- Token-aware chunking (only sends what's needed)
- Module-specific context routing (financial data → financial analysis, not sentiment analysis)

## Tech Stack

- **Python 3.11+** — Async/await, type hints, dataclasses
- **OpenAI API** — GPT-4o-mini for cost-efficient analysis
- **yfinance** — Financial data and market metrics
- **aiohttp** — Async HTTP for news collection
- **tiktoken** — Accurate GPT token counting
- **Rich** — Terminal UI and logging
- **Jinja2** — Report template rendering
- **tenacity** — Retry logic with exponential backoff

## License

MIT License

## Author

**Isaac Lim (임창성)**
- Computer Science @ University of Liverpool
- [GitHub](https://github.com/isaaclim221b)
