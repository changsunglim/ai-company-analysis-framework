# AI Company Analysis Framework

[![CI](https://github.com/changsunglim/ai-company-analysis-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/changsunglim/ai-company-analysis-framework/actions/workflows/ci.yml)

An async LLM pipeline that turns ~120 minutes of manual company research into a
structured, sourced report in ~35 minutes. Runs **free** on Gemini/Groq free-tier
models, or ~$0.005–0.02/run on OpenAI. Used to analyze **70+ companies**.

**Full reports the tool generated:**
- **→ [BlackRock (BLK)](blackrock.html)** — generated free on Gemini, **$0.00**
- **→ [Apple (AAPL)](sample_report.html)**

## How it works

```
COLLECT ───▶ PREPROCESS ───▶ ANALYZE ───▶ REPORT
financial     clean +          5 modules:   Markdown
(yfinance),   deduplicate +    financial,   + JSON
news (RSS),   token-optimized  sentiment,   with sources
competitors   chunking         competitive,
(concurrent)                   risk, growth
```

- **Concurrent collection** — financial data, news, and competitor info gathered in parallel
- **Token-optimized preprocessing** — dedup, relevance filtering, chunking
- **5 analysis modules** — financial health, news sentiment, competitive position, risk, growth
- **Rate-limited async API** — token bucket + exponential backoff
- **Structured output** — Markdown reports + JSON, with source metadata

Code, tests, and setup: **[github.com/changsunglim/ai-company-analysis-framework](https://github.com/changsunglim/ai-company-analysis-framework)**
