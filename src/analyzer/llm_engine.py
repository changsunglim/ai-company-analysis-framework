"""
LLM engine - handles all the OpenAI API calls with rate limiting.
"""

import asyncio
import os
import time
from typing import Any

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.analyzer.prompts import PromptManager
from src.preprocessor.cleaner import ProcessedChunk
from src.utils.async_queue import AsyncTaskQueue, QueueTask
from src.utils.logger import setup_logger

logger = setup_logger("llm_engine")


class LLMAnalyzer:
    """Main LLM interface. Runs analysis modules through OpenAI API."""

    # 모듈별로 어떤 데이터 타입이 관련있는지 매핑
    MODULE_DATA_MAPPING = {
        "financial_analysis": ["financial"],
        "news_sentiment": ["news"],
        "competitive_position": ["industry", "financial"],
        "risk_assessment": ["financial", "news", "industry"],
        "growth_outlook": ["financial", "news", "industry"],
    }

    def __init__(self, config: dict | None = None):
        self.config = config or {}

        # Model: env override wins, else config, else the OpenAI default.
        self.model = os.environ.get("LLM_MODEL") or self.config.get("model", "gpt-4o-mini")
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 4096)

        # rate limit 설정 (env overrides let you dial these down for stricter free tiers)
        rate_config = self.config.get("rate_limit", {})
        self.max_rpm = int(os.environ.get("LLM_MAX_RPM") or rate_config.get("max_requests_per_minute", 20))
        self.max_concurrent = int(os.environ.get("LLM_MAX_CONCURRENT") or rate_config.get("max_concurrent", 3))

        # Provider-agnostic: works with any OpenAI-compatible endpoint.
        # Default is OpenAI. For a free run, set LLM_BASE_URL + LLM_API_KEY + LLM_MODEL
        # to a free provider (Groq, Gemini's OpenAI-compatible endpoint, etc.).
        self.client = AsyncOpenAI(
            base_url=os.environ.get("LLM_BASE_URL") or None,
            api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        )
        self.prompt_manager = PromptManager()
        self.task_queue = AsyncTaskQueue(
            max_concurrent=self.max_concurrent,
            max_per_minute=self.max_rpm,
            retry_attempts=rate_config.get("retry_attempts", 3),
            retry_delay=rate_config.get("retry_delay", 2.0),
            exponential_backoff=rate_config.get("exponential_backoff", True),
        )

        # usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0

    async def analyze(
        self,
        company: str,
        chunks: list[ProcessedChunk],
        modules: list[str] | None = None,
    ) -> dict[str, str]:
        """Run all analysis modules on preprocessed data."""
        if modules is None:
            modules = self.prompt_manager.get_available_modules()
            modules = [m for m in modules if m != "executive_summary"]

        logger.info(
            f"Starting analysis for {company} "
            f"({len(modules)} modules, {len(chunks)} data chunks)"
        )

        chunks_by_type = self._group_chunks(chunks)

        # 각 모듈별로 task 생성
        tasks = []
        for module in modules:
            context = self._build_context(module, chunks_by_type)
            if not context.strip():
                logger.warning(f"No data for module '{module}', skipping")
                continue

            prompt = self.prompt_manager.get_analysis_prompt(
                module=module,
                company=company,
                context=context,
            )

            tasks.append(
                QueueTask(
                    task_id=module,
                    coroutine_fn=self._call_llm,
                    kwargs={
                        "system_prompt": prompt["system"],
                        "user_prompt": prompt["user"],
                        "module_name": prompt["module_name"],
                    },
                )
            )

        # rate limiting 적용해서 실행
        results = await self.task_queue.process_batch(tasks)

        analysis_results = {}
        for task, result in zip(tasks, results):
            if result and not isinstance(result, Exception):
                analysis_results[task.task_id] = result

        # executive summary는 다른 분석 끝나고 생성
        if analysis_results:
            summary = await self._generate_executive_summary(
                company, analysis_results
            )
            if summary:
                analysis_results["executive_summary"] = summary

        logger.info(
            f"Analysis done. API calls: {self.api_calls}, "
            f"tokens: {self.total_prompt_tokens + self.total_completion_tokens:,}, "
            f"cost: ${self.total_cost:.4f}"
        )

        return analysis_results

    async def _call_llm(
        self, system_prompt: str, user_prompt: str, module_name: str
    ) -> str:
        """Single LLM API call with error handling."""
        logger.info(f"Running module: {module_name}")
        start = time.time()

        try:
            response = await self._api_call_with_retry(
                system_prompt, user_prompt
            )

            usage = response.usage
            if usage:
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens
                self._update_cost(usage.prompt_tokens, usage.completion_tokens)

            self.api_calls += 1
            elapsed = time.time() - start

            result = response.choices[0].message.content
            logger.info(
                f"'{module_name}' done ({elapsed:.1f}s, "
                f"{usage.total_tokens if usage else '?'} tokens)"
            )
            return result

        except Exception as e:
            logger.error(f"Module '{module_name}' failed: {e}")
            raise

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying API call (attempt {retry_state.attempt_number})..."
        ),
    )
    async def _api_call_with_retry(self, system_prompt: str, user_prompt: str) -> Any:
        """API call with tenacity retry for rate limits."""
        return await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    async def _generate_executive_summary(
        self, company: str, analysis_results: dict[str, str]
    ) -> str | None:
        """Generate executive summary from all analysis results."""
        combined = "\n\n---\n\n".join(
            f"## {module.replace('_', ' ').title()}\n\n{result}"
            for module, result in analysis_results.items()
        )

        prompt = self.prompt_manager.get_analysis_prompt(
            module="executive_summary",
            company=company,
            context=combined,
        )

        try:
            return await self._call_llm(
                system_prompt=prompt["system"],
                user_prompt=prompt["user"],
                module_name="Executive Summary",
            )
        except Exception as e:
            logger.error(f"Executive summary failed: {e}")
            return None

    def _group_chunks(
        self, chunks: list[ProcessedChunk]
    ) -> dict[str, list[ProcessedChunk]]:
        """Group chunks by their data_type field."""
        grouped: dict[str, list[ProcessedChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.data_type, []).append(chunk)
        return grouped

    def _build_context(
        self,
        module: str,
        chunks_by_type: dict[str, list[ProcessedChunk]],
    ) -> str:
        """Build context string for a module. Only includes relevant data types."""
        relevant_types = self.MODULE_DATA_MAPPING.get(
            module, list(chunks_by_type.keys())
        )

        parts = []
        for dtype in relevant_types:
            for chunk in chunks_by_type.get(dtype, []):
                parts.append(chunk.text)

        return "\n\n".join(parts)

    def _update_cost(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Estimate cost from token usage. Known OpenAI pricing only.

        Free-tier / unknown provider models contribute $0 — we don't invent a price
        for a model we have no rate card for, so a free run honestly reports $0.
        """
        pricing = {
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
        }

        model_price = pricing.get(self.model)
        if model_price is None:
            return  # free-tier or unpriced model → no cost to add
        self.total_cost += (
            prompt_tokens * model_price["input"]
            + completion_tokens * model_price["output"]
        )

    def get_usage_stats(self) -> dict:
        return {
            "model": self.model,
            "api_calls": self.api_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
            "queue_metrics": self.task_queue.get_metrics(),
        }
