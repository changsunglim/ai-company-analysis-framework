"""
LLM analysis engine with rate limiting and error handling.

Core component that interfaces with OpenAI API to perform
multi-dimensional company analysis. Implements robust rate
limiting, retry logic, and token management.
"""

import asyncio
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
    """
    LLM-powered analysis engine with built-in rate limiting.

    Orchestrates multiple analysis modules (financial, sentiment,
    competitive, risk, growth) through the OpenAI API with:
    - Token bucket rate limiting
    - Exponential backoff retry
    - Async batch processing
    - Cost tracking
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

        # API configuration
        self.model = self.config.get("model", "gpt-4o-mini")
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 4096)

        # Rate limiting configuration
        rate_config = self.config.get("rate_limit", {})
        self.max_rpm = rate_config.get("max_requests_per_minute", 20)

        # Initialize components
        self.client = AsyncOpenAI()
        self.prompt_manager = PromptManager()
        self.task_queue = AsyncTaskQueue(
            max_concurrent=3,
            max_per_minute=self.max_rpm,
            retry_attempts=rate_config.get("retry_attempts", 3),
            retry_delay=rate_config.get("retry_delay", 2.0),
            exponential_backoff=rate_config.get(
                "exponential_backoff", True
            ),
        )

        # Usage tracking
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
        """
        Run all analysis modules on preprocessed data.

        Args:
            company: Company name/ticker
            chunks: Preprocessed data chunks
            modules: Specific modules to run (default: all)

        Returns:
            Dict mapping module names to analysis results
        """
        if modules is None:
            modules = self.prompt_manager.get_available_modules()
            # Remove executive_summary — it runs after others
            modules = [m for m in modules if m != "executive_summary"]

        logger.info(
            f"Starting analysis for {company} "
            f"({len(modules)} modules, {len(chunks)} data chunks)"
        )

        # Group chunks by data type for targeted analysis
        chunks_by_type = self._group_chunks(chunks)

        # Build tasks for each analysis module
        tasks = []
        for module in modules:
            context = self._build_context(module, chunks_by_type)
            if not context.strip():
                logger.warning(
                    f"No relevant data for module '{module}', skipping"
                )
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

        # Execute all analysis tasks with rate limiting
        results = await self.task_queue.process_batch(tasks)

        # Map results back to module names
        analysis_results = {}
        for task, result in zip(tasks, results):
            if result and not isinstance(result, Exception):
                analysis_results[task.task_id] = result

        # Generate executive summary from all analysis results
        if analysis_results:
            summary = await self._generate_executive_summary(
                company, analysis_results
            )
            if summary:
                analysis_results["executive_summary"] = summary

        logger.info(
            f"Analysis complete. "
            f"API calls: {self.api_calls}, "
            f"Total tokens: {self.total_prompt_tokens + self.total_completion_tokens:,}, "
            f"Estimated cost: ${self.total_cost:.4f}"
        )

        return analysis_results

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        module_name: str,
    ) -> str:
        """
        Make a single LLM API call with error handling.

        Implements retry logic for transient errors while
        failing fast on permanent errors (e.g., invalid API key).
        """
        logger.info(f"Running analysis module: {module_name}")
        start_time = time.time()

        try:
            response = await self._api_call_with_retry(
                system_prompt, user_prompt
            )

            # Track usage
            usage = response.usage
            if usage:
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens
                self._update_cost(usage.prompt_tokens, usage.completion_tokens)

            self.api_calls += 1
            elapsed = time.time() - start_time

            result = response.choices[0].message.content
            logger.info(
                f"Module '{module_name}' complete "
                f"({elapsed:.1f}s, {usage.total_tokens if usage else '?'} tokens)"
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
    async def _api_call_with_retry(
        self, system_prompt: str, user_prompt: str
    ) -> Any:
        """Make an API call with tenacity retry for rate limits."""
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
        """Generate an executive summary from all analysis results."""
        # Combine all analysis outputs
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
            logger.error(f"Failed to generate executive summary: {e}")
            return None

    def _group_chunks(
        self, chunks: list[ProcessedChunk]
    ) -> dict[str, list[ProcessedChunk]]:
        """Group chunks by data type for targeted module context."""
        grouped: dict[str, list[ProcessedChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.data_type, []).append(chunk)
        return grouped

    def _build_context(
        self,
        module: str,
        chunks_by_type: dict[str, list[ProcessedChunk]],
    ) -> str:
        """
        Build context string for a specific analysis module.

        Maps modules to the most relevant data types to minimize
        token usage while maximizing analytical relevance.
        """
        # Module-to-data-type relevance mapping
        relevance_map = {
            "financial_analysis": ["financial"],
            "news_sentiment": ["news"],
            "competitive_position": ["industry", "financial"],
            "risk_assessment": ["financial", "news", "industry"],
            "growth_outlook": ["financial", "news", "industry"],
        }

        relevant_types = relevance_map.get(
            module, list(chunks_by_type.keys())
        )

        context_parts = []
        for data_type in relevant_types:
            chunks = chunks_by_type.get(data_type, [])
            for chunk in chunks:
                context_parts.append(chunk.text)

        return "\n\n".join(context_parts)

    def _update_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> None:
        """Estimate API cost based on model pricing."""
        # gpt-4o-mini pricing (as of 2024)
        pricing = {
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
        }

        model_price = pricing.get(
            self.model, pricing["gpt-4o-mini"]
        )
        cost = (
            prompt_tokens * model_price["input"]
            + completion_tokens * model_price["output"]
        )
        self.total_cost += cost

    def get_usage_stats(self) -> dict:
        """Return API usage statistics."""
        return {
            "model": self.model,
            "api_calls": self.api_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": (
                self.total_prompt_tokens + self.total_completion_tokens
            ),
            "estimated_cost_usd": round(self.total_cost, 4),
            "queue_metrics": self.task_queue.get_metrics(),
        }
