"""
Async task queue with rate limiting for efficient API utilization.

Designed to minimize API calls while maintaining analysis quality —
a critical constraint when operating in resource-limited environments.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from src.utils.logger import setup_logger

logger = setup_logger("async_queue")


@dataclass
class QueueTask:
    """Represents a single task in the processing queue."""

    task_id: str
    coroutine_fn: Callable[..., Coroutine]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: int = 0  # Lower = higher priority
    created_at: float = field(default_factory=time.time)
    result: Any = None
    error: Exception | None = None


class AsyncTaskQueue:
    """
    Rate-limited async task queue for managing API calls.

    Implements token bucket rate limiting to stay within API quotas
    while maximizing throughput. Supports priority ordering and
    automatic retry with exponential backoff.
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        max_per_minute: int = 20,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        exponential_backoff: bool = True,
    ):
        self.max_concurrent = max_concurrent
        self.max_per_minute = max_per_minute
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._request_times: list[float] = []
        self._lock = asyncio.Lock()

        # Metrics
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_retries = 0

    async def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to comply with rate limits."""
        async with self._lock:
            now = time.time()
            # Remove request timestamps older than 60 seconds
            self._request_times = [
                t for t in self._request_times if now - t < 60
            ]

            if len(self._request_times) >= self.max_per_minute:
                # Calculate wait time until the oldest request expires
                wait_time = 60 - (now - self._request_times[0]) + 0.1
                if wait_time > 0:
                    logger.info(
                        f"Rate limit reached. Waiting {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)

            self._request_times.append(time.time())

    async def _execute_with_retry(self, task: QueueTask) -> Any:
        """Execute a task with retry logic and exponential backoff."""
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                await self._wait_for_rate_limit()
                async with self._semaphore:
                    result = await task.coroutine_fn(
                        *task.args, **task.kwargs
                    )
                    task.result = result
                    self.completed_tasks += 1
                    return result

            except Exception as e:
                last_error = e
                self.total_retries += 1

                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay
                    if self.exponential_backoff:
                        delay *= 2**attempt

                    logger.warning(
                        f"Task '{task.task_id}' failed (attempt {attempt + 1}/"
                        f"{self.retry_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

        task.error = last_error
        self.failed_tasks += 1
        logger.error(
            f"Task '{task.task_id}' failed after {self.retry_attempts} "
            f"attempts: {last_error}"
        )
        return None

    async def process_batch(
        self, tasks: list[QueueTask]
    ) -> list[Any]:
        """
        Process a batch of tasks concurrently with rate limiting.

        Args:
            tasks: List of QueueTask objects to process

        Returns:
            List of results in the same order as input tasks
        """
        self.total_tasks += len(tasks)

        # Sort by priority (lower number = higher priority)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)

        logger.info(
            f"Processing batch of {len(tasks)} tasks "
            f"(max concurrent: {self.max_concurrent}, "
            f"rate limit: {self.max_per_minute}/min)"
        )

        results = await asyncio.gather(
            *[self._execute_with_retry(task) for task in sorted_tasks],
            return_exceptions=True,
        )

        logger.info(
            f"Batch complete: {self.completed_tasks}/{self.total_tasks} "
            f"succeeded, {self.failed_tasks} failed, "
            f"{self.total_retries} retries"
        )

        return results

    def get_metrics(self) -> dict:
        """Return queue performance metrics."""
        return {
            "total_tasks": self.total_tasks,
            "completed": self.completed_tasks,
            "failed": self.failed_tasks,
            "total_retries": self.total_retries,
            "success_rate": (
                f"{self.completed_tasks / max(self.total_tasks, 1) * 100:.1f}%"
            ),
        }
