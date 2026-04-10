"""
Async task queue with token bucket rate limiting.
Built this to avoid getting rate limited by OpenAI lol
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from src.utils.logger import setup_logger

logger = setup_logger("async_queue")


@dataclass
class QueueTask:
    """Single task in the queue."""
    task_id: str
    coroutine_fn: Callable[..., Coroutine]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: int = 0  # lower = higher priority
    created_at: float = field(default_factory=time.time)
    result: Any = None
    error: Exception | None = None


class AsyncTaskQueue:
    """
    Rate-limited async task queue.

    Uses a sliding window to track requests per minute and a semaphore
    for concurrency control. Supports retry with exponential backoff.
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

        # metrics
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_retries = 0

    async def _wait_for_rate_limit(self) -> None:
        """Wait if we're about to exceed the rate limit."""
        async with self._lock:
            now = time.time()
            # 60초 지난 타임스탬프 제거
            self._request_times = [
                t for t in self._request_times if now - t < 60
            ]

            if len(self._request_times) >= self.max_per_minute:
                wait_time = 60 - (now - self._request_times[0]) + 0.1
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)

            self._request_times.append(time.time())

    async def _execute_with_retry(self, task: QueueTask) -> Any:
        """Execute task with retry + backoff."""
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                await self._wait_for_rate_limit()
                async with self._semaphore:
                    result = await task.coroutine_fn(*task.args, **task.kwargs)
                    task.result = result
                    self.completed_tasks += 1
                    return result

            except Exception as e:
                last_error = e
                self.total_retries += 1

                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay
                    if self.exponential_backoff:
                        delay *= (2 ** attempt)

                    logger.warning(
                        f"Task '{task.task_id}' failed (attempt {attempt + 1}/"
                        f"{self.retry_attempts}): {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

        # 전부 실패
        task.error = last_error
        self.failed_tasks += 1
        logger.error(
            f"Task '{task.task_id}' failed after "
            f"{self.retry_attempts} attempts: {last_error}"
        )
        return None

    async def process_batch(self, tasks: list[QueueTask]) -> list[Any]:
        """Process a batch of tasks with rate limiting and concurrency control."""
        self.total_tasks += len(tasks)

        # priority 정렬 (낮을수록 먼저)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)

        logger.info(
            f"Processing {len(tasks)} tasks "
            f"(concurrent: {self.max_concurrent}, "
            f"rate: {self.max_per_minute}/min)"
        )

        results = await asyncio.gather(
            *[self._execute_with_retry(t) for t in sorted_tasks],
            return_exceptions=True,
        )

        logger.info(
            f"Batch done: {self.completed_tasks}/{self.total_tasks} ok, "
            f"{self.failed_tasks} failed, {self.total_retries} retries"
        )

        return results

    def get_metrics(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "completed": self.completed_tasks,
            "failed": self.failed_tasks,
            "total_retries": self.total_retries,
            "success_rate": (
                f"{self.completed_tasks / max(self.total_tasks, 1) * 100:.1f}%"
            ),
        }
