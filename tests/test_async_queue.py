"""
Unit tests for the async task queue.
"""

import asyncio
import unittest
from src.utils.async_queue import AsyncTaskQueue, QueueTask


class TestAsyncTaskQueue(unittest.TestCase):
    """Tests for AsyncTaskQueue rate limiting and retry logic."""

    def test_basic_task_execution(self):
        """Test that tasks execute and return results."""

        async def sample_task(x: int) -> int:
            return x * 2

        async def run():
            queue = AsyncTaskQueue(
                max_concurrent=3, max_per_minute=60
            )
            tasks = [
                QueueTask(
                    task_id=f"task_{i}",
                    coroutine_fn=sample_task,
                    args=(i,),
                )
                for i in range(5)
            ]
            results = await queue.process_batch(tasks)
            return results, queue

        results, queue = asyncio.run(run())
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0], 0)
        self.assertEqual(results[1], 2)
        self.assertEqual(queue.completed_tasks, 5)
        self.assertEqual(queue.failed_tasks, 0)

    def test_retry_on_failure(self):
        """Test that failed tasks are retried."""
        call_count = 0

        async def flaky_task() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"

        async def run():
            queue = AsyncTaskQueue(
                max_concurrent=1,
                max_per_minute=60,
                retry_attempts=3,
                retry_delay=0.1,
            )
            tasks = [
                QueueTask(task_id="flaky", coroutine_fn=flaky_task)
            ]
            results = await queue.process_batch(tasks)
            return results, queue

        results, queue = asyncio.run(run())
        self.assertEqual(results[0], "success")
        self.assertEqual(queue.completed_tasks, 1)

    def test_metrics_tracking(self):
        """Test that queue metrics are properly tracked."""

        async def ok_task() -> str:
            return "ok"

        async def run():
            queue = AsyncTaskQueue(max_concurrent=5, max_per_minute=60)
            tasks = [
                QueueTask(
                    task_id=f"task_{i}", coroutine_fn=ok_task
                )
                for i in range(10)
            ]
            await queue.process_batch(tasks)
            return queue.get_metrics()

        metrics = asyncio.run(run())
        self.assertEqual(metrics["total_tasks"], 10)
        self.assertEqual(metrics["completed"], 10)
        self.assertEqual(metrics["failed"], 0)
        self.assertEqual(metrics["success_rate"], "100.0%")


if __name__ == "__main__":
    unittest.main()
