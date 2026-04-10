"""Tests for the async task queue."""

import asyncio
import unittest
from src.utils.async_queue import AsyncTaskQueue, QueueTask


class TestAsyncTaskQueue(unittest.TestCase):

    def test_basic_execution(self):
        """Tasks should execute and return results."""

        async def double(x: int) -> int:
            return x * 2

        async def run():
            q = AsyncTaskQueue(max_concurrent=3, max_per_minute=60)
            tasks = [
                QueueTask(task_id=f"t{i}", coroutine_fn=double, args=(i,))
                for i in range(5)
            ]
            results = await q.process_batch(tasks)
            return results, q

        results, q = asyncio.run(run())
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0], 0)
        self.assertEqual(results[1], 2)
        self.assertEqual(q.completed_tasks, 5)
        self.assertEqual(q.failed_tasks, 0)

    def test_retry(self):
        """Failed tasks should be retried."""
        attempts = 0

        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise Exception("not yet")
            return "ok"

        async def run():
            q = AsyncTaskQueue(
                max_concurrent=1, max_per_minute=60,
                retry_attempts=3, retry_delay=0.1,
            )
            tasks = [QueueTask(task_id="flaky", coroutine_fn=flaky)]
            results = await q.process_batch(tasks)
            return results, q

        results, q = asyncio.run(run())
        self.assertEqual(results[0], "ok")
        self.assertEqual(q.completed_tasks, 1)

    def test_metrics(self):
        async def ok() -> str:
            return "ok"

        async def run():
            q = AsyncTaskQueue(max_concurrent=5, max_per_minute=60)
            tasks = [
                QueueTask(task_id=f"t{i}", coroutine_fn=ok)
                for i in range(10)
            ]
            await q.process_batch(tasks)
            return q.get_metrics()

        m = asyncio.run(run())
        self.assertEqual(m["total_tasks"], 10)
        self.assertEqual(m["completed"], 10)
        self.assertEqual(m["failed"], 0)
        self.assertEqual(m["success_rate"], "100.0%")


if __name__ == "__main__":
    unittest.main()
