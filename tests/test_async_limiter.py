import asyncio
import unittest

from nano_graphrag._utils import limit_async_func_call


class AsyncLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_return_value_and_metadata(self):
        @limit_async_func_call(1)
        async def example(value):
            """Example docstring."""
            return value

        self.assertEqual(await asyncio.wait_for(example(42), 1), 42)
        self.assertEqual(example.__name__, "example")
        self.assertEqual(example.__doc__, "Example docstring.")
        self.assertTrue(hasattr(example, "__wrapped__"))

    async def test_exception_propagates_and_slot_is_reused(self):
        @limit_async_func_call(1, waitting_time=0)
        async def example(fail):
            if fail:
                raise ValueError("failure")
            return "reused"

        with self.assertRaisesRegex(ValueError, "failure"):
            await asyncio.wait_for(example(True), 1)
        self.assertEqual(await asyncio.wait_for(example(False), 1), "reused")

    async def test_inflight_cancellation_releases_slot(self):
        entered = asyncio.Event()
        hold = asyncio.Event()

        @limit_async_func_call(1, waitting_time=0)
        async def example(block):
            if block:
                entered.set()
                await hold.wait()
            return "reused"

        task = asyncio.create_task(example(True))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(await asyncio.wait_for(example(False), 1), "reused")

    async def test_cancelling_waiter_does_not_release_running_slot(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        third_entered = asyncio.Event()

        @limit_async_func_call(1, waitting_time=0)
        async def example(which):
            if which == "first":
                entered.set()
                await release.wait()
            if which == "third":
                third_entered.set()

        first = asyncio.create_task(example("first"))
        await asyncio.wait_for(entered.wait(), 1)
        waiter = asyncio.create_task(example("waiter"))
        await asyncio.sleep(0)  # Let the waiter reach the limiter's polling loop.
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        third = asyncio.create_task(example("third"))
        try:
            await asyncio.sleep(0)  # Let the third call try to enter.
            self.assertFalse(third_entered.is_set())
            release.set()
            await asyncio.wait_for(asyncio.gather(first, third), 1)
            self.assertTrue(third_entered.is_set())
        finally:
            for task in (first, third):
                task.cancel()
            await asyncio.gather(first, third, return_exceptions=True)

    async def test_concurrency_limit_under_contention(self):
        release = asyncio.Event()
        two_entered = asyncio.Event()
        entered = 0
        active = 0
        peak = 0

        @limit_async_func_call(2, waitting_time=0)
        async def example():
            nonlocal entered, active, peak
            entered += 1
            active += 1
            peak = max(peak, active)
            if entered == 2:
                two_entered.set()
            try:
                await release.wait()
            finally:
                active -= 1

        tasks = [asyncio.create_task(example()) for _ in range(6)]
        try:
            await asyncio.wait_for(two_entered.wait(), 1)
            await asyncio.sleep(0)  # Allow all remaining calls to contend.
            self.assertEqual(entered, 2)
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 1)
            self.assertEqual(entered, 6)
            self.assertEqual(peak, 2)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
