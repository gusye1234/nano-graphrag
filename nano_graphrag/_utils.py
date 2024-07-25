import logging
import inspect
import asyncio
import nest_asyncio
from functools import wraps
from dataclasses import dataclass

logger = logging.getLogger("nano-graphrag")


nest_asyncio.apply()
logger.debug("Apply nest_asyncio patch")


def limit_async_func_call(max_size=8, wait_after_seconds=0.01):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        assert inspect.iscoroutinefunction(func), "func must be a coroutine function"
        current_running = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal current_running
            while True:
                if current_running < max_size:
                    current_running += 1
                    break
                await asyncio.sleep(wait_after_seconds)
            print(f"running {current_running} / {max_size}")
            try:
                result = await func(*args, **kwargs)
            finally:
                current_running -= 1
            return result

        return wait_func

    return final_decro
