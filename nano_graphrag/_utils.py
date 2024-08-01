import os
import json
import logging
import inspect
import asyncio
import nest_asyncio
import tiktoken
import nanoid
from functools import wraps

logger = logging.getLogger("nano-graphrag")
ENCODER = None

nest_asyncio.apply()
logger.debug("Apply nest_asyncio patch")


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def generate_id(prefix: str = "", size=16):
    return prefix + nanoid.generate(size=size)


def limit_async_func_call(max_size=8, wait_after_seconds=0.01):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        assert inspect.iscoroutinefunction(func), "func must be a coroutine function"
        sem = asyncio.Semaphore(max_size)

        @wraps(func)
        async def wait_func(*args, **kwargs):
            async with sem:
                result = await func(*args, **kwargs)
            return result

        return wait_func

    return final_decro


def write_json(json_obj, file_name):
    with open(file_name, "w") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name) as f:
        return json.load(f)
