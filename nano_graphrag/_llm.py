from openai import AsyncOpenAI
from functools import partial
from ._utils import logger, limit_async_func_call

openai_async_client = AsyncOpenAI()


async def openai_complete(model, prompt, system_prompt=None, **kwargs) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    return response.choices[0].message.content


async def gpt_4o_complete(prompt, system_prompt=None, **kwargs) -> str:
    return await openai_complete(
        "gpt-4o", prompt, system_prompt=system_prompt, **kwargs
    )


async def gpt_4o_mini_complete(prompt, system_prompt=None, **kwargs) -> str:
    return await openai_complete(
        "gpt-4o-mini", prompt, system_prompt=system_prompt, **kwargs
    )
