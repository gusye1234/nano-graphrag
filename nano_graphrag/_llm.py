from openai import AsyncOpenAI
from functools import partial
from ._utils import encode_string_by_tiktoken, decode_tokens_by_tiktoken

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


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content,
                "chunk_order_index": index,
            }
        )
    return results
