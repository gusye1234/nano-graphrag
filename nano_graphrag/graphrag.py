import asyncio
from dataclasses import dataclass
from ._llm import gpt_4o_complete, gpt_4o_mini_complete
from ._utils import limit_async_func_call
from .prompt import prompts


@dataclass
class GraphRAG:

    best_model_func: callable = gpt_4o_complete
    best_model_max_async: int = 8

    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_async: int = 8

    def __post_init__(self):
        self.best_model_func = limit_async_func_call(
            max_size=self.best_model_max_async
        )(self.best_model_func)

        self.cheap_model_func = limit_async_func_call(
            max_size=self.cheap_model_max_async
        )(self.cheap_model_func)

    async def aquery(self, query):
        return await self.best_model_func(query)

    def query(self):
        pass

    async def ainsert(self, string_or_strings):
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        pass

    def ainsert(self):
        pass
