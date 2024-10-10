import dspy
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import logging
import asyncio
import time
import shutil
from nano_graphrag.entity_extraction.extract import extract_entities_dspy
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._storage import NetworkXStorage
from nano_graphrag._utils import compute_mdhash_id, compute_args_hash
from nano_graphrag._op import extract_entities

WORKING_DIR = "./nano_graphrag_cache_dspy_entity"

load_dotenv()

logger = logging.getLogger("nano-graphrag")
logger.setLevel(logging.DEBUG)


async def deepseepk_model_if_cache(
    prompt: str, model: str = "deepseek-chat", system_prompt : str = None, history_messages: list = [], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


async def benchmark_entity_extraction(text: str, system_prompt: str, use_dspy: bool = False):
    working_dir = os.path.join(WORKING_DIR, f"use_dspy={use_dspy}")
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)

    start_time = time.time()
    graph_storage = NetworkXStorage(namespace="test", global_config={
        "working_dir": working_dir,
        "entity_summary_to_max_tokens": 500,
        "cheap_model_func": lambda *args, **kwargs: deepseepk_model_if_cache(*args, system_prompt=system_prompt, **kwargs),
        "best_model_func": lambda *args, **kwargs: deepseepk_model_if_cache(*args, system_prompt=system_prompt, **kwargs),
        "cheap_model_max_token_size": 4096,
        "best_model_max_token_size": 4096,
        "tiktoken_model_name": "gpt-4o",
        "hashing_kv": BaseKVStorage(namespace="test", global_config={"working_dir": working_dir}),
        "entity_extract_max_gleaning": 1,
        "entity_extract_max_tokens": 4096,
        "entity_extract_max_entities": 100,
        "entity_extract_max_relationships": 100,
    })
    chunks = {compute_mdhash_id(text, prefix="chunk-"): {"content": text}}
    
    if use_dspy:
        graph_storage = await extract_entities_dspy(chunks, graph_storage, None, graph_storage.global_config)
    else:
        graph_storage = await extract_entities(chunks, graph_storage, None, graph_storage.global_config)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return graph_storage, execution_time


def print_extraction_results(graph_storage: NetworkXStorage):
    print("\nEntities:")
    entities = []
    for node, data in graph_storage._graph.nodes(data=True):
        entity_type = data.get('entity_type', 'Unknown')
        description = data.get('description', 'No description')
        entities.append(f"- {node} ({entity_type}):\n  {description}")
    print("\n".join(entities))

    print("\nRelationships:")
    relationships = []
    for source, target, data in graph_storage._graph.edges(data=True):
        description = data.get('description', 'No description')
        relationships.append(f"- {source} -> {target}:\n  {description}")
    print("\n".join(relationships))


async def run_benchmark(text: str):
    print("\nRunning benchmark with DSPy-AI:")
    system_prompt = """
    You are an expert system specialized in entity and relationship extraction from complex texts. 
    Your task is to thoroughly analyze the given text and extract all relevant entities and their relationships with utmost precision and completeness.
    """
    system_prompt_dspy = f"{system_prompt} Time: {time.time()}."
    lm = dspy.LM(
        model="deepseek/deepseek-chat", 
        model_type="chat",
        api_provider="openai",
        api_key=os.environ["DEEPSEEK_API_KEY"], 
        base_url=os.environ["DEEPSEEK_BASE_URL"], 
        system_prompt=system_prompt, 
        temperature=1.0,
        max_tokens=8192
    )
    dspy.settings.configure(lm=lm, experimental=True)
    graph_storage_with_dspy, time_with_dspy = await benchmark_entity_extraction(text, system_prompt_dspy, use_dspy=True)
    print(f"Execution time with DSPy-AI: {time_with_dspy:.2f} seconds")
    print_extraction_results(graph_storage_with_dspy)

    print("Running benchmark without DSPy-AI:")
    system_prompt_no_dspy = f"{system_prompt} Time: {time.time()}."
    graph_storage_without_dspy, time_without_dspy = await benchmark_entity_extraction(text, system_prompt_no_dspy, use_dspy=False)
    print(f"Execution time without DSPy-AI: {time_without_dspy:.2f} seconds")
    print_extraction_results(graph_storage_without_dspy)

    print("\nComparison:")
    print(f"Time difference: {abs(time_with_dspy - time_without_dspy):.2f} seconds")
    print(f"DSPy-AI is {'faster' if time_with_dspy < time_without_dspy else 'slower'}")

    entities_without_dspy = len(graph_storage_without_dspy._graph.nodes())
    entities_with_dspy = len(graph_storage_with_dspy._graph.nodes())
    relationships_without_dspy = len(graph_storage_without_dspy._graph.edges())
    relationships_with_dspy = len(graph_storage_with_dspy._graph.edges())

    print(f"Entities extracted: {entities_without_dspy} (without DSPy-AI) vs {entities_with_dspy} (with DSPy-AI)")
    print(f"Relationships extracted: {relationships_without_dspy} (without DSPy-AI) vs {relationships_with_dspy} (with DSPy-AI)")


if __name__ == "__main__":
    with open("./tests/zhuyuanzhang.txt", encoding="utf-8-sig") as f:
        text = f.read()

    asyncio.run(run_benchmark(text=text))
