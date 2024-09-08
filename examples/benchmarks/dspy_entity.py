import dspy
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import logging
import asyncio
from nano_graphrag.entity_extraction.extract import extract_entities_dspy
from nano_graphrag._storage import NetworkXStorage, BaseKVStorage
from nano_graphrag._utils import compute_mdhash_id, compute_args_hash

WORKING_DIR = "./nano_graphrag_cache_dspy_entity"

load_dotenv()

logger = logging.getLogger("nano-graphrag")
logger.setLevel(logging.DEBUG)


async def deepseepk_model_if_cache(
    prompt, model: str = "deepseek-chat", system_prompt=None, history_messages=[], **kwargs
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


async def nano_entity_extraction(text: str, system_prompt: str = None):
    graph_storage = NetworkXStorage(namespace="test", global_config={
        "working_dir": WORKING_DIR,
        "entity_summary_to_max_tokens": 500,
        "cheap_model_func": lambda *args, **kwargs: deepseepk_model_if_cache(*args, system_prompt=system_prompt, **kwargs),
        "best_model_func": lambda *args, **kwargs: deepseepk_model_if_cache(*args, system_prompt=system_prompt, **kwargs),
        "cheap_model_max_token_size": 4096,
        "best_model_max_token_size": 4096,
        "tiktoken_model_name": "gpt-4o",
        "hashing_kv": BaseKVStorage(namespace="test", global_config={"working_dir": WORKING_DIR}),
        "entity_extract_max_gleaning": 1,
        "entity_extract_max_tokens": 4096,
        "entity_extract_max_entities": 100,
        "entity_extract_max_relationships": 100,
    })
    chunks = {compute_mdhash_id(text, prefix="chunk-"): {"content": text}}
    graph_storage = await extract_entities_dspy(chunks, graph_storage, None, graph_storage.global_config)

    print("Current Implementation Result:")
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


if __name__ == "__main__":
    system_prompt = """
        You are a world-class AI system, capable of complex reasoning and reflection. 
        Reason through the query, and then provide your final response. 
        If you detect that you made a mistake in your reasoning at any point, correct yourself.
        Think carefully.
    """
    lm = dspy.OpenAI(
        model="deepseek-chat", 
        model_type="chat", 
        api_key=os.environ["DEEPSEEK_API_KEY"], 
        base_url=os.environ["DEEPSEEK_BASE_URL"], 
        system_prompt=system_prompt, 
        temperature=0.3,
        top_p=1,
        max_tokens=4096
    )
    dspy.settings.configure(lm=lm)
    
    with open("./examples/data/test.txt", encoding="utf-8-sig") as f:
        text = f.read()

    asyncio.run(nano_entity_extraction(text, system_prompt))
