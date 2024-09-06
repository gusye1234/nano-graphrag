import dspy
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
import logging
import asyncio
from nano_graphrag._op import extract_entities, extract_entities_dspy
from nano_graphrag._storage import NetworkXStorage, BaseKVStorage
from nano_graphrag._utils import compute_mdhash_id, compute_args_hash
from nano_graphrag.prompt import PROMPTS

WORKING_DIR = "./nano_graphrag_cache_dspy_entity"

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.DEBUG)


class DeepSeek(dspy.Module):
    def __init__(self, model, api_key, **kwargs):
        self.model = model
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.provider = "default", 
        self.history = [] 
        self.kwargs = {
            "temperature": 0.2,
            "max_tokens": 2048,
            **kwargs
        }

    def basic_request(self, prompt, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            **self.kwargs
        )
        self.history.append({"prompt": prompt, "response": response})
        return response 

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.basic_request(prompt, **kwargs)
        completions = [choice.message.content for choice in response.choices]
        return completions 

    def inspect_history(self, n: int = 1):
        if len(self.history) < n:
            return self.history
        return self.history[-n:]


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


class EntityTypeExtractionSignature(dspy.Signature):
    input_text = dspy.InputField(desc="The text to extract entity types from")
    entity_types = dspy.OutputField(desc="List of entity types present in the text")


class EntityExtractionSignature(dspy.Signature):
    input_text = dspy.InputField(desc="The text to extract entities and relationships from")
    entities = dspy.OutputField(desc="List of extracted entities with their types and descriptions")
    relationships = dspy.OutputField(desc="List of relationships between entities, including descriptions and importance scores")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning for entity and relationship extraction")


class EntityExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.type_extractor = dspy.ChainOfThought(EntityTypeExtractionSignature)
        self.cot = dspy.ChainOfThought(EntityExtractionSignature)

    def forward(self, input_text):
        type_result = self.type_extractor(input_text=input_text)
        entity_types = type_result.entity_types
        prompt_template = PROMPTS["entity_extraction"]
        formatted_prompt = prompt_template.format(
            input_text=input_text,
            entity_types=entity_types,
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        )
        return self.cot(input_text=formatted_prompt)


# def extract_entities_dspy(text):
#     dspy_extractor = EntityExtractor()
#     dspy_result = dspy_extractor(input_text=text)
    
#     print("DSPY Result:")
#     print("\nReasoning:")
#     print(dspy_result.reasoning)
#     print("\nEntities:")
#     entities = dspy_result.entities.split(PROMPTS["DEFAULT_RECORD_DELIMITER"])
#     for entity in entities:
#         if entity.strip():
#             print(entity.strip())
    
#     print("\nRelationships:")
#     relationships = dspy_result.relationships.split(PROMPTS["DEFAULT_RECORD_DELIMITER"])
#     for relationship in relationships:
#         if relationship.strip():
#             print(relationship.strip())


async def nano_entity_extraction(text):
    graph_storage = NetworkXStorage(namespace="test", global_config={
        "working_dir": WORKING_DIR,
        "entity_summary_to_max_tokens": 500,
        "cheap_model_func": deepseepk_model_if_cache,
        "best_model_func": deepseepk_model_if_cache,
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
    lm = DeepSeek(model="deepseek-chat", api_key=os.environ["DEEPSEEK_API_KEY"])
    dspy.settings.configure(lm=lm)
    
    with open("./examples/data/test.txt", encoding="utf-8-sig") as f:
        text = f.read()

    asyncio.run(nano_entity_extraction(text))
    # extract_entities_dspy(text)