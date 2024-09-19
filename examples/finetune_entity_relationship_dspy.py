import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2
from dspy.evaluate import Evaluate
import asyncio
import os
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
import logging
import pickle

from nano_graphrag._utils import compute_mdhash_id
from nano_graphrag.entity_extraction.extract import generate_dataset
from nano_graphrag.entity_extraction.module import EntityRelationshipExtractor
from nano_graphrag.entity_extraction.module_typed import TypedEntityRelationshipExtractor
from nano_graphrag.entity_extraction.metric import relationship_similarity_metric, entity_recall_metric


WORKING_DIR = "./nano_graphrag_cache_finetune_entity_relationship_dspy"

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.DEBUG)

np.random.seed(1337)


if __name__ == "__main__":
    system_prompt = """
        You are a world-class AI system, capable of complex reasoning and reflection. 
        Reason through the query, and then provide your final response. 
        If you detect that you made a mistake in your reasoning at any point, correct yourself.
        Think carefully.
    """
    # lm = dspy.OllamaLocal(
    #     model="llama3.1", 
    #     model_type="chat",
    #     system=system_prompt,
    #     temperature=1.0,
    #     top_p=1.0,
    #     max_tokens=4096
    # )
    lm = dspy.OpenAI(
        model="deepseek-chat", 
        model_type="chat", 
        api_key=os.environ["DEEPSEEK_API_KEY"], 
        base_url=os.environ["DEEPSEEK_BASE_URL"], 
        system_prompt=system_prompt, 
        temperature=1.0,
        top_p=1.0,
        max_tokens=4096
    )
    dspy.settings.configure(lm=lm, experimental=True)

    entity_relationship_trainset_path = os.path.join(WORKING_DIR, "entity_relationship_extraction_news_trainset.pkl")
    entity_relationship_valset_path = os.path.join(WORKING_DIR, "entity_relationship_extraction_news_valset.pkl")
    entity_relationship_devset_path = os.path.join(WORKING_DIR, "entity_relationship_extraction_news_devset.pkl")

    trainset = pickle.load(open(entity_relationship_trainset_path, "rb"))
    valset = pickle.load(open(entity_relationship_valset_path, "rb"))
    devset = pickle.load(open(entity_relationship_devset_path, "rb"))

    trainset = [example for example in trainset if len(example.relationships) > 0 and len(example.entities) > 0]
    valset = [example for example in valset if len(example.relationships) > 0 and len(example.entities) > 0]
    devset = [example for example in devset if len(example.relationships) > 0 and len(example.entities) > 0]

    model = TypedEntityRelationshipExtractor()

    # prediction = model(trainset[0].input_text)
    # import pdb; pdb.set_trace()
    # print(prediction)

    optimizer = MIPROv2(
        prompt_model=lm,
        task_model=lm,
        metric=entity_recall_metric,
        init_temperature=1.4,
        num_candidates=3,
        verbose=True
    )
    kwargs = dict(num_threads=os.cpu_count(), display_progress=True, display_table=0)
    miprov2_model = optimizer.compile(
        model, 
        trainset=trainset[:3], 
        requires_permission_to_run=False,
        num_batches=10, 
        max_labeled_demos=2, 
        max_bootstrapped_demos=2, 
        eval_kwargs=kwargs
    )