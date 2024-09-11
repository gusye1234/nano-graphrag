import os
import dspy
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate

from nano_graphrag._utils import logger
from nano_graphrag.entity_extraction.dataset import load_entity_relationship_dataset
from nano_graphrag.entity_extraction.metric import relationship_similarity_metric


def load_compiled_model(model: dspy.Module, dataset_path: str, module_path: str, **kwargs) -> dspy.Module:
    try:
        model.load(module_path)
        logger.info(f"Successfully loaded optimized DSPy module from: {module_path}")
        return model
    except FileNotFoundError:
        logger.warning(f"DSPy module path `{module_path}` does not exist, attempting to fine tune from scratch and save the compiled model...")      
        return compile_model(model=model, dataset_path=dataset_path, module_path=module_path, **kwargs)


def compile_model(model: dspy.Module, dataset_path: str, module_path: str, **kwargs) -> dspy.Module:
    dataset = load_entity_relationship_dataset(filepath=dataset_path)
    evaluate = Evaluate(
        devset=dataset, 
        metric=kwargs.get('metric', relationship_similarity_metric), 
        num_threads=kwargs.get('num_threads', os.cpu_count()), 
        display_progress=kwargs.get('display_progress', True), 
        display_table=kwargs.get('display_table', 5)
    )
    logger.info(f"Evaluating uncompiled DSPy module")
    evaluate(model)
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=kwargs.get('metric', relationship_similarity_metric), 
        num_threads=kwargs.get('num_threads', os.cpu_count()),
        num_candidate_programs=kwargs.get('num_candidate_programs', 5),
        max_labeled_demos=kwargs.get('max_labeled_demos', 8),
    )
    logger.info(f"Optimizing DSPy module")
    optimized_model = optimizer.compile(model, trainset=dataset)
    evaluate(optimized_model)
    optimized_model.save(module_path)
    logger.info(f"Successfully saved optimized DSPy module to {module_path}")
    return optimized_model
