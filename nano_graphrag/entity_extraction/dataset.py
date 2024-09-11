import pickle
import dspy
import asyncio

from nano_graphrag._utils import logger
from nano_graphrag.entity_extraction.module import EntityRelationshipExtractor
from nano_graphrag.base import (
    TextChunkSchema,
)

entity_extractor = EntityRelationshipExtractor()


def load_entity_relationship_dataset(filepath: str) -> list[dspy.Example]:
    with open(filepath, 'rb') as f:
        examples = pickle.load(f)
        logger.info(f"Loaded {len(examples)} examples with keys: {examples[0].keys()}")
    return examples


async def generate_dataset(
    chunks: dict[str, TextChunkSchema],
    filepath: str,
    save_dataset: bool = True
) -> list[dspy.Example]:
    global entity_extractor    
    ordered_chunks = list(chunks.items())

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]) -> dspy.Example:
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        prediction = await asyncio.to_thread(
            entity_extractor, input_text=content
        )
        example = dspy.Example(
            input_text=content, 
            entities=prediction.entities, 
            relationships=prediction.relationships
        ).with_inputs("input_text")
        return example

    examples = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    if save_dataset:
        with open(filepath, 'wb') as f:
            pickle.dump(examples, f)
            logger.info(f"Saved {len(examples)} examples with keys: {examples[0].keys()}")
    
    return examples
