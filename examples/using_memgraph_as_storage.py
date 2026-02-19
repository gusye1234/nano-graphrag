"""
Example: Using Memgraph as the graph storage backend for GraphRAG

This example demonstrates how to use Memgraph instead of the default NetworkX
for graph storage in nano-graphrag. Memgraph is a high-performance graph database
optimized for real-time analytics.

Prerequisites:
1. Install Memgraph (see docs/use_memgraph_for_graphrag.md)
2. Install required packages:
   pip install nano-graphrag neo4j

Note: Memgraph uses the same Bolt protocol as Neo4j, so we use the neo4j Python driver.
"""

import os
import sys
import asyncio
from pathlib import Path

# Import local nano_graphrag
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
def load_env_file():
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print(f"⚠️  No .env file found at {env_path}")

# Load environment variables
load_env_file()

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._storage import MemgraphStorage

# Configuration for Memgraph connection
MEMGRAPH_CONFIG = {
    "memgraph_url": os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"),
    "memgraph_auth": (
        os.environ.get("MEMGRAPH_USER", "memgraph"),
        os.environ.get("MEMGRAPH_PASSWORD", ""),
    )
}

WORKING_DIR = "./nano_graphrag_cache_memgraph_example"

async def main():
    print("Initializing GraphRAG with Memgraph storage...")
    
    # Initialize GraphRAG with Memgraph storage
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        graph_storage_cls=MemgraphStorage,
        addon_params=MEMGRAPH_CONFIG,
        enable_llm_cache=True,
    )
    
    # Sample documents to insert
    documents = [
        "The capital of France is Paris. Paris is known for the Eiffel Tower.",
        "Tokyo is the capital of Japan. Tokyo is famous for its technology and culture.",
        "London is the capital of the United Kingdom. London has the Tower Bridge.",
        "Berlin is the capital of Germany. Berlin is known for its history and art scene.",
        "Rome is the capital of Italy. Rome is famous for the Colosseum and ancient history.",
    ]
    
    print("Inserting documents...")
    await rag.ainsert(documents)
    
    print("Building graph index...")
    await rag.aquery(
        "Tell me about European capitals",
        param=QueryParam(mode="global")
    )
    
    # Query examples
    queries = [
        "What are the famous landmarks in European capitals?",
        "Compare the capitals of France and Germany",
        "Tell me about Asian capitals",
    ]
    
    print("\nRunning queries...")
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)
        
        # Global search
        global_result = await rag.aquery(
            query,
            param=QueryParam(mode="global")
        )
        print(f"Global result: {global_result}")
        
        # Local search
        local_result = await rag.aquery(
            query,
            param=QueryParam(mode="local")
        )
        print(f"Local result: {local_result}")
    
    print("\nExample completed successfully!")
    print(f"Graph data stored in Memgraph at: {MEMGRAPH_CONFIG['memgraph_url']}")

if __name__ == "__main__":
    asyncio.run(main())
