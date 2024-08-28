import asyncio
import time
import numpy as np
from tqdm import tqdm
from nano_graphrag import GraphRAG
from nano_graphrag._storage import NanoVectorDBStorage, HNSWVectorStorage
from nano_graphrag._utils import wrap_embedding_func_with_attrs


WORKING_DIR = "./nano_graphrag_cache_benchmark_hnsw_vs_nano_vector_storage"
DATA_LEN = 100_000
FAKE_DIM = 1024
BATCH_SIZE = 100000


@wrap_embedding_func_with_attrs(embedding_dim=FAKE_DIM, max_token_size=8192)
async def sample_embedding(texts: list[str]) -> np.ndarray:
    return np.float32(np.random.rand(len(texts), FAKE_DIM))


def generate_test_data():
    return {str(i): {"content": f"Test content {i}"} for i in range(DATA_LEN)}


async def benchmark_storage(storage_class, name):
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=sample_embedding)
    storage = storage_class(
        namespace=f"benchmark_{name}",
        global_config=rag.__dict__,
        embedding_func=sample_embedding,
        meta_fields={"content"},
    )

    test_data = generate_test_data()
    
    print(f"Benchmarking {name}...")
    with tqdm(total=DATA_LEN, desc=f"{name} Benchmark") as pbar:
        start_time = time.time()
        for i in range(0, len(test_data), BATCH_SIZE):
            batch = {k: test_data[k] for k in list(test_data.keys())[i:i+BATCH_SIZE]}
            await storage.upsert(batch)
            pbar.update(min(BATCH_SIZE, DATA_LEN - i))
        
        insert_time = time.time() - start_time

        save_start_time = time.time()
        await storage.index_done_callback()
        save_time = time.time() - save_start_time
        pbar.update(1)

        query_vector = np.random.rand(FAKE_DIM)
        query_times = []
        for _ in range(100):
            query_start = time.time()
            await storage.query(query_vector, top_k=10)
            query_times.append(time.time() - query_start)
            pbar.update(1)
    
    avg_query_time = sum(query_times) / len(query_times)
    
    print(f"{name} - Insert: {insert_time:.2f}s, Save: {save_time:.2f}s, Avg Query: {avg_query_time:.4f}s")
    return insert_time, save_time, avg_query_time


async def run_benchmarks():
    print("Running NanoVectorDB benchmark...")
    nano_insert_time, nano_save_time, nano_query_time = await benchmark_storage(NanoVectorDBStorage, "nano")
    
    print("\nRunning HNSWVectorStorage benchmark...")
    hnsw_insert_time, hnsw_save_time, hnsw_query_time = await benchmark_storage(HNSWVectorStorage, "hnsw")
    
    print("\nBenchmark Results:")
    print(f"NanoVectorDB - Insert: {nano_insert_time:.2f}s, Save: {nano_save_time:.2f}s, Avg Query: {nano_query_time:.4f}s")
    print(f"HNSWVectorStorage - Insert: {hnsw_insert_time:.2f}s, Save: {hnsw_save_time:.2f}s, Avg Query: {hnsw_query_time:.4f}s")


if __name__ == "__main__":
    asyncio.run(run_benchmarks())