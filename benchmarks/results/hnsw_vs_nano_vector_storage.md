## `n=100_000, d=1024`
```shell
> python benchmarks/hnsw_vs_nano_vector_storage.py
Running NanoVectorDB benchmark...
INFO:nano-graphrag:Creating working directory ./nano_graphrag_cache_benchmark_hnsw_vs_nano_vector_storage
INFO:nano-graphrag:Load KV full_docs with 0 data
INFO:nano-graphrag:Load KV text_chunks with 0 data
INFO:nano-graphrag:Load KV llm_response_cache with 0 data
INFO:nano-graphrag:Load KV community_reports with 0 data
INFO:nano-vectordb:Init {'embedding_dim': 1024, 'metric': 'cosine', 'storage_file': './nano_graphrag_cache_benchmark_hnsw_vs_nano_vector_storage/vdb_entities.json'} 0 data
INFO:nano-vectordb:Init {'embedding_dim': 1024, 'metric': 'cosine', 'storage_file': './nano_graphrag_cache_benchmark_hnsw_vs_nano_vector_storage/vdb_benchmark_nano.json'} 0 data
Benchmarking nano...
nano Benchmark:   0%|                                                                                                               | 0/100000 [00:00<?, ?it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_nano
nano Benchmark: 100101it [00:04, 22875.64it/s]                                                                                                                 
nano - Insert: 1.41s, Save: 1.71s, Avg Query: 0.0125s

Running HNSWVectorStorage benchmark...
INFO:nano-graphrag:Load KV full_docs with 0 data
INFO:nano-graphrag:Load KV text_chunks with 0 data
INFO:nano-graphrag:Load KV llm_response_cache with 0 data
INFO:nano-graphrag:Load KV community_reports with 0 data
INFO:nano-vectordb:Init {'embedding_dim': 1024, 'metric': 'cosine', 'storage_file': './nano_graphrag_cache_benchmark_hnsw_vs_nano_vector_storage/vdb_entities.json'} 0 data
INFO:nano-graphrag:Created new index for benchmark_hnsw
Benchmarking hnsw...
hnsw Benchmark:   0%|                                                                                                               | 0/100000 [00:00<?, ?it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_hnsw
hnsw Benchmark: 100101it [01:23, 1197.02it/s]                                                                                                                  
hnsw - Insert: 82.94s, Save: 0.50s, Avg Query: 0.0018s

Benchmark Results:
NanoVectorDB - Insert: 1.41s, Save: 1.71s, Avg Query: 0.0125s
HNSWVectorStorage - Insert: 82.94s, Save: 0.50s, Avg Query: 0.0018s
```

## `n=1_000_000, d=1024`
```shell
> python benchmarks/hnsw_vs_nano_vector_storage.py
Running NanoVectorDB benchmark...
INFO:nano-graphrag:Creating working directory ./nano_graphrag_cache_benchmark_hnsw_vs_nano_vector_storage
INFO:nano-graphrag:Load KV full_docs with 0 data
INFO:nano-graphrag:Load KV text_chunks with 0 data
INFO:nano-graphrag:Load KV llm_response_cache with 0 data
INFO:nano-graphrag:Load KV community_reports with 0 data
INFO:nano-vectordb:Init {'embedding_dim': 1024, 'metric': 'cosine', 'storage_file': './nano_graphrag_cache_benchmark_hnsw_vs_nano_vector_storage/vdb_entities.json'} 0 data
INFO:nano-vectordb:Init {'embedding_dim': 1024, 'metric': 'cosine', 'storage_file': './nano_graphrag_cache_benchmark_hnsw_vs_nano_vector_storage/vdb_benchmark_nano.json'} 0 data
Benchmarking nano...
nano Benchmark:   0%|                                                                                                              | 0/1000000 [00:00<?, ?it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_nano
nano Benchmark:  10%|█████████▍                                                                                    | 100000/1000000 [00:01<00:13, 65004.60it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_nano
nano Benchmark:  20%|██████████████████▊                                                                           | 200000/1000000 [00:02<00:11, 67094.28it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_nano
nano Benchmark:  30%|████████████████████████████▏                                                                 | 300000/1000000 [00:04<00:10, 66806.48it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_nano
nano Benchmark:  40%|█████████████████████████████████████▌                                                        | 400000/1000000 [00:06<00:09, 64556.41it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_nano
nano Benchmark:  50%|███████████████████████████████████████████████                                               | 500000/1000000 [00:07<00:08, 61842.55it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_nano
nano Benchmark:  60%|████████████████████████████████████████████████████████▍                                     | 600000/1000000 [00:09<00:06, 59906.38it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_nano
nano Benchmark:  70%|█████████████████████████████████████████████████████████████████▊                            | 700000/1000000 [00:11<00:05, 58056.89it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_nano
nano Benchmark:  80%|███████████████████████████████████████████████████████████████████████████▏                  | 800000/1000000 [00:13<00:03, 55523.96it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_nano
nano Benchmark:  90%|████████████████████████████████████████████████████████████████████████████████████▌         | 900000/1000000 [00:15<00:01, 52749.15it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_nano
nano Benchmark: 1000101it [01:04, 15412.94it/s]                                                                                                                
nano - Insert: 18.19s, Save: 33.64s, Avg Query: 0.1306s

Running HNSWVectorStorage benchmark...
INFO:nano-graphrag:Load KV full_docs with 0 data
INFO:nano-graphrag:Load KV text_chunks with 0 data
INFO:nano-graphrag:Load KV llm_response_cache with 0 data
INFO:nano-graphrag:Load KV community_reports with 0 data
INFO:nano-vectordb:Init {'embedding_dim': 1024, 'metric': 'cosine', 'storage_file': './nano_graphrag_cache_benchmark_hnsw_vs_nano_vector_storage/vdb_entities.json'} 0 data
INFO:nano-graphrag:Created new index for benchmark_hnsw
Benchmarking hnsw...
hnsw Benchmark:   0%|                                                                                                              | 0/1000000 [00:00<?, ?it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_hnsw
hnsw Benchmark:  10%|█████████▌                                                                                     | 100000/1000000 [01:23<12:33, 1194.52it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_hnsw
hnsw Benchmark:  20%|███████████████████                                                                            | 200000/1000000 [02:55<11:45, 1133.51it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_hnsw
hnsw Benchmark:  30%|████████████████████████████▌                                                                  | 300000/1000000 [04:28<10:33, 1104.56it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_hnsw
hnsw Benchmark:  40%|██████████████████████████████████████                                                         | 400000/1000000 [06:01<09:08, 1094.29it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_hnsw
hnsw Benchmark:  50%|███████████████████████████████████████████████▌                                               | 500000/1000000 [07:34<07:40, 1086.34it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_hnsw
hnsw Benchmark:  60%|█████████████████████████████████████████████████████████                                      | 600000/1000000 [09:09<06:12, 1074.65it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_hnsw
hnsw Benchmark:  70%|██████████████████████████████████████████████████████████████████▌                            | 700000/1000000 [10:46<04:43, 1058.71it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_hnsw
hnsw Benchmark:  80%|████████████████████████████████████████████████████████████████████████████                   | 800000/1000000 [12:24<03:11, 1045.96it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_hnsw
hnsw Benchmark:  90%|█████████████████████████████████████████████████████████████████████████████████████▌         | 900000/1000000 [14:08<01:38, 1017.91it/s]INFO:nano-graphrag:Inserting 100000 vectors to benchmark_hnsw
hnsw Benchmark: 1000101it [15:54, 1047.85it/s]                                                                                                                 
hnsw - Insert: 950.02s, Save: 4.21s, Avg Query: 0.0020s

Benchmark Results:
NanoVectorDB - Insert: 18.19s, Save: 33.64s, Avg Query: 0.1306s
HNSWVectorStorage - Insert: 950.02s, Save: 4.21s, Avg Query: 0.0020s
```