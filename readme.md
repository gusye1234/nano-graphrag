<div align="center">
  <h1>nano-GraphRAG</h1>
  <p><strong>A simple, easy-to-hack GraphRAG implementation</strong></p>
   <p><strong>⚠️ It's still under development and not ready yet ⚠️</strong></p>
  <p>
    <img src="https://img.shields.io/badge/in-developing-red">
    <img src="https://img.shields.io/badge/python->=3.9-blue">
    <a href="https://pypi.org/project/nano-graphrag/">
      <img src="https://img.shields.io/pypi/v/nano-graphrag.svg">
    </a>
  </p>
</div>



😭 [GraphRAG](https://arxiv.org/pdf/2404.16130) is good and powerful, but the official [implementation](https://github.com/microsoft/graphrag/tree/main) is difficult/painful to read or hack.

😊 This project provides a simple GraphRAG implementation with minimal(ish?) dependencies.

🎁 `nano-graphrag` is about 700 lines of code, excluding `tests` and prompts. 

👌 Small yet scalable. Fully asynchronous. You can switch any component you want!



## TODO before publishing

- [x] Index
  - [x] Chunking
  - [x] Entity extraction
  - [x] Entity summary
  - [x] Compute communities
  - [x] Entities Embedding
  - [x] Community Report
- [ ] Query
  - [ ] Global
  - [ ] Local



## Install

**Install from PyPi**

```shell
pip install nano-graphrag
```

**Install from source**

```shell
# clone this repo first
cd nano-graphrag
pip install -e .
```



## Quick Start - Not yet

download a copy of A Christmas Carol by Charles Dickens:

```shell
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
```

Use the below python snippet:

```python
from nano_graphrag import GraphRAG

graph_func = GraphRAG(working_dir="./dickens")

with open("./book.txt") as f
    graph_func.insert(f.read())

print(graph_func.query("What are the top themes in this story?"))
```

Next time you initialize a `GraphRAG` from the same `working_dir`, it will reload all the contexts automatically.

### Async Support

For each method `NAME(...)` , there is a corresponding async method `aNAME(...)`

```python
await graph_func.ainsert(...)
await graph_func.aquery(...)
...
```



## Advanced:  Storage

You can replace all storage-related components to your own implementation, `nano-graphrag` mainly uses three kinds of storage:

- `base.BaseKVStorage` for storing key-json pairs of data. 
  - By default we use disk file storage as the backend. 
  -  `GraphRAG(.., key_string_value_json_storage_cls=YOURS,...)`
- `base.BaseVectorStorage` for indexing embeddings. 
  - By default we use [`milvus-lite`](https://github.com/milvus-io/milvus-lite) as the backend.
  - `GraphRAG(.., vector_db_storage_cls=YOURS,...)`
- `base.BaseGraphStorage` for storing knowledge graph. 
  - By default we use [`networkx`](https://github.com/networkx/networkx) as the backend.
  - `GraphRAG(.., graph_storage_cls=YOURS,...)`



## Benchmark - Not yet

...