<div align="center">
  <h1>nano-GraphRAG</h1>
  <p><strong>A simple, easy-to-hack GraphRAG implementation</strong></p>
  <p>
    <img src="https://img.shields.io/badge/in-developing-red">
    <img src="https://img.shields.io/badge/python->=3.9-blue">
    <a href="https://pypi.org/project/nano-graphrag/">
      <img src="https://img.shields.io/pypi/v/nano-graphrag.svg">
    </a>
  </p>
</div>




😭 [GraphRAG](https://arxiv.org/pdf/2404.16130) is good and powerful, but the official [implementation](https://github.com/microsoft/graphrag/tree/main) is not very "easy" to read or hack.

😊 This project aims to provide a simple implementation with few dependencies, while retaining the core functionality.

👌 `nano-graphrag` is about 500-lines of code (excluding `tests` and prompts)



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
curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt > ./book.txt
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



## Benchmark - Not yet

...