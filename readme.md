<div align="center">
  <h1>nano-GraphRAG</h1>
  <p><strong>A simple, easy-to-hack GraphRAG implementation</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python->=3.9.11-blue">
    <a href="https://pypi.org/project/nano-graphrag/">
      <img src="https://img.shields.io/pypi/v/nano-graphrag.svg">
    </a>
  </p>
</div>





😭 [GraphRAG](https://arxiv.org/pdf/2404.16130) is good and powerful, but the official [implementation](https://github.com/microsoft/graphrag/tree/main) is difficult/painful to **read or hack**.

😊 This project provides a **smaller, faster, cleaner GraphRAG**, while remaining the core functionality(see [benchmark](#benchmark) and [issues](#Issues) ).

🎁 Excluding `tests` and prompts,  `nano-graphrag` is about **800 lines of code**.

👌 Small yet [**portable**](#Advances), [**asynchronous**](#Async) and fully typed.



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



## Quick Start

**Please set OpenAI API key in environment: `export OPENAI_API_KEY="sk-..."`.** 

> [!TIP]
>
> If you like to use another LLM: [LLM component](#LLM).
>
> If you like to use another Embedding Model: [Embedding](#Embedding).

download a copy of A Christmas Carol by Charles Dickens:

```shell
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
```

Use the below python snippet:

```python
from nano_graphrag import GraphRAG, QueryParam

graph_func = GraphRAG(working_dir="./dickens")

with open("./book.txt") as f:
    graph_func.insert(f.read())

# Perform global graphrag search
print(graph_func.query("What are the top themes in this story?"))

# Perform local graphrag search (I think is better and more scalable one)
print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="local")))
```

Next time you initialize a `GraphRAG` from the same `working_dir`, it will reload all the contexts automatically.

### Incremental Insert

`nano-graphrag` supports incremental insert, no duplicated computation or data will be added:

```python
with open("./book.txt") as f:
    book = f.read()
    half_len = len(book) // 2
    graph_func.insert(book[:half_len])
    graph_func.insert(book[half_len:])
```

> `nano-graphrag` use md5-hash of the content as the key, so there is no duplicated chunk.
>
> However, each time you insert, the communities of graph will be re-computed and the community reports will be re-generated

### Async

For each method `NAME(...)` , there is a corresponding async method `aNAME(...)`

```python
await graph_func.ainsert(...)
await graph_func.aquery(...)
...
```

### Available Parameters

`GraphRAG` and `QueryParam` are `dataclass` in Python. Use `help(GraphRAG)` and `help(QueryParam)` to see all available parameters! 



## Advances

### Prompt

`nano-graphrag` use prompts from `nano_graphrag.prompt.PROMPTS` dict object. You can play with it and replace any prompt inside.

Some important prompts:

- `PROMPTS["entity_extraction"]` is used to extract the entities and relations from a text chunk.
- `PROMPTS["community_report"]` is used to organize and summary the graph cluster's description.
- `PROMPTS["local_rag_response"]` is the system prompt template of the local search generation.
- `PROMPTS["global_reduce_rag_response"]` is the system prompt template of the global search generation.
- `PROMPTS["fail_response"]` is the fallback response when nothing is related to the user query.

### LLM

In `nano-graphrag`, we requires two types of LLM, a great one and a cheap one. The former is used to plan and respond, the latter is used to summary. By default, the great one is `gpt-4o` and the cheap one is `gpt-4o-mini`

You can implement your own LLM function (refer to `_llm.gpt_4o_complete`):

```python
async def my_llm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
  # pop cache KV database if any
  hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
  # the rest kwargs are for calling LLM, for example, `max_tokens=xxx`
	...
  # YOUR LLM calling
  response = await call_your_LLM(messages, **kwargs)
  return response
```

Replace the default one with:

```python
# Adjust the max token size or the max async requests if needed
GraphRAG(best_model_func=my_llm_complete, best_model_max_token_size=..., best_model_max_async=...)
GraphRAG(cheap_model_func=my_llm_complete, cheap_model_max_token_size=..., cheap_model_max_async=...)
```

You can refer to an [example](./examples/using_deepseek_as_llm.py) that use [`deepseek-chat`](https://platform.deepseek.com/api-docs/) as the LLM model in `nano-graphrag`

### Embedding

You can replace the default embedding functions with any `_utils.EmbedddingFunc` instance.

For example, the default one is using OpenAI embedding API:

```python
@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = AsyncOpenAI()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
```

Replace default embedding function with:

```python
GraphRAG(embedding_func=your_embed_func, embedding_batch_num=..., embedding_func_max_async=...)
```

You can refer to an [example](./examples/using_local_embedding_model.py) that use `sentence-transformer` to locally compute embeddings.

### Storage

You can replace all storage-related components to your own implementation, `nano-graphrag` mainly uses three kinds of storage:

**`base.BaseKVStorage` for storing key-json pairs of data** 

- By default we use disk file storage as the backend. 
- `GraphRAG(.., key_string_value_json_storage_cls=YOURS,...)`

**`base.BaseVectorStorage` for indexing embeddings**

- By default we use [`nano-vectordb`](https://github.com/gusye1234/nano-vectordb) as the backend.
- Check out this [example](./examples/using_milvus_as_vectorDB.py) that use [`milvus-lite`](https://github.com/milvus-io/milvus-lite) as the backend (not available in Windows).
- `GraphRAG(.., vector_db_storage_cls=YOURS,...)`

**`base.BaseGraphStorage` for storing knowledge graph**

- By default we use [`networkx`](https://github.com/networkx/networkx) as the backend.
- `GraphRAG(.., graph_storage_cls=YOURS,...)`

You can refer to `nano_graphrag.base` to see detailed interfaces for each components.



## Benchmark

- [benchmark for English](./benchmark-en.md)
- [benchmark for Chinese](./benchmark-zh.md)



## Issues

- `nano-graphrag` didn't implement the `covariates` feature of `GraphRAG`
- `nano-graphrag` implements the global search different from the original. The original use a map-reduce-like style to fill all the communities into context, while `nano-graphrag` only use the top-K important and central communites (use `QueryParam.global_max_conside_community` to control, default to 512 communities).



## TODO in Next Version

>  If a checkbox is filled meaning someone is on it

- [ ] `nano-graphrag`'s Data Source Id is local, meaning it always starts at 0 at any response and you have to remap it into the current session. So it's kinda useless right now.
- [ ] `nano-graphrag` truncates the community's raw description if it exceed the maximun context size when generating community report, while GraphRAG uses a sub-communities iterative summary to include all. 
- [ ] Add real benchmark with GraphRAG
- [ ] Add new components, see [issue](https://github.com/gusye1234/nano-graphrag/issues/2)

