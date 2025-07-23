1. Install [Memgraph MAGE](https://memgraph.com/docs/getting-started)
2. Start Memgraph server
3. Get the `MEMGRAPH_URL`,  `MEMGRAPH_USER` and `MEMGRAPH_PASSWORD`
   - By default, `MEMGRAPH_URL` is `bolt://localhost:7687` ,  `MEMGRAPH_USER` and `MEMGRAPH_PASSWORD` is empty

Short command for running MAGE on your local machine is: 

```bash 
docker run -it -p 7687:7687 -p 7444:7444 -p 3000:3000 memgraph/memgraph-mage
```

Pass your Memgraph instance to `GraphRAG`:

```python
from nano_graphrag import GraphRAG
from nano_graphrag._storage import MemgraphStorage
import os

memgraph_config = {
  "memgraph_url": os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"),
  "memgraph_auth": (
      os.environ.get("MEMGRAPH_USER", ""),
      os.environ.get("MEMGRAPH_PASSWORD", ""),
  )
}

GraphRAG(
  graph_storage_cls=MemgraphStorage,
  addon_params=memgraph_config,
)
```

