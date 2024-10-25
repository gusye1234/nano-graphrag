1. Install [Neo4j](https://neo4j.com/docs/operations-manual/current/installation/) (please use 5.x version)
2. Install Neo4j GDS (graph data science) [plugin](https://neo4j.com/docs/graph-data-science/current/installation/neo4j-server/)
3. Start neo4j server
4. Get the `NEO4J_URL`,  `NEO4J_USER` and `NEO4J_PASSWORD`
   - By default, `NEO4J_URL` is `neo4j://localhost:7687` ,  `NEO4J_USER` is `neo4j` and `NEO4J_PASSWORD` is `neo4j`

Pass your neo4j instance to `GraphRAG`:

```python
from nano_graphrag import GraphRAG
from nano_graphrag._storage import Neo4jStorage

neo4j_config = {
  "neo4j_url": os.environ.get("NEO4J_URL", "neo4j://localhost:7687"),
  "neo4j_auth": (
      os.environ.get("NEO4J_USER", "neo4j"),
      os.environ.get("NEO4J_PASSWORD", "neo4j"),
  )
}
GraphRAG(
  graph_storage_cls=Neo4jStorage,
  addon_params=neo4j_config,
)
```



