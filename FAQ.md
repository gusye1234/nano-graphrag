### `Leiden.EmptyNetworkError:EmptyNetworkError`

This error is caused by `nano-graphrag` tries to compute communities on an empty network. In most cases, this is caused by the LLM model you're using, it fails to extract any entities or relations, so the graph is empty.

Try to use another bigger LLM.