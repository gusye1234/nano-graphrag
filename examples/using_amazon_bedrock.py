from nano_graphrag import GraphRAG, QueryParam

graph_func = GraphRAG(
    working_dir="../bedrock_example",
    using_amazon_bedrock=True,
    best_model_id="us.anthropic.claude-3-sonnet-20240229-v1:0",
    cheap_model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
)

with open("../tests/mock_data.txt") as f:
    graph_func.insert(f.read())

prompt = "What are the top themes in this story?"

# Perform global graphrag search
print(graph_func.query(prompt, param=QueryParam(mode="global")))

# Perform local graphrag search (I think is better and more scalable one)
print(graph_func.query(prompt, param=QueryParam(mode="local")))
