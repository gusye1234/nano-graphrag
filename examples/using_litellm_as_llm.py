"""Use LiteLLM as the LLM and embedding provider.

LiteLLM supports 100+ providers (OpenAI, Anthropic, Google, Azure,
AWS Bedrock, Ollama, Groq, Mistral, and more) through a unified interface.

Install: pip install litellm

Set the appropriate environment variable for your chosen provider, e.g.:
  export OPENAI_API_KEY="sk-..."
  export ANTHROPIC_API_KEY="sk-ant-..."

Model strings follow the LiteLLM format: "provider/model-name"
  - openai/gpt-4o
  - anthropic/claude-sonnet-4-6
  - groq/llama-4-scout-17b-16e-instruct
  - ollama/llama3
See https://docs.litellm.ai/docs/providers for the full list.
"""

from nano_graphrag import GraphRAG, QueryParam

graph_func = GraphRAG(
    working_dir="../litellm_example",
    using_litellm=True,
    litellm_best_model_id="openai/gpt-4o",
    litellm_cheap_model_id="openai/gpt-4o-mini",
    litellm_embedding_model="openai/text-embedding-3-small",
    litellm_embedding_dim=1536,
)

with open("../tests/mock_data.txt") as f:
    graph_func.insert(f.read())

prompt = "What are the top themes in this story?"

# Perform global graphrag search
print(graph_func.query(prompt, param=QueryParam(mode="global")))

# Perform local graphrag search
print(graph_func.query(prompt, param=QueryParam(mode="local")))
