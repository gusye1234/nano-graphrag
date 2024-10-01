## Next Version

- [ ] Add DSpy for prompt-tuning to make small models(Qwen2 7B, Llama 3.1 8B...) can extract entities. @NumberChiffre @gusye1234
- [ ] Optimize Algorithm: add `global_local` query method, globally rewrite query then perform local search.



## In next few versions

- [ ] Add rate limiter: support token limit (tokens per second, per minute)

- [ ] Add other advanced RAG algorithms, candidates:

  - [ ] [HybridRAG](https://arxiv.org/abs/2408.04948)
  - [ ] [HippoRAG](https://arxiv.org/abs/2405.14831)
  
  
  



## Interesting directions

- [ ] Add [Sciphi Triplex](https://huggingface.co/SciPhi/Triplex) as the entity extraction model.
- [ ] Add new components, see [issue](https://github.com/gusye1234/nano-graphrag/issues/2)