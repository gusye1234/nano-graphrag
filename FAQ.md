### `Leiden.EmptyNetworkError:EmptyNetworkError`

This error is caused by `nano-graphrag` tries to compute communities on an empty network. In most cases, this is caused by the LLM model you're using, it fails to extract any entities or relations, so the graph is empty.

Try to use another bigger LLM, or here are some ideas to fix it:

 - Check the response from the LLM, make sure the result fits the desired response format of the extracting entities prompt. 

    The desired response format is something like that:

    ```text
    ("entity"<|>"Cruz"<|>"person"<|>"Cruz is associated with a vision of control and order, influencing the dynamics among other characters.")
    ```

 - Some LLMs may not return the format like above, so one possible solution is to add a system instruction to the input prompt, such like:
    ```json
    {
        "role": "system",
        "content": "You are an intelligent assistant and will follow the instructions given to you to fulfill the goal. The answer should be in the format as in the given example."
    }
    ```
    You can use this system_prompt as default for your LLM calling funcation
    
