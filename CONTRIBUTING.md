# Contributing to nano-graphrag

### Submit your Contribution through PR

To make a contribution, follow these steps:

1. Fork and clone this repository
3. If you modified the core code (`./nano_graphrag`), please add tests for it
4. **Include proper documentation / docstring or examples**
5. Ensure that all tests pass by running `pytest`
6. Submit a pull request

For more details about pull requests, please read [GitHub's guides](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).



### Only add a dependency when we have to

`nano-graphrag` needs to be `nano` and `light`. If we want to add more features, we add them smartly. Don't introduce a huge dependency just for a simple function.