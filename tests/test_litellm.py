import pytest
import numpy as np
from unittest.mock import AsyncMock, Mock, patch
from nano_graphrag import _llm


@pytest.mark.asyncio
async def test_litellm_complete_if_cache():
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="1"))]
    messages = [{"role": "system", "content": "3"}, {"role": "user", "content": "2"}]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        response = await _llm.litellm_complete_if_cache(
            "anthropic/claude-sonnet-4-6", "2", system_prompt="3"
        )

        mock_acompletion.assert_awaited_once_with(
            model="anthropic/claude-sonnet-4-6",
            messages=messages,
            drop_params=True,
        )
        assert response == "1"


@pytest.mark.asyncio
async def test_litellm_complete_factory():
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="hello"))]
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        func = _llm.create_litellm_complete_function("openai/gpt-4o")
        response = await func("hi", system_prompt="sys")

        mock_acompletion.assert_awaited_once_with(
            model="openai/gpt-4o",
            messages=messages,
            drop_params=True,
        )
        assert response == "hello"


@pytest.mark.asyncio
async def test_litellm_complete_factory_extra_kwargs():
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="ok"))]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        func = _llm.create_litellm_complete_function(
            "openai/gpt-4o", api_key="sk-test"
        )
        response = await func("prompt")

        call_kwargs = mock_acompletion.call_args
        assert call_kwargs.kwargs["api_key"] == "sk-test"
        assert call_kwargs.kwargs["drop_params"] is True
        assert response == "ok"


@pytest.mark.asyncio
async def test_litellm_embedding():
    mock_response = AsyncMock()
    mock_response.data = [{"embedding": [1, 1, 1]}, {"embedding": [2, 2, 2]}]

    with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
        mock_aembedding.return_value = mock_response

        embedding_func = _llm.create_litellm_embedding_function(
            model="openai/text-embedding-3-small",
            embedding_dim=3,
        )
        response = await embedding_func(["Hello", "World"])

        mock_aembedding.assert_awaited_once_with(
            model="openai/text-embedding-3-small",
            input=["Hello", "World"],
        )
        assert np.allclose(response, np.array([[1, 1, 1], [2, 2, 2]]))
