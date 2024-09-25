import pytest
import numpy as np
from unittest.mock import AsyncMock, Mock, patch
from nano_graphrag import _llm


def test_get_openai_async_client_instance():
    with patch("nano_graphrag._llm.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = "CLIENT"
        client = _llm.get_openai_async_client_instance()
    assert client == "CLIENT"


def test_get_azure_openai_async_client_instance():
    with patch("nano_graphrag._llm.AsyncAzureOpenAI") as mock_openai:
        mock_openai.return_value = "AZURE_CLIENT"
        client = _llm.get_azure_openai_async_client_instance()
    assert client == "AZURE_CLIENT"


@pytest.fixture
def mock_openai_client():
    with patch("nano_graphrag._llm.get_openai_async_client_instance") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_azure_openai_client():
    with patch(
        "nano_graphrag._llm.get_azure_openai_async_client_instance"
    ) as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        yield mock_client


@pytest.mark.asyncio
async def test_openai_gpt4o(mock_openai_client):
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="1"))]
    messages = [{"role": "system", "content": "3"}, {"role": "user", "content": "2"}]
    mock_openai_client.chat.completions.create.return_value = mock_response

    response = await _llm.gpt_4o_complete("2", system_prompt="3")

    mock_openai_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        messages=messages,
    )
    assert response == "1"


@pytest.mark.asyncio
async def test_openai_gpt4omini(mock_openai_client):
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="1"))]
    messages = [{"role": "system", "content": "3"}, {"role": "user", "content": "2"}]
    mock_openai_client.chat.completions.create.return_value = mock_response

    response = await _llm.gpt_4o_mini_complete("2", system_prompt="3")

    mock_openai_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=messages,
    )
    assert response == "1"


@pytest.mark.asyncio
async def test_azure_openai_gpt4o(mock_azure_openai_client):
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="1"))]
    messages = [{"role": "system", "content": "3"}, {"role": "user", "content": "2"}]
    mock_azure_openai_client.chat.completions.create.return_value = mock_response

    response = await _llm.azure_gpt_4o_complete("2", system_prompt="3")

    mock_azure_openai_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        messages=messages,
    )
    assert response == "1"


@pytest.mark.asyncio
async def test_azure_openai_gpt4omini(mock_azure_openai_client):
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="1"))]
    messages = [{"role": "system", "content": "3"}, {"role": "user", "content": "2"}]
    mock_azure_openai_client.chat.completions.create.return_value = mock_response

    response = await _llm.azure_gpt_4o_mini_complete("2", system_prompt="3")

    mock_azure_openai_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=messages,
    )
    assert response == "1"


@pytest.mark.asyncio
async def test_openai_embedding(mock_openai_client):
    mock_response = AsyncMock()
    mock_response.data = [Mock(embedding=[1, 1, 1])]
    texts = ["Hello world"]
    mock_openai_client.embeddings.create.return_value = mock_response

    response = await _llm.openai_embedding(texts)

    mock_openai_client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    # print(response)
    assert np.allclose(response, np.array([[1, 1, 1]]))


@pytest.mark.asyncio
async def test_azure_openai_embedding(mock_azure_openai_client):
    mock_response = AsyncMock()
    mock_response.data = [Mock(embedding=[1, 1, 1])]
    texts = ["Hello world"]
    mock_azure_openai_client.embeddings.create.return_value = mock_response

    response = await _llm.azure_openai_embedding(texts)

    mock_azure_openai_client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    # print(response)
    assert np.allclose(response, np.array([[1, 1, 1]]))
