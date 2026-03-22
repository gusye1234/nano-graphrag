import pytest
import numpy as np
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from nano_graphrag import _llm


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

def test_get_minimax_async_client_instance():
    _llm.global_minimax_async_client = None
    with patch("nano_graphrag._llm.AsyncOpenAI") as mock_openai, \
         patch.dict("os.environ", {}, clear=False):
        # Remove MINIMAX_API_KEY if present so we test the fallback
        import os
        os.environ.pop("MINIMAX_API_KEY", None)
        mock_openai.return_value = "MINIMAX_CLIENT"
        client = _llm.get_minimax_async_client_instance()
    assert client == "MINIMAX_CLIENT"
    mock_openai.assert_called_once_with(
        api_key="",
        base_url="https://api.minimax.io/v1",
    )
    _llm.global_minimax_async_client = None


def test_get_minimax_async_client_singleton():
    _llm.global_minimax_async_client = None
    with patch("nano_graphrag._llm.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = "MINIMAX_CLIENT"
        client1 = _llm.get_minimax_async_client_instance()
        client2 = _llm.get_minimax_async_client_instance()
    assert client1 is client2
    mock_openai.assert_called_once()
    _llm.global_minimax_async_client = None


def test_get_minimax_async_client_with_env_key():
    _llm.global_minimax_async_client = None
    with patch("nano_graphrag._llm.AsyncOpenAI") as mock_openai, \
         patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key-123"}):
        mock_openai.return_value = "MINIMAX_CLIENT"
        _llm.get_minimax_async_client_instance()
    mock_openai.assert_called_once_with(
        api_key="test-key-123",
        base_url="https://api.minimax.io/v1",
    )
    _llm.global_minimax_async_client = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_minimax_client():
    _llm.global_minimax_async_client = None
    with patch("nano_graphrag._llm.get_minimax_async_client_instance") as mock:
        mock_client = AsyncMock()
        mock.return_value = mock_client
        yield mock_client
    _llm.global_minimax_async_client = None


# ---------------------------------------------------------------------------
# LLM completion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_minimax_complete_if_cache(mock_minimax_client):
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="answer"))]
    mock_minimax_client.chat.completions.create.return_value = mock_response

    result = await _llm.minimax_complete_if_cache(
        "MiniMax-M2.7", "hello", system_prompt="sys"
    )

    mock_minimax_client.chat.completions.create.assert_awaited_once_with(
        model="MiniMax-M2.7",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
    )
    assert result == "answer"


@pytest.mark.asyncio
async def test_minimax_complete_no_system_prompt(mock_minimax_client):
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="response"))]
    mock_minimax_client.chat.completions.create.return_value = mock_response

    result = await _llm.minimax_complete_if_cache("MiniMax-M2.7", "prompt")

    mock_minimax_client.chat.completions.create.assert_awaited_once_with(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": "prompt"}],
    )
    assert result == "response"


@pytest.mark.asyncio
async def test_minimax_complete_with_history(mock_minimax_client):
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="result"))]
    mock_minimax_client.chat.completions.create.return_value = mock_response

    history = [
        {"role": "user", "content": "prev_q"},
        {"role": "assistant", "content": "prev_a"},
    ]
    result = await _llm.minimax_complete_if_cache(
        "MiniMax-M2.7", "new_q", history_messages=history
    )

    expected = history + [{"role": "user", "content": "new_q"}]
    mock_minimax_client.chat.completions.create.assert_awaited_once_with(
        model="MiniMax-M2.7", messages=expected,
    )
    assert result == "result"


@pytest.mark.asyncio
async def test_minimax_complete_with_cache_miss(mock_minimax_client):
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="fresh"))]
    mock_minimax_client.chat.completions.create.return_value = mock_response

    mock_kv = AsyncMock()
    mock_kv.get_by_id.return_value = None

    result = await _llm.minimax_complete_if_cache(
        "MiniMax-M2.7", "q", hashing_kv=mock_kv
    )

    assert result == "fresh"
    mock_kv.upsert.assert_awaited_once()
    mock_kv.index_done_callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_minimax_complete_with_cache_hit(mock_minimax_client):
    mock_kv = AsyncMock()
    mock_kv.get_by_id.return_value = {"return": "cached"}

    result = await _llm.minimax_complete_if_cache(
        "MiniMax-M2.7", "q", hashing_kv=mock_kv
    )

    assert result == "cached"
    mock_minimax_client.chat.completions.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# Convenience model wrappers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_minimax_m2_7_complete(mock_minimax_client):
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="m2.7"))]
    mock_minimax_client.chat.completions.create.return_value = mock_response

    result = await _llm.minimax_m2_7_complete("prompt", system_prompt="sys")

    mock_minimax_client.chat.completions.create.assert_awaited_once_with(
        model="MiniMax-M2.7",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prompt"},
        ],
    )
    assert result == "m2.7"


@pytest.mark.asyncio
async def test_minimax_m2_5_highspeed_complete(mock_minimax_client):
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="fast"))]
    mock_minimax_client.chat.completions.create.return_value = mock_response

    result = await _llm.minimax_m2_5_highspeed_complete("prompt", system_prompt="sys")

    mock_minimax_client.chat.completions.create.assert_awaited_once_with(
        model="MiniMax-M2.5-highspeed",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prompt"},
        ],
    )
    assert result == "fast"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _make_aiohttp_mock(json_data):
    """Helper to create properly nested aiohttp session/response mocks."""
    mock_resp = MagicMock()
    mock_resp.json = AsyncMock(return_value=json_data)

    # session.post() returns an async context manager
    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_post_cm
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


@pytest.mark.asyncio
async def test_minimax_embedding():
    mock_session = _make_aiohttp_mock({
        "vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "total_tokens": 10,
        "base_resp": {"status_code": 0, "status_msg": "success"},
    })

    with patch("nano_graphrag._llm.aiohttp.ClientSession", return_value=mock_session):
        result = await _llm.minimax_embedding(["hello", "world"])

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))


@pytest.mark.asyncio
async def test_minimax_embedding_error():
    mock_session = _make_aiohttp_mock({
        "base_resp": {"status_code": 1001, "status_msg": "invalid api key"},
    })

    with patch("nano_graphrag._llm.aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(Exception, match="MiniMax embedding error"):
            await _llm.minimax_embedding(["test"])


@pytest.mark.asyncio
async def test_minimax_embedding_attributes():
    assert _llm.minimax_embedding.embedding_dim == 1536
    assert _llm.minimax_embedding.max_token_size == 8192


# ---------------------------------------------------------------------------
# Integration tests (require MINIMAX_API_KEY, skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.skipif(
    not __import__("os").environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)
async def test_integration_minimax_complete():
    _llm.global_minimax_async_client = None
    result = await _llm.minimax_m2_7_complete("Say 'hello' and nothing else.")
    assert "hello" in result.lower()
    _llm.global_minimax_async_client = None


@pytest.mark.asyncio
@pytest.mark.skipif(
    not __import__("os").environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)
async def test_integration_minimax_m2_5_highspeed_complete():
    _llm.global_minimax_async_client = None
    result = await _llm.minimax_m2_5_highspeed_complete(
        "What is 2+2? Answer with just the number."
    )
    assert "4" in result
    _llm.global_minimax_async_client = None


@pytest.mark.asyncio
@pytest.mark.skipif(
    not __import__("os").environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)
async def test_integration_minimax_embedding():
    _llm.global_minimax_async_client = None
    result = await _llm.minimax_embedding(["hello world"])
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1536)
    _llm.global_minimax_async_client = None
