#! python3
# -*- encoding: utf-8 -*-
"""Tests for embedding clients.

@File   :   test_embeddings.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import pytest
from unittest.mock import MagicMock, patch
from isobase.knowledge.embeddings import OpenAIEmbeddingClient


class MockEmbeddingResponse:
    """Mock response from OpenAI embeddings API."""

    def __init__(self, embeddings):
        self.data = [MagicMock(embedding=emb) for emb in embeddings]


def test_openai_client_initialization():
    """Test client initialization."""
    client = OpenAIEmbeddingClient(
        api_key="test_key",
        base_url="https://test.com/v1",
        model="test-model",
        dimensions=1024,
    )

    assert client.model == "test-model"
    assert client.dimensions == 1024


def test_openai_client_empty_api_key():
    """Test client initialization with empty API key."""
    with pytest.raises(ValueError, match="api_key cannot be empty"):
        OpenAIEmbeddingClient(api_key="")


@patch("isobase.knowledge.embeddings.openai.OpenAI")
def test_embed_texts(mock_openai_class):
    """Test embedding multiple texts."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MockEmbeddingResponse([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ])
    mock_client.embeddings.create.return_value = mock_response

    # Create client and embed
    client = OpenAIEmbeddingClient(
        api_key="test_key",
        model="test-model",
        dimensions=3
    )

    embeddings = client.embed_texts(["text1", "text2"])

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]

    # Verify API was called correctly
    mock_client.embeddings.create.assert_called_once()
    call_kwargs = mock_client.embeddings.create.call_args[1]
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["input"] == ["text1", "text2"]
    assert call_kwargs["dimensions"] == 3


@patch("isobase.knowledge.embeddings.openai.OpenAI")
def test_embed_texts_empty(mock_openai_class):
    """Test embedding with empty text list."""
    client = OpenAIEmbeddingClient(api_key="test_key")

    with pytest.raises(ValueError, match="texts cannot be empty"):
        client.embed_texts([])


@patch("isobase.knowledge.embeddings.openai.OpenAI")
def test_embed_query(mock_openai_class):
    """Test embedding a single query."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MockEmbeddingResponse([[0.1, 0.2, 0.3]])
    mock_client.embeddings.create.return_value = mock_response

    # Create client and embed
    client = OpenAIEmbeddingClient(
        api_key="test_key",
        model="test-model"
    )

    embedding = client.embed_query("test query")
    assert embedding == [0.1, 0.2, 0.3]


@patch("isobase.knowledge.embeddings.openai.OpenAI")
def test_embed_query_empty(mock_openai_class):
    """Test embedding with empty query."""
    client = OpenAIEmbeddingClient(api_key="test_key")

    with pytest.raises(ValueError, match="query cannot be empty"):
        client.embed_query("")


# ---------------------------------------------------------------------------
# dimensions property tests
# ---------------------------------------------------------------------------


def test_dimensions_explicit():
    """Explicit dimensions always take priority."""
    client = OpenAIEmbeddingClient(api_key="test_key", dimensions=2048)
    assert client.dimensions == 2048


@patch("isobase.knowledge.embeddings.openai.OpenAI")
def test_dimensions_from_response(mock_openai_class):
    """Dimensions auto-detected from API response after embed_texts()."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MockEmbeddingResponse([[0.1] * 1536])
    mock_client.embeddings.create.return_value = mock_response

    client = OpenAIEmbeddingClient(api_key="test_key")

    client.embed_texts(["test"])
    assert client.dimensions == 1536


@patch("isobase.knowledge.embeddings.openai.OpenAI")
def test_dimensions_from_query_response(mock_openai_class):
    """Dimensions auto-detected from API response after embed_query()."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MockEmbeddingResponse([[0.1] * 768])
    mock_client.embeddings.create.return_value = mock_response

    client = OpenAIEmbeddingClient(api_key="test_key")

    client.embed_query("test query")
    assert client.dimensions == 768


def test_dimensions_unknown_raises():
    """Accessing dimensions before any API call raises RuntimeError."""
    client = OpenAIEmbeddingClient(api_key="test_key")

    with pytest.raises(RuntimeError, match="Dimensions unknown"):
        _ = client.dimensions


def test_dimensions_explicit_overrides_auto():
    """Explicit dimensions take priority even after auto-detection."""
    client = OpenAIEmbeddingClient(api_key="test_key", dimensions=512)
    # Even though we haven't called the API, explicit setting is enough
    assert client.dimensions == 512


@patch("isobase.knowledge.embeddings.openai.OpenAI")
def test_dimensions_cached_after_first_call(mock_openai_class):
    """Dimensions are cached after the first API call."""
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance

    # First call returns 1024-dim embedding
    mock_client_instance.embeddings.create.return_value = MockEmbeddingResponse(
        [[0.1] * 1024]
    )

    client = OpenAIEmbeddingClient(api_key="test_key")
    client.embed_texts(["first call"])

    # Second call returns different dimension (should not happen in practice,
    # but proves we cache from the FIRST response)
    mock_client_instance.embeddings.create.return_value = MockEmbeddingResponse(
        [[0.1] * 2048]
    )
    client.embed_texts(["second call"])

    # Still reports the first detected dimension
    assert client.dimensions == 1024


# ---------------------------------------------------------------------------
# fetch_dimensions tests
# ---------------------------------------------------------------------------


@patch("isobase.knowledge.embeddings.openai.OpenAI")
def test_fetch_dimensions(mock_openai_class):
    """fetch_dimensions() returns dimension count after a single probe request."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MockEmbeddingResponse([[0.1] * 1024])
    mock_client.embeddings.create.return_value = mock_response

    client = OpenAIEmbeddingClient(api_key="test_key")
    dims = client.fetch_dimensions()

    assert dims == 1024
    assert client.dimensions == 1024  # cached


@patch("isobase.knowledge.embeddings.openai.OpenAI")
def test_fetch_dimensions_uses_cache(mock_openai_class):
    """Subsequent fetch_dimensions() calls reuse the cached value."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_client.embeddings.create.return_value = MockEmbeddingResponse([[0.1] * 512])

    client = OpenAIEmbeddingClient(api_key="test_key")
    assert client.fetch_dimensions() == 512
    # Second call should not hit the API again
    assert client.fetch_dimensions() == 512
    assert mock_client.embeddings.create.call_count == 1


def test_fetch_dimensions_explicit():
    """When dimensions are set explicitly, fetch_dimensions() returns immediately."""
    client = OpenAIEmbeddingClient(api_key="test_key", dimensions=2048)
    assert client.fetch_dimensions() == 2048


@patch("isobase.knowledge.embeddings.openai.OpenAI")
def test_fetch_dimensions_api_error(mock_openai_class):
    """fetch_dimensions() raises on API failure."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.embeddings.create.side_effect = Exception("Service down")

    client = OpenAIEmbeddingClient(api_key="test_key")
    with pytest.raises(RuntimeError, match="Failed to fetch dimensions"):
        client.fetch_dimensions()


# ---------------------------------------------------------------------------
# error handling tests
# ---------------------------------------------------------------------------


@patch("isobase.knowledge.embeddings.openai.OpenAI")
def test_embed_texts_with_api_error(mock_openai_class):
    """Test handling of API errors."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.embeddings.create.side_effect = Exception("API Error")

    client = OpenAIEmbeddingClient(api_key="test_key")

    with pytest.raises(RuntimeError, match="Embedding API call failed"):
        client.embed_texts(["test"])


@patch("isobase.knowledge.embeddings.openai.OpenAI")
def test_default_kwargs(mock_openai_class):
    """Default kwargs are passed to every API call."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.embeddings.create.return_value = MockEmbeddingResponse([[0.1, 0.2]])

    client = OpenAIEmbeddingClient(
        api_key="test_key", encoding_format="float", custom_param="value"
    )

    client.embed_texts(["test"])

    call_kwargs = mock_client.embeddings.create.call_args[1]
    assert call_kwargs["encoding_format"] == "float"
    assert call_kwargs["custom_param"] == "value"
