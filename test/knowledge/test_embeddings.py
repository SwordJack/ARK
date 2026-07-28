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
from isobase.knowledge.embeddings import OpenAICompatEmbeddingClient


class MockEmbeddingResponse:
    """Mock response from OpenAI embeddings API."""
    def __init__(self, embeddings):
        self.data = [MagicMock(embedding=emb) for emb in embeddings]


def test_openai_compat_client_initialization():
    """Test client initialization."""
    client = OpenAICompatEmbeddingClient(
        api_key="test_key",
        base_url="https://test.com/v1",
        model="test-model",
        dimensions=1024
    )

    assert client.model == "test-model"
    assert client.dimensions == 1024


def test_openai_compat_client_empty_api_key():
    """Test client initialization with empty API key."""
    with pytest.raises(ValueError, match="api_key cannot be empty"):
        OpenAICompatEmbeddingClient(api_key="")


@patch("isobase.knowledge.embeddings.openai_compat.OpenAI")
def test_embed_texts(mock_openai_class):
    """Test embedding multiple texts."""
    # Setup mock
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MockEmbeddingResponse([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ])
    mock_client.embeddings.create.return_value = mock_response

    # Create client and embed
    client = OpenAICompatEmbeddingClient(
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


@patch("isobase.knowledge.embeddings.openai_compat.OpenAI")
def test_embed_texts_empty(mock_openai_class):
    """Test embedding with empty text list."""
    client = OpenAICompatEmbeddingClient(api_key="test_key")

    with pytest.raises(ValueError, match="texts cannot be empty"):
        client.embed_texts([])


@patch("isobase.knowledge.embeddings.openai_compat.OpenAI")
def test_embed_query(mock_openai_class):
    """Test embedding a single query."""
    # Setup mock
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MockEmbeddingResponse([[0.1, 0.2, 0.3]])
    mock_client.embeddings.create.return_value = mock_response

    # Create client and embed
    client = OpenAICompatEmbeddingClient(
        api_key="test_key",
        model="test-model"
    )

    embedding = client.embed_query("test query")

    assert embedding == [0.1, 0.2, 0.3]


@patch("isobase.knowledge.embeddings.openai_compat.OpenAI")
def test_embed_query_empty(mock_openai_class):
    """Test embedding with empty query."""
    client = OpenAICompatEmbeddingClient(api_key="test_key")

    with pytest.raises(ValueError, match="query cannot be empty"):
        client.embed_query("")


def test_dimensions_explicit():
    """Test dimensions property with explicit value."""
    client = OpenAICompatEmbeddingClient(
        api_key="test_key",
        dimensions=2048
    )

    assert client.dimensions == 2048


def test_dimensions_inference():
    """Test dimensions property inference from model name."""
    # OpenAI models
    client1 = OpenAICompatEmbeddingClient(
        api_key="test_key",
        model="text-embedding-3-small"
    )
    assert client1.dimensions == 1536

    client2 = OpenAICompatEmbeddingClient(
        api_key="test_key",
        model="text-embedding-3-large"
    )
    assert client2.dimensions == 3072

    # Qwen models
    client3 = OpenAICompatEmbeddingClient(
        api_key="test_key",
        model="text-embedding-v4"
    )
    assert client3.dimensions == 1024

    client4 = OpenAICompatEmbeddingClient(
        api_key="test_key",
        model="text-embedding-v3"
    )
    assert client4.dimensions == 1024

    # Unknown model (default)
    client5 = OpenAICompatEmbeddingClient(
        api_key="test_key",
        model="unknown-model"
    )
    assert client5.dimensions == 1536


@patch("isobase.knowledge.embeddings.openai_compat.OpenAI")
def test_embed_texts_with_api_error(mock_openai_class):
    """Test handling of API errors."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Simulate API error
    mock_client.embeddings.create.side_effect = Exception("API Error")

    client = OpenAICompatEmbeddingClient(api_key="test_key")

    with pytest.raises(RuntimeError, match="Embedding API call failed"):
        client.embed_texts(["test"])


@patch("isobase.knowledge.embeddings.openai_compat.OpenAI")
def test_default_kwargs(mock_openai_class):
    """Test default kwargs are passed to API."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MockEmbeddingResponse([[0.1, 0.2]])
    mock_client.embeddings.create.return_value = mock_response

    client = OpenAICompatEmbeddingClient(
        api_key="test_key",
        encoding_format="float",
        custom_param="value"
    )

    client.embed_texts(["test"])

    call_kwargs = mock_client.embeddings.create.call_args[1]
    assert call_kwargs["encoding_format"] == "float"
    assert call_kwargs["custom_param"] == "value"
