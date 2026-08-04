#! python3
# -*- encoding: utf-8 -*-
"""Tests for provider-specific reranker implementations.

@File   :   test_reranker_providers.py
@Created:   2026/08/04 22:44 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import pytest
from unittest.mock import MagicMock, patch

from isobase.knowledge.entities import KnowledgeChunk, RetrievalResult
from isobase.knowledge.retrieval.providers import OpenAIReranker


def test_openai_reranker_initialization():
    """Test reranker initialization."""
    reranker = OpenAIReranker(
        api_key="test_key",
        base_url="https://test.com/v1",
        model="qwen3-rerank",
    )
    assert reranker.model == "qwen3-rerank"
    assert reranker.endpoint == "/reranks"


def test_openai_reranker_custom_endpoint():
    """Endpoint can be customized per provider."""
    reranker = OpenAIReranker(
        api_key="test_key",
        base_url="https://test.com/v1",
        endpoint="/rerank",
    )
    assert reranker.endpoint == "/rerank"


def test_openai_reranker_empty_api_key():
    """Test reranker initialization with empty API key."""
    with pytest.raises(ValueError, match="api_key cannot be empty"):
        OpenAIReranker(api_key="", base_url="https://test.com/v1")


@patch("isobase.knowledge.retrieval.providers.openai_reranker.OpenAI")
def test_rerank_calls_endpoint_and_remaps(mock_openai_class):
    """Full rerank: endpoint called, results remapped with score_source='rerank'."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Simulate API response with reordered results
    mock_item_0 = MagicMock()
    mock_item_0.index = 2
    mock_item_0.relevance_score = 0.95

    mock_item_1 = MagicMock()
    mock_item_1.index = 0
    mock_item_1.relevance_score = 0.88

    mock_response = MagicMock()
    mock_response.results = [mock_item_0, mock_item_1]
    mock_client.post.return_value = mock_response

    reranker = OpenAIReranker(
        api_key="test_key",
        base_url="https://test.com/v1",
    )

    # Build candidates
    chunks = [
        KnowledgeChunk(id="c0", document_id="d1", knowledge_base_id="kb1", content="aaaa", index=0),
        KnowledgeChunk(id="c1", document_id="d1", knowledge_base_id="kb1", content="bbbb", index=1),
        KnowledgeChunk(id="c2", document_id="d1", knowledge_base_id="kb1", content="cccc", index=2),
    ]
    results = [
        RetrievalResult(chunk=chunks[0], score=0.9, score_source="dense"),
        RetrievalResult(chunk=chunks[1], score=0.5, score_source="dense"),
        RetrievalResult(chunk=chunks[2], score=0.3, score_source="dense"),
    ]

    out = reranker.rerank("test query", results, top_k=2)

    assert len(out) == 2
    # First result should be chunk c2 (index 2, score 0.95)
    assert out[0].chunk.id == "c2"
    assert out[0].score == 0.95
    assert out[0].score_source == "rerank"
    # Second result should be chunk c0 (index 0, score 0.88)
    assert out[1].chunk.id == "c0"
    assert out[1].score == 0.88
    assert out[1].score_source == "rerank"

    # Verify endpoint was called correctly
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "/reranks"
    body = call_args[1]["body"]
    assert body["model"] == "qwen3-rerank"
    assert body["query"] == "test query"
    assert body["documents"] == ["aaaa", "bbbb", "cccc"]
    assert body["top_n"] == 2


@patch("isobase.knowledge.retrieval.providers.openai_reranker.OpenAI")
def test_rerank_custom_endpoint_used(mock_openai_class):
    """Custom endpoint path is used in the API call."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.results = []
    mock_client.post.return_value = mock_response

    reranker = OpenAIReranker(
        api_key="test_key",
        base_url="https://test.com/v1",
        endpoint="/rerank",
    )

    chunk = KnowledgeChunk(id="c0", document_id="d1", knowledge_base_id="kb1", content="text", index=0)
    results = [RetrievalResult(chunk=chunk, score=0.9, score_source="dense")]

    reranker.rerank("query", results, top_k=1)

    assert mock_client.post.call_args[0][0] == "/rerank"


@patch("isobase.knowledge.retrieval.providers.openai_reranker.OpenAI")
def test_rerank_empty_results(mock_openai_class):
    """Empty results list returns empty."""
    reranker = OpenAIReranker(
        api_key="test_key",
        base_url="https://test.com/v1",
    )
    out = reranker.rerank("query", [], top_k=5)
    assert out == []


@patch("isobase.knowledge.retrieval.providers.openai_reranker.OpenAI")
def test_rerank_null_query(mock_openai_class):
    """Null query sends empty string."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.results = []
    mock_client.post.return_value = mock_response

    reranker = OpenAIReranker(
        api_key="test_key",
        base_url="https://test.com/v1",
    )

    chunk = KnowledgeChunk(id="c0", document_id="d1", knowledge_base_id="kb1", content="text", index=0)
    results = [RetrievalResult(chunk=chunk, score=0.9, score_source="dense")]

    reranker.rerank(None, results, top_k=1)

    body = mock_client.post.call_args[1]["body"]
    assert body["query"] == ""
