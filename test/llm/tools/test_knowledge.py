#! python3
# -*- coding: utf-8 -*-
"""Tests for knowledge-base LLM tool integration.

@File   : test_knowledge.py
@Created: 2026/08/02 00:05 (UTC+08:00)
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

# Here put the import lib.
from typing import List

from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.embeddings import BaseEmbeddingClient
from isobase.knowledge.stores import MemoryKnowledgeStore
from isobase.llm.entities import ToolCall
from isobase.llm.tools import ToolSet
from isobase.llm.tools.knowledge import create_knowledge_search_tool


class FakeEmbeddingClient(BaseEmbeddingClient):
    """Fake embedding client for deterministic tool tests."""

    def __init__(self, dimensions: int = 3):
        """Initializes the fake embedding client."""
        super().__init__(model="fake-embedding", dimensions=dimensions)

    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embeds texts using simple keyword buckets."""
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str, **kwargs) -> List[float]:
        """Embeds a single query."""
        if not query:
            raise ValueError("query cannot be empty")
        return self._embed(query)

    @property
    def dimensions(self) -> int:
        """Returns the fake embedding dimensions."""
        return self._dimensions

    def fetch_dimensions(self) -> int:
        """Returns the fake embedding dimensions."""
        return self._dimensions

    def _embed(self, text: str) -> List[float]:
        lowered = text.lower()
        return [
            1.0 if "rag" in lowered else 0.0,
            1.0 if "vector" in lowered else 0.0,
            1.0 if "chunk" in lowered else 0.0,
        ]


def _build_service() -> KnowledgeBaseService:
    """Builds a knowledge service for tool integration tests."""
    return KnowledgeBaseService(
        embedding_client=FakeEmbeddingClient(),
        store=MemoryKnowledgeStore(),
        chunker=FixedSizeChunker(chunk_size=100, chunk_overlap=10),
    )


def test_create_knowledge_search_tool_metadata():
    """Knowledge search tool exposes expected metadata and schema."""
    service = _build_service()
    kb = service.create_knowledge_base(name="Test KB")

    tool = create_knowledge_search_tool(service, kb.id, top_k=2)

    assert tool.name == "search_knowledge_base"
    assert "Search indexed private knowledge" in tool.description
    assert tool.parameters_schema["required"] == ["query"]
    assert tool.parameters_schema["properties"]["query"]["type"] == "string"
    assert tool.parameters_schema["properties"]["limit"]["type"] == "string"


def test_knowledge_search_tool_executes_via_toolset():
    """ToolSet executes the knowledge search tool and returns context."""
    service = _build_service()
    kb = service.create_knowledge_base(name="Test KB")
    service.index_text(kb.id, "RAG combines retrieval and generation.", "RAG Doc")

    tool = create_knowledge_search_tool(service, kb.id, top_k=1)
    toolset = ToolSet([tool])
    tool_calls = [
        ToolCall(
            id="call-1",
            name="search_knowledge_base",
            arguments='{"query": "RAG"}',
        )
    ]

    results, latest_results = toolset.execute_tool_calls(tool_calls)

    assert latest_results == {"search_knowledge_base": True}
    assert len(results) == 1
    assert "RAG Doc" in results[0]
    assert "RAG combines retrieval and generation." in results[0]


def test_knowledge_search_tool_limit_overrides_default_top_k():
    """Tool call limit overrides the factory default top_k."""
    service = _build_service()
    kb = service.create_knowledge_base(name="Test KB")
    service.index_text(kb.id, "RAG combines retrieval and generation.", "RAG Doc")
    service.index_text(kb.id, "Vector databases store embeddings.", "Vector Doc")

    tool = create_knowledge_search_tool(service, kb.id, top_k=1)
    toolset = ToolSet([tool])
    tool_calls = [
        ToolCall(
            id="call-1",
            name="search_knowledge_base",
            arguments='{"query": "RAG vector", "limit": 2}',
        )
    ]

    results, _ = toolset.execute_tool_calls(tool_calls)

    assert "RAG Doc" in results[0]
    assert "Vector Doc" in results[0]


def test_knowledge_search_tool_returns_no_results_message():
    """Knowledge search tool returns a stable message for empty results."""
    service = _build_service()
    kb = service.create_knowledge_base(name="Empty KB")
    tool = create_knowledge_search_tool(service, kb.id)

    result = tool.execute('{"query": "missing"}')

    assert result == "No relevant knowledge found for this query."


def test_knowledge_search_tool_errors_use_function_tool_wrapper():
    """FunctionTool converts service errors into tool execution strings."""
    service = _build_service()
    kb = service.create_knowledge_base(name="Test KB")
    tool = create_knowledge_search_tool(service, kb.id)

    result = tool.execute('{"query": ""}')

    assert result.startswith("Error executing tool 'search_knowledge_base':")
    assert "query cannot be empty" in result

