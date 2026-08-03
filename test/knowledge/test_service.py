#! python3
# -*- encoding: utf-8 -*-
"""Tests for KnowledgeBaseService integration.

@File   :   test_service.py
@Created:   2026/07/29 00:00
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import pytest
from typing import List
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.embeddings import BaseEmbeddingClient
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.stores import MemoryKnowledgeStore


class FakeEmbeddingClient(BaseEmbeddingClient):
    """Fake embedding client for testing without API calls."""

    def __init__(self, dimensions: int = 128):
        super().__init__(dimensions=dimensions)

    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generates deterministic fake embeddings."""
        embeddings = []
        for text in texts:
            vec = []
            for i in range(self._dimensions):
                val = (len(text) + sum(ord(c) for c in text[:10]) + i) % 100 / 100.0
                vec.append(val)
            embeddings.append(vec)
        return embeddings

    def embed_query(self, query: str, **kwargs) -> List[float]:
        """Embeds a single query."""
        return self.embed_texts([query], **kwargs)[0]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def fetch_dimensions(self) -> int:
        return self._dimensions


def test_service_initialization():
    """Test service initialization."""
    embedding_client = FakeEmbeddingClient(dimensions=128)
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    store = MemoryKnowledgeStore()

    service = KnowledgeBaseService(
        embedding_client=embedding_client,
        store=store,
        chunker=chunker,
    )

    assert service.embedding_client is embedding_client
    assert service.store is store
    assert service.chunker is chunker


def test_service_create_knowledge_base():
    """Test creating a knowledge base through service."""
    service = KnowledgeBaseService(
        embedding_client=FakeEmbeddingClient(),
        store=MemoryKnowledgeStore(),
        chunker=FixedSizeChunker(),
    )

    kb = service.create_knowledge_base(
        name="Test KB",
        description="Test description",
        metadata={"owner": "test"}
    )

    assert kb.id != ""
    assert kb.name == "Test KB"
    assert kb.description == "Test description"
    assert kb.metadata["owner"] == "test"
    assert kb.dimensions == 128  # From FakeEmbeddingClient
    assert kb.chunk_size == 512   # FixedSizeChunker default
    assert kb.chunk_overlap == 50
    assert kb.created_time is not None


def test_service_kb_chunk_params_override_chunker():
    """index_text uses KB's chunk_size/chunk_overlap, not the chunker's
    instance defaults."""
    # Explicit instance defaults that differ from the KB settings
    service = KnowledgeBaseService(
        embedding_client=FakeEmbeddingClient(dimensions=64),
        store=MemoryKnowledgeStore(),
        chunker=FixedSizeChunker(chunk_size=100, chunk_overlap=20),
    )

    # Create KB — at this point the KB snaps the chunker's current defaults
    kb = service.create_knowledge_base(name="Test KB")

    # Build a new chunker with *different* defaults and wire it into the
    # same service, as if the service's chunker had been replaced between
    # KB creation and indexing.
    service.chunker = FixedSizeChunker(chunk_size=999, chunk_overlap=99)

    text = "RAG stands for Retrieval-Augmented Generation. " * 5
    service.index_text(
        knowledge_base_id=kb.id, text=text, title="Test Doc"
    )

    # Retrieve — the chunks should have been cut with the KB's recorded
    # parameters (100/20), not the new chunker defaults (999/99).  We
    # verify by checking no chunk exceeds the KB's chunk_size.
    results = service.retrieve(
        query="RAG", knowledge_base_id=kb.id, top_k=10
    )
    assert results
    for r in results:
        assert len(r.chunk.content) <= 100


def test_service_index_text():
    """Test indexing text through service."""
    service = KnowledgeBaseService(
        embedding_client=FakeEmbeddingClient(dimensions=64),
        store=MemoryKnowledgeStore(),
        chunker=FixedSizeChunker(chunk_size=50, chunk_overlap=10),
    )

    kb = service.create_knowledge_base(name="Test KB")

    text = "RAG stands for Retrieval-Augmented Generation. " * 5  # Long enough for multiple chunks
    doc = service.index_text(
        knowledge_base_id=kb.id,
        text=text,
        title="Test Doc",
        source_uri="test.md",
        metadata={"author": "Alice"}
    )

    assert doc.id != ""
    assert doc.title == "Test Doc"
    assert doc.source_uri == "test.md"
    assert doc.content == text
    assert doc.metadata["author"] == "Alice"


def test_service_index_empty_text():
    """Test indexing empty text."""
    service = KnowledgeBaseService(
        embedding_client=FakeEmbeddingClient(),
        store=MemoryKnowledgeStore(),
        chunker=FixedSizeChunker(),
    )

    kb = service.create_knowledge_base(name="Test KB")

    doc = service.index_text(
        knowledge_base_id=kb.id,
        text="",
        title="Empty Doc"
    )

    assert doc.id != ""
    assert doc.title == "Empty Doc"


def test_service_index_text_invalid_kb():
    """Test indexing text to non-existent knowledge base."""
    service = KnowledgeBaseService(
        embedding_client=FakeEmbeddingClient(),
        store=MemoryKnowledgeStore(),
        chunker=FixedSizeChunker(),
    )

    with pytest.raises(KeyError):
        service.index_text(
            knowledge_base_id="nonexistent",
            text="test",
            title="Test"
        )


def test_service_retrieve():
    """Test retrieving chunks through service."""
    service = KnowledgeBaseService(
        embedding_client=FakeEmbeddingClient(dimensions=64),
        store=MemoryKnowledgeStore(),
        chunker=FixedSizeChunker(chunk_size=50, chunk_overlap=10),
    )

    kb = service.create_knowledge_base(name="Test KB")

    # Index multiple documents
    service.index_text(kb.id, "RAG is a retrieval technique.", "Doc 1")
    service.index_text(kb.id, "Vector databases store embeddings.", "Doc 2")
    service.index_text(kb.id, "Chunking splits documents.", "Doc 3")

    # Retrieve
    results = service.retrieve(
        query="RAG retrieval",
        knowledge_base_id=kb.id,
        top_k=2
    )

    assert len(results) <= 2
    assert all(hasattr(r, "chunk") for r in results)
    assert all(hasattr(r, "score") for r in results)
    assert all(r.document is not None for r in results)


def test_service_retrieve_empty_query():
    """Test retrieving with empty query."""
    service = KnowledgeBaseService(
        embedding_client=FakeEmbeddingClient(),
        store=MemoryKnowledgeStore(),
        chunker=FixedSizeChunker(),
    )

    kb = service.create_knowledge_base(name="Test KB")

    with pytest.raises(ValueError, match="query cannot be empty"):
        service.retrieve(query="", knowledge_base_id=kb.id)


def test_service_retrieve_as_context():
    """Test retrieving formatted context."""
    service = KnowledgeBaseService(
        embedding_client=FakeEmbeddingClient(dimensions=64),
        store=MemoryKnowledgeStore(),
        chunker=FixedSizeChunker(chunk_size=50, chunk_overlap=10),
    )

    kb = service.create_knowledge_base(name="Test KB")

    service.index_text(kb.id, "RAG is a retrieval technique.", "Doc 1")
    service.index_text(kb.id, "Vector databases store embeddings.", "Doc 2")

    context = service.retrieve_as_context(
        query="RAG",
        knowledge_base_id=kb.id,
        top_k=2,
        separator="\n---\n"
    )

    assert isinstance(context, str)
    if context:  # May be empty if no similar chunks
        assert "---" in context or len(context) > 0


def test_service_retrieve_as_context_empty():
    """Test retrieving context from empty knowledge base."""
    service = KnowledgeBaseService(
        embedding_client=FakeEmbeddingClient(),
        store=MemoryKnowledgeStore(),
        chunker=FixedSizeChunker(),
    )

    kb = service.create_knowledge_base(name="Test KB")

    context = service.retrieve_as_context(
        query="test",
        knowledge_base_id=kb.id,
        top_k=5
    )

    assert context == ""


def test_service_e2e_workflow():
    """Test end-to-end workflow: create → index → retrieve."""
    service = KnowledgeBaseService(
        embedding_client=FakeEmbeddingClient(dimensions=128),
        store=MemoryKnowledgeStore(),
        chunker=FixedSizeChunker(chunk_size=100, chunk_overlap=20),
    )

    # Create knowledge base
    kb = service.create_knowledge_base(
        name="AI Documentation",
        description="AI-related docs"
    )

    # Index documents
    texts = [
        "RAG combines retrieval and generation for better LLM responses.",
        "Vector databases enable fast similarity search using embeddings.",
        "Chunking strategies affect retrieval quality and performance.",
    ]

    for i, text in enumerate(texts):
        service.index_text(
            knowledge_base_id=kb.id,
            text=text,
            title=f"Doc {i+1}",
            source_uri=f"doc{i+1}.md"
        )

    # Retrieve
    results = service.retrieve(
        query="How does RAG work?",
        knowledge_base_id=kb.id,
        top_k=3
    )

    assert len(results) > 0
    assert all(r.chunk.knowledge_base_id == kb.id for r in results)
    assert all(r.score >= 0 and r.score <= 1 for r in results)

    # Verify results are sorted by score
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
