#! python3
# -*- encoding: utf-8 -*-
"""Manual live smoke test for the knowledge base module.

This script is intentionally **not** collected by pytest (filename does not
start with ``test_``) and demonstrates end-to-end knowledge base functionality
with fake embeddings (no API calls required).

Usage:
    Run from the repo root:
        python -m test.knowledge.live.run_knowledge_basic

This example:
    1. Creates a knowledge base
    2. Indexes multiple documents with chunking
    3. Performs similarity search
    4. Demonstrates formatted context retrieval for LLM integration

@File   :   run_knowledge_basic.py
@Created:   2026/07/29 00:28
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from typing import List
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.embeddings import BaseEmbeddingClient
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.stores import MemoryKnowledgeStore


class FakeEmbeddingClient(BaseEmbeddingClient):
    """Fake embedding client for testing without API calls.

    Generates deterministic embeddings based on text length and content.
    NOT suitable for real similarity search, only for demonstrations.
    """

    def __init__(self, dimensions: int = 128):
        self._dimensions = dimensions

    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generates fake embeddings based on text characteristics."""
        embeddings = []
        for text in texts:
            # Simple hash-based fake embedding
            vec = []
            for i in range(self._dimensions):
                # Use text length and character codes for deterministic values
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


def main():
    print("=" * 60)
    print("IsoBase Knowledge Base - Basic Example")
    print("=" * 60)
    print()

    # Initialize components
    print("Initializing knowledge base service...")
    embedding_client = FakeEmbeddingClient(dimensions=128)
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=30)
    store = MemoryKnowledgeStore()

    service = KnowledgeBaseService(
        embedding_client=embedding_client,
        store=store,
        chunker=chunker,
    )
    print("✓ Service initialized")
    print()

    # Create knowledge base
    print("Creating knowledge base...")
    kb = service.create_knowledge_base(
        name="AI Documentation",
        description="Documentation about AI concepts",
        metadata={"owner": "eng-team", "version": "1.0"}
    )
    print(f"✓ Knowledge base created: {kb.name} (ID: {kb.id[:8]}...)")
    print()

    # Index documents
    print("Indexing documents...")

    doc1 = service.index_text(
        knowledge_base_id=kb.id,
        text=(
            "RAG stands for Retrieval-Augmented Generation. It is a technique that combines "
            "information retrieval with text generation. RAG systems first retrieve relevant "
            "documents from a knowledge base, then use those documents as context for generating "
            "responses. This approach helps LLMs provide more accurate and up-to-date information."
        ),
        title="RAG Overview",
        source_uri="docs/rag.md",
        metadata={"author": "Alice", "date": "2024-01-15"}
    )
    print(f"✓ Indexed: {doc1.title}")

    doc2 = service.index_text(
        knowledge_base_id=kb.id,
        text=(
            "Vector databases are specialized storage systems designed for high-dimensional "
            "embeddings. They enable fast similarity search using techniques like approximate "
            "nearest neighbor (ANN) search. Popular vector databases include FAISS, Pinecone, "
            "Qdrant, and Milvus. These systems are essential for RAG applications."
        ),
        title="Vector Databases",
        source_uri="docs/vector-db.md",
        metadata={"author": "Bob", "date": "2024-01-16"}
    )
    print(f"✓ Indexed: {doc2.title}")

    doc3 = service.index_text(
        knowledge_base_id=kb.id,
        text=(
            "Embeddings are dense vector representations of text that capture semantic meaning. "
            "They are created by neural networks trained to place semantically similar texts "
            "close together in vector space. Common embedding models include OpenAI's "
            "text-embedding-3, Alibaba's Qwen embedding models, and open-source models like "
            "sentence-transformers."
        ),
        title="Text Embeddings",
        source_uri="docs/embeddings.md",
        metadata={"author": "Alice", "date": "2024-01-17"}
    )
    print(f"✓ Indexed: {doc3.title}")
    print()

    # Retrieve relevant chunks
    print("Searching knowledge base...")
    print()

    queries = [
        "What is RAG?",
        "Tell me about vector databases",
        "How do embeddings work?",
    ]

    for query in queries:
        print(f"Query: \"{query}\"")
        print("-" * 60)

        results = service.retrieve(
            query=query,
            knowledge_base_id=kb.id,
            top_k=2,
        )

        if not results:
            print("No relevant results found.")
        else:
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  Score: {result.score:.4f}")
                print(f"  Source: {result.document.title}")
                print(f"  Author: {result.document.metadata.get('author', 'Unknown')}")
                print(f"  Content: {result.chunk.content[:150]}...")

        print()
        print()

    # Demonstrate formatted context retrieval
    print("=" * 60)
    print("Formatted Context for LLM")
    print("=" * 60)
    print()

    query = "Explain RAG systems"
    context = service.retrieve_as_context(
        query=query,
        knowledge_base_id=kb.id,
        top_k=2,
    )

    print(f"Query: \"{query}\"")
    print()
    print("Retrieved Context:")
    print(context)
    print()

    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
