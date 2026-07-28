#! python3
# -*- encoding: utf-8 -*-
"""In-memory knowledge store implementation.

@File   :   memory.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import uuid
from typing import Dict, List
from datetime import datetime, timezone

from ..entities import (
    KnowledgeBase,
    KnowledgeDocument,
    KnowledgeChunk,
    RetrievalResult
)
from .base import BaseKnowledgeStore


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Computes cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score in range [-1, 1].
        Returns 0.0 if either vector has zero magnitude.

    Raises:
        ValueError: If vectors have different lengths.

    Example:
        >>> cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        1.0
        >>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
        0.0
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vectors must have the same length: {len(a)} != {len(b)}"
        )

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class MemoryKnowledgeStore(BaseKnowledgeStore):
    """In-memory knowledge store for testing and demos.

    All data is stored in Python dictionaries and lost on restart.
    Not suitable for production use. Use SQL or dedicated vector
    database implementations for persistence.

    Thread-safety: This implementation is NOT thread-safe. Use locks
    if accessing from multiple threads.
    """

    def __init__(self):
        """Initializes an empty in-memory store."""
        self.knowledge_bases: Dict[str, KnowledgeBase] = {}
        self.documents: Dict[str, KnowledgeDocument] = {}
        self.chunks: Dict[str, KnowledgeChunk] = {}
        self.embeddings: Dict[str, List[float]] = {}  # chunk_id -> vector

    def create_knowledge_base(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Creates a new knowledge base.

        Args:
            kb: KnowledgeBase object. If id is empty, generates a UUID.

        Returns:
            The created knowledge base with timestamps.

        Raises:
            ValueError: If a knowledge base with this id already exists.
        """
        if not kb.id:
            kb.id = str(uuid.uuid4())

        if kb.id in self.knowledge_bases:
            raise ValueError(f"Knowledge base {kb.id} already exists")

        now = datetime.now(timezone.utc)
        kb.created_at = now
        kb.updated_at = now

        self.knowledge_bases[kb.id] = kb
        return kb

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBase:
        """Retrieves a knowledge base by ID.

        Args:
            kb_id: Knowledge base identifier.

        Returns:
            The knowledge base object.

        Raises:
            KeyError: If knowledge base not found.
        """
        if kb_id not in self.knowledge_bases:
            raise KeyError(f"Knowledge base {kb_id} not found")
        return self.knowledge_bases[kb_id]

    def add_document(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        """Stores a document.

        Args:
            doc: KnowledgeDocument object. If id is empty, generates a UUID.

        Returns:
            The stored document with timestamps.

        Raises:
            KeyError: If the parent knowledge base does not exist.
        """
        # Verify parent knowledge base exists
        if doc.knowledge_base_id not in self.knowledge_bases:
            raise KeyError(
                f"Knowledge base {doc.knowledge_base_id} not found"
            )

        if not doc.id:
            doc.id = str(uuid.uuid4())

        doc.created_at = datetime.now(timezone.utc)
        self.documents[doc.id] = doc
        return doc

    def add_chunks(
        self,
        chunks: List[KnowledgeChunk],
        embeddings: List[List[float]]
    ) -> None:
        """Stores chunks and their embeddings.

        Args:
            chunks: List of chunks to store.
            embeddings: Corresponding embedding vectors.

        Raises:
            ValueError: If chunk count doesn't match embedding count.
            KeyError: If parent knowledge base does not exist.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match "
                f"embedding count ({len(embeddings)})"
            )

        for chunk, embedding in zip(chunks, embeddings):
            # Verify parent knowledge base exists
            if chunk.knowledge_base_id not in self.knowledge_bases:
                raise KeyError(
                    f"Knowledge base {chunk.knowledge_base_id} not found"
                )

            if not chunk.id:
                chunk.id = str(uuid.uuid4())

            self.chunks[chunk.id] = chunk
            self.embeddings[chunk.id] = embedding

    def search(
        self,
        kb_id: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Searches for similar chunks by vector similarity.

        Uses cosine similarity for comparison. Results are sorted
        by similarity score in descending order.

        Args:
            kb_id: Knowledge base ID to search within.
            query_embedding: Query vector.
            top_k: Maximum number of results.

        Returns:
            List of retrieval results with scores and document references.
            Empty list if knowledge base is empty or has no chunks.
        """
        # Filter chunks by knowledge base
        kb_chunks = [
            (chunk_id, chunk)
            for chunk_id, chunk in self.chunks.items()
            if chunk.knowledge_base_id == kb_id
        ]

        if not kb_chunks:
            return []

        # Compute similarities
        results = []
        for chunk_id, chunk in kb_chunks:
            embedding = self.embeddings[chunk_id]
            score = cosine_similarity(query_embedding, embedding)

            # Optionally fetch parent document
            doc = self.documents.get(chunk.document_id)

            results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                document=doc
            ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:top_k]
