#! python3
# -*- encoding: utf-8 -*-
"""Tests for knowledge retrieval strategies.

@File   :   test_retrieval.py
@Created:   2026/08/03 19:59 (UTC+08:00)
@Author :   SwordJack
#Contact:   https://github.com/SwordJack/
"""

import pytest

from isobase.knowledge.entities import KnowledgeChunk, KnowledgeDocument
from isobase.knowledge.retrieval import (
    DenseRetriever,
    DenseRetrievalItem,
    cosine_similarity,
)


def test_cosine_similarity_identical_vectors():
    """Test cosine similarity of identical vectors."""
    vec = [1.0, 2.0, 3.0]
    similarity = cosine_similarity(vec, vec)
    assert abs(similarity - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    """Test cosine similarity of orthogonal vectors."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    similarity = cosine_similarity(vec1, vec2)
    assert abs(similarity - 0.0) < 1e-6


def test_cosine_similarity_opposite_vectors():
    """Test cosine similarity of opposite vectors."""
    vec1 = [1.0, 0.0]
    vec2 = [-1.0, 0.0]
    similarity = cosine_similarity(vec1, vec2)
    assert abs(similarity - (-1.0)) < 1e-6


def test_cosine_similarity_different_lengths():
    """Test cosine similarity with different vector lengths."""
    vec1 = [1.0, 2.0]
    vec2 = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="Vectors must have the same length"):
        cosine_similarity(vec1, vec2)


def test_cosine_similarity_zero_vector():
    """Test cosine similarity with zero vector."""
    vec1 = [0.0, 0.0, 0.0]
    vec2 = [1.0, 2.0, 3.0]
    similarity = cosine_similarity(vec1, vec2)
    assert similarity == 0.0


def test_dense_retriever_retrieve():
    """Test dense retriever ranking and top_k limiting."""
    retriever = DenseRetriever()

    # Build fake items
    chunk1 = KnowledgeChunk(
        id="c1", document_id="d1", knowledge_base_id="kb1", content="chunk 1", index=0
    )
    chunk2 = KnowledgeChunk(
        id="c2", document_id="d1", knowledge_base_id="kb1", content="chunk 2", index=1
    )
    chunk3 = KnowledgeChunk(
        id="c3", document_id="d2", knowledge_base_id="kb1", content="chunk 3", index=0
    )

    doc1 = KnowledgeDocument(id="d1", knowledge_base_id="kb1", title="Doc 1")
    doc2 = KnowledgeDocument(id="d2", knowledge_base_id="kb1", title="Doc 2")

    items = [
        DenseRetrievalItem(chunk=chunk1, embedding=[1.0, 0.0], document=doc1),
        DenseRetrievalItem(chunk=chunk2, embedding=[0.0, 1.0], document=doc1),
        DenseRetrievalItem(chunk=chunk3, embedding=[0.707, 0.707], document=doc2),
    ]

    # Query embedding is aligned with [1.0, 0.0]
    query_embedding = [1.0, 0.0]

    # Retrieve top 2
    results = retriever.retrieve(query_embedding, items, top_k=2)

    assert len(results) == 2

    # First result should be chunk1 (similarity 1.0)
    assert results[0].chunk.id == "c1"
    assert abs(results[0].score - 1.0) < 1e-6
    assert results[0].document is doc1

    # Second result should be chunk3 (similarity ~0.707)
    assert results[1].chunk.id == "c3"
    assert abs(results[1].score - 0.707) < 1e-3
    assert results[1].document is doc2
