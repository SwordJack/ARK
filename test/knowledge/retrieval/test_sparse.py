#! python3
# -*- encoding: utf-8 -*-
"""Tests for sparse text retrieval (BM25).

@File   :   test_sparse.py
@Created:   2026/08/04 23:36 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import pytest

from isobase.knowledge.entities import KnowledgeChunk, KnowledgeDocument
from isobase.knowledge.retrieval import (
    SparseRetrievalItem,
    SparseRetriever,
    tokenize_text,
)


def test_tokenize_text():
    """Tokenize lowercases and extracts word tokens."""
    assert tokenize_text("Hello World") == ["hello", "world"]
    assert tokenize_text("") == []
    assert tokenize_text("C++ is fun!") == ["c", "is", "fun"]


def test_sparse_retriever_empty_query():
    """Empty query returns empty."""
    retriever = SparseRetriever()
    chunk = KnowledgeChunk(
        id="c1", document_id="d1", knowledge_base_id="kb1", content="hello world", index=0,
    )
    items = [SparseRetrievalItem(chunk=chunk)]
    results = retriever.retrieve("", items)
    assert results == []


def test_sparse_retriever_empty_items():
    """No items returns empty."""
    retriever = SparseRetriever()
    results = retriever.retrieve("hello", [])
    assert results == []


def test_sparse_retriever_basic_ranking():
    """Chunks with matching terms score higher."""
    retriever = SparseRetriever()
    chunk1 = KnowledgeChunk(
        id="c1", document_id="d1", knowledge_base_id="kb1", content="machine learning basics", index=0,
    )
    chunk2 = KnowledgeChunk(
        id="c2", document_id="d1", knowledge_base_id="kb1", content="deep learning and neural networks", index=1,
    )
    chunk3 = KnowledgeChunk(
        id="c3", document_id="d1", knowledge_base_id="kb1", content="cooking recipes for dinner", index=2,
    )
    items = [
        SparseRetrievalItem(chunk=chunk1),
        SparseRetrievalItem(chunk=chunk2),
        SparseRetrievalItem(chunk=chunk3),
    ]

    results = retriever.retrieve("deep learning", items, top_k=3)

    # chunk2 should be first (exact match on "deep learning")
    # chunk1 second (matches "learning")
    # chunk3 last or absent (no match)
    assert len(results) >= 2
    assert results[0].chunk.id == "c2"
    assert results[0].score_source == "sparse"
    # Sorted by descending score
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_sparse_retriever_top_k_trims():
    """top_k limits the number of results."""
    retriever = SparseRetriever()
    chunks = [
        KnowledgeChunk(
            id=f"c{i}", document_id="d1", knowledge_base_id="kb1",
            content=f"python code example {i}", index=i,
        )
        for i in range(10)
    ]
    items = [SparseRetrievalItem(chunk=c) for c in chunks]

    results = retriever.retrieve("python code", items, top_k=3)
    assert len(results) == 3


def test_sparse_retriever_no_match_returns_empty():
    """When no chunk matches the query, returns empty."""
    retriever = SparseRetriever()
    chunk = KnowledgeChunk(
        id="c1", document_id="d1", knowledge_base_id="kb1", content="hello world", index=0,
    )
    items = [SparseRetrievalItem(chunk=chunk)]

    results = retriever.retrieve("zzzzz nonexistent", items)
    assert results == []


def test_sparse_retriever_document_provenance():
    """Document is attached to results when provided."""
    retriever = SparseRetriever()
    doc = KnowledgeDocument(id="d1", knowledge_base_id="kb1", title="Test Doc")
    chunk = KnowledgeChunk(
        id="c1", document_id="d1", knowledge_base_id="kb1", content="hello world", index=0,
    )
    items = [SparseRetrievalItem(chunk=chunk, document=doc)]

    results = retriever.retrieve("hello", items)
    assert len(results) == 1
    assert results[0].document is doc
    assert results[0].document.title == "Test Doc"


def test_sparse_retriever_term_frequency_matters():
    """A chunk with more occurrences of a term scores higher."""
    retriever = SparseRetriever()
    chunk1 = KnowledgeChunk(
        id="c1", document_id="d1", knowledge_base_id="kb1",
        content="python", index=0,
    )
    chunk2 = KnowledgeChunk(
        id="c2", document_id="d1", knowledge_base_id="kb1",
        content="python python python python python", index=1,
    )
    items = [
        SparseRetrievalItem(chunk=chunk1),
        SparseRetrievalItem(chunk=chunk2),
    ]

    results = retriever.retrieve("python", items, top_k=2)
    assert results[0].chunk.id == "c2"
    assert results[0].score > results[1].score


def test_sparse_retriever_idf_matters():
    """Common term contributes less score than rare term."""
    retriever = SparseRetriever()
    common_chunk = KnowledgeChunk(
        id="common", document_id="d1", knowledge_base_id="kb1",
        content="the cat sat on the mat", index=0,
    )

    # All chunks share "the", so its IDF is low.
    chunks = [common_chunk] + [
        KnowledgeChunk(
            id=f"c{i}", document_id="d1", knowledge_base_id="kb1",
            content=f"the cat sat on the rare_term_{i}", index=i + 1,
        )
        for i in range(5)
    ]

    items = [SparseRetrievalItem(chunk=c) for c in chunks]
    results = retriever.retrieve("the rare_term_3", items, top_k=3)

    # "c3" has both "rare_term_3" and "the" — should rank first
    assert results[0].chunk.id == "c3"


def test_sparse_retriever_custom_bm25_params():
    """Custom k1 and b parameters are applied."""
    retriever_default = SparseRetriever(k1=1.5, b=0.75)
    retriever_custom = SparseRetriever(k1=0.1, b=0.1)

    # Vary document lengths so normalization differs across chunks
    chunks = [
        KnowledgeChunk(
            id="c0", document_id="d1", knowledge_base_id="kb1",
            content="python", index=0,
        ),
        KnowledgeChunk(
            id="c1", document_id="d1", knowledge_base_id="kb1",
            content="python python python python python python python python python python", index=1,
        ),
        KnowledgeChunk(
            id="c2", document_id="d1", knowledge_base_id="kb1",
            content="code", index=2,
        ),
    ]
    items = [SparseRetrievalItem(chunk=c) for c in chunks]

    results_default = retriever_default.retrieve("python code", items)
    results_custom = retriever_custom.retrieve("python code", items)

    assert len(results_default) == len(results_custom)
    # Scores differ due to different TF saturation and length normalization
    assert results_default[0].score != results_custom[0].score
