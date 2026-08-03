#! python3
# -*- encoding: utf-8 -*-
"""Tests for knowledge store backends.

@File   :   test_memory.py
@Created:   2026/07/29 00:00 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import pytest
from isobase.knowledge.entities import (
    KnowledgeBase,
    KnowledgeDocument,
    KnowledgeChunk,
)
from isobase.knowledge.stores import MemoryKnowledgeStore


def test_memory_store_create_knowledge_base():
    """Test creating a knowledge base."""
    store = MemoryKnowledgeStore()

    kb = KnowledgeBase(
        id="kb1",
        name="Test KB",
        description="Test description"
    )

    created_kb = store.create_knowledge_base(kb)

    assert created_kb.id == "kb1"
    assert created_kb.name == "Test KB"
    assert created_kb.created_time is not None
    assert created_kb.updated_time is not None


def test_memory_store_create_knowledge_base_auto_id():
    """Test creating a knowledge base with auto-generated ID."""
    store = MemoryKnowledgeStore()

    kb = KnowledgeBase(id="", name="Test KB")
    created_kb = store.create_knowledge_base(kb)

    assert created_kb.id != ""
    assert len(created_kb.id) > 0


def test_memory_store_create_knowledge_base_duplicate():
    """Test creating duplicate knowledge base raises error."""
    store = MemoryKnowledgeStore()

    kb1 = KnowledgeBase(id="kb1", name="Test KB 1")
    store.create_knowledge_base(kb1)

    kb2 = KnowledgeBase(id="kb1", name="Test KB 2")
    with pytest.raises(ValueError, match="already exists"):
        store.create_knowledge_base(kb2)


def test_memory_store_get_knowledge_base():
    """Test retrieving a knowledge base."""
    store = MemoryKnowledgeStore()

    kb = KnowledgeBase(id="kb1", name="Test KB")
    store.create_knowledge_base(kb)

    retrieved_kb = store.get_knowledge_base("kb1")

    assert retrieved_kb.id == "kb1"
    assert retrieved_kb.name == "Test KB"


def test_memory_store_get_knowledge_base_not_found():
    """Test retrieving non-existent knowledge base."""
    store = MemoryKnowledgeStore()

    with pytest.raises(KeyError, match="not found"):
        store.get_knowledge_base("nonexistent")


def test_memory_store_add_document():
    """Test adding a document."""
    store = MemoryKnowledgeStore()

    kb = KnowledgeBase(id="kb1", name="Test KB")
    store.create_knowledge_base(kb)

    doc = KnowledgeDocument(
        id="doc1",
        knowledge_base_id="kb1",
        title="Test Doc",
        content="Test content"
    )

    added_doc = store.add_document(doc)

    assert added_doc.id == "doc1"
    assert added_doc.title == "Test Doc"
    assert added_doc.created_time is not None


def test_memory_store_add_document_auto_id():
    """Test adding a document with auto-generated ID."""
    store = MemoryKnowledgeStore()

    kb = KnowledgeBase(id="kb1", name="Test KB")
    store.create_knowledge_base(kb)

    doc = KnowledgeDocument(
        id="",
        knowledge_base_id="kb1",
        title="Test Doc"
    )

    added_doc = store.add_document(doc)

    assert added_doc.id != ""
    assert len(added_doc.id) > 0


def test_memory_store_add_document_invalid_kb():
    """Test adding document to non-existent knowledge base."""
    store = MemoryKnowledgeStore()

    doc = KnowledgeDocument(
        id="doc1",
        knowledge_base_id="nonexistent",
        title="Test Doc"
    )

    with pytest.raises(KeyError, match="not found"):
        store.add_document(doc)


def test_memory_store_add_chunks():
    """Test adding chunks with embeddings."""
    store = MemoryKnowledgeStore()

    kb = KnowledgeBase(id="kb1", name="Test KB")
    store.create_knowledge_base(kb)

    chunks = [
        KnowledgeChunk(
            id="c1",
            document_id="doc1",
            knowledge_base_id="kb1",
            content="chunk 1",
            index=0
        ),
        KnowledgeChunk(
            id="c2",
            document_id="doc1",
            knowledge_base_id="kb1",
            content="chunk 2",
            index=1
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_chunks(chunks, embeddings)

    assert "c1" in store.chunks
    assert "c2" in store.chunks
    assert store.embeddings["c1"] == [1.0, 0.0, 0.0]
    assert store.embeddings["c2"] == [0.0, 1.0, 0.0]


def test_memory_store_add_chunks_mismatch():
    """Test adding chunks with mismatched embedding count."""
    store = MemoryKnowledgeStore()

    kb = KnowledgeBase(id="kb1", name="Test KB")
    store.create_knowledge_base(kb)

    chunks = [
        KnowledgeChunk(
            id="c1",
            document_id="doc1",
            knowledge_base_id="kb1",
            content="chunk 1",
            index=0
        ),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],  # Extra embedding
    ]

    with pytest.raises(ValueError, match="must match"):
        store.add_chunks(chunks, embeddings)


def test_memory_store_add_chunks_invalid_kb():
    """Test adding chunks to non-existent knowledge base."""
    store = MemoryKnowledgeStore()

    chunks = [
        KnowledgeChunk(
            id="c1",
            document_id="doc1",
            knowledge_base_id="nonexistent",
            content="chunk 1",
            index=0
        ),
    ]

    embeddings = [[1.0, 0.0]]

    with pytest.raises(KeyError, match="not found"):
        store.add_chunks(chunks, embeddings)


def test_memory_store_search():
    """Test searching for similar chunks."""
    store = MemoryKnowledgeStore()

    kb = KnowledgeBase(id="kb1", name="Test KB")
    store.create_knowledge_base(kb)

    doc = KnowledgeDocument(
        id="doc1",
        knowledge_base_id="kb1",
        title="Test Doc"
    )
    store.add_document(doc)

    chunks = [
        KnowledgeChunk(
            id="c1",
            document_id="doc1",
            knowledge_base_id="kb1",
            content="chunk 1",
            index=0
        ),
        KnowledgeChunk(
            id="c2",
            document_id="doc1",
            knowledge_base_id="kb1",
            content="chunk 2",
            index=1
        ),
        KnowledgeChunk(
            id="c3",
            document_id="doc1",
            knowledge_base_id="kb1",
            content="chunk 3",
            index=2
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    store.add_chunks(chunks, embeddings)

    # Search for chunk similar to c1
    query_embedding = [0.9, 0.1, 0.0]
    results = store.search("kb1", query_embedding, top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == "c1"  # Most similar
    assert results[0].score > results[1].score  # Sorted by score
    assert results[0].document.title == "Test Doc"


def test_memory_store_search_empty_kb():
    """Test searching in empty knowledge base."""
    store = MemoryKnowledgeStore()

    kb = KnowledgeBase(id="kb1", name="Test KB")
    store.create_knowledge_base(kb)

    query_embedding = [1.0, 0.0, 0.0]
    results = store.search("kb1", query_embedding, top_k=5)

    assert results == []


def test_memory_store_search_nonexistent_kb():
    """Test searching in non-existent knowledge base."""
    store = MemoryKnowledgeStore()

    query_embedding = [1.0, 0.0, 0.0]
    results = store.search("nonexistent", query_embedding, top_k=5)

    assert results == []


def test_memory_store_search_top_k():
    """Test top_k limiting of search results."""
    store = MemoryKnowledgeStore()

    kb = KnowledgeBase(id="kb1", name="Test KB")
    store.create_knowledge_base(kb)

    # Add 10 chunks
    chunks = [
        KnowledgeChunk(
            id=f"c{i}",
            document_id="doc1",
            knowledge_base_id="kb1",
            content=f"chunk {i}",
            index=i
        )
        for i in range(10)
    ]

    embeddings = [[float(i), 0.0] for i in range(10)]

    store.add_chunks(chunks, embeddings)

    # Search with top_k=3
    query_embedding = [5.0, 0.0]
    results = store.search("kb1", query_embedding, top_k=3)

    assert len(results) == 3


# ---------------------------------------------------------------------
# Document lifecycle
# ---------------------------------------------------------------------

def test_memory_store_get_document():
    """Test retrieving a document by ID."""
    store = MemoryKnowledgeStore()

    kb = KnowledgeBase(id="kb1", name="Test KB")
    store.create_knowledge_base(kb)

    doc = KnowledgeDocument(id="doc1", knowledge_base_id="kb1", title="Test Doc")
    store.add_document(doc)

    retrieved = store.get_document("doc1")
    assert retrieved.id == "doc1"
    assert retrieved.title == "Test Doc"


def test_memory_store_get_document_not_found():
    """Test retrieving non-existent document raises KeyError."""
    store = MemoryKnowledgeStore()

    with pytest.raises(KeyError, match="not found"):
        store.get_document("nonexistent")


def test_memory_store_delete_document():
    """Test deleting a document removes it and its chunks/embeddings."""
    store = MemoryKnowledgeStore()

    store.create_knowledge_base(KnowledgeBase(id="kb1", name="Test KB"))
    store.add_document(KnowledgeDocument(id="doc1", knowledge_base_id="kb1", title="Test Doc"))

    chunks = [
        KnowledgeChunk(id="c1", document_id="doc1", knowledge_base_id="kb1", content="chunk 1", index=0),
        KnowledgeChunk(id="c2", document_id="doc1", knowledge_base_id="kb1", content="chunk 2", index=1),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    store.add_chunks(chunks, embeddings)

    store.delete_document("doc1")

    # Document is gone
    with pytest.raises(KeyError):
        store.get_document("doc1")

    # Chunks and embeddings are gone
    assert "c1" not in store.chunks
    assert "c2" not in store.chunks
    assert "c1" not in store.embeddings
    assert "c2" not in store.embeddings

    # Search returns nothing
    results = store.search("kb1", [1.0, 0.0])
    assert results == []


def test_memory_store_delete_document_not_found():
    """Test deleting non-existent document raises KeyError."""
    store = MemoryKnowledgeStore()

    with pytest.raises(KeyError, match="not found"):
        store.delete_document("nonexistent")


def test_memory_store_delete_document_leaves_other_docs():
    """Test deleting one document does not affect another in the same KB."""
    store = MemoryKnowledgeStore()

    store.create_knowledge_base(KnowledgeBase(id="kb1", name="Test KB"))

    store.add_document(KnowledgeDocument(id="doc1", knowledge_base_id="kb1", title="Keep"))
    store.add_chunks(
        [KnowledgeChunk(id="c1", document_id="doc1", knowledge_base_id="kb1", content="kept", index=0)],
        [[1.0, 0.0]],
    )

    store.add_document(KnowledgeDocument(id="doc2", knowledge_base_id="kb1", title="Remove"))
    store.add_chunks(
        [KnowledgeChunk(id="c2", document_id="doc2", knowledge_base_id="kb1", content="removed", index=0)],
        [[0.0, 1.0]],
    )

    store.delete_document("doc2")

    # doc1 still accessible
    retrieved = store.get_document("doc1")
    assert retrieved.title == "Keep"

    # c1 still searchable
    results = store.search("kb1", [1.0, 0.0])
    assert len(results) == 1
    assert results[0].chunk.content == "kept"


# ---------------------------------------------------------------------
# Knowledge-base lifecycle
# ---------------------------------------------------------------------

def test_memory_store_delete_knowledge_base():
    """Test deleting a knowledge base removes everything."""
    store = MemoryKnowledgeStore()

    store.create_knowledge_base(KnowledgeBase(id="kb1", name="Test KB"))
    store.add_document(KnowledgeDocument(id="doc1", knowledge_base_id="kb1", title="Test Doc"))

    chunks = [
        KnowledgeChunk(id="c1", document_id="doc1", knowledge_base_id="kb1", content="chunk 1", index=0),
    ]
    embeddings = [[1.0, 0.0]]
    store.add_chunks(chunks, embeddings)

    store.delete_knowledge_base("kb1")

    # KB is gone
    with pytest.raises(KeyError):
        store.get_knowledge_base("kb1")

    # Document is gone
    with pytest.raises(KeyError):
        store.get_document("doc1")

    # Chunks and embeddings are gone
    assert "c1" not in store.chunks
    assert "c1" not in store.embeddings


def test_memory_store_delete_knowledge_base_not_found():
    """Test deleting non-existent knowledge base raises KeyError."""
    store = MemoryKnowledgeStore()

    with pytest.raises(KeyError, match="not found"):
        store.delete_knowledge_base("nonexistent")


# ---------------------------------------------------------------------
# List methods
# ---------------------------------------------------------------------

def test_memory_store_list_knowledge_bases():
    """Test listing all knowledge bases."""
    store = MemoryKnowledgeStore()

    store.create_knowledge_base(KnowledgeBase(id="kb1", name="First"))
    store.create_knowledge_base(KnowledgeBase(id="kb2", name="Second"))

    kbs = store.list_knowledge_bases()
    names = {kb.name for kb in kbs}

    assert len(kbs) == 2
    assert names == {"First", "Second"}


def test_memory_store_list_knowledge_bases_empty():
    """Test listing knowledge bases when none exist."""
    store = MemoryKnowledgeStore()
    assert store.list_knowledge_bases() == []


def test_memory_store_list_documents():
    """Test listing documents in a knowledge base."""
    store = MemoryKnowledgeStore()

    store.create_knowledge_base(KnowledgeBase(id="kb1", name="Test KB"))
    store.add_document(KnowledgeDocument(id="doc1", knowledge_base_id="kb1", title="Doc A"))
    store.add_document(KnowledgeDocument(id="doc2", knowledge_base_id="kb1", title="Doc B"))

    docs = store.list_documents("kb1")
    titles = {doc.title for doc in docs}

    assert len(docs) == 2
    assert titles == {"Doc A", "Doc B"}


def test_memory_store_list_documents_empty():
    """Test listing documents in a knowledge base with no documents."""
    store = MemoryKnowledgeStore()
    store.create_knowledge_base(KnowledgeBase(id="kb1", name="Test KB"))

    assert store.list_documents("kb1") == []


def test_memory_store_list_documents_invalid_kb():
    """Test listing documents for non-existent knowledge base."""
    store = MemoryKnowledgeStore()

    with pytest.raises(KeyError, match="not found"):
        store.list_documents("nonexistent")
