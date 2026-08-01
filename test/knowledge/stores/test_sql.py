#! python3
# -*- encoding: utf-8 -*-
"""Tests for SQL knowledge store persistence.

@File   :   test_sql.py
@Created:   2026/08/02 03:16 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import pytest
from isobase.database.sql import SqlDbService
from isobase.knowledge.entities import (
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
)
from isobase.knowledge.stores import SqlKnowledgeStore


def test_sql_store_persists_knowledge_base(tmp_path):
    """Test knowledge base persistence across store instances."""
    db_path = tmp_path / "knowledge.db"
    sql_service = SqlDbService(f"sqlite:///{db_path}")

    store = SqlKnowledgeStore(sql_service)
    kb = KnowledgeBase(
        id="kb1",
        name="Test KB",
        description="Test description",
        embedding_model_id="test-embedding",
        dimensions=3,
        metadata={"owner": "test"},
    )
    store.create_knowledge_base(kb)

    reloaded_store = SqlKnowledgeStore(sql_service)
    reloaded_kb = reloaded_store.get_knowledge_base("kb1")

    assert reloaded_kb.id == "kb1"
    assert reloaded_kb.name == "Test KB"
    assert reloaded_kb.description == "Test description"
    assert reloaded_kb.embedding_model_id == "test-embedding"
    assert reloaded_kb.metadata["owner"] == "test"
    assert reloaded_kb.created_time is not None

    sql_service.close()


def test_sql_store_persists_documents_chunks_and_embeddings(tmp_path):
    """Test document, chunk, and embedding persistence."""
    db_path = tmp_path / "knowledge.db"
    sql_service = SqlDbService(f"sqlite:///{db_path}")

    store = SqlKnowledgeStore(sql_service)
    store.create_knowledge_base(KnowledgeBase(id="kb1", name="Test KB"))
    store.add_document(KnowledgeDocument(
        id="doc1",
        knowledge_base_id="kb1",
        title="Test Doc",
        source_uri="file://test.txt",
        content="Test content",
        metadata={"kind": "text"},
    ))
    store.add_chunks(
        chunks=[
            KnowledgeChunk(
                id="c1",
                document_id="doc1",
                knowledge_base_id="kb1",
                content="first chunk",
                index=0,
                metadata={"heading_path": ["A"]},
            ),
            KnowledgeChunk(
                id="c2",
                document_id="doc1",
                knowledge_base_id="kb1",
                content="second chunk",
                index=1,
            ),
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    reloaded_store = SqlKnowledgeStore(sql_service)
    results = reloaded_store.search("kb1", [0.9, 0.1, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == "c1"
    assert results[0].chunk.metadata["heading_path"] == ["A"]
    assert results[0].document.title == "Test Doc"
    assert results[0].document.source_uri == "file://test.txt"
    assert results[0].document.metadata["kind"] == "text"
    assert results[0].score > results[1].score

    sql_service.close()


def test_sql_store_delete_document_persists(tmp_path):
    """Test deleted document is gone after reload."""
    db_path = tmp_path / "knowledge.db"
    sql_service = SqlDbService(f"sqlite:///{db_path}")

    store = SqlKnowledgeStore(sql_service)
    store.create_knowledge_base(KnowledgeBase(id="kb1", name="Test KB"))
    store.add_document(KnowledgeDocument(id="doc1", knowledge_base_id="kb1", title="Test Doc"))
    store.add_chunks(
        [KnowledgeChunk(id="c1", document_id="doc1", knowledge_base_id="kb1", content="x", index=0)],
        [[1.0, 0.0]],
    )

    store.delete_document("doc1")

    reloaded = SqlKnowledgeStore(sql_service)
    with pytest.raises(KeyError):
        reloaded.get_document("doc1")
    assert reloaded.search("kb1", [1.0, 0.0]) == []

    sql_service.close()


def test_sql_store_delete_knowledge_base_persists(tmp_path):
    """Test deleted knowledge base is gone after reload."""
    db_path = tmp_path / "knowledge.db"
    sql_service = SqlDbService(f"sqlite:///{db_path}")

    store = SqlKnowledgeStore(sql_service)
    store.create_knowledge_base(KnowledgeBase(id="kb1", name="Test KB"))
    store.add_document(KnowledgeDocument(id="doc1", knowledge_base_id="kb1", title="Test Doc"))
    store.add_chunks(
        [KnowledgeChunk(id="c1", document_id="doc1", knowledge_base_id="kb1", content="x", index=0)],
        [[1.0, 0.0]],
    )

    store.delete_knowledge_base("kb1")

    reloaded = SqlKnowledgeStore(sql_service)
    with pytest.raises(KeyError):
        reloaded.get_knowledge_base("kb1")
    with pytest.raises(KeyError):
        reloaded.get_document("doc1")

    sql_service.close()


def test_sql_store_list_knowledge_bases_persists(tmp_path):
    """Test listing knowledge bases across reload."""
    db_path = tmp_path / "knowledge.db"
    sql_service = SqlDbService(f"sqlite:///{db_path}")

    store = SqlKnowledgeStore(sql_service)
    store.create_knowledge_base(KnowledgeBase(id="kb1", name="First"))
    store.create_knowledge_base(KnowledgeBase(id="kb2", name="Second"))

    reloaded = SqlKnowledgeStore(sql_service)
    kbs = reloaded.list_knowledge_bases()

    assert len(kbs) == 2
    assert {kb.name for kb in kbs} == {"First", "Second"}

    sql_service.close()


def test_sql_store_list_documents_persists(tmp_path):
    """Test listing documents across reload."""
    db_path = tmp_path / "knowledge.db"
    sql_service = SqlDbService(f"sqlite:///{db_path}")

    store = SqlKnowledgeStore(sql_service)
    store.create_knowledge_base(KnowledgeBase(id="kb1", name="Test KB"))
    store.add_document(KnowledgeDocument(id="doc1", knowledge_base_id="kb1", title="Doc A"))
    store.add_document(KnowledgeDocument(id="doc2", knowledge_base_id="kb1", title="Doc B"))

    reloaded = SqlKnowledgeStore(sql_service)
    docs = reloaded.list_documents("kb1")

    assert len(docs) == 2
    assert {doc.title for doc in docs} == {"Doc A", "Doc B"}

    sql_service.close()
