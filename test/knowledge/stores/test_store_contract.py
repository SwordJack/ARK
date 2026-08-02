#! python3
# -*- encoding: utf-8 -*-
"""Shared contract tests for knowledge store backends.

@File   :   test_store_contract.py
@Created:   2026/08/03 05:00 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from typing import Callable

import pytest

from isobase.database.sql import SqlDbService
from isobase.knowledge.entities import (
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
)
from isobase.knowledge.stores import MemoryKnowledgeStore, SqlKnowledgeStore
from isobase.knowledge.stores.base import BaseKnowledgeStore
from isobase.knowledge.stores.memory import cosine_similarity


StoreFactory = Callable[[], BaseKnowledgeStore]


@pytest.fixture(params=["memory", "sql"])
def store_factory(request, tmp_path) -> StoreFactory:
    """Creates isolated store instances for shared contract tests.

    MongoDB is covered by live smoke tests because it requires an external
    database service. These shared tests stay self-contained for normal pytest
    runs while keeping the contract reusable for future backends.
    """
    if request.param == "memory":
        return MemoryKnowledgeStore

    if request.param == "sql":
        db_path = tmp_path / "knowledge_contract.db"
        sql_service = SqlDbService(f"sqlite:///{db_path}")
        return lambda: SqlKnowledgeStore(sql_service)

    raise ValueError(f"Unsupported store backend: {request.param}")


@pytest.fixture
def store(store_factory: StoreFactory) -> BaseKnowledgeStore:
    """Creates a store instance for one contract test."""
    return store_factory()


def _create_kb(store: BaseKnowledgeStore, kb_id: str = "kb1") -> KnowledgeBase:
    """Creates a sample knowledge base."""
    return store.create_knowledge_base(KnowledgeBase(
        id=kb_id,
        name=f"Knowledge Base {kb_id}",
        description="Shared contract test KB",
        embedding_model_id="test-embedding",
        dimensions=3,
        metadata={"owner": "contract"},
    ))


def _create_doc(
    store: BaseKnowledgeStore,
    kb_id: str = "kb1",
    doc_id: str = "doc1",
) -> KnowledgeDocument:
    """Creates a sample document."""
    return store.add_document(KnowledgeDocument(
        id=doc_id,
        knowledge_base_id=kb_id,
        title=f"Document {doc_id}",
        source_uri=f"file:///{doc_id}.txt",
        content=f"Full content for {doc_id}",
        metadata={"kind": "contract"},
    ))


def _create_chunks(
    store: BaseKnowledgeStore,
    kb_id: str = "kb1",
    doc_id: str = "doc1",
) -> None:
    """Creates sample chunks and embeddings."""
    store.add_chunks(
        chunks=[
            KnowledgeChunk(
                id="chunk-a",
                document_id=doc_id,
                knowledge_base_id=kb_id,
                content="alpha chunk",
                index=0,
                token_count=2,
                metadata={"heading_path": ["A"]},
            ),
            KnowledgeChunk(
                id="chunk-b",
                document_id=doc_id,
                knowledge_base_id=kb_id,
                content="beta chunk",
                index=1,
                token_count=2,
            ),
            KnowledgeChunk(
                id="chunk-c",
                document_id=doc_id,
                knowledge_base_id=kb_id,
                content="gamma chunk",
                index=2,
                token_count=2,
            ),
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def test_contract_create_get_and_list_knowledge_base(store: BaseKnowledgeStore):
    """All stores create, fetch, and list knowledge bases consistently."""
    kb = _create_kb(store)

    retrieved = store.get_knowledge_base(kb.id)
    listed = store.list_knowledge_bases()

    assert retrieved.id == "kb1"
    assert retrieved.name == "Knowledge Base kb1"
    assert retrieved.description == "Shared contract test KB"
    assert retrieved.embedding_model_id == "test-embedding"
    assert retrieved.metadata["owner"] == "contract"
    assert retrieved.created_time is not None
    assert retrieved.updated_time is not None
    assert [item.id for item in listed] == ["kb1"]


def test_contract_create_duplicate_knowledge_base_raises(store: BaseKnowledgeStore):
    """All stores reject duplicate knowledge base IDs."""
    _create_kb(store)

    with pytest.raises(ValueError, match="already exists"):
        store.create_knowledge_base(KnowledgeBase(id="kb1", name="Duplicate"))


def test_contract_add_get_and_list_document(store: BaseKnowledgeStore):
    """All stores add, fetch, and list documents consistently."""
    _create_kb(store)
    doc = _create_doc(store)

    retrieved = store.get_document(doc.id)
    listed = store.list_documents("kb1")

    assert retrieved.id == "doc1"
    assert retrieved.knowledge_base_id == "kb1"
    assert retrieved.title == "Document doc1"
    assert retrieved.source_uri == "file:///doc1.txt"
    assert retrieved.content == "Full content for doc1"
    assert retrieved.metadata["kind"] == "contract"
    assert retrieved.created_time is not None
    assert [item.id for item in listed] == ["doc1"]


def test_contract_add_document_to_missing_kb_raises(store: BaseKnowledgeStore):
    """All stores reject documents whose parent knowledge base is missing."""
    with pytest.raises(KeyError, match="not found"):
        _create_doc(store, kb_id="missing")


def test_contract_add_chunks_mismatched_embeddings_raises(store: BaseKnowledgeStore):
    """All stores require one embedding per chunk."""
    _create_kb(store)
    _create_doc(store)

    with pytest.raises(ValueError, match="must match"):
        store.add_chunks(
            [KnowledgeChunk(
                id="chunk-a",
                document_id="doc1",
                knowledge_base_id="kb1",
                content="alpha chunk",
                index=0,
            )],
            [[1.0, 0.0], [0.0, 1.0]],
        )


def test_contract_search_returns_sorted_results(store: BaseKnowledgeStore):
    """All stores return search results sorted by descending similarity."""
    _create_kb(store)
    _create_doc(store)
    _create_chunks(store)

    results = store.search("kb1", [0.9, 0.1, 0.0], top_k=3)

    assert len(results) == 3
    assert [result.chunk.id for result in results] == [
        "chunk-a",
        "chunk-b",
        "chunk-c",
    ]
    assert results[0].score > results[1].score
    assert results[1].score > results[2].score
    assert results[0].chunk.content == "alpha chunk"
    assert results[0].chunk.metadata["heading_path"] == ["A"]
    assert results[0].document.id == "doc1"
    assert results[0].document.title == "Document doc1"
    assert results[0].score == pytest.approx(
        cosine_similarity([0.9, 0.1, 0.0], [1.0, 0.0, 0.0])
    )


def test_contract_search_respects_top_k(store: BaseKnowledgeStore):
    """All stores limit search results by top_k."""
    _create_kb(store)
    _create_doc(store)
    _create_chunks(store)

    results = store.search("kb1", [1.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk.id == "chunk-a"


def test_contract_search_empty_or_missing_kb_returns_empty(store: BaseKnowledgeStore):
    """All stores return an empty list for empty or missing knowledge bases."""
    _create_kb(store)

    assert store.search("kb1", [1.0, 0.0, 0.0]) == []
    assert store.search("missing", [1.0, 0.0, 0.0]) == []


def test_contract_search_does_not_cross_knowledge_bases(store: BaseKnowledgeStore):
    """All stores search only within the requested knowledge base."""
    _create_kb(store, "kb1")
    _create_doc(store, "kb1", "doc1")
    store.add_chunks(
        [KnowledgeChunk(
            id="chunk-a",
            document_id="doc1",
            knowledge_base_id="kb1",
            content="kb1 chunk",
            index=0,
        )],
        [[1.0, 0.0]],
    )

    _create_kb(store, "kb2")
    _create_doc(store, "kb2", "doc2")
    store.add_chunks(
        [KnowledgeChunk(
            id="chunk-b",
            document_id="doc2",
            knowledge_base_id="kb2",
            content="kb2 chunk",
            index=0,
        )],
        [[1.0, 0.0]],
    )

    results = store.search("kb1", [1.0, 0.0], top_k=10)

    assert len(results) == 1
    assert results[0].chunk.content == "kb1 chunk"
    assert results[0].document.id == "doc1"


def test_contract_search_vector_dimension_mismatch_raises(store: BaseKnowledgeStore):
    """All stores surface vector dimension mismatches as ValueError."""
    _create_kb(store)
    _create_doc(store)
    _create_chunks(store)

    with pytest.raises(ValueError, match="same length"):
        store.search("kb1", [1.0, 0.0], top_k=1)


def test_contract_delete_document_cascades_index_data(store: BaseKnowledgeStore):
    """All stores remove a document's chunks and embeddings with the document."""
    _create_kb(store)
    _create_doc(store)
    _create_chunks(store)

    store.delete_document("doc1")

    with pytest.raises(KeyError, match="not found"):
        store.get_document("doc1")
    assert store.list_documents("kb1") == []
    assert store.search("kb1", [1.0, 0.0, 0.0]) == []


def test_contract_delete_document_leaves_other_documents(store: BaseKnowledgeStore):
    """All stores delete only the target document's index data."""
    _create_kb(store)
    _create_doc(store, "kb1", "doc1")
    store.add_chunks(
        [KnowledgeChunk(
            id="chunk-a",
            document_id="doc1",
            knowledge_base_id="kb1",
            content="keep chunk",
            index=0,
        )],
        [[1.0, 0.0]],
    )
    _create_doc(store, "kb1", "doc2")
    store.add_chunks(
        [KnowledgeChunk(
            id="chunk-b",
            document_id="doc2",
            knowledge_base_id="kb1",
            content="delete chunk",
            index=0,
        )],
        [[0.0, 1.0]],
    )

    store.delete_document("doc2")

    results = store.search("kb1", [1.0, 0.0], top_k=10)
    assert len(results) == 1
    assert results[0].chunk.content == "keep chunk"
    assert store.get_document("doc1").title == "Document doc1"
    with pytest.raises(KeyError, match="not found"):
        store.get_document("doc2")


def test_contract_delete_knowledge_base_cascades_all_data(store: BaseKnowledgeStore):
    """All stores remove every document and chunk under a deleted KB."""
    _create_kb(store)
    _create_doc(store)
    _create_chunks(store)

    store.delete_knowledge_base("kb1")

    with pytest.raises(KeyError, match="not found"):
        store.get_knowledge_base("kb1")
    with pytest.raises(KeyError, match="not found"):
        store.get_document("doc1")
    assert store.search("kb1", [1.0, 0.0, 0.0]) == []


def test_contract_get_missing_kb_or_document_raises(store: BaseKnowledgeStore):
    """All stores raise KeyError when fetching a missing KB or document."""
    with pytest.raises(KeyError, match="not found"):
        store.get_knowledge_base("missing")

    with pytest.raises(KeyError, match="not found"):
        store.get_document("missing")


def test_contract_list_documents_for_missing_kb_raises(store: BaseKnowledgeStore):
    """All stores raise KeyError when listing documents for a missing KB."""
    with pytest.raises(KeyError, match="not found"):
        store.list_documents("missing")


def test_contract_list_empty_returns_empty_list(store: BaseKnowledgeStore):
    """All stores return empty lists when there is nothing to list."""
    assert store.list_knowledge_bases() == []

    _create_kb(store)
    assert store.list_documents("kb1") == []


def test_contract_add_chunks_to_missing_kb_raises(store: BaseKnowledgeStore):
    """All stores raise KeyError when adding chunks to a missing KB."""
    with pytest.raises(KeyError, match="not found"):
        store.add_chunks(
            [KnowledgeChunk(
                id="chunk-a",
                document_id="doc1",
                knowledge_base_id="missing",
                content="orphan chunk",
                index=0,
            )],
            [[1.0, 0.0]],
        )


def test_contract_delete_missing_entities_raise(store: BaseKnowledgeStore):
    """All stores raise KeyError when deleting missing KBs or documents."""
    with pytest.raises(KeyError, match="not found"):
        store.delete_document("missing")

    with pytest.raises(KeyError, match="not found"):
        store.delete_knowledge_base("missing")


def test_contract_persistent_store_can_reload_data(store_factory: StoreFactory):
    """Persistent stores preserve indexed data across store instances."""
    first_store = store_factory()
    if isinstance(first_store, MemoryKnowledgeStore):
        pytest.skip("MemoryKnowledgeStore is not persistent by design.")

    _create_kb(first_store)
    _create_doc(first_store)
    _create_chunks(first_store)

    reloaded_store = store_factory()
    results = reloaded_store.search("kb1", [1.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk.content == "alpha chunk"
    assert results[0].document.content == "Full content for doc1"
