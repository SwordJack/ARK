#! python3
# -*- encoding: utf-8 -*-
"""Shared contract tests for knowledge store backends.

@File   :   test_store_contract.py
@Created:   2026/08/03 05:00 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from os import path
from typing import Callable
from uuid import uuid4

import pytest

from isobase.database.sql import SqlDbService
from isobase.knowledge.entities import (
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
)
from isobase.knowledge.stores import (
    MemoryKnowledgeStore,
    SqlKnowledgeStore,
)
from isobase.knowledge.stores.base import BaseKnowledgeStore
from isobase.knowledge.stores.memory import cosine_similarity


StoreFactory = Callable[[], BaseKnowledgeStore]


def _mongo_is_reachable() -> bool:
    """Returns True when a MongoDB server is accepting connections."""
    try:
        from pymongo import MongoClient

        client = MongoClient(
            "mongodb://localhost:27017",
            serverSelectionTimeoutMS=2000,
        )
        client.admin.command("ping")
        client.close()
        return True
    except Exception:
        return False


@pytest.fixture(params=["memory", "sql", "mongo"])
def store_factory(request, tmp_path):
    """Creates isolated store instances for shared contract tests."""
    if request.param == "memory":
        yield MemoryKnowledgeStore
        return

    if request.param == "sql":
        db_path = path.join(str(tmp_path), "knowledge_contract.db")

        def _create_sql_store() -> SqlKnowledgeStore:
            """Creates a SQL store connected to the same database path."""
            return SqlKnowledgeStore(SqlDbService(f"sqlite:///{db_path}"))

        yield _create_sql_store
        return

    if request.param == "mongo":
        if not _mongo_is_reachable():
            pytest.skip("MongoDB not available on localhost:27017")

        from isobase.database.mongo import MongoDbService
        from isobase.knowledge.stores import MongoKnowledgeStore

        db_name = f"isobase_contract_{uuid4().hex}"

        def _create_mongo_store() -> MongoKnowledgeStore:
            """Creates a Mongo store connected to the same database."""
            svc = MongoDbService(
                uri="mongodb://localhost:27017",
                database_name=db_name,
                transactions_enabled=False,
            )
            return MongoKnowledgeStore(svc)

        yield _create_mongo_store

        # Tear down: drop the isolated test database.
        try:
            from pymongo import MongoClient

            client = MongoClient(
                "mongodb://localhost:27017",
                serverSelectionTimeoutMS=2000,
            )
            client.drop_database(db_name)
            client.close()
        except Exception:
            pass
        return

    raise ValueError(f"Unsupported store backend: {request.param}")


@pytest.fixture
def store(store_factory: StoreFactory) -> BaseKnowledgeStore:
    """Creates a store instance for one contract test."""
    return store_factory()


# ---------------------------------------------------------------------------
# Test helpers — never pass explicit ids, always use the returned object.
# ---------------------------------------------------------------------------


def _kb(**kwargs) -> KnowledgeBase:
    """Builds a KnowledgeBase DTO. The store auto-generates ``id`` when
    not provided. Pass ``id=...`` explicitly when needed."""
    defaults = {
        "name": "Contract Test KB",
        "description": "Shared contract test KB",
        "embedding_model_id": "test-embedding",
        "dimensions": 3,
        "metadata": {"owner": "contract"},
    }
    defaults.update(kwargs)
    return KnowledgeBase(**defaults)


def _create_kb(store: BaseKnowledgeStore, **kwargs) -> KnowledgeBase:
    """Creates a knowledge base and returns the store-populated DTO."""
    return store.create_knowledge_base(_kb(**kwargs))


def _doc(kb: KnowledgeBase, **kwargs) -> KnowledgeDocument:
    """Builds a KnowledgeDocument DTO. The store auto-generates ``id`` when
    not provided. Pass ``id=...`` explicitly when needed."""
    defaults = {
        "knowledge_base_id": kb.id,
        "title": "Contract Test Doc",
        "source_uri": "file:///contract.md",
        "content": "Full content for contract test document.",
        "metadata": {"kind": "contract"},
    }
    defaults.update(kwargs)
    return KnowledgeDocument(**defaults)


def _create_doc(store: BaseKnowledgeStore, kb: KnowledgeBase, **kwargs) -> KnowledgeDocument:
    """Creates a document and returns the store-populated DTO."""
    return store.add_document(_doc(kb, **kwargs))


def _create_chunks(
    store: BaseKnowledgeStore,
    kb: KnowledgeBase,
    doc: KnowledgeDocument,
) -> None:
    """Creates three sample chunks with embeddings."""
    store.add_chunks(
        chunks=[
            KnowledgeChunk(
                document_id=doc.id,
                knowledge_base_id=kb.id,
                content="alpha chunk",
                index=0,
                token_count=2,
                metadata={"heading_path": ["A"]},
            ),
            KnowledgeChunk(
                document_id=doc.id,
                knowledge_base_id=kb.id,
                content="beta chunk",
                index=1,
                token_count=2,
            ),
            KnowledgeChunk(
                document_id=doc.id,
                knowledge_base_id=kb.id,
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


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------


def test_contract_create_get_and_list_knowledge_base(store: BaseKnowledgeStore) -> None:
    """All stores create, fetch, and list knowledge bases consistently."""
    kb = _create_kb(store)

    retrieved = store.get_knowledge_base(kb.id)
    listed = store.list_knowledge_bases()

    assert retrieved.id == kb.id
    assert retrieved.name == kb.name
    assert retrieved.description == kb.description
    assert retrieved.embedding_model_id == kb.embedding_model_id
    assert retrieved.metadata == kb.metadata
    assert retrieved.created_time is not None
    assert retrieved.updated_time is not None
    assert [item.id for item in listed] == [kb.id]


def test_contract_create_duplicate_knowledge_base_raises(store: BaseKnowledgeStore) -> None:
    """All stores reject duplicate knowledge base IDs."""
    kb = _create_kb(store)

    with pytest.raises(ValueError, match="already exists"):
        store.create_knowledge_base(_kb(id=kb.id, name="Duplicate"))


def test_contract_get_missing_knowledge_base_raises(store: BaseKnowledgeStore) -> None:
    """All stores raise KeyError when fetching a missing KB."""
    with pytest.raises(KeyError, match="not found"):
        store.get_knowledge_base("missing")


def test_contract_list_knowledge_bases_empty(store: BaseKnowledgeStore) -> None:
    """All stores return an empty list when no KBs exist."""
    assert store.list_knowledge_bases() == []


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def test_contract_add_get_and_list_document(store: BaseKnowledgeStore) -> None:
    """All stores add, fetch, and list documents consistently."""
    kb = _create_kb(store)
    doc = _create_doc(store, kb)

    retrieved = store.get_document(doc.id)
    listed = store.list_documents(kb.id)

    assert retrieved.id == doc.id
    assert retrieved.knowledge_base_id == kb.id
    assert retrieved.title == doc.title
    assert retrieved.source_uri == doc.source_uri
    assert retrieved.content == doc.content
    assert retrieved.metadata == doc.metadata
    assert retrieved.created_time is not None
    assert [item.id for item in listed] == [doc.id]


def test_contract_add_document_to_missing_kb_raises(store: BaseKnowledgeStore) -> None:
    """All stores reject documents whose parent knowledge base is missing."""
    fake_kb = KnowledgeBase(
        id="missing",
        name="Ghost",
        dimensions=3,
    )
    with pytest.raises(KeyError, match="not found"):
        store.add_document(_doc(fake_kb))


def test_contract_get_missing_document_raises(store: BaseKnowledgeStore) -> None:
    """All stores raise KeyError when fetching a missing document."""
    with pytest.raises(KeyError, match="not found"):
        store.get_document("missing")


def test_contract_list_documents_for_missing_kb_raises(store: BaseKnowledgeStore) -> None:
    """All stores raise KeyError when listing documents for a missing KB."""
    with pytest.raises(KeyError, match="not found"):
        store.list_documents("missing")


def test_contract_list_documents_empty(store: BaseKnowledgeStore) -> None:
    """All stores return an empty list when a KB has no documents."""
    kb = _create_kb(store)
    assert store.list_documents(kb.id) == []


# ---------------------------------------------------------------------------
# Chunks & embeddings
# ---------------------------------------------------------------------------


def test_contract_add_chunks_mismatched_embeddings_raises(store: BaseKnowledgeStore) -> None:
    """All stores require one embedding per chunk."""
    kb = _create_kb(store)
    doc = _create_doc(store, kb)

    with pytest.raises(ValueError, match="must match"):
        store.add_chunks(
            [KnowledgeChunk(
                document_id=doc.id,
                knowledge_base_id=kb.id,
                content="orphan chunk",
                index=0,
            )],
            [[1.0, 0.0], [0.0, 1.0]],
        )


def test_contract_add_chunks_to_missing_kb_raises(store: BaseKnowledgeStore) -> None:
    """All stores raise KeyError when adding chunks to a missing KB."""
    with pytest.raises(KeyError, match="not found"):
        store.add_chunks(
            [KnowledgeChunk(
                document_id="doc1",
                knowledge_base_id="missing",
                content="orphan chunk",
                index=0,
            )],
            [[1.0, 0.0]],
        )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_contract_search_returns_sorted_results(store: BaseKnowledgeStore) -> None:
    """All stores return search results sorted by descending similarity."""
    kb = _create_kb(store)
    doc = _create_doc(store, kb)
    _create_chunks(store, kb, doc)

    results = store.search(kb.id, [0.9, 0.1, 0.0], top_k=3)

    assert len(results) == 3
    assert [r.chunk.content for r in results] == [
        "alpha chunk",
        "beta chunk",
        "gamma chunk",
    ]
    assert results[0].score > results[1].score
    assert results[1].score > results[2].score
    assert results[0].chunk.metadata["heading_path"] == ["A"]
    assert results[0].document.id == doc.id
    assert results[0].document.title == doc.title
    assert results[0].score == pytest.approx(
        cosine_similarity([0.9, 0.1, 0.0], [1.0, 0.0, 0.0])
    )


def test_contract_search_respects_top_k(store: BaseKnowledgeStore) -> None:
    """All stores limit search results by top_k."""
    kb = _create_kb(store)
    doc = _create_doc(store, kb)
    _create_chunks(store, kb, doc)

    results = store.search(kb.id, [1.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1


def test_contract_search_empty_or_missing_kb_returns_empty(store: BaseKnowledgeStore) -> None:
    """All stores return an empty list for empty or missing knowledge bases."""
    kb = _create_kb(store)

    assert store.search(kb.id, [1.0, 0.0, 0.0]) == []
    assert store.search("missing", [1.0, 0.0, 0.0]) == []


def test_contract_search_does_not_cross_knowledge_bases(store: BaseKnowledgeStore) -> None:
    """All stores search only within the requested knowledge base."""
    kb1 = _create_kb(store, name="KB One")
    doc1 = _create_doc(store, kb1, title="Doc One")
    store.add_chunks(
        [KnowledgeChunk(
            document_id=doc1.id,
            knowledge_base_id=kb1.id,
            content="kb1 chunk",
            index=0,
        )],
        [[1.0, 0.0]],
    )

    kb2 = _create_kb(store, name="KB Two")
    doc2 = _create_doc(store, kb2, title="Doc Two")
    store.add_chunks(
        [KnowledgeChunk(
            document_id=doc2.id,
            knowledge_base_id=kb2.id,
            content="kb2 chunk",
            index=0,
        )],
        [[1.0, 0.0]],
    )

    results = store.search(kb1.id, [1.0, 0.0], top_k=10)

    assert len(results) == 1
    assert results[0].chunk.content == "kb1 chunk"
    assert results[0].document.id == doc1.id


def test_contract_search_vector_dimension_mismatch_raises(store: BaseKnowledgeStore) -> None:
    """All stores surface vector dimension mismatches as ValueError."""
    kb = _create_kb(store)
    doc = _create_doc(store, kb)
    _create_chunks(store, kb, doc)

    with pytest.raises(ValueError, match="same length"):
        store.search(kb.id, [1.0, 0.0], top_k=1)


# ---------------------------------------------------------------------------
# Delete — document
# ---------------------------------------------------------------------------


def test_contract_delete_document_cascades_index_data(store: BaseKnowledgeStore) -> None:
    """All stores remove a document's chunks and embeddings with the document."""
    kb = _create_kb(store)
    doc = _create_doc(store, kb)
    _create_chunks(store, kb, doc)

    store.delete_document(doc.id)

    with pytest.raises(KeyError, match="not found"):
        store.get_document(doc.id)
    assert store.list_documents(kb.id) == []
    assert store.search(kb.id, [1.0, 0.0, 0.0]) == []


def test_contract_delete_document_leaves_other_documents(store: BaseKnowledgeStore) -> None:
    """All stores delete only the target document's index data."""
    kb = _create_kb(store)

    keep_doc = _create_doc(store, kb, title="Keep")
    store.add_chunks(
        [KnowledgeChunk(
            document_id=keep_doc.id,
            knowledge_base_id=kb.id,
            content="keep chunk",
            index=0,
        )],
        [[1.0, 0.0]],
    )

    delete_doc = _create_doc(store, kb, title="Delete")
    store.add_chunks(
        [KnowledgeChunk(
            document_id=delete_doc.id,
            knowledge_base_id=kb.id,
            content="delete chunk",
            index=0,
        )],
        [[0.0, 1.0]],
    )

    store.delete_document(delete_doc.id)

    results = store.search(kb.id, [1.0, 0.0], top_k=10)
    assert len(results) == 1
    assert results[0].chunk.content == "keep chunk"
    assert store.get_document(keep_doc.id).title == "Keep"
    with pytest.raises(KeyError, match="not found"):
        store.get_document(delete_doc.id)


# ---------------------------------------------------------------------------
# Delete — knowledge base
# ---------------------------------------------------------------------------


def test_contract_delete_knowledge_base_cascades_all_data(store: BaseKnowledgeStore) -> None:
    """All stores remove every document and chunk under a deleted KB."""
    kb = _create_kb(store)
    doc = _create_doc(store, kb)
    _create_chunks(store, kb, doc)

    store.delete_knowledge_base(kb.id)

    with pytest.raises(KeyError, match="not found"):
        store.get_knowledge_base(kb.id)
    with pytest.raises(KeyError, match="not found"):
        store.get_document(doc.id)
    assert store.search(kb.id, [1.0, 0.0, 0.0]) == []


# ---------------------------------------------------------------------------
# Delete — error paths
# ---------------------------------------------------------------------------


def test_contract_delete_missing_entities_raise(store: BaseKnowledgeStore) -> None:
    """All stores raise KeyError when deleting missing KBs or documents."""
    with pytest.raises(KeyError, match="not found"):
        store.delete_document("missing")

    with pytest.raises(KeyError, match="not found"):
        store.delete_knowledge_base("missing")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_contract_persistent_store_can_reload_data(store_factory: StoreFactory) -> None:
    """Persistent stores preserve indexed data across store instances."""
    first_store = store_factory()
    if isinstance(first_store, MemoryKnowledgeStore):
        pytest.skip("MemoryKnowledgeStore is not persistent by design.")

    kb = _create_kb(first_store)
    doc = _create_doc(first_store, kb)
    _create_chunks(first_store, kb, doc)

    reloaded_store = store_factory()
    results = reloaded_store.search(kb.id, [1.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk.content == "alpha chunk"
    assert results[0].document.content == "Full content for contract test document."
