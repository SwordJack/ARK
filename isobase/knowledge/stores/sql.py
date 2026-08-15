#! python3
# -*- encoding: utf-8 -*-
"""SQL knowledge store implementation.

@File   :   sql.py
@Created:   2026/08/02 00:00
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import json
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, select
from sqlalchemy.orm import DeclarativeBase

from isobase.database.sql import SqlDbModelMixin, SqlDbService

from ..entities import (
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalResult,
)
from ..retrieval import DenseRetriever, DenseRetrievalItem
from .base import BaseKnowledgeStore


class KnowledgeSqlBase(DeclarativeBase):
    """Declarative base for knowledge SQL models.

    SQLAlchemy forbids inheriting ``DeclarativeBase`` directly; every
    concrete model must inherit from a user-defined subclass that serves
    as the shared metadata registry.  ``create_all(Base)`` creates all
    knowledge tables in one call.
    """

    pass


class KnowledgeBaseModel(SqlDbModelMixin, KnowledgeSqlBase):
    """SQL model for knowledge bases."""

    __tablename__ = "knowledge_bases"

    id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False, default="")
    embedding_model_id = Column(String(255), nullable=False, default="")
    dimensions = Column(Integer, nullable=False)
    chunk_size = Column(Integer, nullable=False)
    chunk_overlap = Column(Integer, nullable=False)
    metadata_json = Column("metadata", Text, nullable=False, default="{}")
    created_time = Column(DateTime(timezone=True), nullable=True)
    updated_time = Column(DateTime(timezone=True), nullable=True)


class KnowledgeDocumentModel(SqlDbModelMixin, KnowledgeSqlBase):
    """SQL model for knowledge documents."""

    __tablename__ = "knowledge_documents"

    id = Column(String(64), primary_key=True)
    knowledge_base_id = Column(
        String(64), ForeignKey("knowledge_bases.id"), nullable=False
    )
    title = Column(String(255), nullable=False)
    source_uri = Column(Text, nullable=False, default="")
    content = Column(Text, nullable=False, default="")
    metadata_json = Column("metadata", Text, nullable=False, default="{}")
    created_time = Column(DateTime(timezone=True), nullable=True)


class KnowledgeChunkModel(SqlDbModelMixin, KnowledgeSqlBase):
    """SQL model for knowledge chunks."""

    __tablename__ = "knowledge_chunks"

    id = Column(String(64), primary_key=True)
    document_id = Column(String(64), nullable=False)
    knowledge_base_id = Column(
        String(64), ForeignKey("knowledge_bases.id"), nullable=False
    )
    content = Column(Text, nullable=False)
    index = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    metadata_json = Column("metadata", Text, nullable=False, default="{}")


class KnowledgeEmbeddingModel(SqlDbModelMixin, KnowledgeSqlBase):
    """SQL model for chunk embeddings."""

    __tablename__ = "knowledge_embeddings"

    chunk_id = Column(
        String(64), ForeignKey("knowledge_chunks.id"), primary_key=True
    )
    embedding = Column(Text, nullable=False)


class SqlKnowledgeStore(BaseKnowledgeStore):
    """SQL-backed knowledge store using JSON text embeddings.

    This implementation stores vectors as JSON text and computes cosine
    similarity in Python.  CRUD operations delegate to
    :class:`SqlDbModelMixin` to stay consistent with the rest of IsoBase.
    """

    def __init__(self, sql_service: Optional[SqlDbService] = None) -> None:
        """Initializes the SQL knowledge store.

        Args:
            sql_service: Configured SQL database service. If omitted, an
                in-memory SQLite database is created.
        """
        if sql_service is None:
            sql_service = SqlDbService("sqlite:///:memory:")
        self.sql_service = sql_service
        self.sql_service.create_all(KnowledgeSqlBase)

        # Wire the service to each model class so that SqlDbModelMixin
        # methods (insert / find_by_id / delete_many / …) operate on
        # the same engine.
        KnowledgeBaseModel.use_sql_service(sql_service)
        KnowledgeDocumentModel.use_sql_service(sql_service)
        KnowledgeChunkModel.use_sql_service(sql_service)
        KnowledgeEmbeddingModel.use_sql_service(sql_service)
        self.retriever = DenseRetriever()

    # ------------------------------------------------------------------
    # Knowledge base CRUD
    # ------------------------------------------------------------------

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

        if KnowledgeBaseModel.find_by_id(kb.id) is not None:
            raise ValueError(f"Knowledge base {kb.id} already exists")

        model = self._kb_to_model(kb)
        model.insert()
        return self._model_to_kb(model)

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBase:
        """Retrieves a knowledge base by ID.

        Args:
            kb_id: Knowledge base identifier.

        Returns:
            The knowledge base object.

        Raises:
            KeyError: If knowledge base not found.
        """
        model = KnowledgeBaseModel.find_by_id(kb_id)
        if model is None:
            raise KeyError(f"Knowledge base {kb_id} not found")
        return self._model_to_kb(model)

    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        """Lists all knowledge bases.

        Returns:
            List of all knowledge bases. May be empty.
        """
        models = KnowledgeBaseModel.find_many()
        return [self._model_to_kb(m) for m in models]

    def delete_knowledge_base(self, kb_id: str) -> None:
        """Deletes a knowledge base and all its documents/chunks/embeddings.

        Args:
            kb_id: Knowledge base identifier.

        Raises:
            KeyError: If knowledge base not found.
        """
        if KnowledgeBaseModel.find_by_id(kb_id) is None:
            raise KeyError(f"Knowledge base {kb_id} not found")

        def _cascade(session: Any) -> None:
            chunks = KnowledgeChunkModel.find_many(
                KnowledgeChunkModel.knowledge_base_id == kb_id,
                session=session,
            )
            chunk_ids = [c.id for c in chunks]
            if chunk_ids:
                KnowledgeEmbeddingModel.delete_many(
                    KnowledgeEmbeddingModel.chunk_id.in_(chunk_ids),
                    session=session,
                )
                KnowledgeChunkModel.delete_many(
                    KnowledgeChunkModel.id.in_(chunk_ids),
                    session=session,
                )
            KnowledgeDocumentModel.delete_many(
                KnowledgeDocumentModel.knowledge_base_id == kb_id,
                session=session,
            )
            # delete_many above already removed the document rows; now
            # delete the KB row itself via the instance method.
            kb_model = KnowledgeBaseModel.find_by_id(kb_id, session=session)
            kb_model.delete(session=session)

        KnowledgeBaseModel.execute_transaction(_cascade)

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------

    def add_document(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        """Stores a document.

        Args:
            doc: KnowledgeDocument object. If id is empty, generates a UUID.

        Returns:
            The stored document with timestamps.

        Raises:
            KeyError: If the parent knowledge base does not exist.
        """
        if KnowledgeBaseModel.find_by_id(doc.knowledge_base_id) is None:
            raise KeyError(f"Knowledge base {doc.knowledge_base_id} not found")

        if not doc.id:
            doc.id = str(uuid.uuid4())

        model = self._doc_to_model(doc)
        model.insert()
        return self._model_to_doc(model)

    def get_document(self, doc_id: str) -> KnowledgeDocument:
        """Retrieves a document by ID.

        Args:
            doc_id: Document identifier.

        Returns:
            The document object.

        Raises:
            KeyError: If document not found.
        """
        model = KnowledgeDocumentModel.find_by_id(doc_id)
        if model is None:
            raise KeyError(f"Document {doc_id} not found")
        return self._model_to_doc(model)

    def get_documents(self, doc_ids: List[str]) -> Dict[str, KnowledgeDocument]:
        """Retrieves multiple documents by ID."""
        unique_ids = list(dict.fromkeys(doc_ids))
        if not unique_ids:
            return {}

        rows = KnowledgeDocumentModel.find_many(
            KnowledgeDocumentModel.id.in_(unique_ids),
        )
        return {row.id: self._model_to_doc(row) for row in rows}

    def list_documents(self, kb_id: str) -> list[KnowledgeDocument]:
        """Lists all documents in a knowledge base.

        Args:
            kb_id: Knowledge base identifier.

        Returns:
            List of documents. May be empty.

        Raises:
            KeyError: If knowledge base not found.
        """
        if KnowledgeBaseModel.find_by_id(kb_id) is None:
            raise KeyError(f"Knowledge base {kb_id} not found")
        models = KnowledgeDocumentModel.find_many(
            KnowledgeDocumentModel.knowledge_base_id == kb_id,
        )
        return [self._model_to_doc(m) for m in models]

    def delete_document(self, doc_id: str) -> None:
        """Deletes a document and all its chunks/embeddings.

        Args:
            doc_id: Document identifier.

        Raises:
            KeyError: If document not found.
        """
        if KnowledgeDocumentModel.find_by_id(doc_id) is None:
            raise KeyError(f"Document {doc_id} not found")

        def _cascade(session: Any) -> None:
            chunks = KnowledgeChunkModel.find_many(
                KnowledgeChunkModel.document_id == doc_id,
                session=session,
            )
            chunk_ids = [c.id for c in chunks]
            if chunk_ids:
                KnowledgeEmbeddingModel.delete_many(
                    KnowledgeEmbeddingModel.chunk_id.in_(chunk_ids),
                    session=session,
                )
                KnowledgeChunkModel.delete_many(
                    KnowledgeChunkModel.id.in_(chunk_ids),
                    session=session,
                )
            doc_model = KnowledgeDocumentModel.find_by_id(doc_id, session=session)
            doc_model.delete(session=session)

        KnowledgeDocumentModel.execute_transaction(_cascade)

    # ------------------------------------------------------------------
    # Chunks & search
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[KnowledgeChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Stores chunks and their embeddings.

        Args:
            chunks: List of chunks to store.
            embeddings: Corresponding embedding vectors.

        Raises:
            ValueError: If len(chunks) != len(embeddings).
            KeyError: If parent knowledge base does not exist.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match "
                f"embedding count ({len(embeddings)})"
            )

        # Verify KB existence once before entering the transaction.
        if chunks:
            kb_id = chunks[0].knowledge_base_id
            if KnowledgeBaseModel.find_by_id(kb_id) is None:
                raise KeyError(f"Knowledge base {kb_id} not found")

        def _bulk_insert(session: Any) -> None:
            for chunk, embedding in zip(chunks, embeddings):
                if not chunk.id:
                    chunk.id = str(uuid.uuid4())

                chunk_model = self._chunk_to_model(chunk)
                chunk_model.insert(session=session)

                embed_model = KnowledgeEmbeddingModel(
                    chunk_id=chunk.id,
                    embedding=json.dumps(embedding),
                )
                embed_model.insert(session=session)

        # Use any model class — they all share the same SqlDbService.
        KnowledgeBaseModel.execute_transaction(_bulk_insert)

    def list_chunks(self, kb_id: str) -> List[KnowledgeChunk]:
        """Lists all chunks in a knowledge base.

        Args:
            kb_id: Knowledge base identifier.

        Returns:
            List of chunks. May be empty.
        """
        rows = KnowledgeChunkModel.find_many(
            KnowledgeChunkModel.knowledge_base_id == kb_id,
        )
        return [self._model_to_chunk(row) for row in rows]

    def search(
        self,
        kb_id: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Searches for similar chunks by vector similarity.

        The search still uses a raw SQLAlchemy JOIN because
        :class:`SqlDbModelMixin` does not expose multi-entity queries.

        Args:
            kb_id: Knowledge base ID to search within.
            query_embedding: Query vector.
            top_k: Maximum number of results.

        Returns:
            List of retrieval results, sorted by score descending.
            Empty list if knowledge base is empty or has no chunks.
        """
        with self.sql_service.create_session() as session:
            rows = session.execute(
                select(KnowledgeChunkModel, KnowledgeEmbeddingModel)
                .join(
                    KnowledgeEmbeddingModel,
                    KnowledgeEmbeddingModel.chunk_id == KnowledgeChunkModel.id,
                )
                .where(KnowledgeChunkModel.knowledge_base_id == kb_id)
            ).all()

            chunks = [self._model_to_chunk(chunk_model) for chunk_model, _ in rows]
            documents = self.get_documents([chunk.document_id for chunk in chunks])

            items = []
            for chunk, (_, embedding_model) in zip(chunks, rows):
                items.append(DenseRetrievalItem(
                    chunk=chunk,
                    embedding=json.loads(embedding_model.embedding),
                    document=documents.get(chunk.document_id),
                ))

        return self.retriever.retrieve(
            query_embedding=query_embedding,
            items=items,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # DTO ↔ model converters
    # ------------------------------------------------------------------
    # created_time / updated_time are intentionally *not* copied into
    # newly constructed model instances — :meth:`SqlDbModelMixin.insert`
    # populates them automatically.

    def _kb_to_model(self, kb: KnowledgeBase) -> KnowledgeBaseModel:
        """Converts a knowledge base DTO to a SQL model."""
        return KnowledgeBaseModel(
            id=kb.id,
            name=kb.name,
            description=kb.description,
            embedding_model_id=kb.embedding_model_id,
            dimensions=kb.dimensions,
            chunk_size=kb.chunk_size,
            chunk_overlap=kb.chunk_overlap,
            metadata_json=self._dump_metadata(kb.metadata),
        )

    def _model_to_kb(self, model: KnowledgeBaseModel) -> KnowledgeBase:
        """Converts a SQL model to a knowledge base DTO."""
        return KnowledgeBase(
            id=model.id,
            name=model.name,
            description=model.description,
            embedding_model_id=model.embedding_model_id,
            dimensions=model.dimensions,
            chunk_size=model.chunk_size,
            chunk_overlap=model.chunk_overlap,
            metadata=self._load_metadata(model.metadata_json),
            created_time=model.created_time,
            updated_time=model.updated_time,
        )

    def _doc_to_model(self, doc: KnowledgeDocument) -> KnowledgeDocumentModel:
        """Converts a document DTO to a SQL model."""
        return KnowledgeDocumentModel(
            id=doc.id,
            knowledge_base_id=doc.knowledge_base_id,
            title=doc.title,
            source_uri=doc.source_uri,
            content=doc.content,
            metadata_json=self._dump_metadata(doc.metadata),
        )

    def _model_to_doc(self, model: KnowledgeDocumentModel) -> KnowledgeDocument:
        """Converts a SQL model to a document DTO."""
        return KnowledgeDocument(
            id=model.id,
            knowledge_base_id=model.knowledge_base_id,
            title=model.title,
            source_uri=model.source_uri,
            content=model.content,
            metadata=self._load_metadata(model.metadata_json),
            created_time=model.created_time,
        )

    def _chunk_to_model(self, chunk: KnowledgeChunk) -> KnowledgeChunkModel:
        """Converts a chunk DTO to a SQL model."""
        return KnowledgeChunkModel(
            id=chunk.id,
            document_id=chunk.document_id,
            knowledge_base_id=chunk.knowledge_base_id,
            content=chunk.content,
            index=chunk.index,
            token_count=chunk.token_count,
            metadata_json=self._dump_metadata(chunk.metadata),
        )

    def _model_to_chunk(self, model: KnowledgeChunkModel) -> KnowledgeChunk:
        """Converts a SQL model to a chunk DTO."""
        return KnowledgeChunk(
            id=model.id,
            document_id=model.document_id,
            knowledge_base_id=model.knowledge_base_id,
            content=model.content,
            index=model.index,
            token_count=model.token_count,
            metadata=self._load_metadata(model.metadata_json),
        )

    def _dump_metadata(self, metadata: Any) -> str:
        """Serializes metadata to JSON text."""
        return json.dumps(metadata)

    def _load_metadata(self, metadata_json: str) -> Any:
        """Deserializes metadata JSON text."""
        return json.loads(metadata_json or "{}")
