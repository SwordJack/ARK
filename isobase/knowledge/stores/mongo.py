#! python3
# -*- encoding: utf-8 -*-
"""MongoDB knowledge store implementation.

@File   :   mongo.py
@Created:   2026/08/02 18:27
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from isobase.database.mongo import MongoDbModel, MongoDbService, mongo_db

from ..entities import (
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalResult,
)
from ..retrieval import DenseRetriever, DenseRetrievalItem
from .base import BaseKnowledgeStore


class KnowledgeBaseMongoModel(MongoDbModel):
    """MongoDB model for knowledge bases."""

    collection_name = "knowledge_bases"

    def __init__(
        self,
        name: str = "",
        description: str = "",
        embedding_model_id: str = "",
        dimensions: int = 1024,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes a knowledge base Mongo model.

        Args:
            name: Human-readable name.
            description: Optional description of the knowledge base purpose.
            embedding_model_id: Identifier of the embedding model used.
            dimensions: Embedding vector dimensions.
            chunk_size: Default chunk size in characters.
            chunk_overlap: Overlap between adjacent chunks in characters.
            metadata: Additional metadata (tags, owner, etc.).
            **kwargs: Forwarded to :class:`MongoDbModel` (e.g. ``_id``,
                ``created_time``, ``updated_time``).
        """
        self.name = name
        self.description = description
        self.embedding_model_id = embedding_model_id
        self.dimensions = dimensions
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata = metadata or {}
        super().__init__(
            **kwargs,
        )


class KnowledgeDocumentMongoModel(MongoDbModel):
    """MongoDB model for knowledge documents."""

    collection_name = "knowledge_documents"

    def __init__(
        self,
        knowledge_base_id: str = "",
        title: str = "",
        source_uri: str = "",
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes a knowledge document Mongo model.

        Args:
            knowledge_base_id: Parent knowledge base ID.
            title: Document title.
            source_uri: Source location (URL, file path, etc.).
            content: Full text content of the document.
            metadata: Additional metadata (author, date, tags, etc.).
            **kwargs: Forwarded to :class:`MongoDbModel` (e.g. ``_id``,
                ``created_time``).
        """
        self.knowledge_base_id = knowledge_base_id
        self.title = title
        self.source_uri = source_uri
        self.content = content
        self.metadata = metadata or {}
        super().__init__(
            **kwargs,
        )


class KnowledgeChunkMongoModel(MongoDbModel):
    """MongoDB model for knowledge chunks."""

    collection_name = "knowledge_chunks"

    def __init__(
        self,
        document_id: str = "",
        knowledge_base_id: str = "",
        content: str = "",
        index: int = 0,
        token_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes a knowledge chunk Mongo model.

        Args:
            document_id: Parent document ID.
            knowledge_base_id: Parent knowledge base ID.
            content: Text content of the chunk.
            index: Position of this chunk within the parent document (0-based).
            token_count: Approximate token count (for cost estimation).
            metadata: Additional metadata (heading path, page number, etc.).
            **kwargs: Forwarded to :class:`MongoDbModel` (e.g. ``_id``).
        """
        self.document_id = document_id
        self.knowledge_base_id = knowledge_base_id
        self.content = content
        self.index = index
        self.token_count = token_count
        self.metadata = metadata or {}
        super().__init__(
            **kwargs,
        )


class KnowledgeEmbeddingMongoModel(MongoDbModel):
    """MongoDB model for chunk embeddings."""

    collection_name = "knowledge_embeddings"

    def __init__(
        self,
        chunk_id: str = "",
        embedding: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes a chunk embedding Mongo model.

        Args:
            chunk_id: Parent chunk ID.
            embedding: Embedding vector as a list of floats.
            **kwargs: Forwarded to :class:`MongoDbModel` (e.g. ``_id``).
        """
        self.chunk_id = chunk_id
        self.embedding = embedding or []
        super().__init__(
            **kwargs,
        )


class MongoKnowledgeStore(BaseKnowledgeStore):
    """MongoDB-backed knowledge store using Mongo documents for embeddings.

    MongoDB owns each document's ``_id``. Returned knowledge DTO ids are the
    string form of those BSON ObjectIds, matching :class:`MongoDbModel.id`.
    """

    def __init__(self, mongo_service: Optional[MongoDbService] = None) -> None:
        """Initializes the MongoDB knowledge store.

        Args:
            mongo_service: Configured MongoDB service. If omitted, the default
                global MongoDB service is used.
        """
        self.mongo_service = mongo_service or mongo_db
        KnowledgeBaseMongoModel.use_mongo_service(self.mongo_service)
        KnowledgeDocumentMongoModel.use_mongo_service(self.mongo_service)
        KnowledgeChunkMongoModel.use_mongo_service(self.mongo_service)
        KnowledgeEmbeddingMongoModel.use_mongo_service(self.mongo_service)
        self.retriever = DenseRetriever()

    def create_knowledge_base(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Creates a new knowledge base.

        Args:
            kb: KnowledgeBase object. MongoDB generates the stored identifier.

        Returns:
            The created knowledge base with MongoDB id and timestamps.

        Raises:
            ValueError: If a knowledge base with this id already exists.
        """
        if kb.id and KnowledgeBaseMongoModel.find_by_id(kb.id) is not None:
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
        model = KnowledgeBaseMongoModel.find_by_id(kb_id)
        if model is None:
            raise KeyError(f"Knowledge base {kb_id} not found")
        return self._model_to_kb(model)

    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        """Lists all knowledge bases.

        Returns:
            List of all knowledge bases. May be empty.
        """
        models = KnowledgeBaseMongoModel.find_many()
        return [self._model_to_kb(m) for m in models]

    def delete_knowledge_base(self, kb_id: str) -> None:
        """Deletes a knowledge base and all its documents/chunks/embeddings.

        Args:
            kb_id: Knowledge base identifier.

        Raises:
            KeyError: If knowledge base not found.
        """
        kb_model = KnowledgeBaseMongoModel.find_by_id(kb_id)
        if kb_model is None:
            raise KeyError(f"Knowledge base {kb_id} not found")

        def _cascade(session: Any) -> None:
            chunks = KnowledgeChunkMongoModel.find_many({"knowledge_base_id": kb_id})
            chunk_ids = [c.id for c in chunks]
            if chunk_ids:
                KnowledgeEmbeddingMongoModel.delete_many(
                    {"chunk_id": {"$in": chunk_ids}},
                    session=session,
                )
                KnowledgeChunkMongoModel.delete_many(
                    {"_id": {"$in": [c._id for c in chunks]}},
                    session=session,
                )
            KnowledgeDocumentMongoModel.delete_many(
                {"knowledge_base_id": kb_id},
                session=session,
            )
            kb_model.delete(session=session)

        KnowledgeBaseMongoModel.execute_transaction(_cascade)

    def add_document(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        """Stores a document.

        Args:
            doc: KnowledgeDocument object. MongoDB generates the stored identifier.

        Returns:
            The stored document with MongoDB id and timestamps.

        Raises:
            KeyError: If the parent knowledge base does not exist.
        """
        if KnowledgeBaseMongoModel.find_by_id(doc.knowledge_base_id) is None:
            raise KeyError(f"Knowledge base {doc.knowledge_base_id} not found")

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
        model = KnowledgeDocumentMongoModel.find_by_id(doc_id)
        if model is None:
            raise KeyError(f"Document {doc_id} not found")
        return self._model_to_doc(model)

    def list_documents(self, kb_id: str) -> list[KnowledgeDocument]:
        """Lists all documents in a knowledge base.

        Args:
            kb_id: Knowledge base identifier.

        Returns:
            List of documents. May be empty.

        Raises:
            KeyError: If knowledge base not found.
        """
        if KnowledgeBaseMongoModel.find_by_id(kb_id) is None:
            raise KeyError(f"Knowledge base {kb_id} not found")
        models = KnowledgeDocumentMongoModel.find_many({"knowledge_base_id": kb_id})
        return [self._model_to_doc(m) for m in models]

    def delete_document(self, doc_id: str) -> None:
        """Deletes a document and all its chunks/embeddings.

        Args:
            doc_id: Document identifier.

        Raises:
            KeyError: If document not found.
        """
        doc_model = KnowledgeDocumentMongoModel.find_by_id(doc_id)
        if doc_model is None:
            raise KeyError(f"Document {doc_id} not found")

        def _cascade(session: Any) -> None:
            chunks = KnowledgeChunkMongoModel.find_many({"document_id": doc_id})
            chunk_ids = [c.id for c in chunks]
            if chunk_ids:
                KnowledgeEmbeddingMongoModel.delete_many(
                    {"chunk_id": {"$in": chunk_ids}},
                    session=session,
                )
                KnowledgeChunkMongoModel.delete_many(
                    {"_id": {"$in": [c._id for c in chunks]}},
                    session=session,
                )
            doc_model.delete(session=session)

        KnowledgeDocumentMongoModel.execute_transaction(_cascade)

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
            KeyError: If parent document or knowledge base does not exist.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match "
                f"embedding count ({len(embeddings)})"
            )

        for chunk in chunks:
            if KnowledgeBaseMongoModel.find_by_id(chunk.knowledge_base_id) is None:
                raise KeyError(f"Knowledge base {chunk.knowledge_base_id} not found")
            if KnowledgeDocumentMongoModel.find_by_id(chunk.document_id) is None:
                raise KeyError(f"Document {chunk.document_id} not found")

        def _bulk_insert(session: Any) -> None:
            for chunk, embedding in zip(chunks, embeddings):
                chunk_model = self._chunk_to_model(chunk)
                chunk_model.insert(session=session)
                chunk.id = chunk_model.id

                embed_model = KnowledgeEmbeddingMongoModel(
                    chunk_id=chunk.id,
                    embedding=embedding,
                )
                embed_model.insert(session=session)

        KnowledgeBaseMongoModel.execute_transaction(_bulk_insert)

    def list_chunks(self, kb_id: str) -> List[KnowledgeChunk]:
        """Lists all chunks in a knowledge base.

        Args:
            kb_id: Knowledge base identifier.

        Returns:
            List of chunks. May be empty.
        """
        chunks = KnowledgeChunkMongoModel.find_many({"knowledge_base_id": kb_id})
        return [self._model_to_chunk(c) for c in chunks]

    def search(
        self,
        kb_id: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Searches for similar chunks by vector similarity.

        Args:
            kb_id: Knowledge base ID to search within.
            query_embedding: Query vector.
            top_k: Maximum number of results.

        Returns:
            List of retrieval results, sorted by score descending.
            Empty list if knowledge base is empty or has no chunks.
        """
        chunks = KnowledgeChunkMongoModel.find_many({"knowledge_base_id": kb_id})
        items = []
        for chunk_model in chunks:
            embedding_model = KnowledgeEmbeddingMongoModel.find_one(
                {"chunk_id": chunk_model.id},
            )
            if embedding_model is None:
                continue

            chunk = self._model_to_chunk(chunk_model)
            doc_model = KnowledgeDocumentMongoModel.find_by_id(chunk.document_id)
            doc = self._model_to_doc(doc_model) if doc_model is not None else None
            items.append(DenseRetrievalItem(
                chunk=chunk,
                embedding=embedding_model.embedding,
                document=doc,
            ))

        return self.retriever.retrieve(
            query_embedding=query_embedding,
            items=items,
            top_k=top_k,
        )

    def _kb_to_model(self, kb: KnowledgeBase) -> KnowledgeBaseMongoModel:
        """Converts a knowledge base DTO to a MongoDB model."""
        return KnowledgeBaseMongoModel(
            name=kb.name,
            description=kb.description,
            embedding_model_id=kb.embedding_model_id,
            dimensions=kb.dimensions,
            chunk_size=kb.chunk_size,
            chunk_overlap=kb.chunk_overlap,
            metadata=kb.metadata,
        )

    def _model_to_kb(self, model: KnowledgeBaseMongoModel) -> KnowledgeBase:
        """Converts a MongoDB model to a knowledge base DTO."""
        return KnowledgeBase(
            id=model.id,
            name=model.name,
            description=model.description,
            embedding_model_id=model.embedding_model_id,
            dimensions=model.dimensions,
            chunk_size=model.chunk_size,
            chunk_overlap=model.chunk_overlap,
            metadata=getattr(model, "metadata", {}),
            created_time=getattr(model, "created_time", None),
            updated_time=getattr(model, "updated_time", None),
        )

    def _doc_to_model(self, doc: KnowledgeDocument) -> KnowledgeDocumentMongoModel:
        """Converts a document DTO to a MongoDB model."""
        return KnowledgeDocumentMongoModel(
            knowledge_base_id=doc.knowledge_base_id,
            title=doc.title,
            source_uri=doc.source_uri,
            content=doc.content,
            metadata=doc.metadata,
        )

    def _model_to_doc(self, model: KnowledgeDocumentMongoModel) -> KnowledgeDocument:
        """Converts a MongoDB model to a document DTO."""
        return KnowledgeDocument(
            id=model.id,
            knowledge_base_id=model.knowledge_base_id,
            title=model.title,
            source_uri=model.source_uri,
            content=model.content,
            metadata=getattr(model, "metadata", {}),
            created_time=getattr(model, "created_time", None),
        )

    def _chunk_to_model(self, chunk: KnowledgeChunk) -> KnowledgeChunkMongoModel:
        """Converts a chunk DTO to a MongoDB model."""
        return KnowledgeChunkMongoModel(
            document_id=chunk.document_id,
            knowledge_base_id=chunk.knowledge_base_id,
            content=chunk.content,
            index=chunk.index,
            token_count=chunk.token_count,
            metadata=chunk.metadata,
        )

    def _model_to_chunk(self, model: KnowledgeChunkMongoModel) -> KnowledgeChunk:
        """Converts a MongoDB model to a chunk DTO."""
        return KnowledgeChunk(
            id=model.id,
            document_id=model.document_id,
            knowledge_base_id=model.knowledge_base_id,
            content=model.content,
            index=model.index,
            token_count=model.token_count,
            metadata=getattr(model, "metadata", {}),
        )
