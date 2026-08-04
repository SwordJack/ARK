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
from ..retrieval import DenseRetriever, DenseRetrievalItem
from .base import BaseKnowledgeStore


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
        self.retriever = DenseRetriever()

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
        kb.created_time = now
        kb.updated_time = now

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

    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        """Lists all knowledge bases.

        Returns:
            List of all knowledge bases. May be empty.
        """
        return list(self.knowledge_bases.values())

    def delete_knowledge_base(self, kb_id: str) -> None:
        """Deletes a knowledge base and all its documents/chunks/embeddings.

        Args:
            kb_id: Knowledge base identifier.

        Raises:
            KeyError: If knowledge base not found.
        """
        if kb_id not in self.knowledge_bases:
            raise KeyError(f"Knowledge base {kb_id} not found")

        # Collect and remove all documents/chunks/embeddings under this KB
        doc_ids = [
            doc_id for doc_id, doc in self.documents.items()
            if doc.knowledge_base_id == kb_id
        ]
        for doc_id in doc_ids:
            del self.documents[doc_id]

        chunk_ids = [
            chunk_id for chunk_id, chunk in self.chunks.items()
            if chunk.knowledge_base_id == kb_id
        ]
        for chunk_id in chunk_ids:
            del self.chunks[chunk_id]
            if chunk_id in self.embeddings:
                del self.embeddings[chunk_id]

        del self.knowledge_bases[kb_id]

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

        doc.created_time = datetime.now(timezone.utc)
        self.documents[doc.id] = doc
        return doc

    def get_document(self, doc_id: str) -> KnowledgeDocument:
        """Retrieves a document by ID.

        Args:
            doc_id: Document identifier.

        Returns:
            The document object.

        Raises:
            KeyError: If document not found.
        """
        if doc_id not in self.documents:
            raise KeyError(f"Document {doc_id} not found")
        return self.documents[doc_id]

    def list_documents(self, kb_id: str) -> list[KnowledgeDocument]:
        """Lists all documents in a knowledge base.

        Args:
            kb_id: Knowledge base identifier.

        Returns:
            List of documents. May be empty.

        Raises:
            KeyError: If knowledge base not found.
        """
        if kb_id not in self.knowledge_bases:
            raise KeyError(f"Knowledge base {kb_id} not found")
        return [
            doc for doc in self.documents.values()
            if doc.knowledge_base_id == kb_id
        ]

    def delete_document(self, doc_id: str) -> None:
        """Deletes a document and all its chunks/embeddings.

        Args:
            doc_id: Document identifier.

        Raises:
            KeyError: If document not found.
        """
        if doc_id not in self.documents:
            raise KeyError(f"Document {doc_id} not found")

        del self.documents[doc_id]

        chunk_ids = [
            chunk_id for chunk_id, chunk in self.chunks.items()
            if chunk.document_id == doc_id
        ]
        for chunk_id in chunk_ids:
            del self.chunks[chunk_id]
            if chunk_id in self.embeddings:
                del self.embeddings[chunk_id]

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

    def list_chunks(self, kb_id: str) -> List[KnowledgeChunk]:
        """Lists all chunks in a knowledge base.

        Args:
            kb_id: Knowledge base identifier.

        Returns:
            List of chunks. May be empty.
        """
        return [
            chunk for chunk in self.chunks.values()
            if chunk.knowledge_base_id == kb_id
        ]

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

        items = [
            DenseRetrievalItem(
                chunk=chunk,
                embedding=self.embeddings[chunk_id],
                document=self.documents.get(chunk.document_id),
            )
            for chunk_id, chunk in kb_chunks
        ]
        return self.retriever.retrieve(
            query_embedding=query_embedding,
            items=items,
            top_k=top_k,
        )
