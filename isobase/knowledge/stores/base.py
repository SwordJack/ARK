#! python3
# -*- encoding: utf-8 -*-
"""Abstract base class for knowledge storage backends.

@File   :   base.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from abc import ABC, abstractmethod
from typing import List
from ..entities import KnowledgeBase, KnowledgeDocument, KnowledgeChunk, RetrievalResult


class BaseKnowledgeStore(ABC):
    """Abstract storage interface for knowledge base data.

    A knowledge store manages persistence for knowledge bases, documents,
    chunks, and their embedding vectors. Different implementations support
    different backends (memory, SQL, vector databases).
    """

    # ---- knowledge base -----------------------------------------------------------------

    @abstractmethod
    def create_knowledge_base(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Creates a new knowledge base.

        Args:
            kb: KnowledgeBase object to create. The id may be generated
                if not provided.

        Returns:
            The created knowledge base with id and timestamps populated.

        Raises:
            ValueError: If a knowledge base with the same id already exists.
        """
        pass

    @abstractmethod
    def get_knowledge_base(self, kb_id: str) -> KnowledgeBase:
        """Retrieves a knowledge base by ID.

        Args:
            kb_id: Knowledge base identifier.

        Returns:
            The knowledge base object.

        Raises:
            KeyError: If knowledge base not found.
        """
        pass

    @abstractmethod
    def list_knowledge_bases(self) -> List[KnowledgeBase]:
        """Lists all knowledge bases.

        Returns:
            List of all knowledge bases. May be empty.
        """
        pass

    @abstractmethod
    def delete_knowledge_base(self, kb_id: str) -> None:
        """Deletes a knowledge base and all its documents/chunks/embeddings.

        Args:
            kb_id: Knowledge base identifier.

        Raises:
            KeyError: If knowledge base not found.
        """
        pass

    # ---- document -----------------------------------------------------------------------

    @abstractmethod
    def add_document(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        """Stores a document.

        Args:
            doc: KnowledgeDocument object to store. The id may be generated
                if not provided.

        Returns:
            The stored document with id and timestamps populated.

        Raises:
            KeyError: If the parent knowledge base does not exist.
        """
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> KnowledgeDocument:
        """Retrieves a document by ID.

        Args:
            doc_id: Document identifier.

        Returns:
            The document object.

        Raises:
            KeyError: If document not found.
        """
        pass

    @abstractmethod
    def list_documents(self, kb_id: str) -> List[KnowledgeDocument]:
        """Lists all documents in a knowledge base.

        Args:
            kb_id: Knowledge base identifier.

        Returns:
            List of documents. May be empty.

        Raises:
            KeyError: If knowledge base not found.
        """
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        """Deletes a document and all its chunks/embeddings.

        The document is removed together with every chunk and embedding
        that belongs to it.  This is the recommended way to remove a
        document's footprint from a knowledge base.

        Args:
            doc_id: Document identifier.

        Raises:
            KeyError: If document not found.
        """
        pass

    # ---- chunks & search ----------------------------------------------------------------

    @abstractmethod
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
            ValueError: If len(chunks) != len(embeddings).
            KeyError: If parent document or knowledge base does not exist.
        """
        pass

    @abstractmethod
    def search(
        self,
        kb_id: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Searches for similar chunks by vector similarity.

        Args:
            kb_id: Knowledge base ID to search within.
            query_embedding: Query vector.
            top_k: Maximum number of results to return.

        Returns:
            List of retrieval results, sorted by score descending.
            Empty list if knowledge base is empty or not found.

        Note:
            Implementations should use cosine similarity or equivalent
            distance metric for vector comparison.
        """
        pass
