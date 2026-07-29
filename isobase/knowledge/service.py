#! python3
# -*- encoding: utf-8 -*-
"""High-level knowledge base service for indexing and retrieval.

@File   :   service.py
@Created:   2026/07/29 00:16
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import uuid
from typing import Any, Dict, List, Optional

from .entities import (
    KnowledgeBase,
    KnowledgeDocument,
    KnowledgeChunk,
    RetrievalResult
)
from .embeddings.base import BaseEmbeddingClient
from .chunking.base import BaseChunker
from .stores.base import BaseKnowledgeStore


class KnowledgeBaseService:
    """High-level service for indexing and retrieving knowledge.

    Orchestrates the full RAG pipeline: chunking, embedding, storage,
    and retrieval. This is the main entry point for knowledge base
    operations.

    Example usage:
        service = KnowledgeBaseService(
            embedding_client=OpenAIEmbeddingClient(...),
            store=MemoryKnowledgeStore(),
            chunker=FixedSizeChunker(chunk_size=512, chunk_overlap=50)
        )

        kb = service.create_knowledge_base(name="Documentation")
        service.index_text(kb.id, text="RAG is...", title="RAG Overview")
        results = service.retrieve(query="What is RAG?", knowledge_base_id=kb.id)
    """

    def __init__(
        self,
        embedding_client: BaseEmbeddingClient,
        store: BaseKnowledgeStore,
        chunker: BaseChunker,
    ):
        """Initializes the knowledge base service.

        Args:
            embedding_client: Client for generating text embeddings.
            store: Storage backend for knowledge bases and vectors.
            chunker: Chunking strategy for splitting documents.
        """
        self.embedding_client = embedding_client
        self.store = store
        self.chunker = chunker

    def create_knowledge_base(
        self,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBase:
        """Creates a new knowledge base.

        Args:
            name: Human-readable name for the knowledge base.
            description: Optional description of the knowledge base purpose.
            metadata: Additional metadata (tags, owner, etc.).

        Returns:
            The created knowledge base with generated id.

        Example:
            kb = service.create_knowledge_base(
                name="Product Documentation",
                description="Internal product docs and FAQs",
                metadata={"owner": "eng-team"}
            )
        """
        kb = KnowledgeBase(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            embedding_model_id=getattr(self.embedding_client, "model", "unknown"),
            dimensions=self.embedding_client.dimensions,
            chunk_size=getattr(self.chunker, "chunk_size", 0),
            chunk_overlap=getattr(self.chunker, "chunk_overlap", 0),
            metadata=metadata or {},
        )
        return self.store.create_knowledge_base(kb)

    def index_text(
        self,
        knowledge_base_id: str,
        text: str,
        title: str = "",
        source_uri: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeDocument:
        """Indexes a text document into a knowledge base.

        The document is chunked, embedded, and stored for retrieval.

        Args:
            knowledge_base_id: Target knowledge base ID.
            text: Text content to index.
            title: Document title (defaults to "Untitled").
            source_uri: Source URL or file path.
            metadata: Additional metadata (author, date, tags, etc.).

        Returns:
            The created document with its indexed chunks.

        Raises:
            KeyError: If knowledge base does not exist.
            RuntimeError: If embedding API call fails.

        Example:
            doc = service.index_text(
                knowledge_base_id=kb.id,
                text="RAG stands for Retrieval-Augmented Generation...",
                title="RAG Overview",
                source_uri="docs/rag.md",
                metadata={"author": "Alice"}
            )
        """
        # Verify knowledge base exists
        self.store.get_knowledge_base(knowledge_base_id)

        # Create document record
        doc = KnowledgeDocument(
            id=str(uuid.uuid4()),
            knowledge_base_id=knowledge_base_id,
            title=title or "Untitled",
            source_uri=source_uri,
            content=text,
            metadata=metadata or {},
        )
        doc = self.store.add_document(doc)

        # Chunk text
        chunk_texts = self.chunker.chunk(text)

        if not chunk_texts:
            # Empty document, nothing to index
            return doc

        # Create chunk records
        chunks = [
            KnowledgeChunk(
                id=str(uuid.uuid4()),
                document_id=doc.id,
                knowledge_base_id=knowledge_base_id,
                content=chunk_text,
                index=idx,
                token_count=len(chunk_text.split()),  # Rough approximation
                metadata={},
            )
            for idx, chunk_text in enumerate(chunk_texts)
        ]

        # Embed chunks
        embeddings = self.embedding_client.embed_texts([c.content for c in chunks])

        # Store chunks and embeddings
        self.store.add_chunks(chunks, embeddings)

        return doc

    def retrieve(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieves relevant chunks for a query.

        Args:
            query: Query text.
            knowledge_base_id: Knowledge base to search.
            top_k: Maximum number of results to return.

        Returns:
            List of retrieval results, sorted by relevance (score descending).
            Empty list if no relevant chunks found.

        Raises:
            ValueError: If query is empty.
            RuntimeError: If embedding API call fails.

        Example:
            results = service.retrieve(
                query="What is RAG?",
                knowledge_base_id=kb.id,
                top_k=3
            )
            for result in results:
                print(f"Score: {result.score:.3f}")
                print(f"Content: {result.chunk.content}")
        """
        if not query:
            raise ValueError("query cannot be empty")

        # Embed query
        query_embedding = self.embedding_client.embed_query(query)

        # Search
        results = self.store.search(
            kb_id=knowledge_base_id,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results

    def retrieve_as_context(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 5,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """Retrieves and formats results as LLM context.

        Convenience method for generating formatted context strings
        ready for injection into LLM prompts.

        Args:
            query: Query text.
            knowledge_base_id: Knowledge base to search.
            top_k: Maximum number of results.
            separator: Separator between chunks in the output.

        Returns:
            Formatted context string with source attribution.
            Empty string if no results found.

        Example:
            context = service.retrieve_as_context(
                query="What is RAG?",
                knowledge_base_id=kb.id,
                top_k=3
            )
            prompt = f"Context:\\n{context}\\n\\nQuestion: {query}\\nAnswer:"
        """
        results = self.retrieve(query, knowledge_base_id, top_k)

        if not results:
            return ""

        return separator.join(str(result) for result in results)
