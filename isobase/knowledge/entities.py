#! python3
# -*- encoding: utf-8 -*-
"""Core data structures for knowledge base operations.

@File   :   entities.py
@Created:   2026/07/29 00:16
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime


@dataclass
class KnowledgeBase:
    """Represents a knowledge base container.

    A knowledge base is a logical collection of indexed documents with
    associated configuration (embedding model, chunking strategy, etc.).

    Attributes:
        id: Unique identifier for the knowledge base.
        name: Human-readable name.
        description: Optional description of the knowledge base purpose.
        embedding_model_id: Identifier of the embedding model used.
        dimensions: Embedding vector dimensions.
        chunk_size: Default chunk size in characters.
        chunk_overlap: Overlap between adjacent chunks in characters.
        metadata: Additional metadata (tags, owner, etc.).
        created_time: Timestamp of creation.
        updated_time: Timestamp of last update.
    """

    id: str
    name: str
    description: str = ""
    embedding_model_id: str = ""
    dimensions: int = 1024
    chunk_size: int = 512
    chunk_overlap: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: Optional[datetime] = None
    updated_time: Optional[datetime] = None


@dataclass
class KnowledgeDocument:
    """Represents a source document in a knowledge base.

    A document is the top-level unit of indexing. It contains the original
    content and is split into chunks for embedding and retrieval.

    Attributes:
        id: Unique identifier for the document.
        knowledge_base_id: Parent knowledge base ID.
        title: Document title.
        source_uri: Source location (URL, file path, etc.).
        content: Full text content of the document.
        metadata: Additional metadata (author, date, tags, etc.).
        created_time: Timestamp of document creation.
    """

    id: str
    knowledge_base_id: str
    title: str
    source_uri: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: Optional[datetime] = None


@dataclass
class KnowledgeChunk:
    """Represents an indexed text chunk.

    Chunks are the unit of retrieval. Each chunk is embedded and stored
    with its vector representation for similarity search.

    Attributes:
        id: Unique identifier for the chunk.
        document_id: Parent document ID.
        knowledge_base_id: Parent knowledge base ID.
        content: Text content of the chunk.
        index: Position of this chunk within the parent document (0-based).
        token_count: Approximate token count (for cost estimation).
        metadata: Additional metadata (page number, section, etc.).
    """

    id: str
    document_id: str
    knowledge_base_id: str
    content: str
    index: int
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Represents a retrieved chunk with provenance.

    Returned by search operations, containing the chunk, its similarity score,
    and optionally the parent document for context.

    Attributes:
        chunk: The retrieved chunk.
        score: Similarity score (higher is more relevant).
        document: Optional parent document (for displaying source).
    """

    chunk: KnowledgeChunk
    score: float
    document: Optional[KnowledgeDocument] = None

    def __str__(self) -> str:
        """Formats the result for LLM context injection.

        Returns a human-readable representation with source attribution
        suitable for including in LLM prompts.
        """
        source = f" ({self.document.title})" if self.document else ""
        return f"[Score: {self.score:.3f}{source}]\n{self.chunk.content}"
